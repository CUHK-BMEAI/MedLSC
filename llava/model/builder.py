import json
import os.path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from llava.medlsc_utils import medlsc as mslora
from llava.medlsc_utils.lora_utils import get_all_linear_names, add_lora_into_model_by_name
from llava.model import LlavaMistralForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def _load_partial_state_dict(model, state_dict):
    model_state = model.state_dict()
    updated_state = dict(model_state)
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if target.shape == value.shape:
            updated_state[key] = value.to(device=target.device, dtype=target.dtype)
            continue
        if target.ndim != value.ndim:
            continue
        copied = target.clone()
        slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, value.shape))
        copied[slices] = value[slices].to(device=target.device, dtype=target.dtype)
        updated_state[key] = copied
    model.load_state_dict(updated_state, strict=False)


def _init_global_routing_modules(model, routing_global_enable, routing_projector_hidden, routing_projector_trainable, max_task):
    if not routing_global_enable:
        return

    hidden_size = int(model.config.hidden_size)
    hidden_features = int(routing_projector_hidden) if int(routing_projector_hidden) > 0 else max(1, hidden_size // 2)

    model.routing_projector = mslora.AllocationProjector(
        in_features=hidden_size,
        hidden_features=hidden_features,
        out_features=hidden_size,
        trainable=bool(routing_projector_trainable),
    )
    model.routing_keys = torch.nn.Parameter(torch.empty(int(max_task), hidden_size))
    torch.nn.init.normal_(model.routing_keys, mean=0.0, std=0.02)


def load_pretrained_model(
        model_path,
        model_name,
        lora_paths=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda"
):
    print(f'Model Name: {model_name}')
    kwargs = {}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaMistralForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=False,
        use_flash_attention_2=False,
        **kwargs
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        model.config.vision_tower = model.config.mm_vision_tower
        model.get_model().initialize_vision_modules(
            model.config,
        )
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    cl_lora_weights = []
    if lora_paths is not None and len(lora_paths) > 0:##

        lora_names = get_all_linear_names(
            model,
            exclude_keywords=('vision', 'mm_projector', 'lm_head', 'routing_projector', 'allocation_projector')
        )

        cfg_path = os.path.join(os.path.dirname(lora_paths[-1]), 'config.json')
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        mslora_cfg = data['mslora_cfg']
        mslora_cfg['max_task'] = len(lora_paths)

        routing_global_enable = bool(data.get('routing_global_enable', False))
        routing_projector_hidden = int(data.get('routing_projector_hidden', 0))
        routing_projector_trainable = bool(data.get('routing_projector_trainable', True))
        routing_temperature = float(data.get('routing_temperature', mslora_cfg.get('allocation_temperature', 1.0)))

        model.config.routing_global_enable = routing_global_enable
        model.config.routing_projector_hidden = routing_projector_hidden
        model.config.routing_projector_trainable = routing_projector_trainable
        model.config.routing_temperature = routing_temperature
        model.config.routing_use_task_mask = bool(data.get('routing_use_task_mask', False))

        _init_global_routing_modules(
            model=model,
            routing_global_enable=routing_global_enable,
            routing_projector_hidden=routing_projector_hidden,
            routing_projector_trainable=routing_projector_trainable,
            max_task=len(lora_paths),
        )

        print(mslora_cfg)
        if 'adding_layers' in mslora_cfg:
            adding_layers = mslora_cfg.get('adding_layers')
            lora_names = [n for n in lora_names if any(f".{adding_layer}." in n for adding_layer in adding_layers)]

        print(f'lora names: ', lora_names)
        add_lora_into_model_by_name(model, names=lora_names, mslora_cfg=mslora_cfg)


        cl_lora_weights = []
        for lora_path in lora_paths:
            print(f'loading previous lora weight from {lora_path}')
            weight_to_load = torch.load(lora_path, 'cpu')
            # print(list(weight_to_load.keys()))
            cur_cl_lora_weight = {}
            for k, v in weight_to_load.items():
                if 'cl_lora_weight' in k:
                    cur_cl_lora_weight[k] = v
            cl_lora_weights.append(cur_cl_lora_weight)

            print('=' * 90)
            for k, v in model.state_dict().items():
                if any(k in kk for kk in weight_to_load.keys()):
                    print(f'matched: ', k)

            model.load_state_dict(weight_to_load, strict=False)

            routing_candidate_paths = [
                os.path.join(os.path.dirname(lora_path), 'routing.bin'),
                os.path.join(os.path.dirname(lora_path), 'routing_projector.bin'),
            ]
            for routing_path in routing_candidate_paths:
                if os.path.exists(routing_path):
                    routing_weight = torch.load(routing_path, map_location='cpu')
                    _load_partial_state_dict(model, routing_weight)
                    print(f'Loaded routing weights from {routing_path}')
                    break

    model.only_ortho = False
    vision_tower.to(device=device, dtype=kwargs['torch_dtype'])
    model.model.mm_projector.to(device=device, dtype=kwargs['torch_dtype'])
    model.to(device=device, dtype=kwargs['torch_dtype'])
    return tokenizer, model, image_processor, context_len, cl_lora_weights