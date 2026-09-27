import logging
import copy
import os
import pathlib
import sys
from functools import partial

# Prefer this checkout over another installed LLaVA package.
REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from arguments import ModelArguments, TrainingArguments
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN
from llava.medlsc_utils import medlsc
from llava.medlsc_utils.lora_utils import get_all_linear_names, add_lora_into_model_by_name, is_adapter_weight_key
from llava.train.trainer import LLaVATrainer
from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM
from llava.train.data import *

local_rank = None

from transformers import set_seed


def set_all_seeds(seed):
    print(f"##########  SET SEDD: {seed}")
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_trainable_state_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


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


def _load_previous_router_state(model, checkpoint_path):
    routing_candidates = []
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        routing_candidates.append(os.path.join(checkpoint_dir, "routing.bin"))
        routing_candidates.append(os.path.join(checkpoint_dir, "routing_projector.bin"))

    for candidate in routing_candidates:
        if os.path.exists(candidate):
            routing_state = torch.load(candidate, map_location="cpu")
            _load_partial_state_dict(model, routing_state)
            print(f"Loaded routing weights from {candidate}")
            return


def _freeze_previous_routing_keys(model, current_task_index):
    if not hasattr(model, "routing_keys"):
        return

    routing_keys = model.routing_keys
    if routing_keys.ndim != 2:
        return

    current_task_index = int(current_task_index)
    if current_task_index <= 0:
        return

    active_rows = torch.zeros_like(routing_keys)
    active_rows[current_task_index:, :] = 1.0

    def mask_routing_key_grad(grad):
        if grad is None:
            return None
        return grad * active_rows.to(device=grad.device, dtype=grad.dtype)

    routing_keys.register_hook(mask_routing_key_grad)
    routing_keys.requires_grad = True


def _snapshot_previous_routing_state(model, previous_task_count):
    if not hasattr(model, "routing_projector") or not hasattr(model, "routing_keys"):
        return

    previous_task_count = int(previous_task_count)
    if previous_task_count <= 0:
        return

    model.old_routing_projector = copy.deepcopy(model.routing_projector)
    model.old_routing_projector.requires_grad_(False)
    model.old_routing_projector.eval()

    old_routing_keys = torch.zeros_like(model.routing_keys.detach())
    copy_task_count = min(previous_task_count, old_routing_keys.shape[0], model.routing_keys.shape[0])
    if copy_task_count > 0:
        old_routing_keys[:copy_task_count] = model.routing_keys[:copy_task_count].detach().clone()

    model.old_routing_keys = old_routing_keys
    model.old_routing_task_count = copy_task_count


def _normalize_training_phase(training_phase):
    if training_phase is None:
        return "joint"
    phase = str(training_phase).strip().lower()
    if phase in {"lora", "lora_only"}:
        return "lora"
    if phase in {"router", "router_only"}:
        return "router"
    return "joint"


def _set_phase_trainability(model, training_phase, current_task_id, router_trainable=True):
    phase = _normalize_training_phase(training_phase)
    current_task_id = int(current_task_id)
    model.requires_grad_(False)

    if hasattr(model, "config"):
        model.config.training_phase = phase

    if phase in {"joint", "lora"}:
        current_lora_token = f"task_{current_task_id}_lora"
        for name, param in model.named_parameters():
            if "cl_lora_pool" in name and current_lora_token in name:
                param.requires_grad = True

    effective_router_trainable = phase == "router" or (phase == "joint" and router_trainable)
    if hasattr(model, "config"):
        model.config.routing_projector_trainable = bool(effective_router_trainable)

    if effective_router_trainable:
        router_tokens = ("routing_projector", "routing_keys", "allocation_projector", "allocation_keys")
        for name, param in model.named_parameters():
            if any(token in name for token in router_tokens):
                param.requires_grad = True
        if hasattr(model, "routing_keys"):
            _freeze_previous_routing_keys(model, current_task_id)


def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    seed = training_args.seed
    if seed == 42:
        print("###### Using Default random seed: 42")
        set_seed(42)
    else:
        set_all_seeds(seed)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    config = LlavaMistralConfig.from_pretrained(model_args.model_path)
    config.max_task = model_args.max_task
    config.alpha = model_args.alpha
    config.beta = model_args.beta
    config.kl_loss = bool(model_args.kl_loss)
    config.kl_weight = float(model_args.kl_weight)
    config.router_supervision_loss = bool(model_args.router_supervision_loss)
    config.router_supervision_weight = float(model_args.router_supervision_weight)
    current_task_id = int(model_args.current_task_id) if int(getattr(model_args, "current_task_id", -1)) >= 0 else (
        len(model_args.previous_lora_path) if model_args.previous_lora_path is not None else 0
    )
    config.current_task_id = current_task_id
    config.training_phase = _normalize_training_phase(model_args.training_phase)
    config.routing_use_task_mask = bool(model_args.routing_use_task_mask)

    model = LlavaMistralForCausalLM.from_pretrained(
        model_args.model_path,
        config=config,
        low_cpu_mem_usage=False,
        use_flash_attention_2=False,
        **bnb_model_from_pretrained_args,
    )
    setattr(model, "current_task_id", current_task_id)

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
        model.get_model().initialize_vision_modules(model.config)
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    model.model.mm_projector.to(dtype=compute_dtype, device=training_args.device)
    model.to(dtype=compute_dtype, device=training_args.device)

    if model_args.lora_enable:
        model.requires_grad_(False)

        routing_global_enable = bool(model_args.routing_global_enable)
        model.config.routing_global_enable = routing_global_enable

        if routing_global_enable:
            routing_projector_hidden = (
                None if int(model_args.routing_projector_hidden) <= 0 else int(model_args.routing_projector_hidden)
            )
            routing_hidden_size = routing_projector_hidden or max(1, model.config.hidden_size // 2)
            model.routing_projector = medlsc.AllocationProjector(
                in_features=model.config.hidden_size,
                hidden_features=routing_hidden_size,
                out_features=model.config.hidden_size,
                trainable=model_args.routing_projector_trainable,
            ).to(dtype=compute_dtype, device=training_args.device)
            model.routing_keys = torch.nn.Parameter(
                torch.empty(model_args.max_task, model.config.hidden_size, device=training_args.device, dtype=compute_dtype)
            )
            torch.nn.init.normal_(model.routing_keys, mean=0.0, std=0.02)
            model.config.routing_temperature = float(model_args.routing_temperature)
            model.config.routing_projector_hidden = int(routing_hidden_size)
            model.config.routing_projector_trainable = bool(model_args.routing_projector_trainable)
            model.config.routing_allocation_enable = bool(model_args.routing_allocation_enable)
            model.config.routing_global_enable = True
            model.config.routing_use_task_mask = bool(model_args.routing_use_task_mask)
            _freeze_previous_routing_keys(model, model_args.max_task - 1)

        medlsc_cfg = {
            "max_task": model_args.max_task,
            "lora_rank": model_args.lora_rank,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
            "allocation_enable": bool(model_args.routing_allocation_enable) and (not routing_global_enable),
            "allocation_projector_hidden": None if int(model_args.routing_projector_hidden) <= 0 else int(model_args.routing_projector_hidden),
            "allocation_temperature": float(model_args.routing_temperature),
            "allocation_projector_trainable": bool(model_args.routing_projector_trainable),
            "routing_use_task_mask": bool(model_args.routing_use_task_mask),
        }
        model.config.medlsc_cfg = medlsc_cfg

        lora_names = get_all_linear_names(
            model,
            exclude_keywords=("vision", "mm_projector", "lm_head", "routing_projector", "allocation_projector"),
        )

        if model_args.adding_layers is not None:
            model.config.medlsc_cfg["adding_layers"] = model_args.adding_layers
            print(model_args.adding_layers)
            lora_names = [
                n for n in lora_names if any(f".{adding_layer}." in n for adding_layer in model_args.adding_layers)
            ]

        print("lora names:", lora_names)
        add_lora_into_model_by_name(model, names=lora_names, medlsc_cfg=medlsc_cfg)

        if model_args.previous_lora_path is not None:
            print(f"loading previous lora weight from {model_args.previous_lora_path}")
            for path in model_args.previous_lora_path:
                weight_to_load = torch.load(path, "cpu")
                for k, _ in model.state_dict().items():
                    if is_adapter_weight_key(k) and k in weight_to_load:
                        del weight_to_load[k]
                model.load_state_dict(weight_to_load, strict=False)
                _load_previous_router_state(model, path)
            _snapshot_previous_routing_state(model, current_task_id)

        if bool(model_args.routing_use_task_mask):
            model.task_mask = torch.zeros(1024).to(device=model.device, dtype=torch.bool)
            model.task_mask[: model_args.max_task] = True

        if model_args.is_same_modality is not None:
            print(f"Is same_modality {model_args.is_same_modality}")
            model.is_same_modality = model_args.is_same_modality
            assert len(model.is_same_modality) == model_args.max_task - 1

        _set_phase_trainability(
            model,
            model_args.training_phase,
            current_task_id,
            router_trainable=bool(model_args.routing_projector_trainable),
        )

    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        gc_kwargs = {"use_reentrant": False}
        if getattr(training_args, "gradient_checkpointing_kwargs", None):
            gc_kwargs.update(training_args.gradient_checkpointing_kwargs)

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        backbone = model.get_model() if hasattr(model, "get_model") else model
        if hasattr(backbone, "_gradient_checkpointing_func"):
            backbone._gradient_checkpointing_func = partial(torch_checkpoint, use_reentrant=gc_kwargs["use_reentrant"])
        for module in model.modules():
            if hasattr(module, "_gradient_checkpointing_func"):
                module._gradient_checkpointing_func = partial(
                    torch_checkpoint,
                    use_reentrant=gc_kwargs["use_reentrant"],
                )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    training_args.use_im_start_end = model.config.mm_use_im_start_end

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and training_args.bf16:
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    data_args.mm_use_im_start_end = model.config.mm_use_im_start_end
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model.config.alpha = model_args.alpha
    model.config.beta = model_args.beta
    model.only_ortho = model.config.only_ortho = model_args.only_ortho

    if getattr(model.config, "alpha", None) is not None:
        alpha = model.config.alpha
        if alpha == -1:
            model.alpha = torch.nn.Parameter(torch.tensor(-4.6, requires_grad=True))
        else:
            model.alpha = model.config.alpha

    if getattr(model.config, "beta", None) is not None:
        beta = model.config.beta
        if beta == -1:
            model.beta = torch.nn.Parameter(torch.tensor(-4.6, requires_grad=True))
        else:
            model.beta = model.config.beta

    print("=" * 90)
    print(f"alpha: {model.alpha},\n beta: {model.beta}")

    print("=" * 90)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"trainable_param: {name}, shape: {param.shape}")
    print("=" * 90)

    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # Continual stages should come from explicit previous_lora_path / routing loads.
    # Do not auto-resume DeepSpeed optimizer state from checkpoint-* directories,
    # because that replays the old ZeRO checkpoint contract instead of the CL stage contract.
    trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    trainer._save(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="sdpa")

