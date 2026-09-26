import torch

from llava.medlsc_utils import medlsc as mslora
from torch import nn
from tqdm import tqdm
import loralib


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_state_dict_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_LinearWithLora(
    module: nn.Linear,
    mslora_cfg: dict,
    fan_in_fan_out=False,
) -> mslora.Linear:
    out_module = mslora.Linear(
        **mslora_cfg,
        in_features=module.in_features,
        out_features=module.out_features,
        fan_in_fan_out=fan_in_fan_out,
        bias=module.bias is not None,
    )
    out_module.load_state_dict(get_state_dict_maybe_zero_3(module.named_parameters()), strict=False)
    return out_module


def get_EmbeddingWithLora(module: nn.Embedding, lora_cfg: dict) -> loralib.Embedding:
    out_module = loralib.Embedding(
        num_embeddings=module.num_embeddings,
        embedding_dim=module.embedding_dim,
        **lora_cfg,
    )
    out_module.load_state_dict(module.state_dict(), strict=False)
    return out_module


def set_module_by_name(
    model: nn.Module,
    name: str,
    target_module: nn.Module,
):
    split_names = name.split(".")

    if len(split_names) == 1:
        if split_names[0].isdigit():
            model[int(split_names[0])] = target_module
        else:
            setattr(model, split_names[0], target_module)
        return

    if split_names[0].isdigit():
        cur_module = model[int(split_names[0])]
    else:
        cur_module = getattr(model, split_names[0])

    for part in split_names[1:-1]:
        if part.isdigit():
            cur_module = cur_module[int(part)]
        else:
            cur_module = getattr(cur_module, part)

    last_name = split_names[-1]
    if last_name.isdigit():
        cur_module[int(last_name)] = target_module
    else:
        setattr(cur_module, last_name, target_module)


def add_lora_into_model_by_name(
    model: nn.Module,
    names: list,
    mslora_cfg: dict,
    fan_in_fan_out: bool = False,
):
    with tqdm(total=len(names), desc="Adding LoRA modules (prog): ") as pbar:
        for name, module in model.named_modules():
            if name in names:
                if isinstance(module, nn.Linear):
                    set_module_by_name(model, name, get_LinearWithLora(module, mslora_cfg, fan_in_fan_out))
                elif isinstance(module, nn.Embedding):
                    print("Not support cl lora for nn.Embedding, will skip it...")
                pbar.update(1)


def get_all_linear_names(
    model: nn.Module,
    exclude_keywords: tuple = ("vision",),
    include_embedding=False,
):
    names = []
    for n, m in model.named_modules():
        if all(kw not in n for kw in exclude_keywords) and isinstance(m, nn.Linear) and n not in names:
            names.append(n)
        elif all(kw not in n for kw in exclude_keywords) and include_embedding and isinstance(m, nn.Embedding):
            names.append(n)
    return names
