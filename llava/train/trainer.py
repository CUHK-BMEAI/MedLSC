import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_trainable_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def _split_trainable_state(named_params):
    named_params = list(named_params)
    routing_state = {}
    lora_state = {}
    routing_keys = ("routing_projector", "routing_keys", "allocation_projector", "allocation_keys")
    for name, tensor in dict(named_params).items():
        if any(key in name for key in routing_keys):
            routing_state[name] = tensor
        elif "lora_" in name:
            lora_state[name] = tensor
        else:
            continue
    routing_state = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in routing_state.items()}
    lora_state = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in lora_state.items()}
    return lora_state, routing_state


def _resolve_current_task_id(model, default: Optional[int] = -1):
    candidates = [
        getattr(model, "current_task_id", None),
        getattr(getattr(model, "module", None), "current_task_id", None),
        getattr(getattr(model, "config", None), "current_task_id", None),
        getattr(getattr(getattr(model, "module", None), "config", None), "current_task_id", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except Exception:
            continue
    return default if default is None else int(default)


class LLaVATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss

        components = getattr(model, "_last_loss_components", None) or {}
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1)))
        should_print = self.is_world_process_zero() and (int(self.state.global_step) % logging_steps == 0)
        if should_print and components:
            fields = [f"total={components.get('loss_total')}"]
            if components.get("loss_ce") is not None:
                fields.append(f"ce={components.get('loss_ce')}")
            if components.get("loss_kl") is not None:
                fields.append(f"kl={components.get('loss_kl')}")
            if components.get("loss_router_supervision") is not None:
                fields.append(f"router_sup={components.get('loss_router_supervision')}")
            if components.get("loss_reg") is not None:
                fields.append(f"reg={components.get('loss_reg')}")
            if components.get("loss_ortho") is not None:
                fields.append(f"ortho={components.get('loss_ortho')}")
            expected_router_label_preview = getattr(model, "_last_expected_router_label_preview", None)
            if expected_router_label_preview is not None:
                fields.append(f"expect_label={expected_router_label_preview}")
            expected_router_labels = getattr(model, "_last_expected_router_labels", None)
            if expected_router_labels is not None:
                fields.append(f"expect_label_ids={expected_router_labels}")
                if len(expected_router_labels) > 0:
                    fields.append(f"expect_label_first={int(expected_router_labels[0])}")
            resolved_task_id = _resolve_current_task_id(model)
            fields.append(f"current_task_id={resolved_task_id}")
            fields.append(f"router_expert_target={resolved_task_id}")
            loss_task_id = components.get("router_supervision_task_id")
            if loss_task_id is not None:
                fields.append(f"router_sup_task_id={int(loss_task_id)}")
            target_prob = components.get("router_supervision_target_prob")
            if target_prob is not None:
                fields.append(f"target_prob={target_prob}")
            print(f"[step {int(self.state.global_step)}] loss components: " + ", ".join(fields))

        log_payload = {k: v for k, v in components.items() if v is not None}
        if log_payload:
            self.log(log_payload)

        return (loss, outputs) if return_outputs else loss

    def _save_trainable_weights(self, output_dir: Optional[str]):
        if output_dir is None:
            return
        lora_state, routing_state = _split_trainable_state(self.model.named_parameters())
        os.makedirs(output_dir, exist_ok=True)
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "save_pretrained"):
            model_config.save_pretrained(output_dir)

        # Save per-stage filtered file cl_lora_task{N}.bin containing only 
        # the current-task LoRA params (to reduce storage space, no cumulative backup)
        
        # Attempt to detect current task id from model config and save filtered file
        current_task = None
        try:
            current_task = _resolve_current_task_id(self.model, default=None)
        except Exception:
            current_task = None

        if current_task is not None:
            key_sub = f"task_{current_task}_lora"
            current_stage_lora = {k: v for k, v in lora_state.items() if key_sub in k}
            if len(current_stage_lora) > 0:
                try:
                    torch.save(current_stage_lora, os.path.join(output_dir, f"cl_lora_task{current_task}.bin"))
                except Exception:
                    pass

        if len(routing_state) > 0:
            try:
                torch.save(routing_state, os.path.join(output_dir, "routing.bin"))
            except Exception:
                pass

    def _save_checkpoint(self, model, trial, metrics=None):
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self._save_trainable_weights(output_dir)


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self._save_trainable_weights(output_dir)