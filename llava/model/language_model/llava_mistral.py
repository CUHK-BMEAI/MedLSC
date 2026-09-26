import math
from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from transformers import AutoConfig, AutoModelForCausalLM, MistralConfig, MistralModel, MistralForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ...medlsc_utils import medlsc as mslora


def calculate_euclidean_distance(a, b):
    euclidean_distance = torch.norm(torch.abs(a - b))
    return euclidean_distance


def get_sim_score(cur_lora, other_lora):
    total_sim = 0
    for w1, w2 in zip(cur_lora, other_lora):
        dis = calculate_euclidean_distance(w1, w2.detach().clone())
        sim = 1 / torch.exp(dis)
        total_sim += sim
    total_sim /= len(cur_lora)
    return total_sim


def get_param(param):
    from deepspeed import zero

    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data
    else:
        param = param
    return param


def compute_R(P_list, Q_list):
    R = 0.0
    I = torch.eye(P_list[0].size(0), device=P_list[0].device, dtype=P_list[0].dtype)
    for P, Q in zip(P_list, Q_list):
        PP_T = torch.matmul(P, P.T)
        QQ_T = torch.matmul(Q.T, Q)
        R += torch.norm(PP_T - I, p="fro") ** 2 + torch.norm(QQ_T - I, p="fro") ** 2
    R /= len(Q_list)
    return R


def _compute_global_routing_weights(owner, inputs_embeds, attention_mask=None):
    if (not hasattr(owner, "routing_projector")) or (not hasattr(owner, "routing_keys")):
        return None

    if inputs_embeds.dim() > 2:
        if attention_mask is not None:
            mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).unsqueeze(-1)
            pooled = inputs_embeds.masked_fill(mask == 0, torch.finfo(inputs_embeds.dtype).min)
        else:
            pooled = inputs_embeds
        pooled = pooled.max(dim=1, keepdim=True).values
    else:
        pooled = inputs_embeds.unsqueeze(1)

    routing_repr = owner.routing_projector(pooled)
    scores = torch.matmul(routing_repr, owner.routing_keys.t()).squeeze(1)
    temperature = float(getattr(owner.config, "routing_temperature", 1.0))
    if temperature <= 0:
        temperature = float(inputs_embeds.size(-1) ** 0.5)
    scores = scores / temperature
    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    return torch.softmax(scores, dim=-1)


def _compute_global_routing_weights_from_state(
    projector,
    routing_keys,
    inputs_embeds,
    attention_mask=None,
    temperature: float = 1.0,
    num_active_tasks: Optional[int] = None,
    pad_to: Optional[int] = None,
):
    if projector is None or routing_keys is None:
        return None

    if num_active_tasks is None:
        num_active_tasks = routing_keys.shape[0]
    num_active_tasks = int(num_active_tasks)
    if num_active_tasks <= 0:
        return None

    if inputs_embeds.dim() > 2:
        if attention_mask is not None:
            mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).unsqueeze(-1)
            pooled = inputs_embeds.masked_fill(mask == 0, torch.finfo(inputs_embeds.dtype).min)
        else:
            pooled = inputs_embeds
        pooled = pooled.max(dim=1, keepdim=True).values
    else:
        pooled = inputs_embeds.unsqueeze(1)

    routing_repr = projector(pooled)
    active_keys = routing_keys[:num_active_tasks]
    scores = torch.matmul(routing_repr, active_keys.t()).squeeze(1)
    if temperature <= 0:
        temperature = float(inputs_embeds.size(-1) ** 0.5)
    scores = scores / temperature
    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    weights = torch.softmax(scores, dim=-1)

    if pad_to is not None and pad_to > num_active_tasks:
        pad_width = int(pad_to - num_active_tasks)
        weights = torch.cat(
            [weights, torch.zeros(weights.size(0), pad_width, device=weights.device, dtype=weights.dtype)],
            dim=-1,
        )
    return weights


def _call_linear_with_routing(module, x, task_mask=None, routing_weights: Optional[torch.Tensor] = None):
    if isinstance(module, mslora.Linear):
        return module(x, task_mask=task_mask, routing_weights=routing_weights)
    return module(x)


def _routing_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
    task_mask=None,
    routing_weights: Optional[torch.Tensor] = None,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = _call_linear_with_routing(self.q_proj, hidden_states, task_mask=task_mask, routing_weights=routing_weights)
    key_states = _call_linear_with_routing(self.k_proj, hidden_states, task_mask=task_mask, routing_weights=routing_weights)
    value_states = _call_linear_with_routing(self.v_proj, hidden_states, task_mask=task_mask, routing_weights=routing_weights)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = _call_linear_with_routing(self.o_proj, attn_output, task_mask=task_mask, routing_weights=routing_weights)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _routing_mlp_forward(self, x, task_mask=None, routing_weights: Optional[torch.Tensor] = None):
    return _call_linear_with_routing(
        self.down_proj,
        self.act_fn(
            _call_linear_with_routing(self.gate_proj, x, task_mask=task_mask, routing_weights=routing_weights)
        ) * _call_linear_with_routing(self.up_proj, x, task_mask=task_mask, routing_weights=routing_weights),
        task_mask=task_mask,
        routing_weights=routing_weights,
    )


def _routing_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.Tensor] = None,
    task_mask=None,
    routing_weights: Optional[torch.Tensor] = None,
):
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
        task_mask=task_mask,
        routing_weights=routing_weights,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states, task_mask=task_mask, routing_weights=routing_weights)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def _patch_routing_modules(model: MistralModel):
    for decoder_layer in model.layers:
        decoder_layer.self_attn.forward = MethodType(_routing_attention_forward, decoder_layer.self_attn)
        decoder_layer.mlp.forward = MethodType(_routing_mlp_forward, decoder_layer.mlp)
        decoder_layer.forward = MethodType(_routing_decoder_layer_forward, decoder_layer)


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral_prog"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)
        _patch_routing_modules(self)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_mask=None,
        routing_weights=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        padding_mask = None

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        elif 0 in attention_mask:
            padding_mask = attention_mask

        if (
            padding_mask is not None
            and hasattr(self.config, "_flash_attn_2_enabled")
            and self.config._flash_attn_2_enabled
        ):
            is_padding_right = padding_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        attention_mask = self.create_extended_attention_mask_for_decoder(
            (batch_size, seq_length),
            attention_mask,
            inputs_embeds.device,
        )
        attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        if routing_weights is None:
                            hidden_states, attention_mask, position_ids = inputs
                            routing_kwarg = None
                        else:
                            hidden_states, attention_mask, position_ids, routing_kwarg = inputs
                        return module(
                            hidden_states,
                            attention_mask,
                            position_ids,
                            past_key_value,
                            output_attentions,
                            use_cache,
                            padding_mask=padding_mask,
                            task_mask=task_mask,
                            routing_weights=routing_kwarg,
                        )

                    return custom_forward

                checkpoint_inputs = [hidden_states, attention_mask, position_ids]
                if routing_weights is not None:
                    checkpoint_inputs.append(routing_weights)
                layer_outputs = torch_checkpoint(
                    create_custom_forward(decoder_layer),
                    *checkpoint_inputs,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                    task_mask=task_mask,
                    routing_weights=routing_weights,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def _compute_global_routing_weights(self, inputs_embeds, attention_mask=None):
        if not bool(getattr(self.config, "routing_global_enable", False)):
            return None
        return _compute_global_routing_weights(self, inputs_embeds, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        task_mask=None,
        routing_weights=None,
        replay_task_mark=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Keep mask optional for routing; only use binary mask when explicitly requested.
        use_binary_mask = bool(getattr(self.config, "routing_use_task_mask", False))
        if task_mask is None and use_binary_mask:
            task_mask = getattr(self, "task_mask", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                cls_tokens,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        routing_weights = routing_weights if routing_weights is not None else self._compute_global_routing_weights(
            inputs_embeds, attention_mask=attention_mask
        )

        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_mask=task_mask,
            routing_weights=routing_weights,
        )

        hidden_states = backbone_output[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        standard_loss = None
        # Compute CE loss only when not in router-only training phase.
        if labels is not None and not (
            self.training and getattr(self.config, "training_phase", "joint") == "router"
        ):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            standard_loss = loss

        loss_tensor = loss
        kl_loss_value = None
        router_supervision_loss_value = None
        reg_loss_value = None
        ortho_loss_value = None
        expected_router_labels = None
        expected_router_label_preview = None
        router_supervision_target_prob = None

        if (
            self.training
            and bool(getattr(self.config, "kl_loss", False))
            and getattr(self.config, "training_phase", "joint") == "router"
        ):
            if replay_task_mark is not None and routing_weights is not None and hasattr(self, "old_routing_projector"):
                replay_task_mark = torch.as_tensor(replay_task_mark, device=inputs_embeds.device).view(-1)
                replay_mask = replay_task_mark != -1
                if torch.any(replay_mask) and hasattr(self, "old_routing_keys"):
                    current_task_count = routing_weights.shape[-1]
                    old_task_count = int(getattr(self, "old_routing_task_count", 0))
                    old_temperature = float(getattr(self.config, "routing_temperature", 1.0))
                    old_routing_weights = _compute_global_routing_weights_from_state(
                        self.old_routing_projector,
                        self.old_routing_keys,
                        inputs_embeds,
                        attention_mask=attention_mask,
                        temperature=old_temperature,
                        num_active_tasks=old_task_count,
                        pad_to=current_task_count,
                    )
                    if old_routing_weights is not None:
                        current_replay = routing_weights[replay_mask]
                        old_replay = old_routing_weights[replay_mask]
                        kl_fn = nn.KLDivLoss(reduction="none")
                        kl_values = kl_fn(
                            torch.log(current_replay.clamp_min(1e-6)),
                            old_replay,
                        ).sum(dim=-1)
                        kl_loss = kl_values.mean()
                        kl_loss_value = kl_loss
                        kl_weight = float(getattr(self.config, "kl_weight", 1.0))
                        if loss_tensor is None:
                            loss_tensor = kl_weight * kl_loss
                        else:
                            loss_tensor = loss_tensor + kl_weight * kl_loss

        if (
            self.training
            and bool(getattr(self.config, "router_supervision_loss", False))
            and routing_weights is not None
            and getattr(self.config, "training_phase", "joint") in ("router", "calibration")
        ):
            current_task_id = int(getattr(self.config, "current_task_id", 0))
            target_task_ids = torch.full(
                (routing_weights.size(0),),
                current_task_id,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            # For replayed samples, use their original task IDs
            if replay_task_mark is not None:
                replay_task_mark_tensor = torch.as_tensor(replay_task_mark, device=inputs_embeds.device).view(-1)
                replay_mask = replay_task_mark_tensor != -1
                if torch.any(replay_mask):
                    target_task_ids[replay_mask] = replay_task_mark_tensor[replay_mask].long()
            target_task_ids = target_task_ids.long().clamp_(0, routing_weights.size(-1) - 1)
            expected_router_labels = target_task_ids.detach().cpu().tolist()
            if target_task_ids.numel() > 0:
                # Show distribution of target tasks (mixed labels from current + replay)
                preview = [0] * int(routing_weights.size(-1))
                unique_targets, counts = torch.unique(target_task_ids, return_counts=True)
                for task_id, count in zip(unique_targets, counts):
                    preview[int(task_id.item())] = int(count.item())
                expected_router_label_preview = preview
                # Log probability for the first target (may be replay or current task)
                router_supervision_target_prob = float(
                    routing_weights[0, int(target_task_ids[0].item())].detach().item()
                )
            router_supervision_loss = -torch.log(routing_weights.clamp_min(1e-6))[torch.arange(
                routing_weights.size(0), device=inputs_embeds.device
            ), target_task_ids].mean()
            router_supervision_loss_value = router_supervision_loss
            sup_w = float(getattr(self.config, "router_supervision_weight", 1.0))
            if loss_tensor is None:
                loss_tensor = sup_w * router_supervision_loss
            else:
                loss_tensor = loss_tensor + sup_w * router_supervision_loss

        if self.training:
            reg_loss = None
            ortho_loss = None

            active_task_count = None
            if task_mask is not None:
                active_task_count = int(torch.sum(task_mask).item())
            else:
                active_task_count = int(getattr(self.config, "max_task", 0))

            if active_task_count > 0:
                is_same_modality = getattr(self, "is_same_modality", None)
                if is_same_modality is not None and active_task_count > 1:
                    all_lora_weights = {}
                    for i in range(active_task_count):
                        cur_lora_weight = []
                        for n, param in self.named_parameters():
                            if f"task_{i}_lora" in n:
                                cur_lora_weight.append(get_param(param))
                        all_lora_weights[f"task_{i}_lora"] = cur_lora_weight

                    cur_lora = all_lora_weights.pop(f"task_{active_task_count - 1}_lora")
                    for i, (_, v) in enumerate(all_lora_weights.items()):
                        sim_score = get_sim_score(cur_lora, v)
                        if i < len(is_same_modality) and is_same_modality[i] == 1:
                            sim_score = 1 - sim_score
                        if reg_loss is None:
                            reg_loss = sim_score
                        else:
                            reg_loss += sim_score

                lora_As = []
                lora_Bs = []
                for n, param in self.named_parameters():
                    if "lora_A" in n:
                        lora_As.append(get_param(param))
                    if "lora_B" in n:
                        lora_Bs.append(get_param(param))
                if len(lora_As) > 0 and len(lora_Bs) > 0:
                    ortho_loss = compute_R(lora_As, lora_Bs)

                if reg_loss is not None:
                    reg_term = torch.exp(self.alpha) * reg_loss if isinstance(self.alpha, torch.nn.Parameter) else self.alpha * reg_loss
                    if ortho_loss is not None:
                        reg_term = reg_term + (
                            torch.exp(self.beta) * ortho_loss if isinstance(self.alpha, torch.nn.Parameter) else self.beta * ortho_loss
                        )
                    reg_loss_value = reg_loss
                    ortho_loss_value = ortho_loss
                    if loss_tensor is not None:
                        loss_tensor = loss_tensor + reg_term

        def _loss_to_float(value):
            if isinstance(value, torch.Tensor):
                return float(value.detach().item())
            return None

        self._last_loss_components = {
            "loss_total": _loss_to_float(loss_tensor),
            "loss_ce": _loss_to_float(standard_loss),
            "loss_kl": _loss_to_float(kl_loss_value),
            "loss_router_supervision": _loss_to_float(router_supervision_loss_value),
            "loss_reg": _loss_to_float(reg_loss_value),
            "loss_ortho": _loss_to_float(ortho_loss_value),
            "router_supervision_target_prob": router_supervision_target_prob,
            "router_supervision_task_id": int(getattr(self.config, "current_task_id", -1)),
        }
        self._last_expected_router_labels = expected_router_labels
        self._last_expected_router_label_preview = expected_router_label_preview
        if not return_dict:
            output = (logits,) + backbone_output[1:]
            output = (loss_tensor,) + output if loss_tensor is not None else output
        else:
            output = CausalLMOutputWithPast(
                loss=loss_tensor,
                logits=logits,
                past_key_values=backbone_output.past_key_values,
                hidden_states=backbone_output.hidden_states,
                attentions=backbone_output.attentions,
            )
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        task_mask=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                cls_tokens,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
            self.cls_tokens = cls_tokens
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        routing_weights = self._compute_global_routing_weights(inputs_embeds, attention_mask=attention_mask)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            task_mask=task_mask,
            routing_weights=routing_weights,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        task_mask=None,
        routing_weights=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if task_mask is not None:
            inputs["task_mask"] = task_mask
        if routing_weights is not None:
            inputs["routing_weights"] = routing_weights
        return inputs


AutoConfig.register("llava_mistral_prog", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)