"""Exact weighted LoRA-delta aggregation for MedLSC.

For every sample and routed layer, evaluate the mathematically exact mixture

    delta(x) = sum_i w_i * B_i(A_i(x))
    y        = base(x) + delta(x)

This is equivalent to merging the complete LoRA delta matrices

    Delta W_mix = sum_i w_i * (B_i A_i)

and then applying ``Delta W_mix x``, but it avoids materializing the full
``[out_features, in_features]`` matrix.

Unlike the V6 factor-merge ablation, this does **not** compute

    (sum_i w_i B_i) (sum_j w_j A_j) x

so it does not introduce cross-expert terms ``B_i A_j`` for ``i != j``.
"""

from __future__ import annotations

from typing import Optional

import torch

from llava.medlsc_utils import medlsc


def _exact_delta_fusion(
    self: medlsc.Linear,
    x: torch.Tensor,
    task_mask: Optional[torch.BoolTensor],
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    active_indices = self._task_indices(task_mask, device=x.device)
    if active_indices.numel() == 0:
        return self._base_linear(x)

    active_modules = [
        self.cl_lora_pool[f"task_{int(index)}_lora"]
        for index in active_indices.tolist()
    ]

    weights = routing_weights
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    if weights.size(-1) != active_indices.numel():
        weights = weights.index_select(dim=-1, index=active_indices)

    weights = weights.to(device=x.device, dtype=x.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    if weights.size(0) == 1 and x.size(0) != 1:
        weights = weights.expand(x.size(0), -1)
    if weights.size(0) != x.size(0):
        raise ValueError(
            "Exact-delta routing batch mismatch: "
            f"weights={tuple(weights.shape)}, input={tuple(x.shape)}"
        )

    # Stack factors by expert:
    #   A: [experts, rank, in_features]
    #   B: [experts, out_features, rank]
    #
    # The LoRA dropout is zero in the training scripts.  Applying the first
    # module dropout preserves the existing no-op behavior while keeping this
    # path compatible with the LoraModule interface.
    a_bank = torch.stack([module.lora_A for module in active_modules], dim=0)
    b_bank = torch.stack([module.lora_B for module in active_modules], dim=0)
    a_bank = a_bank.to(device=x.device, dtype=x.dtype)
    b_bank = b_bank.to(device=x.device, dtype=x.dtype)

    routed_x = active_modules[0].lora_dropout(x)

    # low_rank: [batch, ..., experts, rank] = A_i(x)
    low_rank = torch.einsum("b...d,erd->b...er", routed_x, a_bank)

    # expert_delta: [batch, ..., experts, out_features] = B_i(A_i(x))
    expert_delta = torch.einsum("b...er,eor->b...eo", low_rank, b_bank)

    # fused_delta: [batch, ..., out_features] = sum_i w_i * B_i(A_i(x))
    fused_delta = torch.einsum("be,b...eo->b...o", weights, expert_delta)

    scaling = float(active_modules[0].scaling)
    return self._base_linear(x) + fused_delta * scaling


def apply_exact_delta_merge_patch() -> None:
    """Install the exact weighted-delta fusion operator once."""
    if getattr(medlsc.Linear, "_v7_exact_delta_merge_patch", False):
        return
    medlsc.Linear._fuse_with_routing_weights = _exact_delta_fusion
    medlsc.Linear._v7_exact_delta_merge_patch = True
    print(
        "[V7 DELTA MERGE] Using exact weighted LoRA deltas: "
        "delta=sum_i(w_i * B_i(A_i(x)))."
    )
