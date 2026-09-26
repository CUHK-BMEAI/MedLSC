"""Department-aware hybrid routing for MSLoRA.

This module is intentionally lightweight and monkey-patch based so the original
model files can stay untouched.  It combines:

  1. query-key routing probabilities from the calibrated router;
  2. per-sample anchor probabilities/scores; and
  3. a department-aware candidate mask.

Candidate rule:
  anchor top-k experts -> inferred department(s)
  candidates = all experts in inferred department(s) + top-k outside experts
               selected by query-key score

Final weights:
  log p = lambda_Q(x) log p_query_masked + lambda_A(x) log p_anchor_masked
  weights = softmax(log p) over candidates; non-candidates get zero.
"""

from __future__ import annotations

from types import MethodType
from typing import Sequence

import torch
import torch.nn as nn


STANDARD_DATASETS = [
    "covid-CXP",
    "slake-ctxr",
    "iu-x-ray",
    "slake-mri",
    "PCAM",
    "pathvqa",
    "HAM_skin8",
    "derm",
    "Yangxi",
    "oct-c8",
    "cervical",
    "kvasir",
    "hyperkvasir",
]


DEPARTMENT_BY_TAG = {
    "covid-CXP": "radiology",
    "slake-ctxr": "radiology",
    "iu-x-ray": "radiology",
    "slake-mri": "radiology",
    "PCAM": "pathology",
    "pathvqa": "pathology",
    "HAM_skin8": "dermatology",
    "derm": "dermatology",
    "Yangxi": "ophthalmology",
    "oct-c8": "ophthalmology",
    "cervical": "gynecology",
    "kvasir": "gastroenterology",
    "hyperkvasir": "gastroenterology",
}


def dataset_tags_for_order(task_order: str, active_count: int) -> list[str]:
    tags = list(STANDARD_DATASETS)
    if str(task_order).lower() == "reverse":
        tags.reverse()
    return tags[: int(active_count)]


def _as_probabilities(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.float()
    if values.dim() == 1:
        values = values.unsqueeze(0)
    # If values look like probabilities, only renormalize.  Otherwise interpret
    # them as raw scores and softmax them.
    row_sum = values.sum(dim=-1, keepdim=True)
    nonnegative = bool(torch.all(values >= 0).detach().cpu().item())
    close_to_one = bool(torch.allclose(row_sum.detach().cpu(), torch.ones_like(row_sum).detach().cpu(), atol=1e-3))
    if nonnegative and close_to_one:
        return values / row_sum.clamp_min(eps)
    return torch.softmax(values, dim=-1)


class DepartmentAdaptiveFusionGate(nn.Module):
    """SMoLoRA-style learned fusion gate for anchor-vs-query routing scores."""

    def __init__(self, max_experts: int, init_anchor_coefficient: float = 0.2):
        super().__init__()
        self.max_experts = int(max_experts)
        init_anchor = min(max(float(init_anchor_coefficient), 1e-4), 1.0 - 1e-4)
        init_query = 1.0 - init_anchor
        self.anchor_fc = nn.Linear(self.max_experts, 1, bias=True)
        self.query_fc = nn.Linear(self.max_experts, 1, bias=True)
        nn.init.zeros_(self.anchor_fc.weight)
        nn.init.zeros_(self.query_fc.weight)
        with torch.no_grad():
            self.anchor_fc.bias.fill_(torch.log(torch.tensor(init_anchor)).item())
            self.query_fc.bias.fill_(torch.log(torch.tensor(init_query)).item())

    def _fit_width(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] == self.max_experts:
            return values
        if values.shape[-1] > self.max_experts:
            return values[..., : self.max_experts]
        pad = self.max_experts - values.shape[-1]
        return torch.cat([values, values.new_zeros(*values.shape[:-1], pad)], dim=-1)

    def forward(self, anchor_probs: torch.Tensor, query_probs: torch.Tensor) -> torch.Tensor:
        gate_dtype = self.anchor_fc.weight.dtype
        anchor_features = self._fit_width(anchor_probs).to(dtype=gate_dtype)
        query_features = self._fit_width(query_probs).to(dtype=gate_dtype)
        anchor_score = self.anchor_fc(anchor_features)
        query_score = self.query_fc(query_features)
        # Column 0 = lambda_anchor, column 1 = lambda_query.
        return torch.softmax(torch.cat([anchor_score, query_score], dim=-1), dim=-1)


def install_department_anchor_router(
    model,
    active_dataset_tags: Sequence[str],
    anchor_coefficient: float = 0.2,
    anchor_department_topk: int = 1,
    key_outside_topk: int = 3,
    fallback: str = "oracle_task",
    adaptive_fusion: bool = True,
) -> None:
    if not 0.0 <= float(anchor_coefficient) <= 1.0:
        raise ValueError("anchor_coefficient must be between 0 and 1")
    if int(anchor_department_topk) < 1:
        raise ValueError("anchor_department_topk must be >= 1")
    if int(key_outside_topk) < 0:
        raise ValueError("key_outside_topk must be >= 0")

    query_router = model._compute_global_routing_weights
    active_dataset_tags = list(active_dataset_tags)
    department_tags = [DEPARTMENT_BY_TAG.get(tag, tag) for tag in active_dataset_tags]
    unique_departments = {name: idx for idx, name in enumerate(sorted(set(department_tags)))}
    department_ids_cpu = torch.tensor([unique_departments[name] for name in department_tags], dtype=torch.long)
    department_names = {idx: name for name, idx in unique_departments.items()}

    model._external_department_anchor_weights = None
    model._department_anchor_fallback = str(fallback)
    model._department_anchor_active_tags = active_dataset_tags
    model._department_ids_cpu = department_ids_cpu
    model._department_names = department_names
    model._last_query_weights = None
    model._last_anchor_weights = None
    model._last_hybrid_weights = None
    model._last_fusion_weights = None
    model._last_candidate_mask = None
    model._last_anchor_department_tags = None
    model._last_key_outside_experts = None

    if bool(adaptive_fusion):
        if not hasattr(model, "department_anchor_fusion"):
            model.department_anchor_fusion = DepartmentAdaptiveFusionGate(
                max_experts=len(active_dataset_tags),
                init_anchor_coefficient=float(anchor_coefficient),
            )
        first_param = next(model.parameters())
        model.department_anchor_fusion.to(device=first_param.device, dtype=first_param.dtype)
        model.department_anchor_fusion.train(model.training)

    def _fallback_anchor(owner, query_weights: torch.Tensor) -> torch.Tensor:
        batch, experts = query_weights.shape
        out = torch.zeros(batch, experts, device=query_weights.device, dtype=torch.float32)
        current_task_id = int(getattr(owner.config, "current_task_id", 0))
        current_task_id = max(0, min(current_task_id, experts - 1))
        out[:, current_task_id] = 1.0
        return out

    def _build_candidate_mask(query_float: torch.Tensor, anchor_float: torch.Tensor):
        batch, experts = query_float.shape
        dept_ids = department_ids_cpu[:experts].to(device=query_float.device)
        candidate = torch.zeros_like(query_float, dtype=torch.bool)
        anchor_dept_names_batch: list[list[str]] = []
        outside_batch: list[list[int]] = []
        for b in range(batch):
            cur_anchor_topk = min(int(anchor_department_topk), experts)
            anchor_top = torch.topk(anchor_float[b], k=cur_anchor_topk).indices
            selected_depts = torch.unique(dept_ids[anchor_top])
            same_department = torch.isin(dept_ids, selected_depts)
            cur_mask = same_department.clone()

            outside = ~same_department
            outside_indices = torch.nonzero(outside, as_tuple=False).flatten()
            chosen_outside: list[int] = []
            if int(key_outside_topk) > 0 and outside_indices.numel() > 0:
                cur_k = min(int(key_outside_topk), int(outside_indices.numel()))
                outside_scores = query_float[b, outside_indices]
                outside_top_local = torch.topk(outside_scores, k=cur_k).indices
                outside_top = outside_indices[outside_top_local]
                cur_mask[outside_top] = True
                chosen_outside = [int(i) for i in outside_top.detach().cpu().tolist()]
            candidate[b] = cur_mask
            dept_list = [int(i) for i in selected_depts.detach().cpu().tolist()]
            anchor_dept_names_batch.append([department_names[i] for i in dept_list])
            outside_batch.append(chosen_outside)
        return candidate, anchor_dept_names_batch, outside_batch

    def hybrid_weights(owner, inputs_embeds, attention_mask=None):
        query_weights = query_router(inputs_embeds, attention_mask=attention_mask)
        if query_weights is None:
            return None
        query_float = query_weights.float()
        anchor_weights = owner._external_department_anchor_weights
        if anchor_weights is None:
            if str(getattr(owner, "_department_anchor_fallback", "oracle_task")) == "none":
                raise RuntimeError("Department-anchor routing expected anchor weights, but none were provided")
            anchor_float = _fallback_anchor(owner, query_float)
        else:
            anchor_float = _as_probabilities(anchor_weights.to(device=query_float.device), eps=1e-8)
            if anchor_float.shape[-1] > query_float.shape[-1]:
                anchor_float = anchor_float[:, : query_float.shape[-1]]
            if anchor_float.shape[-1] < query_float.shape[-1]:
                pad = query_float.shape[-1] - anchor_float.shape[-1]
                anchor_float = torch.cat(
                    [anchor_float, torch.zeros(anchor_float.size(0), pad, device=query_float.device)],
                    dim=-1,
                )
        if anchor_float.size(0) == 1 and query_float.size(0) > 1:
            anchor_float = anchor_float.expand(query_float.size(0), -1)
        if anchor_float.shape != query_float.shape:
            raise RuntimeError(f"Query/anchor shape mismatch: {tuple(query_float.shape)} != {tuple(anchor_float.shape)}")

        candidate_mask, anchor_dept_names, key_outside = _build_candidate_mask(query_float, anchor_float)
        eps = 1e-8
        masked_query = query_float.masked_fill(~candidate_mask, 0.0)
        masked_anchor = anchor_float.masked_fill(~candidate_mask, 0.0)
        query_norm = masked_query / masked_query.sum(dim=-1, keepdim=True).clamp_min(eps)
        anchor_norm = masked_anchor / masked_anchor.sum(dim=-1, keepdim=True).clamp_min(eps)

        fusion_gate = getattr(owner, "department_anchor_fusion", None)
        if bool(adaptive_fusion) and fusion_gate is not None:
            fusion = fusion_gate(anchor_norm, query_norm).to(device=query_norm.device, dtype=query_norm.dtype)
            lambda_anchor = fusion[:, 0:1]
            lambda_query = fusion[:, 1:2]
        else:
            lambda_anchor = torch.full(
                (query_norm.size(0), 1),
                float(anchor_coefficient),
                device=query_norm.device,
                dtype=query_norm.dtype,
            )
            lambda_query = 1.0 - lambda_anchor

        combined = (
            lambda_query * torch.log(query_norm.clamp_min(eps))
            + lambda_anchor * torch.log(anchor_norm.clamp_min(eps))
        )
        combined = combined.masked_fill(~candidate_mask, torch.finfo(combined.dtype).min)
        hybrid = torch.softmax(combined, dim=-1)

        owner._last_query_weights = query_float.detach().cpu()
        owner._last_anchor_weights = anchor_float.detach().cpu()
        owner._last_hybrid_weights = hybrid.detach().cpu()
        owner._last_fusion_weights = torch.cat([lambda_anchor, lambda_query], dim=-1).detach().cpu()
        owner._last_candidate_mask = candidate_mask.detach().cpu()
        owner._last_anchor_department_tags = anchor_dept_names
        owner._last_key_outside_experts = key_outside
        return hybrid.to(dtype=inputs_embeds.dtype)

    model._compute_global_routing_weights = MethodType(hybrid_weights, model)
