#!/usr/bin/env python3
"""Inference-only department-aware fusion of a calibrated query-key router and CLIP anchors.

For every sample, anchors first infer one or two likely departments.  The
candidate set contains all same-department experts plus top-k outside experts
selected by key-based routing score.  The final weights are computed only over
this candidate set:

    C(x)        = same_department(anchor_topk) union topk_other(query_score)
    log p_hyb   = (1-s) * log p_query[C] + s * log p_anchor[C]
    weights[C]  = softmax(log p_hyb)
    weights[~C] = 0

The resulting global weight vector is supplied to every MedLSC-enabled layer,
where V7 evaluates the exact weighted LoRA-delta formula:

    delta(x) = sum_i w_i * B_i(A_i(x))
    y        = base(x) + delta(x)

No router, LoRA, anchor, or base-model parameter is trained by this program.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Sequence
from uuid import uuid4

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed


REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from train_anchor_idea import (  # noqa: E402
    STANDARD_DATASETS,
    DatasetSpec,
    extract_prompt,
    image_value,
    load_records,
    resolve_image,
    torch_load_cpu,
    unwrap_state_dict,
)
from eval_anchor_idea import (  # noqa: E402
    answer_type,
    decode_generation,
    generation_prompt,
    ground_truth,
    load_routing_from_feature_cache,
)

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.eval.report_results import get_metrics  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model import LlavaMistralForCausalLM  # noqa: E402
from llava.medlsc_utils.exact_delta_merge_patch import apply_exact_delta_merge_patch  # noqa: E402
from llava.medlsc_utils import medlsc  # noqa: E402
from llava.medlsc_utils.department_anchor_router_v10 import DepartmentAdaptiveFusionGate  # noqa: E402
from llava.medlsc_utils.lora_utils import add_lora_into_model_by_name, get_all_linear_names, get_adapter_config  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


apply_exact_delta_merge_patch()


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


def stage_directory(checkpoint_root: Path, datasets: Sequence[DatasetSpec], stage_id: int) -> Path:
    return checkpoint_root / "_".join(spec.tag for spec in datasets[: stage_id + 1])


def resolve_stage_files(
    checkpoint_root: Path,
    datasets: Sequence[DatasetSpec],
    stage_id: int,
) -> tuple[Path, list[Path], Path]:
    current_stage_dir = stage_directory(checkpoint_root, datasets, stage_id)
    config_file = current_stage_dir / "config.json"
    routing_file = current_stage_dir / "routing.bin"
    if not config_file.is_file():
        raise FileNotFoundError(f"Stage configuration not found: {config_file}")
    if not routing_file.is_file():
        raise FileNotFoundError(
            f"Calibrated router checkpoint not found: {routing_file}. "
            "The Stage 2/Stage 3 checkpoint must contain routing.bin."
        )
    lora_paths: list[Path] = []
    for task_id in range(stage_id + 1):
        task_stage_dir = stage_directory(checkpoint_root, datasets, task_id)
        lora_file = task_stage_dir / f"cl_lora_task{task_id}.bin"
        if not lora_file.is_file():
            raise FileNotFoundError(f"Task-{task_id} LoRA checkpoint not found: {lora_file}")
        lora_paths.append(lora_file)
    return config_file, lora_paths, routing_file


def load_partial_state(model, state: dict[str, torch.Tensor]) -> tuple[int, int]:
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in state.items():
        target = model_state.get(key)
        if target is None or target.shape != value.shape:
            skipped += 1
            continue
        compatible[key] = value.to(dtype=target.dtype)
    model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped


def load_calibrated_model(
    model_path: Path,
    config_file: Path,
    lora_paths: Sequence[Path],
    routing_file: Path,
    device: str,
    dtype_name: str,
    anchor_coefficient: float,
):
    disable_torch_init()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    config_data = json.loads(config_file.read_text(encoding="utf-8"))
    medlsc_cfg = dict(get_adapter_config(config_data))
    if not medlsc_cfg:
        raise ValueError(f"No medlsc_cfg found in {config_file}")
    active_experts = len(lora_paths)
    medlsc_cfg["max_task"] = active_experts

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaMistralForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=False,
        use_flash_attention_2=False,
        torch_dtype=dtype,
    )
    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        model.config.vision_tower = model.config.mm_vision_tower
        model.get_model().initialize_vision_modules(model.config)
        vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    model.config.max_task = active_experts
    model.config.routing_global_enable = True
    model.config.routing_use_task_mask = False
    model.config.routing_temperature = float(
        config_data.get("routing_temperature", medlsc_cfg.get("allocation_temperature", 1.0))
    )
    hidden_size = int(model.config.hidden_size)
    projector_hidden = int(config_data.get("routing_projector_hidden", 0))
    if projector_hidden <= 0:
        projector_hidden = max(1, hidden_size // 2)
    model.routing_projector = medlsc.AllocationProjector(
        in_features=hidden_size,
        hidden_features=projector_hidden,
        out_features=hidden_size,
        trainable=False,
    )
    model.routing_keys = torch.nn.Parameter(
        torch.empty(active_experts, hidden_size), requires_grad=False
    )
    torch.nn.init.normal_(model.routing_keys, mean=0.0, std=0.02)
    adaptive_fusion_enabled = bool(config_data.get("department_adaptive_fusion_enable", True))
    if adaptive_fusion_enabled:
        model.department_anchor_fusion = DepartmentAdaptiveFusionGate(
            max_experts=active_experts,
            init_anchor_coefficient=float(config_data.get("department_anchor_coefficient", anchor_coefficient)),
        )

    lora_names = get_all_linear_names(
        model,
        exclude_keywords=("vision", "mm_projector", "lm_head", "routing_projector", "allocation_projector"),
    )
    if "adding_layers" in medlsc_cfg:
        adding_layers = medlsc_cfg["adding_layers"]
        lora_names = [name for name in lora_names if any(f".{layer}." in name for layer in adding_layers)]
    add_lora_into_model_by_name(model, names=lora_names, medlsc_cfg=medlsc_cfg)

    model_keys = set(model.state_dict())
    for expert_id, lora_path in enumerate(lora_paths):
        expected = f"task_{expert_id}_lora"
        state = unwrap_state_dict(torch_load_cpu(lora_path))
        relevant = {key: value for key, value in state.items() if expected in key}
        matched = sum(key in model_keys for key in relevant)
        if not relevant or matched == 0:
            raise RuntimeError(
                f"LoRA expert {expert_id} did not match: file={lora_path}, "
                f"relevant={len(relevant)}, matched={matched}"
            )
        model.load_state_dict(relevant, strict=False)
        print(f"[LoRA] loaded expert={expert_id:02d}, matched={matched}, file={lora_path}")

    routing_state = unwrap_state_dict(torch_load_cpu(routing_file))
    routing_matched, routing_skipped = load_partial_state(model, routing_state)
    required_tokens = ("routing_projector", "routing_keys")
    required_matches = [key for key in routing_state if key in model_keys and any(token in key for token in required_tokens)]
    if not required_matches:
        raise RuntimeError(
            f"No global query-key router tensors matched from {routing_file}. "
            f"Example checkpoint keys: {list(routing_state)[:10]}"
        )
    print(
        f"[ROUTER] calibrated checkpoint={routing_file}, matched={routing_matched}, "
        f"skipped={routing_skipped}, global_tensors={len(required_matches)}"
    )

    vision_tower.to(device=device, dtype=dtype)
    model.model.mm_projector.to(device=device, dtype=dtype)
    model.to(device=device, dtype=dtype)
    model.eval()
    return tokenizer, model, image_processor, dtype


def install_department_hybrid_router(
    model,
    seen_datasets: Sequence[DatasetSpec],
    anchor_coefficient: float,
    anchor_department_topk: int,
    key_outside_topk: int,
) -> None:
    if not 0.0 <= anchor_coefficient <= 1.0:
        raise ValueError("anchor coefficient must be between 0 and 1")
    if anchor_department_topk < 1:
        raise ValueError("anchor_department_topk must be >= 1")
    if key_outside_topk < 0:
        raise ValueError("key_outside_topk must be >= 0")
    query_router = model._compute_global_routing_weights
    model._external_anchor_weights = None
    model._last_query_weights = None
    model._last_anchor_weights = None
    model._last_hybrid_weights = None
    model._last_fusion_weights = None
    model._last_candidate_mask = None
    model._last_anchor_department_ids = None
    model._last_anchor_department_tags = None
    model._last_key_outside_experts = None

    department_tags = [DEPARTMENT_BY_TAG.get(spec.tag, spec.tag) for spec in seen_datasets]
    unique_departments = {name: idx for idx, name in enumerate(sorted(set(department_tags)))}
    department_ids = torch.tensor([unique_departments[name] for name in department_tags], dtype=torch.long)
    model._department_ids_cpu = department_ids
    model._department_names = {idx: name for name, idx in unique_departments.items()}

    def build_candidate_mask(query_float: torch.Tensor, anchor_float: torch.Tensor) -> tuple[torch.Tensor, list[list[int]], list[list[str]], list[list[int]]]:
        batch, experts = query_float.shape
        dept_ids = owner_dept_ids = model._department_ids_cpu[:experts].to(device=query_float.device)
        candidate = torch.zeros_like(query_float, dtype=torch.bool)
        anchor_dept_ids_batch: list[list[int]] = []
        anchor_dept_names_batch: list[list[str]] = []
        outside_batch: list[list[int]] = []
        for b in range(batch):
            cur_anchor_topk = min(anchor_department_topk, experts)
            anchor_top = torch.topk(anchor_float[b], k=cur_anchor_topk).indices
            selected_depts = torch.unique(dept_ids[anchor_top])
            same_department = torch.isin(dept_ids, selected_depts)
            cur_mask = same_department.clone()

            outside = ~same_department
            outside_indices = torch.nonzero(outside, as_tuple=False).flatten()
            chosen_outside: list[int] = []
            if key_outside_topk > 0 and outside_indices.numel() > 0:
                cur_k = min(key_outside_topk, int(outside_indices.numel()))
                outside_scores = query_float[b, outside_indices]
                outside_top_local = torch.topk(outside_scores, k=cur_k).indices
                outside_top = outside_indices[outside_top_local]
                cur_mask[outside_top] = True
                chosen_outside = [int(i) for i in outside_top.detach().cpu().tolist()]
            candidate[b] = cur_mask
            dept_list = [int(i) for i in selected_depts.detach().cpu().tolist()]
            anchor_dept_ids_batch.append(dept_list)
            anchor_dept_names_batch.append([model._department_names[i] for i in dept_list])
            outside_batch.append(chosen_outside)
        return candidate, anchor_dept_ids_batch, anchor_dept_names_batch, outside_batch

    def hybrid_weights(owner, inputs_embeds, attention_mask=None):
        query_weights = query_router(inputs_embeds, attention_mask=attention_mask)
        anchor_weights = owner._external_anchor_weights
        if query_weights is None:
            raise RuntimeError("The calibrated query-key router returned no weights")
        if anchor_weights is None:
            raise RuntimeError("Anchor weights were not set before generation")
        anchor_weights = anchor_weights.to(device=query_weights.device, dtype=torch.float32)
        if anchor_weights.dim() == 1:
            anchor_weights = anchor_weights.unsqueeze(0)
        query_float = query_weights.float()
        if query_float.shape != anchor_weights.shape:
            raise RuntimeError(
                f"Query/anchor weight shape mismatch: {tuple(query_float.shape)} != {tuple(anchor_weights.shape)}"
            )
        candidate_mask, anchor_dept_ids, anchor_dept_names, key_outside = build_candidate_mask(
            query_float, anchor_weights
        )
        eps = 1e-8
        neg_inf = torch.finfo(query_float.dtype).min
        masked_query = query_float.masked_fill(~candidate_mask, 0.0)
        masked_anchor = anchor_weights.masked_fill(~candidate_mask, 0.0)
        query_norm = masked_query / masked_query.sum(dim=-1, keepdim=True).clamp_min(eps)
        anchor_norm = masked_anchor / masked_anchor.sum(dim=-1, keepdim=True).clamp_min(eps)

        fusion_gate = getattr(owner, "department_anchor_fusion", None)
        if fusion_gate is not None:
            fusion = fusion_gate(anchor_norm, query_norm).to(device=query_float.device, dtype=query_float.dtype)
            lambda_anchor = fusion[:, 0:1]
            lambda_query = fusion[:, 1:2]
        else:
            lambda_anchor = torch.full(
                (query_float.size(0), 1),
                float(anchor_coefficient),
                device=query_float.device,
                dtype=query_float.dtype,
            )
            lambda_query = 1.0 - lambda_anchor

        combined_log_score = (
            lambda_query * torch.log(query_norm.clamp_min(eps))
            + lambda_anchor * torch.log(anchor_norm.clamp_min(eps))
        )
        combined_log_score = combined_log_score.masked_fill(~candidate_mask, neg_inf)
        hybrid = torch.softmax(combined_log_score, dim=-1)
        owner._last_query_weights = query_float.detach().cpu()
        owner._last_anchor_weights = anchor_weights.detach().cpu()
        owner._last_hybrid_weights = hybrid.detach().cpu()
        owner._last_fusion_weights = torch.cat([lambda_anchor, lambda_query], dim=-1).detach().cpu()
        owner._last_candidate_mask = candidate_mask.detach().cpu()
        owner._last_anchor_department_ids = anchor_dept_ids
        owner._last_anchor_department_tags = anchor_dept_names
        owner._last_key_outside_experts = key_outside
        return hybrid.to(dtype=inputs_embeds.dtype)

    model._compute_global_routing_weights = MethodType(hybrid_weights, model)


def evaluate_dataset(
    tokenizer,
    model,
    image_processor,
    model_dtype,
    data_root: Path,
    spec: DatasetSpec,
    seen_datasets: Sequence[DatasetSpec],
    anchor_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    question_file = data_root / spec.data_dir / spec.test_file
    rows = [row for row in load_records(question_file) if image_value(row) and extract_prompt(row)]
    if args.max_samples_per_dataset > 0:
        rows = rows[: args.max_samples_per_dataset]
    anchor_rows = list(anchor_rows[: len(rows)])
    if len(rows) != len(anchor_rows):
        raise ValueError(f"Test/anchor length mismatch for {spec.tag}: {len(rows)} != {len(anchor_rows)}")

    result_dir = output_dir / spec.tag
    result_dir.mkdir(parents=True, exist_ok=True)
    answer_file = result_dir / "answers.jsonl"
    task_mask = torch.zeros(1024, dtype=torch.bool, device=args.device)
    task_mask[: len(seen_datasets)] = True
    true_expert_id = seen_datasets.index(spec)

    with answer_file.open("w", encoding="utf-8") as handle:
        for index, (row, anchor_row) in enumerate(
            tqdm(zip(rows, anchor_rows), total=len(rows), desc=f"hybrid {spec.tag}")
        ):
            expected_image = str(anchor_row.get("image", ""))
            if expected_image and expected_image != image_value(row):
                raise ValueError(
                    f"Cache/test order mismatch for {spec.tag} at {index}: "
                    f"{expected_image!r} != {image_value(row)!r}"
                )
            anchor_weights = torch.tensor(
                anchor_row["expert_weights"], dtype=torch.float32, device=args.device
            ).unsqueeze(0)
            model._external_anchor_weights = anchor_weights

            question = extract_prompt(row)
            clean_question, prompt = generation_prompt(model, question, args.conv_mode)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(args.device)
            resolved_image = resolve_image(data_root, spec, question_file, image_value(row))
            with Image.open(resolved_image) as image:
                rgb_image = image.convert("RGB")
            image_tensor = process_images([rgb_image], image_processor, model.config)[0]
            rgb_image.close()
            image_tensor = image_tensor.unsqueeze(0).to(device=args.device, dtype=model_dtype)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    task_mask=task_mask,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            output_text = decode_generation(tokenizer, output_ids, input_ids)
            query_weights = model._last_query_weights[0].tolist()
            used_anchor_weights = model._last_anchor_weights[0].tolist()
            hybrid_weights = model._last_hybrid_weights[0].tolist()
            fusion_weights = (
                model._last_fusion_weights[0].tolist()
                if getattr(model, "_last_fusion_weights", None) is not None
                else [args.anchor_coefficient, 1.0 - args.anchor_coefficient]
            )
            candidate_mask = [bool(v) for v in model._last_candidate_mask[0].tolist()]
            candidate_experts = [i for i, keep in enumerate(candidate_mask) if keep]
            candidate_tags = [seen_datasets[i].tag for i in candidate_experts]
            anchor_department_tags = model._last_anchor_department_tags[0]
            key_outside_experts = model._last_key_outside_experts[0]
            query_selected = int(torch.tensor(query_weights).argmax().item())
            anchor_selected = int(torch.tensor(used_anchor_weights).argmax().item())
            hybrid_selected = int(torch.tensor(hybrid_weights).argmax().item())
            output = {
                "question_id": row.get("question_id", row.get("id", index)),
                "prompt": clean_question,
                "image": image_value(row),
                "text": output_text,
                "gt": ground_truth(row),
                "answer_type": answer_type(row),
                "answer_id": str(uuid4()),
                "model_id": get_model_name_from_path(str(args.model_path)),
                "metadata": {
                    "baseline": "v10_adaptive_hybrid_department_anchor_delta_merge",
                    "stage_id": args.stage_id,
                    "anchor_coefficient": args.anchor_coefficient,
                    "adaptive_fusion_weights_anchor_query": fusion_weights,
                    "query_weights": query_weights,
                    "anchor_weights": used_anchor_weights,
                    "hybrid_weights": hybrid_weights,
                    "candidate_mask": candidate_mask,
                    "candidate_experts": candidate_experts,
                    "candidate_tags": candidate_tags,
                    "candidate_size": len(candidate_experts),
                    "anchor_department_topk": args.anchor_department_topk,
                    "anchor_department_tags": anchor_department_tags,
                    "key_outside_topk": args.key_outside_topk,
                    "key_outside_experts": key_outside_experts,
                    "key_outside_tags": [seen_datasets[i].tag for i in key_outside_experts],
                    "operator": "exact_delta_merge",
                    "query_selected_expert": query_selected,
                    "anchor_selected_expert": anchor_selected,
                    "hybrid_selected_expert": hybrid_selected,
                    "hybrid_selected_tag": seen_datasets[hybrid_selected].tag,
                    "query_correct": query_selected == true_expert_id,
                    "anchor_correct": anchor_selected == true_expert_id,
                    "hybrid_correct": hybrid_selected == true_expert_id,
                },
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()

    if not args.skip_metrics:
        get_metrics(
            SimpleNamespace(
                result_file=str(answer_file),
                metric_output_file=str(result_dir / "metrics.txt"),
                hit_sample_file=None,
                not_hit_sample_file=None,
            )
        )
    return answer_file


def parse_dataset_selection(value: str, seen: Sequence[DatasetSpec]) -> list[DatasetSpec]:
    if value.strip().lower() in {"all", "all-seen"}:
        return list(seen)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    by_tag = {spec.tag: spec for spec in seen}
    unknown = [tag for tag in requested if tag not in by_tag]
    if unknown:
        raise ValueError(f"Datasets unavailable at this stage: {unknown}; seen={list(by_tag)}")
    return [by_tag[tag] for tag in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--anchor-file", required=True, type=Path)
    parser.add_argument("--test-feature-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage-id", type=int, default=12)
    parser.add_argument("--task-order", choices=("standard", "reverse"), default="standard")
    parser.add_argument("--datasets", default="all-seen")
    parser.add_argument("--anchor-coefficient", type=float, default=0.2)
    parser.add_argument("--anchor-department-topk", type=int, default=1)
    parser.add_argument("--key-outside-topk", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--conv-mode", default="mistral_instruct")
    parser.add_argument("--max-samples-per-dataset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--skip-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    datasets = list(STANDARD_DATASETS)
    if args.task_order == "reverse":
        datasets.reverse()
    if not 0 <= args.stage_id < len(datasets):
        raise ValueError(f"stage-id must be between 0 and {len(datasets) - 1}")
    for name in (
        "model_path",
        "data_root",
        "checkpoint_root",
        "anchor_file",
        "test_feature_cache_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen = datasets[: args.stage_id + 1]
    eval_datasets = parse_dataset_selection(args.datasets, seen)
    config_file, lora_paths, routing_file = resolve_stage_files(
        args.checkpoint_root, datasets, args.stage_id
    )
    anchor_by_dataset = load_routing_from_feature_cache(
        args.anchor_file,
        args.test_feature_cache_dir,
        seen,
        args.stage_id,
    )
    tokenizer, model, image_processor, model_dtype = load_calibrated_model(
        args.model_path,
        config_file,
        lora_paths,
        routing_file,
        args.device,
        args.dtype,
        args.anchor_coefficient,
    )
    install_department_hybrid_router(
        model,
        seen,
        args.anchor_coefficient,
        args.anchor_department_topk,
        args.key_outside_topk,
    )
    result_files = []
    for spec in eval_datasets:
        result_files.append(
            evaluate_dataset(
                tokenizer,
                model,
                image_processor,
                model_dtype,
                args.data_root,
                spec,
                seen,
                anchor_by_dataset[spec.tag],
                args.output_dir,
                args,
            )
        )
    summary = {
        "stage_id": args.stage_id,
        "stage_tag": seen[-1].tag,
        "task_order": args.task_order,
        "anchor_coefficient": args.anchor_coefficient,
        "adaptive_fusion": hasattr(model, "department_anchor_fusion"),
        "anchor_department_topk": args.anchor_department_topk,
        "key_outside_topk": args.key_outside_topk,
        "department_by_tag": DEPARTMENT_BY_TAG,
        "checkpoint_root": str(args.checkpoint_root),
        "routing_checkpoint": str(routing_file),
        "lora_checkpoints": [str(path) for path in lora_paths],
        "datasets": [spec.tag for spec in eval_datasets],
        "result_files": [str(path) for path in result_files],
    }
    (args.output_dir / "department_hybrid_delta_merge_inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[DONE] Department-aware hybrid query-anchor delta-merge evaluation: {args.output_dir}")


if __name__ == "__main__":
    main()
