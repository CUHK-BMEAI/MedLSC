#!/usr/bin/env python3
"""Build CLIP anchors and test whether they route samples to the right LoRA expert.

This is deliberately independent of the existing MedLSC training/evaluation files.
It validates the same standalone per-dataset LoRA checkpoints used by
``train.py``, builds one image/text anchor per task from
the training split, and measures anchor-routing accuracy on the test splits.

No LLaVA generation is performed: the purpose of this program is to isolate the
quality of anchor-based expert allocation from answer-generation quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


IMAGE_TOKEN = "<image>"


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    name: str
    data_dir: str
    train_file: str
    test_file: str


STANDARD_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("covid-CXP", "COVID-CXP", "COVID_CXP", "train.json", "test.jsonl"),
    DatasetSpec("slake-ctxr", "Slake-CTXR", "Slake-VQARad", "train_ct_xray.json", "test_ct_xray.jsonl"),
    DatasetSpec("iu-x-ray", "IU-X-Ray", "IU-X-Ray", "train.json", "test.jsonl"),
    DatasetSpec("slake-mri", "Slake-MRI", "Slake-VQARad", "train_mri.json", "test_mri.jsonl"),
    DatasetSpec("PCAM", "PCAM", "PCam", "train.json", "test.jsonl"),
    DatasetSpec("pathvqa", "PathVQA", "PathVQA", "train.json", "test.jsonl"),
    DatasetSpec("HAM_skin8", "HAM/Skin8", "HAM_skin8", "train.json", "test.jsonl"),
    DatasetSpec("derm", "Derm/Fitzpatrick", "Fitzpatrick", "train.json", "test.jsonl"),
    DatasetSpec("Yangxi", "Yangxi", "Yangxi", "train.json", "test.jsonl"),
    DatasetSpec("oct-c8", "OCT-C8", "Retinal_OCT_C8", "train.json", "test.jsonl"),
    DatasetSpec("cervical", "Cervical", "cervical", "train.json", "test.jsonl"),
    DatasetSpec("kvasir", "Kvasir-VQA", "Kvasir-VQA", "train.json", "test.jsonl"),
    DatasetSpec("hyperkvasir", "HyperKvasir", "HyperKvasir", "train.json", "test.jsonl"),
)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                rows.append(row)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a list of objects in {path}")
    return data


def extract_prompt(row: dict[str, Any]) -> str:
    for key in ("text", "question", "prompt"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).replace(IMAGE_TOKEN, " ").strip()
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", ""))).lower()
            if role in {"human", "user"}:
                return str(turn.get("value", turn.get("content", ""))).replace(IMAGE_TOKEN, " ").strip()
    return ""


def image_value(row: dict[str, Any]) -> str:
    value = row.get("image", "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def resolve_image(data_root: Path, spec: DatasetSpec, question_file: Path, value: str) -> Path:
    path = Path(value).expanduser()
    candidates = (
        path,
        data_root / path,
        data_root / spec.data_dir / path,
        question_file.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Image not found for {spec.tag}: {value!r}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def stable_offset(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int, tag: str) -> list[dict[str, Any]]:
    usable = [row for row in rows if image_value(row) and extract_prompt(row)]
    if limit > 0 and len(usable) > limit:
        return random.Random(seed + stable_offset(tag)).sample(usable, limit)
    return usable


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        for key in ("state_dict", "model", "module"):
            nested = value.get(key)
            if isinstance(nested, dict) and any(torch.is_tensor(item) for item in nested.values()):
                value = nested
                break
    if not isinstance(value, dict):
        raise ValueError(f"LoRA checkpoint has unsupported type: {type(value)}")
    return {str(key): tensor for key, tensor in value.items() if torch.is_tensor(tensor)}


def find_lora_file(root: Path, tag: str) -> Path:
    task_dir = root / tag
    candidates = (
        task_dir / "cl_lora_task0.bin",
        task_dir / "cl_lora.bin",
        root / f"{tag}.bin",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if task_dir.is_dir():
        matches = sorted(task_dir.glob("cl_lora*.bin"))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(
        f"LoRA checkpoint for {tag!r} not found under {root}. Expected {task_dir / 'cl_lora_task0.bin'}"
    )


def validate_lora_experts(root: Path, datasets: Sequence[DatasetSpec]) -> list[dict[str, Any]]:
    """Load every standalone LoRA exactly once and record its intended expert slot."""
    manifest: list[dict[str, Any]] = []
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    for expert_id, spec in enumerate(datasets):
        path = find_lora_file(root, spec.tag)
        state = unwrap_state_dict(torch_load_cpu(path))
        lora_state = {key: value for key, value in state.items() if "lora" in key.lower()}
        if not lora_state:
            raise ValueError(f"No LoRA tensors found in {path}")

        # Compare topology after hiding the source task-0 slot. This catches a
        # mismatched rank or target-module set before expensive anchor extraction.
        canonical_shapes = {
            key.replace("task_0_lora", "task_*_lora").replace("task0", "task*"): tuple(value.shape)
            for key, value in lora_state.items()
        }
        if reference_shapes is None:
            reference_shapes = canonical_shapes
        else:
            common = set(reference_shapes).intersection(canonical_shapes)
            incompatible = [key for key in common if reference_shapes[key] != canonical_shapes[key]]
            if incompatible:
                first = incompatible[0]
                raise ValueError(
                    f"LoRA topology mismatch for expert {expert_id} ({spec.tag}) at {first}: "
                    f"{reference_shapes[first]} != {canonical_shapes[first]}"
                )

        tensor_bytes = sum(value.numel() * value.element_size() for value in lora_state.values())
        manifest.append(
            {
                "expert_id": expert_id,
                "dataset_tag": spec.tag,
                "source_path": str(path),
                "source_slot": 0,
                "target_slot": expert_id,
                "tensor_count": len(lora_state),
                "tensor_bytes": tensor_bytes,
                "file_bytes": path.stat().st_size,
            }
        )
        print(
            f"[LoRA] expert={expert_id:02d} tag={spec.tag:<12} "
            f"tensors={len(lora_state):4d} file={path}"
        )
    return manifest


class ClipEncoder:
    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=self.dtype).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, images: Sequence[Image.Image], texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        )
        return F.normalize(image_features.float(), dim=-1), F.normalize(text_features.float(), dim=-1)


def batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def open_batch(
    rows: Sequence[dict[str, Any]], data_root: Path, spec: DatasetSpec, source_file: Path
) -> tuple[list[Image.Image], list[str]]:
    images: list[Image.Image] = []
    texts: list[str] = []
    try:
        for row in rows:
            with Image.open(resolve_image(data_root, spec, source_file, image_value(row))) as image:
                images.append(image.convert("RGB"))
            texts.append(extract_prompt(row))
    except Exception:
        for image in images:
            image.close()
        raise
    return images, texts


def build_anchors(
    encoder: ClipEncoder,
    data_root: Path,
    datasets: Sequence[DatasetSpec],
    batch_size: int,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_anchors: list[torch.Tensor] = []
    text_anchors: list[torch.Tensor] = []
    counts: list[int] = []
    for expert_id, spec in enumerate(datasets):
        source_file = data_root / spec.data_dir / spec.train_file
        rows = sample_rows(load_records(source_file), max_samples, seed, spec.tag)
        if not rows:
            raise ValueError(f"No usable anchor rows for {spec.tag}: {source_file}")
        image_sum: torch.Tensor | None = None
        text_sum: torch.Tensor | None = None
        count = 0
        progress = tqdm(total=len(rows), desc=f"anchors {expert_id:02d} {spec.tag}")
        for batch in batched(rows, batch_size):
            images, texts = open_batch(batch, data_root, spec, source_file)
            try:
                image_features, text_features = encoder.encode(images, texts)
            finally:
                for image in images:
                    image.close()
            batch_image_sum = image_features.sum(0).cpu()
            batch_text_sum = text_features.sum(0).cpu()
            image_sum = batch_image_sum if image_sum is None else image_sum + batch_image_sum
            text_sum = batch_text_sum if text_sum is None else text_sum + batch_text_sum
            count += len(batch)
            progress.update(len(batch))
        progress.close()
        assert image_sum is not None and text_sum is not None
        image_anchors.append(F.normalize((image_sum / count).unsqueeze(0), dim=-1).squeeze(0))
        text_anchors.append(F.normalize((text_sum / count).unsqueeze(0), dim=-1).squeeze(0))
        counts.append(count)
    return torch.stack(image_anchors), torch.stack(text_anchors), torch.tensor(counts, dtype=torch.long)


def compute_routing_weights(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_anchors: torch.Tensor,
    text_anchors: torch.Tensor,
    temperature: float,
    image_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if not 0.0 <= image_weight <= 1.0:
        raise ValueError("image_weight must be between zero and one")
    image_similarity = image_features @ image_anchors.T
    text_similarity = text_features @ text_anchors.T
    similarity = image_weight * image_similarity + (1.0 - image_weight) * text_similarity
    weights = torch.softmax(similarity / temperature, dim=-1)
    return weights, similarity, image_similarity, text_similarity


def save_confusion_csv(path: Path, confusion: torch.Tensor, tags: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *tags])
        for tag, row in zip(tags, confusion.tolist()):
            writer.writerow([tag, *row])


def cache_test_features(
    encoder: ClipEncoder,
    data_root: Path,
    datasets: Sequence[DatasetSpec],
    output_dir: Path,
    batch_size: int,
    max_samples: int,
    seed: int,
) -> list[Path]:
    """Encode every test sample once so all continual stages can reuse it."""
    cache_dir = output_dir / "test_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    for true_id, spec in enumerate(datasets):
        source_file = data_root / spec.data_dir / spec.test_file
        rows = sample_rows(load_records(source_file), max_samples, seed, f"test:{spec.tag}")
        if not rows:
            raise ValueError(f"No usable test rows for {spec.tag}: {source_file}")
        image_batches: list[torch.Tensor] = []
        text_batches: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        progress = tqdm(total=len(rows), desc=f"encode  {true_id:02d} {spec.tag}")
        for batch in batched(rows, batch_size):
            images, texts = open_batch(batch, data_root, spec, source_file)
            try:
                image_features, text_features = encoder.encode(images, texts)
            finally:
                for image in images:
                    image.close()
            # FP16 halves cache size. Similarities are still evaluated in FP32.
            image_batches.append(image_features.cpu().half())
            text_batches.append(text_features.cpu().half())
            for row in batch:
                metadata.append(
                    {
                        "question_id": row.get("question_id", row.get("id")),
                        "image": image_value(row),
                        "prompt": extract_prompt(row),
                    }
                )
            progress.update(len(batch))
        progress.close()
        cache_path = cache_dir / f"{true_id:02d}_{spec.tag}_test_features.pt"
        torch.save(
            {
                "dataset_tag": spec.tag,
                "true_expert_id": true_id,
                "image_features": torch.cat(image_batches, dim=0),
                "text_features": torch.cat(text_batches, dim=0),
                "metadata": metadata,
            },
            cache_path,
        )
        cache_paths.append(cache_path)
        print(f"[INFO] Cached {len(rows)} test features: {cache_path}")
    return cache_paths


def evaluate_cached_stage(
    encoder: ClipEncoder,
    datasets: Sequence[DatasetSpec],
    cache_paths: Sequence[Path],
    image_anchors: torch.Tensor,
    text_anchors: torch.Tensor,
    stage_id: int,
    output_dir: Path,
    batch_size: int,
    temperature: float,
    image_weight: float,
) -> dict[str, Any]:
    """Evaluate tasks 0..stage_id using only experts 0..stage_id."""
    active_datasets = list(datasets[: stage_id + 1])
    tags = [spec.tag for spec in active_datasets]
    stage_dir = output_dir / f"stage_{stage_id:02d}_{datasets[stage_id].tag}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    image_anchors_device = F.normalize(image_anchors[: stage_id + 1].float(), dim=-1).to(encoder.device)
    text_anchors_device = F.normalize(text_anchors[: stage_id + 1].float(), dim=-1).to(encoder.device)
    confusion = torch.zeros(stage_id + 1, stage_id + 1, dtype=torch.long)
    dataset_metrics: list[dict[str, Any]] = []
    total = top1_correct = top3_correct = 0
    predictions_path = stage_dir / "routing_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as prediction_file:
        for true_id, spec in enumerate(active_datasets):
            cached = torch_load_cpu(cache_paths[true_id])
            image_features = cached["image_features"]
            text_features = cached["text_features"]
            metadata = cached["metadata"]
            dataset_total = int(image_features.shape[0])
            dataset_top1 = dataset_top3 = 0
            progress = tqdm(total=dataset_total, desc=f"stage {stage_id:02d} test {spec.tag}")
            for start in range(0, dataset_total, batch_size):
                end = min(start + batch_size, dataset_total)
                image_batch = image_features[start:end].float().to(encoder.device)
                text_batch = text_features[start:end].float().to(encoder.device)
                weights, similarity, image_sim, text_sim = compute_routing_weights(
                    image_batch,
                    text_batch,
                    image_anchors_device,
                    text_anchors_device,
                    temperature,
                    image_weight,
                )
                top_k = min(3, stage_id + 1)
                top_ids = weights.topk(top_k, dim=-1).indices.cpu()
                weights_cpu = weights.cpu()
                similarity_cpu = similarity.cpu()
                image_sim_cpu = image_sim.cpu()
                text_sim_cpu = text_sim.cpu()
                for local_index, sample in enumerate(metadata[start:end]):
                    predicted_id = int(top_ids[local_index, 0])
                    top3_ids = [int(value) for value in top_ids[local_index].tolist()]
                    is_top1 = predicted_id == true_id
                    is_top3 = true_id in top3_ids
                    confusion[true_id, predicted_id] += 1
                    dataset_top1 += int(is_top1)
                    dataset_top3 += int(is_top3)
                    output = {
                        **sample,
                        "stage_id": stage_id,
                        "stage_tag": datasets[stage_id].tag,
                        "dataset_tag": spec.tag,
                        "true_expert_id": true_id,
                        "predicted_expert_id": predicted_id,
                        "predicted_expert_tag": tags[predicted_id],
                        "top1_correct": is_top1,
                        "top3_expert_ids": top3_ids,
                        "top3_expert_tags": [tags[value] for value in top3_ids],
                        "expert_weights": weights_cpu[local_index].tolist(),
                        "combined_similarity": similarity_cpu[local_index].tolist(),
                        "image_similarity": image_sim_cpu[local_index].tolist(),
                        "text_similarity": text_sim_cpu[local_index].tolist(),
                    }
                    prediction_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                progress.update(end - start)
            progress.close()
            total += dataset_total
            top1_correct += dataset_top1
            top3_correct += dataset_top3
            metrics = {
                "expert_id": true_id,
                "dataset_tag": spec.tag,
                "samples": dataset_total,
                "top1_correct": dataset_top1,
                "top1_accuracy": dataset_top1 / dataset_total,
                "top3_correct": dataset_top3,
                "top3_accuracy": dataset_top3 / dataset_total,
            }
            dataset_metrics.append(metrics)
            print(
                f"[STAGE {stage_id:02d}] expert={true_id:02d} tag={spec.tag:<12} n={dataset_total:6d} "
                f"top1={metrics['top1_accuracy']:.4f} top3={metrics['top3_accuracy']:.4f}"
            )

    summary = {
        "stage_id": stage_id,
        "stage_tag": datasets[stage_id].tag,
        "active_experts": stage_id + 1,
        "active_dataset_tags": tags,
        "samples": total,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / total,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_correct / total,
        "macro_top1_accuracy": sum(item["top1_accuracy"] for item in dataset_metrics) / len(dataset_metrics),
        "current_task_top1_accuracy": dataset_metrics[-1]["top1_accuracy"],
        "temperature": temperature,
        "image_weight": image_weight,
        "text_weight": 1.0 - image_weight,
        "datasets": dataset_metrics,
        "confusion_matrix": confusion.tolist(),
    }
    save_confusion_csv(stage_dir / "routing_confusion.csv", confusion, tags)
    (stage_dir / "routing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[STAGE {stage_id:02d} SUMMARY] experts={stage_id + 1} "
        f"top1={summary['top1_accuracy']:.4f} top3={summary['top3_accuracy']:.4f}"
    )
    return summary


def save_stagewise_summary(output_dir: Path, datasets: Sequence[DatasetSpec], summaries: Sequence[dict[str, Any]]) -> None:
    tags = [spec.tag for spec in datasets]
    matrix: list[list[float | None]] = [[None for _ in datasets] for _ in datasets]
    for summary in summaries:
        stage_id = int(summary["stage_id"])
        for metric in summary["datasets"]:
            matrix[stage_id][int(metric["expert_id"])] = float(metric["top1_accuracy"])
    aggregate = {
        "dataset_tags": tags,
        "stage_summaries": list(summaries),
        "stage_by_dataset_top1_accuracy": matrix,
    }
    (output_dir / "stagewise_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "stagewise_top1_accuracy.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage_id", "stage_tag", *tags])
        for stage_id, row in enumerate(matrix):
            writer.writerow(
                [stage_id, tags[stage_id], *["" if value is None else f"{value:.8f}" for value in row]]
            )
    with (output_dir / "stagewise_overview.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["stage_id", "stage_tag", "active_experts", "samples", "micro_top1", "macro_top1", "current_task_top1", "top3"]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary["stage_id"],
                    summary["stage_tag"],
                    summary["active_experts"],
                    summary["samples"],
                    f"{summary['top1_accuracy']:.8f}",
                    f"{summary['macro_top1_accuracy']:.8f}",
                    f"{summary['current_task_top1_accuracy']:.8f}",
                    f"{summary['top3_accuracy']:.8f}",
                ]
            )


def evaluate_router(
    encoder: ClipEncoder,
    data_root: Path,
    datasets: Sequence[DatasetSpec],
    image_anchors: torch.Tensor,
    text_anchors: torch.Tensor,
    output_dir: Path,
    batch_size: int,
    max_samples: int,
    seed: int,
    temperature: float,
    image_weight: float,
) -> dict[str, Any]:
    image_anchors_device = F.normalize(image_anchors.float(), dim=-1).to(encoder.device)
    text_anchors_device = F.normalize(text_anchors.float(), dim=-1).to(encoder.device)
    tags = [spec.tag for spec in datasets]
    confusion = torch.zeros(len(datasets), len(datasets), dtype=torch.long)
    dataset_metrics: list[dict[str, Any]] = []
    total = top1_correct = top3_correct = 0
    predictions_path = output_dir / "routing_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as prediction_file:
        for true_id, spec in enumerate(datasets):
            source_file = data_root / spec.data_dir / spec.test_file
            rows = sample_rows(load_records(source_file), max_samples, seed, f"test:{spec.tag}")
            if not rows:
                raise ValueError(f"No usable test rows for {spec.tag}: {source_file}")
            dataset_total = dataset_top1 = dataset_top3 = 0
            progress = tqdm(total=len(rows), desc=f"route   {true_id:02d} {spec.tag}")
            for batch in batched(rows, batch_size):
                images, texts = open_batch(batch, data_root, spec, source_file)
                try:
                    image_features, text_features = encoder.encode(images, texts)
                finally:
                    for image in images:
                        image.close()
                weights, similarity, image_sim, text_sim = compute_routing_weights(
                    image_features,
                    text_features,
                    image_anchors_device,
                    text_anchors_device,
                    temperature,
                    image_weight,
                )
                top_k = min(3, len(datasets))
                top_ids = weights.topk(top_k, dim=-1).indices.cpu()
                weights_cpu = weights.cpu()
                similarity_cpu = similarity.cpu()
                image_sim_cpu = image_sim.cpu()
                text_sim_cpu = text_sim.cpu()
                for index, row in enumerate(batch):
                    predicted_id = int(top_ids[index, 0])
                    top3_ids = [int(value) for value in top_ids[index].tolist()]
                    is_top1 = predicted_id == true_id
                    is_top3 = true_id in top3_ids
                    confusion[true_id, predicted_id] += 1
                    dataset_total += 1
                    dataset_top1 += int(is_top1)
                    dataset_top3 += int(is_top3)
                    output = {
                        "question_id": row.get("question_id", row.get("id")),
                        "dataset_tag": spec.tag,
                        "true_expert_id": true_id,
                        "predicted_expert_id": predicted_id,
                        "predicted_expert_tag": tags[predicted_id],
                        "top1_correct": is_top1,
                        "top3_expert_ids": top3_ids,
                        "top3_expert_tags": [tags[value] for value in top3_ids],
                        "expert_weights": weights_cpu[index].tolist(),
                        "combined_similarity": similarity_cpu[index].tolist(),
                        "image_similarity": image_sim_cpu[index].tolist(),
                        "text_similarity": text_sim_cpu[index].tolist(),
                        "image": image_value(row),
                        "prompt": extract_prompt(row),
                    }
                    prediction_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                progress.update(len(batch))
            progress.close()
            total += dataset_total
            top1_correct += dataset_top1
            top3_correct += dataset_top3
            metrics = {
                "expert_id": true_id,
                "dataset_tag": spec.tag,
                "samples": dataset_total,
                "top1_correct": dataset_top1,
                "top1_accuracy": dataset_top1 / dataset_total,
                "top3_correct": dataset_top3,
                "top3_accuracy": dataset_top3 / dataset_total,
            }
            dataset_metrics.append(metrics)
            print(
                f"[ROUTING] expert={true_id:02d} tag={spec.tag:<12} n={dataset_total:6d} "
                f"top1={metrics['top1_accuracy']:.4f} top3={metrics['top3_accuracy']:.4f}"
            )

    summary = {
        "samples": total,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / total,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_correct / total,
        "temperature": temperature,
        "image_weight": image_weight,
        "text_weight": 1.0 - image_weight,
        "datasets": dataset_metrics,
        "confusion_matrix": confusion.tolist(),
    }
    save_confusion_csv(output_dir / "routing_confusion.csv", confusion, tags)
    (output_dir / "routing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def select_datasets(selection: str, task_order: str) -> list[DatasetSpec]:
    ordered = list(STANDARD_DATASETS)
    if task_order == "reverse":
        ordered.reverse()
    if selection.strip().lower() == "all":
        return ordered
    requested = [value.strip() for value in selection.split(",") if value.strip()]
    by_tag = {spec.tag: spec for spec in ordered}
    unknown = [tag for tag in requested if tag not in by_tag]
    if unknown:
        raise ValueError(f"Unknown dataset tags: {unknown}. Valid tags: {list(by_tag)}")
    # Preserve continual-learning order rather than the comma-list order.
    requested_set = set(requested)
    return [spec for spec in ordered if spec.tag in requested_set]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--lora-checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--anchor-model-path", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--datasets", default="all", help="Comma-separated tags or 'all'.")
    parser.add_argument("--task-order", choices=("standard", "reverse"), default="standard")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-anchor-samples", type=int, default=0, help="0 uses the full training split.")
    parser.add_argument("--max-test-samples", type=int, default=0, help="0 uses the full test split.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-anchors", type=Path, default=None)
    parser.add_argument("--skip-lora-validation", action="store_true")
    parser.add_argument(
        "--evaluation-mode",
        choices=("stage-wise", "final"),
        default="stage-wise",
        help="stage-wise evaluates every prefix; final evaluates only the complete selected expert bank.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.lora_checkpoint_root = args.lora_checkpoint_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = select_datasets(args.datasets, args.task_order)
    if not datasets:
        raise ValueError("No datasets selected")

    if args.skip_lora_validation:
        lora_manifest: list[dict[str, Any]] = []
    else:
        lora_manifest = validate_lora_experts(args.lora_checkpoint_root, datasets)

    encoder = ClipEncoder(args.anchor_model_path, args.device)
    if args.reuse_anchors is not None:
        state = torch_load_cpu(args.reuse_anchors.expanduser().resolve())
        expected_tags = [spec.tag for spec in datasets]
        if list(state.get("dataset_tags", [])) != expected_tags:
            raise ValueError(
                f"Anchor task order mismatch: file={state.get('dataset_tags')} requested={expected_tags}"
            )
        image_anchors = state["image_anchors"].float()
        text_anchors = state["text_anchors"].float()
        counts = state["counts"].long()
        anchor_file = args.reuse_anchors.expanduser().resolve()
        print(f"[INFO] Reusing anchors: {anchor_file}")
    else:
        image_anchors, text_anchors, counts = build_anchors(
            encoder,
            args.data_root,
            datasets,
            args.batch_size,
            args.max_anchor_samples,
            args.seed,
        )
        anchor_file = args.output_dir / "anchor_lora_router.pt"
        torch.save(
            {
                "format_version": 1,
                "image_anchors": image_anchors,
                "text_anchors": text_anchors,
                "counts": counts,
                "dataset_tags": [spec.tag for spec in datasets],
                "expert_ids": list(range(len(datasets))),
                "anchor_model_path": args.anchor_model_path,
                "temperature": args.temperature,
                "image_weight": args.image_weight,
                "lora_experts": lora_manifest,
            },
            anchor_file,
        )
        print(f"[INFO] Saved anchor bank: {anchor_file}")

    manifest = {
        "task_order": args.task_order,
        "dataset_tags": [spec.tag for spec in datasets],
        "anchor_file": str(anchor_file),
        "anchor_counts": counts.tolist(),
        "anchor_model_path": args.anchor_model_path,
        "lora_experts": lora_manifest,
    }
    (args.output_dir / "expert_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cache_paths = cache_test_features(
        encoder,
        args.data_root,
        datasets,
        args.output_dir,
        args.batch_size,
        args.max_test_samples,
        args.seed,
    )
    if args.evaluation_mode == "stage-wise":
        stage_ids = range(len(datasets))
    else:
        stage_ids = [len(datasets) - 1]
    summaries = [
        evaluate_cached_stage(
            encoder,
            datasets,
            cache_paths,
            image_anchors,
            text_anchors,
            stage_id,
            args.output_dir,
            args.batch_size,
            args.temperature,
            args.image_weight,
        )
        for stage_id in stage_ids
    ]
    save_stagewise_summary(args.output_dir, datasets, summaries)
    summary = summaries[-1]
    print("=" * 80)
    print(f"Final evaluated stage top-1 accuracy: {summary['top1_accuracy']:.4f}")
    print(f"Final evaluated stage top-3 accuracy: {summary['top3_accuracy']:.4f}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
