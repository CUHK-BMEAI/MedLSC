#!/usr/bin/env python3
"""
v5: Continual Router Learning with Task Calibration and Evaluation
===========================================================================

This version adds:
  1. Calibration phase after router training (loss = supervised_loss + task_loss, LR=2e-5, 1 epoch)
  2. Evaluation on 200 sampled training records after each phase
  3. Final full-dataset test set evaluation across all 14 datasets

Minimal changes from v2 to maintain control.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


class DatasetConfig:
    """Configuration for a single dataset in the reversed pipeline."""

    def __init__(
        self,
        task_id: int,
        name: str,
        tag: str,
        group: str,
        data_dir: str,
        json_file: str,
        epochs: int = 1,
        batch_size: int = 16,
    ):
        self.task_id = task_id
        self.name = name
        self.tag = tag
        self.group = group
        self.data_dir = data_dir
        self.json_file = json_file
        self.epochs = epochs
        self.batch_size = batch_size


class TrainSeparateV5:
    """
    Orchestrates continual router learning with pre-trained dataset-specific LoRAs,
    plus calibration phase and evaluation.

    Pipeline (standard order):
    1. COVID-CXP (task 0)
    2. Slake-CTXR (task 1)
    3. IU-X-Ray (task 2)
    4. Slake-MRI (task 3)
    5. PCAM (task 4)
    6. PathVQA (task 5)
    7. HAM/Skin8 (task 6)
    8. Derm/Fitzpatrick (task 7)
    9. Yangxi (task 8)
    10. OCT-C8 (task 9)
    11. Cervical (task 10)
    12. Kvasir-VQA (task 11)
    13. HyperKvasir (task 12)
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_dir = Path(args.repo_dir)
        self.checkpoint_root = Path(args.checkpoint_root)
        self.lora_checkpoint_root = Path(args.lora_checkpoint_root)
        self.data_root = Path(args.data_root)
        self.python_bin = args.python_bin
        self.devices = args.devices
        self.port = args.port
        self.use_deepspeed = args.use_deepspeed
        self.deepspeed_config = Path(args.deepspeed_config)

        self.sample_per_task = int(os.getenv("SAMPLE_PER_TASK", 280))
        self.sample_seed = int(os.getenv("SAMPLE_SEED", 42))
        
        # Calibration config (new)
        self.calibration_lr = float(os.getenv("CALIBRATION_LR", 2e-5))
        self.calibration_epoch = int(os.getenv("CALIBRATION_EPOCH", 1))
        
        # Evaluation config (new)
        self.eval_sample_size = int(os.getenv("EVAL_SAMPLE_SIZE", 200))
        self.eval_sample_seed = int(os.getenv("EVAL_SAMPLE_SEED", 123))
        self.task_order = str(os.getenv("TASK_ORDER", "standard")).strip().lower()
        if self.task_order not in {"standard", "reverse", "reversed"}:
            raise ValueError(
                f"Unsupported TASK_ORDER={self.task_order!r}. Expected 'standard' or 'reverse'."
            )
        self.is_reversed_order = self.task_order in {"reverse", "reversed"}

        # Model paths
        self.pretrained_model_path = Path(args.pretrained_model_path)
        self.train_script = self.repo_dir / "llava" / "train" / "train_cali.py"
        self.eval_script = self.repo_dir / "llava" / "eval" / "eval_medlsc_cr_with_routing.py"
        self.report_script = self.repo_dir / "llava" / "eval" / "report_results.py"

        # Output directory
        run_name = "finetune_Hospital_PROG_STANDARD_V12-64-64_llava_med_v1.5"
        if self.is_reversed_order:
            run_name += "_REVERSED"
        self.output_base = self.checkpoint_root / run_name
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.prep_dir = self.output_base / "prepared_json"
        self.prep_dir.mkdir(parents=True, exist_ok=True)

        # Previous LoRA tracking for continual learning
        self.previous_lora_paths: List[Path] = []
        self.last_lora_file: Optional[Path] = None

        # Define datasets in standard order
        self.datasets = self._create_dataset_configs()

    def _create_dataset_configs(self) -> List[DatasetConfig]:
        """Create dataset configurations in standard order (13-task)."""
        epoch_default = int(os.getenv("EPOCH_ALL", 4))
        configs = [
            DatasetConfig(
                task_id=0,
                name="COVID-CXP",
                tag="covid-CXP",
                group="radiology",
                data_dir="COVID_CXP",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_COVID_CXP", epoch_default)),
                batch_size=int(os.getenv("BATCH_COVID_CXP", 32)),
            ),
            DatasetConfig(
                task_id=1,
                name="Slake-CTXR",
                tag="slake-ctxr",
                group="radiology",
                data_dir="Slake-VQARad",
                json_file="train_ct_xray.json",
                epochs=int(os.getenv("EPOCH_SLAKE_CTXR", epoch_default)),
                batch_size=int(os.getenv("BATCH_SLAKE_CTXR", 32)),
            ),
            DatasetConfig(
                task_id=2,
                name="IU-X-Ray",
                tag="iu-x-ray",
                group="radiology",
                data_dir="IU-X-Ray",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_IU_XRAY", epoch_default)),
                batch_size=int(os.getenv("BATCH_IU_XRAY", 16)),
            ),
            DatasetConfig(
                task_id=3,
                name="Slake-MRI",
                tag="slake-mri",
                group="radiology",
                data_dir="Slake-VQARad",
                json_file="train_mri.json",
                epochs=int(os.getenv("EPOCH_SLAKE_MRI", epoch_default)),
                batch_size=int(os.getenv("BATCH_SLAKE_MRI", 32)),
            ),
            DatasetConfig(
                task_id=4,
                name="PCAM",
                tag="PCAM",
                group="pathology",
                data_dir="PCam",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_PCAM", epoch_default)),
                batch_size=int(os.getenv("BATCH_PCAM", 16)),
            ),
            DatasetConfig(
                task_id=5,
                name="PathVQA",
                tag="pathvqa",
                group="pathology",
                data_dir="PathVQA",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_PATHVQA", epoch_default)),
                batch_size=int(os.getenv("BATCH_PATHVQA", 16)),
            ),
            DatasetConfig(
                task_id=6,
                name="HAM/Skin8",
                tag="HAM_skin8",
                group="derm",
                data_dir="HAM_skin8",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_HAM_SKIN8", epoch_default)),
                batch_size=int(os.getenv("BATCH_HAM_SKIN8", 16)),
            ),
            DatasetConfig(
                task_id=7,
                name="Derm/Fitzpatrick",
                tag="derm",
                group="derm",
                data_dir="Fitzpatrick",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_DERM", epoch_default)),
                batch_size=int(os.getenv("BATCH_DERM", 16)),
            ),
            DatasetConfig(
                task_id=8,
                name="Yangxi",
                tag="Yangxi",
                group="ophthalmology",
                data_dir="Yangxi",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_YANGXI", epoch_default)),
                batch_size=int(os.getenv("BATCH_YANGXI", 16)),
            ),
            DatasetConfig(
                task_id=9,
                name="OCT-C8",
                tag="oct-c8",
                group="ophthalmology",
                data_dir="Retinal_OCT_C8",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_OCT_C8", epoch_default)),
                batch_size=int(os.getenv("BATCH_OCT_C8", 16)),
            ),
            DatasetConfig(
                task_id=10,
                name="Cervical",
                tag="cervical",
                group="cervical",
                data_dir="cervical",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_CERVICAL", epoch_default)),
                batch_size=int(os.getenv("BATCH_CERVICAL", 16)),
            ),
            DatasetConfig(
                task_id=11,
                name="Kvasir-VQA",
                tag="kvasir",
                group="endoscopy",
                data_dir="Kvasir-VQA",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_KVASIR", epoch_default)),
                batch_size=int(os.getenv("BATCH_KVASIR", 16)),
            ),
            DatasetConfig(
                task_id=12,
                name="HyperKvasir",
                tag="hyperkvasir",
                group="endoscopy",
                data_dir="HyperKvasir",
                json_file="train.json",
                epochs=int(os.getenv("EPOCH_HYPERKVASIR", epoch_default)),
                batch_size=int(os.getenv("BATCH_HYPERKVASIR", 16)),
            ),
        ]
        if self.is_reversed_order:
            configs = list(reversed(configs))
        for idx, dataset in enumerate(configs):
            dataset.task_id = idx
        return configs

    def _get_lora_checkpoint_path(self, dataset_tag: str) -> Path:
        """Get the path to the pre-trained LoRA checkpoint directory for a dataset."""
        checkpoint_path = self.lora_checkpoint_root / dataset_tag
        lora_file = checkpoint_path / "cl_lora_task0.bin"
        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_file}")
        return checkpoint_path

    def _resolve_lora_source_file(self, checkpoint_path: Path) -> Path:
        """Resolve a dataset LoRA source path to an actual checkpoint file."""
        if checkpoint_path.is_file():
            return checkpoint_path

        candidates = [
            checkpoint_path / "cl_lora_task0.bin",
            checkpoint_path / "cl_lora.bin",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"No standalone LoRA checkpoint found under: {checkpoint_path}")

    def _prepare_current_task_lora(self, source_lora_file: Path, dataset: DatasetConfig, output_dir: Path) -> Path:
        """
        Remap the standalone current-task expert into the correct task slot.

        This guarantees that task-k router training sees the current expert in
        task_{k}_lora, rather than incorrectly loading it into task 0.
        """
        remap_dir = output_dir / "remapped_loras"
        remapped_lora_file = remap_dir / f"cl_lora_task{dataset.task_id}.bin"

        if dataset.task_id == 0:
            remapped_lora_file.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source_lora_file, remapped_lora_file)
            print(f"[INFO] Copied task-0 LoRA without remap: {source_lora_file} -> {remapped_lora_file}")
            return remapped_lora_file

        remap_script = self.repo_dir / "llava" / "remap_lora_task_slot.py"
        cmd = [
            self.python_bin,
            str(remap_script),
            "--input",
            str(source_lora_file),
            "--target-task-id",
            str(dataset.task_id),
            "--output",
            str(remapped_lora_file),
        ]
        print(f"[INFO] Remapping current expert into task slot: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return remapped_lora_file

    def _prepare_json(self, raw_json_path: Path, image_base_dir: Path, output_json_path: Path) -> None:
        """Convert relative image paths in JSON to absolute paths."""
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {raw_json_path}, got {type(data)}")

        for row in data:
            image = row.get("image", "")
            if isinstance(image, str) and image and not os.path.isabs(image):
                row["image"] = str((image_base_dir / image).resolve())

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load_json_records(self, json_path: Path) -> List[dict]:
        if json_path.suffix.lower() == ".jsonl":
            records: List[dict] = []
            with json_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"Expected JSON object per line in {json_path} at line {line_no}, got {type(row)}"
                        )
                    records.append(row)
            return records

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {json_path}, got {type(data)}")
        return data

    def _sample_records(self, records: List[dict], n: int, rng: random.Random) -> List[dict]:
        if n <= 0:
            return []
        if n >= len(records):
            out = list(records)
            rng.shuffle(out)
            return out
        return rng.sample(records, n)

    def _sample_records_stratified(
        self,
        records: List[dict],
        n: int,
        rng: random.Random,
        key_fn: Callable[[dict], str],
    ) -> Tuple[List[dict], Dict[str, int]]:
        buckets: Dict[str, List[dict]] = defaultdict(list)
        for row in records:
            buckets[key_fn(row)].append(row)

        keys = list(buckets.keys())
        if not keys:
            return [], {}

        rng.shuffle(keys)
        base = n // len(keys)
        remainder = n % len(keys)
        target = {k: base for k in keys}
        for k in keys[:remainder]:
            target[k] += 1

        selected: List[dict] = []
        remainder_pool: List[dict] = []
        for k in keys:
            rows = list(buckets[k])
            rng.shuffle(rows)
            take = min(target[k], len(rows))
            selected.extend(rows[:take])
            remainder_pool.extend(rows[take:])

        needed = n - len(selected)
        if needed > 0 and remainder_pool:
            if needed >= len(remainder_pool):
                selected.extend(remainder_pool)
            else:
                selected.extend(rng.sample(remainder_pool, needed))

        rng.shuffle(selected)
        bucket_sizes = {k: len(v) for k, v in buckets.items()}
        return selected, bucket_sizes

    def _sample_records_stratified_fixed(
        self,
        records: List[dict],
        n: int,
        rng: random.Random,
        key_fn: Callable[[dict], str],
        desired_keys: List[str],
    ) -> Tuple[List[dict], Dict[str, int]]:
        if not desired_keys:
            return [], {}

        buckets: Dict[str, List[dict]] = {k: [] for k in desired_keys}
        other: List[dict] = []

        for row in records:
            key = key_fn(row)
            if key in buckets:
                buckets[key].append(row)
            else:
                other.append(row)

        base = n // len(desired_keys)
        remainder = n % len(desired_keys)
        shuffled_keys = list(desired_keys)
        rng.shuffle(shuffled_keys)
        target = {k: base for k in desired_keys}
        for k in shuffled_keys[:remainder]:
            target[k] += 1

        selected: List[dict] = []
        remainder_pool: List[dict] = list(other)
        for k in desired_keys:
            rows = list(buckets[k])
            rng.shuffle(rows)
            take = min(target[k], len(rows))
            selected.extend(rows[:take])
            remainder_pool.extend(rows[take:])

        needed = n - len(selected)
        if needed > 0 and remainder_pool:
            if needed >= len(remainder_pool):
                selected.extend(remainder_pool)
            else:
                selected.extend(rng.sample(remainder_pool, needed))

        rng.shuffle(selected)
        bucket_sizes = {k: len(v) for k, v in buckets.items()}
        if other:
            bucket_sizes["Other"] = len(other)
        return selected, bucket_sizes

    def _count_key_hist(self, records: List[dict], key_fn: Callable[[dict], str]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for row in records:
            counts[key_fn(row)] += 1
        return dict(counts)

    def _covid_modality_key(self, row: dict) -> str:
        image = str(row.get("image", "")).replace("\\", "/")
        task = str(row.get("task", "")).lower()

        if "covid-disease-ct" in task:
            return "CT"
        if "covid-disease-x-ray" in task:
            return "X-Ray"

        parts = [p.lower() for p in Path(image).parts]
        for part in parts:
            if part.startswith("ct ") or part.startswith("ct-") or part.startswith("ct_") or part == "ct":
                return "CT"
            if part.startswith("x-ray") or part.startswith("xray") or part.startswith("x_ray"):
                return "X-Ray"

        conv = row.get("conversations")
        if isinstance(conv, list) and conv:
            last = conv[-1]
            if isinstance(last, dict):
                val = str(last.get("value", "")).strip().lower()
                if val == "ct":
                    return "CT"
                if val in {"x-ray", "xray"}:
                    return "X-Ray"

        return "Unknown"

    def _covid_source_key(self, row: dict) -> str:
        image = str(row.get("image", "")).replace("\\", "/").lower()
        parts = [p.lower() for p in Path(image).parts]
        if "covid" in parts:
            return "COVID"
        if "cxp" in parts:
            return "CXP"
        if "/covid/" in image:
            return "COVID"
        if "/cxp/" in image:
            return "CXP"
        return "Other"

    def _slake_mri_source_key(self, row: dict) -> str:
        image = str(row.get("image", "")).replace("\\", "/").lower()
        if "/slake_images/" in image:
            return "Slake"
        if "/vqarad_images/" in image:
            return "RadVQA"
        if "radimagenet-vqa" in image or "radimagenet_vqa" in image or "radimagenetvqa" in image:
            return "RadImageNetVQA"
        return "Other"

    def _slake_ctxr_source_key(self, row: dict) -> str:
        image = str(row.get("image", "")).replace("\\", "/").lower()
        if "/slake_images/" in image:
            return "Slake"
        if "/vqarad_images/" in image:
            return "RadVQA"
        return "Other"

    def _ham_skin8_source_key(self, row: dict) -> str:
        image = str(row.get("image", "")).replace("\\", "/").lower()
        task = str(row.get("task", "")).lower()

        if "/ham/" in image:
            return "HAM"
        if "/skin_8/" in image or "/skin8/" in image:
            return "skin8"
        if "ham" in task:
            return "HAM"
        if "skin8" in task or "skin_8" in task:
            return "skin8"
        return "Other"

    def _ensure_prepared_json(self, dataset: DatasetConfig) -> Path:
        prep_json = self.prep_dir / f"task_{dataset.tag}_abs.json"
        if prep_json.exists():
            return prep_json

        raw_json = self.data_root / dataset.data_dir / dataset.json_file
        if not raw_json.exists():
            raise FileNotFoundError(f"Dataset JSON not found: {raw_json}")

        image_base_dir = self.data_root / dataset.data_dir
        if dataset.tag in {"HAM_skin8", "covid-CXP"}:
            image_base_dir = self.data_root

        self._prepare_json(raw_json, image_base_dir, prep_json)
        return prep_json

    def _ensure_sampled_json(self, dataset: DatasetConfig) -> Path:
        prep_json = self._ensure_prepared_json(dataset)
        if self.sample_per_task <= 0:
            return prep_json

        sampled_json = self.prep_dir / f"task_{dataset.tag}_sampled_{self.sample_per_task}.json"
        if sampled_json.exists():
            return sampled_json

        records = self._load_json_records(prep_json)
        if self.sample_per_task >= len(records):
            print(
                f"[INFO] Sample size >= dataset size for {dataset.tag}. Using full set: {len(records)}"
            )
            return prep_json

        rng = random.Random(self.sample_seed + dataset.task_id)

        if dataset.tag == "covid-CXP":
            covid_rows = [row for row in records if self._covid_source_key(row) == "COVID"]
            cxp_rows = [row for row in records if self._covid_source_key(row) == "CXP"]
            other_rows = [row for row in records if self._covid_source_key(row) == "Other"]

            base = self.sample_per_task // 2
            remainder = self.sample_per_task % 2
            order = ["COVID", "CXP"]
            rng.shuffle(order)
            target = {"COVID": base, "CXP": base}
            if remainder:
                target[order[0]] += 1

            covid_sampled, covid_buckets = self._sample_records_stratified_fixed(
                covid_rows,
                target["COVID"],
                rng,
                self._covid_modality_key,
                ["CT", "X-Ray"],
            )
            cxp_sampled = self._sample_records(cxp_rows, target["CXP"], rng)

            sampled = list(covid_sampled) + list(cxp_sampled)

            if len(sampled) < self.sample_per_task:
                selected_ids = {id(row) for row in sampled}
                remainder_pool = [
                    row for row in (covid_rows + cxp_rows + other_rows) if id(row) not in selected_ids
                ]
                needed = self.sample_per_task - len(sampled)
                if remainder_pool:
                    if needed >= len(remainder_pool):
                        sampled.extend(remainder_pool)
                    else:
                        sampled.extend(rng.sample(remainder_pool, needed))

            buckets = {
                "COVID": len(covid_rows),
                "CXP": len(cxp_rows),
                "Other": len(other_rows),
            }
            selected_sources = self._count_key_hist(sampled, self._covid_source_key)
            selected_modalities = self._count_key_hist(covid_sampled, self._covid_modality_key)
            print(
                f"[INFO] Stratified sampling for {dataset.tag}: buckets={buckets}, selected={selected_sources}, covid_modalities={selected_modalities}"
            )
        elif dataset.tag == "HAM_skin8":
            sampled, buckets = self._sample_records_stratified_fixed(
                records,
                self.sample_per_task,
                rng,
                self._ham_skin8_source_key,
                ["HAM", "skin8"],
            )
            selected = self._count_key_hist(sampled, self._ham_skin8_source_key)
            print(
                f"[INFO] Stratified sampling for {dataset.tag}: buckets={buckets}, selected={selected}"
            )
        elif dataset.tag == "slake-ctxr":
            sampled, buckets = self._sample_records_stratified_fixed(
                records,
                self.sample_per_task,
                rng,
                self._slake_ctxr_source_key,
                ["Slake", "RadVQA"],
            )
            selected = self._count_key_hist(sampled, self._slake_ctxr_source_key)
            print(
                f"[INFO] Stratified sampling for {dataset.tag}: buckets={buckets}, selected={selected}"
            )
        elif dataset.tag == "slake-mri":
            sampled, buckets = self._sample_records_stratified_fixed(
                records,
                self.sample_per_task,
                rng,
                self._slake_mri_source_key,
                ["Slake", "RadVQA", "RadImageNetVQA"],
            )
            selected = self._count_key_hist(sampled, self._slake_mri_source_key)
            print(
                f"[INFO] Stratified sampling for {dataset.tag}: buckets={buckets}, selected={selected}"
            )
        else:
            sampled = self._sample_records(records, self.sample_per_task, rng)

        sampled_json.parent.mkdir(parents=True, exist_ok=True)
        with sampled_json.open("w", encoding="utf-8") as f:
            json.dump(sampled, f, ensure_ascii=False)

        print(
            f"[INFO] Wrote sampled set for {dataset.tag}: {len(sampled)} -> {sampled_json}"
        )
        return sampled_json

    def _find_stage_lora_file(self, stage_dir: Path, task_id: int) -> Path:
        """Find the saved LoRA file for a completed stage."""
        preferred = stage_dir / f"cl_lora_task{task_id}.bin"
        if preferred.exists():
            return preferred

        matches = sorted(stage_dir.glob("cl_lora_task*.bin"))
        if matches:
            return matches[0]

        fallback = stage_dir / "cl_lora.bin"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(f"No LoRA checkpoint found in completed stage directory: {stage_dir}")

    def _initialize_resume_state(self) -> Tuple[int, str]:
        """
        Initialize chain suffix and previous LoRA paths when resuming mid-curriculum.

        Returns:
            start_task_id: first task id to train in this run
            chain_suffix: accumulated chain suffix from completed tasks
        """
        start_task_id = int(getattr(self.args, "start_from_task", 0))
        if start_task_id <= 0:
            return 0, ""

        if start_task_id >= len(self.datasets):
            raise ValueError(
                f"start_from_task={start_task_id} is out of range for {len(self.datasets)} datasets"
            )

        chain_parts: List[str] = []
        resumed_lora_paths: List[Path] = []

        for dataset in self.datasets[:start_task_id]:
            chain_parts.append(dataset.tag)
            stage_dir = self.output_base / "_".join(chain_parts)
            if not stage_dir.exists():
                raise FileNotFoundError(
                    f"Cannot resume from task {start_task_id}: completed stage directory missing: {stage_dir}"
                )
            resumed_lora_paths.append(self._find_stage_lora_file(stage_dir, dataset.task_id))

        self.previous_lora_paths = resumed_lora_paths
        self.last_lora_file = resumed_lora_paths[-1] if resumed_lora_paths else None
        chain_suffix = "_".join(chain_parts)

        print("=" * 80)
        print(f"[INFO] Resuming continual pipeline from task_id={start_task_id} ({self.datasets[start_task_id].name})")
        print(f"[INFO] Reused chain suffix: {chain_suffix}")
        print("[INFO] Loaded previous stage LoRAs:")
        for path in resumed_lora_paths:
            print(f"  - {path}")
        print("=" * 80)

        return start_task_id, chain_suffix

    def _build_prog_replay_json(
        self,
        current_task_id: int,
        current_task_json: Path,
        output_json: Path,
        replay_specs: List[Tuple[int, str]],
    ) -> None:
        """Build the replay buffer JSON for continual learning."""
        cmd = [
            self.python_bin,
            str(self.repo_dir / "llava" / "build_replay_buffer.py"),
            "--current-task-json",
            str(current_task_json),
            "--current-task-id",
            str(current_task_id),
            "--replay-per-task",
            str(int(os.getenv("REPLAY_PER_TASK", 200))),
            "--seed",
            str(int(os.getenv("REPLAY_SEED", 42))),
            "--output",
            str(output_json),
        ]

        for task_id, task_json in replay_specs:
            cmd.extend(["--task-spec", f"{task_id}:{task_json}"])

        print(f"[INFO] Building replay buffer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def _compute_same_modality_flags(self, current_dataset: DatasetConfig) -> List[int]:
        """Compute is_same_modality flags for all previous datasets."""
        flags = []
        for prev_dataset in self.datasets[: current_dataset.task_id]:
            flags.append(1 if prev_dataset.group == current_dataset.group else 0)
        return flags

    def _build_deepspeed_launcher(self) -> List[str]:
        """Build the deepspeed launcher command."""
        if self.use_deepspeed:
            return ["deepspeed"]
        else:
            return [self.python_bin, "-m", "deepspeed.launcher.launch"]

    def _copy_pretrained_lora_config(self, src_path: Path, dst_path: Path) -> None:
        """Copy config.json and routing.bin from pre-trained LoRA checkpoint."""
        src_config = src_path / "config.json"
        dst_config = dst_path / "config.json"
        if src_config.exists() and not dst_config.exists():
            import shutil
            shutil.copy2(src_config, dst_config)
            print(f"[INFO] Copied config from {src_config} to {dst_config}")

    def _train_router_phase(
        self,
        dataset: DatasetConfig,
        lora_checkpoint_path: Path,
        chain_suffix: str,
        previous_lora_paths: List[Path],
    ) -> Path:
        """
        Train the router for a single dataset using pre-trained LoRA weights.
        """

        # Prepare input data
        train_json = self._ensure_sampled_json(dataset)

        # Build replay buffer if this is not the first task
        replay_specs = []
        if dataset.task_id > 0:
            replay_output_json = (
                self.output_base / f"{chain_suffix}" / f"replay_{int(os.getenv('REPLAY_PER_TASK', 200))}.json"
            )
            replay_specs = [
                (i, str(self._ensure_sampled_json(self.datasets[i])))
                for i in range(dataset.task_id)
            ]
            self._build_prog_replay_json(dataset.task_id, train_json, replay_output_json, replay_specs)
            train_json = replay_output_json

        # Setup output directory
        output_suffix = chain_suffix
        output_dir = self.output_base / output_suffix
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute same_modality flags
        same_modality_flags = self._compute_same_modality_flags(dataset)

        current_source_lora_file = self._resolve_lora_source_file(lora_checkpoint_path)
        current_task_lora_file = self._prepare_current_task_lora(
            current_source_lora_file,
            dataset,
            output_dir,
        )

        # Build training command
        max_task = dataset.task_id + 1

        cmd = self._build_deepspeed_launcher()
        cmd.extend([
            "--include",
            f"localhost:{self.devices}",
            "--master_port",
            str(self.port),
            str(self.train_script),
            "--deepspeed",
            str(self.deepspeed_config),
            "--model_path",
            str(self.pretrained_model_path),
            "--lora_enable",
            "True",
            "--lora_rank",
            str(self.args.lora_rank),
            "--lora_alpha",
            str(self.args.lora_alpha),
            "--lora_dropout",
            "0.0",
            "--max_task",
            str(max_task),
            "--current_task_id",
            str(dataset.task_id),
            "--training_phase",
            "router",
            "--seed",
            "42",
            "--alpha",
            "0",
            "--beta",
            "0",
            "--routing_global_enable",
            str(os.getenv("ROUTING_GLOBAL_ENABLE", "True")),
            "--routing_allocation_enable",
            str(os.getenv("ROUTING_ALLOCATION_ENABLE", "True")),
            "--routing_projector_hidden",
            str(os.getenv("ROUTING_PROJECTOR_HIDDEN", 64)),
            "--routing_temperature",
            str(os.getenv("ROUTING_TEMPERATURE", 1.0)),
            "--routing_projector_trainable",
            str(os.getenv("ROUTING_PROJECTOR_TRAINABLE", "True")),
            "--routing_use_task_mask",
            str(os.getenv("ROUTING_USE_TASK_MASK", "False")),
            "--kl_loss",
            str(os.getenv("KL_LOSS", "True")),
            "--kl_weight",
            str(os.getenv("KL_WEIGHT", 1)),
            "--router_supervision_loss",
            str(os.getenv("ROUTER_SUPERVISION_LOSS", "True")),
            "--router_supervision_weight",
            str(os.getenv("ROUTER_SUPERVISION_WEIGHT", 1.0)),
            "--data_path",
            str(train_json),
            "--image_folder",
            str(self.data_root),
            "--bf16",
            "True",
            "--output_dir",
            str(output_dir),
            "--num_train_epochs",
            str(dataset.epochs),
            "--per_device_train_batch_size",
            str(dataset.batch_size),
            "--gradient_accumulation_steps",
            str(self.args.gradient_accumulation_steps),
            "--evaluation_strategy",
            "no",
            "--save_strategy",
            "no",
            "--save_steps",
            "100",
            "--save_total_limit",
            "1",
            "--learning_rate",
            str(self.args.learning_rate),
            "--weight_decay",
            "0.",
            "--warmup_ratio",
            "0.03",
            "--lr_scheduler_type",
            "cosine",
            "--logging_steps",
            "1",
            "--tf32",
            "True",
            "--model_max_length",
            str(self.args.model_max_length),
            "--gradient_checkpointing",
            "True",
            "--dataloader_num_workers",
            "4",
            "--report_to",
            "wandb",
        ])

        # For the first task: load pre-trained LoRA as the foundation via previous_lora_path
        # For subsequent tasks: load all accumulated LoRAs from previous router training
        effective_lora_paths = list(previous_lora_paths) + [current_task_lora_file]

        if effective_lora_paths:
            cmd.extend(["--previous_lora_path"])
            cmd.extend([str(lora_path) for lora_path in effective_lora_paths])

        # Add same_modality flags
        if same_modality_flags:
            cmd.extend(["--is_same_modality"])
            cmd.extend([str(flag) for flag in same_modality_flags])

        print("=" * 80)
        print(f"Dataset: {dataset.name} (task_id={dataset.task_id})")
        print("Training phase: router (pre-trained LoRA base)")
        print(f"Output dir: {output_dir}")
        print(f"Source LoRA: {current_source_lora_file}")
        print(f"Remapped current-task LoRA: {current_task_lora_file}")
        print(f"Previous task LoRAs: {len(previous_lora_paths)}")
        print(f"Sample per task: {self.sample_per_task}")
        print("=" * 80)

        # Execute training
        print(f"[INFO] Training command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed for {dataset.name}")

        # Find the generated LoRA file
        lora_files = list(output_dir.glob("cl_lora*.bin"))
        if not lora_files:
            # Try the generic cl_lora.bin
            if (output_dir / "cl_lora.bin").exists():
                lora_file = output_dir / "cl_lora.bin"
            else:
                raise FileNotFoundError(f"No LoRA checkpoint found in {output_dir}")
        else:
            lora_file = lora_files[0]

        routing_file = output_dir / "routing.bin"
        if not routing_file.exists():
            raise FileNotFoundError(f"No routing checkpoint found in {output_dir}")

        print(f"[INFO] Task {dataset.task_id} router training completed. LoRA: {lora_file}")
        return lora_file

    def _calibrate_router_phase(
        self,
        dataset: DatasetConfig,
        output_dir: Path,
        previous_lora_paths: List[Path],
        current_lora_file: Path,
    ) -> None:
        """
                Calibrate the router in a single step with combined losses.

                Loss = supervised_loss + CE loss.
                KL remains disabled by default (via shell config).
        """

        # Use the same training data as router training
        train_json = self._ensure_sampled_json(dataset)
        
        # Build replay buffer if this is not the first task
        if dataset.task_id > 0:
            replay_output_json = (
                output_dir / f"replay_calib_{int(os.getenv('REPLAY_PER_TASK', 200))}.json"
            )
            replay_specs = [
                (i, str(self._ensure_sampled_json(self.datasets[i])))
                for i in range(dataset.task_id)
            ]
            self._build_prog_replay_json(dataset.task_id, train_json, replay_output_json, replay_specs)
            train_json = replay_output_json

        # Compute same_modality flags
        same_modality_flags = self._compute_same_modality_flags(dataset)

        max_task = dataset.task_id + 1

        base_cmd = self._build_deepspeed_launcher()
        base_cmd.extend([
            "--include",
            f"localhost:{self.devices}",
            "--master_port",
            str(self.port),
            str(self.train_script),
            "--deepspeed",
            str(self.deepspeed_config),
            "--model_path",
            str(self.pretrained_model_path),
            "--lora_enable",
            "True",
            "--lora_rank",
            str(self.args.lora_rank),
            "--lora_alpha",
            str(self.args.lora_alpha),
            "--lora_dropout",
            "0.0",
            "--max_task",
            str(max_task),
            "--current_task_id",
            str(dataset.task_id),
            "--seed",
            "42",
            "--alpha",
            "0",
            "--beta",
            "0",
            "--routing_global_enable",
            str(os.getenv("ROUTING_GLOBAL_ENABLE", "True")),
            "--routing_allocation_enable",
            str(os.getenv("ROUTING_ALLOCATION_ENABLE", "True")),
            "--routing_projector_hidden",
            str(os.getenv("ROUTING_PROJECTOR_HIDDEN", 64)),
            "--routing_temperature",
            str(os.getenv("ROUTING_TEMPERATURE", 1.0)),
            "--routing_projector_trainable",
            str(os.getenv("ROUTING_PROJECTOR_TRAINABLE", "True")),
            "--routing_use_task_mask",
            str(os.getenv("ROUTING_USE_TASK_MASK", "False")),
            "--kl_loss",
            "False",
            "--router_supervision_loss",
            str(os.getenv("ROUTER_SUPERVISION_LOSS", "True")),
            "--router_supervision_weight",
            str(os.getenv("ROUTER_SUPERVISION_WEIGHT", 1.0)),
            "--data_path",
            str(train_json),
            "--image_folder",
            str(self.data_root),
            "--bf16",
            "True",
            "--output_dir",
            str(output_dir),
            "--per_device_train_batch_size",
            str(dataset.batch_size),
            "--gradient_accumulation_steps",
            str(self.args.gradient_accumulation_steps),
            "--evaluation_strategy",
            "no",
            "--save_strategy",
            "no",
            "--save_steps",
            "100",
            "--save_total_limit",
            "1",
            "--learning_rate",
            str(self.calibration_lr),
            "--weight_decay",
            "0.",
            "--warmup_ratio",
            "0.03",
            "--lr_scheduler_type",
            "cosine",
            "--logging_steps",
            "1",
            "--tf32",
            "True",
            "--model_max_length",
            str(self.args.model_max_length),
            "--gradient_checkpointing",
            "True",
            "--dataloader_num_workers",
            "4",
            "--report_to",
            "wandb",
        ])

        # Match v4 ordering: completed previous task LoRAs first, current task last.
        effective_lora_paths = list(previous_lora_paths) + [current_lora_file]

        if effective_lora_paths:
            base_cmd.extend(["--previous_lora_path"])
            base_cmd.extend([str(lora_path) for lora_path in effective_lora_paths])

        # Add same_modality flags
        if same_modality_flags:
            base_cmd.extend(["--is_same_modality"])
            base_cmd.extend([str(flag) for flag in same_modality_flags])

        print("=" * 80)
        print(f"Dataset: {dataset.name} (task_id={dataset.task_id})")
        print("Training phase: calibration (supervised_loss + ce_loss)")
        print(f"Calibration LR: {self.calibration_lr}")
        print(f"Calibration epoch: {self.calibration_epoch}")
        print(f"Output dir: {output_dir}")
        print("=" * 80)

        cmd = list(base_cmd)
        cmd.extend([
            "--training_phase",
            "calibration",
            "--num_train_epochs",
            str(self.calibration_epoch),
        ])
        print(f"[INFO] Calibration command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Calibration failed for {dataset.name}")

        print(f"[INFO] Task {dataset.task_id} calibration completed.")

    def _sample_eval_data(self, dataset: DatasetConfig) -> Path:
        """
        Sample 200 random records from the current task training set for evaluation.
        """
        prep_json = self._ensure_prepared_json(dataset)
        records = self._load_json_records(prep_json)
        
        rng = random.Random(self.eval_sample_seed + dataset.task_id)
        sampled = self._sample_records(records, self.eval_sample_size, rng)
        
        # Write sampled records as JSONL (one JSON object per line)
        eval_jsonl = self.output_base / f"eval_sample_task{dataset.task_id}_{self.eval_sample_size}.jsonl"
        eval_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with eval_jsonl.open("w", encoding="utf-8") as f:
            for i, row in enumerate(sampled):
                text = row.get("text")
                gt = row.get("gt")

                # Convert training-format conversations into eval-format fields.
                conversations = row.get("conversations")
                if isinstance(conversations, list):
                    if text is None:
                        for turn in conversations:
                            if isinstance(turn, dict) and str(turn.get("from", "")).lower() == "human":
                                text = turn.get("value")
                                break
                    if gt is None:
                        for turn in conversations:
                            if isinstance(turn, dict) and str(turn.get("from", "")).lower() == "gpt":
                                gt = turn.get("value")
                                break

                if text is None:
                    text = ""
                if gt is None:
                    gt = ""

                question_id = row.get("question_id", row.get("id", i))
                eval_row = {
                    "question_id": question_id,
                    "image": row.get("image", ""),
                    "text": text,
                    "gt": gt,
                    "answer_type": row.get("answer_type", "OPEN"),
                }
                f.write(json.dumps(eval_row, ensure_ascii=False) + "\n")

        print(f"[INFO] Sampled {len(sampled)} records for evaluation: {eval_jsonl}")
        return eval_jsonl

    def _eval_on_sampled_data(
        self,
        dataset: DatasetConfig,
        eval_json: Path,
        stage_label: str,
        output_dir: Path,
        previous_lora_paths: List[Path],
        current_lora_file: Path,
    ) -> None:
        """
        Evaluate on sampled data using eval_medlsc_cr_with_routing.py
        """
        max_task = dataset.task_id + 1
        
        # Match v4 ordering: completed previous task LoRAs first, current task last.
        effective_lora_paths = list(previous_lora_paths) + [current_lora_file]
        
        
        result_dir = output_dir / f"eval_{stage_label}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.python_bin,
            str(self.eval_script),
            "--conv-mode",
            "mistral_instruct",
            "--model-path",
            str(self.pretrained_model_path),
            "--task_mask",
            str(max_task),
            "--question-file",
            str(eval_json),
            "--image-folder",
            "/",
            "--answers-file",
            str(result_dir / "answer-file.jsonl"),
            "--temperature",
            "0.0",
        ]
        
        # Add previous LoRA paths as separate CLI args (avoid single space-separated string)
        if effective_lora_paths:
            cmd.extend(["--previous_lora_path"])
            cmd.extend([str(p) for p in effective_lora_paths])

        print("=" * 80)
        print(f"Evaluating on {stage_label} with {len(self._load_json_records(eval_json))} samples")
        print("=" * 80)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.devices
        result = subprocess.run(cmd, check=False, env=env)
        if result.returncode != 0:
            print(f"[WARNING] Evaluation failed for {stage_label}, continuing...")
            return
        
        # Generate metrics
        metrics_cmd = [
            self.python_bin,
            str(self.report_script),
            "--result_file",
            str(result_dir / "answer-file.jsonl"),
            "--metric_output_file",
            str(result_dir / "metrics.txt"),
        ]
        
        metrics_result = subprocess.run(metrics_cmd, check=False)
        if metrics_result.returncode != 0:
            print(f"[WARNING] Metrics generation failed for {stage_label}, continuing...")
        
        # Print metrics
        metrics_file = result_dir / "metrics.txt"
        if metrics_file.exists():
            print(f"\n[INFO] Metrics for {stage_label}:")
            with metrics_file.open("r") as f:
                print(f.read())

    def run(self) -> None:
        """Execute the full continual router learning pipeline with calibration and evaluation."""
        print("=" * 80)
        print("v5: Router Training + Calibration + Evaluation")
        print("=" * 80)
        print(f"Checkpoint root: {self.output_base}")
        print(f"Pre-trained LoRA root: {self.lora_checkpoint_root}")
        print(f"Data root: {self.data_root}")
        print(f"Devices: {self.devices}")
        print(f"Sample per task: {self.sample_per_task}")
        print(f"Calibration LR: {self.calibration_lr}")
        print(f"Calibration epoch: {self.calibration_epoch}")
        print(f"Eval sample size: {self.eval_sample_size}")
        print(f"Task order: {'reversed' if self.is_reversed_order else 'standard'}")
        print("=" * 80)

        start_task_id, chain_suffix = self._initialize_resume_state()
        if start_task_id <= 0:
            self.previous_lora_paths = []

        for dataset in self.datasets[start_task_id:]:
            print(f"\n[INFO] Processing dataset {dataset.task_id + 1}/13: {dataset.name}")

            # Get pre-trained LoRA checkpoint for this dataset
            lora_ckpt_path = self._get_lora_checkpoint_path(dataset.tag)

            # Update chain suffix
            if not chain_suffix:
                chain_suffix = dataset.tag
            else:
                chain_suffix = f"{chain_suffix}_{dataset.tag}"

            # Train router phase with pre-trained LoRA
            lora_file = self._train_router_phase(
                dataset,
                lora_ckpt_path,
                chain_suffix,
                self.previous_lora_paths,
            )

            output_dir = self.output_base / chain_suffix
            
            # NEW: Evaluate after router training
            print(f"\n[INFO] Evaluating after router training...")
            eval_json = self._sample_eval_data(dataset)
            self._eval_on_sampled_data(
                dataset,
                eval_json,
                f"after_router_task{dataset.task_id}",
                output_dir,
                self.previous_lora_paths,
                lora_file,
            )
            
            # NEW: Calibration phase
            print(f"\n[INFO] Starting calibration phase...")
            self._calibrate_router_phase(
                dataset,
                output_dir,
                self.previous_lora_paths,
                lora_file,
            )
            
            # NEW: Evaluate after calibration
            print(f"\n[INFO] Evaluating after calibration...")
            self._eval_on_sampled_data(
                dataset,
                eval_json,
                f"after_calib_task{dataset.task_id}",
                output_dir,
                self.previous_lora_paths,
                lora_file,
            )

            # Add to previous LoRAs for next task
            self.previous_lora_paths.append(lora_file)
            self.last_lora_file = lora_file

        print("\n" + "=" * 80)
        print("v5 Training Completed Successfully!")
        print(f"Final checkpoint: {self.output_base / chain_suffix}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="v5: Continual Router Learning with Task Calibration and Evaluation"
    )

    parser.add_argument("--repo_dir", default=None, help="Repository root directory")
    parser.add_argument(
        "--checkpoint_root",
        default=None,
        help="Root directory for saving checkpoints (default: {repo_dir}/checkpoints)",
    )
    parser.add_argument(
        "--lora_checkpoint_root",
        default=None,
        help="Root directory containing pre-trained dataset-specific LoRA checkpoints",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Root directory for training data (default: {repo_dir}/data/MedLSC)",
    )
    parser.add_argument(
        "--pretrained_model_path",
        default=None,
        help="Path to pretrained base model (default: {workspace_dir}/pretrained_models/llava_med_v1.5)",
    )
    parser.add_argument("--python_bin", default="python", help="Python executable path")
    parser.add_argument("--devices", default="0,1", help="CUDA devices to use (comma-separated)")
    parser.add_argument("--port", type=int, default=29500, help="Master port for distributed training")
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        default=True,
        help="Whether to use deepspeed launcher",
    )
    parser.add_argument(
        "--deepspeed_config",
        default=None,
        help="Path to deepspeed config (default: {repo_dir}/scripts/base/zero1.json)",
    )
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--model_max_length", type=int, default=2048, help="Maximum model token length")
    parser.add_argument(
        "--start_from_task",
        type=int,
        default=0,
        help="Resume the curriculum from this task id, reusing completed stage checkpoints before it.",
    )

    args = parser.parse_args()

    # Set defaults for computed paths
    if not args.repo_dir:
        args.repo_dir = str(Path(__file__).resolve().parent)

    repo_path = Path(args.repo_dir).resolve()
    args.repo_dir = str(repo_path)
    workspace_dir = repo_path.parent

    if not args.checkpoint_root:
        args.checkpoint_root = str(repo_path / "checkpoints")

    if not args.lora_checkpoint_root:
        args.lora_checkpoint_root = str(Path(args.checkpoint_root) / "finetune_lora_each-64-64_llava_med_v1.5")

    if not args.data_root:
        args.data_root = str(repo_path / "data" / "MedLSC")

    if not args.pretrained_model_path:
        candidate = workspace_dir / "pretrained_models" / "llava_med_v1.5"
        if not candidate.exists():
            candidate = repo_path / "pretrained_models" / "llava_med_v1.5"
        args.pretrained_model_path = str(candidate)

    if not args.deepspeed_config:
        args.deepspeed_config = str(repo_path / "scripts" / "base" / "zero1.json")

    # Validate paths
    repo_path = Path(args.repo_dir)
    if not repo_path.exists():
        print(f"ERROR: Repository directory not found: {repo_path}")
        sys.exit(1)

    lora_ckpt_root = Path(args.lora_checkpoint_root)
    if not lora_ckpt_root.exists():
        print(f"ERROR: LoRA checkpoint root directory not found: {lora_ckpt_root}")
        sys.exit(1)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: Data root directory not found: {data_root}")
        sys.exit(1)

    pretrained_model_path = Path(args.pretrained_model_path)
    if not pretrained_model_path.exists():
        print(f"ERROR: Pretrained model directory not found: {pretrained_model_path}")
        sys.exit(1)

    # Create trainer and run
    trainer = TrainSeparateV5(args)
    trainer.run()


if __name__ == "__main__":
    main()
