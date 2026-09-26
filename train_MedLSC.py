#!/usr/bin/env python3
"""V10 orchestration: department-aware calibration with learned adaptive anchor/query fusion."""

import json
import os
import subprocess
from pathlib import Path

import train as v5


_V5Base = v5.TrainSeparateV5


class TrainSeparateV10DepartmentAnchorAdaptiveFusionDeltaMerge(_V5Base):
    """Reuse V5 curriculum/checkpoint handling with a learned Eq.16 fusion gate."""

    def __init__(self, args):
        super().__init__(args)

        self.train_script = self.repo_dir / "llava" / "train" / "train_cali_department_anchor_adaptive_fusion_delta_merge.py"
        self.eval_script = (
            self.repo_dir
            / "llava"
            / "eval"
            / "eval_medlsc_cr_with_routing_department_anchor_adaptive_fusion_delta_merge.py"
        )

        order = "REVERSED" if self.is_reversed_order else "STANDARD"
        run_name = (
            f"finetune_Hospital_PROG_{order}_V10_DEPT_ANCHOR_ADAPTIVE_FUSION_DELTA_MERGE-"
            f"{args.lora_rank}-{args.lora_alpha}_llava_med_v1.5"
        )
        self.output_base = self.checkpoint_root / run_name
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.prep_dir = self.output_base / "prepared_json"
        self.prep_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "version": "v10_department_anchor_adaptive_fusion_delta_merge",
            "routing_training": "supervised query-key router",
            "calibration": "router supervision loss + task loss + learned adaptive anchor/query fusion gate",
            "stage3_and_inference_routing": "anchor-inferred department mask + query-key top-k outsiders + learned adaptive hybrid score",
            "adaptive_fusion": {
                "formula": "z_i(x) = lambda_Q(x) log p_i^Q(x) + lambda_A(x) log p_i^A(x)",
                "gate": "[lambda_A(x), lambda_Q(x)] = softmax([fc_A(p^A_C(x)), fc_Q(p^Q_C(x))])",
                "initialization": "fc weights are zero; biases initialize to ANCHOR_COEFFICIENT and 1-ANCHOR_COEFFICIENT",
                "trained_in": "calibration phase",
            },
            "adapter_operator": {
                "formula": "delta(x) = sum_i weight_i * B_i(A_i(x))",
                "equivalent_delta_weight": "DeltaW_mix = sum_i weight_i * (B_i A_i)",
                "forward": "base(x) + scaling * sum_i weight_i * B_i(A_i(x))",
                "rank_note": "The exact merged DeltaW may have rank up to num_experts * rank.",
                "cross_terms": "No B_i A_j cross-expert terms are introduced.",
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
            },
        }
        with (self.output_base / "v10_department_anchor_adaptive_fusion_delta_merge_operator.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)

        print("=" * 80)
        print("MSLoRA V10: learned adaptive anchor/query fusion + exact delta merge")
        print(f"Output root: {self.output_base}")
        print("Eq.16: learned lambda_A(x), lambda_Q(x) replace the fixed anchor coefficient.")
        print("Operator: delta(x)=sum_i w_i * B_i(A_i(x)); no cross-expert terms.")
        print("=" * 80)

    def _prepare_evaluation_anchors(self) -> None:
        """Build frozen CLIP anchors and full test caches before model training."""
        anchor_script = self.repo_dir / "establish_anchor.py"
        anchor_dir = Path(os.environ.get(
            "ANCHOR_RESULTS_DIR", str(self.output_base / "anchors")
        )).expanduser().resolve()
        data_root = Path(os.environ.get(
            "ANCHOR_DATA_ROOT", str(self.data_root)
        )).expanduser().resolve()
        task_order = "reverse" if self.is_reversed_order else "standard"
        command = [
            self.python_bin, str(anchor_script),
            "--data-root", str(data_root),
            "--lora-checkpoint-root", str(self.lora_checkpoint_root),
            "--output-dir", str(anchor_dir),
            "--task-order", task_order,
            "--datasets", "all",
            "--anchor-model-path", os.environ.get(
                "ANCHOR_MODEL_PATH", "openai/clip-vit-large-patch14-336"
            ),
            "--device", os.environ.get("ANCHOR_DEVICE", "cuda:0"),
            "--batch-size", os.environ.get("ANCHOR_BATCH_SIZE", "16"),
            "--max-anchor-samples", "0",
            "--max-test-samples", "0",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.devices.split(",")[0].strip()
        print(f"[MedLSC] Preparing evaluation anchors and caches: {anchor_dir}")
        # A separate process releases CLIP memory before DeepSpeed starts.
        subprocess.run(command, check=True, env=env)

        anchor_file = anchor_dir / "anchor_lora_router.pt"
        cache_dir = anchor_dir / "test_feature_cache"
        if not anchor_file.is_file() or anchor_file.stat().st_size == 0:
            raise FileNotFoundError(f"Anchor preparation did not produce: {anchor_file}")
        for dataset in self.datasets:
            matches = list(cache_dir.glob(f"*_{dataset.tag}_test_features.pt"))
            if len(matches) != 1 or matches[0].stat().st_size == 0:
                raise RuntimeError(
                    f"Expected one nonempty test-feature cache for {dataset.tag} "
                    f"in {cache_dir}; found {len(matches)}."
                )
        assets = {
            "anchor_results_dir": str(anchor_dir),
            "anchor_file": str(anchor_file),
            "test_feature_cache_dir": str(cache_dir),
            "data_root": str(data_root),
            "task_order": task_order,
            "dataset_tags": [dataset.tag for dataset in self.datasets],
        }
        manifest = self.output_base / "evaluation_assets.json"
        manifest.write_text(json.dumps(assets, indent=2) + "\n", encoding="utf-8")
        print(f"[MedLSC] Evaluation artifacts ready; paths saved to {manifest}")

    def run(self) -> None:
        self._prepare_evaluation_anchors()
        super().run()


def main():
    # V5's parser/path validation is retained; only the instantiated trainer is
    # replaced.  This preserves START_FROM_TASK and reverse-order support.
    v5.TrainSeparateV5 = TrainSeparateV10DepartmentAnchorAdaptiveFusionDeltaMerge
    v5.main()


if __name__ == "__main__":
    main()
