#!/usr/bin/env python3
"""V10 orchestration: department-aware calibration with learned adaptive anchor/query fusion."""

import json

import train as v5


_V5Base = v5.TrainSeparateV5


class TrainSeparateV10DepartmentAnchorAdaptiveFusionDeltaMerge(_V5Base):
    """Reuse V5 curriculum/checkpoint handling with a learned fusion gate."""

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
        print("V10: learned adaptive anchor/query fusion + exact delta merge")
        print(f"Output root: {self.output_base}")
        print("Eq.16: learned lambda_A(x), lambda_Q(x) replace the fixed anchor coefficient.")
        print("Operator: delta(x)=sum_i w_i * B_i(A_i(x)); no cross-expert terms.")
        print("=" * 80)


def main():
    # V5's parser/path validation is retained; only the instantiated trainer is
    # replaced.  This preserves START_FROM_TASK and reverse-order support.
    v5.TrainSeparateV5 = TrainSeparateV10DepartmentAnchorAdaptiveFusionDeltaMerge
    v5.main()


if __name__ == "__main__":
    main()
