#!/usr/bin/env python3
"""Evaluate V10 adaptive-fusion checkpoints with exact weighted LoRA-delta fusion."""

import runpy
from pathlib import Path

from llava.medlsc_utils.exact_delta_merge_patch import apply_exact_delta_merge_patch


apply_exact_delta_merge_patch()

runpy.run_path(
    str(Path(__file__).with_name("eval_medlsc_cr_with_routing.py")),
    run_name="__main__",
)
