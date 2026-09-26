#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def remap_state_dict(state_dict, target_task_id: int):
    src_token = "task_0_lora"
    dst_token = f"task_{int(target_task_id)}_lora"
    remapped = {}
    for key, value in state_dict.items():
        remapped[key.replace(src_token, dst_token)] = value
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a single-expert LoRA checkpoint into a target task slot."
    )
    parser.add_argument("--input", required=True, help="Path to source cl_lora_task0.bin or cl_lora.bin")
    parser.add_argument("--target-task-id", type=int, required=True, help="Target task slot id")
    parser.add_argument("--output", required=True, help="Path to save remapped checkpoint")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    state_dict = torch.load(input_path, map_location="cpu")
    remapped = remap_state_dict(state_dict, target_task_id=args.target_task_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(remapped, output_path)
    print(f"Remapped {input_path} -> {output_path} (task_{args.target_task_id})")


if __name__ == "__main__":
    main()
