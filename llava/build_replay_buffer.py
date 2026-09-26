#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def parse_task_spec(raw: str) -> Tuple[int, Path]:
    parts = raw.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --task-spec '{raw}'. Expected format: task_id:data_json")
    return int(parts[0]), Path(parts[1])


def load_json_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def sample_records(records: List[dict], n: int, rng: random.Random) -> List[dict]:
    if n <= 0:
        return []
    if n >= len(records):
        out = list(records)
        rng.shuffle(out)
        return out
    return rng.sample(records, n)


def normalize_image_path(item: dict, dataset_root: Path) -> dict:
    out = dict(item)
    image = out.get("image")
    if not isinstance(image, str) or image == "":
        return out

    image_path = Path(image)
    if image_path.is_absolute():
        return out

    out["image"] = str(dataset_root / image_path)
    return out


def tag_records(records: List[dict], task_id: int) -> List[dict]:
    tagged = []
    for item in records:
        obj = dict(item)
        obj["replay_task_mark"] = int(task_id)
        tagged.append(obj)
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a prog replay buffer JSON by mixing current-task data with fixed random replay from prior tasks."
    )
    parser.add_argument(
        "--current-task-json",
        type=Path,
        required=True,
        help="Current task train.json to keep as the main dataset.",
    )
    parser.add_argument(
        "--current-task-id",
        type=int,
        required=True,
        help="Current incremental task id.",
    )
    parser.add_argument(
        "--replay-per-task",
        type=int,
        default=200,
        help="Fixed number of replay samples to draw from each prior task.",
    )
    parser.add_argument(
        "--task-spec",
        action="append",
        required=True,
        help="One prior task spec: task_id:data_json (repeat this flag per prior task).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    merged: List[dict] = []

    current_records = load_json_records(args.current_task_json)
    current_records = [normalize_image_path(item, args.current_task_json.parent) for item in current_records]
    merged.extend(tag_records(current_records, -1))
    print(
        f"Current task {args.current_task_id}: source={args.current_task_json} total={len(current_records)} picked={len(current_records)}"
    )

    for raw in args.task_spec:
        task_id, json_path = parse_task_spec(raw)
        if task_id >= args.current_task_id:
            continue

        records = load_json_records(json_path)
        picked = sample_records(records, args.replay_per_task, rng)
        picked = [normalize_image_path(item, json_path.parent) for item in picked]
        merged.extend(tag_records(picked, task_id))
        print(
            f"Replay task {task_id}: source={json_path} total={len(records)} picked={len(picked)}"
        )

    rng.shuffle(merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"Wrote {len(merged)} samples to {args.output}")


if __name__ == "__main__":
    main()