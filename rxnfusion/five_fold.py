from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .predict import choose_device, predict_rows, read_config, write_outputs
from .data import filter_split, read_rows


def parse_folds(raw: str) -> list[int]:
    folds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not folds or any(value < 1 for value in folds):
        raise ValueError("--folds must contain positive integers")
    return folds


def find_dataset(root: Path, task: str, fold: int) -> Path:
    candidates = [
        root / task / "5_fold" / f"{fold}_fold" / "test.csv",
        root / task / f"{fold}_fold" / "test.csv",
        root / task / f"fold_{fold}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No test CSV found; checked: " + ", ".join(map(str, candidates)))


def find_checkpoint(root: Path, task: str, fold: int) -> Path:
    candidates = [root / task / f"fold_{fold}", root / task / f"{fold}_fold"]
    for directory in candidates:
        files = sorted(directory.glob("*.ckpt")) if directory.is_dir() else []
        if files:
            return files[-1]
    raise FileNotFoundError(f"No checkpoint found under {root / task} for fold {fold}")


def scalar_metrics(value: dict, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(scalar_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = float(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference over supplied five-fold checkpoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = read_config(args.config)
    task = str(config.get("task", args.config.stem))
    device = choose_device(args.device)
    completed: dict[str, dict] = {}
    for fold in parse_folds(args.folds):
        data_path = find_dataset(args.dataset_root, task, fold)
        checkpoint = find_checkpoint(args.checkpoint_root, task, fold)
        rows = filter_split(read_rows(data_path), config)
        predictions, metrics = predict_rows(rows, config, checkpoint, batch_size=args.batch_size, max_length=int(args.max_length or config.get("max_length", 768)), device=device)
        fold_dir = args.output_root / task / f"fold_{fold}"
        write_outputs(predictions, metrics, fold_dir / "predictions.csv", fold_dir / "metrics.json")
        completed[str(fold)] = metrics
        print(f"fold {fold}: {metrics}")

    numeric: dict[str, list[float]] = {}
    for metrics in completed.values():
        for key, value in scalar_metrics(metrics).items():
            numeric.setdefault(key, []).append(value)
    summary = {key: {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "folds": values} for key, values in sorted(numeric.items())}
    summary_payload = {"task": task, "folds": sorted(int(fold) for fold in completed), "summary": summary}
    summary_path = args.output_root / task / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
