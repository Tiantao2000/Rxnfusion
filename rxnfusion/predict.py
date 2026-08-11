from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers.utils import logging as transformers_logging

from .bio_tokenizer import BioReactTokenizer
from .data import as_float, filter_split, reaction_pair, read_rows
from .metrics import classification, json_ready, regression
from .models import load_model
from .tokenizer import MolformerTokenizer

ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def choose_device(raw: str) -> torch.device:
    return torch.device("cuda:0" if raw == "auto" and torch.cuda.is_available() else ("cpu" if raw == "auto" else raw))


def batched(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict_rows(rows: list[dict[str, str]], config: dict[str, Any], checkpoint: Path, *, batch_size: int, max_length: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformers_logging.set_verbosity_error()
    kind = str(config["model_kind"])
    dtype = torch.float32
    labels = config.get("label_columns") or [config.get("label_column", "label")]
    labels = [str(item) for item in labels]
    if kind == "kcat":
        tokenizer = BioReactTokenizer(config.get("vocab_path"))
        model = load_model(checkpoint, kind="kcat", vocab_size=tokenizer.vocab_size, num_labels=1, problem_type="regression", device=device, dtype=dtype)
    else:
        tokenizer = MolformerTokenizer(vocab_file=str(config.get("vocab_path", ROOT / "assets" / "molformer_vocab.json")))
        model = load_model(
            checkpoint,
            kind=kind,
            vocab_size=int(config.get("vocab_size", tokenizer.vocab_size)),
            num_labels=int(config.get("num_labels", 1)),
            problem_type=str(config.get("problem_type", "regression")),
            label_sizes={str(k): int(v) for k, v in config.get("label_sizes", {}).items()} or None,
            device=device,
            dtype=dtype,
        )

    result_rows: list[dict[str, Any]] = []
    true_values: list[Any] = []
    predicted_values: list[Any] = []
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for batch in batched(rows, batch_size):
            if kind == "kcat":
                reactions = [str(row[config.get("reaction_column", "reaction")]) for row in batch]
                sequences = [str(row[config.get("sequence_column", "sequence")]) for row in batch]
                tokens = tokenizer.batch_encode_pairs(reactions, sequences, max_length=max_length, padding=True)
                tensors = {key: torch.tensor(tokens[key], dtype=torch.long, device=device) for key in ("input_ids", "attention_mask", "token_type_ids")}
                inputs_embeds = model.embed(tensors["input_ids"]) + model.token_type_embed(tensors["token_type_ids"].clamp(0, 1))
                output = model(inputs_embeds=inputs_embeds, attention_mask=tensors["attention_mask"], return_dict=True)
            else:
                pairs = [reaction_pair(row, config) for row in batch]
                encoded = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], max_length=max_length, truncation=True, padding=True, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], return_dict=True)

            if kind == "hierarchical":
                logits_by_label = output.logits
                for offset, row in enumerate(batch):
                    item = dict(row)
                    row_true = []
                    row_pred = []
                    for label in labels:
                        logits = logits_by_label[label][offset]
                        probs = torch.softmax(logits, dim=-1)
                        pred = int(torch.argmax(probs))
                        value = as_float(row.get(label))
                        true = int(value) if value is not None else None
                        item[f"pred_{label}"] = pred
                        item[f"pred_{label}_prob"] = float(probs[pred])
                        row_true.append(true)
                        row_pred.append(pred)
                    item["correct"] = int(all(t is not None and t == p for t, p in zip(row_true, row_pred)))
                    result_rows.append(item)
                continue

            logits = output.logits
            if kind == "kcat" or config.get("problem_type") == "regression":
                values = logits.flatten().float().cpu().tolist()
                for row, pred in zip(batch, values):
                    item = dict(row)
                    item["prediction"] = pred
                    result_rows.append(item)
                    true = as_float(row.get(labels[0]))
                    if true is not None:
                        true_values.append(true)
                        predicted_values.append(pred)
            else:
                probs = torch.softmax(logits, dim=-1).float().cpu()
                preds = torch.argmax(probs, dim=-1).tolist()
                for row, pred, prob in zip(batch, preds, probs.tolist()):
                    item = dict(row)
                    item["prediction"] = int(pred)
                    item["prediction_probability"] = float(prob[pred])
                    result_rows.append(item)
                    true = as_float(row.get(labels[0]))
                    if true is not None:
                        true_values.append(int(true))
                        predicted_values.append(int(pred))
                        probabilities.append(prob)

    if kind == "hierarchical":
        metrics: dict[str, Any] = {"task_type": "hierarchical_classification", "num_samples": len(result_rows)}
        for label in labels:
            truth = [int(as_float(row[label])) for row in rows if as_float(row.get(label)) is not None]
            pred = [int(item[f"pred_{label}"]) for item, row in zip(result_rows, rows) if as_float(row.get(label)) is not None]
            metrics[label] = classification(truth, pred, [])
    elif (true_values and config.get("problem_type") == "regression") or kind == "kcat":
        metrics = regression(true_values, predicted_values) if true_values else {"task_type": "regression", "num_samples": len(result_rows)}
    elif true_values:
        metrics = classification(true_values, predicted_values, probabilities)
    else:
        metrics = {"task_type": str(config.get("problem_type", "prediction")), "num_samples": len(result_rows)}
    return result_rows, json_ready(metrics)


def write_outputs(rows: list[dict[str, Any]], metrics: dict[str, Any], predictions: Path, metrics_path: Path) -> None:
    predictions.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one inference checkpoint on a CSV.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = read_config(args.config)
    max_length = int(args.max_length or config.get("max_length", 768))
    rows = filter_split(read_rows(args.input), config)
    if not rows:
        raise ValueError("No rows remain after split filtering")
    predictions, metrics = predict_rows(rows, config, args.checkpoint, batch_size=args.batch_size, max_length=max_length, device=choose_device(args.device))
    write_outputs(predictions, metrics, args.predictions, args.metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
