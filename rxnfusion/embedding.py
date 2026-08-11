from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers.utils import logging as transformers_logging

from .models import load_model
from .tokenizer import MolformerTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = ROOT / "assets" / "molformer_vocab.json"


def pooled_embedding(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return torch.cat([hidden[:, 0], mean], dim=-1)


def embed_reactions(
    reactions: list[str],
    checkpoint: str | Path,
    *,
    batch_size: int = 16,
    max_length: int = 300,
    device: str = "auto",
    dtype: str = "float32",
    vocab: str | Path = DEFAULT_VOCAB,
) -> torch.Tensor:
    if not reactions:
        raise ValueError("At least one reaction is required")
    transformers_logging.set_verbosity_error()
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_device = torch.device(device)
    tokenizer = MolformerTokenizer(vocab_file=str(vocab))
    model = load_model(checkpoint, kind="embedding", vocab_size=tokenizer.vocab_size, device=torch_device, dtype=dtype_map[dtype])
    output = []
    with torch.inference_mode():
        for start in range(0, len(reactions), batch_size):
            tokens = tokenizer(
                reactions[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(torch_device) for key, value in tokens.items()}
            result = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], return_dict=True)
            output.append(pooled_embedding(result.last_hidden_state, tokens["attention_mask"]).float().cpu())
    return torch.cat(output)


def _read_reactions(args: argparse.Namespace) -> list[str]:
    values = list(args.reactions) + list(args.reaction)
    if args.input_file:
        values.extend(line.strip() for line in args.input_file.read_text().splitlines() if line.strip())
    if args.input_csv:
        with args.input_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if args.csv_column not in (reader.fieldnames or []):
                raise ValueError(f"Missing column {args.csv_column!r} in {args.input_csv}")
            values.extend(row[args.csv_column].strip() for row in reader if row[args.csv_column].strip())
    if not values:
        raise ValueError("Pass a reaction, --input-file, or --input-csv")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RxnLM reaction embeddings.")
    parser.add_argument("reactions", nargs="*")
    parser.add_argument("--reaction", action="append", default=[])
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--csv-column", default="reaction")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()
    reactions = _read_reactions(args)
    embeddings = embed_reactions(reactions, args.checkpoint, batch_size=args.batch_size, max_length=args.max_length, device=args.device, dtype=args.dtype, vocab=args.vocab)
    if not args.output:
        for reaction, vector in zip(reactions, embeddings.tolist()):
            print(json.dumps({"reaction": reaction, "embedding": vector}))
    elif args.output.suffix.lower() == ".npy":
        import numpy as np
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, embeddings.numpy())
    elif args.output.suffix.lower() == ".pt":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"reactions": reactions, "embeddings": embeddings}, args.output)
    elif args.output.suffix.lower() == ".jsonl":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for reaction, vector in zip(reactions, embeddings.tolist()):
                handle.write(json.dumps({"reaction": reaction, "embedding": vector}) + "\n")
    else:
        raise ValueError("Output must use .npy, .pt, or .jsonl")
    print(f"embedded {len(reactions)} reactions -> {tuple(embeddings.shape)}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
