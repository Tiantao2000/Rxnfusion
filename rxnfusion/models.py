from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .modeling import (
    RxnLMConfig,
    RxnLMForHierarchicalClassification,
    RxnLMForMaskedLM,
    RxnLMForSequenceClassification,
)


def _state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint}")
    return state


def _load_compatible(model: torch.nn.Module, state: dict[str, torch.Tensor], *, prefix: str = "model.") -> None:
    current = model.state_dict()
    filtered = {}
    for key, value in state.items():
        candidate = key[len(prefix):] if key.startswith(prefix) else key
        if candidate in current and tuple(value.shape) == tuple(current[candidate].shape):
            filtered[candidate] = value
    if not filtered:
        raise ValueError("No compatible model weights were found in the checkpoint.")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    if missing:
        print(f"warning: {loaded}/{len(current)} tensors loaded; missing={missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"warning: ignored unexpected tensors={unexpected[:5]}", file=sys.stderr)


def load_model(
    checkpoint: str | Path,
    *,
    kind: str,
    vocab_size: int = 2362,
    num_labels: int = 1,
    problem_type: str = "regression",
    label_sizes: dict[str, int] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    config = RxnLMConfig(
        vocab_size=vocab_size,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    if kind == "embedding":
        model = RxnLMForMaskedLM(config)
    elif kind == "hierarchical":
        if not label_sizes:
            raise ValueError("label_sizes is required for hierarchical tasks")
        model = RxnLMForHierarchicalClassification(config, label_sizes=label_sizes)
    elif kind in {"reaction", "kcat"}:
        model = RxnLMForSequenceClassification(config)
        if kind == "kcat":
            model.token_type_embed = torch.nn.Embedding(2, config.hidden_size)
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    _load_compatible(model, _state_dict(checkpoint))
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model


def checkpoint_hparams(checkpoint: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    return dict(payload.get("hyper_parameters", {}))
