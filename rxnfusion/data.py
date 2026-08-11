from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV input is supported.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def reaction_pair(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    reactants = cell(row[config.get("reactants_column", "reactants")])
    products = cell(row[config.get("products_column", "products")])
    reagents_column = config.get("reagents_column", "reagents")
    reagents = cell(row.get(reagents_column, "")) if reagents_column else ""
    if reagents:
        text_a = f"{reactants}>{reagents}"
    else:
        text_a = reactants
    return text_a, products


def filter_split(rows: Iterable[dict[str, str]], config: dict[str, Any]) -> list[dict[str, str]]:
    split_column = config.get("split_column")
    split_value = config.get("split_value")
    if not split_column or split_value is None:
        return list(rows)
    return [row for row in rows if cell(row.get(split_column)) == str(split_value)]


def as_float(value: Any) -> float | None:
    text = cell(value)
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
