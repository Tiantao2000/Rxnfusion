from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch

from .embedding import embed_reactions


def read_reactions(path: Path, reaction_column: str, label_column: str) -> tuple[list[str], list[str]]:
    reactions, labels = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for row in reader:
            if reaction_column in fields:
                reaction = row.get(reaction_column, "")
            else:
                reactants = row.get("reactants", "")
                reagents = row.get("reagents", "")
                products = row.get("products", "")
                reaction = f"{reactants}>{reagents}>{products}" if reagents else f"{reactants}>>{products}"
            if reaction:
                reactions.append(reaction)
                labels.append(row.get(label_column, "unknown"))
    if len(reactions) < 2:
        raise ValueError("TMAP needs at least two reactions")
    return reactions, labels


def angular(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denominator == 0 else 1.0 - float(np.dot(a, b) / denominator)


def build_edges(features: np.ndarray, neighbors: int, trees: int, seed: int) -> list[tuple[int, int, float]]:
    from annoy import AnnoyIndex
    index = AnnoyIndex(features.shape[1], "angular")
    if hasattr(index, "set_seed"):
        index.set_seed(seed)
    for row_id, vector in enumerate(features):
        index.add_item(row_id, vector.tolist())
    index.build(trees)
    edges, seen = [], set()
    for row_id in range(len(features)):
        for neighbor in index.get_nns_by_item(row_id, min(neighbors + 1, len(features))):
            if row_id == neighbor:
                continue
            edge = tuple(sorted((row_id, neighbor)))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append((edge[0], edge[1], angular(features[edge[0]], features[edge[1]])))
    return edges


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an RxnLM-only TMAP from reaction embeddings.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reaction-column", default="reaction")
    parser.add_argument("--label-column", default="class")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--annoy-trees", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    reactions, labels = read_reactions(args.input, args.reaction_column, args.label_column)
    if args.limit:
        reactions, labels = reactions[: args.limit], labels[: args.limit]
    features = embed_reactions(reactions, args.checkpoint, batch_size=args.batch_size, max_length=args.max_length, device=args.device).numpy()
    edges = build_edges(features, args.knn, args.annoy_trees, seed=42)
    import tmap
    layout = tmap.layout_from_edge_list(len(reactions), edges)
    x, y, source, target, _ = layout
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "features.npy", features)
    with (args.output_dir / "layout.pkl").open("wb") as handle:
        pickle.dump({"x": list(map(float, x)), "y": list(map(float, y)), "s": list(map(int, source)), "t": list(map(int, target)), "edges": edges}, handle)
    with (args.output_dir / "layout.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "label", "reaction"])
        writer.writerows((i, x[i], y[i], labels[i], reactions[i]) for i in range(len(reactions)))
    try:
        from faerun import Faerun
        classes = sorted(set(labels))
        groups = [classes.index(label) for label in labels]
        faerun = Faerun(clear_color="#ffffff", coords=False, view="front")
        faerun.add_scatter("RxnLM", {"x": x, "y": y, "c": [groups], "labels": reactions}, categorical=[True], has_legend=True)
        faerun.add_tree("RxnLM_tree", {"from": source, "to": target}, point_helper="RxnLM")
        faerun.plot(str(args.output_dir / "tmap"), template="reaction_smiles")
        print(f"wrote {args.output_dir / 'tmap.html'}")
    except ImportError:
        print("Faerun is unavailable; wrote features and layout files only.")
    print(json.dumps({"reactions": len(reactions), "embedding_shape": list(features.shape), "edges": len(edges)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
