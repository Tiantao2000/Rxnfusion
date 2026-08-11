from __future__ import annotations

import math
from typing import Any


def regression(y_true: list[float], y_pred: list[float]) -> dict[str, Any]:
    n = len(y_true)
    errors = [p - y for y, p in zip(y_true, y_pred)]
    abs_errors = sorted(abs(e) for e in errors)
    mse = sum(e * e for e in errors) / n if n else float("nan")
    mean_true = sum(y_true) / n if n else float("nan")
    mean_pred = sum(y_pred) / n if n else float("nan")
    ss_tot = sum((y - mean_true) ** 2 for y in y_true)
    covariance = sum((y - mean_true) * (p - mean_pred) for y, p in zip(y_true, y_pred))
    pred_var = sum((p - mean_pred) ** 2 for p in y_pred)
    pearson_den = math.sqrt(ss_tot * pred_var)
    return {
        "task_type": "regression",
        "num_samples": n,
        "mae": sum(abs_errors) / n if n else float("nan"),
        "mse": mse,
        "rmse": math.sqrt(mse) if math.isfinite(mse) else float("nan"),
        "r2": 1.0 - sum(e * e for e in errors) / ss_tot if ss_tot else None,
        "pearson_correlation": covariance / pearson_den if pearson_den else None,
        "max_absolute_error": max(abs_errors) if abs_errors else float("nan"),
        "median_absolute_error": abs_errors[(n - 1) // 2] if abs_errors else float("nan"),
        "mean_error": sum(errors) / n if n else float("nan"),
        "true_mean": mean_true,
        "prediction_mean": mean_pred,
        "true_std": math.sqrt(ss_tot / n) if n else float("nan"),
        "prediction_std": math.sqrt(pred_var / n) if n else float("nan"),
    }


def classification(y_true: list[int], y_pred: list[int], probabilities: list[list[float]]) -> dict[str, Any]:
    n_classes = max([*y_true, *y_pred], default=-1) + 1
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(y_true, y_pred):
        if true >= 0 and pred >= 0:
            matrix[true][pred] += 1
    recalls = []
    f1s = []
    for cls in range(n_classes):
        tp = matrix[cls][cls]
        support = sum(matrix[cls])
        fp = sum(matrix[row][cls] for row in range(n_classes)) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "task_type": "classification",
        "num_samples": len(y_true),
        "num_classes": n_classes,
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else None,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "confusion_matrix": matrix,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
