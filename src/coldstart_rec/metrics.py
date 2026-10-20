"""Accuracy, ranking, and recommendation-exposure metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


def regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    """Return RMSE and MAE for aligned tensors."""
    if predictions.numel() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "count": 0}
    error = predictions.float().reshape(-1) - targets.float().reshape(-1)
    return {
        "rmse": float(torch.sqrt(torch.mean(error.square())).item()),
        "mae": float(torch.mean(error.abs()).item()),
        "count": int(error.numel()),
    }


def ranking_metrics_for_user(
    scores: np.ndarray,
    ratings: np.ndarray,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict[str, float]:
    """Rank a user's timestamp-held-out rated movies.

    The candidate set is the user's query set, not unlabeled catalog items.
    This avoids treating unobserved MovieLens entries as negative feedback.
    """
    if scores.ndim != 1 or ratings.ndim != 1 or scores.shape != ratings.shape:
        raise ValueError("scores and ratings must be aligned one-dimensional arrays")
    relevant = ratings >= relevance_threshold
    relevant_count = int(relevant.sum())
    if scores.size == 0:
        return {
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"hitrate@{k}": 0.0,
        }
    cutoff = min(k, scores.size)
    order = np.argsort(-scores, kind="stable")[:cutoff]
    hits = relevant[order].astype(np.float64)
    recall = float(hits.sum() / relevant_count) if relevant_count else 0.0
    discounts = 1.0 / np.log2(np.arange(cutoff, dtype=np.float64) + 2.0)
    dcg = float((hits * discounts).sum())
    ideal_count = min(relevant_count, cutoff)
    idcg = float(discounts[:ideal_count].sum())
    ndcg = dcg / idcg if idcg else 0.0
    return {
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg,
        f"hitrate@{k}": float(hits.any()),
    }


def mean_metric_dict(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average finite values for each key."""
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    output: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        finite = [value for value in values if math.isfinite(value)]
        output[key] = float(np.mean(finite)) if finite else float("nan")
    return output


def exposure_by_gender(
    recommended_items: Sequence[np.ndarray],
    genders: Sequence[int],
    movie_genres: np.ndarray,
    genre_names: Iterable[str],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Return per-position multi-label genre exposure for male/female users.

    A movie contributes once to every genre tag it carries, so each group's
    genre rates need not sum to one.
    """
    names = list(genre_names)
    if movie_genres.shape[1] != len(names):
        raise ValueError("genre name count differs from movie feature width")
    totals: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(movie_genres.shape[1], dtype=np.float64)
    )
    positions: dict[int, int] = defaultdict(int)
    for items, gender in zip(recommended_items, genders, strict=True):
        selected = np.asarray(items, dtype=np.int64)[:k]
        if selected.size:
            totals[int(gender)] += movie_genres[selected].sum(axis=0)
            positions[int(gender)] += int(selected.size)

    labels = {0: "male", 1: "female", 2: "unknown"}
    result: dict[str, dict[str, float]] = {}
    for group, values in sorted(totals.items()):
        denominator = max(positions[group], 1)
        result[labels.get(group, str(group))] = {
            name: float(value / denominator)
            for name, value in zip(names, values, strict=True)
        }
    return result
