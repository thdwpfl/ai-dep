"""Vectorized construction of timestamp-respecting cold-start episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .data import MovieLens1MData


@dataclass
class EpisodeBatch:
    """A padded batch of independent unseen-user tasks."""

    user_indices: Tensor
    genders: Tensor
    support_items: Tensor
    support_ratings: Tensor
    support_mask: Tensor
    query_items: Tensor
    query_ratings: Tensor
    query_mask: Tensor

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        return EpisodeBatch(
            **{
                field: getattr(self, field).to(device)
                for field in self.__dataclass_fields__
            }
        )


def make_episode_batch(
    data: MovieLens1MData,
    user_indices: Sequence[int] | np.ndarray,
    support_sizes: int | Sequence[int] | np.ndarray,
    *,
    query_size: int | None,
    rng: np.random.Generator,
    device: torch.device | str | None = None,
) -> EpisodeBatch:
    """Collate chronological prefixes and sampled or complete future queries."""
    users = np.asarray(user_indices, dtype=np.int64)
    if users.ndim != 1 or users.size == 0:
        raise ValueError("user_indices must be a non-empty one-dimensional sequence")
    if np.isscalar(support_sizes):
        supports = np.full(users.size, int(support_sizes), dtype=np.int64)
    else:
        supports = np.asarray(support_sizes, dtype=np.int64)
    if supports.shape != users.shape or np.any(supports < 0):
        raise ValueError("support_sizes must be non-negative and aligned with users")
    if query_size is not None and query_size <= 0:
        raise ValueError("query_size must be positive or None")

    histories = [data.history_by_index(int(user)) for user in users]
    actual_supports = np.minimum(
        supports, np.asarray([len(history.ratings) - 1 for history in histories])
    )
    max_support = int(actual_supports.max(initial=0))
    support_items = np.zeros((users.size, max_support), dtype=np.int64)
    support_ratings = np.zeros((users.size, max_support), dtype=np.float32)
    support_mask = np.zeros((users.size, max_support), dtype=np.bool_)

    query_item_rows: list[np.ndarray] = []
    query_rating_rows: list[np.ndarray] = []
    for row, (history, support) in enumerate(zip(histories, actual_supports, strict=True)):
        support = int(support)
        if support:
            support_items[row, :support] = history.item_indices[:support]
            support_ratings[row, :support] = history.ratings[:support]
            support_mask[row, :support] = True
        remaining_items = history.item_indices[support:]
        remaining_ratings = history.ratings[support:]
        if query_size is not None and len(remaining_items) > query_size:
            selected = np.sort(
                rng.choice(len(remaining_items), size=query_size, replace=False)
            )
            remaining_items = remaining_items[selected]
            remaining_ratings = remaining_ratings[selected]
        query_item_rows.append(remaining_items.astype(np.int64, copy=False))
        query_rating_rows.append(remaining_ratings.astype(np.float32, copy=False))

    max_query = max(len(row) for row in query_item_rows)
    query_items = np.zeros((users.size, max_query), dtype=np.int64)
    query_ratings = np.zeros((users.size, max_query), dtype=np.float32)
    query_mask = np.zeros((users.size, max_query), dtype=np.bool_)
    for row, (items, ratings) in enumerate(
        zip(query_item_rows, query_rating_rows, strict=True)
    ):
        length = len(items)
        query_items[row, :length] = items
        query_ratings[row, :length] = ratings
        query_mask[row, :length] = True

    target_device = device if device is not None else "cpu"
    return EpisodeBatch(
        user_indices=torch.as_tensor(users, dtype=torch.long, device=target_device),
        genders=torch.as_tensor(
            data.user_genders[users], dtype=torch.long, device=target_device
        ),
        support_items=torch.as_tensor(
            support_items, dtype=torch.long, device=target_device
        ),
        support_ratings=torch.as_tensor(
            support_ratings, dtype=torch.float32, device=target_device
        ),
        support_mask=torch.as_tensor(
            support_mask, dtype=torch.bool, device=target_device
        ),
        query_items=torch.as_tensor(
            query_items, dtype=torch.long, device=target_device
        ),
        query_ratings=torch.as_tensor(
            query_ratings, dtype=torch.float32, device=target_device
        ),
        query_mask=torch.as_tensor(query_mask, dtype=torch.bool, device=target_device),
    )


def user_batches(
    user_indices: Sequence[int] | np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    *,
    shuffle: bool,
) -> list[np.ndarray]:
    """Return stable batches with an optional deterministic RNG shuffle."""
    users = np.asarray(user_indices, dtype=np.int64).copy()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shuffle:
        rng.shuffle(users)
    return [users[start : start + batch_size] for start in range(0, len(users), batch_size)]

