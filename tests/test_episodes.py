from __future__ import annotations

import numpy as np

from coldstart_rec.data import MovieLens1MData
from coldstart_rec.episodes import make_episode_batch


def toy_data() -> MovieLens1MData:
    arrays = {
        "movie_ids": np.arange(1, 7, dtype=np.int32),
        "movie_titles": np.asarray([f"M{i}" for i in range(6)]),
        "movie_years": np.arange(2000, 2006, dtype=np.int16),
        "movie_years_normalized": np.linspace(-1, 1, 6, dtype=np.float32),
        "movie_genres": np.eye(18, dtype=np.uint8)[:6],
        "dee_item_genre_membership": np.column_stack(
            (np.arange(6, dtype=np.int32), np.arange(6, dtype=np.int32))
        ),
        "user_ids": np.asarray([10, 11], dtype=np.int32),
        "user_genders": np.asarray([0, 1], dtype=np.uint8),
        "user_splits": np.asarray([0, 0], dtype=np.uint8),
        "user_ptr": np.asarray([0, 4, 10], dtype=np.int64),
        "rating_item_indices": np.asarray(
            [0, 1, 2, 3, 0, 1, 2, 3, 4, 5], dtype=np.int32
        ),
        "rating_values": np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1, 5], dtype=np.uint8),
        "rating_timestamps": np.arange(10, dtype=np.int64),
    }
    return MovieLens1MData(arrays, {"schema_version": 1})


def test_batch_has_chronological_support_and_masked_queries() -> None:
    batch = make_episode_batch(
        toy_data(),
        [0, 1],
        [1, 3],
        query_size=2,
        rng=np.random.default_rng(4),
    )
    assert batch.support_items.shape == (2, 3)
    assert batch.support_mask.tolist() == [[True, False, False], [True, True, True]]
    assert batch.support_items[0, 0].item() == 0
    assert batch.query_items.shape == (2, 2)
    assert batch.query_mask.all()

