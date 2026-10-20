from __future__ import annotations

import numpy as np

from coldstart_rec.groups import balanced_group_item_sample, paper_genre_membership


def test_paper_genre_membership_excludes_ambiguous_movies() -> None:
    names = ["Action", "Romance", "Comedy"]
    tags = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # Supply the rest of the paper's named genres as empty columns.
    remaining = ["Crime", "Film-Noir", "War", "Children's", "Fantasy", "Musical"]
    tags = np.pad(tags, ((0, 0), (0, len(remaining))))
    membership = paper_genre_membership(tags, names + remaining)
    np.testing.assert_array_equal(
        membership,
        np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32),
    )


def test_balanced_group_sample() -> None:
    membership = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    result = balanced_group_item_sample(
        membership, size=20, rng=np.random.default_rng(7)
    )
    counts = membership[result].sum(axis=0)
    np.testing.assert_array_equal(counts, [10, 10])

