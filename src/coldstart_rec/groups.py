"""MovieLens group definitions used by DEE and exposure diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PAPER_MALE_PREFERRED = ("Action", "Crime", "Film-Noir", "War")
PAPER_FEMALE_PREFERRED = ("Children's", "Fantasy", "Musical", "Romance")
PAPER_GROUP_NAMES = ("male-preferred genres", "female-preferred genres")


def paper_genre_membership(
    movie_genres: np.ndarray, genre_names: Sequence[str]
) -> np.ndarray:
    """Build the two categorical item groups used in the DEE paper.

    Movies tagged by exactly one of the two preferred-genre sets enter that
    group. Movies tagged by both or neither are excluded from DEE (an all-zero
    row) because the source paper does not specify how to assign ambiguity.
    They remain available to the recommender and all accuracy metrics.
    """
    names = list(genre_names)
    male_columns = [names.index(name) for name in PAPER_MALE_PREFERRED]
    female_columns = [names.index(name) for name in PAPER_FEMALE_PREFERRED]
    has_male = movie_genres[:, male_columns].any(axis=1)
    has_female = movie_genres[:, female_columns].any(axis=1)
    membership = np.zeros((movie_genres.shape[0], 2), dtype=np.float32)
    membership[has_male & ~has_female, 0] = 1.0
    membership[has_female & ~has_male, 1] = 1.0
    return membership


def balanced_group_item_sample(
    membership: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample roughly equal candidate counts from each non-overlapping group."""
    if membership.ndim != 2:
        raise ValueError("membership must be [movies, groups]")
    if size < membership.shape[1]:
        raise ValueError("sample size must cover every item group")
    per_group = size // membership.shape[1]
    pieces: list[np.ndarray] = []
    for group in range(membership.shape[1]):
        population = np.flatnonzero(membership[:, group] > 0)
        if population.size == 0:
            raise ValueError(f"item group {group} is empty")
        take = per_group + int(group < size % membership.shape[1])
        pieces.append(rng.choice(population, size=take, replace=take > population.size))
    result = np.concatenate(pieces)
    rng.shuffle(result)
    return result.astype(np.int64, copy=False)

