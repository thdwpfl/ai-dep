"""Reusable, stateless inference API for completely unseen users."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .checkpoint import load_model_checkpoint
from .data import GENRES, MovieLens1MData, load_processed_data, normalize_gender

DISPLAY_TRAILING_ARTICLES = ("The", "A", "An")


def format_movie_title(title: str) -> str:
    """Move MovieLens' trailing English article to a natural display position.

    The canonical catalog title remains untouched for reproducibility. Only the
    primary display title is changed, including when an alternate title follows
    in parentheses: ``Confessional, The (Le Confessionnal)`` becomes
    ``The Confessional (Le Confessionnal)``.
    """
    primary, separator, alternate = str(title).partition(" (")
    for article in DISPLAY_TRAILING_ARTICLES:
        suffix = f", {article}"
        if primary.endswith(suffix) and len(primary) > len(suffix):
            primary = f"{article} {primary[:-len(suffix)]}"
            break
    return primary + separator + alternate


@dataclass(frozen=True)
class MovieInfo:
    """Catalog metadata exposed by search."""

    movie_id: int
    item_index: int
    title: str
    year: int | None
    genres: tuple[str, ...]

    @property
    def display_title(self) -> str:
        """Return a presentation title while retaining ``title`` as source data."""
        return format_movie_title(self.title)

    @property
    def display(self) -> str:
        year = f" ({self.year})" if self.year is not None else ""
        return f"{self.display_title}{year}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation(MovieInfo):
    """A recommended catalog item with its model score."""

    predicted_score: float


class MovieRecommender:
    """Load-once API that never mutates its saved global checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_path: str | Path = "data/processed",
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.data_path = Path(data_path).resolve()
        self.data: MovieLens1MData = load_processed_data(self.data_path)
        self.device = torch.device(device)
        self.model, self.checkpoint = load_model_checkpoint(
            self.checkpoint_path, self.device
        )
        expected = self.checkpoint.get("dataset_fingerprint")
        actual = self.data.manifest["artifact"]["sha256"]
        if expected != actual:
            raise ValueError("Checkpoint and catalog fingerprints differ")
        self._catalog = [self._movie_info(index) for index in range(self.data.num_movies)]
        rating_counts = np.bincount(
            self.data.rating_item_indices, minlength=self.data.num_movies
        )
        self._popular_item_indices = np.argsort(
            -rating_counts, kind="stable"
        ).astype(np.int64)

    def _movie_info(self, item_index: int) -> MovieInfo:
        genre_indices = np.flatnonzero(self.data.movie_genres[item_index])
        year_value = int(self.data.movie_years[item_index])
        return MovieInfo(
            movie_id=int(self.data.movie_ids[item_index]),
            item_index=int(item_index),
            title=str(self.data.movie_titles[item_index]),
            year=year_value if year_value >= 0 else None,
            genres=tuple(GENRES[index] for index in genre_indices),
        )

    @property
    def catalog(self) -> tuple[MovieInfo, ...]:
        return tuple(self._catalog)

    @property
    def popular_catalog(self) -> tuple[MovieInfo, ...]:
        """Return the catalog ordered by MovieLens rating count."""
        return tuple(self._catalog[int(index)] for index in self._popular_item_indices)

    def search(self, query: str, limit: int = 25) -> list[MovieInfo]:
        """Search canonical and naturally ordered display titles locally."""
        if limit < 1:
            return []
        needle = query.strip().casefold()
        if not needle:
            return self._catalog[:limit]
        return [
            movie
            for movie in self._catalog
            if needle in movie.title.casefold()
            or needle in movie.display_title.casefold()
        ][:limit]

    @torch.inference_mode()
    def recommend(
        self,
        gender: str | int | None,
        ratings_by_movie_id: Mapping[int, float] | None = None,
        *,
        top_k: int = 10,
    ) -> list[Recommendation]:
        """Condition on supplied ratings and score the warm movie catalog."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        ratings = dict(ratings_by_movie_id or {})
        support_indices: list[int] = []
        support_values: list[float] = []
        for movie_id, rating in ratings.items():
            if int(movie_id) not in self.data.movie_id_to_index:
                raise KeyError(f"Unknown MovieLens movie ID: {movie_id}")
            value = float(rating)
            if not np.isfinite(value) or not 1.0 <= value <= 5.0:
                raise ValueError(f"Rating for movie {movie_id} must be between 1 and 5")
            support_indices.append(self.data.movie_id_to_index[int(movie_id)])
            support_values.append(value)

        gender_index = int(gender) if isinstance(gender, int) else normalize_gender(gender)
        support_items = torch.as_tensor(
            support_indices, dtype=torch.long, device=self.device
        ).reshape(1, -1)
        support_ratings = torch.as_tensor(
            support_values, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        support_mask = torch.ones_like(support_items, dtype=torch.bool)
        genders = torch.tensor([gender_index], dtype=torch.long, device=self.device)
        candidates = torch.arange(
            self.data.num_movies, dtype=torch.long, device=self.device
        ).reshape(1, -1)
        scores = self.model(
            support_items,
            support_ratings,
            support_mask,
            genders,
            candidates,
        ).squeeze(0)
        if support_indices:
            scores[torch.as_tensor(support_indices, device=self.device)] = -torch.inf
        count = min(top_k, self.data.num_movies - len(support_indices))
        values, indices = torch.topk(scores, k=count)
        output: list[Recommendation] = []
        for score, item_index in zip(
            values.cpu().tolist(), indices.cpu().tolist(), strict=True
        ):
            info = self._catalog[int(item_index)]
            output.append(
                Recommendation(
                    **info.to_dict(),
                    predicted_score=round(float(score), 4),
                )
            )
        return output

    @staticmethod
    def genre_distribution(
        recommendations: list[Recommendation],
    ) -> dict[str, float]:
        """Return per-recommendation multi-label genre exposure."""
        if not recommendations:
            return {}
        counts = {genre: 0 for genre in GENRES}
        for recommendation in recommendations:
            for genre in recommendation.genres:
                counts[genre] += 1
        denominator = len(recommendations)
        return {
            genre: count / denominator for genre, count in counts.items() if count
        }
