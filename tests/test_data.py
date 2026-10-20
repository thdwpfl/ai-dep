"""Focused tests for MovieLens parsing, splitting, serialization, and episodes."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coldstart_rec.data import (  # noqa: E402
    GENRES,
    GENDER_TO_INDEX,
    ColdStartEpisodeDataset,
    load_processed_data,
    normalize_gender,
    preprocess_ml1m,
    safe_extract_zip,
)


def _write_toy_ml1m(root: Path) -> Path:
    raw = root / "ml-1m"
    raw.mkdir(parents=True)
    users = []
    ratings = []
    timestamp_base = 1_000_000
    for user_id in range(1, 13):
        gender = "M" if user_id % 2 else "F"
        users.append(f"{user_id}::{gender}::25::4::0000{user_id}")
        # Intentionally written out of timestamp order.
        ratings.extend(
            [
                f"{user_id}::1::4::{timestamp_base + user_id * 10 + 3}",
                f"{user_id}::2::2::{timestamp_base + user_id * 10 + 1}",
                f"{user_id}::3::5::{timestamp_base + user_id * 10 + 2}",
            ]
        )
    movies = [
        "1::Amélie (2001)::Comedy|Romance",
        "2::Toy Story (1995)::Animation|Children's|Comedy",
        "3::No Year Film::Drama",
    ]
    (raw / "users.dat").write_text("\n".join(users) + "\n", encoding="latin-1")
    (raw / "movies.dat").write_text("\n".join(movies) + "\n", encoding="latin-1")
    (raw / "ratings.dat").write_text("\n".join(ratings) + "\n", encoding="latin-1")
    return raw


def test_preprocess_load_split_and_episode(tmp_path: Path) -> None:
    raw = _write_toy_ml1m(tmp_path / "raw")
    artifact, manifest_path = preprocess_ml1m(
        raw, tmp_path / "processed", seed=17, split_ratios=(0.5, 0.25, 0.25)
    )
    assert artifact.is_file()
    data = load_processed_data(artifact.parent)

    assert data.movie_genres.shape == (3, len(GENRES))
    assert data.movie_genres.dtype == np.uint8
    assert data.movie_titles.tolist() == ["Amélie", "Toy Story", "No Year Film"]
    assert data.movie_years.tolist() == [2001, 1995, -1]
    assert data.movie_years_normalized[2] == 0.0
    assert data.dee_item_genre_membership.shape[1] == 2
    assert set(data.dee_item_genre_membership[:, 1]) <= set(range(18))

    split_sets = [set(data.split_user_indices(name).tolist()) for name in ("train", "val", "test")]
    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
    assert set.union(*split_sets) == set(range(12))
    for name in ("train", "val", "test"):
        genders = set(data.user_genders[data.split_user_indices(name)].tolist())
        assert genders == {GENDER_TO_INDEX["M"], GENDER_TO_INDEX["F"]}

    history = data.history_by_id(1)
    assert history.timestamps.tolist() == sorted(history.timestamps.tolist())
    assert history.movie_ids.tolist() == [2, 3, 1]
    episode = data.episode_by_index(history.user_index, support_size=1)
    assert episode.support_movie_ids.tolist() == [2]
    assert episode.query_movie_ids.tolist() == [3, 1]
    assert episode.support_genres.shape == (1, 18)
    dataset = ColdStartEpisodeDataset(data, "test", support_size=1)
    assert len(dataset) > 0
    assert len(dataset[0].query_ratings) == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 17
    assert manifest["no_user_overlap"] is True
    assert manifest["counts"]["ratings"] == 36
    assert manifest["gender_to_index"]["U"] == 2


def test_split_is_deterministic(tmp_path: Path) -> None:
    raw = _write_toy_ml1m(tmp_path / "raw")
    preprocess_ml1m(raw, tmp_path / "first", seed=123)
    preprocess_ml1m(raw, tmp_path / "second", seed=123)
    first = load_processed_data(tmp_path / "first")
    second = load_processed_data(tmp_path / "second")
    np.testing.assert_array_equal(first.user_splits, second.user_splits)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("male", GENDER_TO_INDEX["M"]),
        ("F", GENDER_TO_INDEX["F"]),
        (None, GENDER_TO_INDEX["U"]),
        ("prefer-not-to-answer", GENDER_TO_INDEX["U"]),
    ],
)
def test_unknown_gender_mapping(value: str | None, expected: int) -> None:
    assert normalize_gender(value) == expected


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()
