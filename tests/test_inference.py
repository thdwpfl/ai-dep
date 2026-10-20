from __future__ import annotations

from pathlib import Path

import pytest

from coldstart_rec.inference import MovieRecommender, format_movie_title

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/checkpoints/fair_best.pt"
BASELINE_CHECKPOINT = ROOT / "artifacts/checkpoints/baseline_best.pt"
STRESS_BASELINE_CHECKPOINT = (
    ROOT / "artifacts/checkpoints/stress_baseline_best.pt"
)
STRESS_FAIR_CHECKPOINT = ROOT / "artifacts/checkpoints/stress_fair_best.pt"
DATA = ROOT / "data/processed"


@pytest.mark.integration
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="full trained checkpoint not ready")
def test_trained_inference_excludes_support_and_handles_zero_ratings() -> None:
    engine = MovieRecommender(CHECKPOINT, DATA)
    zero = engine.recommend("unknown", {}, top_k=10)
    assert len(zero) == 10
    assert all(1.0 <= row.predicted_score <= 5.0 for row in zero)
    rated_id = engine.catalog[0].movie_id
    conditioned = engine.recommend("female", {rated_id: 5}, top_k=10)
    assert len(conditioned) == 10
    assert rated_id not in {row.movie_id for row in conditioned}
    profile = {1: 5, 260: 5, 1196: 5, 1721: 2}
    male = engine.recommend("male", profile, top_k=20)
    female = engine.recommend("female", profile, top_k=20)
    male_signature = [(row.movie_id, row.predicted_score) for row in male]
    female_signature = [(row.movie_id, row.predicted_score) for row in female]
    assert male_signature != female_signature


def test_missing_checkpoint_is_reported() -> None:
    with pytest.raises(FileNotFoundError):
        MovieRecommender(ROOT / "does-not-exist.pt", DATA)


def test_movie_titles_move_trailing_english_articles_for_display() -> None:
    assert format_movie_title("Shawshank Redemption, The") == (
        "The Shawshank Redemption"
    )
    assert format_movie_title("Walk in the Clouds, A") == "A Walk in the Clouds"
    assert format_movie_title("Awfully Big Adventure, An") == (
        "An Awfully Big Adventure"
    )
    assert format_movie_title("Confessional, The (Le Confessionnal)") == (
        "The Confessional (Le Confessionnal)"
    )
    assert format_movie_title("Seven Samurai (Shichinin no samurai)") == (
        "Seven Samurai (Shichinin no samurai)"
    )


@pytest.mark.integration
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="full trained checkpoint not ready")
def test_catalog_display_hides_ids_and_popular_order_is_complete() -> None:
    engine = MovieRecommender(CHECKPOINT, DATA)
    assert "· id " not in engine.catalog[0].display.lower()
    assert len(engine.popular_catalog) == len(engine.catalog)
    assert {movie.item_index for movie in engine.popular_catalog} == {
        movie.item_index for movie in engine.catalog
    }
    matches = engine.search("The Shawshank Redemption")
    assert matches and matches[0].movie_id == 318


@pytest.mark.integration
@pytest.mark.skipif(
    not BASELINE_CHECKPOINT.exists(), reason="full baseline checkpoint not ready"
)
def test_baseline_checkpoint_is_dee_free_and_serves_recommendations() -> None:
    engine = MovieRecommender(BASELINE_CHECKPOINT, DATA)
    assert engine.checkpoint["config"]["train"]["dee_coefficient"] == 0.0
    rated_id = engine.catalog[0].movie_id
    recommendations = engine.recommend("unknown", {rated_id: 4}, top_k=10)
    assert len(recommendations) == 10
    assert rated_id not in {row.movie_id for row in recommendations}


@pytest.mark.integration
@pytest.mark.skipif(
    not STRESS_BASELINE_CHECKPOINT.exists() or not STRESS_FAIR_CHECKPOINT.exists(),
    reason="controlled-bias checkpoints not ready",
)
def test_controlled_bias_and_dee_checkpoints_are_distinct_and_gender_sensitive() -> None:
    """The comparison sites must serve the intended, genuinely different models."""
    stress = MovieRecommender(STRESS_BASELINE_CHECKPOINT, DATA)
    fair = MovieRecommender(STRESS_FAIR_CHECKPOINT, DATA)

    stress_config = stress.checkpoint["config"]
    fair_config = fair.checkpoint["config"]
    assert stress_config["model"]["architecture"] == "explicit_gender_genre"
    assert fair_config["model"]["architecture"] == "explicit_gender_genre"
    assert stress_config["train"]["dee_coefficient"] == 0.0
    assert stress_config["train"]["stress_coefficient"] > 0.0
    assert fair_config["train"]["dee_coefficient"] > 0.0
    assert fair_config["train"]["stress_coefficient"] == 0.0

    profile = {1: 5, 260: 5, 1196: 5, 1721: 2}
    stress_male = stress.recommend("male", profile, top_k=10)
    stress_female = stress.recommend("female", profile, top_k=10)
    fair_male = fair.recommend("male", profile, top_k=10)
    fair_female = fair.recommend("female", profile, top_k=10)

    stress_overlap = len(
        {row.movie_id for row in stress_male}
        & {row.movie_id for row in stress_female}
    )
    fair_overlap = len(
        {row.movie_id for row in fair_male}
        & {row.movie_id for row in fair_female}
    )
    assert stress_overlap < fair_overlap
    assert [(row.movie_id, row.predicted_score) for row in stress_male] != [
        (row.movie_id, row.predicted_score) for row in stress_female
    ]
