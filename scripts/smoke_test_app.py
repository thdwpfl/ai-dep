#!/usr/bin/env python3
"""Run the complete two-screen Streamlit flow headlessly."""

from __future__ import annotations

import argparse
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINIMUM_RATINGS = 3


def button_by_label(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def run_smoke(
    app_path: Path,
    *,
    expected_checkpoint: Path | None = None,
    expected_model_label: str | None = None,
    expected_notice: str | None = None,
) -> int:
    """Exercise onboarding, recommendation, editing, and reset for one app."""
    app = AppTest.from_file(app_path, default_timeout=60).run()
    if app.exception:
        raise RuntimeError(f"App startup exceptions: {app.exception}")
    if expected_checkpoint is not None:
        actual_checkpoint = Path(app.session_state["active_checkpoint"]).resolve()
        if actual_checkpoint != expected_checkpoint.resolve():
            raise AssertionError(
                f"Expected checkpoint {expected_checkpoint}, got {actual_checkpoint}"
            )
    if expected_model_label is not None:
        actual_label = app.session_state["active_model_label"]
        if actual_label != expected_model_label:
            raise AssertionError(
                f"Expected model label {expected_model_label!r}, got {actual_label!r}"
            )
        initial_text = " ".join(markdown.value for markdown in app.markdown)
        if expected_model_label not in initial_text:
            raise AssertionError("The active model label was not visible in the UI")
    if expected_notice is not None:
        initial_text = " ".join(
            [
                *(markdown.value for markdown in app.markdown),
                *(warning.value for warning in app.warning),
            ]
        )
        if expected_notice not in initial_text:
            raise AssertionError("The model notice was not visible on the setup screen")

    recommend = button_by_label(app, "내 취향 분석하고 추천받기 →")
    if not recommend.disabled:
        raise AssertionError("Recommendation must be disabled before three ratings")

    app.selectbox[0].set_value("여성").run()
    options = list(app.multiselect[0].options[:MINIMUM_RATINGS])
    app.multiselect[0].set_value(options).run()
    if app.exception:
        raise RuntimeError(f"App movie-selection exceptions: {app.exception}")
    if len(app.radio) != MINIMUM_RATINGS:
        raise AssertionError("Every selected movie must expose a direct rating control")

    for radio in app.radio:
        radio.set_value(5)
    app.run()
    recommend = button_by_label(app, "내 취향 분석하고 추천받기 →")
    if recommend.disabled:
        raise AssertionError("Recommendation stayed disabled after three ratings")
    recommend.click().run()
    if app.exception:
        raise RuntimeError(f"App recommendation exceptions: {app.exception}")

    recommendations = app.session_state["recommendation_rows"]
    if app.session_state["ui_page"] != "results" or len(recommendations) != 10:
        raise AssertionError("The results screen did not contain ten recommendations")
    if app.session_state["recommendation_gender"] != "female":
        raise AssertionError("The selected website gender did not reach inference")
    rated_ids = set(app.session_state["saved_ratings"])
    if rated_ids.intersection(row.movie_id for row in recommendations):
        raise AssertionError("A rated movie leaked into the recommendations")
    visible_text = " ".join(
        [*(markdown.value for markdown in app.markdown),
         *(caption.value for caption in app.caption)]
    )
    if "당신을 위한 추천 10편" not in visible_text:
        raise AssertionError("The second-screen recommendation heading was not rendered")
    if "선택한 성별 ‘여성’ 정보와" not in visible_text:
        raise AssertionError("The results screen did not confirm the applied gender")
    if expected_notice is not None:
        results_text = " ".join(
            [
                visible_text,
                *(warning.value for warning in app.warning),
            ]
        )
        if expected_notice not in results_text:
            raise AssertionError("The model notice was not visible on the results screen")
    if len(app.metric) != 10 or "genre-chip" not in visible_text:
        raise AssertionError("Recommendation score cards or genres were not rendered")
    if "· id " in visible_text.lower():
        raise AssertionError("A MovieLens ID leaked into the website")

    button_by_label(app, "← 별점 수정하기").click().run()
    if app.exception or app.session_state["ui_page"] != "setup":
        raise RuntimeError("The edit-ratings action did not return to setup")
    if len(app.multiselect[0].value) != MINIMUM_RATINGS:
        raise AssertionError("The edit-ratings action did not preserve movie choices")
    if app.selectbox[0].value != "여성":
        raise AssertionError("The edit-ratings action did not preserve gender")
    if len(app.radio) != MINIMUM_RATINGS or any(radio.value != 5 for radio in app.radio):
        raise AssertionError("The edit-ratings action did not preserve ratings")

    button_by_label(app, "처음부터").click().run()
    if app.exception or app.multiselect[0].value:
        raise RuntimeError("The session reset did not clear the user's movie choices")
    if not button_by_label(app, "내 취향 분석하고 추천받기 →").disabled:
        raise AssertionError("Recommendation must be disabled again after reset")

    print(
        "Streamlit two-screen smoke test passed with 3 ratings and "
        f"10 recommendations ({app_path.name})."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=PROJECT_ROOT / "app.py")
    parser.add_argument("--expected-checkpoint", type=Path)
    parser.add_argument("--expected-model-label")
    parser.add_argument("--expected-notice")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    app_path = arguments.app
    if not app_path.is_absolute():
        app_path = PROJECT_ROOT / app_path
    raise SystemExit(
        run_smoke(
            app_path,
            expected_checkpoint=arguments.expected_checkpoint,
            expected_model_label=arguments.expected_model_label,
            expected_notice=arguments.expected_notice,
        )
    )
