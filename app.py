"""Two-step, session-only Streamlit interface for movie recommendations."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from coldstart_rec.inference import MovieInfo, MovieRecommender  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/fair_best.pt"
DEFAULT_DATA = PROJECT_ROOT / "data/processed"
MIN_RATINGS = 3
MAX_RATINGS = 12
GENRE_LABELS = {
    "Action": "액션",
    "Adventure": "모험",
    "Animation": "애니메이션",
    "Children's": "어린이",
    "Comedy": "코미디",
    "Crime": "범죄",
    "Documentary": "다큐멘터리",
    "Drama": "드라마",
    "Fantasy": "판타지",
    "Film-Noir": "필름 누아르",
    "Horror": "공포",
    "Musical": "뮤지컬",
    "Mystery": "미스터리",
    "Romance": "로맨스",
    "Sci-Fi": "SF",
    "Thriller": "스릴러",
    "War": "전쟁",
    "Western": "서부극",
}


@st.cache_resource(show_spinner="추천 모델을 불러오는 중이에요…")
def load_engine(checkpoint: str, data_path: str) -> MovieRecommender:
    """Cache immutable model/catalog state; user ratings never enter this cache."""
    return MovieRecommender(checkpoint, data_path, device="cpu")


def reset_session() -> None:
    """Erase every transient value associated with this browser session."""
    for key in list(st.session_state):
        del st.session_state[key]


def go_to_setup() -> None:
    """Return to the rating screen while retaining the user's current choices."""
    saved_selection = st.session_state.get("saved_movie_selection", [])
    saved_ratings = st.session_state.get("saved_ratings", {})
    saved_gender = st.session_state.get("saved_gender_selection", "응답하지 않음")
    st.session_state["movie_selection"] = list(saved_selection)
    st.session_state["gender_selection"] = saved_gender
    for movie_id, rating in saved_ratings.items():
        st.session_state[f"rating_{movie_id}"] = int(rating)
    st.session_state["ui_page"] = "setup"


def inject_styles() -> None:
    """Apply lightweight styling without external assets or tracking."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .hero {
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(239, 68, 68, 0.24);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.22), rgba(30, 41, 59, 0.18));
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            font-size: clamp(1.8rem, 4vw, 2.7rem);
        }
        .hero p {
            margin: 0;
            opacity: .82;
            font-size: 1.02rem;
        }
        .steps {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin: .25rem 0 1.15rem 0;
            font-size: .9rem;
        }
        .step {
            padding: .38rem .72rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, .14);
            color: rgba(148, 163, 184, 1);
        }
        .step.active {
            color: white;
            background: rgb(220, 38, 38);
            font-weight: 700;
        }
        .step-line {
            width: 32px;
            height: 1px;
            background: rgba(148, 163, 184, .4);
        }
        .model-badge {
            display: inline-block;
            margin: 0 0 .8rem 0;
            padding: .3rem .66rem;
            border-radius: 999px;
            color: rgb(254, 202, 202);
            background: rgba(239, 68, 68, .11);
            border: 1px solid rgba(239, 68, 68, .23);
            font-size: .8rem;
            font-weight: 650;
        }
        .genre-chip {
            display: inline-block;
            margin: .12rem .28rem .12rem 0;
            padding: .2rem .52rem;
            border-radius: 999px;
            background: rgba(239, 68, 68, .12);
            border: 1px solid rgba(239, 68, 68, .26);
            font-size: .78rem;
        }
        .rank {
            color: rgb(248, 113, 113);
            font-weight: 800;
            letter-spacing: .04em;
            margin-bottom: .25rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_steps(current: int) -> None:
    first_class = "step active" if current == 1 else "step"
    second_class = "step active" if current == 2 else "step"
    st.markdown(
        f"""
        <div class="steps" aria-label="추천 진행 단계">
          <span class="{first_class}">1&nbsp; 취향 입력</span>
          <span class="step-line"></span>
          <span class="{second_class}">2&nbsp; 추천 결과</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_badge(model_label: str) -> None:
    """Make the active checkpoint family explicit without exposing internals."""
    st.markdown(
        f'<span class="model-badge">{html.escape(model_label)}</span>',
        unsafe_allow_html=True,
    )


def render_model_notice(model_notice: str | None) -> None:
    """Show an optional model-specific warning immediately below its badge."""
    if model_notice:
        st.warning(model_notice)


def movie_label(movie: MovieInfo) -> str:
    """Format a searchable label without exposing internal or MovieLens IDs."""
    genres = " · ".join(GENRE_LABELS.get(genre, genre) for genre in movie.genres[:3])
    suffix = f" — {genres}" if genres else ""
    return f"{movie.display}{suffix}"


def genre_chips(genres: tuple[str, ...]) -> str:
    """Render trusted catalog genres as escaped visual chips."""
    return " ".join(
        f'<span class="genre-chip">{html.escape(GENRE_LABELS.get(genre, genre))}</span>'
        for genre in genres
    )


def render_setup(
    engine: MovieRecommender,
    model_label: str,
    model_notice: str | None = None,
) -> None:
    """Render step one: select and explicitly rate watched movies."""
    render_steps(1)
    render_model_badge(model_label)
    render_model_notice(model_notice)
    st.markdown(
        """
        <div class="hero">
          <h1>🎬 당신의 취향을 알려주세요</h1>
          <p>본 영화 중 최소 3편을 고르고 별점을 남기면, 취향에 맞는 영화 10편을 찾아드려요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1])
    with left:
        st.markdown("#### 1. 본 영화를 찾아보세요")
        st.caption("제목을 입력해 검색할 수 있어요. 많이 평가할수록 취향을 더 잘 반영합니다.")
    with right:
        gender_label = st.selectbox(
            "성별 (선택사항)",
            ["응답하지 않음", "여성", "남성"],
            key="gender_selection",
            help="응답하지 않아도 추천을 받을 수 있어요.",
        )

    catalog_by_index = {movie.item_index: movie for movie in engine.catalog}
    popular_indices = [movie.item_index for movie in engine.popular_catalog]
    selected_indices = st.multiselect(
        f"본 영화 선택 (최소 {MIN_RATINGS}편)",
        options=popular_indices,
        format_func=lambda index: movie_label(catalog_by_index[int(index)]),
        max_selections=MAX_RATINGS,
        placeholder="예: Toy Story, Matrix, Titanic…",
        key="movie_selection",
        label_visibility="collapsed",
    )

    ratings: dict[int, float] = {}
    if selected_indices:
        st.markdown("#### 2. 별점을 바로 선택하세요")
        st.caption("1점은 별로였어요, 5점은 정말 좋았어요.")
        rating_columns = st.columns(2)
        for position, item_index in enumerate(selected_indices):
            movie = catalog_by_index[int(item_index)]
            with rating_columns[position % 2]:
                with st.container(border=True):
                    st.markdown(f"**{movie.display_title}**")
                    year_text = str(movie.year) if movie.year is not None else "연도 정보 없음"
                    st.caption(year_text)
                    st.markdown(genre_chips(movie.genres), unsafe_allow_html=True)
                    rating = st.radio(
                        f"{movie.display_title} 별점",
                        options=[1, 2, 3, 4, 5],
                        index=None,
                        format_func=lambda value: "★" * value,
                        horizontal=True,
                        key=f"rating_{movie.movie_id}",
                        label_visibility="collapsed",
                    )
                    if rating is not None:
                        ratings[movie.movie_id] = float(rating)
    else:
        st.info("위 검색창에서 본 영화를 선택하면 바로 별점을 줄 수 있어요.")

    rated_count = len(ratings)
    st.progress(
        min(rated_count / MIN_RATINGS, 1.0),
        text=f"별점 완료 {rated_count}/{MIN_RATINGS}편",
    )
    all_selected_rated = bool(selected_indices) and rated_count == len(selected_indices)
    ready = rated_count >= MIN_RATINGS and all_selected_rated
    if rated_count < MIN_RATINGS:
        st.caption(f"추천을 시작하려면 {MIN_RATINGS - rated_count}편의 별점이 더 필요해요.")
    elif not all_selected_rated:
        st.caption("선택한 모든 영화에 별점을 남겨주세요.")
    else:
        st.success("취향 입력 완료! 이제 맞춤 추천을 받을 수 있어요.")

    action_left, action_right = st.columns([4, 1])
    with action_left:
        recommend_clicked = st.button(
            "내 취향 분석하고 추천받기 →",
            type="primary",
            width="stretch",
            disabled=not ready,
        )
    with action_right:
        st.button("처음부터", on_click=reset_session, width="stretch")

    if recommend_clicked:
        gender = {
            "여성": "female",
            "남성": "male",
            "응답하지 않음": "unknown",
        }[gender_label]
        with st.spinner("입력한 별점을 바탕으로 취향을 분석하고 있어요…"):
            try:
                recommendations = engine.recommend(gender, ratings, top_k=10)
            except Exception:
                st.error("추천을 만드는 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요.")
                return
            st.session_state["saved_movie_selection"] = [
                int(index) for index in selected_indices
            ]
            st.session_state["saved_ratings"] = dict(ratings)
            st.session_state["saved_gender_selection"] = gender_label
            st.session_state["recommendation_gender"] = gender
            st.session_state["recommendation_rows"] = recommendations
            st.session_state["recommendation_rating_count"] = rated_count
            st.session_state["ui_page"] = "results"
        st.rerun()


def render_results(
    engine: MovieRecommender,
    model_label: str,
    model_notice: str | None = None,
) -> None:
    """Render step two as scannable recommendation cards."""
    recommendations = st.session_state.get("recommendation_rows")
    if not recommendations:
        st.session_state["ui_page"] = "setup"
        st.rerun()

    render_steps(2)
    render_model_badge(model_label)
    render_model_notice(model_notice)
    st.markdown(
        """
        <div class="hero">
          <h1>🍿 당신을 위한 추천 10편</h1>
          <p>평가한 영화는 제외하고, 입력한 취향과 잘 맞는 순서로 골랐어요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rating_count = int(st.session_state.get("recommendation_rating_count", 0))
    gender_label = st.session_state.get("saved_gender_selection", "응답하지 않음")
    if gender_label == "응답하지 않음":
        profile_summary = "성별 정보 없이"
    else:
        profile_summary = f"선택한 성별 ‘{gender_label}’ 정보와"
    st.caption(
        f"{profile_summary} {rating_count}편의 별점을 모델에 반영한 세션 전용 추천이에요."
    )

    action_left, action_right, spacer = st.columns([1.3, 1, 3])
    with action_left:
        st.button("← 별점 수정하기", on_click=go_to_setup, width="stretch")
    with action_right:
        st.button("처음부터", on_click=reset_session, width="stretch")
    with spacer:
        st.empty()

    cards = st.columns(2)
    for position, recommendation in enumerate(recommendations, start=1):
        with cards[(position - 1) % 2]:
            with st.container(border=True):
                st.markdown(f'<div class="rank">TOP {position}</div>', unsafe_allow_html=True)
                st.markdown(f"### {recommendation.display_title}")
                year_text = (
                    str(recommendation.year)
                    if recommendation.year is not None
                    else "연도 정보 없음"
                )
                st.caption(year_text)
                st.markdown(
                    genre_chips(recommendation.genres), unsafe_allow_html=True
                )
                st.metric(
                    "예상 취향 점수",
                    f"{recommendation.predicted_score:.2f} / 5",
                )

    with st.expander("추천 장르 구성 보기"):
        distribution = engine.genre_distribution(recommendations)
        diagnostic = pd.DataFrame(
            {
                "장르": [GENRE_LABELS.get(genre, genre) for genre in distribution],
                "추천 영화 중 비율": list(distribution.values()),
            }
        ).set_index("장르")
        st.bar_chart(diagnostic)
        st.caption("한 영화가 여러 장르를 가질 수 있어 전체 비율의 합은 100%를 넘을 수 있어요.")


def main(
    *,
    default_checkpoint: Path = DEFAULT_CHECKPOINT,
    model_label: str = "DEE 공정성 적용 모델",
    page_title: str = "내 취향 영화 추천",
    model_notice: str | None = None,
) -> None:
    """Run the shared two-step UI for a selected trained checkpoint."""
    st.set_page_config(page_title=page_title, page_icon="🎬", layout="wide")
    inject_styles()
    checkpoint = os.environ.get("COLDSTART_CHECKPOINT", str(default_checkpoint))
    data_path = os.environ.get("COLDSTART_DATA", str(DEFAULT_DATA))
    active_model_label = os.environ.get("COLDSTART_MODEL_LABEL", model_label)
    if not Path(checkpoint).is_file():
        st.error(f"학습된 모델을 찾을 수 없어요: {checkpoint}")
        st.stop()
    try:
        engine = load_engine(checkpoint, data_path)
    except Exception as exc:
        st.error(f"추천 모델을 불러오지 못했어요: {exc}")
        st.stop()

    st.session_state["active_checkpoint"] = str(Path(checkpoint).resolve())
    st.session_state["active_model_label"] = active_model_label
    st.session_state.setdefault("ui_page", "setup")
    if st.session_state["ui_page"] == "results":
        render_results(engine, active_model_label, model_notice)
    else:
        render_setup(engine, active_model_label, model_notice)

    st.divider()
    st.caption("🔒 입력한 성별과 별점은 이 브라우저 세션 안에서만 사용되며 저장되지 않아요.")


if __name__ == "__main__":
    main()
