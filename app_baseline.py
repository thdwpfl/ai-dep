"""Streamlit entry point for the trained recommendation baseline (no DEE)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main  # noqa: E402

BASELINE_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/baseline_best.pt"


if __name__ == "__main__":
    main(
        default_checkpoint=BASELINE_CHECKPOINT,
        model_label="Baseline 추천 모델 · DEE 미적용",
        page_title="Baseline 영화 추천",
    )
