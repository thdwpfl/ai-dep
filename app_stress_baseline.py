"""Streamlit entry point for the controlled-bias stress baseline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main  # noqa: E402

STRESS_BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/checkpoints/stress_baseline_best.pt"
)
STRESS_NOTICE = (
    "⚠️ 이 모델은 DEE의 완화 효과를 검증하기 위해 성별×장르 편향을 의도적으로 "
    "주입한 통제 실험용 모델입니다. 실제 성별에 따른 영화 취향을 대표하지 않으며, "
    "일반적인 추천이나 개인에 대한 판단에 사용하면 안 됩니다."
)


if __name__ == "__main__":
    main(
        default_checkpoint=STRESS_BASELINE_CHECKPOINT,
        model_label="통제 편향 Stress Baseline · DEE 미적용",
        page_title="통제 편향 Stress Baseline",
        model_notice=STRESS_NOTICE,
    )
