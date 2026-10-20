"""Streamlit entry point for the DEE-mitigated controlled-bias model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main  # noqa: E402

STRESS_FAIR_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints/stress_fair_best.pt"
STRESS_NOTICE = (
    "⚠️ 이 모델은 DEE의 완화 효과를 검증하기 위해 성별×장르 편향을 의도적으로 "
    "주입한 통제 실험에서 학습했습니다. DEE를 적용했더라도 실제 성별에 따른 영화 "
    "취향을 대표하지 않으며, 일반적인 추천이나 개인에 대한 판단에 사용하면 안 됩니다."
)


if __name__ == "__main__":
    main(
        default_checkpoint=STRESS_FAIR_CHECKPOINT,
        model_label="통제 편향 완화 모델 · DEE 적용",
        page_title="DEE 통제 편향 완화 모델",
        model_notice=STRESS_NOTICE,
    )
