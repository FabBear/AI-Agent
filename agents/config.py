# ── XGBoost 학습 라벨링 임계값 (MLOps data_labeling.ipynb 섹션 7) ──
Q_THR: float = 30.0
W_THR: float = 1.0
WIP_THR: float = 3.0
AVAIL_THR: float = 0.5
U_HI: float = 0.8
U_LO: float = 0.5

# ── 모델 파일 경로 ────────────────────────────────────────────────────
# Simulation 레포에서 학습 후 내보낸 모델을 AI-Agent 에서 직접 참조
MODEL_DIR: str = "../Simulation/simulation/ML/out"
MODEL_FILENAME: str = "bottleneck_xgb.ubj"
FEATURE_COLS_FILENAME: str = "bottleneck_feature_cols.json"

# ── CASCADE 분석 ──────────────────────────────────────────────────────
# DAG 탐색 최대 홉 수
MAX_CASCADE_HOPS: int = 5
# CT 전파 감쇠율 (홉당 이만큼 흡수)
CT_DECAY: float = 0.7
# 정규화 기준값
CT_BASELINE_MIN: float = 60.0  # 60분 이상이면 impact 1.0
LOT_BASELINE: float = 30.0  # 30 lots 이상이면 impact 1.0

# ── 종합 점수 가중치 ──────────────────────────────────────────────────
# impact_score = W_CAPACITY*capacity + W_CT*ct + W_LOT*lot_risk
W_CAPACITY: float = 0.40
W_CT: float = 0.35
W_LOT: float = 0.25

# composite_score = PROB_WEIGHT*prob + IMPACT_WEIGHT*impact_score
PROB_WEIGHT: float = 0.50
IMPACT_WEIGHT: float = 0.50

# ── 심각도 임계값 (composite_score 기준) ─────────────────────────────
CRITICAL_SCORE: float = 0.75
HIGH_SCORE: float = 0.55
MEDIUM_SCORE: float = 0.35

# ── LLM 설정 (전 Agent 공통) ──────────────────────────────────────────
# 환경변수로 override 가능: LLM_MODEL=gpt-4o uv run python run_detection.py
import os as _os
LLM_MODEL: str = _os.getenv("LLM_MODEL", "gpt-5.4-mini")
LLM_TEMPERATURE: float = float(_os.getenv("LLM_TEMPERATURE", "0.2"))
