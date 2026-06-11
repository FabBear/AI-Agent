# ── XGBoost 학습 라벨링 임계값 (MLOps data_labeling.ipynb 섹션 7) ──
Q_THR: float = 30.0
W_THR: float = 1.0
WIP_THR: float = 3.0
AVAIL_THR: float = 0.5
U_HI: float = 0.8
U_LO: float = 0.5

# ── 모델 파일 경로 ────────────────────────────────────────────────────
# Simulation 레포에서 학습 후 내보낸 모델을 AI-Agent 에서 직접 참조
MODEL_DIR: str = "models"
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

# ── 비교분석 Agent (compare_agent) ────────────────────────────────────
# 5개 KPI 방향 — 도메인 합의 사항. 병목 TG 관점:
#   lower_better  : 값이 낮아질수록 개선 (mean_delta < 0 = 개선)
#   higher_better : 값이 높아질수록 개선 (mean_delta > 0 = 개선)
# utilization_avg=lower_better 근거: cascade_analyzer/impact_calculator.py가
#   utilization을 "stress 신호"로 해석 (utilization↑ = 병목 신호) — 시스템 일관성.
KPI_DIRECTIONS_COMPARE: dict[str, str] = {
    "q_time_min": "lower_better",
    "wip": "lower_better",
    "wait_ratio": "lower_better",
    "utilization_avg": "lower_better",
    "available_tool_ratio": "higher_better",
}

# 5개 KPI 가중치 (composite_score 합산용, 합=1.0). 도메인 합의 필요.
W_KPI_Q_TIME: float = 0.40
W_KPI_WIP: float = 0.25
W_KPI_WAIT_RATIO: float = 0.15
W_KPI_UTIL: float = 0.10
W_KPI_AVAIL: float = 0.10
# 불확실성(CI 폭) 페널티 가중치
W_KPI_UNCERTAINTY: float = 0.10

# 의사결정 임계값 — 데이터 분포로 튜닝 필요
COMPARE_EQUIVALENCE_EPS: float = 0.05   # 1위와 점수 차이가 이 이하면 통계적 동등
COMPARE_NO_EFFECT_MAX: float = 0.01     # 최고 점수가 이 이하면 "효과 미검증"

# KPI별 "최소 의미있는 변화량" — 후보 간 Δ가 이 이하면 의미 없음으로 본다.
# 운영자 합의 필요 (코드 베이스에 별도 정답 없음).
MIN_DELTA_Q_TIME: float = 5.0
MIN_DELTA_WIP: float = 1.0
MIN_DELTA_WAIT_RATIO: float = 0.02
MIN_DELTA_UTIL: float = 0.02
MIN_DELTA_AVAIL: float = 0.02

# action_kind 운영 메타데이터.
#   effort        : 1(가벼움) ~ 4(무거움). 동등 케이스 tie-break + LLM 근거에 노출.
#   scope         : 영향 범위 — single_lot < tool_local < toolgroup < fab_wide
#   reversibility : 되돌리기 난이도 — high(쉬움) / medium / low(어려움)
#   description_ko: 한국어 라벨
# 현재 solution_generator가 만드는 건 DISPATCH_RULE_OVERRIDE만 (action_mapper.py).
# LOT_HOLD/SET_SUPER_HOT은 FabEnv 지원 — 향후 확장 대비.
ACTION_KIND_METADATA: dict[str, dict] = {
    "LOT_HOLD": {
        "effort": 1,
        "scope": "single_lot",
        "reversibility": "high",
        "description_ko": "단일 lot 일시 보류",
    },
    "SET_SUPER_HOT": {
        "effort": 2,
        "scope": "single_lot",
        "reversibility": "high",
        "description_ko": "단일 lot 우선처리 지정",
    },
    "DISPATCH_RULE_OVERRIDE": {
        "effort": 4,
        "scope": "fab_wide",
        "reversibility": "low",
        "description_ko": "FAB 디스패치 룰 변경",
    },
    "UNKNOWN": {
        "effort": 99,
        "scope": "unknown",
        "reversibility": "unknown",
        "description_ko": "미정",
    },
}
