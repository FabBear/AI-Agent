"""report_agent JSON v2 컨트랙트 — Pydantic 모델.

설계 원칙
=========
1. 한 문자열에 여러 정보 안 뭉친다. 값과 단위, before·after·delta 모두 별 필드.
   - X "Release Interval 60.0→62.2분"
   - O { current: 60.0, target: 62.2, delta: 2.2, unit: "min" }

2. 색·이모지·뱃지는 enum 토큰만. FE 디자인 시스템이 표현을 결정한다.
   - X "🚨 CRITICAL"
   - O severity="Critical", severity_token="danger", severity_priority=0

3. "데이터 없음" 표현은 `available: bool` 플래그. FE가 카드 자체를
   안 그릴 수 있게.

4. 업스트림에 없는 데이터는 모델에 키조차 없다. (tool_status,
   affected_lots_detail 같은 mock 잔재 키 없음.)

5. 모든 최상위 블록은 항상 출력된다. FE TypeScript 타입과 예측 가능한
   컨트랙트를 위해 — 값이 없으면 None/[] 로 나가더라도 키는 존재한다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ── 토큰 enum ─────────────────────────────────────────────────────────────────

class SeverityToken(str, Enum):
    DANGER  = "danger"     # Critical
    WARN    = "warn"       # High
    CAUTION = "caution"    # Medium
    OK      = "ok"         # Low


class ConfidenceToken(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class DecisionStatusToken(str, Enum):
    WINNER     = "winner"       # clear_winner
    EQUIVALENT = "equivalent"   # equivalent_candidates
    TENTATIVE  = "tentative"    # no_meaningful_effect


class VerdictToken(str, Enum):
    IMPROVED  = "improved"
    WORSENED  = "worsened"
    NEUTRAL   = "neutral"       # unchanged
    BASELINE  = "baseline"


class ApprovalToken(str, Enum):
    APPROVED      = "approved"
    REJECTED      = "rejected"
    AUTO_APPROVED = "auto_approved"


class ThresholdStateToken(str, Enum):
    OK      = "ok"
    WARN    = "warn"
    CAUTION = "caution"
    DANGER  = "danger"


class DirectionToken(str, Enum):
    BOTTLENECK_POSITIVE = "bottleneck_positive"  # SHAP > 0
    BOTTLENECK_NEGATIVE = "bottleneck_negative"  # SHAP < 0


class ReliabilityToken(str, Enum):
    HIGH = "HIGH"
    MED  = "MED"
    LOW  = "LOW"


# ── 공통 원자 ─────────────────────────────────────────────────────────────────

class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=False)


class KpiChange(_Base):
    """before/after/delta/pct_change 분해 — '인사: 안녕 / 이름: ...' 원칙."""
    kpi: str
    unit: str = ""
    now: float | int | None = None
    after: float | int | None = None
    delta: float | int | None = None
    pct_change: float | None = None
    reliability_token: ReliabilityToken | None = None


# ── 1. meta ──────────────────────────────────────────────────────────────────

class Meta(_Base):
    toolgroup: str
    process_name: str
    area_name: str | None = None
    severity: str                           # 원본 enum 값
    severity_token: SeverityToken
    severity_priority: int                  # 0=danger ~ 3=ok
    detected_at: str
    generated_at: str
    snapshot_time: float | None = None
    horizon_min: int | None = None
    schema_version: str = "report/1.0"


# ── 2. approval ──────────────────────────────────────────────────────────────

class Approval(_Base):
    status: str                             # "승인" / "반려"
    status_token: ApprovalToken
    approver_name: str | None = None
    approver_role: str | None = None
    approved_at: str | None = None
    comment: str | None = None
    rejection_reason: str | None = None
    selected_label: str | None = None


# ── 3. risk ──────────────────────────────────────────────────────────────────

class Risk(_Base):
    score: float | None = None              # 0-100
    score_unit: str = "score"
    score_threshold_state: ThresholdStateToken | None = None
    composite_score: float | None = None    # 0-1
    probability: float | None = None        # 0-1


# ── 4. confidence ────────────────────────────────────────────────────────────

class Confidence(_Base):
    level: str | None = None                # "HIGH" / "MEDIUM" / "LOW"
    level_token: ConfidenceToken | None = None
    needs_more_data: bool = False
    g_star_probability: float | None = None


# ── 5. if_no_action (compare.current_state.natural_forecast_2h) ──────────────

class IfNoAction(_Base):
    available: bool = False
    will_get_worse: bool | None = None
    horizon_min: int | None = None
    kpi_changes: list[KpiChange] = []


# ── 6. bottleneck_kpis (anchor TG 현재 KPI 카드) ──────────────────────────────

class KpiCard(_Base):
    key: str                                # "risk_score", "q_time_min", ...
    label: str
    value: float | int | None = None
    unit: str = ""
    threshold_state: ThresholdStateToken | None = None
    prev_value: float | int | None = None
    delta: float | int | None = None
    pct_change: float | None = None


# ── 7. bottleneck_trend (5점 시계열) ──────────────────────────────────────────

class TrendPoint(_Base):
    time_label: str                         # "T-240분"
    offset_min: int | None = None           # -240
    values: dict[str, float | None] = {}    # {feature_name: value}


# ── 8. diffusion ─────────────────────────────────────────────────────────────

class AffectedProcess(_Base):
    toolgroup: str
    utilization_pct: float | None = None
    wait_ratio: float | None = None
    wip: int | None = None
    impact_score: float | None = None
    data_quality_flags: list[str] = []      # ["wait_ratio_anomaly", ...]


class ForwardSimResult(_Base):
    toolgroup: str
    q_time_min_future: float | None = None
    wait_ratio_future: float | None = None
    wip_future: int | None = None
    is_bottleneck_predicted: bool | None = None


class ForwardSimulation(_Base):
    available: bool = False
    horizon_min: int | None = None
    results: list[ForwardSimResult] = []


class Diffusion(_Base):
    bottleneck_location: str
    diffusion_path: list[str] = []
    line_stop_expected_min: float | None = None
    risk_level: str                         # 원본 severity
    risk_level_token: SeverityToken
    at_risk_lots: float | int | None = None
    impact_pct: float | None = None
    affected_toolgroups: list[str] = []
    high_impact_processes: list[AffectedProcess] = []
    low_impact_processes: list[AffectedProcess] = []
    forward_simulation: ForwardSimulation = Field(default_factory=ForwardSimulation)


# ── 9. cause ─────────────────────────────────────────────────────────────────

class CausePrimary(_Base):
    category: str
    feature: str
    confidence: str
    confidence_token: ConfidenceToken
    reasoning: str = ""


class CauseDismissed(_Base):
    name: str
    reason: str = ""


class GStarSignificantKpi(_Base):
    kpi: str
    delta_mean: float
    t_p_adj: float
    significant: bool


class GStarMonteCarlo(_Base):
    """Monte Carlo 30회 시뮬 결과 — '30회 중 28회 병목 확인'."""
    n_total: int
    n_alarm: int
    alarm_ratio_pct: float | None = None


class CauseGStar(_Base):
    confirmed: bool
    probability: float | None = None
    monte_carlo: GStarMonteCarlo | None = None
    upstream_confirmed_toolgroups: list[str] = []
    significant_kpis: list[GStarSignificantKpi] = []


class ConsensusAxes(_Base):
    """SHAP·트렌드·업스트림·G* 4축 합의 보드."""
    shap_supports: bool
    trend_supports: bool
    upstream_supports: bool
    g_star_supports: bool
    axes_agreed_count: int = Field(ge=0, le=4)


class CauseCategoryItem(_Base):
    name: str
    features: list[str] = []
    shap_share_pct: float = 0.0
    n_trend_significant: int = 0
    upstream_match: bool = False
    g_star_confirmed: bool = False
    total_score: float = 0.0
    confidence: str
    confidence_token: ConfidenceToken


class ShapTopItem(_Base):
    rank: int
    feature: str
    value: float | None = None
    shap: float | None = None
    contribution_pct: float | None = None
    direction_token: DirectionToken | None = None


class EvidenceMatrixItem(_Base):
    feature: str
    votes: int = 0
    score: float = 0.0
    shap_rank: int | None = None
    shap_value: float | None = None
    trend_slope: float | None = None
    trend_r2: float | None = None
    trend_significant: bool = False
    upstream_match: bool = False
    g_star_p_value: float | None = None
    g_star_significant: bool = False
    confidence: str
    confidence_token: ConfidenceToken


class TrendSeriesFeature(_Base):
    values: list[float | None] = []
    slope_per_hour: float | None = None
    r2: float | None = None
    significant: bool = False


class TrendSeries(_Base):
    time_labels: list[str] = []             # ["T-240분", "T-180분", ...]
    time_offsets_min: list[int] = []        # [-240, -180, -120, -60, 0]
    features: dict[str, TrendSeriesFeature] = {}


class Cause(_Base):
    summary: str = ""
    primary: CausePrimary | None = None
    secondary_categories: list[str] = []
    dismissed: list[CauseDismissed] = []
    needs_more_data: bool = False
    g_star: CauseGStar | None = None
    consensus_axes: ConsensusAxes | None = None
    categories: list[CauseCategoryItem] = []
    shap_top: list[ShapTopItem] = []
    evidence_matrix: list[EvidenceMatrixItem] = []
    trend_series: TrendSeries | None = None
    upstream_suspects: list[str] = []


# ── 10. actions ──────────────────────────────────────────────────────────────

class ReleaseIntervalParam(_Base):
    """Release Interval은 의미 단위로 4개 필드로 분해."""
    current: float | None = None
    target: float | None = None
    delta: float | None = None
    unit: str = "min"


class LotAdjustmentParam(_Base):
    lot_plan_id: int | None = None
    lot_type: str | None = None
    product_name: str | None = None
    release_time: float | None = None
    whatif_release_time: float | None = None
    action_kind: str | None = None
    priority: int | None = None
    time_to_due: float | None = None
    zone: str | None = None


class ActionParams(_Base):
    release_interval: ReleaseIntervalParam | None = None
    lot_priority_rule: str | None = None
    superhotlot_enable: bool | None = None
    lot_adjustments: list[LotAdjustmentParam] = []


class KpiImpactItem(_Base):
    kpi: str
    unit: str = ""
    now: float | int | None = None
    after: float | int | None = None
    delta: float | int | None = None
    pct_change: float | None = None
    verdict: str | None = None
    verdict_token: VerdictToken | None = None
    confidence: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_width: float | None = None
    paired_t_p: float | None = None
    significant: bool | None = None


class OperationalAttrs(_Base):
    effort: int | None = None               # 0~4
    effort_max: int = 4
    scope: str | None = None                # "none" / "local" / "fab_wide"
    reversibility: str | None = None        # "high" / "med" / "low"


class SimulationStats(_Base):
    paired_n: int | None = None
    confidence: float | None = None
    verdict: str | None = None


class ActionCandidate(_Base):
    label: str
    kind: str
    is_baseline: bool = False
    is_approved: bool = False
    description: str = ""
    target_toolgroups: list[str] = []
    params: ActionParams | None = None
    kpi_impact: list[KpiImpactItem] = []
    operational: OperationalAttrs | None = None
    simulation: SimulationStats | None = None
    composite_score: float | None = None
    tradeoffs: list[str] = []
    comparison_basis: str = ""
    per_tg_forecasts: dict[str, dict[str, dict[str, float]]] = {}


class WhyNotOther(_Base):
    label: str
    reason: str


class Recommendation(_Base):
    headline: str = ""
    primary_reason: str = ""
    plan_description: str | None = None
    effect_and_risk: str | None = None
    approval_reason: str | None = None
    confidence_level: str | None = None
    confidence_token: ConfidenceToken | None = None
    tradeoffs: list[str] = []
    why_not_others: list[WhyNotOther] = []
    caveats: list[str] = []
    tiebreaker_chain: list[str] = []
    selected_by: str | None = None


class ImmediateAction(_Base):
    order: int
    text: str


class MonitoringCheck(_Base):
    kpi: str
    target: float | str | None = None
    unit: str = ""
    check_after_min: int | None = None


class Playbook(_Base):
    available: bool = False
    immediate_actions: list[ImmediateAction] = []
    monitoring: list[MonitoringCheck] = []
    rollback_condition: str | None = None


class Actions(_Base):
    available: bool = False
    decision_status: str | None = None
    decision_status_token: DecisionStatusToken | None = None
    decision_caveat: str = ""
    tiebreaker_used: str | None = None
    equivalent_set: list[str] = []
    approved_label: str | None = None
    candidates: list[ActionCandidate] = []
    recommendation: Recommendation | None = None
    playbook: Playbook = Field(default_factory=Playbook)


# ── 11. data_quality ─────────────────────────────────────────────────────────

class DataQualityWarning(_Base):
    code: str                               # "wait_ratio_anomaly", ...
    toolgroup: str | None = None
    field: str | None = None
    value: float | None = None
    message: str = ""


class DataQuality(_Base):
    status: str = "ok"                      # "ok" / "warning" / "error"
    warnings: list[DataQualityWarning] = []


# ── 12. provenance ───────────────────────────────────────────────────────────

class Provenance(_Base):
    snapshot_time: float | None = None
    t0: float | None = None
    run_id: str | None = None
    scenario_id: str | None = None
    target_toolgroups: list[str] = []


# ── 13. sections + rendered (마크다운) ────────────────────────────────────────

class Sections(_Base):
    """LLM이 만드는 마크다운 섹션. PR #3에서 1회 호출로 통합 예정."""
    header: str = ""
    review: str = ""
    summary: str = ""
    diffusion: str = ""
    cause: str = ""
    actions: str = ""


class Rendered(_Base):
    markdown: str = ""                      # 전체 마크다운 — 다운로드/PDF용


# ── 루트 ─────────────────────────────────────────────────────────────────────

class ReportV2(_Base):
    schema_version: str = "report/1.0"
    meta: Meta
    approval: Approval | None = None
    risk: Risk = Field(default_factory=Risk)
    confidence: Confidence = Field(default_factory=Confidence)
    if_no_action: IfNoAction = Field(default_factory=IfNoAction)
    bottleneck_kpis: list[KpiCard] = []
    bottleneck_trend: list[TrendPoint] = []
    diffusion: Diffusion | None = None
    cause: Cause | None = None
    actions: Actions = Field(default_factory=Actions)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    provenance: Provenance = Field(default_factory=Provenance)
    sections: Sections = Field(default_factory=Sections)
    rendered: Rendered = Field(default_factory=Rendered)
