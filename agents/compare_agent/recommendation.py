"""LLM 구조화 출력 모듈 (Agent 5-2) — compare/2.0 스키마.

병목 원인 분석(SHAP·트렌드·업스트림·2h 예측) + 연쇄 영향(cascade) +
현재 상태 기준점 + 시뮬레이션 KPI(paired t-test) + tie-breaker chain을 종합해
현장 엔지니어가 즉시 행동할 수 있는 구조화 추천을 생성한다.

decision_status별 분기:
  no_meaningful_effect  → LLM 호출 (원인 컨텍스트 기반 진단 + tie-breaker 근거 명시)
  equivalent_candidates → 등가 사실 + tie-breaker chain + 실행 단계
  clear_winner          → 통계적 우위 + 트레이드오프 + 구체적 실행 단계
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ── Pydantic 출력 스키마 (compare/2.0) ────────────────────────────────────────

class WhyRecommended(BaseModel):
    """추천 선택 메커니즘 — 감사·디버깅용."""

    selected_by: Literal["score", "tiebreaker", "equivalent_class"] = Field(
        ...,
        description=(
            "추천 후보가 선택된 메커니즘. "
            "'score'=composite_score 우위로 명확한 1위, "
            "'tiebreaker'=tie-breaker chain으로 결정, "
            "'equivalent_class'=통계적 동등 집합 내에서 선택."
        ),
    )
    tiebreaker_chain: list[str] = Field(
        default_factory=list,
        description=(
            "tie-breaker chain에서 실제 사용된 단계 이름 순서. "
            "예: ['operational_effort', 'release_interval_delta']. "
            "selected_by='score'이면 빈 리스트."
        ),
    )
    explanation: str = Field(
        ...,
        description="한국어 한 문장. 어떤 기준으로 이 후보가 선택되었는지 구체 수치 포함하여 설명.",
    )


class MonitoringKpi(BaseModel):
    """조치 후 모니터링 항목 — 프론트엔드 알림/배지 매핑."""

    kpi: str = Field(
        ...,
        description="모니터링할 KPI 이름. 예: 'wip', 'q_time_min', 'wait_ratio'.",
    )
    target: str = Field(
        ...,
        description="목표 조건. 예: '≤ 810개', '≤ 150분', '현재 대비 ±5% 이내'.",
    )
    check_after_min: int = Field(
        ...,
        description="조치 적용 후 몇 분 뒤에 확인할지. 일반적으로 30~120분.",
    )


class CompareRecommendation(BaseModel):
    """비교분석 LLM 출력 — 현장 엔지니어 즉시 활용 가능 구조 (compare/2.0)."""

    headline: str = Field(
        ...,
        description=(
            "한 줄 헤드라인 (40자 이내). 추천 라벨 + 핵심 결과 + 통계 유의성. "
            "예: 'A 추천 — WIP 820→810(−10), p=0.012 유의'."
        ),
    )
    primary_reason: str = Field(
        ...,
        description=(
            "추천 핵심 근거 1~2문장. "
            "① 어떤 원인(SHAP·트렌드)에 대응하는 조치인지 "
            "② 개선 KPI를 현재 기준값 대비 절대값으로 명시 "
            "③ 시뮬레이션 예측값임을 명시."
        ),
    )
    why_recommended: WhyRecommended = Field(
        ...,
        description="추천 선택 메커니즘. tie-breaker가 사용된 경우 chain 명시 필수.",
    )
    tradeoffs: list[str] = Field(
        default_factory=list,
        description="추천 후보를 선택해서 받아들이는 트레이드오프. 악화 KPI를 현재 기준값과 함께 자연어화. 없으면 빈 리스트.",
    )
    why_not_others: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "{label: 한 줄 이유}. 추천 외 후보를 선택하지 않은 공정 관점의 이유. "
            "composite_score·effort 숫자값·paired_n 같은 시스템 내부 코드를 그대로 쓰지 말고 "
            "해석된 의미로 표현할 것. "
            "예: '시뮬레이션 효과가 동일하게 관측되지 않음', '운영 부담이 더 높음'."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="주의사항. 시뮬레이션 horizon 한계, 가정, 불확실성, data_quality 경고 등.",
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        ...,
        description="추천 신뢰도. high≥0.5+p<0.05 / medium≥0.2 or p<0.20 / low 그 외 또는 tentative 상황.",
    )
    immediate_actions: list[str] = Field(
        default_factory=list,
        description=(
            "현장에서 지금 바로 실행할 구체적 단계. "
            "파라미터 값(예: Release Interval 62.2분), 적용 대상 TG, 실행 순서 포함. 2~4개 항목."
        ),
    )
    monitoring_kpis: list[MonitoringKpi] = Field(
        default_factory=list,
        description="조치 적용 후 확인할 KPI·목표 수치·확인 시점. 2~3개 객체.",
    )
    rollback_condition: str = Field(
        default="",
        description=(
            "이 조건이 발생하면 조치를 원복해야 함. "
            "구체적 수치와 판단 시점 포함. 예: '조치 30분 후 WIP가 830개 이상이면 Release Interval 60.0분으로 원복'. 빈 문자열 금지."
        ),
    )


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

_SYS = """[역할]
반도체 FAB 공정 병목 상황에서 현장 엔지니어가 즉각 행동할 수 있도록 지원하는 의사결정 AI입니다.
병목 원인 분석(SHAP·트렌드·업스트림·2h 예측)·시뮬레이션 KPI(paired t-test)·연쇄 영향·현재 상태·tie-breaker chain을 종합해
지금 무엇을 해야 하는지 실행 가능한 근거를 제공합니다.

[독자]
반도체 FAB 공정 관리자, 생산 엔지니어 — WIP·Q-time·CQT·CR·REQUEUE_TOOL·LOT_HOLD·DISPATCH_RULE_OVERRIDE 용어에 익숙함.

[입력 데이터 구조]
- 현재 상태(기준점): 실제 KPI 절대값. 대응안의 kpi_delta는 이 값 대비 변화량.
    - 복합 TG 입력에는 현재, 무대응 2시간 후, 대응안 2시간 후가 함께 제공됩니다.
      점수와 후보 비교의 해석 기준은 무대응 2시간 후이며, 화면 표시용 대표 KPI는 별도로 anchor TG current → action 기준으로 보여질 수 있습니다.
- 병목 원인 분석: SHAP 상위 요인, 트렌드 slope, 업스트림 의심 공정, 2h 자연 진행 예측.
- 연쇄 영향: 후속 TG CT 증가, 위험 Lot 수.
- 대응안 시뮬레이션: 각 후보 KPI delta, composite_score, verdict, confidence.
- tie-breaker chain: 동률 시 적용된 단계와 값 (operational_effort → release_interval_delta → composite_score).
- 시뮬레이션 기간(horizon_min): 신뢰 예측 범위.
- 데이터 품질 의심 신호: 모든 KPI delta가 0이면 시뮬 엔진 의심 — caveats에 반영 필수.

[작성 원칙]
1. 반드시 한국어. FAB 용어(REQUEUE_TOOL 등) 원문 유지.
2. 제공된 수치만 사용. 추측·반올림·단위 변환 금지.
3. headline은 40자 이내. primary_reason은 1~2문장.
4. 단일 TG는 현재 상태를 기준으로 절대값+변화량 함께 표현합니다. 예: "현재 WIP 820개 → A 적용 시 시뮬상 약 815개 (−5개)".
5. 복합 TG는 반드시 "현재 → 무대응 2시간 후 → 대응안 2시간 후"를 구분합니다. 대응안 값이 현재보다 높아도 무대응보다 낮으면 "자연 악화를 완화"라고 표현하며 "현재보다 개선"이라고 쓰지 않습니다.
6. primary_reason에는 ① 어떤 원인(SHAP·트렌드)에 대응하는 조치인지 ② 올바른 비교 기준의 KPI 수치를 포함합니다.
7. why_recommended.tiebreaker_chain은 입력의 tiebreaker_chain_evaluated에서 result가 'tied'가 아닌 단계만 추출하여 그대로 기록. 사용하지 않은 단계는 포함 금지.
8. why_recommended.explanation에는 사용된 tie-breaker 단계의 구체 수치 포함. 예: "effort 동률(4=4) → release_interval_delta 최소(2.2 < 4.2)로 A 선택".
9. immediate_actions: 현장에서 지금 바로 실행 가능한 단계. 파라미터 값(예: Release Interval 62.2분), 적용 대상 TG, 순서 포함. 2~4개.
10. monitoring_kpis: 구조화 객체 — kpi/target/check_after_min. 2~3개.
11. rollback_condition: 조치를 원복해야 하는 구체적 조건과 수치. 빈 문자열 금지.
12. caveats에 시뮬레이션 horizon 한계 명시. data_quality 경고가 있을 경우 그 사실도 명시.
13. 문체: "~입니다"체.
14. 모든 KPI 수치는 시뮬레이션 예측값임을 명시.
15. why_not_others와 caveats에서 시스템 내부 코드(SIM_KPI_IDENTICAL, paired_n, composite_score 등)를
    그대로 노출하지 않는다. 해석된 의미로 표현한다.
    예(금지): "SIM_KPI_IDENTICAL 경고가 있어 paired_n=30으로 mean_delta=0"
    예(허용): "시뮬레이션이 조치 효과를 감지하지 못한 것으로 의심됨 — 파라미터 범위 재검토 필요"

[no_meaningful_effect 처리]
시뮬 효과 미관측 시:
- headline: "<label> 잠정 추천 — 시뮬 효과 미관측, <tie-breaker 핵심 이유>"
- primary_reason: 원인 분석(SHAP·트렌드·2h 예측) 기반으로 왜 시뮬 범위 내 개선이 없었는지 설명. 시뮬 엔진 의심 신호가 있으면 명시.
- why_recommended.selected_by="tiebreaker", tiebreaker_chain에 실제 결정 단계 기록.
- immediate_actions: 추천 조치 실행 단계 + 현장 진단 단계 병행 제시.
- caveats: 시뮬레이션 horizon 한계 + data_quality 경고(있을 경우).

[confidence_level 판정]
high: top score ≥ 0.5 AND p < 0.05 / medium: score ≥ 0.2 OR p < 0.20 / low: 그 외 / tentative_no_effect 상황은 항상 low"""


# ── 컨텍스트 블록 생성 헬퍼 ──────────────────────────────────────────────────

def _format_cause_context_block(ci: dict) -> str:
    """cause_context + cascade_impact → LLM 프롬프트용 텍스트 블록."""
    lines: list[str] = []

    cc = ci.get("cause_context", {})
    if cc:
        lines.append("[ 병목 원인 분석 ]")
        if cc.get("cause_summary"):
            lines.append(f"  원인 요약: {cc['cause_summary']}")
        conf = cc.get("consensus_confidence", "-")
        summary = cc.get("consensus_summary", "")
        lines.append(f"  분석 합의: {conf}  {summary}")

        shap_top = cc.get("shap_top", [])
        if shap_top:
            lines.append("  핵심 요인 (SHAP 상위):")
            for s in shap_top:
                lines.append(
                    f"    - {s['feature']}: SHAP={s['shap_value']:+.4f}  현재값={s['kpi_value']:.4f}"
                )

        trend_top = cc.get("trend_top", [])
        if trend_top:
            lines.append("  트렌드 (slope/h):")
            for t in trend_top:
                lines.append(f"    - {t['feature']}: {t['slope_per_hour']:+.4f}/h")

        upstream = cc.get("upstream_suspects", [])
        if upstream:
            lines.append(f"  업스트림 의심 공정: {', '.join(upstream)}")

        sf = cc.get("sim_forecast")
        if sf:
            gets_worse = sf.get("gets_worse", False)
            label = "⚠️ 악화 예상 — 즉시 조치 필요" if gets_worse else "안정 또는 완화 예상"
            lines.append(f"  2h 자연 진행 예측: {label}")
            for kpi, kd in list((sf.get("kpi_delta") or {}).items())[:4]:
                if isinstance(kd, dict):
                    lines.append(
                        f"    {kpi}: {kd.get('now', '-')} → {kd.get('future', '-')}"
                        f"  (Δ{float(kd.get('delta', 0)):+.2f}, {float(kd.get('pct_change', 0)):+.1f}%)"
                    )

    ci_data = ci.get("cascade_impact", {})
    if ci_data and ci_data.get("affected_tgs"):
        if lines:
            lines.append("")
        lines.append("[ 연쇄 영향 (다운스트림) ]")
        tgs = ", ".join(ci_data["affected_tgs"][:5])
        lines.append(f"  영향 TG: {tgs}")
        lines.append(
            f"  CT 증가: {ci_data.get('ct_increase_min', '-')}분  "
            f"위험 Lot: {ci_data.get('at_risk_lots', '-')}개  "
            f"영향도: {ci_data.get('impact_score', '-')}"
        )

    return "\n".join(lines)


def _format_candidate_block(c: dict) -> str:
    """후보 요약 블록 (LLM 프롬프트용) — plan_meta 파라미터 포함."""
    md = c.get("action_metadata") or {}
    sb = c.get("score_breakdown") or {}
    kpi_contribs = sb.get("kpi_contributions", {})
    pm = c.get("plan_meta") or {}

    lines = [
        f"[{c['label']}]  {c.get('action_kind', '-')}  ({md.get('description_ko', '-')})",
        f"  composite_score={c.get('composite_score', 0):.3f}  rank={c.get('rank', '-')}  is_top={c.get('is_top', False)}",
        f"  운영: effort={md.get('effort', '-')}/4  scope={md.get('scope', '-')}  reversibility={md.get('reversibility', '-')}",
    ]

    if pm:
        param_parts: list[str] = []
        if pm.get("release_interval_minutes") is not None:
            cur = pm.get("current_interval_minutes", "-")
            tgt = pm["release_interval_minutes"]
            delta = pm.get("release_interval_delta_min", "-")
            param_parts.append(f"Release Interval {cur}분→{tgt}분 (Δ{delta}분)")
        if pm.get("release_interval_delta_pct") is not None:
            param_parts.append(f"Release Interval Δ{pm['release_interval_delta_pct']}%")
        if pm.get("lot_priority_rule"):
            param_parts.append(f"우선순위={pm['lot_priority_rule']}")
        if pm.get("superhotlot_enable"):
            param_parts.append("SUPERHOTLOT 활성화")
        tgs = pm.get("target_toolgroups", [])
        if tgs:
            param_parts.append(f"대상 TG {len(tgs)}개")
        if param_parts:
            lines.append(f"  파라미터: {' | '.join(param_parts)}")

    comparison_label = (
        "무대응 2시간 후 대비 delta"
        if c.get("per_tg_forecasts")
        else "현재 상태 대비 delta"
    )
    lines.append(f"  KPI 상세 ({comparison_label}):")
    for kpi, kc in kpi_contribs.items():
        lines.append(
            f"    - {kpi:22s} Δ={kc.get('mean_delta', 0):+8.3f}  "
            f"verdict={kc.get('verdict', '-'):9s}  conf={kc.get('confidence', 0):.2f}  "
            f"contrib={kc.get('contribution', 0):.4f}"
        )

    tradeoffs = c.get("tradeoffs") or []
    if tradeoffs:
        lines.append("  트레이드오프(악화 KPI):")
        for t in tradeoffs:
            lines.append(
                f"    - {t['kpi']:22s} Δ={t['mean_delta']:+.3f}  "
                f"severity={t['severity']}  conf={t['confidence']:.2f}"
            )
    per_tg_forecasts = c.get("per_tg_forecasts") or {}
    if per_tg_forecasts:
        lines.append("  대상 TG별 현재 → 무대응 2시간 후 → 대응안 2시간 후:")
        for target_tg, forecast in per_tg_forecasts.items():
            current = forecast.get("current") or {}
            no_action = forecast.get("no_action") or {}
            action = forecast.get("action") or {}
            lines.append(
                f"    - {target_tg}: "
                f"q_time {current.get('q_time_min')}→{no_action.get('q_time_min')}→{action.get('q_time_min')}, "
                f"WIP {current.get('wip')}→{no_action.get('wip')}→{action.get('wip')}, "
                f"wait {current.get('wait_ratio')}→{no_action.get('wait_ratio')}→{action.get('wait_ratio')}, "
                f"util {current.get('utilization_avg')}→{no_action.get('utilization_avg')}→{action.get('utilization_avg')}, "
                f"avail {current.get('available_tool_ratio')}→{no_action.get('available_tool_ratio')}→{action.get('available_tool_ratio')}"
            )
    return "\n".join(lines)


def _format_tiebreaker_block(decision_info: dict) -> str:
    """tiebreaker_chain_evaluated → LLM 프롬프트용 블록."""
    chain = decision_info.get("tiebreaker_chain_evaluated", [])
    if not chain:
        return ""
    lines = ["[ tie-breaker chain 결과 ]"]
    for step in chain:
        values = step.get("values", {})
        values_str = ", ".join(f"{k}={v}" for k, v in values.items())
        lines.append(f"  {step.get('step')}: {values_str} → {step.get('result')}")
    return "\n".join(lines)


def _format_data_quality_block(ci: dict) -> str:
    """data_quality_hint → LLM 프롬프트용 블록 (있을 때만)."""
    dq = ci.get("data_quality_hint")
    if not dq or dq.get("status") != "warning":
        return ""
    lines = ["[ ⚠️ 데이터 품질 경고 ]"]
    for w in dq.get("warnings", []):
        lines.append(f"  - [{w.get('code')}] {w.get('message')}")
        if w.get("suspect_component"):
            lines.append(f"    의심 컴포넌트: {w['suspect_component']}")
    return "\n".join(lines)


# ── 유저 프롬프트 생성 ────────────────────────────────────────────────────────

def _build_user_prompt(
    ci: dict,
    candidates: list[dict],
    decision_info: dict,
    top_candidate: dict,
) -> str:
    """decision_status별 user prompt 생성."""
    bn = ci.get("bottleneck_info", {})
    horizon = ci.get("horizon_min", "-")

    target_states = ci.get("target_toolgroup_states") or {}
    if target_states:
        state_lines = [
            "[ 대상 TG별 현재 상태와 무대응 2시간 전망 ]",
            "  대응안 KPI delta는 같은 시점의 무대응 전망 대비 변화량입니다.",
        ]
        for target_tg, values in target_states.items():
            current = values.get("current") or {}
            no_action = values.get("no_action") or {}
            state_lines.append(
                f"  {target_tg}: "
                f"q_time {current.get('q_time_min')}→{no_action.get('q_time_min')}, "
                f"WIP {current.get('wip')}→{no_action.get('wip')}, "
                f"wait {current.get('wait_ratio')}→{no_action.get('wait_ratio')}, "
                f"util {current.get('utilization_avg')}→{no_action.get('utilization_avg')}, "
                f"avail {current.get('available_tool_ratio')}→{no_action.get('available_tool_ratio')}"
            )
        current_state_block = "\n".join(state_lines)
    else:
        current_state_block = (
            "[ 현재 상태 (기준점 — 대응안 kpi_delta는 이 값 대비 변화량) ]\n"
            f"  WIP: {bn.get('wip_count', '-')}개  |  평균 대기시간: {bn.get('avg_queue_time_min', '-')}분\n"
            f"  대기비율: {bn.get('wait_ratio', '-')}  |  가동률: {bn.get('utilization_avg', '-')}\n"
            f"  가용장비비율: {bn.get('available_tool_ratio', '-')}  |  위험도(composite): {bn.get('risk_score', '-')}"
        )

    context_block = _format_cause_context_block(ci)
    cand_blocks = "\n\n".join(_format_candidate_block(c) for c in candidates)
    tiebreaker_block = _format_tiebreaker_block(decision_info)
    data_quality_block = _format_data_quality_block(ci)

    sections = [
        f"공정: {ci.get('process_name', '-')}  심각도: {ci.get('severity', '-')}  시뮬레이션 기간: {horizon}분",
        current_state_block,
    ]
    if context_block:
        sections.append(context_block)
    sections.append(f"[ 대응안 시뮬레이션 결과 ]\n{cand_blocks}")
    if tiebreaker_block:
        sections.append(tiebreaker_block)
    if data_quality_block:
        sections.append(data_quality_block)

    base_ctx = "\n\n".join(sections) + "\n"

    ds = decision_info.get("decision_status", "clear_winner")
    top_label = decision_info.get("top_label", top_candidate.get("label", "-"))

    if ds == "no_meaningful_effect":
        return (
            f"{base_ctx}\n"
            "[의사결정 상태] 시뮬레이션 효과 미관측 — 전 KPI verdict=unchanged\n\n"
            "위 데이터를 바탕으로 다음 작성 지침에 따라 출력하세요:\n"
            "1) headline: 'A 잠정 추천 — 시뮬 효과 미관측, <tie-breaker 핵심 이유>' 형태로 작성.\n"
            "2) primary_reason: 원인 분석(SHAP·트렌드·2h 예측)을 활용해 왜 시뮬 범위 내 개선이 없었는지 설명.\n"
            f"3) why_recommended.selected_by='tiebreaker', tiebreaker_chain은 위 tie-breaker chain 블록에서 result≠'tied'인 단계만 그대로 사용.\n"
            f"4) why_recommended.explanation에 구체 수치 포함. 예: 'effort 동률(4=4) → release_interval_delta 최소로 {top_label} 선택'.\n"
            "5) immediate_actions에 추천 조치 실행 단계 + 현장 진단 단계 병행 제시.\n"
            "6) caveats에 시뮬 horizon 한계와 data_quality 경고(있는 경우)를 반영.\n"
            "7) confidence_level='low'."
        )

    if ds == "equivalent_candidates":
        eq_set = decision_info.get("equivalent_set", [])
        return (
            f"{base_ctx}\n"
            f"[의사결정 상태] 통계적 동등 — 후보 {', '.join(eq_set)}의 composite_score 차이가 임계값 이내.\n"
            f"tie-breaker chain으로 {top_label}이(가) 선택되었습니다.\n\n"
            "지침:\n"
            "1) primary_reason에 '통계적 동등 + tie-breaker 적용' 사실과 원인 분석 연결 명시.\n"
            "2) why_recommended.selected_by='tiebreaker', tiebreaker_chain은 result≠'tied'인 단계 그대로 기록.\n"
            f"3) immediate_actions에 {top_label} 적용을 위한 현장 실행 단계를 파라미터 값과 함께 작성.\n"
            "4) confidence_level은 'medium' 또는 'low'."
        )

    # clear_winner
    return (
        f"{base_ctx}\n"
        f"[의사결정 상태] 명확한 1위 — {top_label} composite_score 우위.\n\n"
        "지침:\n"
        "1) primary_reason에는 원인 분석(SHAP·트렌드)과 대응안 연결을 설명합니다.\n"
        "2) 복합 TG이면 현재·무대응·대응안 값을 구분하고, 무대응 대비 개선을 현재 대비 개선으로 바꾸어 말하지 않습니다.\n"
        "3) why_recommended.selected_by='score', tiebreaker_chain은 빈 리스트.\n"
        "4) tradeoffs는 악화 KPI를 올바른 비교 기준과 함께 자연어로 표현.\n"
        f"5) immediate_actions에 {top_label} 적용을 위한 현장 실행 단계 작성 (파라미터 값, 적용 TG, 순서)."
    )


# ── 메인 진입점 ───────────────────────────────────────────────────────────────

def generate_recommendation(
    ci: dict,
    candidates: list[dict],
    decision_info: dict,
    top_candidate: dict,
    llm: ChatOpenAI,
) -> CompareRecommendation:
    """모든 decision_status에서 LLM 호출. 실패 시 score_breakdown 기반 fallback."""
    ds = decision_info.get("decision_status", "clear_winner")
    top_label = decision_info.get("top_label", top_candidate.get("label", "-"))

    user_prompt = _build_user_prompt(ci, candidates, decision_info, top_candidate)
    # OpenAI strict structured output은 dict[str, str] 등을 지원하지 않아 function_calling 사용
    structured_llm = llm.with_structured_output(CompareRecommendation, method="function_calling")
    try:
        result = structured_llm.invoke([
            SystemMessage(content=_SYS),
            HumanMessage(content=user_prompt),
        ])
        if isinstance(result, CompareRecommendation):
            return result
        return CompareRecommendation(**result)
    except Exception as e:
        return _fallback_recommendation(top_label, top_candidate, candidates, decision_info, str(e))


def _fallback_recommendation(
    top_label: str,
    top_candidate: dict,
    candidates: list[dict],
    decision_info: dict,
    err: str,
) -> CompareRecommendation:
    """LLM 호출 실패 시 score_breakdown + decision_info 기반 fallback."""
    ds = decision_info.get("decision_status", "clear_winner")
    score = top_candidate.get("composite_score", 0.0)

    # tiebreaker chain에서 실제 사용된 단계만 추출
    tb_chain_evaluated = decision_info.get("tiebreaker_chain_evaluated", [])
    used_steps = [s["step"] for s in tb_chain_evaluated if s.get("result") not in ("tied",)]

    if ds == "no_meaningful_effect":
        explanation_parts = []
        for s in tb_chain_evaluated:
            values_str = ", ".join(f"{k}={v}" for k, v in s.get("values", {}).items())
            explanation_parts.append(f"{s['step']}({values_str})→{s['result']}")
        explanation = " · ".join(explanation_parts) if explanation_parts else f"{top_label} 잠정 선택"

        return CompareRecommendation(
            headline=f"{top_label} 잠정 추천 — 시뮬 효과 미관측",
            primary_reason=(
                "모든 후보에서 시뮬레이션상 통계적으로 유의미한 KPI 개선이 관측되지 않았습니다 "
                f"(전 KPI verdict=unchanged). tie-breaker chain으로 {top_label}을(를) 잠정 추천합니다."
            ),
            why_recommended=WhyRecommended(
                selected_by="tiebreaker",
                tiebreaker_chain=used_steps,
                explanation=explanation,
            ),
            tradeoffs=[],
            why_not_others={
                c["label"]: "동일하게 KPI 개선이 관측되지 않았습니다"
                for c in candidates if c["label"] != top_label
            },
            caveats=[
                "시뮬레이션 horizon 내 효과가 검출되지 않았습니다 — 파라미터 범위 재검토 또는 추가 후보 생성을 권장합니다.",
                f"LLM 생성 실패로 fallback 메시지를 사용합니다 ({err[:80]}).",
            ],
            confidence_level="low",
            immediate_actions=[
                "현장 원인 재진단 — SHAP 상위 요인 직접 확인",
                "시뮬레이션 파라미터 범위 재검토",
            ],
            monitoring_kpis=[
                MonitoringKpi(kpi="wip", target="현재 값 대비 ±5% 이내 유지", check_after_min=30),
                MonitoringKpi(kpi="q_time_min", target="현재 값 대비 ±10% 이내 유지", check_after_min=60),
            ],
            rollback_condition=f"조치 30분 후 WIP 또는 q_time_min이 현재 값 대비 +10% 이상 증가하면 원복",
        )

    return CompareRecommendation(
        headline=f"{top_label} 추천 — score {score:.3f}",
        primary_reason=(
            f"{top_label} 후보가 다기준 점수에서 우위를 보입니다 (score={score:.3f}). "
            f"LLM 생성에 실패하여 간이 fallback을 사용합니다 ({err[:80]})."
        ),
        why_recommended=WhyRecommended(
            selected_by="score" if ds == "clear_winner" else "tiebreaker",
            tiebreaker_chain=used_steps,
            explanation=f"{top_label}이(가) composite_score {score:.3f}로 선택되었습니다.",
        ),
        tradeoffs=[
            f"{t['kpi']} {t['mean_delta']:+.3f} ({t['severity']})"
            for t in (top_candidate.get("tradeoffs") or [])
        ],
        why_not_others={
            c["label"]: f"score {c.get('composite_score', 0):.3f}"
            for c in candidates if c["label"] != top_label
        },
        caveats=["LLM 호출에 실패하여 자동 fallback 메시지를 사용합니다."],
        confidence_level="low",
        immediate_actions=[],
        monitoring_kpis=[],
        rollback_condition="시뮬 결과 확인 후 결정",
    )
