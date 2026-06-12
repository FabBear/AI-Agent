"""report_agent — narrative 생성 (LLM) + section 렌더링 (코드).

PR #3 설계
=========
- LLM은 narrative(짧은 문장)만 작성한다. 표·수치는 코드가 만든다.
- 4개 섹션의 narrative를 1번의 structured-output 호출로 동시에 받는다.
  · 토큰 비용 ~70% 절감
  · 일관성: 동일 컨텍스트에서 4개 결론이 나오므로 "심화 중 vs 정상" 모순 없음
- 표는 ReportV2(builder.py 산출)에서 결정론적으로 렌더링한다.
  · LLM 환각 / `-` 빈칸 문제 원천 차단
  · 컬럼·순서 회귀 테스트 가능
- 마크다운 외형은 기존과 유사하게 유지한다.

호출 흐름
========
narrate(report_v2) → ReportNarration         # 단일 LLM 호출
render_sections(report_v2, narration) → 4 sections dict
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents import config
from agents.report_agent.adapter import pick_approved_option  # noqa: F401  # 외부 import 보존
from agents.report_agent.schema import (
    ActionCandidate,
    ReportV2,
    SeverityToken,
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")


# ── LLM ──────────────────────────────────────────────────────────────────────

_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                "  프로젝트 루트의 .env 파일에 OPENAI_API_KEY를 추가하세요."
            )
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            max_completion_tokens=1500,   # PR #2 이전: 4 × 3000. 현재: 1회 1500.
        )
    return _llm


# ── narrative 구조 (LLM이 채울 부분) ──────────────────────────────────────────

class ReportNarration(BaseModel):
    """LLM이 1회 호출로 만드는 4개 섹션 narrative.

    LLM은 표 · 수치 · 마크다운 헤더를 만들지 않는다. 오직 문장만.
    각 필드 description이 LLM 프롬프트 역할도 한다 (structured output).
    """

    tldr_3lines: list[str] = Field(
        ...,
        min_length=3, max_length=3,
        description=(
            "3줄 핵심 요약. 각 줄은 한 문장. "
            "줄 1: 병목 상황 + risk_score 언급. "
            "줄 2: 생산 영향 + load_ratio(wait_ratio) 언급. "
            "줄 3: 긴급 조치 필요성. "
            "문체는 '~됨/~함/~임' 으로 통일. 마크다운 기호 없이 순수 문장만."
        ),
    )
    diffusion_interpretation: str = Field(
        ...,
        description=(
            "확산 영향에 대한 한~두 문장 해석. "
            "diffusion_path를 자연어로 풀어내고, bottleneck_trend를 보고 "
            "Q-time/WIP가 심화/완화/유지 중인지 한 문장으로 요약. "
            "표는 만들지 말 것 — 표는 코드가 생성함. 문체 '~됨/~함/~임'."
        ),
    )
    cause_judgment: str = Field(
        ...,
        description=(
            "원인 분석 판정. judgment.primary_reasoning을 그대로 인용하거나 "
            "한 문장 요약. 이어서 secondary_causes가 있으면 '보조 원인: ...', "
            "dismissed가 있으면 '기각: ... (이유: ...)' 추가. 2~4문장 이내."
        ),
    )
    actions_summary: str = Field(
        ...,
        description=(
            "승인된 대응안 요약 narrative. 'kpi_impact를 해석하여 한~두 문장'. "
            "예: 'A 대응안은 Q-time을 145.3분에서 100.1분으로 감소시킬 것으로 "
            "예측됨(시뮬레이션 기준).' 반려 상황이면 '반려됨 — 사유: ...' 한 줄. "
            "수치는 데이터에 있는 값을 그대로 사용. 문체 '~됨/~함/~임'."
        ),
    )


# ── 시스템 프롬프트 (기존 _SYS 보존 + 표 작성 지시 제거) ────────────────────────

_SYS_NARRATE = """[역할]
당신은 반도체 FAB 공정 병목 대응 의사결정 보고서의 narrative(서술 문장)를
작성하는 전문 AI입니다. 표 · KPI 카드 · 수치 비교는 별도 코드가 결정론적으로
생성하므로, 당신은 반드시 문장(narrative)만 작성합니다.

[독자]
- 반도체 FAB 공정 관리자, 생산 엔지니어
- WIP · Q-time · CT · TH · Load Ratio · CR · SHAP · SuperHotLot ·
  Dispatch Rule 등 FAB 공정 용어에 익숙합니다
- 병목 발생 시 빠른 의사결정이 필요하므로 핵심 수치와 근거 중심으로 판단합니다
- 모호한 표현을 신뢰하지 않으며, 데이터에 기반한 명확한 서술을 요구합니다

[도메인 지식 — 이 기준으로 데이터를 해석하세요]
risk_score 등급:
  · 정상   : 25 이하
  · MEDIUM : 26~47  → 주의 필요
  · HIGH   : 47~60  → 비정상 진입, 즉각 모니터링
  · CRITICAL: 60 이상 → 즉시 대응 필요

Load Ratio 해석:
  · < 0.75       : 여유 있음
  · 0.75 ~ 0.90  : 주의
  · 0.90 ~ 1.00  : capacity 여유 낮음
  · > 1.00       : 과부하 / 2차 병목 가능

[작성 원칙]
1. 반드시 한국어. 장비명·KPI·Lot명·대응안 종류는 원문 유지.
2. 제공된 수치는 그대로 사용. 반올림·단위변환·재계산 금지.
3. 데이터에 없는 내용은 절대 추측·창작 금지. 없으면 '-' 또는 생략.
4. 객관적·사실 기반 단문. "아마도" "가능성이 있습니다" 금지.
   단, 시뮬레이션 예측값은 "시뮬레이션 기준" 또는 "예측값" 명시.
5. 능동태 단문. 한 문장 최대 두 줄.
6. 서론·결론·부연 설명 추가 금지.
7. 문체 통일: "~됨", "~함", "~임". (~습니다 · ~이다 · ~합니다 금지)

[금지 사항 — 매우 중요]
- 마크다운 표(| ... |) 작성 금지. 표는 별도 코드가 만듭니다.
- 마크다운 헤더(##, ###) 작성 금지. 섹션 구조는 코드가 만듭니다.
- 수치 환각 금지. 입력 데이터에 없는 숫자는 절대 만들지 않습니다.
- 영어와 한국어 혼용 금지(지정된 FAB 용어 제외)."""


# ── 핵심 LLM 호출 ────────────────────────────────────────────────────────────

def narrate(report_v2: ReportV2) -> ReportNarration:
    """단일 LLM 호출로 4개 narrative 동시 생성 (structured output).

    실패 시 결정론적 fallback narration을 반환한다.
    """
    llm = _get_llm().with_structured_output(ReportNarration)
    user_prompt = _build_user_prompt(report_v2)

    try:
        result = llm.invoke([
            {"role": "system", "content": _SYS_NARRATE},
            {"role": "user", "content": user_prompt},
        ])
        if not isinstance(result, ReportNarration):
            result = ReportNarration.model_validate(result)
        return result
    except Exception:
        return _fallback_narration(report_v2)


def _build_user_prompt(report_v2: ReportV2) -> str:
    """LLM에게 ReportV2의 핵심 데이터만 보여준다.

    표 데이터 전체를 다 넣지 않는다 — 표는 코드가 만들고, LLM은
    narrative에 필요한 핵심 사실만 받는다. 토큰 절약 + 환각 방지.
    """
    meta = report_v2.meta
    risk = report_v2.risk
    diffusion = report_v2.diffusion
    cause = report_v2.cause
    actions = report_v2.actions
    inact = report_v2.if_no_action

    cause_snip = {}
    if cause:
        cause_snip = {
            "summary": cause.summary,
            "primary": cause.primary.model_dump() if cause.primary else None,
            "secondary_categories": cause.secondary_categories,
            "dismissed": [d.model_dump() for d in cause.dismissed],
            "needs_more_data": cause.needs_more_data,
            "consensus_axes": cause.consensus_axes.model_dump() if cause.consensus_axes else None,
        }

    actions_snip = {
        "available": actions.available,
        "decision_status": actions.decision_status,
        "approved_label": actions.approved_label,
        "approved_candidate": None,
    }
    if actions.available:
        approved = next((c for c in actions.candidates if c.is_approved), None)
        if approved:
            actions_snip["approved_candidate"] = {
                "label": approved.label,
                "kind": approved.kind,
                "description": approved.description,
                "kpi_impact_summary": [
                    {
                        "kpi": k.kpi,
                        "now": k.now,
                        "after": k.after,
                        "delta": k.delta,
                        "pct_change": k.pct_change,
                        "unit": k.unit,
                        "verdict": k.verdict,
                    }
                    for k in approved.kpi_impact
                ],
            }

    approval = report_v2.approval
    approval_snip = approval.model_dump() if approval else None

    import json as _json
    payload = {
        "process_name": meta.process_name,
        "severity": meta.severity,
        "risk_score": risk.score,
        "if_no_action": inact.model_dump(),
        "bottleneck_kpis_summary": [
            {"key": k.key, "value": k.value, "unit": k.unit, "delta": k.delta}
            for k in report_v2.bottleneck_kpis
        ],
        "diffusion": {
            "bottleneck_location": diffusion.bottleneck_location if diffusion else None,
            "diffusion_path": diffusion.diffusion_path if diffusion else [],
            "line_stop_expected_min": diffusion.line_stop_expected_min if diffusion else None,
            "high_impact_count": len(diffusion.high_impact_processes) if diffusion else 0,
        },
        "bottleneck_trend": [t.model_dump() for t in report_v2.bottleneck_trend],
        "cause": cause_snip,
        "actions": actions_snip,
        "approval": approval_snip,
    }
    return (
        "아래 보고서 데이터를 바탕으로 4개 섹션의 narrative만 작성하세요.\n"
        "표는 만들지 마세요 — 코드가 만듭니다.\n\n"
        f"```json\n{_json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )


def _fallback_narration(report_v2: ReportV2) -> ReportNarration:
    """LLM 호출 실패 시 결정론적 narration. 'LLM 실패' 표시 포함."""
    meta = report_v2.meta
    risk = report_v2.risk
    return ReportNarration(
        tldr_3lines=[
            f"{meta.process_name}에서 {meta.severity} 수준 병목이 감지됨 (risk_score {risk.score}).",
            "정확한 영향·KPI는 아래 표 참조.",
            "[LLM narrative 생성 실패 — 데이터만 표시됨]",
        ],
        diffusion_interpretation="[LLM narrative 생성 실패 — 표 데이터만 표시됨]",
        cause_judgment="[LLM narrative 생성 실패 — 표 데이터만 표시됨]",
        actions_summary="[LLM narrative 생성 실패 — 표 데이터만 표시됨]",
    )


# ── 표 렌더링 (결정론적, 코드만) ──────────────────────────────────────────────

def _md_cell(v) -> str:
    """마크다운 셀 표시. None / "" → '-'."""
    if v is None or v == "":
        return "-"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        # 자연스러운 소수점: 정수면 int, 아니면 4자리
        if v.is_integer():
            return str(int(v))
        return f"{v:.4g}"
    return str(v)


def _render_kpi_table(report_v2: ReportV2) -> str:
    """## 1. 요약 — 핵심 지표 표."""
    bn_info_rows = []
    bn_info_rows.append(("심각도", report_v2.meta.severity))
    if report_v2.risk.score is not None:
        bn_info_rows.append(("risk_score", _md_cell(report_v2.risk.score)))

    # KPI 카드 → 표
    kpi_label_map = {
        "q_time_min": "현재 평균 대기시간",
        "wait_ratio": "load_ratio (wait_ratio)",
        "wip": "WIP",
        "utilization_avg": "가동률",
        "available_tool_ratio": "가용 호기 비율",
        "max_util": "최대 가동률",
    }
    for card in report_v2.bottleneck_kpis:
        if card.key == "risk_score":
            continue
        label = kpi_label_map.get(card.key, card.label)
        unit_suffix = ""
        if card.key == "q_time_min":
            unit_suffix = "분"
        elif card.key == "utilization_avg" or card.key == "max_util":
            # 비율을 % 로 표시
            if card.value is not None:
                display = f"{card.value * 100:.1f}%"
                bn_info_rows.append((label, display))
                continue
        elif card.key == "wip":
            unit_suffix = "개"
        bn_info_rows.append((label, _md_cell(card.value) + unit_suffix))

    lines = ["| 지표 | 값 |", "|------|----|"]
    for label, val in bn_info_rows:
        lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def _render_affected_table(report_v2: ReportV2) -> str:
    """확산 — 공정별 가동률 표 (high_impact 상위)."""
    diffusion = report_v2.diffusion
    if diffusion is None or not diffusion.high_impact_processes:
        return "| 공정 | 가동률(%) | wait_ratio | WIP | 상태 |\n|------|-----------|-----------|-----|------|\n| - | - | - | - | - |"

    lines = ["| 공정 | 가동률(%) | wait_ratio | WIP | 상태 |", "|------|-----------|-----------|-----|------|"]
    for p in diffusion.high_impact_processes:
        status = "영향"
        if p.data_quality_flags:
            status = f"영향 ⚠️ {','.join(p.data_quality_flags)}"
        lines.append(
            f"| {p.toolgroup} | {_md_cell(p.utilization_pct)} | "
            f"{_md_cell(p.wait_ratio)} | {_md_cell(p.wip)} | {status} |"
        )
    return "\n".join(lines)


def _render_forward_sim_table(report_v2: ReportV2) -> str:
    diffusion = report_v2.diffusion
    if diffusion is None or not diffusion.forward_simulation.available:
        return "| Tool Group | wait_ratio (미래) | WIP (미래) | 병목 예측 |\n|------------|-----------------|----------|----------|\n| - | - | - | - |"

    fwd = diffusion.forward_simulation
    lines = ["| Tool Group | wait_ratio (미래) | WIP (미래) | 병목 예측 |",
             "|------------|-----------------|----------|----------|"]
    for r in fwd.results:
        bn = "✅ 병목" if r.is_bottleneck_predicted else "✅ 정상"
        lines.append(
            f"| {r.toolgroup} | {_md_cell(r.wait_ratio_future)} | "
            f"{_md_cell(r.wip_future)} | {bn} |"
        )
    return "\n".join(lines)


def _render_top_causes_table(report_v2: ReportV2) -> str:
    cause = report_v2.cause
    if cause is None or not cause.shap_top:
        return "| 순위 | 원인(feature) | 기여도(%) | 현재값 |\n|------|--------------|----------|--------|\n| - | - | - | - |"

    # evidence_matrix에서 votes/confidence 보강
    evid_map = {e.feature: e for e in cause.evidence_matrix}
    lines = ["| 순위 | 원인(feature) | 기여도(%) | 현재값 | 4개 분석 수렴 | 신뢰도 |",
             "|------|--------------|----------|--------|-------------|--------|"]
    for s in cause.shap_top[:5]:
        e = evid_map.get(s.feature)
        votes = f"{e.votes}/4" if e else "-"
        conf = e.confidence if e else "-"
        lines.append(
            f"| {s.rank} | {s.feature} | {_md_cell(s.contribution_pct)} | "
            f"{_md_cell(s.value)} | {votes} | {conf} |"
        )
    return "\n".join(lines)


def _render_shap_table(report_v2: ReportV2) -> str:
    cause = report_v2.cause
    if cause is None or not cause.shap_top:
        return "| 피처명 | 현재값 | 기여도(%) | 방향 |\n|--------|--------|----------|------|\n| - | - | - | - |"

    lines = ["| 피처명 | 현재값 | 기여도(%) | 방향 |", "|--------|--------|----------|------|"]
    for s in cause.shap_top[:5]:
        direction = "병목 쪽으로 기여(+)" if s.direction_token and s.direction_token.value == "bottleneck_positive" else "병목 완화(-)"
        lines.append(
            f"| {s.feature} | {_md_cell(s.value)} | {_md_cell(s.contribution_pct)} | {direction} |"
        )
    return "\n".join(lines)


def _render_trend_table(report_v2: ReportV2) -> str:
    if not report_v2.bottleneck_trend:
        return "| 시각 | (데이터 없음) |\n|------|--------------|"

    points = report_v2.bottleneck_trend
    feat_keys = list(points[0].values.keys())
    if not feat_keys:
        return "| 시각 | (데이터 없음) |\n|------|--------------|"

    header = "| 시각 | " + " | ".join(feat_keys) + " |"
    sep = "|------|" + "---------|" * len(feat_keys)
    lines = [header, sep]
    for p in points:
        cells = [p.time_label] + [_md_cell(p.values.get(k)) for k in feat_keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_trend_stats_table(report_v2: ReportV2) -> str:
    cause = report_v2.cause
    if cause is None or cause.trend_series is None or not cause.trend_series.features:
        return "| 피처 | 시간당 변화율 | R² | 통계 유의 |\n|------|------------|-----|---------|\n| - | - | - | - |"

    lines = ["| 피처 | 시간당 변화율 | R² | 통계 유의 |", "|------|------------|-----|---------|"]
    for feat, info in cause.trend_series.features.items():
        slope = info.slope_per_hour
        slope_disp = "-" if slope is None else (f"+{slope:.4g}/h" if slope >= 0 else f"{slope:.4g}/h")
        r2_disp = _md_cell(info.r2)
        sig = "✅" if info.significant else "❌"
        lines.append(f"| {feat} | {slope_disp} | {r2_disp} | {sig} |")
    return "\n".join(lines)


def _render_evidence_table(report_v2: ReportV2) -> str:
    cause = report_v2.cause
    if cause is None or not cause.evidence_matrix:
        return "| 피처 | SHAP | 트렌드 유의 | 업스트림 일치 | G* 유의 | 수렴 수 | 신뢰도 |\n|------|------|-----------|------------|--------|--------|--------|\n| - | - | - | - | - | - | - |"

    lines = ["| 피처 | SHAP | 트렌드 유의 | 업스트림 일치 | G* 유의 | 수렴 수 | 신뢰도 |",
             "|------|------|-----------|------------|--------|--------|--------|"]
    for e in cause.evidence_matrix[:8]:
        shap_arrow = "-" if e.shap_value is None else (f"+{e.shap_value:.4g}" if e.shap_value >= 0 else f"{e.shap_value:.4g}")
        lines.append(
            f"| {e.feature} | {shap_arrow} | {_md_cell(e.trend_significant)} | "
            f"{_md_cell(e.upstream_match)} | {_md_cell(e.g_star_significant)} | "
            f"{e.votes}/4 | {e.confidence} |"
        )
    return "\n".join(lines)


def _render_candidates_compare_table(report_v2: ReportV2) -> str:
    actions = report_v2.actions
    if not actions.candidates:
        return "| 대응안 | 종류 | 설명 | 대기시간 변화 | WIP 변화 | 시뮬 신뢰도 | 운영 부담 | 영향 범위 | 가역성 |\n|--------|------|------|-------------|---------|-----------|----------|----------|--------|\n| - | - | - | - | - | - | - | - | - |"

    lines = ["| 대응안 | 종류 | 설명 | 대기시간 변화 | WIP 변화 | 시뮬 신뢰도 | 운영 부담 | 영향 범위 | 가역성 |",
             "|--------|------|------|-------------|---------|-----------|----------|----------|--------|"]
    for c in actions.candidates:
        label = c.label + (" ✅" if c.is_approved else "")
        kind = c.kind
        desc = (c.description or "")[:40]
        q_kpi = next((k for k in c.kpi_impact if k.kpi == "q_time_min"), None)
        wip_kpi = next((k for k in c.kpi_impact if k.kpi == "wip"), None)
        q_delta = _md_cell(q_kpi.delta) + "분" if q_kpi and q_kpi.delta is not None else "-"
        wip_delta = _md_cell(wip_kpi.delta) if wip_kpi and wip_kpi.delta is not None else "-"
        sim_conf = "-"
        if c.simulation and c.simulation.confidence is not None:
            sim_conf = f"{int(c.simulation.confidence * 100)}%"
        op_effort = "-"
        op_scope = "-"
        op_rev = "-"
        if c.operational:
            op_effort = f"{c.operational.effort}/{c.operational.effort_max}" if c.operational.effort is not None else "-"
            op_scope = c.operational.scope or "-"
            op_rev = c.operational.reversibility or "-"
        lines.append(
            f"| {label} | {kind} | {desc} | {q_delta} | {wip_delta} | "
            f"{sim_conf} | {op_effort} | {op_scope} | {op_rev} |"
        )
    return "\n".join(lines)


# ── 섹션 렌더링 (narrative + 표를 합쳐 마크다운 섹션 생성) ────────────────────

def _render_summary(report_v2: ReportV2, narration: ReportNarration) -> str:
    quote = "\n".join(f"> **{line}**" for line in narration.tldr_3lines)
    return (
        "## 1. 요약\n\n"
        f"{quote}\n\n"
        f"{_render_kpi_table(report_v2)}\n\n"
        "---"
    )


def _render_diffusion(report_v2: ReportV2, narration: ReportNarration) -> str:
    diffusion = report_v2.diffusion
    fwd_available = diffusion.forward_simulation.available if diffusion else False
    horizon = diffusion.forward_simulation.horizon_min if (diffusion and fwd_available) else 120
    path_arrow = " → ".join(diffusion.diffusion_path) if diffusion else "-"
    risk_lvl = diffusion.risk_level if diffusion else "-"
    line_stop = diffusion.line_stop_expected_min if diffusion else None
    line_stop_disp = f"{int(line_stop)}분" if line_stop is not None else "-"

    return (
        "## 2. 확산 영향 분석\n\n"
        f"**탐지시각**: {report_v2.meta.detected_at}\n\n"
        f"### ① 병목 발생 여부 및 위치\n"
        f"{narration.diffusion_interpretation}\n"
        f"- **확산 경로**: {path_arrow}\n\n"
        f"### ② 확산 현황 (공정별 가동률 현황)\n\n"
        f"{_render_affected_table(report_v2)}\n\n"
        f"### ③ 위험도 및 전 라인 정지 예상 시간\n"
        f"- **위험도**: {risk_lvl}\n"
        f"- **전 라인 정지 예상**: {line_stop_disp} 이내\n\n"
        f"### ④ 추이 데이터\n\n"
        f"{_render_trend_table(report_v2)}\n\n"
        f"### ⑤ Forward Simulation ({horizon}분 후 예측)\n\n"
        f"{_render_forward_sim_table(report_v2)}\n\n"
        "---"
    )


def _render_cause(report_v2: ReportV2, narration: ReportNarration) -> str:
    cause = report_v2.cause
    upstream = (cause.upstream_suspects if cause else []) or []
    upstream_disp = (
        f"과부하 공정: {' → '.join(upstream)} (WIP 과공급으로 병목 TG에 유입)"
        if upstream else "업스트림 과부하 공정 없음"
    )

    # G* Monte Carlo block
    gstar_block = ""
    if cause and cause.g_star:
        g = cause.g_star
        if g.monte_carlo:
            mc = g.monte_carlo
            gstar_block = (
                f"\n### G* 인과 확증 (Monte Carlo)\n"
                f"- **확증 여부**: {'✅ 확정' if g.confirmed else '❌ 미확정'} "
                f"(probability {_md_cell(g.probability)})\n"
                f"- **시뮬 결과**: 총 {mc.n_total}회 중 {mc.n_alarm}회 병목 확인 "
                f"({_md_cell(mc.alarm_ratio_pct)}%)\n"
            )
            if g.upstream_confirmed_toolgroups:
                gstar_block += f"- **업스트림 인과 확증 TG**: {', '.join(g.upstream_confirmed_toolgroups)}\n"

    consensus_block = ""
    if cause and cause.consensus_axes:
        ax = cause.consensus_axes
        consensus_block = (
            f"\n### 4축 합의 ({ax.axes_agreed_count}/4)\n"
            f"- SHAP: {'✅' if ax.shap_supports else '❌'} / "
            f"트렌드: {'✅' if ax.trend_supports else '❌'} / "
            f"업스트림: {'✅' if ax.upstream_supports else '❌'} / "
            f"G*: {'✅' if ax.g_star_supports else '❌'}\n"
        )

    return (
        "## 3. 원인 분석 TOP\n\n"
        f"{_render_top_causes_table(report_v2)}\n\n"
        f"### 판정 근거\n{narration.cause_judgment}\n\n"
        f"### 업스트림 과부하 공정\n{upstream_disp}\n"
        f"{consensus_block}"
        f"{gstar_block}\n"
        f"### 4가지 분석 수렴 증거\n\n"
        f"{_render_evidence_table(report_v2)}\n\n"
        f"### ML 모델 SHAP 분석 (Top features)\n\n"
        f"{_render_shap_table(report_v2)}\n\n"
        f"### 트렌드 악화 속도\n\n"
        f"{_render_trend_stats_table(report_v2)}\n\n"
        "---"
    )


def _render_actions(report_v2: ReportV2, narration: ReportNarration) -> str:
    actions = report_v2.actions
    approval = report_v2.approval

    # 1) 후보가 아예 없는 경우 — PR #1과 동일 결정론적 안내
    if not actions.available:
        is_rejected = approval and approval.status_token and approval.status_token.value == "rejected"
        return _no_actions_section(
            {"rejection_reason": approval.rejection_reason if approval else None},
            bool(is_rejected),
        )

    # 2) 반려된 경우
    is_rejected = approval and approval.status_token and approval.status_token.value == "rejected"

    # 의사결정 상태 뱃지
    decision_badge_map = {
        "winner": "✅ 명확한 1위",
        "equivalent": "⚠️ 통계적 동등 (운영 부담 tie-break 적용)",
        "tentative": "🚨 효과 미검증 (운영 부담 최저 잠정 추천)",
    }
    decision_label = "-"
    if actions.decision_status_token:
        decision_label = decision_badge_map.get(
            actions.decision_status_token.value, actions.decision_status
        )

    # 추천 신뢰도
    rec = actions.recommendation
    confidence_disp = "-"
    if rec and rec.confidence_level:
        confidence_disp = rec.confidence_level

    if is_rejected:
        rej_reason = approval.rejection_reason or "-"
        header = (
            "## 4. 승인된 대응안\n\n"
            f"### ① 의사결정 상태\n- **상태**: {decision_label}\n\n"
            f"### ② 승인된 대응안 예상 효과\n"
            f"승인된 대응안 없음. (반려됨)\n"
            f"- **반려 사유**: {rej_reason}\n\n"
        )
    else:
        # 승인된 대응안 narrative
        approved = next((c for c in actions.candidates if c.is_approved), None)
        kind = approved.kind if approved else "-"
        desc = approved.description if approved else "-"
        primary_reason = rec.primary_reason if rec else "-"
        header = (
            "## 4. 승인된 대응안\n\n"
            f"### ① 의사결정 상태\n"
            f"- **상태**: {decision_label}\n"
            f"- **추천 신뢰도**: {confidence_disp}\n\n"
            f"### ② 승인된 대응안 예상 효과\n"
            f"- **대응안**: {kind} — {desc}\n"
            f"- **핵심 근거**: {primary_reason}\n"
            f"- **예상 효과 요약**: {narration.actions_summary}\n\n"
        )

    # 트레이드오프 / 다른 후보 / 주의사항
    tradeoffs_block = ""
    if rec and rec.tradeoffs:
        tradeoffs_block = "### ③ 받아들이는 트레이드오프\n" + "\n".join(f"- {t}" for t in rec.tradeoffs) + "\n\n"
    else:
        tradeoffs_block = "### ③ 받아들이는 트레이드오프\n- 통계적으로 유의미한 악화 KPI 없음\n\n"

    why_not_block = ""
    if rec and rec.why_not_others:
        why_not_block = "### ④ 다른 후보를 선택하지 않은 이유\n" + "\n".join(
            f"- **{w.label}**: {w.reason}" for w in rec.why_not_others
        ) + "\n\n"
    else:
        why_not_block = "### ④ 다른 후보를 선택하지 않은 이유\n- -\n\n"

    caveats_block = ""
    if rec and rec.caveats:
        caveats_block = "### ⑤ 주의사항\n" + "\n".join(f"- {c}" for c in rec.caveats) + "\n\n"
    else:
        caveats_block = "### ⑤ 주의사항\n- 특이사항 없음\n\n"

    # Playbook (PR #2에서 새로 추가)
    pb = actions.playbook
    playbook_block = ""
    if pb and pb.available:
        playbook_block = "### ⑥ 현장 Playbook\n\n#### 즉시 실행\n"
        for a in pb.immediate_actions:
            playbook_block += f"{a.order}. {a.text}\n"
        if pb.monitoring:
            playbook_block += "\n#### 모니터링\n"
            for m in pb.monitoring:
                target_disp = _md_cell(m.target) + (f" {m.unit}" if m.unit else "")
                playbook_block += f"- T+{m.check_after_min}분: {m.kpi} ≤ {target_disp}\n"
        if pb.rollback_condition:
            playbook_block += f"\n#### 롤백 조건\n- ⚠️ {pb.rollback_condition}\n"
        playbook_block += "\n"

    # 대응안 비교표
    compare_block = (
        "### ⑦ 대응안 비교\n\n"
        f"{_render_candidates_compare_table(report_v2)}\n\n"
    )

    return (
        f"{header}"
        f"{tradeoffs_block}"
        f"{why_not_block}"
        f"{caveats_block}"
        f"{playbook_block}"
        f"{compare_block}"
        "---"
    )


def render_sections(report_v2: ReportV2, narration: ReportNarration) -> dict[str, str]:
    """4개 섹션 마크다운을 한 번에 생성. 표는 코드, narrative는 LLM."""
    return {
        "summary":   _render_summary(report_v2, narration),
        "diffusion": _render_diffusion(report_v2, narration),
        "cause":     _render_cause(report_v2, narration),
        "actions":   _render_actions(report_v2, narration),
    }


# ── PR #1에서 추가된 결정론적 fallback (보존) ────────────────────────────────

def _no_actions_section(approval_info: dict, is_rejected: bool) -> str:
    """업스트림이 대응안 후보를 만들지 못한 경우의 결정론적 섹션."""
    if is_rejected:
        reason = approval_info.get("rejection_reason") or "-"
        return (
            "## 4. 승인된 대응안\n\n"
            "### ① 의사결정 상태\n"
            "- **상태**: 반려됨\n\n"
            "### ② 승인된 대응안\n"
            "승인된 대응안 없음. (반려됨)\n"
            f"- **반려 사유**: {reason}\n\n"
            "---"
        )
    return (
        "## 4. 승인된 대응안\n\n"
        "### ① 의사결정 상태\n"
        "- **상태**: 대응안 후보 없음 — 업스트림(compare_agent)에서 검증된 후보가 생성되지 않음\n\n"
        "### ② 안내\n"
        "이 보고서는 병목 감지·원인·확산 분석까지 수행되었으나, "
        "효과 검증을 통과한 대응안 후보가 도출되지 않아 대응안 섹션은 비워둡니다.\n"
        "- 시뮬레이션 결과 또는 데이터 품질을 확인한 뒤 재실행을 권장합니다.\n\n"
        "---"
    )


# ── 레거시 API 호환 — node.py가 새 함수로 옮기기 전 단계용 ─────────────────────
# PR #3 완료 후엔 node.py가 새 narrate / render_sections 만 호출하므로
# 아래 함수들은 사실상 사용되지 않는다. 안전을 위해 thin wrapper로 유지한다.

def write_summary(state: dict) -> dict:
    """[DEPRECATED] PR #3 이전 호환용. 신규 코드는 narrate/render_sections 사용."""
    rv2 = state.get("report_v2")
    if rv2 is None:
        return {"section_summary": ""}
    sections = render_sections(rv2, narrate(rv2))
    return {"section_summary": sections["summary"]}


def write_diffusion(state: dict) -> dict:
    """[DEPRECATED] 위와 동일."""
    rv2 = state.get("report_v2")
    if rv2 is None:
        return {"section_diffusion": ""}
    sections = render_sections(rv2, narrate(rv2))
    return {"section_diffusion": sections["diffusion"]}


def write_cause(state: dict) -> dict:
    """[DEPRECATED] 위와 동일."""
    rv2 = state.get("report_v2")
    if rv2 is None:
        return {"section_cause": ""}
    sections = render_sections(rv2, narrate(rv2))
    return {"section_cause": sections["cause"]}


def write_actions(state: dict) -> dict:
    """[DEPRECATED] 위와 동일."""
    rv2 = state.get("report_v2")
    if rv2 is None:
        return {"section_actions": ""}
    sections = render_sections(rv2, narrate(rv2))
    return {"section_actions": sections["actions"]}
