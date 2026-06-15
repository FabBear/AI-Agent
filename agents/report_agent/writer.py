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
narrate_with_reflection(report_v2) → ReportNarration   # Guardrail + Reflection 루프
  └─ narrate(report_v2, feedback) → ReportNarration    # 단일 LLM 호출 (재시도 포함)
render_sections(report_v2, narration) → 4 sections dict
"""

from __future__ import annotations

import os
import re
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


# ── 대응안 레이블 한국어 변환 ──────────────────────────────────────────────────

_ACTION_LABEL_KO: dict[str, str] = {
    "현재상태": "현상유지안",
    "conservative": "보수안",
    "standard": "표준안",
    "aggressive": "강화안",
}


def _label_ko(label: str | None) -> str:
    if not label:
        return label or ""
    return _ACTION_LABEL_KO.get(label, label)


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
            "줄 1: 어느 공정에서 어떤 수준의 문제가 발생했는지 (심각도 포함). "
            "줄 2: 생산에 어떤 영향이 있는지 — 대기 시간·설비 포화 상태를 현장 언어로. "
            "줄 3: 현재 어떤 대응이 결정되었는지 (또는 조치 필요 상태). "
            "시스템 코드·변수명(risk_score, wait_ratio, composite_score 등)은 절대 쓰지 말 것. "
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
            "원인 분석 판정. 현장 엔지니어가 읽는다고 생각하고 공정 현상 언어로 설명. "
            "분석 방법론 이름(SHAP, G*, 4축 합의, Monte Carlo 등)은 절대 쓰지 말고 "
            "그 결론이 의미하는 공정 현상을 표현할 것. "
            "예(금지): 'SHAP 기여율 83%가 설비_포화를 지목함' "
            "예(허용): '설비 가동률이 한계에 근접해 투입량을 소화하지 못하는 상태가 주원인임' "
            "예(금지): '4축 합의 4/4로 신뢰도 HIGH' "
            "예(허용): '여러 방법으로 분석해도 같은 원인이 나와 신뢰도가 높음' "
            "secondary_causes가 있으면 '보조 원인: ...', "
            "dismissed가 있으면 '기각: ... (이유: ...)' 추가. 2~4문장 이내. "
            "문체 '~됨/~함/~임'."
        ),
    )
    actions_summary: str = Field(
        ...,
        description=(
            "승인된 대응안 요약 narrative. 대표 TG의 kpi_impact를 해석하여 "
            "한~두 문장으로 요약할 것. TG 값을 평균 내거나 합산하지 말 것. "
            "반려 상황이면 '반려됨 — 사유: ...' 한 줄. "
            "수치는 데이터에 있는 값을 그대로 사용. 문체 '~됨/~함/~임'."
        ),
    )


class CritiqueResult(BaseModel):
    """LLM Critic이 narrative 검토 후 반환하는 결과."""

    issues: list[str] = Field(default_factory=list)
    needs_revision: bool = Field(default=False)
    score: float = Field(default=1.0, description="0~1, 높을수록 좋음. 0.7 이상이면 needs_revision=False.")


# ── 시스템 프롬프트 (기존 _SYS 보존 + 표 작성 지시 제거) ────────────────────────

_SYS_NARRATE = """[역할]
당신은 반도체 FAB 공정 병목 대응 의사결정 보고서의 narrative(서술 문장)를
작성하는 전문 AI입니다. 표 · KPI 카드 · 수치 비교는 별도 코드가 결정론적으로
생성하므로, 당신은 반드시 문장(narrative)만 작성합니다.

[독자]
- 반도체 FAB 현장 엔지니어, 공정 관리자
- WIP · Q-time · CT · Load Ratio · SuperHotLot · Dispatch Rule 등
  FAB 공정 용어에 익숙하지만, SHAP · G* · Monte Carlo 등 ML/통계 방법론 용어는 모름
- "어떤 공정 현상이 원인인지"를 알고 싶어 하며, "어떤 분석 도구가 뭘 했는지"는 관심 없음
- 병목 발생 원인과 확신 수준을 현장 언어로 설명받기를 원함

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
- 영어와 한국어 혼용 금지(지정된 FAB 용어 제외).
- 분석 시스템 내부 코드 서술 금지: "total score", "composite_score", "SHAP 기여율 X%", "paired_n" 등
  분석 도구 점수·코드는 이미 표에 있으므로 서술에 다시 쓰지 않는다.
  대신 그 수치가 의미하는 공정 현상을 쓴다.
  예(금지): "설비_포화율 total score 0.830과 SHAP 기여율 83.0%가 이를 지지함"
  예(허용): "설비 가동률이 한계에 근접한 상태가 지속되어 투입량을 소화하지 못하는 상황임"."""

_SYS_CRITIQUE = """당신은 반도체 FAB 병목 보고서 narrative의 품질을 검토하는 reviewer입니다.
작성된 narrative가 실제 데이터와 일치하는지, 논리적으로 일관되는지 검토합니다.

[검토 항목]
1. 수치 일치: narrative에 언급된 수치(risk_score, q_time 등)가 실제 데이터와 맞는가
2. 논리 일관성: cause_judgment가 실제 cause 데이터의 primary_category와 일치하는가
3. 행동 가능성: actions.available=False인데 조치를 권고하지 않는가
4. 심각도 톤: 문체의 긴박감이 severity와 맞는가 (Low면 "즉각/긴급" 표현 없어야 함)

[score 기준]
- 1.0: 완벽, 수정 불필요
- 0.7~0.9: 경미한 문제, 수정 선택적
- 0.5~0.7: 수정 필요
- 0.5 미만: 심각한 오류, 반드시 수정

score >= 0.7이면 needs_revision=False로 설정하세요.
issues는 구체적인 문제점 목록입니다. 문제 없으면 빈 리스트."""


# ── 핵심 LLM 호출 ────────────────────────────────────────────────────────────

def narrate(report_v2: ReportV2, feedback: list[str] | None = None, historical: dict | None = None) -> ReportNarration:
    """단일 LLM 호출로 4개 narrative 동시 생성 (structured output).

    feedback이 있으면 이전 작성의 문제점을 프롬프트에 포함해 재생성한다.
    실패 시 결정론적 fallback narration을 반환한다.
    """
    llm = _get_llm().with_structured_output(ReportNarration)
    user_prompt = _build_user_prompt(report_v2, feedback, historical)

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


def _build_user_prompt(report_v2: ReportV2, feedback: list[str] | None = None, historical: dict | None = None) -> str:
    """LLM에게 ReportV2의 핵심 데이터만 보여준다.

    표 데이터 전체를 다 넣지 않는다 — 표는 코드가 만들고, LLM은
    narrative에 필요한 핵심 사실만 받는다. 토큰 절약 + 환각 방지.
    feedback이 있으면 이전 작성의 문제점을 프롬프트 끝에 추가한다.
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
        "approved_label": _label_ko(actions.approved_label),
        "approved_candidate": None,
    }
    if actions.available:
        approved = next((c for c in actions.candidates if c.is_approved), None)
        if approved:
            actions_snip["approved_candidate"] = {
                "label": _label_ko(approved.label),
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
    base = (
        "아래 보고서 데이터를 바탕으로 4개 섹션의 narrative만 작성하세요.\n"
        "표는 만들지 마세요 — 코드가 만듭니다.\n\n"
        f"```json\n{_json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )
    if historical:
        hist_lines = []
        if historical.get("repeat_count"):
            hist_lines.append(f"- 반복 이력: {historical['repeat_count']}")
        if historical.get("past_effectiveness"):
            hist_lines.append(f"- 과거 조치 효과: {historical['past_effectiveness']}")
        if hist_lines:
            base += "\n\n[과거 이력 — DB 조회 결과, narrative에 활용하세요]\n" + "\n".join(hist_lines)
    if feedback:
        feedback_text = "\n".join(f"- {v}" for v in feedback)
        base += f"\n\n[이전 작성의 문제점 — 반드시 수정]\n{feedback_text}"
    return base


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


# ── Guardrail + Reflection ────────────────────────────────────────────────────

def check_guardrails(report_v2: ReportV2, narration: ReportNarration) -> list[str]:
    """코드 규칙 기반 위반 검사. LLM 비용 없음.

    반환값이 빈 리스트면 통과, 아니면 위반 메시지 목록.
    """
    violations: list[str] = []

    # 규칙 1: 대응안 없는데 조치 권고
    if not report_v2.actions.available:
        all_text = " ".join(narration.tldr_3lines) + " " + narration.actions_summary
        action_words = ["권고", "조치를 취", "즉시 실행", "대응 실시", "실행할 것"]
        if any(w in all_text for w in action_words):
            violations.append(
                "actions_unavailable: 대응안 후보가 없는 상황인데 조치를 권고하는 표현이 있음 — "
                "actions_summary와 tldr에서 권고 표현을 제거하고 '대응안 없음' 사실을 반영할 것"
            )

    # 규칙 2: 반려됐는데 긍정적 실행 언급 (단어 경계 주의: "승인" 단독은 FP 위험)
    if (report_v2.approval
            and report_v2.approval.status_token
            and report_v2.approval.status_token.value == "rejected"):
        positive_exec_words = ["승인됨", "승인되었", "실행됨", "실행 예정", "진행됨", "진행 예정"]
        if any(w in narration.actions_summary for w in positive_exec_words):
            violations.append(
                "approval_rejected: 대응안이 반려된 상황인데 actions_summary에서 긍정적 실행 표현이 있음 — "
                "반려 사실과 반려 사유를 반영할 것"
            )

    # 규칙 3: Low 심각도인데 긴박한 표현
    if report_v2.meta.severity == "Low":
        urgent_words = ["즉각", "즉시", "긴급", "심각한 위험", "위기"]
        if any(w in " ".join(narration.tldr_3lines) for w in urgent_words):
            violations.append(
                "severity_mismatch: 심각도가 Low인데 tldr에 긴박한 표현이 사용됨 — "
                "Low 수준에 맞는 적절한 톤으로 조정할 것"
            )

    # 규칙 4: wait_ratio 이상치가 있는데 데이터 품질 언급 없음
    # builder.py _WAIT_RATIO_ANOMALY_THRESHOLD=10.0 초과 시 DataQuality.warnings에 기록됨
    if report_v2.data_quality and report_v2.data_quality.warnings:
        anomaly_codes = {w.code for w in report_v2.data_quality.warnings}
        if "wait_ratio_anomaly" in anomaly_codes:
            quality_words = ["데이터 품질", "이상치", "anomaly", "확인 필요", "wait_ratio 이상"]
            all_text = narration.diffusion_interpretation + " " + " ".join(narration.tldr_3lines)
            if not any(w in all_text for w in quality_words):
                violations.append(
                    "wait_ratio_anomaly: wait_ratio 이상치(>10)가 감지됐으나 narrative에 데이터 품질 주의 언급이 없음 — "
                    "diffusion_interpretation에 'wait_ratio 이상치 감지 — 단위/계산 확인 필요' 문구를 추가할 것"
                )

    return violations


def critique(report_v2: ReportV2, narration: ReportNarration) -> CritiqueResult:
    """LLM이 narrative와 실제 데이터를 비교해 논리 일관성을 검토한다.

    실패 시 needs_revision=False(현재 narration 사용)를 반환한다.
    """
    import json as _json

    llm = _get_llm().with_structured_output(CritiqueResult)

    cause = report_v2.cause
    actions = report_v2.actions
    data_snip = {
        "severity": report_v2.meta.severity,
        "risk_score": report_v2.risk.score,
        "cause_primary": cause.primary.model_dump() if (cause and cause.primary) else None,
        "cause_summary": cause.summary if cause else None,
        "actions_available": actions.available,
        "approved_label": _label_ko(actions.approved_label),
        "approval_status": report_v2.approval.status if report_v2.approval else None,
    }

    user_content = (
        f"[실제 데이터]\n```json\n{_json.dumps(data_snip, ensure_ascii=False, indent=2)}\n```\n\n"
        f"[작성된 narrative]\n"
        f"tldr: {narration.tldr_3lines}\n"
        f"diffusion: {narration.diffusion_interpretation}\n"
        f"cause: {narration.cause_judgment}\n"
        f"actions: {narration.actions_summary}\n\n"
        "위 narrative를 검토하고 CritiqueResult를 반환하세요."
    )

    try:
        result = llm.invoke([
            {"role": "system", "content": _SYS_CRITIQUE},
            {"role": "user", "content": user_content},
        ])
        if not isinstance(result, CritiqueResult):
            result = CritiqueResult.model_validate(result)
        return result
    except Exception:
        return CritiqueResult(issues=[], needs_revision=False, score=1.0)


def narrate_with_reflection(report_v2: ReportV2, max_retries: int = 2, historical: dict | None = None) -> ReportNarration:
    """Guardrail + Reflection 루프로 narrative 품질을 보장한다.

    흐름:
      1. narrate() 로 초안 생성 (historical context 포함)
      2. check_guardrails() 로 코드 규칙 위반 검사 → 위반 있으면 피드백과 함께 재생성
      3. critique() 로 LLM 논리 검토 → 문제 있고 score가 개선될 여지 있으면 재생성
      4. max_retries 초과 또는 개선 없으면 현재 narration 반환
    """
    from agents.logger import get_logger
    _log = get_logger(__name__)

    narration = narrate(report_v2, historical=historical)
    prev_score = 0.0

    for attempt in range(max_retries):
        # Step 1: Guardrail (코드 기반, LLM 비용 없음)
        violations = check_guardrails(report_v2, narration)
        if violations:
            _log.info(f"[Reflection] guardrail 위반 {len(violations)}건 → 재생성 (attempt {attempt + 1})")
            narration = narrate(report_v2, feedback=violations, historical=historical)
            continue

        # Step 2: Critique (LLM 기반)
        try:
            result = critique(report_v2, narration)
        except Exception as e:
            _log.warning(f"[Reflection] critique 실패 — 현재 narration 사용: {e}")
            break

        _log.info(f"[Reflection] score={result.score:.2f}, needs_revision={result.needs_revision} (attempt {attempt + 1})")

        if not result.needs_revision:
            break
        if result.score <= prev_score:
            _log.info("[Reflection] score 개선 없음 — 루프 종료")
            break

        prev_score = result.score
        narration = narrate(report_v2, feedback=result.issues, historical=historical)

    return narration


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
        bn_info_rows.append(("위험 점수", _md_cell(report_v2.risk.score)))

    # KPI 카드 → 표
    kpi_label_map = {
        "q_time_min": "현재 평균 대기시간",
        "wait_ratio": "대기 누적",
        "wip": "WIP",
        "utilization_avg": "가동률",
        "available_tool_ratio": "가용장비비율",
        "max_util": "최대 가동률",
    }
    for card in report_v2.bottleneck_kpis:
        if card.key == "risk_score":
            continue
        label = kpi_label_map.get(card.key, card.label)
        unit_suffix = ""
        if card.key == "q_time_min":
            unit_suffix = "분"
        elif card.key in ("utilization_avg", "max_util", "available_tool_ratio"):
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
        return "| 공정 | 가동률(%) | 대기 누적 | WIP | 상태 |\n|------|-----------|---------|-----|------|\n| - | - | - | - | - |"

    lines = ["| 공정 | 가동률(%) | 대기 누적 | WIP | 상태 |", "|------|-----------|---------|-----|------|"]
    for p in diffusion.high_impact_processes:
        if p.utilization_pct is not None and p.utilization_pct >= 90:
            status = "⚠️ 주의"
        else:
            status = "모니터링"
        if p.data_quality_flags:
            status += " ⚠️"
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
        bn = "병목 지속 예측" if r.is_bottleneck_predicted else "자연 해소 예측"
        lines.append(
            f"| {r.toolgroup} | {_md_cell(r.wait_ratio_future)} | "
            f"{_md_cell(r.wip_future)} | {bn} |"
        )
    return "\n".join(lines)


def _render_top_causes_table(report_v2: ReportV2) -> str:
    cause = report_v2.cause
    if cause is None or not cause.shap_top:
        return "| 순위 | 원인(feature) | 기여도(%) | 현재값 |\n|------|--------------|----------|--------|\n| - | - | - | - |"

    # evidence_matrix에서 confidence 보강
    evid_map = {e.feature: e for e in cause.evidence_matrix}
    lines = ["| 순위 | 원인 지표 | 기여도(%) | 현재값 | 신뢰도 |",
             "|------|---------|----------|--------|--------|"]
    for s in cause.shap_top[:5]:
        e = evid_map.get(s.feature)
        conf = e.confidence if e else "-"
        lines.append(
            f"| {s.rank} | {s.feature} | {_md_cell(s.contribution_pct)} | "
            f"{_md_cell(s.value)} | {conf} |"
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
        return "| 피처 | 시간당 변화율 |\n|------|------------|\n| - | - |"

    lines = ["| 피처 | 시간당 변화율 |", "|------|------------|"]
    for feat, info in cause.trend_series.features.items():
        slope = info.slope_per_hour
        slope_disp = "-" if slope is None else (f"+{slope:.4g}/h" if slope >= 0 else f"{slope:.4g}/h")
        lines.append(f"| {feat} | {slope_disp} |")
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

    lines = ["| 대응안 | 종류 | 설명 | 대기시간 변화 | WIP 변화 |",
             "|--------|------|------|-------------|---------|"]
    for c in actions.candidates:
        label = _label_ko(c.label) + (" ✅" if c.is_approved else "")
        kind = c.kind
        desc = (c.description or "")[:40]
        q_kpi = next((k for k in c.kpi_impact if k.kpi == "q_time_min"), None)
        wip_kpi = next((k for k in c.kpi_impact if k.kpi == "wip"), None)
        q_delta = _md_cell(q_kpi.delta) + "분" if q_kpi and q_kpi.delta is not None else "-"
        wip_delta = _md_cell(wip_kpi.delta) if wip_kpi and wip_kpi.delta is not None else "-"
        lines.append(
            f"| {label} | {kind} | {desc} | {q_delta} | {wip_delta} |"
        )
    return "\n".join(lines)


def _render_approved_tg_forecasts(report_v2: ReportV2) -> str:
    approved = next(
        (candidate for candidate in report_v2.actions.candidates if candidate.is_approved),
        None,
    )
    if approved is None or not approved.per_tg_forecasts:
        return ""

    lines = [
        "#### 대상 TG별 KPI 전망",
        "",
        "| Tool Group | q_time_min | WIP | wait_ratio | utilization_avg | available_tool_ratio |",
        "|------------|------------|-----|------------|-----------------|----------------------|",
    ]
    for target_tg, forecast in approved.per_tg_forecasts.items():
        current = forecast.get("current") or {}
        action = forecast.get("action") or current

        def sequence(kpi: str) -> str:
            return f"{_md_cell(current.get(kpi))} → {_md_cell(action.get(kpi))}"

        lines.append(
            f"| {target_tg} | {sequence('q_time_min')} | {sequence('wip')} | "
            f"{sequence('wait_ratio')} | {sequence('utilization_avg')} | "
            f"{sequence('available_tool_ratio')} |"
        )
    return "\n".join(lines) + "\n\n"


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
        f"### ④ 무대응 시 {horizon}분 후 예측\n\n"
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

    # 분석 합의 블록 (평어)
    consensus_block = ""
    if cause and cause.consensus_axes:
        ax = cause.consensus_axes
        count = ax.axes_agreed_count
        if count == 4:
            consensus_text = "여러 방법으로 분석해도 모두 같은 원인을 지목함"
        elif count >= 3:
            consensus_text = f"여러 방법으로 분석한 결과 {count}가지가 같은 원인을 지목함"
        elif count >= 2:
            consensus_text = f"여러 방법으로 분석한 결과 {count}가지가 같은 원인을 지목함 — 추가 확인 권장"
        else:
            consensus_text = f"분석 방법 간 의견 불일치 — 신뢰도 낮음"
        consensus_block = f"- **분석 합의**: {consensus_text}\n"

    # 별도 통계 검증 블록 (평어, 원시 숫자 제거)
    gstar_block = ""
    if cause and cause.g_star:
        g = cause.g_star
        if g.confirmed:
            tgs = f" (확인 공정: {', '.join(g.upstream_confirmed_toolgroups)})" if g.upstream_confirmed_toolgroups else ""
            gstar_block = f"- **별도 통계 검증**: 동일 원인 확인됨{tgs}\n"
        else:
            gstar_block = "- **별도 통계 검증**: 미확정\n"

    extra_block = ""
    if consensus_block or gstar_block:
        extra_block = "\n" + consensus_block + gstar_block

    return (
        "## 3. 원인 분석 TOP\n\n"
        f"{_render_top_causes_table(report_v2)}\n\n"
        f"### 판정 근거\n{narration.cause_judgment}\n"
        f"{extra_block}\n"
        f"### 업스트림 과부하 공정\n{upstream_disp}\n\n"
        f"### 악화 추세\n\n"
        f"{_render_trend_stats_table(report_v2)}\n\n"
        "---"
    )


def _render_rag_section(rag_evidence: dict) -> str:
    """### ⑤ 사례 기반 근거 — 대응안별 RAG 평가 + 인사이트 + 참조 사례."""
    candidates = rag_evidence.get("candidates") or []
    comparison = rag_evidence.get("comparison") or {}
    common_hits = rag_evidence.get("common_hits") or []

    if not any(c.get("evidence") for c in candidates) and not common_hits:
        return ""

    _effect = {"high": "높음", "medium": "보통", "low": "낮음", "unknown": "-"}
    _risk   = {"high": "높음", "medium": "보통", "low": "낮음", "unknown": "-"}

    lines: list[str] = ["### ⑤ 사례 기반 근거\n"]

    # 대응안별 평가 표
    rows = [(c, c.get("evidence") or {}) for c in candidates if c.get("evidence")]
    if rows:
        lines += ["| 대응안 | 효과 전망 | 리스크 | 비고 |",
                  "|--------|----------|--------|------|"]
        for c, ev in rows:
            label = _ACTION_LABEL_KO.get(c.get("label", ""), c.get("label", "-"))
            effect  = _effect.get(ev.get("effect_outlook", "unknown"), "-")
            risk    = _risk.get(ev.get("risk_level", "unknown"), "-")
            raw = (ev.get("candidate_summary") or "").replace("\n", " ").replace("|", "／")
            summary = raw[:80].rsplit(" ", 1)[0] if len(raw) > 80 else raw
            lines.append(f"| {label} | {effect} | {risk} | {summary} |")
        lines.append("")

    # 종합 인사이트
    insight = comparison.get("rag_summary") or comparison.get("overall_comment") or ""
    if insight:
        lines.append(f"**종합 인사이트**: {insight}\n")

    # 참조 사례 — 같은 TG 그룹이면 헤더 하나로 묶어 표시
    if common_hits:
        def _extract_case_label(title: str) -> str:
            m = re.search(r'\b(conservative|standard|aggressive)\b', title, re.IGNORECASE)
            return m.group(1).lower() if m else ""

        def _outcome_sentence(summary: str) -> str:
            parts = [s.strip().rstrip(".") for s in summary.split(". ") if len(s.strip()) > 10]
            if len(parts) >= 2:
                return f"{parts[-2]} → {parts[-1]}"
            return parts[-1] if parts else summary[:150]

        tg_codes_h = [h.get("tg_code", "") for h in common_hits[:3]]
        all_same_tg = len(set(tg_codes_h)) == 1 and bool(tg_codes_h[0])

        if all_same_tg:
            tg = tg_codes_h[0].replace("+", " + ")
            cause = (common_hits[0].get("bottleneck_cause_type") or "").strip()
            header = f"**참조 사례** — {tg} {cause} 복합 대응" if cause else f"**참조 사례** — {tg} 복합 대응"
            lines.append(header)
            for hit in common_hits[:3]:
                title = hit.get("report_title") or hit.get("case_id") or "-"
                lbl = _extract_case_label(title)
                ko_lbl = _ACTION_LABEL_KO.get(lbl, lbl or title)
                outcome = _outcome_sentence(hit.get("cause_summary") or "")
                lines.append(f"- {ko_lbl}: {outcome}")
        else:
            lines.append("**참조 사례**")
            for hit in common_hits[:3]:
                title = hit.get("report_title") or hit.get("case_id") or "-"
                lbl = _extract_case_label(title)
                display = _ACTION_LABEL_KO.get(lbl, title) if lbl else title
                outcome = _outcome_sentence(hit.get("cause_summary") or "")
                lines.append(f"- {display}: {outcome}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_actions(report_v2: ReportV2, narration: ReportNarration, rag_evidence: dict | None = None) -> str:
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
        "winner": "✅ 명확한 최우선 대응안",
        "equivalent": "⚠️ 효과 동등 — 운영 부담이 가장 낮은 대응안 선택",
        "tentative": "🚨 잠정 선택 — 시뮬레이션 효과 미검증, 운영 부담이 가장 낮은 대응안으로 선택",
    }
    decision_label = "-"
    if actions.decision_status_token:
        decision_label = decision_badge_map.get(
            actions.decision_status_token.value, actions.decision_status
        )

    rec = actions.recommendation

    if is_rejected:
        rej_reason = approval.rejection_reason or "-"
        decision_block = (
            "## 4. 승인된 대응안\n\n"
            f"### ① 결정 사항\n"
            f"- **상태**: {decision_label}\n"
            f"- **결과**: 승인된 대응안 없음 (반려됨)\n"
            f"- **반려 사유**: {rej_reason}\n\n"
        )
        return (
            f"{decision_block}"
            "---"
        )

    # 승인된 대응안
    approved = next((c for c in actions.candidates if c.is_approved), None)
    kind = approved.kind if approved else "-"
    desc = approved.description if approved else "-"
    primary_reason = rec.primary_reason if rec else "-"

    decision_block = (
        "## 4. 승인된 대응안\n\n"
        f"### ① 결정 사항\n"
        f"- **상태**: {decision_label}\n"
        f"- **대응안**: {kind} — {desc}\n"
        f"- **핵심 근거**: {primary_reason}\n"
        f"- **예상 효과**: {narration.actions_summary}\n\n"
    )
    decision_block += _render_approved_tg_forecasts(report_v2)

    # ② 결정 근거 (트레이드오프 + 다른 후보 이유 + 주의사항)
    grounds_lines = ["### ② 결정 근거\n"]

    if rec and rec.tradeoffs:
        grounds_lines.append("**받아들이는 트레이드오프**")
        grounds_lines.extend(f"- {t}" for t in rec.tradeoffs)
    else:
        grounds_lines.append("**받아들이는 트레이드오프**")
        grounds_lines.append("- 통계적으로 유의미한 악화 KPI 없음")

    grounds_lines.append("")

    if rec and rec.why_not_others:
        grounds_lines.append("**다른 후보를 선택하지 않은 이유**")
        grounds_lines.extend(f"- **{_label_ko(w.label)}**: {w.reason}" for w in rec.why_not_others)
    else:
        grounds_lines.append("**다른 후보를 선택하지 않은 이유**")
        grounds_lines.append("- -")

    grounds_lines.append("")

    if rec and rec.caveats:
        grounds_lines.append("**주의사항**")
        grounds_lines.extend(f"- {c}" for c in rec.caveats)
    else:
        grounds_lines.append("**주의사항**")
        grounds_lines.append("- 특이사항 없음")

    grounds_block = "\n".join(grounds_lines) + "\n\n"

    # ③ Playbook
    pb = actions.playbook
    playbook_block = ""
    if pb and pb.available:
        playbook_block = "### ③ 현장 조치 가이드\n\n#### 즉시 실행\n"
        for a in pb.immediate_actions:
            playbook_block += f"- {a.text}\n"
        if pb.monitoring:
            playbook_block += "\n#### 모니터링\n"
            for m in pb.monitoring:
                target_disp = _md_cell(m.target) + (f" {m.unit}" if m.unit else "")
                playbook_block += f"- T+{m.check_after_min}분: {m.kpi} ≤ {target_disp}\n"
        playbook_block += "\n"

    # ④ 대응안 비교
    compare_block = (
        "### ④ 대응안 비교\n\n"
        f"{_render_candidates_compare_table(report_v2)}\n\n"
    )

    # ⑤ 사례 기반 근거 (RAG)
    rag_block = _render_rag_section(rag_evidence) if rag_evidence else ""

    return (
        f"{decision_block}"
        f"{grounds_block}"
        f"{playbook_block}"
        f"{compare_block}"
        f"{rag_block}"
        "---"
    )


def render_sections(report_v2: ReportV2, narration: ReportNarration, rag_evidence: dict | None = None) -> dict[str, str]:
    """4개 섹션 마크다운을 한 번에 생성. 표는 코드, narrative는 LLM."""
    return {
        "summary":   _render_summary(report_v2, narration),
        "diffusion": _render_diffusion(report_v2, narration),
        "cause":     _render_cause(report_v2, narration),
        "actions":   _render_actions(report_v2, narration, rag_evidence=rag_evidence),
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
