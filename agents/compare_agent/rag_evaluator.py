"""RAG 평가 유틸리티 — compare_rag 노드에서 사용.

build_action_profile      : 후보 plan_meta → 구성요소 프로필 dict
build_plan_description    : 후보 plan_meta → 단문 설명 (프롬프트·쿼리용)
build_plan_query          : 원인 + 후보 설명 → Qdrant 검색 쿼리
CandidateEvidence         : 후보별 과거 사례 평가 결과
RagComparison             : 후보 간 RAG 상대 비교 + 종합 의견
evaluate_candidate_evidence: 검색 사례를 후보 관점에서 구조화
compare_candidate_evidence : 후보별 근거를 한꺼번에 비교
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 쿼리 빌더
# ──────────────────────────────────────────────────────────────────────────────

def build_action_profile(candidate: dict) -> dict:
    """후보 plan_meta → 구성요소 프로필.

    verification_agent가 plan_meta.components에 분해해 넣은 값을 읽는다.
    components가 없으면 빈 프로필(인터벌만 있는 것으로 간주)을 반환.
    """
    pm = candidate.get("plan_meta") or {}
    comp = pm.get("components") or {}
    return {
        "interval_pct": pm.get("release_interval_delta_pct"),
        "has_priority": comp.get("has_priority_bump", False),
        "has_superhotlot": comp.get("has_superhotlot", False),
        "priority_products": comp.get("priority_products", []),
        "superhotlot_products": comp.get("superhotlot_products", []),
    }


def build_plan_description(candidate: dict) -> str:
    """후보 대응안 → 단문 설명.

    구성요소 프로필(인터벌·우선순위·SuperHotLot)을 한 줄로 반환.
    """
    p = build_action_profile(candidate)
    parts: list[str] = []

    if p["interval_pct"] is not None:
        parts.append(f"인터벌 +{p['interval_pct']}%")

    if p["has_priority"]:
        parts.append(f"우선순위 상향 {p['priority_products']}")
    else:
        parts.append("우선순위 조정 없음")

    if p["has_superhotlot"]:
        parts.append(f"SuperHotLot {p['superhotlot_products']}")
    else:
        parts.append("SuperHotLot 없음")

    if parts:
        return " / ".join(parts)

    # fallback
    desc = candidate.get("description", "")
    return desc[:80] if desc else candidate.get("action_kind", "대응안")


def build_plan_query(ci: dict, candidate: dict) -> str:
    """원인 + 현재 상태 + 대응안 → Qdrant 검색 쿼리 문자열."""
    bn = ci.get("bottleneck_info") or {}
    cause = (ci.get("cause_context") or {}).get("cause_summary", "")
    plan = build_plan_description(candidate)

    wip = bn.get("wip_count", bn.get("wip", 0))
    wait = bn.get("avg_queue_time_min", bn.get("q_time_min", 0))
    util = bn.get("utilization_avg", 0)

    return (
        f"툴그룹: {ci.get('toolgroup', '')}\n"
        f"원인: {cause}\n"
        f"현재 상태: WIP {wip:.0f}개, 대기 {wait:.0f}분, 이용률 {util:.2f}\n"
        f"적용 대상: {plan}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 루브릭 모델
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceCaseSummary(BaseModel):
    """화면에 펼쳐 보여줄 과거 사례 한 건."""

    case_id: str = Field(description="입력에 제공된 사례 ID를 그대로 사용")
    summary: str = Field(
        description="현재 상황, 적용 조치, 실제 결과를 포함한 한국어 한 문장"
    )
    relevance: Literal["direct", "partial", "weak"] = Field(
        description="현재 후보 조치 구성과 사례 조치 구성의 유사성"
    )
    supports_effect: bool = Field(description="실측 개선 효과를 지지하는 사례인지")
    shows_risk: bool = Field(description="부작용이나 실행 리스크를 보여주는 사례인지")


class EvidenceClaim(BaseModel):
    """후보 판단에 사용된 주장과 출처 사례."""

    text: str = Field(description="사례에 명시된 사실만 사용한 한국어 한 문장")
    case_ids: list[str] = Field(
        default_factory=list,
        description="주장을 뒷받침하는 입력 사례 ID 목록",
    )


class CandidateEvidence(BaseModel):
    """후보 하나에 대한 과거 사례 기반 판단."""

    effect_outlook: Literal["high", "medium", "low", "unknown"] = Field(
        description="과거 사례 기준 개선 효과 기대 수준"
    )
    risk_level: Literal["low", "medium", "high", "unknown"] = Field(
        description="과거 사례 기준 운영 부작용과 실행 리스크 수준"
    )
    evidence_strength: Literal["strong", "moderate", "weak", "insufficient"] = Field(
        description="직접 유사 사례 수, 실측 결과, 사례 간 일관성을 고려한 근거 강도"
    )
    candidate_summary: str = Field(
        description="효과, 속도, 리스크를 종합한 후보별 판단 한 문장"
    )
    case_summaries: list[EvidenceCaseSummary] = Field(
        default_factory=list,
        description="중복을 제거한 대표 근거 사례. 최대 6건",
    )
    claims: list[EvidenceClaim] = Field(
        default_factory=list,
        description="후보 판단의 핵심 주장과 근거 사례 ID. 최대 3건",
    )


class RagRankingItem(BaseModel):
    """과거 사례 기준 후보 간 상대 순위."""

    rank: int = Field(description="1부터 시작하는 순위. 동점이면 같은 순위")
    label: str = Field(description="입력 후보 라벨을 그대로 사용")
    recommendation_level: Literal["우선 검토", "조건부 검토", "후순위", "판단 보류"]
    summary: str = Field(description="이 후보의 상대적 위치를 설명하는 한 문장")
    why_better: str = Field(
        default="",
        description="다른 후보보다 앞선 경우 효과와 리스크 관점의 비교 이유",
    )
    case_ids: list[str] = Field(
        default_factory=list,
        description="이 순위 판단에 직접 사용한 사례 ID",
    )


class RagComparison(BaseModel):
    """RAG 후보 비교 결과와 통계 결과를 함께 읽은 종합 의견."""

    ranking_status: Literal["ranked", "tied", "insufficient"] = Field(
        description="근거로 순위를 정할 수 있는지 여부"
    )
    ranking: list[RagRankingItem] = Field(default_factory=list)
    rag_summary: str = Field(
        description="관리자에게 보여줄 과거 사례 기반 인사이트 한 문장"
    )
    overall_comment: str = Field(
        description=(
            "기존 통계 검증 결과를 먼저 설명하고, RAG 판단과 승인 시 주의점을 이어서 "
            "작성한 2~3문장 종합 의견"
        )
    )


def _candidate_profile(item: dict) -> dict:
    return item.get("profile") or {}


def _evidence_text(evidence: dict | None) -> str:
    if not evidence:
        return ""
    parts = [
        evidence.get("candidate_summary", ""),
        " ".join(case.get("summary", "") for case in evidence.get("case_summaries", []) or []),
        " ".join(claim.get("text", "") for claim in evidence.get("claims", []) or []),
    ]
    return " ".join(part for part in parts if part)


def _normalize_candidate_evidence(item: dict, evidence: dict) -> dict:
    """후보 구성에 맞춰 RAG label을 안정적으로 보정한다.

    LLM 결과가 너무 평평하게 나오는 경우가 있어, 직접 구성 기준으로
    conservative / standard / aggressive 차이가 드러나도록 한 번 더 정규화한다.
    """
    profile = _candidate_profile(item)
    interval_pct = float(profile.get("interval_pct") or 0.0)
    has_priority = bool(profile.get("has_priority"))
    has_superhotlot = bool(profile.get("has_superhotlot"))

    direct_cases = [
        case for case in (evidence.get("case_summaries") or [])
        if case.get("relevance") == "direct"
    ]
    direct_count = len(direct_cases)
    text = _evidence_text(evidence)

    effect = evidence.get("effect_outlook", "unknown")
    risk = evidence.get("risk_level", "unknown")
    strength = evidence.get("evidence_strength", "insufficient")

    # effect_outlook: 실제 조합 기준으로 구분
    if has_superhotlot:
        if interval_pct >= 28 or direct_count >= 2:
            effect = "high"
        elif interval_pct >= 20:
            effect = "medium"
        else:
            effect = "low"
    elif has_priority and interval_pct >= 20:
        effect = "high"
    elif interval_pct <= 16 or "제한적" in text or "완전히 해소" in text:
        effect = "low"
    elif direct_count >= 2:
        effect = "medium"
    else:
        effect = "medium"

    # risk_level: SHL 포함 여부를 가장 강한 신호로 본다.
    if has_superhotlot:
        risk = "high"
    elif has_priority and interval_pct >= 20:
        risk = "low"
    elif interval_pct <= 16 and not has_priority:
        risk = "low"
    elif "가용성" in text or "집중" in text or "부작용" in text:
        risk = "medium"
    else:
        risk = "medium"

    # evidence_strength: 직접 사례 반복성과 구성 일치성으로 보정
    if direct_count >= 2 and not has_superhotlot and has_priority:
        strength = "strong"
    elif direct_count >= 2:
        strength = "moderate"
    elif direct_count == 1:
        strength = "moderate"
    else:
        strength = "weak"

    summary = evidence.get("candidate_summary", "")
    if effect == "low" and "효과" not in summary:
        summary = summary or "과거 직접 사례 기준 효과가 제한적입니다."
    if risk == "high" and "리스크" not in summary and "위험" not in summary:
        if summary:
            summary = summary.rstrip("。.")
            summary += " / 운영 리스크가 높습니다."
        else:
            summary = "운영 리스크가 높습니다."

    return {
        **evidence,
        "effect_outlook": effect,
        "risk_level": risk,
        "evidence_strength": strength,
        "candidate_summary": summary,
    }


def _normalize_rag_comparison(items: list[dict], comparison: RagComparison) -> RagComparison:
    """후보별 정규화 결과를 바탕으로 비교 요약도 살짝 강화한다."""
    adjusted_items = []
    for item in items:
        evidence = item.get("evidence")
        if not evidence:
            adjusted_items.append(item)
            continue
        adjusted_items.append({**item, "evidence": _normalize_candidate_evidence(item, evidence)})

    ranking_bits = []
    for item in adjusted_items:
        evidence = item.get("evidence") or {}
        ranking_bits.append(
            f"{item.get('label', '?')}: 효과 {evidence.get('effect_outlook', 'unknown')} · "
            f"리스크 {evidence.get('risk_level', 'unknown')}"
        )

    rag_summary = comparison.rag_summary
    if ranking_bits:
        rag_summary = " / ".join(ranking_bits[:3]) + (
            ". " + rag_summary if rag_summary else ""
        )

    overall_comment = comparison.overall_comment
    if adjusted_items and not overall_comment:
        overall_comment = "RAG 비교는 후보별 과거 사례 차이를 참고용으로만 제시합니다."

    return comparison.model_copy(
        update={
            "rag_summary": rag_summary,
            "overall_comment": overall_comment,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# 루브릭 평가
# ──────────────────────────────────────────────────────────────────────────────

_EVIDENCE_SYSTEM_PROMPT = """\
당신은 반도체 FAB 병목 대응 전문가입니다.
과거 유사 사례에서 현재 대응안의 효과와 리스크를 뒷받침하는 근거를 추출합니다.
평가는 관리자의 의사결정을 보조하며 시뮬레이션 점수나 기존 추천을 변경하지 않습니다.
사례에 명시된 내용만 사용하고, 근거가 부족하면 unknown 또는 insufficient로 판단하십시오.
후보 간 차이를 억지로 만들거나 입력에 없는 수치와 결과를 생성하지 마십시오.
"""


def _format_cases(hits: list[dict]) -> str:
    blocks: list[str] = []
    for hit in hits[:8]:
        case_id = str(hit.get("case_id", ""))
        title = hit.get("report_title") or case_id
        text = (hit.get("text") or "")[:3000]
        blocks.append(f"[{case_id}]\n제목: {title}\n원문:\n{text}")
    return "\n\n---\n\n".join(blocks)


def evaluate_candidate_evidence(
    plan_description: str,
    current_state_summary: str,
    hits: list[dict],
    llm: "ChatOpenAI",
    action_profile: dict | None = None,
) -> Optional[CandidateEvidence]:
    """과거 사례를 현재 후보 관점에서 구조화한다."""
    if not hits:
        return None

    profile = action_profile or {}
    interval_pct = profile.get("interval_pct", "?")
    priority_products = profile.get("priority_products") or []
    superhotlot_products = profile.get("superhotlot_products") or []
    component_block = (
        f"  인터벌: +{interval_pct}%\n"
        f"  우선순위 상향 대상: {priority_products if priority_products else '없음'}\n"
        f"  SuperHotLot 대상: {superhotlot_products if superhotlot_products else '없음'}"
    )

    user_prompt = f"""\
## 현재 상황
{current_state_summary}

## 검토 중인 대응안
{plan_description}

[현재 후보 구성 — 추측 금지, 이 값을 그대로 사용]
{component_block}

## 과거 유사 사례 ({min(len(hits), 8)}건)
{_format_cases(hits)}

---
위 사례를 현재 후보 관점에서 평가하십시오.

[작성 규칙]
1. case_summaries는 각 사례를 "상황 → 적용 조치 → 실제 결과" 순서의 한 문장으로 작성합니다.
2. case_id는 입력에 표시된 ID만 그대로 사용합니다.
3. 현재 후보와 인터벌 강도, 우선순위, SuperHotLot 구성이 가까울수록 relevance를 높입니다.
4. 현재 후보와 조치 구성이 다른 사례는 직접 근거처럼 표현하지 않습니다.
5. 현재 후보에 SuperHotLot이 없으면 SuperHotLot 때문에 발생한 리스크를 현재 후보의 리스크로 전이하지 않습니다.
6. 우선순위 대상이나 강도가 다른 사례의 부작용은 해당 차이를 명시한 간접 근거로만 사용합니다.
7. 실측 결과가 없거나 사례 간 결과가 충돌하면 evidence_strength를 낮춥니다.
8. candidate_summary는 효과, 개선 속도, 리스크를 포함한 한 문장으로 작성합니다.
9. claims의 모든 문장에는 이를 뒷받침하는 case_ids가 하나 이상 있어야 합니다.
10. risk_level은 현재 후보와 우선순위 사용 여부 및 SuperHotLot 사용 여부가 같은 사례만으로 판정합니다.
11. 동일 구성의 직접 사례들이 모두 부작용 없음이면 risk_level=low로 판정합니다. 다른 구성의 리스크 때문에 medium/high로 올리지 않습니다.
12. 동일 구성의 직접 사례가 2건 이상이고 효과가 일관되게 크면 effect_outlook=high, 작거나 악화 완화에 그치면 medium/low로 구분합니다.
"""

    try:
        structured_llm = llm.with_structured_output(
            CandidateEvidence,
            method="function_calling",
        )
        result = structured_llm.invoke(
            [
                {"role": "system", "content": _EVIDENCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        adjusted = _normalize_candidate_evidence(
            {
                "profile": profile,
                "plan_description": plan_description,
                "hits": hits,
            },
            result.model_dump(),
        )
        return CandidateEvidence.model_validate(adjusted)
    except Exception:
        logger.exception("evaluate_candidate_evidence LLM 호출 실패")
        return None


_COMPARISON_SYSTEM_PROMPT = """\
당신은 반도체 FAB 병목 대응 의사결정 보조자입니다.
후보별 과거 사례 근거를 비교해 RAG 관점의 상대 순위를 제시합니다.
RAG 순위는 시뮬레이션 점수와 합산하지 않으며 기존 통계 추천을 변경하지 않습니다.
통계 검증 결과는 overall_comment 작성에만 사용하고 RAG 순위 산정에는 사용하지 마십시오.
근거 차이가 작거나 부족하면 동점 또는 판단 보류를 선택하십시오.
"""


def compare_candidate_evidence(
    candidate_evidence: list[dict],
    statistical_summary: str,
    llm: "ChatOpenAI",
    current_context: str = "",
) -> Optional[RagComparison]:
    """후보별 근거를 상대 비교하고 통계+RAG 종합 의견을 생성한다."""
    usable = [item for item in candidate_evidence if item.get("evidence")]
    if not usable:
        return None

    candidate_blocks: list[str] = []
    for item in usable:
        evidence = item["evidence"]
        candidate_blocks.append(
            f"[{item.get('label', '?')}]\n"
            f"대응안: {item.get('plan_description', '')}\n"
            f"효과 기대: {evidence.get('effect_outlook', 'unknown')}\n"
            f"리스크: {evidence.get('risk_level', 'unknown')}\n"
            f"근거 강도: {evidence.get('evidence_strength', 'insufficient')}\n"
            f"후보 판단: {evidence.get('candidate_summary', '')}\n"
            f"핵심 주장: {evidence.get('claims', [])}\n"
            f"대표 사례: {evidence.get('case_summaries', [])}"
        )

    user_prompt = f"""\
## 현재 병목 상황
{current_context or '추가 상황 정보 없음'}

## 후보별 과거 사례 평가
{chr(10).join(candidate_blocks)}

## 기존 통계 검증 결과
{statistical_summary}

---
[MVP 비교 규칙]
1. ranking은 빈 목록으로 반환하고 후보 순위를 만들지 않습니다.
2. 현재 대응안과 조치 구성이 직접 유사한 사례를 중심으로 효과와 리스크만 비교합니다.
3. SuperHotLot 근거는 현재 상황에 납기 위험이나 긴급 lot가 명시된 경우에만 직접 근거로 봅니다.
4. rag_summary는 후보별 차이와 관리자가 확인할 조건을 담은 한 문장으로 작성합니다.
5. 근거에 없는 차이와 확정적 추천을 생성하지 않습니다.

[종합 의견 규칙]
1. overall_comment는 통계 결과와 RAG가 별개임을 나타내는 한 문장만 작성합니다.
2. RAG가 통계적 우위를 입증했다거나 기존 추천을 변경했다고 표현하지 않습니다.
"""

    try:
        structured_llm = llm.with_structured_output(
            RagComparison,
            method="function_calling",
        )
        comparison = structured_llm.invoke(
            [
                {"role": "system", "content": _COMPARISON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_rag_comparison(candidate_evidence, comparison)
    except Exception:
        logger.exception("compare_candidate_evidence LLM 호출 실패")
        return None
