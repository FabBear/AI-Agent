"""챗봇 응답 후처리 — 트레일러 파싱, 수치 환각 검증, 산술루프 제거, forecast 경계,
신뢰도 산정, 카드 선택. 전부 결정론적(LLM/DB 없음)."""

import logging
import re

from langchain_core.messages import HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

FOLLOWUPS_MARKER = "<<<FOLLOWUPS>>>"
SPOKEN_MARKER = "<<<SPOKEN>>>"
UI_MARKER = "<<<UI>>>"
UI_TYPES = {"status", "trend", "lot", "cases"}
DATA_NUMERIC_TERMS = (
    "wip", "윕", "재공", "lot", "가동률", "가용률", "q-time", "qtime", "대기", "설비", "장비", "툴",
    "tg", "공정", "구역", "oee", "병목", "확률", "임계값", "threshold",
)
CALCULATION_TERMS = (
    "합계", "총합", "전체 합", "평균", "차이", "증감", "증가", "감소", "증가율", "감소율", "변화율",
    "비율", "퍼센트", "percent", "%", "%p", "몇 배", "더하면", "계산",
)


def _parse_trailers(text: str) -> tuple[str, str, list[str], str]:
    """본문 + 트레일러(SPOKEN/FOLLOWUPS/UI)를 분리. 마커 출현 순서와 무관하게 파싱.
    반환: (body, spoken, followups, ui_type)."""
    markers = [("spoken", SPOKEN_MARKER), ("followups", FOLLOWUPS_MARKER), ("ui", UI_MARKER)]
    positions = sorted((idx, key, m) for key, m in markers if (idx := text.find(m)) != -1)
    if not positions:
        return text.strip(), "", [], ""
    body = text[: positions[0][0]].strip()
    sections: dict[str, str] = {}
    for i, (idx, key, marker) in enumerate(positions):
        start = idx + len(marker)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[key] = text[start:end].strip()
    spoken = " ".join(sections.get("spoken", "").split())[:200]
    followups = []
    for line in sections.get("followups", "").splitlines():
        cleaned = re.sub(r"^[\s\-\*\d\.\)·•]+", "", line).strip().strip("\"'")
        if cleaned:
            followups.append(cleaned[:60])
    ui_raw = (sections.get("ui", "").strip().split() or [""])[0].lower()
    ui_type = ui_raw if ui_raw in UI_TYPES else ""
    return body, spoken, followups[:3], ui_type


def _number_variants(nums: list[str]) -> set[str]:
    """Equivalent numeric renderings allowed by answer formatting.

    Examples:
    - tool "2,291" and answer "2291"
    - tool "100.0" and answer "100"
    - tool ratio "0.91" and answer "91%"
    """
    out: set[str] = set()
    for raw in nums:
        compact = raw.replace(",", "")
        out.add(raw)
        out.add(compact)
        try:
            value = float(compact)
        except (ValueError, OverflowError):
            continue
        out.add(f"{value:.1f}")
        out.add(f"{value:.2f}")
        if value.is_integer():
            out.add(str(int(value)))
        trimmed = f"{value:.4f}".rstrip("0").rstrip(".")
        if trimmed:
            out.add(trimmed)
        if 0 <= value <= 1:
            pct = value * 100
            out.add(f"{pct:.1f}")
            out.add(f"{pct:.2f}")
            if pct.is_integer():
                out.add(str(int(pct)))
            pct_trimmed = f"{pct:.4f}".rstrip("0").rstrip(".")
            if pct_trimmed:
                out.add(pct_trimmed)
    return out


def _validate_numbers(answer: str, messages: list) -> list[str]:
    """수치 환각 감시(재생성 없음, LLM 비용 0).
    답변 속 숫자가 도구 결과(ToolMessage)에 존재하는지 점검 — 불일치 의심 숫자 목록을 반환하고 경고 로그도 남긴다."""
    tool_text = " ".join(str(m.content) for m in messages if isinstance(m, ToolMessage))
    if not tool_text:
        compact = re.sub(r"\s+", "", answer.lower())
        has_data_term = any(term in compact for term in DATA_NUMERIC_TERMS)
        has_calc_term = any(term in compact for term in CALCULATION_TERMS)
        if not (has_data_term or has_calc_term):
            return []
        suspects = [n for n in re.findall(r"\d+(?:[.,]\d+)?", answer) if len(n.replace(",", "").replace(".", "")) > 1]
        suspects = list(dict.fromkeys(suspects))
        if suspects:
            logger.warning("chat 수치 검증: 도구 없이 데이터/계산 숫자 %s (환각 의심)", suspects[:8])
        return suspects
    question_text = " ".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    allowed_nums = _number_variants(re.findall(r"\d+(?:[.,]\d+)?", f"{tool_text} {question_text}"))
    suspects = []
    for n in re.findall(r"\d+(?:[.,]\d+)?", answer):
        if len(n.replace(",", "").replace(".", "")) <= 1:
            continue
        if _number_variants([n]).isdisjoint(allowed_nums):
            suspects.append(n)
    suspects = list(dict.fromkeys(suspects))
    if suspects:
        logger.warning("chat 수치 검증: 도구 결과에 없는 숫자 %s (환각 의심)", suspects[:8])
    return suspects


def _has_repetition_loop(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 80:
        return False
    for size in range(16, min(90, len(compact) // 3)):
        for start in range(0, max(1, min(len(compact) - size * 3, 240))):
            chunk = compact[start:start + size]
            if chunk and compact.count(chunk) >= 4:
                return True
    return len(re.findall(r"\d+\s*\+", text or "")) >= 8


def _sanitize_answer(answer: str) -> tuple[str, list[str]]:
    """반복 계산식/루프 문장을 제거해 응답 품질 저하를 막는다."""
    warnings: list[str] = []
    kept: list[str] = []
    for block in re.split(r"\n{2,}", answer or ""):
        if _has_repetition_loop(block):
            warnings.append("반복 계산식이 감지되어 해당 문단을 제거했습니다.")
            continue
        kept.append(block)
    sanitized = "\n\n".join(part.strip() for part in kept if part.strip()).strip()
    if not sanitized and answer.strip():
        sanitized = (
            "응답 생성 중 반복 계산식이 감지되어 원문을 표시하지 않았습니다. "
            "현재 WIP나 최근 추세 기준으로 다시 조회해 주세요."
        )
    return sanitized, warnings


def _is_future_wip_forecast(message: str) -> bool:
    compact = re.sub(r"[\s_#-]+", "", (message or "").lower())
    return (
        any(term in compact for term in ("내일", "다음날", "익일", "미래", "예측", "forecast"))
        and any(term in compact for term in ("wip", "윕", "재공", "물량"))
    )


def _enforce_forecast_boundary(message: str, answer: str) -> str:
    if not _is_future_wip_forecast(message):
        return answer
    if "예측" in answer and ("지원하지" in answer or "지원되지" in answer or "현재 지원" in answer):
        return answer
    return "내일/미래 WIP 수치 예측은 현재 지원하지 않습니다. " + answer.lstrip()


def _confidence(tools_used: list[str], sources: list[dict], suspects: list[str]) -> tuple[str, list[str]]:
    """답변 신뢰도 휴리스틱(결정론적): 수치 불일치 의심 → LOW,
    도구 실데이터 또는 RAG 출처 기반 → HIGH, 둘 다 없으면(모델 자체 지식) → MEDIUM."""
    if suspects:
        return "LOW", [f"답변 수치 일부({', '.join(suspects[:5])})가 조회 데이터와 일치하지 않을 수 있습니다."]
    if tools_used or sources:
        return "HIGH", []
    return "MEDIUM", []


def _pick_ui(ui_cards: dict, ui_type: str, tools_used: list[str] | None = None) -> dict | None:
    """하이브리드: LLM이 고른 카드 우선. 마커 누락 시 **마지막에 만들어진 카드**로 폴백
    (insertion order = LLM이 실제 마지막에 호출한 도구 → 질문과 가장 관련 높음)."""
    tools_used = tools_used or []
    lot_tools = {"get_lot_status", "get_top_toolgroups", "get_tool_status"}
    if "lot" in ui_cards and lot_tools.intersection(tools_used) and ui_type in ("", "status", "lot"):
        return ui_cards["lot"]
    ui = ui_cards.get(ui_type)
    if ui is None and ui_cards:
        return next(reversed(ui_cards.values()))
    return ui
