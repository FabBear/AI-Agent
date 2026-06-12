"""챗봇 응답 후처리 순수 함수 회귀 — 트레일러 파싱·수치검증·산술루프 제거·forecast 경계·신뢰도·카드 선택.
LLM/DB 없이 결정론적으로 동작하므로 리팩터링 전후 동작 스냅샷으로 쓴다."""

from langchain_core.messages import HumanMessage, ToolMessage

from app.chatbot.postprocess import (
    _confidence,
    _enforce_forecast_boundary,
    _is_future_wip_forecast,
    _parse_trailers,
    _pick_ui,
    _sanitize_answer,
    _validate_numbers,
)


def test_parse_trailers_splits_body_and_trailers_order_independent():
    text = (
        "본문 결론입니다.\n"
        "<<<UI>>>\nlot\n"
        "<<<SPOKEN>>>\n지금 WIP가 높습니다.\n"
        "<<<FOLLOWUPS>>>\n1. 다음 질문 하나\n2. 다음 질문 둘\n3. 다음 질문 셋\n"
    )
    body, spoken, followups, ui_type = _parse_trailers(text)
    assert body == "본문 결론입니다."
    assert spoken == "지금 WIP가 높습니다."
    assert followups == ["다음 질문 하나", "다음 질문 둘", "다음 질문 셋"]
    assert ui_type == "lot"


def test_parse_trailers_without_markers_returns_plain_body():
    body, spoken, followups, ui_type = _parse_trailers("그냥 답변")
    assert body == "그냥 답변"
    assert spoken == "" and followups == [] and ui_type == ""


def test_parse_trailers_ignores_unknown_ui_type():
    _, _, _, ui_type = _parse_trailers("본문\n<<<UI>>>\nbogus")
    assert ui_type == ""


def test_number_validation_allows_format_variants_and_flags_unknowns():
    messages = [
        HumanMessage(content="최근 6시간 WIP 어때?"),
        ToolMessage(content="WIP 2291 Lot, 가동률 0.91", tool_call_id="t1"),
    ]
    assert _validate_numbers("최근 6시간 기준 WIP는 2,291 Lot이고 가동률은 91%입니다.", messages) == []
    assert _validate_numbers("현재 WIP는 2,500 Lot입니다.", messages) == ["2,500"]


def test_number_validation_skips_plain_numbers_when_no_tool_messages():
    assert _validate_numbers("테스트 숫자 9999입니다.", [HumanMessage(content="안녕")]) == []


def test_number_validation_flags_data_numbers_when_no_tool_messages():
    assert _validate_numbers("WIP 합계는 9999 Lot입니다.", [HumanMessage(content="WIP 합계 알려줘")]) == ["9999"]


def test_sanitizer_removes_repeated_arithmetic_loop():
    answer = (
        "정상 결론 문장입니다.\n\n"
        + " = ".join(["1+29+34+55+88"] * 6)
    )
    sanitized, warnings = _sanitize_answer(answer)
    assert "정상 결론 문장입니다." in sanitized
    assert "1+29+34+55+88 = 1+29+34" not in sanitized
    assert warnings


def test_forecast_boundary_prepends_unsupported_notice():
    out = _enforce_forecast_boundary("내일 Litho WIP 예측해줘", "최근 6시간 추세는 보합입니다.")
    assert "예측은 현재 지원하지 않습니다" in out
    assert "최근 6시간 추세는 보합입니다." in out


def test_forecast_boundary_noop_for_non_forecast():
    assert _enforce_forecast_boundary("지금 WIP 어때?", "현재 WIP는 32입니다.") == "현재 WIP는 32입니다."
    assert _is_future_wip_forecast("내일 wip 예측") is True
    assert _is_future_wip_forecast("지금 가동률") is False


def test_confidence_levels():
    assert _confidence([], [], ["423"])[0] == "LOW"
    assert _confidence(["get_fab_status"], [], [])[0] == "HIGH"
    assert _confidence([], [{"title": "x"}], [])[0] == "HIGH"
    assert _confidence([], [], [])[0] == "MEDIUM"


def test_pick_ui_prefers_llm_choice_then_lot_for_lot_tools_then_last():
    cards = {"status": {"type": "status"}, "lot": {"type": "lot"}}
    assert _pick_ui(cards, "status", ["get_fab_status"])["type"] == "status"
    assert _pick_ui(cards, "", ["get_top_toolgroups"])["type"] == "lot"
    only = {"trend": {"type": "trend"}}
    assert _pick_ui(only, "", ["get_kpi_trend"])["type"] == "trend"
    assert _pick_ui({}, "", []) is None
