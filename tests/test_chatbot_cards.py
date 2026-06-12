"""챗봇 generative UI 카드 빌더·포매터 회귀 — DB 없이 가짜 row(dict)로 카드 JSON 구조 검증.
숫자는 항상 도구 실데이터에서 오고 LLM은 카드 종류만 고르므로, 카드 구조가 깨지지 않는지 스냅샷."""

import datetime as dt

from app.chatbot.ui_cards import (
    _fab_status_lookup,
    _fmt_case_detail,
    _fmt_lot,
    _fmt_tools,
    _fmt_top_tgs,
    _fmt_trend,
    _parse_jsonb,
    _status_card,
    _tool_activity_card,
    _tool_card,
    _top_tg_card,
    _trend_card,
)

LIVE = (
    "[현재 FAB 실시간 현황 · 기준 2020-01-26T12:00:00Z]\n"
    "전체: 가동률 27.8%, WIP 1141 Lot, 설비 가동 600/대기 908/셋업 0/비가동 54, 가용률 95.6%\n"
    "구역별(WIP 많은 순):\n"
    "- Def_Met: WIP 471, 평균가동률 24.1%, 가용률 100.0%\n"
    "- Dry_Etch: WIP 318, 평균가동률 26.6%, 가용률 97.8%"
)


def test_status_card_parses_overall_and_areas():
    card = _status_card(LIVE)
    assert card["type"] == "status"
    assert card["props"]["overall"]["wip"] == 1141
    assert card["props"]["overall"]["down"] == 54
    assert card["props"]["areas"][0] == {"name": "Def_Met", "wip": 471, "util": 24.1, "avail": 100.0}
    assert "2020-01-26T12:00:00Z" in card["props"]["title"]


def test_status_card_none_when_unparseable():
    assert _status_card("형식이 다른 텍스트") is None
    assert _status_card(None) is None


def test_fab_status_lookup_filters_area_lines():
    out = _fab_status_lookup(LIVE, "Def_Met")
    assert "Def_Met" in out and "Dry_Etch" not in out.split("Def_Met")[0]
    assert "찾지 못했습니다" in _fab_status_lookup(LIVE, "NoSuchArea")


def _trend_rows(label, bucket_hours):
    return [
        {"bucket": dt.datetime(2020, 1, 26, h, 0), "label": label, "wip": 10, "util": 0.25, "qtime": 100}
        for h in bucket_hours
    ]


def test_trend_card_single_label_uses_real_name():
    card = _trend_card(_trend_rows("Def_Met", [11, 12]), area="Def", hours=6)
    assert card["props"]["title"] == "Def_Met 추세 (최근 6시간)"
    assert [s["name"] for s in card["props"]["series"]] == ["WIP", "가동률%", "Q-time(분)"]


def test_trend_card_multi_label_falls_back_to_overall():
    rows = _trend_rows("Def_Met", [11]) + _trend_rows("Litho", [11])
    card = _trend_card(rows, area="", hours=6, group_by="tg")
    assert card["props"]["title"] == "전체 TG별 추세 (최근 6시간)"


def test_trend_formatter_includes_tool_calculated_summary():
    rows = [
        {"bucket": dt.datetime(2020, 1, 26, 11, 0), "label": "Def_Met", "wip": 10, "util": 0.2, "qtime": 100},
        {"bucket": dt.datetime(2020, 1, 26, 12, 0), "label": "Def_Met", "wip": 15, "util": 0.4, "qtime": 120},
    ]
    out = _fmt_trend(rows, area="Def_Met")
    assert "계산 요약" in out
    assert "WIP 변화 +5" in out
    assert "WIP 변화율 +50.0%" in out


def test_top_tg_card_builds_lot_table():
    rows = [{
        "area_name": "Def_Met", "tg_code": "DefMet_FE_43", "tg_name": "t",
        "utilization_rate": 0.489, "wip_count": 12, "avg_qtime_min": 423,
        "wait_ratio": 3.2, "available_tool_ratio": 1.0, "bottleneck_prob": 0.1, "risk_grade": "LOW",
    }]
    card = _top_tg_card(rows, area="", metric="util", order="desc", limit=5)
    assert card["type"] == "lot"
    assert card["props"]["rows"][0]["label"] == "Def_Met/DefMet_FE_43"
    assert card["props"]["rows"][0]["util"] == 48.9


def test_lot_and_top_tg_formatters_include_aggregate_summary():
    rows = [
        {
            "area_name": "Def_Met", "tg_code": "DefMet_FE_43", "tg_name": "t",
            "utilization_rate": 0.5, "wip_count": 12, "avg_qtime_min": 40,
            "wait_ratio": 3.2, "available_tool_ratio": 1.0, "risk_grade": "LOW",
        },
        {
            "area_name": "Litho", "tg_code": "Litho_FE_1", "tg_name": "l",
            "utilization_rate": 0.25, "wip_count": 8, "avg_qtime_min": 20,
            "wait_ratio": 1.0, "available_tool_ratio": 0.9, "risk_grade": "MEDIUM",
        },
    ]
    assert "WIP 합계 20 Lot" in _fmt_lot(rows, area="")
    assert "평균 가동률 37.5%" in _fmt_top_tgs(rows, area="", metric="wip", order="desc")


def test_tool_card_uses_custom_columns():
    rows = [{
        "area_name": "Def_Met", "tg_code": "DefMet_FE_43", "tool_code": "DM43-01",
        "utilization_rate": 0.3, "oee_estimate": 55.2, "avg_qtime_min": 10,
        "queue_lot_count": 4, "setup_ratio": 0.0, "down_ratio": 0.1,
    }]
    card = _tool_card(rows, tg="DefMet_FE_43")
    assert card["props"]["labelHeader"] == "툴"
    assert [c["key"] for c in card["props"]["columns"]] == ["util", "oee", "queue", "down"]
    assert card["props"]["rows"][0]["down"] == 10.0


def test_tool_formatter_includes_aggregate_summary():
    rows = [
        {
            "area_name": "Def_Met", "tg_code": "DefMet_FE_43", "tool_code": "DM43-01",
            "utilization_rate": 0.3, "oee_estimate": 55.2, "avg_qtime_min": 10,
            "queue_lot_count": 4, "setup_ratio": 0.0, "down_ratio": 0.1,
        },
        {
            "area_name": "Def_Met", "tg_code": "DefMet_FE_43", "tool_code": "DM43-02",
            "utilization_rate": 0.5, "oee_estimate": 65.2, "avg_qtime_min": 20,
            "queue_lot_count": 6, "setup_ratio": 0.0, "down_ratio": 0.2,
        },
    ]
    out = _fmt_tools(rows, tg="DefMet_FE_43")
    assert "대기 합계 10 Lot" in out
    assert "평균 가동률 40.0%" in out


def test_tool_activity_card_three_series():
    rows = [{"bucket": dt.datetime(2020, 1, 26, 11, 0), "tool_code": "DM43-01",
             "util": 0.31, "queue": 4, "qtime": 12, "down": 0.05}]
    card = _tool_activity_card(rows, hours=6)
    assert card["type"] == "trend"
    assert card["props"]["title"] == "DM43-01 활동 추이 (최근 6시간)"


def test_parse_jsonb_handles_string_and_native():
    assert _parse_jsonb('[{"a":1}]') == [{"a": 1}]
    assert _parse_jsonb([{"a": 1}]) == [{"a": 1}]
    assert _parse_jsonb("not json") is None
    assert _parse_jsonb(None) is None


def test_fmt_case_detail_sorts_shap_and_reads_jsonb_string():
    row = {
        "area_name": "Def_Met", "tg_code": "DefMet_FE_43",
        "detected_at": dt.datetime(2020, 1, 26, 9, 30), "risk_grade": "HIGH",
        "bottleneck_prob": 0.87, "status": "RESOLVED", "resolved_at": dt.datetime(2020, 1, 26, 11, 0),
        "bottleneck_cause_type": "WIP_SURGE",
        "shap_features": '[{"feature":"wait_ratio","value":0.42},{"feature":"setup","value":0.05},{"feature":"wip","value":-0.6}]',
        "diffusion_affected_tg_ids": '["a","b"]',
        "report_summary": "요약." * 5, "root_cause_text": None,
    }
    plans = [{
        "plan_seq": 1, "plan_type": "DISPATCH", "plan_title": "우선순위 부스트", "plan_detail": "d",
        "est_throughput_delta": 1.2, "est_avg_wait_delta": -35, "est_delivery_compliance_delta": 0.8,
        "est_delay_delta": -2, "actual_throughput_delta": None, "actual_avg_wait_delta": None,
        "validated_at": None, "selected": True,
    }]
    hitl = [{"re_decision_seq": 1, "decision": "APPROVED", "comment": "ok", "decided_at": dt.datetime(2020, 1, 26, 10, 0)}]
    out = _fmt_case_detail(row, plans, hitl)
    assert "wip=-0.6" in out and "wait_ratio=0.42" in out
    assert "확산 영향 TG: 2곳" in out
    assert "우선순위 부스트 [선택됨]" in out
    assert "APPROVED" in out
    assert "리포트 요약" in out
