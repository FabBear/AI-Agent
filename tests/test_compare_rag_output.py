from types import SimpleNamespace

from agents.compare_agent.node import (
    _build_action_candidates,
    _build_hitl_prompt,
    _build_rag_block,
    _build_rag_statistical_summary,
    _sanitize_candidate_evidence,
    _sanitize_rag_comparison,
    _rank_candidates,
)
from agents.compare_agent.rag_evaluator import (
    CandidateEvidence,
    RagComparison,
    compare_candidate_evidence,
    evaluate_candidate_evidence,
)
from agents.display import print_verification_results
from agents.report_agent.writer import _render_candidates_compare_table
from demo.verification_data import build_demo_verification_results


class _FakeStructuredLlm:
    def __init__(self, result):
        self.result = result
        self.messages = None
        self.schema = None

    def with_structured_output(self, schema, method):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.result


def _rag_evidence() -> dict:
    return {
        "candidates": [
            {
                "label": "standard",
                "profile": {
                    "interval_pct": 22,
                    "priority_products": ["Product_3"],
                    "superhotlot_products": [],
                },
                "hits": [
                    {
                        "case_id": "CASE-004",
                        "report_title": "설비 포화 대응 보고서",
                        "report_url": "/reports/CASE-004",
                    }
                ],
                "evidence": {
                    "effect_outlook": "high",
                    "risk_level": "low",
                    "evidence_strength": "strong",
                    "candidate_summary": "개선 효과가 반복됐고 운영 리스크가 낮았습니다.",
                    "case_summaries": [
                        {
                            "case_id": "CASE-004",
                            "summary": (
                                "중간 포화에서 인터벌 +22%와 우선순위를 적용해 "
                                "대기시간 41% 감소"
                            ),
                            "relevance": "direct",
                            "supports_effect": True,
                            "shows_risk": False,
                        }
                    ],
                    "claims": [],
                },
            }
        ],
        "comparison": {
            "ranking_status": "ranked",
            "ranking": [
                {
                    "rank": 1,
                    "label": "standard",
                    "recommendation_level": "우선 검토",
                    "summary": "효과와 리스크의 균형이 가장 좋습니다.",
                    "why_better": "",
                    "case_ids": ["CASE-004"],
                }
            ],
            "rag_summary": "과거 사례 기준 standard를 우선 검토합니다.",
            "overall_comment": (
                "통계 검증에서는 후보 간 우위를 확인하지 못했습니다. "
                "과거 사례 기준으로는 standard가 우선 검토 대상입니다."
            ),
        },
    }


def test_build_rag_block_shows_mvp_cases_candidate_summary_and_insight():
    text = "\n".join(_build_rag_block(_rag_evidence()))

    assert "1순위" not in text
    assert "[ RAG 유사 사례 참고 ]" in text
    assert "설비 포화 대응 보고서" in text
    assert "[보기: /reports/CASE-004]" in text
    assert "standard: 효과 높음 · 리스크 낮음" in text
    assert "[ RAG 인사이트 ]" in text
    assert "근거 충분" not in text
    assert "적합:" not in text
    assert "① 실측 효과" not in text


def test_sanitize_candidate_evidence_removes_unknown_case_ids():
    evidence = _rag_evidence()["candidates"][0]["evidence"]
    evidence["case_summaries"].append(
        {
            "case_id": "MADE-UP",
            "summary": "입력에 없는 사례",
            "relevance": "direct",
            "supports_effect": True,
            "shows_risk": False,
        }
    )
    evidence["claims"] = [
        {"text": "검증된 주장", "case_ids": ["CASE-004", "MADE-UP"]},
        {"text": "근거 없는 주장", "case_ids": ["MADE-UP"]},
    ]

    sanitized = _sanitize_candidate_evidence(
        evidence,
        [{"case_id": "CASE-004"}],
    )

    assert [case["case_id"] for case in sanitized["case_summaries"]] == ["CASE-004"]
    assert sanitized["claims"] == [
        {"text": "검증된 주장", "case_ids": ["CASE-004"]}
    ]


def test_sanitize_comparison_removes_unknown_labels_and_case_ids():
    comparison = _rag_evidence()["comparison"]
    comparison["ranking"].append(
        {
            "rank": 2,
            "label": "unknown",
            "recommendation_level": "후순위",
            "summary": "입력에 없는 후보",
            "why_better": "",
            "case_ids": ["MADE-UP"],
        }
    )
    comparison["ranking"][0]["case_ids"].append("MADE-UP")

    sanitized = _sanitize_rag_comparison(
        comparison,
        [{"label": "standard", "hits": [{"case_id": "CASE-004"}]}],
    )

    assert [item["label"] for item in sanitized["ranking"]] == ["standard"]
    assert sanitized["ranking"][0]["case_ids"] == ["CASE-004"]


def test_statistical_summary_keeps_baseline_comparison_separate():
    summary = _build_rag_statistical_summary(
        {
            "decision_info": {
                "decision_status": "equivalent_candidates",
                "decision_caveat": "통계적 동등",
            },
            "action_candidates": [
                {
                    "label": "standard",
                    "composite_score": 0.45,
                    "score_breakdown": {
                        "kpi_contributions": {
                            "q_time_min": {
                                "mean_delta": -0.28,
                                "verdict": "improved",
                            },
                            "wip": {
                                "mean_delta": -1.4,
                                "verdict": "improved",
                            },
                        }
                    },
                }
            ],
        }
    )

    assert "120분 paired 검증" in summary
    assert "후보 간 우위가 뚜렷하지 않음" in summary
    assert "q_time_min Δ-0.2800(improved)" in summary
    assert "wip Δ-1.4000(improved)" in summary
    assert "t+20h" not in summary


def test_candidate_evidence_prompt_includes_real_case_ids_and_summary_rules():
    expected = CandidateEvidence(
        effect_outlook="high",
        risk_level="low",
        evidence_strength="strong",
        candidate_summary="효과가 확인됐고 리스크가 낮습니다.",
        case_summaries=[],
        claims=[],
    )
    llm = _FakeStructuredLlm(expected)

    result = evaluate_candidate_evidence(
        plan_description="인터벌 +22% / 우선순위 상향",
        current_state_summary="WIP 80개, 대기 95분",
        hits=[
            {
                "case_id": "CASE-004",
                "report_title": "설비 포화 대응 보고서",
                "text": "인터벌 +22%와 우선순위를 적용해 대기시간 41% 감소",
            }
        ],
        llm=llm,
        action_profile={
            "interval_pct": 22,
            "has_priority": True,
            "has_superhotlot": False,
            "priority_products": ["Product_3"],
            "superhotlot_products": [],
        },
    )

    prompt = llm.messages[1]["content"]
    assert result.effect_outlook == "high"
    assert result.risk_level == "low"
    assert result.evidence_strength == "weak"
    assert result.candidate_summary == expected.candidate_summary
    assert llm.schema is CandidateEvidence
    assert "[CASE-004]" in prompt
    assert "상황 → 적용 조치 → 실제 결과" in prompt
    assert "후보 간 차이를 억지로" in llm.messages[0]["content"]


def test_comparison_prompt_separates_rag_ranking_from_statistical_comment():
    expected = RagComparison(
        ranking_status="tied",
        ranking=[],
        rag_summary="과거 사례만으로 우위를 정하기 어렵습니다.",
        overall_comment="통계 결과와 RAG 판단을 함께 확인해야 합니다.",
    )
    llm = _FakeStructuredLlm(expected)

    result = compare_candidate_evidence(
        candidate_evidence=[
            {
                "label": "standard",
                "profile": {
                    "interval_pct": 22,
                    "has_priority": True,
                    "has_superhotlot": False,
                },
                "plan_description": "인터벌 +22%",
                "evidence": {
                    "effect_outlook": "high",
                    "risk_level": "low",
                    "evidence_strength": "strong",
                    "candidate_summary": "효과와 리스크의 균형이 좋습니다.",
                    "claims": [],
                    "case_summaries": [],
                },
            }
        ],
        statistical_summary="후보 간 통계적 우위를 확인할 수 없음",
        llm=llm,
        current_context="납기 위험과 긴급 lot는 확인되지 않음",
    )

    prompt = llm.messages[1]["content"]
    assert result.ranking_status == expected.ranking_status
    assert result.ranking == expected.ranking
    assert result.rag_summary == (
        "standard: 효과 high · 리스크 low. "
        "과거 사례만으로 우위를 정하기 어렵습니다."
    )
    assert result.overall_comment == expected.overall_comment
    assert llm.schema is RagComparison
    assert "통계 검증 결과는 overall_comment 작성에만 사용" in llm.messages[0]["content"]
    assert "후보 간 통계적 우위를 확인할 수 없음" in prompt
    assert "납기 위험과 긴급 lot는 확인되지 않음" in prompt
    assert "RAG가 통계적 우위를 입증했다거나" in prompt


def test_demo_defmet43_data_makes_standard_clear_winner():
    solutions = [
        {
            "plan_id": "conservative",
            "target_toolgroups": ["DefMet_FE_43"],
            "release_interval_delta_pct": 15.0,
            "lot_adjustments": [],
            "cause_complexity": "single",
        },
        {
            "plan_id": "standard",
            "target_toolgroups": ["DefMet_FE_43"],
            "release_interval_delta_pct": 22.0,
            "lot_adjustments": [{
                "lot_plan_id": 1,
                "lot_type": "Regular_Lot_3",
                "product_name": "Product_3",
                "release_time": 3205,
                "whatif_release_time": 3205,
                "action_kind": "LOT_PRIORITY",
                "priority": 20,
                "time_to_due": 74362,
                "zone": "warn_lower",
            }],
            "cause_complexity": "single",
        },
        {
            "plan_id": "aggressive",
            "target_toolgroups": ["DefMet_FE_43"],
            "release_interval_delta_pct": 28.0,
            "lot_adjustments": [{
                "lot_plan_id": 2,
                "lot_type": "Regular_Lot_4",
                "product_name": "Product_4",
                "release_time": 3205,
                "whatif_release_time": 3205,
                "action_kind": "SET_SUPER_HOT",
                "priority": 30,
                "time_to_due": 42583,
                "zone": "danger",
            }],
            "cause_complexity": "single",
        },
    ]
    groups = build_demo_verification_results(solutions, 3180.0)
    verified_candidates = [
        verified
        for group in groups
        for verified in group["verified_candidates"]
    ]

    candidates = _build_action_candidates(verified_candidates)
    enriched, _scored, decision = _rank_candidates(candidates)
    score_by_label = {
        candidate["label"]: candidate["composite_score"]
        for candidate in enriched
    }

    assert decision["decision_status"] == "clear_winner"
    assert decision["top_label"] == "standard"
    assert score_by_label["standard"] == 0.3809
    assert score_by_label["aggressive"] == 0.1784
    assert score_by_label["conservative"] == 0.0943
    assert all(candidate["paired_n"] == 10 for candidate in enriched)


def test_demo_3780_multi_tg_uses_no_action_baseline_and_standard_wins():
    solutions = [
        {
            "plan_id": "conservative",
            "target_toolgroups": ["DE_FE_1", "Diffusion_FE_125"],
            "release_interval_delta_pct": 15.0,
            "lot_adjustments": [],
            "cause_complexity": "single",
        },
        {
            "plan_id": "standard",
            "target_toolgroups": ["DE_FE_1", "Diffusion_FE_125"],
            "release_interval_delta_pct": 22.0,
            "lot_adjustments": [{
                "lot_plan_id": 1,
                "lot_type": "Regular_Lot_3",
                "product_name": "Product_3",
                "release_time": 3825,
                "whatif_release_time": 3825,
                "action_kind": "LOT_PRIORITY",
                "priority": 20,
                "time_to_due": 74357,
                "zone": "warn_lower",
            }],
            "cause_complexity": "single",
        },
        {
            "plan_id": "aggressive",
            "target_toolgroups": ["DE_FE_1", "Diffusion_FE_125"],
            "release_interval_delta_pct": 28.0,
            "lot_adjustments": [{
                "lot_plan_id": 2,
                "lot_type": "Regular_Lot_4",
                "product_name": "Product_4",
                "release_time": 3825,
                "whatif_release_time": 3825,
                "action_kind": "SET_SUPER_HOT",
                "priority": 30,
                "time_to_due": 42577,
                "zone": "danger",
            }],
            "cause_complexity": "single",
        },
    ]
    groups = build_demo_verification_results(solutions, 3780.0)
    verified_candidates = [
        verified
        for group in groups
        for verified in group["verified_candidates"]
    ]
    candidates = _build_action_candidates(verified_candidates)
    enriched, _scored, decision = _rank_candidates(candidates)
    by_label = {candidate["label"]: candidate for candidate in enriched}

    assert decision["top_label"] == "standard"
    assert by_label["standard"]["composite_score"] == 0.6508
    assert by_label["aggressive"]["composite_score"] == 0.5758
    assert by_label["conservative"]["composite_score"] == 0.0825
    assert by_label["standard"]["kpi_stats"]["wip"]["mean_delta"] == -4.0
    assert (
        by_label["aggressive"]["per_tg_forecasts"]["Diffusion_FE_125"]
        ["action"]["wip"]
        == 20.0
    )
    assert {
        tradeoff["kpi"]
        for tradeoff in by_label["aggressive"]["tradeoffs"]
    } == {"wip", "available_tool_ratio"}


def test_multi_tg_action_option_uses_anchor_tg_without_average():
    from agents.compare_agent.node import _build_action_option

    candidate = {
        "label": "standard",
        "action_kind": "DISPATCH_RULE_OVERRIDE",
        "description": "표준 조정안",
        "score_breakdown": {"kpi_contributions": {}},
        "per_tg_forecasts": {
            "DE_FE_1": {
                "current": {"q_time_min": 62.96, "wip": 10.0},
                "no_action": {"q_time_min": 96.39, "wip": 12.0},
                "action": {"q_time_min": 48.0, "wip": 8.0},
            },
            "Diffusion_FE_125": {
                "current": {"q_time_min": 35.98, "wip": 5.0},
                "no_action": {"q_time_min": 46.26, "wip": 18.0},
                "action": {"q_time_min": 30.0, "wip": 14.0},
            },
        },
        "plan_meta": {},
    }
    decision_info = {
        "decision_status": "clear_winner",
        "top_label": "standard",
    }

    option = _build_action_option(
        candidate,
        {},
        decision_info,
        anchor_toolgroup="DE_FE_1",
    )

    assert option["kpi_impact"]["q_time_min"]["now"] == 62.96
    assert option["kpi_impact"]["q_time_min"]["after"] == 48.0
    assert option["kpi_impact"]["q_time_min"]["delta"] == -14.96
    assert option["kpi_impact"]["wip"]["delta"] == -2.0
    assert option["per_tg_forecasts"] == candidate["per_tg_forecasts"]
    assert "평균" not in option["comparison_basis"]
    assert option["comparison_basis"] == "DE_FE_1 현재 상태 대비 대응안 2시간 후"


def test_report_candidate_table_keeps_original_anchor_summary_format():
    candidate = SimpleNamespace(
        label="standard",
        is_approved=True,
        kind="DISPATCH_RULE_OVERRIDE",
        description="표준 조정안",
        per_tg_forecasts={},
        kpi_impact=[
            SimpleNamespace(kpi="q_time_min", delta=-14.96),
            SimpleNamespace(kpi="wip", delta=-2.0),
        ],
        simulation=None,
        operational=None,
    )
    report = SimpleNamespace(
        actions=SimpleNamespace(candidates=[candidate]),
    )

    table = _render_candidates_compare_table(report)

    assert "| 대기시간 변화 | WIP 변화 |" in table
    assert "-14.96분" in table
    assert "-2" in table
    assert "TG별 대기시간 변화" not in table
    assert "평균" not in table


def test_hitl_summary_keeps_original_compact_format():
    kpi_impact = {
        "q_time_min": {"now": 1.34, "after": 1.06, "delta": -0.28},
        "wip": {"now": 7, "after": 5.6, "delta": -1.4},
        "wait_ratio": {"now": 0.22, "after": 0.16, "delta": -0.06},
        "utilization_avg": {
            "now": 0.1504,
            "after": 0.1304,
            "delta": -0.02,
        },
        "available_tool_ratio": {"now": 1.0, "after": 1.0, "delta": 0.0},
    }
    result = {
        "meta": {
            "scenario_name": "글로벌 플랜 A/B",
            "anchor_toolgroup": "DefMet_FE_43",
            "target_toolgroups": ["DefMet_FE_43"],
        },
        "current_state": {
            "kpi": {
                "wip": {"value": 7, "unit": "lots"},
                "q_time_min": {"value": 1.34, "unit": "min"},
                "wait_ratio": {"value": 0.22, "unit": "ratio"},
                "utilization_avg": {"value": 0.1504, "unit": "ratio"},
                "available_tool_ratio": {"value": 1.0, "unit": "ratio"},
                "risk_score": {"value": 0.7848, "unit": "score"},
            },
            "natural_forecast_2h": None,
        },
        "recommendation": {
            "confidence_level": "HIGH",
            "recommended_label": "standard",
            "recommendation_status": "ai_recommended",
            "headline": "표준안을 우선 검토합니다.",
            "primary_reason": "5개 KPI 균형이 가장 좋습니다.",
        },
        "action_options": [{
            "label": "standard",
            "badge": "ai_recommended",
            "composite_score": 0.3809,
            "simulation": {"simulation_confidence": 0.992},
            "kpi_impact": kpi_impact,
        }],
        "data_quality": {"status": "ok"},
        "decision_meta": {
            "decision_status": "clear_winner",
            "decision_caveat": "",
        },
    }

    prompt = _build_hitl_prompt(result)

    assert "q_time Δ-0.3분" in prompt
    assert "WIP" in prompt
    assert "wait_ratio Δ" not in prompt
    assert "utilization_avg Δ" not in prompt
    assert "available_tool_ratio Δ" not in prompt
    assert "risk_score" in prompt


def test_hitl_summary_adds_readable_per_tg_five_kpis():
    result = {
        "meta": {
            "scenario_name": "글로벌 플랜 A/B",
            "anchor_toolgroup": "DE_FE_1",
            "target_toolgroups": ["DE_FE_1", "Diffusion_FE_125"],
        },
        "current_state": {"kpi": {}, "natural_forecast_2h": None},
        "recommendation": {
            "confidence_level": "HIGH",
            "recommended_label": "standard",
            "recommendation_status": "ai_recommended",
            "headline": "",
            "primary_reason": "",
        },
        "action_options": [{
            "label": "standard",
            "badge": "ai_recommended",
            "composite_score": 0.6508,
            "simulation": {"simulation_confidence": 0.99},
            "kpi_impact": {
                "q_time_min": {"delta": -14.96},
                "wip": {"delta": -2.0},
            },
            "per_tg_forecasts": {
                "DE_FE_1": {
                    "current": {
                        "q_time_min": 62.96,
                        "wip": 10,
                        "wait_ratio": 0.25,
                        "utilization_avg": 0.9943,
                        "available_tool_ratio": 1.0,
                    },
                    "action": {
                        "q_time_min": 48.0,
                        "wip": 8,
                        "wait_ratio": 0.18,
                        "utilization_avg": 0.88,
                        "available_tool_ratio": 1.0,
                    },
                },
            },
        }],
        "data_quality": {"status": "ok"},
        "decision_meta": {
            "decision_status": "clear_winner",
            "decision_caveat": "",
        },
    }

    prompt = _build_hitl_prompt(result)

    assert "standard [AI 추천]  score 0.651" in prompt
    assert "└ DE_FE_1" in prompt
    assert "q_time 62.96→48.0분" in prompt
    assert "WIP 10→8" in prompt
    assert "wait 0.25→0.18" in prompt
    assert "util 0.9943→0.88" in prompt
    assert "avail 1.0→1.0" in prompt


def test_verification_output_shows_current_to_action_only(capsys):
    print_verification_results([{
        "demo_mock": True,
        "verified_candidates": [{
            "label": "standard",
            "paired_n": 10,
            "verdict": "improved",
            "aggregation_rule": "복합 TG 최악값 기준",
            "kpi_stats": {},
            "per_tg_forecasts": {
                "DE_FE_1": {
                    "current": {
                        "q_time_min": 62.96,
                        "wip": 10.0,
                        "wait_ratio": 0.25,
                        "utilization_avg": 0.9943,
                        "available_tool_ratio": 1.0,
                    },
                    "no_action": {
                        "q_time_min": 96.39,
                        "wip": 12.0,
                        "wait_ratio": 0.5,
                        "utilization_avg": 1.0,
                        "available_tool_ratio": 1.0,
                    },
                    "action": {
                        "q_time_min": 58.0,
                        "wip": 8.0,
                        "wait_ratio": 0.18,
                        "utilization_avg": 0.88,
                        "available_tool_ratio": 1.0,
                    },
                },
            },
        }],
    }])

    output = capsys.readouterr().out

    assert "TG별 현재 → 대응안 2h" in output
    assert "q 62.96→58.0분" in output
    assert "62.96→96.39→58.0" not in output
    assert "검정 변화" in output
    assert "평균 변화" not in output
