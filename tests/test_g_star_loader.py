from __future__ import annotations

import json

from agents.cause_analyzer.g_star_loader import load_g_star


def test_load_g_star_reads_inline_kpi_evidence(tmp_path):
    (tmp_path / "g_star_T3180.json").write_text(
        json.dumps(
            {
                "anchor_tg": "DefMet_FE_43",
                "toolgroups": ["DefMet_FE_43"],
                "t0_sim_minute": 3180,
                "alarm_threshold": 0.7,
                "n_g_star": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "agent_handoff_g_star_analysis.json").write_text(
        json.dumps(
            {
                "g_star_analysis": {
                    "g_star_toolgroups": ["DefMet_FE_43", "WE_FE_84"],
                    "g_star_kpi_evidence": [
                        {
                            "toolgroup": "DefMet_FE_43",
                            "kpi": "utilization_avg",
                            "delta_mean": 0.399,
                            "t_p_adj": 0.00001,
                            "kpi_significant": 1,
                        },
                        {
                            "toolgroup": "DefMet_FE_43",
                            "kpi": "wait_ratio",
                            "delta_mean": -0.231,
                            "t_p_adj": 1.0,
                            "kpi_significant": 0,
                        },
                        {
                            "toolgroup": "WE_FE_84",
                            "kpi": "q_time_min",
                            "delta_mean": 118.127,
                            "t_p_adj": 0.0,
                            "kpi_significant": 1,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = load_g_star(3180, tmp_path)

    assert result is not None
    assert result.toolgroups == ["DefMet_FE_43", "WE_FE_84"]
    assert result.kpi_evidence["DefMet_FE_43"][0].kpi == "utilization_avg"
    assert result.kpi_evidence["DefMet_FE_43"][0].delta_mean == 0.399
    assert result.kpi_evidence["DefMet_FE_43"][0].t_p_adj == 0.00001
    assert result.kpi_evidence["DefMet_FE_43"][0].significant is True
    assert result.kpi_evidence["DefMet_FE_43"][1].significant is False
    assert result.kpi_evidence["WE_FE_84"][0].kpi == "q_time_min"
    assert result.kpi_evidence["WE_FE_84"][0].t_p_adj == 0.0
