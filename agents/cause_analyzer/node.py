"""LangGraph 노드: alerts → cause_reports (SHAP + 트렌드 + 업스트림 + Forward 시뮬 + G*)."""

from pathlib import Path

from agents.cascade_analyzer.dag_builder import build_dag
from agents.cause_analyzer.consensus_checker import check_consensus
from agents.cause_analyzer.g_star_loader import load_g_star
from agents.cause_analyzer.llm_summarizer import summarize
from agents.cause_analyzer.shap_analyzer import get_shap_top
from agents.cause_analyzer.trend_analyzer import get_trend_top
from agents.cause_analyzer.upstream_tracker import find_upstream_suspects
from agents.data.kpi_loader import load_kpi_window
from agents.logger import get_logger
from agents.schemas.alert import SeverityLevel
from agents.schemas.cause import CauseReport, ConsensusResult, GStarKpiResult, KpiComparison, SimForecast
from agents.state import PipelineState

_ANALYZE_SEVERITIES = {SeverityLevel.CRITICAL, SeverityLevel.HIGH}

_log = get_logger(__name__)

_DEFAULT_CSV = (
    Path(__file__).parent.parent.parent.parent / "Simulation" / "simulation" / "sample_csv"
)


def analyze_cause(
    state: PipelineState,
    csv_dir: str | Path = _DEFAULT_CSV,
    run_sim: bool = True,
) -> PipelineState:
    if not state["kpi_snapshot"]:
        return {**state, "cause_reports": []}
    alerts = [a for a in state["alerts"] if a.severity in _ANALYZE_SEVERITIES]
    kpi_map = {k.toolgroup: k for k in state["kpi_snapshot"]}
    snapshot_time = state["kpi_snapshot"][0].snapshot_time

    G = build_dag(csv_dir)
    window = load_kpi_window(csv_dir, snapshot_time, n_snapshots=6)

    # G* 분석 결과 로드 (배치로 사전 생성된 파일)
    g_star = load_g_star(t0=snapshot_time)
    if g_star:
        _log.info(f"[G*] {len(g_star.toolgroups)}개 TG 로드 (anchor={g_star.anchor_tg})")
    else:
        _log.info("[G*] 파일 없음 — G* 미반영")
    g_star_toolgroups = g_star.toolgroups if g_star else []

    # Forward 시뮬: 한 번만 실행해서 결과 재사용
    forward_kpis: dict = {}
    if run_sim:
        try:
            from agents.sim_runner.trigger import run_forward
            from agents.sim_runner.forecaster import load_forward_kpis

            _log.info("[Forward Sim] 2시간 후 예측 시뮬레이션 실행 중...")
            fwd_csv_dir = run_forward(t0=snapshot_time, horizon_min=120.0)
            forward_kpis = load_forward_kpis(fwd_csv_dir)
            _log.info(f"[Forward Sim] 완료 — {len(forward_kpis)}개 TG 결과")
        except Exception as e:
            _log.warning(f"[Forward Sim 스킵] {type(e).__name__}: {e}")

    prev_kpi_list = state["prev_kpi_snapshot"] or None

    reports: list[CauseReport] = []
    for alert in alerts:
        tg = alert.toolgroup
        kpi = kpi_map.get(tg)
        if kpi is None:
            continue

        shap_top = get_shap_top(kpi, top_n=4, prev_kpi_list=prev_kpi_list)
        trend_top = get_trend_top(window, tg, top_n=3)
        upstream_suspects = find_upstream_suspects(G, tg, kpi_map)

        # TG별 Forward 시뮬 비교
        sim_forecast: SimForecast | None = None
        future_kpi = forward_kpis.get(tg)
        if future_kpi is not None:
            from agents.sim_runner.forecaster import compare_kpis

            comparison = compare_kpis(kpi, future_kpi)
            kpi_delta = {k: KpiComparison(**v) for k, v in comparison.items()}
            gets_worse = any(
                v.pct_change > 5 for k, v in kpi_delta.items() if k in ("wip", "wait_ratio")
            )
            sim_forecast = SimForecast(
                t0=snapshot_time,
                t_future=future_kpi.snapshot_time,
                kpi_delta=kpi_delta,
                gets_worse=gets_worse,
            )

        # 합의 분석 (G* 포함)
        consensus_raw = check_consensus(
            shap_top, trend_top, upstream_suspects, sim_forecast,
            toolgroup=tg,
            g_star_toolgroups=g_star_toolgroups,
            g_star_evidence=g_star.kpi_evidence if g_star else None,
            g_star_tg_proba=g_star.tg_proba if g_star else None,
            g_star_n_total=g_star.n_total_tg if g_star else 0,
        )
        consensus = ConsensusResult(
            agreed_features=consensus_raw.agreed_features,
            conflicted_features=consensus_raw.conflicted_features,
            upstream_aligns=consensus_raw.upstream_aligns,
            sim_aligns=consensus_raw.sim_aligns,
            g_star_confirmed=consensus_raw.g_star_confirmed,
            g_star_upstream_confirmed=consensus_raw.g_star_upstream_confirmed,
            g_star_sig_kpis=[
                GStarKpiResult(
                    kpi=e.kpi, delta_mean=e.delta_mean,
                    t_p_adj=e.t_p_adj, significant=e.significant,
                )
                for e in consensus_raw.g_star_sig_kpis
            ],
            g_star_proba=consensus_raw.g_star_proba,
            g_star_n_total=consensus_raw.g_star_n_total,
            g_star_n_alarm=consensus_raw.g_star_n_alarm,
            g_star_toolgroups_all=consensus_raw.g_star_toolgroups_all,
            confidence_level=consensus_raw.confidence_level,
            summary=consensus_raw.summary,
        )

        cause_summary = summarize(
            tg, shap_top, trend_top, upstream_suspects, sim_forecast, consensus_raw
        )

        reports.append(
            CauseReport(
                toolgroup=tg,
                snapshot_time=snapshot_time,
                shap_top=shap_top,
                trend_top=trend_top,
                upstream_suspects=upstream_suspects,
                sim_forecast=sim_forecast,
                consensus=consensus,
                cause_summary=cause_summary,
            )
        )

    return {**state, "cause_reports": reports}
