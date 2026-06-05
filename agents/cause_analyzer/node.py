"""LangGraph 노드: alerts → cause_reports (SHAP + 트렌드 + 업스트림 + Forward 시뮬)."""

from pathlib import Path

from agents.cascade_analyzer.dag_builder import build_dag
from agents.cause_analyzer.llm_summarizer import summarize
from agents.cause_analyzer.shap_analyzer import get_shap_top
from agents.cause_analyzer.trend_analyzer import get_trend_top
from agents.cause_analyzer.upstream_tracker import find_upstream_suspects
from agents.data.kpi_loader import load_kpi_window
from agents.logger import get_logger
from agents.schemas.cause import CauseReport, KpiComparison, SimForecast
from agents.state import PipelineState

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
    alerts = state["alerts"]
    kpi_map = {k.toolgroup: k for k in state["kpi_snapshot"]}
    snapshot_time = state["kpi_snapshot"][0].snapshot_time

    G = build_dag(csv_dir)
    window = load_kpi_window(csv_dir, snapshot_time, n_snapshots=6)

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

    prev_kpi_list = state.get("prev_kpi_snapshot") or None

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
            # WIP·wait_ratio는 순간 카운트라 신뢰성 높음
            # q_time_min은 윈도우 평균이라 forward sim에서 리셋되므로 제외
            gets_worse = any(
                v.pct_change > 5 for k, v in kpi_delta.items() if k in ("wip", "wait_ratio")
            )
            sim_forecast = SimForecast(
                t0=snapshot_time,
                t_future=future_kpi.snapshot_time,
                kpi_delta=kpi_delta,
                gets_worse=gets_worse,
            )

        cause_summary = summarize(tg, shap_top, trend_top, upstream_suspects, sim_forecast)

        reports.append(
            CauseReport(
                toolgroup=tg,
                snapshot_time=snapshot_time,
                shap_top=shap_top,
                trend_top=trend_top,
                upstream_suspects=upstream_suspects,
                sim_forecast=sim_forecast,
                cause_summary=cause_summary,
            )
        )

    return {**state, "cause_reports": reports}
