"""LangGraph 노드: alerts → cause_reports.

흐름:
  1. SHAP / 트렌드 / 업스트림 / G* 4개 분석
  2. Evidence Aggregator — 피처별 votes·confidence 계산
  3. LLM Judge — 증거 수렴 기반 root cause 판정
  4. needs_more_data=True 시 window 확장 후 재시도 (최대 2회)
"""

from __future__ import annotations

from pathlib import Path

from agents.cascade_analyzer.dag_builder import build_dag
from agents.cause_analyzer.evidence_aggregator import aggregate_evidence
from agents.cause_analyzer.g_star_loader import load_g_star
from agents.cause_analyzer.llm_judge import judge
from agents.cause_analyzer.shap_analyzer import get_shap_top
from agents.cause_analyzer.trend_analyzer import get_trend_top
from agents.cause_analyzer.upstream_tracker import find_upstream_suspects
from agents.data.kpi_loader import load_kpi_window
from agents.logger import get_logger
from agents.schemas.alert import SeverityLevel
from agents.schemas.cause import (
    CauseReport,
    ConsensusResult,
    GStarKpiResult,
    KpiComparison,
    SimForecast,
)
from agents.state import PipelineState

_ANALYZE_SEVERITIES = {SeverityLevel.CRITICAL, SeverityLevel.HIGH}
_MAX_RETRIES = 2

_log = get_logger(__name__)


def analyze_cause(
    state: PipelineState,
    run_sim: bool = True,
) -> PipelineState:
    if not state["kpi_snapshot"]:
        return {**state, "cause_reports": []}

    alerts = [a for a in state["alerts"] if a.severity in _ANALYZE_SEVERITIES]
    kpi_map = {k.toolgroup: k for k in state["kpi_snapshot"]}
    snapshot_time = state["kpi_snapshot"][0].snapshot_time

    G = build_dag()
    window = load_kpi_window(snapshot_time, n_snapshots=6)

    # snapshot_time은 time_step 기준 시뮬 tick — G* 경로와 일치
    _SIM_CSV_DIR = (
        Path(__file__).parent.parent.parent.parent
        / "Simulation" / "simulation" / "sim_csv_out"
    )
    fwd_base_dir = _SIM_CSV_DIR / f"fwd_base_t{int(snapshot_time)}"
    g_star = load_g_star(t0=snapshot_time, out_dir=fwd_base_dir)
    if g_star:
        _log.info(f"[G*] {len(g_star.toolgroups)}개 TG 로드 (anchor={g_star.anchor_tg})")
    else:
        _log.info("[G*] 파일 없음 — G* 미반영")

    forward_kpis: dict = {}
    manifest_csv = fwd_base_dir / "runs_manifest.csv"
    if manifest_csv.is_file():
        from agents.sim_runner.forecaster import load_forward_kpis_median
        forward_kpis = load_forward_kpis_median(manifest_csv)
        if forward_kpis:
            _log.info(f"[Forward KPI] G* baseline 30회 중위값 — {len(forward_kpis)}개 TG")
    if not forward_kpis and run_sim:
        try:
            from agents.sim_runner.forecaster import load_forward_kpis
            from agents.sim_runner.trigger import run_forward
            _log.info("[Forward Sim] G* baseline 없음 — 1회 fallback")
            fwd_csv_dir = run_forward(t0=snapshot_time, horizon_min=120.0)
            forward_kpis = load_forward_kpis(fwd_csv_dir)
            _log.info(f"[Forward Sim] 완료 — {len(forward_kpis)}개 TG")
        except Exception as e:
            _log.warning(f"[Forward Sim 스킵] {type(e).__name__}: {e}")

    prev_kpi_list = state["prev_kpi_snapshot"] or None
    reports: list[CauseReport] = []

    for alert in alerts:
        tg = alert.toolgroup
        kpi = kpi_map.get(tg)
        if kpi is None:
            continue

        # ── 1. SHAP
        shap_top = get_shap_top(kpi, top_n=4, prev_kpi_list=prev_kpi_list)

        # ── 2. 업스트림 역추적
        upstream_suspects = find_upstream_suspects(G, tg, kpi_map)

        # ── 3. G* KPI 증거
        tg_g_star_evidence = (
            g_star.kpi_evidence.get(tg, []) if g_star and g_star.kpi_evidence else []
        )

        # ── 4. 트렌드 + Evidence Aggregation + LLM Judge (재시도 루프)
        trend_top = get_trend_top(window, tg, top_n=3)
        evidence_bundle = []
        categories = []
        judgment = None

        for retry_n in range(_MAX_RETRIES + 1):
            evidence_bundle, categories = aggregate_evidence(
                shap_top, trend_top, upstream_suspects, tg_g_star_evidence
            )
            judgment = judge(
                tg, evidence_bundle, categories, upstream_suspects,
                g_star=g_star,
                retry_n=retry_n,
            )

            if not judgment.needs_more_data or retry_n >= _MAX_RETRIES:
                if retry_n > 0:
                    _log.info(
                        f"[Cause] {tg} retry #{retry_n} 완료 — "
                        f"confidence={judgment.primary_confidence}"
                    )
                break

            _log.info(f"[Cause] {tg} retry #{retry_n + 1} 요청 (LLM: needs_more_data=True)")
            n_snaps = 6 * (retry_n + 2)
            n_hops = 3 + retry_n + 1
            window_r = load_kpi_window(snapshot_time, n_snapshots=n_snaps)
            trend_top = get_trend_top(window_r, tg, top_n=5)
            upstream_suspects = find_upstream_suspects(G, tg, kpi_map, max_hops=n_hops)

        # ── 5. Forward 시뮬레이션 비교
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

        # ── 6. ConsensusResult 역호환 구성
        g_star_set = set(g_star.toolgroups if g_star else [])
        g_star_confirmed = tg in g_star_set
        g_star_sig_kpis = []
        if g_star_confirmed and g_star and g_star.kpi_evidence:
            raw_evs = g_star.kpi_evidence.get(tg, [])
            g_star_sig_kpis = [
                GStarKpiResult(
                    kpi=e.kpi, delta_mean=e.delta_mean,
                    t_p_adj=e.t_p_adj, significant=e.significant,
                )
                for e in sorted(
                    raw_evs,
                    key=lambda e: (0 if e.significant else 1, -abs(e.delta_mean)),
                )
            ]

        agreed = [judgment.primary_cause] + list(judgment.secondary_causes or [])
        consensus = ConsensusResult(
            agreed_features=agreed,
            conflicted_features=list(judgment.dismissed or []),
            upstream_aligns=bool(upstream_suspects),
            sim_aligns=bool(sim_forecast and sim_forecast.gets_worse),
            g_star_confirmed=g_star_confirmed,
            g_star_upstream_confirmed=sorted(set(upstream_suspects) & g_star_set),
            g_star_sig_kpis=g_star_sig_kpis,
            g_star_proba=(g_star.tg_proba.get(tg, 0.0) if g_star and g_star.tg_proba else 0.0),
            g_star_n_total=(g_star.n_total_tg if g_star else 0),
            g_star_n_alarm=len(g_star_set),
            g_star_toolgroups_all=sorted(g_star_set),
            confidence_level=judgment.primary_confidence,
            summary=judgment.primary_reasoning,
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
                evidence_bundle=evidence_bundle,
                cause_categories=categories,
                judgment=judgment,
                cause_summary=judgment.cause_summary,
            )
        )

    return {**state, "cause_reports": reports}
