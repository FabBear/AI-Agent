"""
병목 감지 → 확산 영향 → 원인 분석 파이프라인 실행 스크립트.

사용:
    uv run python run_detection.py
    uv run python run_detection.py --snapshot 3000
    uv run python run_detection.py --cause-only DefMEt_FE_118
"""

from __future__ import annotations

import argparse
from pathlib import Path

import agents.token_tracker as token_tracker
from agents.data.kpi_loader import load_kpi_snapshot, load_kpi_window
from agents.display import print_alert_table, print_cause_reports, print_solutions
from agents.pipeline import build_pipeline
from agents.state import PipelineState

_DEFAULT_CSV = Path(__file__).parent.parent / "Simulation" / "simulation" / "sample_csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--snapshot", type=float, default=None)
    parser.add_argument("--cause-only", type=str, default=None)
    args = parser.parse_args()

    print(f"\n📂  데이터: {args.csv_dir}")
    kpi_list = load_kpi_snapshot(args.csv_dir, snapshot_time=args.snapshot)
    if not kpi_list:
        print("\n✅  KPI 데이터 없음\n")
        return

    snapshot_time = kpi_list[0].snapshot_time
    print(f"⏱   snapshot_time = {snapshot_time:.0f} min  ({snapshot_time / 60:.1f} h)")
    print(f"📊  분석 대상: {len(kpi_list)}개 toolgroup\n")

    window = load_kpi_window(args.csv_dir, snapshot_time, n_snapshots=2)
    window_times = sorted(window.keys())
    prev_kpi_list = window[window_times[0]] if len(window_times) >= 2 else []

    initial_state: PipelineState = {
        "kpi_snapshot": kpi_list,
        "prev_kpi_snapshot": prev_kpi_list,
        "potential_bottlenecks": [],
        "alerts": [],
        "cause_reports": [],
        "cascade_report": None,
        "solution_candidates": [],
        "hitl_approved": None,
    }

    state = build_pipeline(csv_dir=args.csv_dir).invoke(initial_state)

    alerts = state["alerts"]
    if not alerts:
        print("\n✅  병목 없음\n")
        return

    print(f"[2단계] 확산 영향 분석 완료 → {len(alerts)}개 알림\n")
    print_alert_table(alerts)

    reports = state["cause_reports"]
    if args.cause_only:
        reports = [r for r in reports if r.toolgroup == args.cause_only]
    print_cause_reports(reports)

    solutions = state["solution_candidates"]
    if args.cause_only:
        solutions = [s for s in solutions if s["toolgroup"] == args.cause_only]
    print_solutions(solutions)

    token_tracker.print_summary()
    token_tracker.save_log()


if __name__ == "__main__":
    main()
