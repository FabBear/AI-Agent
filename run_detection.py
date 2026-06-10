"""
병목 감지 → 확산 영향 → 원인 분석 파이프라인 실행 스크립트.

사용:
    uv run python run_detection.py
    uv run python run_detection.py --snapshot 3000
    uv run python run_detection.py --cause-only DefMEt_FE_118
    uv run python run_detection.py --auto-approve
"""

from __future__ import annotations

import argparse
import os

import agents.token_tracker as token_tracker
from agents.data.kpi_loader import load_kpi_snapshot, load_kpi_window
from agents.display import print_alert_table, print_cause_reports, print_solutions, print_report_results
from agents.pipeline import build_pipeline
from agents.state import PipelineState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=float, default=None)
    parser.add_argument("--cause-only", type=str, default=None)
    parser.add_argument("--auto-approve", action="store_true", help="HITL 자동 승인 (AI 추천안 자동 선택)")
    args = parser.parse_args()

    if args.auto_approve:
        os.environ["AUTO_APPROVE"] = "1"
        print("⚡  AUTO_APPROVE 모드: HITL 자동 승인 활성화\n")

    kpi_list = load_kpi_snapshot(snapshot_time=args.snapshot)
    if not kpi_list:
        print("\n✅  KPI 데이터 없음\n")
        return

    snapshot_time = kpi_list[0].snapshot_time
    print(f"⏱   snapshot_time = {snapshot_time:.0f} epoch-min")
    print(f"📊  분석 대상: {len(kpi_list)}개 toolgroup\n")

    window = load_kpi_window(snapshot_time, n_snapshots=2)
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
        "verification_results": [],
        "compare_inputs": [],
        "compare_formatted": [],
        "compare_results": [],
        "report_draft": [],
        "report_results": [],
    }

    pipeline = build_pipeline()
    alerted = False

    for chunk in pipeline.stream(initial_state):
        for node_name, state in chunk.items():

            if node_name == "cascade":
                alerts = state.get("alerts", [])
                if not alerts:
                    print("\n✅  병목 없음\n")
                    return
                alerted = True
                print(f"[2단계] 확산 영향 분석 완료 → {len(alerts)}개 알림\n")
                print_alert_table(alerts)

            elif node_name == "cause":
                reports = state.get("cause_reports", [])
                if args.cause_only:
                    reports = [r for r in reports if r.toolgroup == args.cause_only]
                print_cause_reports(reports)

            elif node_name == "solution":
                solutions = state.get("solution_candidates", [])
                if args.cause_only:
                    solutions = [s for s in solutions if s.get("toolgroup") == args.cause_only]
                print_solutions(solutions)

            elif node_name == "report_save":
                print_report_results(state.get("report_results", []))

    if not alerted:
        print("\n✅  병목 없음\n")
        return

    token_tracker.print_summary()
    token_tracker.save_log()


if __name__ == "__main__":
    main()
