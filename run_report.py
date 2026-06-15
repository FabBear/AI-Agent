"""
HITL 승인 후 최종보고서 생성 스크립트.

사용법:
    uv run python run_report.py --token <hitl_token>
    uv run python run_report.py --token <hitl_token> --approved-by 김관리자 --role 공정관리자
    uv run python run_report.py --token <hitl_token> --reject --reason "추가 검토 필요"
    uv run python run_report.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent
_PENDING_DIR = _ROOT / "hitl_pending"


def list_pending() -> None:
    if not _PENDING_DIR.exists() or not any(_PENDING_DIR.glob("*.json")):
        print("대기 중인 HITL 요청이 없습니다.")
        return
    print(f"{'='*60}")
    print("대기 중인 HITL 요청 목록")
    print(f"{'='*60}")
    for f in sorted(_PENDING_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        toolgroups = [cf["toolgroup"] for cf in data.get("compare_formatted", [])]
        created_at = data.get("created_at", "-")
        token = data.get("hitl_token", f.stem)
        print(f"  token    : {token}")
        print(f"  생성시각 : {created_at}")
        print(f"  toolgroup: {', '.join(toolgroups)}")
        print(f"  실행 명령: python run_report.py --token {token}")
        print()


def run_report_generation(
    token: str,
    approved_by: str,
    role: str,
    comment: str,
    reject: bool,
    reason: str,
) -> None:
    pending_path = _PENDING_DIR / f"{token}.json"
    if not pending_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {pending_path}")
        print("  python run_report.py --list  로 토큰 목록을 확인하세요.")
        sys.exit(1)

    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    if reject:
        approval_info = {
            "status": "반려",
            "approved_by": approved_by,
            "approved_role": role,
            "approved_at": now_str,
            "comment": None,
            "rejection_reason": reason or "사유 미입력",
        }
        print(f"\n[보고서 생성 시작 — 반려] token={token}")
        print(f"  검토자  : {approved_by} ({role})")
        print(f"  반려 사유: {approval_info['rejection_reason']}\n")
    else:
        approval_info = {
            "status": "승인",
            "approved_by": approved_by,
            "approved_role": role,
            "approved_at": now_str,
            "comment": comment or "즉시 적용 승인",
            "rejection_reason": None,
        }
        print(f"\n[보고서 생성 시작 — 승인] token={token}")
        print(f"  승인자: {approved_by} ({role})")
        print(f"  의견  : {approval_info['comment']}\n")

    state = _reconstruct_state(pending, approval_info)

    from agents.pipeline import build_report_pipeline
    from agents.display import print_report_results

    pipeline = build_report_pipeline()
    result = pipeline.invoke(state)

    report_results = result.get("report_results", [])
    print_report_results(report_results)
    print(f"\n[보고서 생성 완료] 보고서 {len(report_results)}개 생성")


def _reconstruct_state(pending: dict, approval_info: dict) -> dict:
    from agents.schemas.alert import BottleneckAlert
    from agents.schemas.kpi import ToolGroupKPI
    from agents.schemas.cause import CauseReport

    compare_formatted = pending["compare_formatted"]

    compare_results = [
        {
            "toolgroup": cf["toolgroup"],
            "recommendation": cf["recommendation"],
            "approval_info": approval_info,
            "action_effects": cf.get("action_effects", []),
            "result_v2": cf.get("result_v2"),
            "json_output_path": "",
        }
        for cf in compare_formatted
    ]

    return {
        "kpi_snapshot":        [ToolGroupKPI.model_validate(k) for k in pending.get("kpi_snapshot", [])],
        "prev_kpi_snapshot":   [ToolGroupKPI.model_validate(k) for k in pending.get("prev_kpi_snapshot", [])],
        "potential_bottlenecks": [],
        "alerts":              [BottleneckAlert.model_validate(a) for a in pending.get("alerts", [])],
        "cause_reports":       [CauseReport.model_validate(r) for r in pending.get("cause_reports", [])],
        "cascade_report":      pending.get("cascade_report"),
        "solution_candidates": pending.get("solution_candidates", []),
        "hitl_approved":       True,
        "hitl_token":          pending.get("hitl_token"),
        "verification_results": [],
        "compare_inputs":      [],
        "compare_formatted":   compare_formatted,
        "compare_results":     compare_results,
        "report_draft":        [],
        "report_results":      [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HITL 승인 후 최종보고서 생성")
    parser.add_argument("--token", type=str, help="HITL 토큰 (분석 완료 시 출력됨)")
    parser.add_argument("--approved-by", type=str, default="관리자", help="승인자 이름")
    parser.add_argument("--role", type=str, default="공정관리자", help="승인자 직책")
    parser.add_argument("--comment", type=str, default="", help="승인 의견")
    parser.add_argument("--reject", action="store_true", help="반려 처리")
    parser.add_argument("--reason", type=str, default="", help="반려 사유 (--reject 시)")
    parser.add_argument("--list", action="store_true", help="대기 중인 HITL 목록 출력")
    args = parser.parse_args()

    if args.list:
        list_pending()
        return

    if not args.token:
        parser.error("--token 이 필요합니다. (python run_report.py --list 로 토큰 확인)")

    run_report_generation(
        token=args.token,
        approved_by=args.approved_by,
        role=args.role,
        comment=args.comment,
        reject=args.reject,
        reason=args.reason,
    )


if __name__ == "__main__":
    main()
