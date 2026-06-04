#!/usr/bin/env python3
"""
보고서 생성 Agent — LangGraph 기반

입력: mock_report_input.json 형태의 dict
출력: reports/ 폴더에 Markdown + JSON 저장

실행:
    python -m agents.report_agent.node agents/report_agent/mock_report_input.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.report_agent.writer import (
    ReportState,
    write_summary,
    write_diffusion,
    write_cause,
    write_actions,
)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = _ROOT / "report_agent_out"


# ── Non-LLM Node 함수들 ───────────────────────────────────────────────────────

def node_write_header(state: ReportState) -> dict:
    """헤더 — LLM 불필요, 구조화 데이터로 직접 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    severity = state.get("severity", "MEDIUM")
    badge = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}.get(severity, severity)
    header = (
        f"# FAB 병목 대응 보고서\n\n"
        f"| 항목 | 내용 |\n"
        f"|------|------|\n"
        f"| 공정명 | `{state.get('process_name', '-')}` |\n"
        f"| 심각도 | **{badge}** |\n"
        f"| 탐지시각 | {state.get('detected_at', '-')} |\n"
        f"| 보고서 생성일시 | {now} |\n\n"
        f"---"
    )
    return {"section_header": header}


def node_write_approval(state: ReportState) -> dict:
    """5. 승인 정보 — LLM 불필요, 구조화 데이터로 직접 생성"""
    ai = state.get("approval_info") or {}
    is_rejected = ai.get("status") == "반려"
    detected_at = state.get("detected_at", "-")

    if is_rejected:
        section = (
            f"## 5. 승인 정보\n\n"
            f"| 항목 | 내용 |\n"
            f"|------|------|\n"
            f"| 탐지시각 | {detected_at} |\n"
            f"| 검토자 | {ai.get('approved_by', '-')} ({ai.get('approved_role', '-')}) |\n"
            f"| 상태 | 반려 |\n"
            f"| 반려일시 | {ai.get('approved_at', '-')} |\n"
            f"| 반려 사유 | {ai.get('rejection_reason', '-')} |"
        )
    else:
        section = (
            f"## 5. 승인 정보\n\n"
            f"| 항목 | 내용 |\n"
            f"|------|------|\n"
            f"| 탐지시각 | {detected_at} |\n"
            f"| 승인자 | {ai.get('approved_by', '-')} ({ai.get('approved_role', '-')}) |\n"
            f"| 상태 | 승인 |\n"
            f"| 승인일시 | {ai.get('approved_at', '-')} |\n"
            f"| 의견 | {ai.get('comment', '-')} |"
        )
    return {"section_approval": section}


def node_assemble(state: ReportState) -> dict:
    """모든 섹션을 합쳐서 최종 Markdown 완성"""
    sections = [
        state.get("section_header", ""),
        state.get("section_summary", ""),
        state.get("section_diffusion", ""),
        state.get("section_cause", ""),
        state.get("section_actions", ""),
        state.get("section_approval", ""),
    ]
    final_report = "\n\n".join(s for s in sections if s)
    return {"final_report": final_report}


def node_save(state: ReportState) -> dict:
    """Markdown + JSON 파일로 저장"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"report_{state.get('process_name', 'unknown')}_{timestamp}"

    md_path = REPORTS_DIR / f"{base}.md"
    md_path.write_text(state.get("final_report", ""), encoding="utf-8")

    report_json = {
        "meta": {
            "process_name": state.get("process_name", "-"),
            "severity": state.get("severity", "-"),
            "detected_at": state.get("detected_at", "-"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "bottleneck_info":      state.get("bottleneck_info", {}),
        "fab_kpi":              state.get("fab_kpi", {}),
        "bottleneck_trend":     state.get("bottleneck_trend", []),
        "feature_trend":        state.get("feature_trend", []),
        "shap_analysis":        state.get("shap_analysis", {}),
        "tool_status":          state.get("tool_status", []),
        "affected_lots_detail": state.get("affected_lots_detail", []),
        "diffusion_analysis":   state.get("diffusion_analysis", {}),
        "cause_analysis":       state.get("cause_analysis", []),
        "action_effects":       state.get("action_effects", []),
        "recommendation":       state.get("recommendation", {}),
        "approval_info":        state.get("approval_info", {}),
        "report_sections": {
            "header":    state.get("section_header", ""),
            "summary":   state.get("section_summary", ""),
            "diffusion": state.get("section_diffusion", ""),
            "cause":     state.get("section_cause", ""),
            "actions":   state.get("section_actions", ""),
            "approval":  state.get("section_approval", ""),
        },
        "full_markdown": state.get("final_report", ""),
    }
    json_path = REPORTS_DIR / f"{base}.json"
    json_path.write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"보고서 저장 완료: {md_path}")
    print(f"JSON 저장 완료:   {json_path}")
    return {"output_path": str(md_path), "json_output_path": str(json_path)}


# ── Graph 구성 ─────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ReportState)

    g.add_node("header",    node_write_header)
    g.add_node("summary",   write_summary)
    g.add_node("diffusion", write_diffusion)
    g.add_node("cause",     write_cause)
    g.add_node("actions",   write_actions)
    g.add_node("approval",  node_write_approval)
    g.add_node("assemble",  node_assemble)
    g.add_node("save",      node_save)

    g.set_entry_point("header")
    g.add_edge("header",    "summary")
    g.add_edge("summary",   "diffusion")
    g.add_edge("diffusion", "cause")
    g.add_edge("cause",     "actions")
    g.add_edge("actions",   "approval")
    g.add_edge("approval",  "assemble")
    g.add_edge("assemble",  "save")
    g.add_edge("save",      END)

    return g.compile()


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def run_report_agent(input_data: dict) -> dict:
    graph = build_graph()
    result = graph.invoke(input_data)
    return {
        "final_report":     result["final_report"],
        "output_path":      result["output_path"],
        "json_output_path": result["json_output_path"],
    }


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "mock_report_input.json"
    )

    if not input_path.is_file():
        print(f"입력 파일을 찾을 수 없습니다: {input_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    result = run_report_agent(data)
    print("\n" + "=" * 60)
    print(result["final_report"])
