#!/usr/bin/env python3
"""
비교분석 Agent (Agent 05) — LangGraph 기반

입력: mock_compare_input.json 형태의 dict
출력: compare_out/ 폴더에 JSON 저장 (action_effects + recommendation + approval_info)
      → report_agent 입력과 직접 병합 가능

실행:
    python -m agents.compare_agent.node agents/compare_agent/mock_compare_input.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.compare_agent.scorer import rank_score

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
COMPARE_OUT_DIR = _ROOT / "compare_agent_out"

load_dotenv(_ROOT / ".env")

# ── LLM 설정 ──────────────────────────────────────────────────────────────────
_MODEL = "gpt-4o-mini"
_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                "  프로젝트 루트의 .env 파일에 OPENAI_API_KEY를 추가하세요."
            )
        _llm = ChatOpenAI(model=_MODEL, api_key=api_key, temperature=0.2, max_tokens=1000)
    return _llm


def _llm_write(system_prompt: str, user_prompt: str) -> str:
    response = _get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content.strip()


# ── State 정의 ────────────────────────────────────────────────────────────────

class CompareState(TypedDict):
    # ── 입력 (Agent 04로부터 받는 데이터) ──
    process_name: str
    severity: str
    snapshot_time: float
    t0: float
    horizon_min: float
    bottleneck_info: dict
    tool_status: list
    affected_lots_detail: list
    action_candidates: list       # [{label, action_kind, description, kpi_delta, ...}]

    # ── 계산 결과 ──
    scored_actions: list          # [{label, score, rank, badge, ...}]
    recommendation_text: str      # LLM 생성 추천 근거

    # ── 출력 (Agent 06 호환) ──
    action_effects: list          # report_agent 형식 그대로
    recommendation: dict          # {action_label, action_kind, reason}
    hitl_prompt: str              # 관리자에게 보여줄 텍스트
    approval_info: dict           # 승인/반려 결과
    output_path: str
    json_output_path: str


# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────

_SYS = """[역할]
당신은 반도체 FAB 공정 병목 대응 의사결정을 지원하는 AI입니다.
여러 대응안(A/B/C)의 시뮬레이션 KPI 비교 결과를 바탕으로,
공정 관리자가 즉시 판단할 수 있는 추천 근거를 작성합니다.

[독자]
- 반도체 FAB 공정 관리자, 생산 엔지니어
- WIP · Q-time · CQT · CR · REQUEUE_TOOL · LOT_HOLD 등 FAB 용어에 익숙함

[작성 원칙]
1. 반드시 한국어로 작성합니다. FAB 용어(REQUEUE_TOOL 등)는 원문 유지.
2. 제공된 수치는 반드시 그대로 사용합니다. 반올림·단위 변환 금지.
3. 데이터에 없는 내용은 절대 추측하지 않습니다.
4. 문체는 "~됨", "~함", "~임"으로 통일합니다.
5. 2~3문장 이내로 작성합니다."""


# ── Node 함수들 ───────────────────────────────────────────────────────────────

def node_validate_input(state: CompareState) -> dict:
    """입력 파싱 및 필드 검증 — LLM 불필요"""
    candidates = state.get("action_candidates")
    if not candidates:
        raise ValueError("action_candidates가 비어 있거나 누락되었습니다.")
    required = {"label", "action_kind", "description", "kpi_delta", "simulation_confidence"}
    for c in candidates:
        missing = required - set(c.keys())
        if missing:
            raise ValueError(f"action_candidates[{c.get('label','?')}] 누락 필드: {missing}")
    return {}


def node_llm_recommend(state: CompareState) -> dict:
    """LLM — 추천 근거 2~3문장 생성"""
    scored_map = {s["label"]: s for s in state["scored_actions"]}
    top = min(state["scored_actions"], key=lambda x: x["rank"])
    top_candidate = next(c for c in state["action_candidates"] if c["label"] == top["label"])

    lines = []
    for c in state["action_candidates"]:
        s = scored_map[c["label"]]
        kd = c["kpi_delta"]
        lines.append(
            f"- {c['label']} ({c['action_kind']}): badge={s['badge']}, "
            f"avg_q_time_delta={kd.get('avg_queue_time_min',0)}분, "
            f"throughput_delta={kd.get('throughput_delta',0)}, "
            f"wip_delta={kd.get('wip_count',0)}, "
            f"신뢰도={c.get('simulation_confidence','-')}"
        )

    baseline = top_candidate.get("baseline_metrics", {})
    whatif   = top_candidate.get("whatif_metrics", {})
    before_after = (
        f"현재(baseline): avg_queue_time={baseline.get('avg_queue_time_min','-')}분, "
        f"WIP={baseline.get('wip_count','-')}, throughput={baseline.get('throughput_24h','-')}\n"
        f"{top['label']} 적용 후(시뮬레이션 예측): avg_queue_time={whatif.get('avg_queue_time_min','-')}분, "
        f"WIP={whatif.get('wip_count','-')}, throughput={whatif.get('throughput_24h','-')}"
    )

    prompt = f"""다음은 FAB 병목 대응안 비교 결과입니다. 순위는 avg_queue_time 개선량(병목 해소 효과) 기준으로 결정됐습니다.

[목적]
이 설명문은 공정 관리자가 대응안 승인 여부를 결정하기 위해 읽습니다.
관리자가 즉시 판단할 수 있도록 핵심 수치 중심으로 작성하세요.

공정: {state['process_name']}  심각도: {state['severity']}
현재 WIP: {state['bottleneck_info'].get('wip_count','-')}  avg_queue_time: {state['bottleneck_info'].get('avg_queue_time_min','-')}분

대응안 비교:
{chr(10).join(lines)}

AI 추천 대응안: {top['label']} — {top_candidate['description']}
{before_after}

[출력 구조 — 반드시 이 순서로 정확히 2문장만 작성]
1문장: {top['label']}을 추천하는 핵심 근거. 반드시 before→after 수치(avg_queue_time)를 포함하고, 시뮬레이션 예측값임을 명시할 것.
2문장: 나머지 대응안을 선택하지 않는 이유. 반드시 수치를 비교하여 차이를 명시할 것.

[출력 예시 — 형식 참고용, 내용은 위 데이터 기준으로 작성]
"A(REQUEUE_TOOL) 적용 시 시뮬레이션 기준 avg_queue_time이 187분 → 142분으로 45분 감소가 예측되어 병목 해소 효과가 가장 큼.
B는 10분 감소에 그치며 throughput이 -1로 생산성도 저하되고, C는 22분 감소로 A의 절반 수준에 불과함."

[금지]
- "가장 효과적입니다", "적절합니다", "좋은 선택입니다" 등 수치 없는 모호한 표현
- 시뮬레이션 예측값을 확정된 사실처럼 서술하는 것
- 데이터에 없는 내용 추측
- 3문장 이상 작성"""

    text = _llm_write(_SYS, prompt)
    return {"recommendation_text": text}


def node_format_output(state: CompareState) -> dict:
    """action_effects + recommendation + hitl_prompt 생성 — LLM 불필요"""
    scored_map = {s["label"]: s for s in state["scored_actions"]}
    top_label = next(
        (s["label"] for s in sorted(state["scored_actions"], key=lambda x: x["rank"])),
        None
    )

    action_effects = []
    for c in state["action_candidates"]:
        s = scored_map[c["label"]]
        label_display = f"{c['label']} {s['badge']}" if s["label"] == top_label else c["label"]
        action_effects.append({
            "label": label_display,
            "action_kind": c["action_kind"],
            "description": c["description"],
            "simulation_confidence": c.get("simulation_confidence"),
            "kpi_delta": c["kpi_delta"],
        })

    top_candidate = next(
        (c for c in state["action_candidates"] if c["label"] == top_label),
        state["action_candidates"][0],
    )
    recommendation = {
        "action_label": top_label,
        "action_kind": top_candidate["action_kind"],
        "reason": state["recommendation_text"],
    }

    lines = [
        "",
        "=" * 58,
        f"[비교분석 완료] {state['process_name']} 관리자 승인 필요",
        "=" * 58,
        f"AI 추천 대응안: {top_label} — {top_candidate['description']}",
        f"추천 근거: {state['recommendation_text']}",
        "",
        "[ 대응안 요약 ]",
    ]
    for c in state["action_candidates"]:
        s = scored_map[c["label"]]
        kd = c["kpi_delta"]
        lines.append(
            f"  {c['label']} {s['badge']:10s}"
            f"  q_time {kd.get('avg_queue_time_min',0):+.0f}분"
            f"  WIP {kd.get('wip_count',0):+d}"
            f"  TH {kd.get('throughput_delta',0):+d}"
            f"  신뢰도 {int(c.get('simulation_confidence',0)*100)}%"
        )
    lines += ["", "승인할 대응안을 입력하세요 (A / B / C / 반려): ", ""]

    return {
        "action_effects": action_effects,
        "recommendation": recommendation,
        "hitl_prompt": "\n".join(lines),
    }


def node_hitl(state: CompareState) -> dict:
    """HITL — 터미널에서 관리자 입력 받기"""
    print(state["hitl_prompt"])

    valid = {c["label"] for c in state["action_candidates"]} | {"반려"}
    while True:
        raw = input(">>> ").strip().upper()
        if raw in valid or raw == "반려":
            break
        print(f"유효하지 않은 입력입니다. ({' / '.join(sorted(valid))}) 중 선택하세요.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if raw == "반려":
        approval_info = {
            "status": "반려",
            "approved_by": None,
            "approved_role": None,
            "approved_at": now,
            "comment": None,
            "rejection_reason": input("반려 사유를 입력하세요 >>> ").strip(),
        }
    else:
        approver = input("승인자 이름을 입력하세요 >>> ").strip()
        role     = input("직책을 입력하세요 >>> ").strip()
        comment  = input("의견 (없으면 Enter) >>> ").strip() or "즉시 적용 승인"
        approval_info = {
            "status": "승인",
            "approved_by": approver,
            "approved_role": role,
            "approved_at": now,
            "comment": comment,
            "rejection_reason": None,
        }
    return {"approval_info": approval_info}


def node_save(state: CompareState) -> dict:
    """결과 JSON 저장 — report_agent 입력으로 바로 병합 가능"""
    COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"compare_{state['process_name']}_{ts}"
    json_path = COMPARE_OUT_DIR / f"{name}.json"

    output = {
        "process_name": state["process_name"],
        "severity": state["severity"],
        "snapshot_time": state["snapshot_time"],
        "action_effects": state["action_effects"],
        "recommendation": state["recommendation"],
        "approval_info": state["approval_info"],
    }
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n[저장 완료] {json_path}")
    return {"json_output_path": str(json_path)}


# ── Graph 구성 ─────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(CompareState)
    g.add_node("validate_input", node_validate_input)
    g.add_node("rank_score",     rank_score)
    g.add_node("llm_recommend",  node_llm_recommend)
    g.add_node("format_output",  node_format_output)
    g.add_node("hitl",           node_hitl)
    g.add_node("save",           node_save)

    g.set_entry_point("validate_input")
    g.add_edge("validate_input", "rank_score")
    g.add_edge("rank_score",     "llm_recommend")
    g.add_edge("llm_recommend",  "format_output")
    g.add_edge("format_output",  "hitl")
    g.add_edge("hitl",           "save")
    g.add_edge("save",           END)
    return g


# ── 진입점 ────────────────────────────────────────────────────────────────────

def run(input_data: dict) -> dict:
    graph = build_graph().compile()
    return graph.invoke(input_data)


def main() -> int:
    if len(sys.argv) < 2:
        print(f"사용법: python {sys.argv[0]} <input_json>", file=sys.stderr)
        return 1
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
