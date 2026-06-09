#!/usr/bin/env python3
"""비교분석 Agent (Agent 05) — PipelineState 통합 노드."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents import config

if TYPE_CHECKING:
    from agents.state import PipelineState

_ROOT = Path(__file__).parent.parent.parent
COMPARE_OUT_DIR = _ROOT / "compare_agent_out"

load_dotenv(_ROOT / ".env")

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
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=1000,
        )
    return _llm


def _llm_write(system_prompt: str, user_prompt: str) -> str:
    response = _get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content.strip()


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


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _build_action_candidates(verified_candidates: list[dict]) -> list[dict]:
    """verified_candidates → compare action_candidates 변환."""
    from agents.verification_agent.sim_executor import HORIZON_MIN  # noqa: F401

    candidates = []
    for vc in verified_candidates:
        action_rows = vc.get("action_rows", [])
        action_kind = action_rows[0]["action_kind"] if action_rows else "UNKNOWN"
        kpi_stats = vc.get("kpi_stats", {})
        p_val = vc.get("paired_t_p")
        confidence = (
            round(max(0.0, min(1.0, 1.0 - float(p_val))), 4)
            if p_val is not None else 0.5
        )
        candidates.append({
            "label": vc["label"],
            "action_kind": action_kind,
            "description": vc.get("name", ""),
            "simulation_confidence": confidence,
            "kpi_delta": {
                "avg_queue_time_min": kpi_stats.get("q_time_min", {}).get("mean_delta", 0.0),
                "wip_count": int(round(kpi_stats.get("wip", {}).get("mean_delta", 0.0))),
                "throughput_delta": 0,
            },
            "kpi_stats": kpi_stats,
            "verdict": vc.get("verdict", "unknown"),
            "paired_n": vc.get("paired_n", 0),
            "paired_t_p": p_val,
        })
    return candidates


def _build_bottleneck_info(toolgroup: str, alert, kpi) -> dict:
    info: dict = {
        "tool_group": toolgroup,
        "risk_score": round(float(alert.composite_score), 4),
    }
    if kpi:
        info.update({
            "wip_count": int(kpi.wip),
            "avg_queue_time_min": round(float(kpi.q_time_min), 2),
            "wait_ratio": round(float(kpi.wait_ratio), 4),
            "available_tool_ratio": round(float(kpi.available_tool_ratio), 4),
            "utilization_avg": round(float(kpi.utilization_avg), 4),
        })
    return info


def _rank_candidates(candidates: list[dict]) -> list[dict]:
    def rank_key(c):
        kd = c["kpi_delta"]
        return (
            -kd.get("avg_queue_time_min", 0.0),
            kd.get("throughput_delta", 0.0),
            -kd.get("wip_count", 0.0),
        )
    sorted_c = sorted(candidates, key=rank_key, reverse=True)
    rank_map = {c["label"]: i + 1 for i, c in enumerate(sorted_c)}
    return [
        {
            "label": c["label"],
            "rank": rank_map[c["label"]],
            "badge": "✅ AI추천" if rank_map[c["label"]] == 1 else "⚪ 선택가능",
        }
        for c in candidates
    ]


# ── Pipeline 노드 ─────────────────────────────────────────────────────────────

def compare_rank(state: "PipelineState") -> dict:
    """Agent 5-1: 입력 변환 + 대응안 순위 결정 (LLM 불필요)."""
    from agents.logger import get_logger
    from agents.verification_agent.sim_executor import HORIZON_MIN
    _log = get_logger(__name__)

    verification_results = state.get("verification_results", [])
    if not verification_results:
        _log.info("[Compare] verification_results 없음 — 스킵")
        return {"compare_inputs": []}

    alerts = state.get("alerts", [])
    kpi_snapshot = state.get("kpi_snapshot", [])
    alert_map = {a.toolgroup: a for a in alerts}
    kpi_map = {k.toolgroup: k for k in kpi_snapshot}
    t0 = kpi_snapshot[0].snapshot_time if kpi_snapshot else 0.0

    compare_inputs: list[dict] = []

    # GlobalSolutionPlan 포맷(plan_id 키 존재): Plan A/B를 하나의 그룹으로 합산
    global_plan_groups = [g for g in verification_results if "plan_id" in g]
    per_tg_groups = [g for g in verification_results if "plan_id" not in g]

    if global_plan_groups:
        from agents.schemas.alert import SeverityLevel
        critical_alerts = [a for a in alerts if a.severity == SeverityLevel.CRITICAL]
        anchor_alert = (
            max(critical_alerts, key=lambda a: a.composite_score)
            if critical_alerts else None
        )
        if anchor_alert is None:
            _log.warning("[Compare] GlobalSolutionPlan: CRITICAL anchor alert 없음 — 스킵")
        else:
            anchor_tg = anchor_alert.toolgroup
            target_tgs = global_plan_groups[0].get("target_toolgroups", [])
            all_verified: list[dict] = []
            for g in global_plan_groups:
                all_verified.extend(g.get("verified_candidates", []))
            candidates = _build_action_candidates(all_verified)
            scored = _rank_candidates(candidates)
            compare_inputs.append({
                "toolgroup": anchor_tg,
                "process_name": f"글로벌 플랜 A/B ({', '.join(target_tgs[:3])}{'...' if len(target_tgs) > 3 else ''})",
                "severity": anchor_alert.severity.value,
                "snapshot_time": float(global_plan_groups[0].get("snapshot_time", t0)),
                "t0": float(t0),
                "horizon_min": HORIZON_MIN,
                "bottleneck_info": _build_bottleneck_info(anchor_tg, anchor_alert, kpi_map.get(anchor_tg)),
                "action_candidates": candidates,
                "scored_actions": scored,
            })

    # 기존 per-TG 포맷
    for result_group in per_tg_groups:
        verified_candidates = result_group.get("verified_candidates", [])
        if not verified_candidates:
            _log.warning("[Compare] verified_candidates 없음 — 스킵")
            continue

        tg = result_group["toolgroup"]
        alert = alert_map.get(tg)
        if alert is None:
            _log.warning(f"[Compare] {tg}: alert 없음 — 스킵")
            continue

        candidates = _build_action_candidates(verified_candidates)
        scored = _rank_candidates(candidates)

        compare_inputs.append({
            "toolgroup": tg,
            "process_name": tg,
            "severity": result_group.get("severity", alert.severity.value),
            "snapshot_time": float(result_group.get("snapshot_time", t0)),
            "t0": float(t0),
            "horizon_min": HORIZON_MIN,
            "bottleneck_info": _build_bottleneck_info(tg, alert, kpi_map.get(tg)),
            "action_candidates": candidates,
            "scored_actions": scored,
        })

    _log.info(f"[Compare] rank 완료 — {len(compare_inputs)}개 공정")
    return {"compare_inputs": compare_inputs}


def compare_llm(state: "PipelineState") -> dict:
    """Agent 5-2: LLM 추천 근거 생성 + 출력 포맷 구성."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    compare_inputs = state.get("compare_inputs", [])
    if not compare_inputs:
        return {"compare_formatted": []}

    compare_formatted: list[dict] = []

    for ci in compare_inputs:
        tg = ci["toolgroup"]
        candidates = ci["action_candidates"]
        scored_actions = ci["scored_actions"]
        scored_map = {s["label"]: s for s in scored_actions}
        top = min(scored_actions, key=lambda x: x["rank"]) if scored_actions else None
        top_label = top["label"] if top else (candidates[0]["label"] if candidates else None)
        top_candidate = next(
            (c for c in candidates if c["label"] == top_label),
            candidates[0] if candidates else {},
        )

        # LLM 추천 근거
        lines = []
        for c in candidates:
            s = scored_map.get(c["label"], {})
            kd = c["kpi_delta"]
            lines.append(
                f"- {c['label']} ({c['action_kind']}): badge={s.get('badge', '-')}, "
                f"avg_q_time_delta={kd.get('avg_queue_time_min', 0)}분, "
                f"throughput_delta={kd.get('throughput_delta', 0)}, "
                f"wip_delta={kd.get('wip_count', 0)}, "
                f"신뢰도={c.get('simulation_confidence', '-')}"
            )

        prompt = (
            f"다음은 FAB 병목 대응안 비교 결과입니다. 순위는 avg_queue_time 개선량 기준입니다.\n\n"
            f"공정: {ci['process_name']}  심각도: {ci['severity']}\n"
            f"현재 WIP: {ci['bottleneck_info'].get('wip_count', '-')}  "
            f"avg_queue_time: {ci['bottleneck_info'].get('avg_queue_time_min', '-')}분\n\n"
            f"대응안 비교:\n{chr(10).join(lines)}\n\n"
            f"AI 추천 대응안: {top_label} — {top_candidate.get('description', '')}\n\n"
            f"[출력 구조 — 반드시 이 순서로 정확히 2문장만 작성]\n"
            f"1문장: {top_label}을 추천하는 핵심 근거. 시뮬레이션 예측값임을 명시할 것.\n"
            f"2문장: 나머지 대응안을 선택하지 않는 이유. 수치를 비교하여 명시할 것."
        )

        try:
            rec_text = _llm_write(_SYS, prompt)
        except Exception as e:
            _log.error(f"[Compare] {tg} LLM 실패: {e}")
            rec_text = f"{top_label} 대응안이 KPI 개선 효과가 가장 큼."

        # action_effects 포맷
        action_effects = []
        for c in candidates:
            s = scored_map.get(c["label"], {})
            label_display = (
                f"{c['label']} {s.get('badge', '')}" if c["label"] == top_label else c["label"]
            )
            action_effects.append({
                "label": label_display,
                "action_kind": c["action_kind"],
                "description": c["description"],
                "simulation_confidence": c.get("simulation_confidence"),
                "kpi_delta": c["kpi_delta"],
            })

        recommendation = {
            "action_label": top_label,
            "action_kind": top_candidate.get("action_kind", ""),
            "reason": rec_text,
        }

        # HITL 프롬프트
        hitl_lines = [
            "",
            "=" * 58,
            f"[비교분석 완료] {ci['process_name']} 관리자 승인 필요",
            "=" * 58,
            f"AI 추천 대응안: {top_label} — {top_candidate.get('description', '')}",
            f"추천 근거: {rec_text}",
            "",
            "[ 대응안 요약 ]",
        ]
        for c in candidates:
            s = scored_map.get(c["label"], {})
            kd = c["kpi_delta"]
            hitl_lines.append(
                f"  {c['label']} {s.get('badge', ''):10s}"
                f"  q_time {kd.get('avg_queue_time_min', 0):+.0f}분"
                f"  WIP {kd.get('wip_count', 0):+d}"
                f"  TH {kd.get('throughput_delta', 0):+d}"
                f"  신뢰도 {int(c.get('simulation_confidence', 0) * 100)}%"
            )
        hitl_lines += ["", "승인할 대응안을 입력하세요 (A / B / C / 반려): ", ""]

        compare_formatted.append({
            "toolgroup": tg,
            "process_name": ci["process_name"],
            "severity": ci["severity"],
            "snapshot_time": ci["snapshot_time"],
            "action_candidates": candidates,
            "action_effects": action_effects,
            "recommendation": recommendation,
            "hitl_prompt": "\n".join(hitl_lines),
        })

        _log.info(f"[Compare] {tg} LLM 완료")

    return {"compare_formatted": compare_formatted}


def compare_hitl(state: "PipelineState") -> dict:
    """Agent 5-3: 관리자 승인(HITL) — Webhook / Auto / Terminal 모드."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    compare_formatted = state.get("compare_formatted", [])
    if not compare_formatted:
        return {"compare_results": []}

    webhook_mode = os.environ.get("WEBHOOK_MODE", "").lower() in ("1", "true", "yes")
    auto_approve  = os.environ.get("AUTO_APPROVE", "").lower() in ("1", "true", "yes")

    if webhook_mode:
        return _hitl_webhook(state, compare_formatted, _log)
    elif auto_approve:
        return _hitl_auto(compare_formatted, _log)
    else:
        return _hitl_terminal(compare_formatted, _log)


# ── HITL 모드별 구현 ──────────────────────────────────────────────────────────

def _hitl_webhook(state: "PipelineState", compare_formatted: list[dict], _log) -> dict:
    """Webhook 모드: 상태 직렬화 → hitl_pending/ 저장 → Phase 1 종료."""
    import uuid as _uuid
    from datetime import datetime as _dt, timezone as _tz

    hitl_token = str(_uuid.uuid4())

    pending = {
        "hitl_token": hitl_token,
        "created_at": _dt.now(_tz.utc).isoformat(),
        "compare_formatted": compare_formatted,
        "alerts": [a.model_dump() for a in state.get("alerts", [])],
        "kpi_snapshot": [k.model_dump() for k in state.get("kpi_snapshot", [])],
        "prev_kpi_snapshot": [k.model_dump() for k in state.get("prev_kpi_snapshot", [])],
        "cause_reports": [r.model_dump() for r in state.get("cause_reports", [])],
        "cascade_report": state.get("cascade_report"),
        "solution_candidates": state.get("solution_candidates", []),
    }

    pending_dir = _ROOT / "hitl_pending"
    pending_dir.mkdir(exist_ok=True)
    pending_path = pending_dir / f"{hitl_token}.json"
    pending_path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info(f"[HITL-Webhook] 상태 저장: {pending_path.name}")

    _notify_spring_boot(hitl_token, compare_formatted, _log)

    print(f"\n{'='*60}")
    print(f"[Phase 1 완료] HITL 대기 중")
    print(f"  token : {hitl_token}")
    print(f"  파일  : {pending_path}")
    print(f"  Phase 2 실행: python run_phase2.py --token {hitl_token}")
    print(f"{'='*60}\n")

    return {"compare_results": [], "hitl_approved": None, "hitl_token": hitl_token}


def _notify_spring_boot(hitl_token: str, compare_formatted: list[dict], _log) -> None:
    """Spring Boot에 HITL 대기 요청 전송 (실패해도 Phase 1 계속 진행)."""
    import urllib.request as _req
    import urllib.error as _err
    from datetime import datetime as _dt, timezone as _tz

    spring_url = os.environ.get("SPRING_BOOT_URL", "").strip()
    if not spring_url:
        _log.debug("[HITL-Webhook] SPRING_BOOT_URL 미설정 — Spring Boot 알림 스킵")
        return

    internal_token = os.environ.get("INTERNAL_API_TOKEN", "")
    payload = json.dumps({
        "hitlToken": hitl_token,
        "toolgroups": [cf["toolgroup"] for cf in compare_formatted],
        "severity": compare_formatted[0]["severity"] if compare_formatted else "CRITICAL",
        "compareSummary": {
            "recommendations": [
                {
                    "toolgroup": cf["toolgroup"],
                    "severity": cf["severity"],
                    "recommendedAction": cf["recommendation"].get("action_label"),
                    "reason": cf["recommendation"].get("reason"),
                    "actionEffects": cf.get("action_effects", []),
                }
                for cf in compare_formatted
            ]
        },
    }, ensure_ascii=False).encode("utf-8")

    request = _req.Request(
        f"{spring_url}/api/internal/agent-hitl/pending",
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-Internal-Token": internal_token,
            "X-Event-Timestamp": _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        method="POST",
    )
    try:
        with _req.urlopen(request, timeout=10) as resp:
            _log.info(f"[HITL-Webhook] Spring Boot 등록 완료: HTTP {resp.status}")
    except _err.HTTPError as e:
        _log.warning(f"[HITL-Webhook] Spring Boot 등록 실패: HTTP {e.code} — Phase 1은 계속 진행")
    except Exception as e:
        _log.warning(f"[HITL-Webhook] Spring Boot 연결 실패: {e} — Phase 1은 계속 진행")


def _hitl_auto(compare_formatted: list[dict], _log) -> dict:
    """AUTO_APPROVE 모드: AI 추천 대응안 자동 승인."""
    COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_results: list[dict] = []

    for cf in compare_formatted:
        tg = cf["toolgroup"]
        print(cf["hitl_prompt"])

        rec_label = cf["recommendation"].get("action_label", "A")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"[AUTO-APPROVE] AI 추천 대응안 자동 승인: {rec_label}")
        approval_info = {
            "status": "승인",
            "approved_by": "AUTO",
            "approved_role": "SYSTEM",
            "approved_at": now,
            "comment": "자동 승인 (AUTO_APPROVE 모드)",
            "rejection_reason": None,
        }
        compare_results.append(_build_compare_result(cf, approval_info))
        _log.info(f"[Compare] {tg} AUTO-APPROVE 완료")

    _log.info(f"[Compare] 완료 — {len(compare_results)}개 공정")
    return {"compare_results": compare_results}


def _hitl_terminal(compare_formatted: list[dict], _log) -> dict:
    """Terminal 모드: 터미널 입력으로 승인/반려."""
    COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_results: list[dict] = []

    for cf in compare_formatted:
        tg = cf["toolgroup"]
        print(cf["hitl_prompt"])

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        valid = {c["label"] for c in cf["action_candidates"]} | {"반려"}
        while True:
            raw = input(">>> ").strip().upper()
            if raw in valid:
                break
            print(f"유효하지 않은 입력입니다. ({' / '.join(sorted(valid))}) 중 선택하세요.")

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
            role = input("직책을 입력하세요 >>> ").strip()
            comment = input("의견 (없으면 Enter) >>> ").strip() or "즉시 적용 승인"
            approval_info = {
                "status": "승인",
                "approved_by": approver,
                "approved_role": role,
                "approved_at": now,
                "comment": comment,
                "rejection_reason": None,
            }

        compare_results.append(_build_compare_result(cf, approval_info))
        _log.info(f"[Compare] {tg} HITL 완료")

    _log.info(f"[Compare] 완료 — {len(compare_results)}개 공정")
    return {"compare_results": compare_results}


def _build_compare_result(cf: dict, approval_info: dict) -> dict:
    """compare_formatted + approval_info → compare_result (파일 저장 포함)."""
    tg = cf["toolgroup"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = COMPARE_OUT_DIR / f"compare_{tg}_{ts}.json"
    json_path.write_text(
        json.dumps({
            "process_name": cf["process_name"],
            "severity": cf["severity"],
            "snapshot_time": cf["snapshot_time"],
            "action_effects": cf["action_effects"],
            "recommendation": cf["recommendation"],
            "approval_info": approval_info,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[저장 완료] {json_path}")
    return {
        "toolgroup": tg,
        "json_output_path": str(json_path),
        "recommendation": cf["recommendation"],
        "approval_info": approval_info,
        "action_effects": cf["action_effects"],
    }
