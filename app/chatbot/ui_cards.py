"""도구 결과 텍스트 포매터 + Generative UI 카드 빌더(프론트 렌더용 구조화 데이터).
숫자는 항상 DB 도구 실데이터에서 옴 — LLM은 어떤 카드를 보일지만 고른다(하이브리드)."""

import json
import re

_TG_METRIC_LABELS = {"util": "가동률", "wip": "WIP", "qtime": "Q-time", "wait": "대기", "bottleneck": "병목확률"}


def _pct(value) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def _num(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct_change(start: float, end: float) -> str:
    if start == 0:
        return "계산불가"
    return f"{((end - start) / abs(start)) * 100:+.1f}%"


def _fab_status_lookup(live_status: str | None, area: str = "") -> str:
    """전달받은 실시간 현황 텍스트에서 전체/특정 구역 현황을 추려 반환."""
    if not live_status or not live_status.strip():
        return "현재 실시간 현황 데이터가 없습니다."
    if not area or not area.strip():
        return live_status
    needle = area.strip().lower()
    lines = [
        line for line in live_status.splitlines()
        if line.startswith("전체") or line.startswith("[") or needle in line.lower()
    ]
    matched = [line for line in lines if needle in line.lower()]
    if not matched:
        return f"'{area}' 구역을 현재 현황에서 찾지 못했습니다. 전체 현황:\n{live_status}"
    return "\n".join(lines)

def _fmt_trend(rows: list, area: str) -> str:
    if not rows:
        return f"'{area or '전체'}' 추세 데이터가 없습니다."
    by_area: dict[str, list] = {}
    for r in rows:
        by_area.setdefault(r["label"], []).append(r)
    blocks = []
    for name, recs in by_area.items():
        recs = sorted(recs, key=lambda r: r["bucket"])
        pts = [
            f"{r['bucket']:%H:%M} WIP {_num(r['wip']):.0f}·가동률 {_pct(r['util'])}·Q-time {_num(r['qtime']):.0f}m"
            for r in recs
        ]
        first, last = recs[0], recs[-1]
        first_wip = _num(first["wip"])
        last_wip = _num(last["wip"])
        avg_wip = _avg([_num(r["wip"]) for r in recs])
        avg_util = _avg([_num(r["util"]) for r in recs]) * 100
        avg_qtime = _avg([_num(r["qtime"]) for r in recs])
        summary = (
            f"계산 요약: 최신 WIP {last_wip:.0f}, 평균 WIP {avg_wip:.1f}, "
            f"WIP 변화 {last_wip - first_wip:+.0f}, WIP 변화율 {_pct_change(first_wip, last_wip)}, "
            f"평균 가동률 {avg_util:.1f}%, 평균 Q-time {avg_qtime:.1f}분."
        )
        blocks.append(f"[{name}] " + " → ".join(pts) + "\n" + summary)
    return "\n".join(blocks)


def _fmt_lot(rows: list, area: str) -> str:
    if not rows:
        return f"'{area or '전체'}' WIP/대기 데이터가 없습니다."
    total_wip = sum(int(_num(r["wip_count"])) for r in rows)
    avg_util = _avg([_num(r["utilization_rate"]) for r in rows]) * 100
    avg_qtime = _avg([_num(r["avg_qtime_min"]) for r in rows])
    top = max(rows, key=lambda r: _num(r["wip_count"]))
    summary = (
        f"계산 요약(조회된 {len(rows)}개 TG 기준): WIP 합계 {total_wip} Lot, "
        f"평균 가동률 {avg_util:.1f}%, 평균 Q-time {avg_qtime:.1f}분, "
        f"최대 WIP {top['area_name']}/{top['tg_code']} {int(_num(top['wip_count']))} Lot."
    )
    lines = [
        f"- {r['area_name']}/{r['tg_code']}: WIP {r['wip_count'] or 0} Lot, 대기 {float(r['wait_ratio'] or 0):.1f}, "
        f"Q-time {float(r['avg_qtime_min'] or 0):.0f}분, 가동률 {_pct(r['utilization_rate'])}"
        for r in rows
    ]
    return "구역별 WIP·대기 현황(개별 Lot 추적 데이터는 없음):\n" + summary + "\n" + "\n".join(lines)


def _fmt_top_tgs(rows: list, area: str, metric: str, order: str) -> str:
    label = _TG_METRIC_LABELS.get(metric, metric)
    if not rows:
        return f"'{area or '전체'}' 툴그룹 현황 데이터가 없습니다."
    total_wip = sum(int(_num(r["wip_count"])) for r in rows)
    avg_util = _avg([_num(r["utilization_rate"]) for r in rows]) * 100
    avg_qtime = _avg([_num(r["avg_qtime_min"]) for r in rows])
    top = rows[0]
    summary = (
        f"계산 요약(조회된 {len(rows)}개 TG 기준): WIP 합계 {total_wip} Lot, "
        f"평균 가동률 {avg_util:.1f}%, 평균 Q-time {avg_qtime:.1f}분, "
        f"1위 {top['area_name']}/{top['tg_code']}."
    )
    lines = [
        f"{i}. {r['area_name']}/{r['tg_code']} ({r['tg_name']}): 가동률 {_pct(r['utilization_rate'])}, "
        f"WIP {int(r['wip_count'] or 0)} Lot, Q-time {float(r['avg_qtime_min'] or 0):.0f}분, "
        f"가용률 {_pct(r['available_tool_ratio'])}, 위험 {r['risk_grade'] or '-'}"
        for i, r in enumerate(rows, 1)
    ]
    return f"툴그룹 {label} {'낮은' if order == 'asc' else '높은'} 순:\n" + summary + "\n" + "\n".join(lines)


def _fmt_tools(rows: list, tg: str) -> str:
    if not rows:
        return f"'{tg or '전체'}' 툴 현황 데이터가 없습니다."
    head = f"[{rows[0]['area_name']}/{rows[0]['tg_code']}] " if tg else ""
    total_queue = sum(int(_num(r["queue_lot_count"])) for r in rows)
    avg_util = _avg([_num(r["utilization_rate"]) for r in rows]) * 100
    avg_oee = _avg([_num(r["oee_estimate"]) for r in rows])
    avg_down = _avg([_num(r["down_ratio"]) for r in rows]) * 100
    summary = (
        f"계산 요약(조회된 {len(rows)}개 툴 기준): 대기 합계 {total_queue} Lot, "
        f"평균 가동률 {avg_util:.1f}%, 평균 OEE {avg_oee:.1f}, 평균 다운율 {avg_down:.1f}%."
    )
    lines = [
        f"- {r['tool_code']}: 가동률 {_pct(r['utilization_rate'])}, OEE {float(r['oee_estimate'] or 0):.1f}, "
        f"대기 {int(r['queue_lot_count'] or 0)} Lot, Q-time {float(r['avg_qtime_min'] or 0):.0f}분, "
        f"셋업 {_pct(r['setup_ratio'])}, 다운 {_pct(r['down_ratio'])}"
        for r in rows
    ]
    return head + "개별 툴(설비) 현황:\n" + summary + "\n" + "\n".join(lines)


def _parse_jsonb(value):
    """asyncpg는 JSONB 코덱 미설정 시 str로 반환 → 방어적 파싱(실패 시 None)."""
    if value is None or isinstance(value, list | dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _value(row, key: str):
    try:
        return row[key]
    except (KeyError, TypeError):
        return None


def _first_value(row, *keys: str):
    for key in keys:
        value = _value(row, key)
        if value is not None:
            return value
    return None


def _fmt_delta(value, digits: int = 1, suffix: str = "") -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):+.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return None


def _fmt_case_detail(row, plans: list, hitl: list) -> str:
    lines = [
        f"[{row['area_name']}/{row['tg_code']}] {row['detected_at']:%m-%d %H:%M} 감지 — "
        f"위험 {row['risk_grade']}, 확률 {_pct(row['bottleneck_prob'])}, 상태 {row['status']}"
        + (f" (해결 {row['resolved_at']:%m-%d %H:%M})" if row["resolved_at"] else "")
    ]
    if row["bottleneck_cause_type"]:
        lines.append(f"원인 유형: {row['bottleneck_cause_type']}")
    shap = _parse_jsonb(row["shap_features"]) or []
    if isinstance(shap, list) and shap:
        def _absval(f: dict) -> float:
            try:
                return abs(float(f.get("value", 0)))
            except (TypeError, ValueError):
                return 0.0
        top = sorted((f for f in shap if isinstance(f, dict)), key=_absval, reverse=True)[:3]
        feats = ", ".join(f"{f.get('feature') or f.get('name') or '?'}={f.get('value')}" for f in top)
        if feats:
            lines.append(f"주요 원인 피처(SHAP): {feats}")
    diffusion = _parse_jsonb(row["diffusion_affected_tg_ids"]) or []
    if isinstance(diffusion, list) and diffusion:
        lines.append(f"확산 영향 TG: {len(diffusion)}곳")
    if plans:
        lines.append("대응안:")
        for p in plans:
            metrics = [
                ("가동률", _fmt_delta(_value(p, "est_util_delta"), 2)),
                ("Q-time", _fmt_delta(_first_value(p, "est_q_time_delta", "est_avg_wait_delta"), 1, "분")),
                ("WIP", _fmt_delta(_value(p, "est_wip_delta"), 1, " Lot")),
                ("대기율", _fmt_delta(_value(p, "est_wait_ratio_delta"), 4)),
            ]
            if all(value is None for _, value in metrics):
                metrics = [
                    ("처리량", _fmt_delta(_value(p, "est_throughput_delta"), 1)),
                    ("대기", _fmt_delta(_value(p, "est_avg_wait_delta"), 0, "분")),
                    ("납기", _fmt_delta(_value(p, "est_delivery_compliance_delta"), 1, "%p")),
                ]
            metric_text = ", ".join(f"{label} {value}" for label, value in metrics if value is not None)
            d = f"{p['plan_seq']}. {p['plan_title']}" + (" [선택됨]" if p["selected"] else "")
            if metric_text:
                d += f": {metric_text}"
            if p["validated_at"]:
                actual_metrics = [
                    ("처리량", _fmt_delta(_value(p, "actual_throughput_delta"), 1)),
                    ("대기", _fmt_delta(_value(p, "actual_avg_wait_delta"), 0, "분")),
                ]
                actual_text = ", ".join(f"{label} {value}" for label, value in actual_metrics if value is not None)
                if actual_text:
                    d += f" (실측: {actual_text})"
            lines.append("- " + d)
    if hitl:
        h = hitl[0]
        lines.append(
            f"HITL 결정: {h['decision']} ({h['decided_at']:%m-%d %H:%M})"
            + (f" — {h['comment']}" if h["comment"] else "")
        )
    else:
        lines.append("HITL 결정 이력 없음")
    summary = row["report_summary"] or row["root_cause_text"]
    if summary:
        lines.append(f"리포트 요약: {str(summary)[:300]}")
    return "\n".join(lines)


def _fmt_tool_activity(rows: list, tool: str) -> str:
    if not rows:
        return f"'{tool or '?'}' 설비 활동 데이터가 없습니다."
    pts = [
        f"{r['bucket']:%H:%M} 가동률 {_pct(r['util'])}·대기 {float(r['queue'] or 0):.0f} Lot·"
        f"Q-time {float(r['qtime'] or 0):.0f}m·다운 {_pct(r['down'])}"
        for r in rows
    ]
    return f"[{rows[0]['tool_code']}] " + " → ".join(pts)


def _fmt_cases(rows: list, area: str) -> str:
    if not rows:
        return f"'{area or '전체'}' 최근 병목 케이스가 없습니다."
    lines = [
        f"- {r['detected_at']:%m-%d %H:%M} {r['area_name']}/{r['tg_code']}: "
        f"위험 {r['risk_grade']}, 확률 {_pct(r['bottleneck_prob'])}, 상태 {r['status']}"
        + (f" (해결 {r['resolved_at']:%m-%d %H:%M})" if r["resolved_at"] else "")
        for r in rows
    ]
    return "최근 병목 케이스:\n" + "\n".join(lines)

def _status_card(live_status: str | None) -> dict | None:
    """Spring buildLiveFabContext가 만든 현황 텍스트(형식을 우리가 통제)를 구조화 카드로.
    형식: '전체: 가동률 21.8%, WIP 572 Lot, 설비 가동 440/대기 1071/셋업 0/비가동 51, 가용률 96.5%'
          '- {구역}: WIP n, 평균가동률 u%, 가용률 a%'"""
    if not live_status:
        return None
    overall = None
    m = re.search(
        r"전체:\s*가동률\s*([\d.]+)%,\s*WIP\s*(\d+)\s*Lot,\s*설비 가동\s*(\d+)/대기\s*(\d+)/셋업\s*(\d+)/비가동\s*(\d+),\s*가용률\s*([\d.]+)%",
        live_status,
    )
    if m:
        overall = {
            "util": float(m.group(1)), "wip": int(m.group(2)),
            "run": int(m.group(3)), "idle": int(m.group(4)),
            "setup": int(m.group(5)), "down": int(m.group(6)),
            "avail": float(m.group(7)),
        }
    areas = [
        {"name": a.group(1).strip(), "wip": int(a.group(2)), "util": float(a.group(3)), "avail": float(a.group(4))}
        for a in re.finditer(r"^-\s*(.+?):\s*WIP\s*(\d+),\s*평균가동률\s*([\d.]+)%,\s*가용률\s*([\d.]+)%", live_status, re.M)
    ]
    if overall is None and not areas:
        return None
    measured = re.search(r"기준\s*([^\]]+)\]", live_status)
    return {
        "type": "status",
        "props": {
            "title": "FAB 실시간 현황" + (f" · {measured.group(1).strip()}" if measured else ""),
            "overall": overall,
            "areas": areas[:10],
        },
    }


def _trend_card(rows: list, area: str, hours: int, group_by: str = "area") -> dict | None:
    if not rows:
        return None
    by_bucket: dict = {}
    labels: set[str] = set()
    for r in rows:
        b = r["bucket"]
        labels.add(str(r["label"]))
        agg = by_bucket.setdefault(b, {"wip": [], "util": [], "qtime": []})
        agg["wip"].append(float(r["wip"] or 0))
        agg["util"].append(float(r["util"] or 0) * 100)
        agg["qtime"].append(float(r["qtime"] or 0))
    buckets = sorted(by_bucket)
    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else 0  # noqa: E731
    subject = next(iter(labels)) if len(labels) == 1 else (f"{area} 일대" if area else "전체")
    unit = " TG별" if group_by == "tg" and len(labels) > 1 else ""
    return {
        "type": "trend",
        "props": {
            "title": f"{subject}{unit} 추세 (최근 {hours}시간)",
            "labels": [b.strftime("%H:%M") for b in buckets],
            "series": [
                {"name": "WIP", "data": [avg(by_bucket[b]["wip"]) for b in buckets]},
                {"name": "가동률%", "data": [avg(by_bucket[b]["util"]) for b in buckets]},
                {"name": "Q-time(분)", "data": [avg(by_bucket[b]["qtime"]) for b in buckets]},
            ],
        },
    }


def _lot_card(rows: list, area: str) -> dict | None:
    if not rows:
        return None
    top = sorted(rows, key=lambda r: float(r["wip_count"] or 0), reverse=True)[:6]
    return {
        "type": "lot",
        "props": {
            "title": f"{area or '전체'} 구역별 WIP·대기",
            "rows": [
                {
                    "label": f"{r['area_name']}/{r['tg_code']}",
                    "wip": int(r["wip_count"] or 0),
                    "wait": round(float(r["wait_ratio"] or 0), 1),
                    "qtime": round(float(r["avg_qtime_min"] or 0)),
                    "util": round(float(r["utilization_rate"] or 0) * 100, 1),
                }
                for r in top
            ],
        },
    }


def _top_tg_card(rows: list, area: str, metric: str, order: str, limit: int) -> dict | None:
    if not rows:
        return None
    label = _TG_METRIC_LABELS.get(metric, metric)
    return {
        "type": "lot",
        "props": {
            "title": f"{area or '전체'} 툴그룹 {label} {'하위' if order == 'asc' else '상위'} {min(limit, len(rows))}",
            "rows": [
                {
                    "label": f"{r['area_name']}/{r['tg_code']}",
                    "wip": int(r["wip_count"] or 0),
                    "wait": round(float(r["wait_ratio"] or 0), 1),
                    "qtime": round(float(r["avg_qtime_min"] or 0)),
                    "util": round(float(r["utilization_rate"] or 0) * 100, 1),
                }
                for r in rows
            ],
        },
    }


def _tool_card(rows: list, tg: str) -> dict | None:
    if not rows:
        return None
    subject = f"{rows[0]['area_name']}/{rows[0]['tg_code']}" if tg else "전체"
    return {
        "type": "lot",
        "props": {
            "title": f"{subject} 툴 현황",
            "labelHeader": "툴",
            "columns": [
                {"key": "util", "label": "가동률", "unit": "%"},
                {"key": "oee", "label": "OEE"},
                {"key": "queue", "label": "대기 Lot"},
                {"key": "down", "label": "다운율", "unit": "%"},
            ],
            "rows": [
                {
                    "label": r["tool_code"],
                    "util": round(float(r["utilization_rate"] or 0) * 100, 1),
                    "oee": round(float(r["oee_estimate"] or 0), 1),
                    "queue": int(r["queue_lot_count"] or 0),
                    "down": round(float(r["down_ratio"] or 0) * 100, 1),
                }
                for r in rows[:10]
            ],
        },
    }


def _tool_activity_card(rows: list, hours: int) -> dict | None:
    if not rows:
        return None
    return {
        "type": "trend",
        "props": {
            "title": f"{rows[0]['tool_code']} 활동 추이 (최근 {hours}시간)",
            "labels": [r["bucket"].strftime("%H:%M") for r in rows],
            "series": [
                {"name": "가동률%", "data": [round(float(r["util"] or 0) * 100, 1) for r in rows]},
                {"name": "대기 Lot", "data": [round(float(r["queue"] or 0), 1) for r in rows]},
                {"name": "Q-time(분)", "data": [round(float(r["qtime"] or 0), 1) for r in rows]},
            ],
        },
    }


def _cases_card(rows: list, area: str) -> dict | None:
    return {
        "type": "cases",
        "props": {
            "title": f"{area or '전체'} 최근 병목 케이스",
            "rows": [
                {
                    "when": r["detected_at"].strftime("%m-%d %H:%M"),
                    "where": f"{r['area_name']}/{r['tg_code']}",
                    "grade": r["risk_grade"],
                    "prob": round(float(r["bottleneck_prob"] or 0) * 100, 1),
                    "status": r["status"],
                }
                for r in rows[:6]
            ],
        },
    }
