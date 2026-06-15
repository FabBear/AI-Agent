"""파이프라인 결과 터미널 출력."""

from __future__ import annotations

from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.cause import CauseReport

_COLORS = {
    SeverityLevel.CRITICAL: "\033[91m",
    SeverityLevel.HIGH: "\033[93m",
    SeverityLevel.MEDIUM: "\033[94m",
    SeverityLevel.LOW: "\033[37m",
}
_RESET = "\033[0m"
_ICONS = {
    SeverityLevel.CRITICAL: "🔴",
    SeverityLevel.HIGH: "🟠",
    SeverityLevel.MEDIUM: "🟡",
    SeverityLevel.LOW: "🔵",
}


def print_alert_table(alerts: list[BottleneckAlert]) -> None:
    print("─" * 90)
    print(
        f"  {'':2} {'심각도':<10} {'Toolgroup':<28} {'종합':>5}  {'확률':>5}  {'영향':>5}  {'후속TG':>4}  {'CT증가':>7}  {'위험lot':>6}"
    )
    print("─" * 90)
    for a in alerts:
        c = _COLORS[a.severity]
        icon = _ICONS[a.severity]
        imp = a.impact
        print(
            f"  {icon} {c}{a.severity.value:<8}{_RESET}  "
            f"{a.toolgroup:<28} "
            f"{a.composite_score:>4.0%}  "
            f"{a.probability:>4.0%}  "
            f"{imp.impact_score:>4.0%}  "
            f"{len(imp.affected_tgs):>4}개  "
            f"{imp.ct_increase_min:>6.1f}분  "
            f"{imp.at_risk_lots:>5.0f}개"
        )
    print("─" * 90)
    counts = {s: sum(1 for a in alerts if a.severity == s) for s in SeverityLevel}
    print(
        f"\n  🔴 Critical {counts[SeverityLevel.CRITICAL]}  "
        f"🟠 High {counts[SeverityLevel.HIGH]}  "
        f"🟡 Medium {counts[SeverityLevel.MEDIUM]}  "
        f"🔵 Low {counts[SeverityLevel.LOW]}  (전체 {len(alerts)}개)"
    )


def print_cause_reports(reports: list[CauseReport]) -> None:
    if not reports:
        return
    print("=" * 70)
    print("  원인 분석 결과 (Evidence-based)")
    print("=" * 70)
    for r in reports:
        print(f"\n▶ {r.toolgroup}  (snapshot={r.snapshot_time:.0f}min)\n")

        # ── SHAP
        print("  [① SHAP 기여도]")
        for s in r.shap_top:
            bar = "█" * int(abs(s.shap_value) * 30)
            sign = "+" if s.shap_value > 0 else "-"
            print(f"    {s.feature:<28} {sign}{bar}  ({s.shap_value:+.3f}, 현재={s.kpi_value:.3f})")

        # ── 트렌드 (significant 표시)
        print("\n  [② KPI 트렌드]")
        if r.trend_top:
            for t in r.trend_top:
                arrow = "↑" if t.slope_per_hour > 0 else "↓"
                sig_mark = " ★유의" if t.significant else ""
                r2_str = f"R²={t.r2:.2f}" if t.r2 > 0 else ""
                print(f"    {t.feature:<28} {arrow} {t.slope_per_hour:+.4f}/h  {r2_str}{sig_mark}")
        else:
            print("    (데이터 없음)")

        # ── 통계적 분석 (G* KPI 검정 + 시뮬 예측)
        print(f"\n  [③ 통계적 분석]")

        if r.consensus.g_star_sig_kpis:
            conf_label = "TG 포함(통계 확인)" if r.consensus.g_star_confirmed else "TG 미포함"
            print(f"  ┌ G* KPI 검정  [{conf_label}]  (p<0.05=유의)")
            print(f"  {'지표':<28} {'Δ평균':>8}  {'p-value':>8}  결과")
            print(f"  {'─' * 58}")
            for e in r.consensus.g_star_sig_kpis:
                verdict = "★ 유의" if e.significant else "─"
                print(f"  {e.kpi:<28} {e.delta_mean:>+8.3f}  {e.t_p_adj:>8.4f}  {verdict}")
        else:
            print(f"  ┌ G* KPI 검정  (데이터 없음)")

        if r.sim_forecast:
            f = r.sim_forecast
            horizon_h = (f.t_future - f.t0) / 60
            horizon_label = f"{horizon_h:.0f}h 후"
            print(f"\n  ┌ 시뮬 예측  t={f.t0:.0f} → t={f.t_future:.0f}  (+{horizon_h:.0f}h)")
            print(f"  {'지표':<22} {'현재':>8}  {horizon_label:>8}  {'변화':>8}")
            print(f"  {'─' * 54}")
            for kpi, comp in f.kpi_delta.items():
                arrow = "↑" if comp.delta > 0 else "↓" if comp.delta < 0 else "─"
                warn = " ⚠" if abs(comp.pct_change) > 20 else ""
                print(
                    f"  {kpi:<22} {comp.now:>8.2f}  {comp.future:>8.2f}"
                    f"  {comp.pct_change:>+6.1f}% {arrow}{warn}"
                )
            status = "⚠ 악화 예상" if f.gets_worse else "✓ 안정 유지"
            print(f"  {'─' * 54}")
            print(f"  전망: {status}")

        # ── 카테고리 수렴
        if r.cause_categories:
            print(f"\n  [④ 카테고리 수렴 분석] — 4가지 분석 종합 (score 순)")
            bar_max = max((c.total_score for c in r.cause_categories), default=1.0) or 1.0
            for cat in r.cause_categories:
                bar_len = int(cat.total_score / bar_max * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                trend_str = f"Trend★={cat.n_trend_significant}" if cat.n_trend_significant else "Trend=0"
                g_str = "G*=확인" if cat.g_star_confirmed else "G*=✗"
                extras = f"{trend_str} | {g_str}"
                conf_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(cat.confidence, "")
                print(
                    f"    {cat.name:<10} {bar}  "
                    f"SHAP={cat.shap_share_pct:5.1f}%  {extras:<20}  "
                    f"score={cat.total_score:.3f}  {conf_icon}[{cat.confidence}]"
                )

        # ── LLM 판정
        if r.judgment:
            j = r.judgment
            conf_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(j.primary_confidence, "")
            print(f"\n  [⑤ LLM 판정]")
            cat_str = f"  ({j.primary_category})" if j.primary_category else ""
            print(f"    {conf_icon} 주요 원인: {j.primary_cause}{cat_str}  [{j.primary_confidence}]")
            print(f"    근거: {j.primary_reasoning}")
            if j.secondary_causes:
                print(f"    보조 원인: {', '.join(j.secondary_causes)}")
            if j.dismissed:
                print(f"    기각: {', '.join(j.dismissed)}")
                if j.dismissed_reason:
                    print(f"           → {j.dismissed_reason}")

        print(f"\n  📝 {r.cause_summary}")
        print("─" * 70)


def _print_composite_solutions(solutions: list[dict]) -> None:
    """GlobalCompositeCandidate 포맷 출력."""
    tgs = solutions[0].get("target_toolgroups", [])
    complexity = solutions[0].get("cause_complexity", "-")
    hitl = any(s.get("hitl_escalation_recommended") for s in solutions)
    print(f"\n▶ 글로벌 복합 대응안  ({len(solutions)}개 강도)")
    print(f"  대상 TG     : {', '.join(tgs)}")
    print(f"  원인 복잡도 : {complexity}")
    if hitl:
        reason = next((s.get("escalation_reason", "") for s in solutions if s.get("hitl_escalation_recommended")), "")
        print(f"  ⚠ HITL 에스컬레이션 권고: {reason}")
    print()

    _level_name = {"conservative": "보수적 조정안", "standard": "표준 조정안", "aggressive": "강화 조정안"}
    for s in solutions:
        plan_id = s.get("plan_id", "-")
        delta_pct = s.get("release_interval_delta_pct", 0)
        adjustments = s.get("lot_adjustments", [])
        danger = [a for a in adjustments if a.get("zone") == "danger"]
        warn_upper = [a for a in adjustments if a.get("zone") == "warn_upper"]
        warn_lower = [a for a in adjustments if a.get("zone") == "warn_lower"]

        print(f"  [{plan_id.upper()}] {_level_name.get(plan_id, plan_id)}")
        print(f"      Release Interval 조정 : {delta_pct:+.1f}%")
        if adjustments:
            print(f"      lot 조정 ({len(adjustments)}건)")
            for zone_key, zone_label, action_label in [
                ("danger",     "위험구간", "→ SuperHotLot (priority 30)"),
                ("warn_upper", "경고상단", "→ priority 30           "),
                ("warn_lower", "경고하단", "→ priority 20           "),
            ]:
                group = [a for a in adjustments if a.get("zone") == zone_key]
                if not group:
                    continue
                print(f"        [{zone_label}] {action_label}  ({len(group)}건)")
                type_counts: dict[str, int] = {}
                for a in group:
                    t = a.get("lot_type", "-")
                    type_counts[t] = type_counts.get(t, 0) + 1
                type_seq: dict[str, int] = {}
                for a in group:
                    raw   = a.get("lot_type", "-")
                    type_seq[raw] = type_seq.get(raw, 0) + 1
                    lot_id = f"{raw}_{type_seq[raw]}" if type_counts[raw] > 1 else raw
                    prod   = a.get("product_name", "-")
                    rel    = a.get("release_time", 0)
                    t2d    = a.get("time_to_due", 0)
                    print(f"          {lot_id:<26} {prod:<12}  rel={rel:.0f}  t2due={t2d:.0f}분")
        else:
            print(f"      lot 조정 : 없음 (납기 위험 없음)")
    print("─" * 70)


def print_solutions(solutions: list[dict]) -> None:
    if not solutions:
        return
    print("=" * 70)
    print("  대응안 생성 결과")
    print("=" * 70)

    # 신규: GlobalCompositeCandidate 포맷
    if solutions[0].get("plan_id") in ("conservative", "standard", "aggressive"):
        _print_composite_solutions(solutions)
        return

    # 레거시: per-TG 포맷
    for entry in solutions:
        tg = entry.get("toolgroup", "-")
        candidates = entry.get("candidates", [])
        print(f"\n▶ {tg}  ({len(candidates)}개 후보)")
        for c in candidates:
            rank = c.get("rank", "-")
            name = c.get("name", "")
            params = c.get("params", {})
            delta_pct = params.get("release_interval_delta_pct")
            priority = params.get("lot_priority_rule") or params.get("priority_direction") or "변경 없음"
            superhotlot = "활성화" if params.get("superhotlot_enable") else "비활성화"
            effect = c.get("expected_effect", "")
            delta_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "변경 없음"
            print(f"  [{rank}] {name}")
            print(f"      Release Interval 조정 : {delta_str}")
            print(f"      투입 우선순위          : {priority}")
            print(f"      SUPERHOTLOT           : {superhotlot}")
            if effect:
                print(f"      기대 효과             : {effect[:200]}")
        print("─" * 70)


def print_report_results(report_results: list[dict]) -> None:
    if not report_results:
        return
    print("=" * 70)
    print("  최종보고서 생성 결과")
    print("=" * 70)
    for r in report_results:
        tg = r.get("toolgroup", "-")
        md = r.get("output_path", "")
        js = r.get("json_output_path", "")
        print(f"\n▶ {tg}")
        if md:
            print(f"  📄 Markdown: {md}")
        if js:
            print(f"  📋 JSON:     {js}")
    print("─" * 70)
