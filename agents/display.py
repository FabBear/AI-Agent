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
    print("  원인 분석 결과")
    print("=" * 70)
    for r in reports:
        print(f"\n▶ {r.toolgroup}  (snapshot={r.snapshot_time:.0f}min)\n")

        print("  [SHAP 기여도]")
        for s in r.shap_top:
            bar = "█" * int(abs(s.shap_value) * 30)
            sign = "+" if s.shap_value > 0 else "-"
            print(f"    {s.feature:<25} {sign}{bar}  ({s.shap_value:+.3f}, 현재={s.kpi_value:.3f})")

        print("\n  [KPI 트렌드]")
        for t in r.trend_top:
            arrow = "↑" if t.slope_per_hour > 0 else "↓"
            print(f"    {t.feature:<25} {arrow} {t.slope_per_hour:+.4f}/h  {t.values}")

        if r.upstream_suspects:
            print(f"\n  [업스트림 과부하]  {', '.join(r.upstream_suspects)}")

        if r.sim_forecast:
            f = r.sim_forecast
            print(f"\n  [2시간 후 시뮬 예측]  t={f.t0:.0f} → t={f.t_future:.0f}")
            print(f"  {'지표':<22} {'현재':>8}  {'2h 후':>8}  {'변화':>8}")
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

        if r.consensus.g_star_confirmed and r.consensus.g_star_sig_kpis:
            print(f"\n  [G* T-test — 병목 원인 검증]  신뢰도={r.consensus.confidence_level}")
            has_cause = any(e.significant for e in r.consensus.g_star_sig_kpis)
            for e in r.consensus.g_star_sig_kpis:
                if e.significant:
                    verdict = "★ 통계적 원인 확인 (비정상 상승)"
                else:
                    verdict = "정상 범위 (원인 아님)"
                print(f"    {e.kpi:<25} Δ={e.delta_mean:+.1f}  p={e.t_p_adj:.4f}  {verdict}")
            if not has_cause:
                print(f"    → 통계적으로 확인된 영구 원인 없음 — 일시적 과부하 가능성")

        print(f"\n  📝 {r.cause_summary}")
        print("─" * 70)


def print_solutions(solutions: list[dict]) -> None:
    if not solutions:
        return
    print("=" * 70)
    print("  대응안 생성 결과 (Lot Release 테이블 조정)")
    print("=" * 70)
    for plan in solutions:
        plan_id = plan["plan_id"]
        cur = plan.get("current_interval_minutes", 0.0)
        interval = plan["release_interval_minutes"]
        priority = plan.get("lot_priority_rule") or "변경 없음"
        superhotlot = "활성화" if plan.get("superhotlot_enable") else "비활성화"
        tgs = plan.get("target_toolgroups", [])
        conf = plan["confidence"]
        bar = "●●●" if conf >= 0.8 else "●●○" if conf >= 0.6 else "●○○"

        print(f"\n▶ 플랜 {plan_id}  신뢰도 {bar} {conf:.0%}")
        print(f"   Release Interval : {cur:.1f}분 → {interval:.1f}분  (+{interval - cur:.1f}분)")
        print(f"   투입 우선순위     : {priority} 적용")
        print(f"   SUPERHOTLOT      : {superhotlot}  (대상: {', '.join(tgs) if tgs else '없음'} 통과 대기 lot)")

        effect = plan.get("expected_effect") or plan.get("description", "")
        if effect:
            print(f"   기대 효과: {effect[:200]}")
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
