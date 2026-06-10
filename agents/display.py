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

        # ── 업스트림
        if r.upstream_suspects:
            print(f"\n  [③ 업스트림 과부하]  {', '.join(r.upstream_suspects)}")

        # ── G* T-test
        if r.consensus.g_star_confirmed and r.consensus.g_star_sig_kpis:
            print(f"\n  [④ G* T-test]")
            for e in r.consensus.g_star_sig_kpis:
                verdict = "★ 통계 확인" if e.significant else "비유의"
                print(f"    {e.kpi:<28} Δ={e.delta_mean:+.1f}  p={e.t_p_adj:.4f}  {verdict}")

        # ── Evidence 수렴 요약 (핵심 신규)
        if r.evidence_bundle:
            print(f"\n  [Evidence 수렴] — 4개 분석 기준 피처별 votes (●=지지, ○=미지지)")
            for ev in r.evidence_bundle:
                filled = "●" * ev.votes
                empty  = "○" * (4 - ev.votes)
                conf_label = {"HIGH": "HIGH ★", "MEDIUM": "MEDIUM", "LOW": "LOW"}.get(ev.confidence, ev.confidence)
                sources = []
                if ev.shap_value is not None and ev.shap_value > 0:
                    sources.append("SHAP")
                if ev.trend_significant:
                    sources.append("Trend")
                if ev.upstream_match:
                    sources.append("Upstream")
                if ev.g_star_significant:
                    sources.append("G*")
                src_str = f"  ← {', '.join(sources)}" if sources else ""
                print(f"    {ev.feature:<28} {filled}{empty}  [{conf_label}]{src_str}")

        # ── LLM 판정 결과 (핵심 신규)
        if r.judgment:
            j = r.judgment
            conf_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(j.primary_confidence, "")
            print(f"\n  [LLM 판정]")
            print(f"    {conf_icon} 주요 원인: {j.primary_cause}  [{j.primary_confidence}]")
            print(f"    근거: {j.primary_reasoning}")
            if j.secondary_causes:
                print(f"    보조 원인: {', '.join(j.secondary_causes)}")
            if j.dismissed:
                print(f"    기각: {', '.join(j.dismissed)}")
                if j.dismissed_reason:
                    print(f"           → {j.dismissed_reason}")

        # ── 시뮬 예측
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
        print(f"\n▶ 플랜 {plan_id}")
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
