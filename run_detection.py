"""
병목 감지 → 확산 영향 → 원인 분석 파이프라인 실행 스크립트.

사용:
    uv run python run_detection.py
    uv run python run_detection.py --snapshot 3000
    uv run python run_detection.py --cause-only DefMEt_FE_118   # 특정 TG 원인만
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.bottleneck_detector.detector import detect
from agents.cascade_analyzer.node import analyze_cascade
from agents.cause_analyzer.node import analyze_cause
from agents.data.kpi_loader import load_kpi_snapshot
from agents.schemas.alert import SeverityLevel
from agents.solution_generator.node import generate_solutions
from agents.state import PipelineState

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

_DEFAULT_CSV = Path(__file__).parent.parent / "Simulation" / "simulation" / "sample_csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--snapshot", type=float, default=None)
    parser.add_argument("--cause-only", type=str, default=None, help="특정 TG 원인 분석만 출력")
    args = parser.parse_args()

    print(f"\n📂  데이터: {args.csv_dir}")
    kpi_list = load_kpi_snapshot(args.csv_dir, snapshot_time=args.snapshot)
    snapshot_time = kpi_list[0].snapshot_time if kpi_list else 0
    print(f"⏱   snapshot_time = {snapshot_time:.0f} min  ({snapshot_time / 60:.1f} h)")
    print(f"📊  분석 대상: {len(kpi_list)}개 toolgroup\n")

    # ── Step 1: XGBoost 병목 감지 ────────────────────────────────
    potential = detect(kpi_list)
    print(f"[1단계] XGBoost 병목 후보: {len(potential)}개")

    # ── Step 2: 확산 영향 + 심각도 결정 ──────────────────────────
    state: PipelineState = {
        "kpi_snapshot": kpi_list,
        "potential_bottlenecks": potential,
        "alerts": [],
        "cause_reports": [],
        "cascade_report": None,
        "solution_candidates": [],
        "hitl_approved": None,
    }
    state = analyze_cascade(state, csv_dir=args.csv_dir)
    alerts = state["alerts"]

    if not alerts:
        print("\n✅  병목 없음\n")
        return

    print(f"[2단계] 확산 영향 분석 완료 → {len(alerts)}개 알림\n")
    _print_alert_table(alerts)

    # ── Step 3: 원인 분석 ─────────────────────────────────────────
    print("\n[3단계] 원인 분석 중...\n")
    state = analyze_cause(state, csv_dir=args.csv_dir)
    reports = state["cause_reports"]

    # 출력 대상 필터링
    if args.cause_only:
        reports = [r for r in reports if r.toolgroup == args.cause_only]

    _print_cause_reports(reports)

    # ── Step 4: 대응안 생성 ───────────────────────────────────────
    print("\n[4단계] 대응안 생성 중...\n")
    state = generate_solutions(state)
    solutions = state["solution_candidates"]

    if args.cause_only:
        solutions = [s for s in solutions if s["toolgroup"] == args.cause_only]

    _print_solutions(solutions)


def _print_alert_table(alerts: list) -> None:
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


def _print_cause_reports(reports: list) -> None:
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

        print(f"\n  📝 {r.cause_summary}")
        print("─" * 70)


def _print_solutions(solutions: list[dict]) -> None:
    if not solutions:
        return
    print("=" * 70)
    print("  대응안 생성 결과")
    print("=" * 70)

    _CONF_BAR = {(0.8, 1.0): "●●●", (0.6, 0.8): "●●○", (0.0, 0.6): "●○○"}

    for sol in solutions:
        tg = sol["toolgroup"]
        sev = sol["severity"]
        score = sol["composite_score"]
        print(f"\n▶ {tg}  [{sev}]  종합점수={score:.0%}\n")

        for cand in sol["candidates"]:
            conf = cand["confidence"]
            bar = next(v for (lo, hi), v in _CONF_BAR.items() if lo <= conf < hi)
            params = {k: v for k, v in cand["params"].items() if v is not None}
            print(f"  [{cand['rank']}] {cand['name']}  신뢰도 {bar} {conf:.0%}")
            print(f"      파라미터: {params}")
            print(f"      목표 KPI: {cand['target_kpi']}")
            print(f"      효과: {cand['expected_effect'][:120]}...")
            print(f"      근거: {cand['rationale']}")
            print()
        print("─" * 70)


if __name__ == "__main__":
    main()
