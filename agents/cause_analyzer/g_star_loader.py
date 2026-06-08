"""G* 분석 handoff JSON + KPI evidence 로더."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_G_STAR_OUT = (
    Path(__file__).parent.parent.parent.parent
    / "Simulation" / "simulation" / "out" / "ml_g_star_e2e"
)


@dataclass
class KpiEvidence:
    kpi: str
    delta_mean: float      # forward - baseline 평균 차이
    t_p_adj: float         # FDR 보정 p-value
    significant: bool      # kpi_significant == 1


@dataclass
class GStarResult:
    anchor_tg: str
    toolgroups: list[str]                          # G* 유의미 TG 목록
    t0_sim_minute: float
    alarm_threshold: float = 0.7
    n_g_star: int = 0
    n_total_tg: int = 0                            # 전체 TG 수
    tg_proba: dict[str, float] = field(default_factory=dict)  # TG별 ML 확률
    kpi_evidence: dict[str, list[KpiEvidence]] = field(default_factory=dict)


def load_g_star(t0: float, out_dir: Path | None = None) -> GStarResult | None:
    """t0 시점의 G* 분석 결과를 로드한다. 파일 없으면 None 반환."""
    base = out_dir or _G_STAR_OUT

    # 1. ML G* 기본 파일 로드
    ml_candidates = [
        base / f"g_star_T{int(t0)}.json",
        base / f"g_star_T{t0}.json",
    ]
    ml_data = None
    for path in ml_candidates:
        if path.exists():
            ml_data = json.loads(path.read_text(encoding="utf-8"))
            break

    if ml_data is None:
        return None

    result = GStarResult(
        anchor_tg=ml_data.get("anchor_tg", ""),
        toolgroups=ml_data.get("toolgroups", []),
        t0_sim_minute=ml_data.get("t0_sim_minute", t0),
        alarm_threshold=ml_data.get("alarm_threshold", 0.7),
        n_g_star=ml_data.get("n_g_star", 0),
    )

    # audit CSV에서 TG별 확률 로드
    audit_candidates = list(base.glob(f"ml_alarm_audit_t{int(t0)}*.csv"))
    if audit_candidates:
        try:
            import pandas as pd
            audit_df = pd.read_csv(audit_candidates[0])
            result.n_total_tg = len(audit_df)
            result.tg_proba = {
                str(row["toolgroup"]): float(row["proba"])
                for _, row in audit_df.iterrows()
                if "toolgroup" in row and "proba" in row
            }
        except Exception:
            pass

    # 2. 통계 검정 결과 (agent_handoff_g_star_analysis.json + evidence CSV) 로드
    handoff_path = base / "agent_handoff_g_star_analysis.json"
    if handoff_path.exists():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        gsa = handoff.get("g_star_analysis", {})

        # 통계 검정으로 확정된 TG 목록으로 업데이트
        stat_tgs = gsa.get("g_star_toolgroups", [])
        if stat_tgs:
            result.toolgroups = stat_tgs

        # evidence CSV 로드
        evidence_csv = gsa.get("evidence_csv", "")
        if evidence_csv:
            evidence_path = Path(evidence_csv)
            if not evidence_path.is_absolute():
                evidence_path = base / evidence_csv
            if evidence_path.exists():
                result.kpi_evidence = _load_evidence(evidence_path)

    return result


def _load_evidence(path: Path) -> dict[str, list[KpiEvidence]]:
    """evidence CSV → {toolgroup: [KpiEvidence]} 딕셔너리."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        evidence: dict[str, list[KpiEvidence]] = {}
        for _, row in df.iterrows():
            tg = str(row.get("toolgroup", ""))
            kpi = str(row.get("kpi", ""))
            if not tg or not kpi:
                continue
            ev = KpiEvidence(
                kpi=kpi,
                delta_mean=float(row.get("delta_mean") or 0.0),
                t_p_adj=float(row.get("t_p_adj") or 1.0),
                significant=bool(int(row.get("kpi_significant", 0))),
            )
            evidence.setdefault(tg, []).append(ev)
        return evidence
    except Exception:
        return {}
