"""mes_lot_release_plan 조회 — solution_generator의 lot 분류에 사용."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from agents.logger import get_logger
from agents.sim_runner.db_connector import get_session

_log = get_logger(__name__)

_SIM_CSV_OUT = (
    Path(__file__).parent.parent.parent.parent
    / "Simulation" / "simulation" / "sim_csv_out"
)


def _find_baseline_scenario_id(t0: float) -> str | None:
    """fwd_base_t{T0}/runs_manifest.csv에서 baseline scenario_id 반환."""
    manifest = _SIM_CSV_OUT / f"fwd_base_t{int(t0)}" / "runs_manifest.csv"
    if not manifest.is_file():
        _log.warning(f"[ReleasePlan] runs_manifest.csv 없음: {manifest}")
        return None
    try:
        df = pd.read_csv(manifest)
        ok = df[df["status"] == "ok"]
        if ok.empty:
            return None
        return str(ok.iloc[0]["scenario_id"])
    except Exception as e:
        _log.warning(f"[ReleasePlan] manifest 읽기 실패: {e}")
        return None


def _find_forward_scenario_id(t0: float) -> str | None:
    """runs_manifest가 없을 때 FORWARD 시나리오 fallback — t0에 해당하는 가장 최근 시나리오."""
    session = get_session()
    try:
        row = session.execute(
            text("""
                SELECT scenario_id FROM mes_scenario
                WHERE mode = 'FORWARD'
                  AND t0_sim_minute = :t0
                  AND status = 'VALIDATED'
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"t0": t0},
        ).fetchone()
        return str(row[0]) if row else None
    except Exception as e:
        _log.warning(f"[ReleasePlan] FORWARD 시나리오 조회 실패: {e}")
        return None
    finally:
        session.close()


def load_release_plan(t0: float) -> list[dict]:
    """baseline 시나리오의 mes_lot_release_plan 전체 반환.

    baseline_scenario_id 탐색 순서:
      1. fwd_base_t{T0}/runs_manifest.csv
      2. DB에서 FORWARD 시나리오 fallback

    Returns:
        source_lot_release_id, product_name, release_time, due_date_sim 필드를 포함한 dict 리스트.
        조회 실패 시 빈 리스트 반환.
    """
    scenario_id = _find_baseline_scenario_id(t0) or _find_forward_scenario_id(t0)
    if scenario_id is None:
        _log.warning(f"[ReleasePlan] t0={t0} baseline 시나리오 없음 — 빈 리스트 반환")
        return []

    session = get_session()
    try:
        rows = session.execute(
            text("""
                SELECT id, lot_type, product_name, release_time,
                       due_date_sim, priority, is_super_hot, lots_count
                FROM mes_lot_release_plan
                WHERE scenario_id = :sid
                ORDER BY release_time
            """),
            {"sid": scenario_id},
        ).fetchall()

        result = [
            {
                "lot_plan_id": int(r[0]),
                "lot_type": str(r[1]),
                "product_name": str(r[2]),
                "release_time": float(r[3] or 0),
                "due_date_sim": float(r[4] or 0),
                "priority": int(r[5] or 10),
                "is_super_hot": bool(r[6]),
                "lots_count": int(r[7] or 1),
            }
            for r in rows
        ]
        _log.info(f"[ReleasePlan] scenario={scenario_id} → {len(result)}개 release_plan 행 조회")
        return result

    except Exception as e:
        _log.error(f"[ReleasePlan] mes_lot_release_plan 조회 실패: {e}")
        return []
    finally:
        session.close()
