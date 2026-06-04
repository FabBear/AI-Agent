"""
FORWARD 시나리오 생성 → VALIDATED 승격 → run_sim_forward_once.py 호출.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from sqlalchemy import text  # noqa: F401 (used in SQL strings)

from agents.sim_runner.db_connector import get_session
from agents.sim_runner.snapshot_builder import insert_t0_snapshot

_SIM_ROOT = Path(__file__).parent.parent.parent.parent / "Simulation" / "simulation"
_VENV_PYTHON = _SIM_ROOT / ".venv" / "bin" / "python"
_RUNNER = _SIM_ROOT / "run_sim_forward_once.py"
_CSV_OUT = _SIM_ROOT / "sim_forward_out"


def run_forward(t0: float, horizon_min: float = 120.0) -> Path:
    """
    T0 스냅샷 생성 → FORWARD 시나리오 생성 → 실행 → 출력 CSV 디렉토리 반환.
    """
    scenario_id = f"AGENT_FWD_{int(t0)}_{uuid.uuid4().hex[:6]}"
    csv_dir = _CSV_OUT / scenario_id
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [Forward Sim] scenario_id={scenario_id}  t0={t0}  horizon={horizon_min}min")

    # 1. mes_scenario 생성 (DRAFT) — FK 제약 때문에 스냅샷보다 먼저
    _create_scenario(scenario_id, t0, horizon_min)

    # 2. T0 스냅샷 삽입
    wip_n, tool_n = insert_t0_snapshot(scenario_id, t0)
    print(f"  [Forward Sim] T0 스냅샷: WIP {wip_n}개, Tool {tool_n}개")

    # 3. VALIDATED 승격
    _promote_to_validated(scenario_id)

    # 4. run_sim_forward_once.py 호출
    print("  [Forward Sim] 시뮬레이션 실행 중...")
    result = subprocess.run(
        [str(_VENV_PYTHON), str(_RUNNER), "--scenario-id", scenario_id, "--csv-dir", str(csv_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(_SIM_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Forward sim failed:\n{result.stderr[-1000:]}")

    print("  [Forward Sim] 완료")
    return csv_dir


def _create_scenario(scenario_id: str, t0: float, horizon: float) -> None:
    session = get_session()
    try:
        session.execute(
            text("""
            INSERT INTO mes_scenario
                (scenario_id, description, mode, t0_sim_minute, horizon_minutes,
                 use_master_lot_release, status, created_by, trigger_meta, created_at)
            VALUES
                (:sid, :desc, 'FORWARD', :t0, :horizon,
                 true, 'DRAFT', 'ai_agent',
                 '{"source": "agent", "purpose": "bottleneck_forecast"}',
                 NOW())
        """),
            {
                "sid": scenario_id,
                "desc": f"AI-Agent 2h forward forecast from t={t0}",
                "t0": t0,
                "horizon": horizon,
            },
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _promote_to_validated(scenario_id: str) -> None:
    session = get_session()
    try:
        session.execute(
            text("""
            UPDATE mes_scenario SET status = 'VALIDATED'
            WHERE scenario_id = :sid AND status = 'DRAFT'
        """),
            {"sid": scenario_id},
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
