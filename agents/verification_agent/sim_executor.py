"""WHATIF 시뮬레이션 30회 paired 실행.

trigger.py(FORWARD) 패턴을 WHATIF 모드로 확장:
  1. runs_manifest.csv에서 baseline scenario_id 조회 (fwd_base_t{T0})
  2. seed별 WHATIF mes_scenario 생성 (DRAFT)
  3. baseline 스냅샷 데이터 복사 (wip / tool / queue / release_plan)
  4. mes_whatif_action rows 삽입
  5. VALIDATED 승격
  6. subprocess → run_sim_forward_once.py --seed {seed}
  7. 30쌍 paired 결과 반환 (paired t-test 용)
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from agents.logger import get_logger
from agents.sim_runner.db_connector import get_session

_PARALLEL_WORKERS = int(os.environ.get("SIM_PARALLEL_WORKERS", min(8, os.cpu_count() or 4)))

_log = get_logger(__name__)

_SIM_ROOT = Path(__file__).parent.parent.parent.parent / "Simulation" / "simulation"
_VENV_PYTHON = _SIM_ROOT / ".venv" / "bin" / "python"
_RUNNER = _SIM_ROOT / "run_sim_forward_once.py"
_VERIFY_OUT = _SIM_ROOT / "sim_verify_out"

HORIZON_MIN = 120.0


# ── manifest 경로 (T0 기반 동적) ──────────────────────────────────────────────

def _manifest_root(t0: float) -> Path:
    return _SIM_ROOT / "sim_csv_out" / f"fwd_base_t{int(t0)}"


def _manifest_file(t0: float) -> Path:
    return _manifest_root(t0) / "runs_manifest.csv"


# ── manifest 읽기 ──────────────────────────────────────────────────────────────

def find_baseline_scenario(t0: float) -> str | None:
    """runs_manifest.csv에서 baseline scenario_id를 반환 (T0 기반 경로)."""
    mf = _manifest_file(t0)
    if not mf.is_file():
        _log.warning(f"[Exec] runs_manifest.csv 없음: {mf}")
        return None
    try:
        df = pd.read_csv(mf)
        ok = df[df["status"] == "ok"]
        if ok.empty:
            return None
        return str(ok.iloc[0]["scenario_id"])
    except Exception as e:
        _log.warning(f"[Exec] runs_manifest.csv 읽기 실패: {e}")
        return None


def _read_manifest_runs(t0: float) -> list[dict]:
    """runs_manifest.csv의 OK 런 목록 전체 반환 (T0 기반 경로)."""
    mf = _manifest_file(t0)
    if not mf.is_file():
        return []
    try:
        df = pd.read_csv(mf)
        ok = df[df["status"] == "ok"].sort_values("run_index")
        return ok.to_dict("records")
    except Exception as e:
        _log.warning(f"[Exec] manifest 읽기 실패: {e}")
        return []


def _resolve_baseline_run_csv_dir(t0: float, original_csv_dir: str) -> Path | None:
    """개별 런의 baseline CSV 경로 탐색.

    탐색 순서:
      1. 매니페스트 원본 경로
      2. 로컬 sim_csv_out/fwd_base_t{T0}/runs/<run_name>/
    """
    original = Path(original_csv_dir)
    if original.is_dir():
        return original
    local = _manifest_root(t0) / "runs" / original.name
    if local.is_dir():
        return local
    return None


# ── DB 조작 ───────────────────────────────────────────────────────────────────

def _create_whatif_scenario(
    scenario_id: str,
    t0: float,
    horizon_min: float,
    baseline_scenario_id: str,
) -> None:
    session = get_session()
    try:
        session.execute(
            text("""
                INSERT INTO mes_scenario
                    (scenario_id, description, mode, t0_sim_minute, horizon_minutes,
                     use_master_lot_release, baseline_scenario_id,
                     status, created_by, trigger_meta, created_at)
                VALUES
                    (:sid, :desc, 'WHATIF', :t0, :horizon,
                     false, :baseline,
                     'DRAFT', 'verification_agent',
                     CAST(:meta AS jsonb), NOW())
            """),
            {
                "sid": scenario_id,
                "desc": f"Verification whatif t0={t0} baseline={baseline_scenario_id}",
                "t0": t0,
                "horizon": horizon_min,
                "baseline": baseline_scenario_id,
                "meta": json.dumps({"source": "verification_agent"}),
            },
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _copy_snapshots(
    baseline_id: str,
    whatif_id: str,
    release_interval_multiplier: float,
) -> None:
    """baseline 스냅샷 데이터를 WHATIF scenario_id로 복사."""
    session = get_session()
    try:
        session.execute(
            text("""
                INSERT INTO mes_wip_snapshot
                    (scenario_id, snapshot_time, lot_id, route_id, current_step_seq,
                     status, tool_group, tool_id, queue_position, due_date_sim,
                     priority, rem_steps, processing_remaining_min,
                     wafers_per_lot, product, is_super_hot)
                SELECT :wid, snapshot_time, lot_id, route_id, current_step_seq,
                       status, tool_group, tool_id, queue_position, due_date_sim,
                       priority, rem_steps, processing_remaining_min,
                       wafers_per_lot, product, is_super_hot
                FROM mes_wip_snapshot
                WHERE scenario_id = :bid
                ON CONFLICT DO NOTHING
            """),
            {"wid": whatif_id, "bid": baseline_id},
        )

        session.execute(
            text("""
                INSERT INTO mes_tool_snapshot
                    (scenario_id, tool_id, tool_group, op_state, current_setup, held_lot_id)
                SELECT :wid, tool_id, tool_group, op_state, current_setup, held_lot_id
                FROM mes_tool_snapshot
                WHERE scenario_id = :bid
                ON CONFLICT DO NOTHING
            """),
            {"wid": whatif_id, "bid": baseline_id},
        )

        session.execute(
            text("""
                INSERT INTO mes_tool_queue_snapshot
                    (scenario_id, tool_id, position, lot_id, route_id,
                     step_seq, due_date_sim, priority)
                SELECT :wid, tool_id, position, lot_id, route_id,
                       step_seq, due_date_sim, priority
                FROM mes_tool_queue_snapshot
                WHERE scenario_id = :bid
                ON CONFLICT DO NOTHING
            """),
            {"wid": whatif_id, "bid": baseline_id},
        )

        mult = release_interval_multiplier
        session.execute(
            text("""
                INSERT INTO mes_lot_release_plan
                    (scenario_id, source_lot_release_id, product_name, route_name,
                     release_time, lots_count, release_interval, lot_name_prefix,
                     lot_type, priority, due_date_sim, wafers_per_lot, is_super_hot)
                SELECT :wid, source_lot_release_id, product_name, route_name,
                       release_time, lots_count,
                       CASE WHEN release_interval IS NOT NULL
                            THEN release_interval * :mult ELSE NULL END,
                       lot_name_prefix, lot_type, priority, due_date_sim,
                       wafers_per_lot, is_super_hot
                FROM mes_lot_release_plan
                WHERE scenario_id = :bid
            """),
            {"wid": whatif_id, "bid": baseline_id, "mult": mult},
        )

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _insert_whatif_actions(scenario_id: str, action_rows: list[dict]) -> None:
    if not action_rows:
        return
    session = get_session()
    try:
        for row in action_rows:
            payload = row.get("payload_json")
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload)
            elif not isinstance(payload, str):
                payload = "{}"
            session.execute(
                text("""
                    INSERT INTO mes_whatif_action
                        (scenario_id, seq, action_kind, effective_time,
                         lot_id, route_id, step_seq, tool_group, tool_id,
                         payload_json, source)
                    VALUES
                        (:sid, :seq, :kind, :eff,
                         :lot, :route, :step, :tg, :tid,
                         CAST(:payload AS jsonb), :src)
                """),
                {
                    "sid": scenario_id,
                    "seq": row.get("seq", 0),
                    "kind": row["action_kind"],
                    "eff": row["effective_time"],
                    "lot": row.get("lot_id"),
                    "route": row.get("route_id"),
                    "step": row.get("step_seq"),
                    "tg": row.get("tool_group"),
                    "tid": row.get("tool_id"),
                    "payload": payload,
                    "src": row.get("source", "AGENT"),
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


# ── 메인 실행 함수 ─────────────────────────────────────────────────────────────

def run_whatif_paired(
    t0: float,
    horizon_min: float,
    action_rows: list[dict],
    release_interval_multiplier: float,
    label: str,
) -> tuple[str, list[dict], str]:
    """
    WHATIF 30회 paired 실행 (seed=baseline 런과 동일).

    각 seed별로 WHATIF 시나리오를 생성하고 시뮬을 실행.
    paired t-test용 30쌍 pairs 반환.

    Returns:
        group_id:    이 검증 세션의 그룹 ID (prefix)
        pairs:       [{run_index, seed, baseline_csv_dir, whatif_csv_dir, ...}]
        baseline_id: FWD_BASE_T26820
    """
    baseline_id = find_baseline_scenario(t0)
    if baseline_id is None:
        raise RuntimeError(
            f"runs_manifest.csv에서 baseline 시나리오를 찾을 수 없습니다 "
            f"(manifest={_manifest_file(t0)})."
        )

    manifest_runs = _read_manifest_runs(t0)
    if not manifest_runs:
        raise RuntimeError("runs_manifest.csv에 OK 런이 없습니다.")

    group_id = f"VERIFY_{label}_{int(t0)}_{uuid.uuid4().hex[:4]}"

    # ── 1단계: DB 셋업 (순차) ────────────────────────────────────────────────
    ready_runs: list[dict] = []
    for run in manifest_runs:
        run_index = int(run["run_index"])
        seed = int(run["seed"])
        run_id = str(run.get("run_id", ""))

        baseline_csv_dir = _resolve_baseline_run_csv_dir(t0, str(run["csv_dir"]))
        if baseline_csv_dir is None:
            _log.warning(f"[Exec] run_{run_index:02d} baseline CSV 없음, 스킵")
            continue

        whatif_id = f"{group_id}_R{run_index:02d}"
        whatif_csv_dir = _VERIFY_OUT / group_id / f"run_{run_index:02d}"
        whatif_csv_dir.mkdir(parents=True, exist_ok=True)

        try:
            _create_whatif_scenario(whatif_id, t0, horizon_min, baseline_id)
            _copy_snapshots(baseline_id, whatif_id, release_interval_multiplier)
            _insert_whatif_actions(whatif_id, action_rows)
            _promote_to_validated(whatif_id)
        except Exception as e:
            _log.error(f"[Exec] run_{run_index:02d} DB 셋업 실패: {e}")
            continue

        ready_runs.append({
            "run_index": run_index,
            "seed": seed,
            "run_id": run_id,
            "whatif_id": whatif_id,
            "whatif_csv_dir": whatif_csv_dir,
            "baseline_csv_dir": baseline_csv_dir,
        })

    # ── 2단계: 시뮬레이션 병렬 실행 ─────────────────────────────────────────
    def _run_one(r: dict) -> dict | None:
        result = subprocess.run(
            [
                str(_VENV_PYTHON), str(_RUNNER),
                "--scenario-id", r["whatif_id"],
                "--csv-dir", str(r["whatif_csv_dir"]),
                "--seed", str(r["seed"]),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_SIM_ROOT),
        )
        if result.returncode != 0:
            _log.error(
                f"[Exec] run_{r['run_index']:02d} 시뮬 실패 [{r['whatif_id']}]:\n"
                f"{result.stderr[-500:]}"
            )
            return None
        return r

    pairs: list[dict] = []
    workers = min(_PARALLEL_WORKERS, len(ready_runs))
    _log.info(f"[Exec] {group_id} — {len(ready_runs)}개 병렬 실행 (workers={workers})")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, r): r for r in ready_runs}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                _log.error(f"[Exec] run_{r['run_index']:02d} 예외: {e}")
                result = None
            if result is not None:
                pairs.append({
                    "run_index": r["run_index"],
                    "seed": r["seed"],
                    "baseline_csv_dir": str(r["baseline_csv_dir"]),
                    "whatif_csv_dir": str(r["whatif_csv_dir"]),
                    "whatif_scenario_id": r["whatif_id"],
                    "baseline_scenario_id": baseline_id,
                    "baseline_run_id": r["run_id"],
                })

    if not pairs:
        raise RuntimeError(f"[Exec] {label}: 성공한 paired 런이 없습니다.")

    _log.info(f"[Exec] {group_id} 완료 — {len(pairs)}/{len(manifest_runs)} paired runs")
    return group_id, pairs, baseline_id


# ── 사후 정리 ──────────────────────────────────────────────────────────────────

def cleanup_verify_scenarios(scenario_ids: list[str]) -> None:
    """VERIFY 시나리오 데이터를 DB와 CSV에서 삭제한다.

    순서:
      1. mes_scenario_run에서 simulation_run_id 수집 (삭제 전)
      2. mes_scenario 삭제 → CASCADE로 하위 테이블 자동 삭제
      3. simulation_log / lot_event_log / lot_release_ledger / tool_state_log 삭제
      4. simulation_run 삭제
      5. sim_verify_out 디렉토리 삭제
    """
    import shutil

    if not scenario_ids:
        return

    session = get_session()
    try:
        # 1. simulation_run_id 수집
        placeholders = ", ".join(f":sid{i}" for i in range(len(scenario_ids)))
        params = {f"sid{i}": sid for i, sid in enumerate(scenario_ids)}

        rows = session.execute(
            text(f"SELECT simulation_run_id FROM mes_scenario_run WHERE scenario_id IN ({placeholders})"),
            params,
        ).fetchall()
        run_ids = [r[0] for r in rows if r[0]]

        # 2. mes_scenario 삭제 (CASCADE: wip/tool/queue/release_plan/whatif_action/kpi_diff/scenario_run 등)
        session.execute(
            text(f"DELETE FROM mes_scenario WHERE scenario_id IN ({placeholders})"),
            params,
        )

        if run_ids:
            run_placeholders = ", ".join(f":rid{i}" for i in range(len(run_ids)))
            run_params = {f"rid{i}": rid for i, rid in enumerate(run_ids)}

            # 3. 로그 테이블 삭제
            for table in ("simulation_log", "lot_event_log", "lot_release_ledger", "tool_state_log"):
                session.execute(
                    text(f"DELETE FROM {table} WHERE run_id IN ({run_placeholders})"),
                    run_params,
                )

            # 4. simulation_run 삭제
            session.execute(
                text(f"DELETE FROM simulation_run WHERE run_id IN ({run_placeholders})"),
                run_params,
            )

        session.commit()
        _log.info(f"[Cleanup] {len(scenario_ids)}개 VERIFY 시나리오 DB 삭제 완료")

    except Exception as e:
        session.rollback()
        _log.warning(f"[Cleanup] DB 삭제 실패: {e}")
    finally:
        session.close()

    # 5. sim_verify_out CSV 디렉토리 삭제
    group_ids = set()
    for sid in scenario_ids:
        # VERIFY_PLAN_A_4740_f67e_R01 → VERIFY_PLAN_A_4740_f67e
        parts = sid.rsplit("_R", 1)
        if len(parts) == 2 and parts[1].isdigit():
            group_ids.add(parts[0])

    for gid in group_ids:
        group_dir = _VERIFY_OUT / gid
        if group_dir.exists():
            try:
                shutil.rmtree(group_dir)
                _log.info(f"[Cleanup] CSV 디렉토리 삭제: {group_dir.name}")
            except Exception as e:
                _log.warning(f"[Cleanup] CSV 디렉토리 삭제 실패 {group_dir}: {e}")
