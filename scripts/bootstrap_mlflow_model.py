"""Register the bundled demo XGBoost model as the MLflow production model.

This script is intentionally idempotent. It is used after DB seed so the
runtime stack has a real MLflow Registry alias and a matching ACTIVE row in
public.th_ml_model_version.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.xgboost
import psycopg2
import xgboost as xgb
from mlflow.tracking import MlflowClient

from app.config import get_settings


BOOTSTRAP_TAG = "fabbear.bootstrap"
MODEL_SHA_TAG = "fabbear.model_sha256"


def _project_root() -> Path:
    return PROJECT_ROOT


def _resolve_model_dir() -> Path:
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    if model_dir.is_absolute():
        return model_dir
    return _project_root() / model_dir


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_bootstrap_version(
    client: MlflowClient,
    model_name: str,
    model_sha: str,
) -> tuple[str, str] | None:
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None

    for version in sorted(versions, key=lambda item: int(item.version), reverse=True):
        run = client.get_run(version.run_id)
        tags = run.data.tags
        if tags.get(BOOTSTRAP_TAG) == "true" and tags.get(MODEL_SHA_TAG) == model_sha:
            return version.version, version.run_id
    return None


def _register_model(
    client: MlflowClient,
    model_name: str,
    model_path: Path,
    model_sha: str,
    feature_cols: list[str],
    train_meta: dict[str, Any],
) -> tuple[str, str]:
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "FabGuard_Bottleneck_Prediction"))
    with mlflow.start_run(run_name="seed-production-model") as run:
        run_id = run.info.run_id
        metrics = train_meta.get("metrics", {})
        mlflow.set_tag(BOOTSTRAP_TAG, "true")
        mlflow.set_tag(MODEL_SHA_TAG, model_sha)
        mlflow.set_tag("fabbear.source", "AI-Agent/models/bottleneck_xgb.ubj")
        mlflow.log_param("model_type", train_meta.get("model_type", "kpi_snapshot"))
        mlflow.log_param("feature_count", len(feature_cols))
        if "lookahead_min" in train_meta:
            mlflow.log_param("lookahead_min", train_meta["lookahead_min"])

        mlflow.log_metric("accuracy", _as_float(os.environ.get("BOOTSTRAP_MODEL_ACCURACY"), 0.9440))
        mlflow.log_metric("val_best_f1", _as_float(os.environ.get("BOOTSTRAP_MODEL_F1"), 0.9125))
        mlflow.log_metric(
            "auc_roc",
            _as_float(os.environ.get("BOOTSTRAP_MODEL_AUC"), _as_float(metrics.get("roc_auc"), 0.9720)),
        )

        feature_file = _resolve_model_dir() / "bottleneck_feature_cols.json"
        train_meta_file = _resolve_model_dir() / "train_meta.json"
        if feature_file.exists():
            mlflow.log_artifact(str(feature_file))
        if train_meta_file.exists():
            mlflow.log_artifact(str(train_meta_file))

        mlflow.xgboost.log_model(
            booster,
            artifact_path="model",
            registered_model_name=model_name,
            model_format="ubj",
        )

    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
    if not versions:
        raise RuntimeError(f"MLflow model version was not created for run_id={run_id}")
    version = max(versions, key=lambda item: int(item.version)).version
    return version, run_id


def _sync_db(
    model_name: str,
    version: str,
    run_id: str,
    feature_cols: list[str],
    train_meta: dict[str, Any],
) -> str:
    settings = get_settings()
    metrics = train_meta.get("metrics", {})
    accuracy = _as_float(os.environ.get("BOOTSTRAP_MODEL_ACCURACY"), 0.9440)
    f1_score = _as_float(os.environ.get("BOOTSTRAP_MODEL_F1"), 0.9125)
    auc_roc = _as_float(os.environ.get("BOOTSTRAP_MODEL_AUC"), _as_float(metrics.get("roc_auc"), 0.9720))
    train_rows = int(os.environ.get("BOOTSTRAP_MODEL_TRAIN_ROWS", "100000"))

    with psycopg2.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.th_ml_model_version
                SET status = 'RETIRED'
                WHERE status = 'ACTIVE'
                  AND (model_name <> %s OR mlflow_version <> %s)
                """,
                (model_name, str(version)),
            )
            cur.execute(
                """
                INSERT INTO public.th_ml_model_version (
                    mlflow_run_id,
                    mlflow_version,
                    model_name,
                    trained_at,
                    accuracy,
                    f1_score,
                    auc_roc,
                    train_row_count,
                    feature_list,
                    status
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    'ACTIVE'
                )
                ON CONFLICT (model_name, mlflow_version)
                DO UPDATE SET
                    mlflow_run_id = EXCLUDED.mlflow_run_id,
                    trained_at = EXCLUDED.trained_at,
                    accuracy = EXCLUDED.accuracy,
                    f1_score = EXCLUDED.f1_score,
                    auc_roc = EXCLUDED.auc_roc,
                    train_row_count = EXCLUDED.train_row_count,
                    feature_list = EXCLUDED.feature_list,
                    status = 'ACTIVE'
                RETURNING model_version_id
                """,
                (
                    run_id,
                    str(version),
                    model_name,
                    datetime.now(UTC),
                    accuracy,
                    f1_score,
                    auc_roc,
                    train_rows,
                    json.dumps(feature_cols, ensure_ascii=False),
                ),
            )
            model_version_id = str(cur.fetchone()[0])
    return model_version_id


def main() -> None:
    settings = get_settings()
    model_name = settings.mlflow_model_name
    alias = settings.mlflow_model_stage.lower()
    model_dir = _resolve_model_dir()
    model_path = model_dir / "bottleneck_xgb.ubj"
    if not model_path.exists():
        raise FileNotFoundError(f"Bundled model not found: {model_path}")

    feature_cols = _read_json(model_dir / "bottleneck_feature_cols.json", [])
    train_meta = _read_json(model_dir / "train_meta.json", {})
    model_sha = _sha256(model_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    existing = _find_bootstrap_version(client, model_name, model_sha)
    if existing:
        version, run_id = existing
    else:
        version, run_id = _register_model(
            client,
            model_name,
            model_path,
            model_sha,
            feature_cols,
            train_meta,
        )

    client.set_registered_model_alias(model_name, alias, version)
    model_version_id = _sync_db(model_name, version, run_id, feature_cols, train_meta)

    print(
        json.dumps(
            {
                "model_name": model_name,
                "alias": alias,
                "mlflow_version": str(version),
                "mlflow_run_id": run_id,
                "db_model_version_id": model_version_id,
                "source": "existing" if existing else "registered",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
