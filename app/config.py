"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://fabbear_user:fabbear_pw@localhost:5432/fabbear"
    spring_base_url: str = "http://localhost:8080"
    internal_api_token: str = "dev-internal-token"
    openai_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_cases: str = "bottleneck_cases"
    mlflow_tracking_uri: str = "http://localhost:5500"
    # [MLOps] MLflow Registry에서 끌어올 모델/스테이지 및 교체(refresh) 주기(초)
    mlflow_model_name: str = "FabBear_Bottleneck_Model"
    mlflow_model_stage: str = "Production"
    model_refresh_interval_sec: int = 300
    agent_csv_dir: str = "../Simulation/simulation/sim_csv_out"
    model_dir: str = "models"
    agent_step_timeout_sec: int = 30
    pipeline_timeout_sec: int = 3600
    # 케이스 병렬 처리 수. 실제 sim 부하는 sim_executor의 전역 세마포어(VERIFY_MAX_CONCURRENT_SIMS)가
    # 캡하므로 케이스를 병렬로 풀어도 동시 sim은 그 값을 안 넘는다. 1로 두면 케이스가 순차라 느림.
    max_concurrent_pipelines: int = 8
    risk_critical_threshold: float = 0.90
    risk_high_threshold: float = 0.85
    risk_medium_threshold: float = 0.70
    bottleneck_threshold: float = 0.50
    # [MLOps] 전체 예측 적재 시 isBottleneckPred 판정 임계값 (ml_g_star/detector와 일치)
    alarm_proba_threshold: float = 0.7
    # [MLOps] 라이브 추론 결과 적재용 run_id (Drift 평가 시 배치 식별자)
    live_run_id: str = "live_inference_run"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
