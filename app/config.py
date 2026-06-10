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
    qdrant_collection_docs: str = "knowledge_docs"
    mlflow_tracking_uri: str = "http://localhost:5000"
    agent_csv_dir: str = "../Simulation/simulation/sim_csv_out"
    model_dir: str = "models"
    agent_step_timeout_sec: int = 30
    pipeline_timeout_sec: int = 1200
    risk_critical_threshold: float = 0.90
    risk_high_threshold: float = 0.85
    risk_medium_threshold: float = 0.70
    bottleneck_threshold: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
