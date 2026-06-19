"""Simulation DB connection — uses same Postgres as Simulation repo."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Simulation DB — default matches Simulation/docker-compose.yml
_SIM_DB_URL = os.getenv(
    "SIM_DATABASE_URL",
    "postgresql+psycopg://fabbear_user@localhost:5432/fabbear",
)
# Schema where Simulation ORM tables live (matches Simulation/simulation/schema_config.py)
_SIM_SCHEMA = os.getenv("POSTGRES_SCHEMA", "simulation")

_engine = None
_Session = None


def get_session():
    global _engine, _Session
    if _engine is None:
        connect_args: dict = {}
        if _SIM_SCHEMA and _SIM_SCHEMA != "public":
            connect_args["options"] = f'-csearch_path="{_SIM_SCHEMA}",public'
        # 커넥션 상한 8 (pool_size 5 + max_overflow 3). 기본값(5+10=15)은 sim 서브프로세스 풀과
        # 합산 시 Postgres max_connections 압박 → 명시적으로 낮춤.
        _engine = create_engine(
            _SIM_DB_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=3,
            connect_args=connect_args,
        )
        _Session = sessionmaker(bind=_engine)
    return _Session()


def query_df(sql: str, params: dict | None = None):
    """Run raw SQL and return pd.DataFrame."""
    import pandas as pd

    with get_session() as session:
        result = session.execute(text(sql), params or {})
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=cols)
