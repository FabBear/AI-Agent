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
    "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
)

_engine = None
_Session = None


def get_session():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(_SIM_DB_URL, pool_pre_ping=True)
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
