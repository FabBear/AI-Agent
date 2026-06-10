"""Simulation DB connection — uses same Postgres as Simulation repo."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Simulation DB — default matches Simulation/docker-compose.yml
_SIM_DB_URL = os.getenv(
    "SIM_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
)
# Schema where Simulation ORM tables live (matches Simulation/simulation/schema_config.py)
_SIM_SCHEMA = os.getenv("POSTGRES_SCHEMA", "simulation")

_engine = None
_Session = None


def get_session():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(_SIM_DB_URL, pool_pre_ping=True)

        if _SIM_SCHEMA and _SIM_SCHEMA != "public":
            @event.listens_for(_engine, "connect")
            def _set_search_path(dbapi_conn, _record):
                cur = dbapi_conn.cursor()
                cur.execute(f"SET search_path TO {_SIM_SCHEMA}, public")
                cur.close()

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
