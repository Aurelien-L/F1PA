"""
Database configuration and connection utilities for F1PA PostgreSQL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class DBConfig:
    """PostgreSQL database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "f1pa_db"
    user: str = "f1pa"
    password: str = "f1pa"

    @classmethod
    def from_env(cls) -> DBConfig:
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "f1pa_db"),
            user=os.getenv("POSTGRES_USER", "f1pa"),
            password=os.getenv("POSTGRES_PASSWORD", "f1pa"),
        )

    def get_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


def create_db_engine(config: Optional[DBConfig] = None) -> Engine:
    """
    Create SQLAlchemy engine for PostgreSQL.

    Args:
        config: Database configuration (uses env defaults if None)

    Returns:
        SQLAlchemy Engine
    """
    if config is None:
        config = DBConfig.from_env()

    conn_str = config.get_connection_string()
    return create_engine(conn_str, echo=False)


def test_connection(engine: Engine) -> bool:
    """
    Test database connection.

    Args:
        engine: SQLAlchemy Engine

    Returns:
        True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 AS test"))
            row = result.fetchone()
            return row[0] == 1
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


def _log(msg: str) -> None:
    print(f"[db_config] {msg}", flush=True)
