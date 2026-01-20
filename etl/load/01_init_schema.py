"""
Initialize PostgreSQL database schema.

Reads schema.sql and executes DDL statements to create tables, indexes, and constraints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import text

from db_config import DBConfig, create_db_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_SQL_PATH = PROJECT_ROOT / "etl" / "load" / "schema.sql"


def _log(msg: str) -> None:
    print(f"[01_init_schema] {msg}", flush=True)


def init_schema(schema_path: Path, config: DBConfig) -> None:
    """
    Execute schema SQL file to initialize database.

    Args:
        schema_path: Path to schema.sql file
        config: Database configuration
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    _log(f"Reading schema: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    engine = create_db_engine(config)

    _log(f"Connecting to: {config.get_connection_string()}")

    # Execute schema as a single transaction
    with engine.begin() as conn:
        _log("Executing schema DDL...")
        conn.execute(text(schema_sql))
        _log("Schema created successfully")

    # Verify tables created
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
        )
        tables = [row[0] for row in result]
        _log(f"Tables created: {tables}")

        expected_tables = {"dim_circuits", "dim_drivers", "dim_sessions", "fact_laps"}
        if set(tables) >= expected_tables:
            _log("All expected tables present")
        else:
            missing = expected_tables - set(tables)
            raise RuntimeError(f"Missing tables: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize F1PA PostgreSQL schema")
    parser.add_argument(
        "--schema",
        type=str,
        default=str(SCHEMA_SQL_PATH),
        help="Path to schema.sql file",
    )
    parser.add_argument("--host", type=str, default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", type=str, default="f1pa_db", help="Database name")
    parser.add_argument("--user", type=str, default="f1pa", help="Database user")
    parser.add_argument("--password", type=str, default="f1pa", help="Database password")

    args = parser.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = PROJECT_ROOT / schema_path

    config = DBConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )

    init_schema(schema_path, config)
    _log("Schema initialization complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
