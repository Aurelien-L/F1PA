"""
Load dim_sessions dimension table.

Reads sessions_scope CSV and populates dim_sessions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db_config import DBConfig, create_db_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_TRANSFORM_DIR = PROJECT_ROOT / "data" / "transform"
DEFAULT_SESSIONS_PATH = DATA_TRANSFORM_DIR / "sessions_scope_2023_2024_2025.csv"


def _log(msg: str) -> None:
    print(f"[04_load_dim_sessions] {msg}", flush=True)


def load_sessions(sessions_path: Path, config: DBConfig, truncate: bool = True) -> int:
    """
    Load dim_sessions from sessions scope.

    Args:
        sessions_path: Path to sessions_scope CSV
        config: Database configuration
        truncate: Whether to truncate table before loading

    Returns:
        Number of rows inserted
    """
    if not sessions_path.exists():
        raise FileNotFoundError(f"Sessions file not found: {sessions_path}")

    _log(f"Reading sessions: {sessions_path}")
    df = pd.read_csv(sessions_path)

    required_cols = ["session_key", "meeting_key", "year", "circuit_key"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select session columns (align with dim_sessions schema)
    session_cols = [
        "session_key",
        "meeting_key",
        "year",
        "session_name",
        "session_type",
        "date_start",
        "date_end",
        "gmt_offset",
        "circuit_key",
    ]

    existing_cols = [col for col in session_cols if col in df.columns]
    df_sessions = df[existing_cols].copy()

    # Parse dates
    for col in ["date_start", "date_end"]:
        if col in df_sessions.columns:
            df_sessions[col] = pd.to_datetime(df_sessions[col], errors="coerce", utc=True)

    # Deduplicate on session_key
    initial_count = len(df_sessions)
    df_sessions = df_sessions.drop_duplicates(subset=["session_key"], keep="first")
    if initial_count != len(df_sessions):
        _log(f"Deduplication: {initial_count} -> {len(df_sessions)}")

    # Sort by year, meeting_key
    df_sessions = df_sessions.sort_values(["year", "meeting_key"]).reset_index(drop=True)

    # Handle missing columns
    for col in session_cols:
        if col not in df_sessions.columns:
            df_sessions[col] = None

    engine = create_db_engine(config)

    with engine.begin() as conn:
        if truncate:
            _log("Truncating dim_sessions...")
            conn.execute(text("TRUNCATE TABLE dim_sessions CASCADE"))

        _log(f"Inserting {len(df_sessions)} sessions...")
        df_sessions.to_sql(
            "dim_sessions",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
        )

    _log(f"Loaded {len(df_sessions)} sessions into dim_sessions")

    # Summary
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT year, COUNT(*) as sessions
                FROM dim_sessions
                GROUP BY year
                ORDER BY year
            """)
        )
        _log("\nSessions by year:")
        for row in result:
            _log(f"  {row[0]}: {row[1]} sessions")

    return len(df_sessions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load dim_sessions from sessions scope")
    parser.add_argument(
        "--sessions",
        type=str,
        default=str(DEFAULT_SESSIONS_PATH),
        help="Path to sessions_scope CSV",
    )
    parser.add_argument("--no-truncate", action="store_true", help="Do not truncate table before loading")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", type=str, default="f1pa_db")
    parser.add_argument("--user", type=str, default="f1pa")
    parser.add_argument("--password", type=str, default="f1pa")

    args = parser.parse_args()

    sessions_path = Path(args.sessions)
    if not sessions_path.is_absolute():
        sessions_path = PROJECT_ROOT / sessions_path

    config = DBConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )

    rows = load_sessions(sessions_path, config, truncate=not args.no_truncate)
    _log(f"Load complete: {rows} sessions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
