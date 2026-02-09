"""
Load dim_circuits dimension table.

Extracts unique circuits from sessions_scope CSV and populates dim_circuits.
Includes wikipedia_circuit_url and station_id from Transform output.
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
    print(f"[02_load_dim_circuits] {msg}", flush=True)


def load_circuits(sessions_path: Path, config: DBConfig, truncate: bool = True) -> int:
    """
    Load dim_circuits from sessions scope data.

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

    required_cols = ["circuit_key", "circuit_short_name", "location", "country_name", "country_code"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select circuit columns
    circuit_cols = [
        "circuit_key",
        "circuit_short_name",
        "location",
        "country_name",
        "country_code",
        "wikipedia_circuit_url",
        "station_id",
    ]
    existing_cols = [col for col in circuit_cols if col in df.columns]

    df_circuits = df[existing_cols].copy()

    # Deduplicate on circuit_key
    initial_count = len(df_circuits)
    df_circuits = df_circuits.drop_duplicates(subset=["circuit_key"], keep="first")
    _log(f"Unique circuits: {initial_count} -> {len(df_circuits)}")

    # Sort by circuit_key
    df_circuits = df_circuits.sort_values("circuit_key").reset_index(drop=True)

    # Handle missing columns
    for col in circuit_cols:
        if col not in df_circuits.columns:
            df_circuits[col] = None

    engine = create_db_engine(config)

    with engine.begin() as conn:
        if truncate:
            _log("Truncating dim_circuits...")
            conn.execute(text("TRUNCATE TABLE dim_circuits CASCADE"))

        _log(f"Inserting {len(df_circuits)} circuits...")
        df_circuits.to_sql(
            "dim_circuits",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
        )

    _log(f"Loaded {len(df_circuits)} circuits into dim_circuits")
    return len(df_circuits)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load dim_circuits from sessions scope")
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

    rows = load_circuits(sessions_path, config, truncate=not args.no_truncate)
    _log(f"Load complete: {rows} circuits")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
