"""
Load fact_laps fact table.

Reads dataset_ml CSV (output from Transform) and populates fact_laps.
This is the main table containing all lap-level features and target variable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db_config import DBConfig, create_db_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATASET_PATH = DATA_PROCESSED_DIR / "dataset_ml_lap_level_2023_2024_2025.csv"


def _log(msg: str) -> None:
    print(f"[05_load_fact_laps] {msg}", flush=True)


def load_laps(dataset_path: Path, config: DBConfig, truncate: bool = True, batch_size: int = 5000) -> int:
    """
    Load fact_laps from ML dataset.

    Args:
        dataset_path: Path to dataset_ml CSV
        config: Database configuration
        truncate: Whether to truncate table before loading
        batch_size: Number of rows per batch insert

    Returns:
        Number of rows inserted
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    _log(f"Reading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    _log(f"Dataset shape: {df.shape}")

    # Required columns (align with fact_laps schema)
    required_cols = [
        "meeting_key",
        "session_key",
        "driver_number",
        "lap_number",
        "year",
        "circuit_key",
        "lap_duration",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select all columns for fact_laps
    fact_cols = [
        # Primary key
        "meeting_key",
        "session_key",
        "driver_number",
        "lap_number",
        # Context
        "year",
        "circuit_key",
        "lap_hour_utc",
        # Sport features
        "st_speed",
        "i1_speed",
        "i2_speed",
        "duration_sector_1",
        "duration_sector_2",
        "duration_sector_3",
        # Weather features
        "temp",
        "rhum",
        "pres",
        "wspd",
        "wdir",
        "prcp",
        "cldc",
        # Target
        "lap_duration",
        # Metadata
        "__source_file",
    ]

    existing_cols = [col for col in fact_cols if col in df.columns]
    df_laps = df[existing_cols].copy()

    # Rename __source_file to source_file
    if "__source_file" in df_laps.columns:
        df_laps = df_laps.rename(columns={"__source_file": "source_file"})

    # Parse dates
    if "lap_hour_utc" in df_laps.columns:
        df_laps["lap_hour_utc"] = pd.to_datetime(df_laps["lap_hour_utc"], errors="coerce", utc=True)

    # Cast numeric types
    int_cols = ["meeting_key", "session_key", "driver_number", "lap_number", "year", "circuit_key"]
    for col in int_cols:
        if col in df_laps.columns:
            df_laps[col] = pd.to_numeric(df_laps[col], errors="coerce").astype("Int64")

    # Deduplicate on composite key (should already be unique)
    key_cols = ["meeting_key", "session_key", "driver_number", "lap_number"]
    initial_count = len(df_laps)
    df_laps = df_laps.drop_duplicates(subset=key_cols, keep="first")
    if initial_count != len(df_laps):
        _log(f"WARNING: Duplicates removed: {initial_count} -> {len(df_laps)}")

    # Filter out rows with null target
    df_laps = df_laps[df_laps["lap_duration"].notna()].copy()
    _log(f"After filtering null targets: {len(df_laps)} laps")

    engine = create_db_engine(config)

    with engine.begin() as conn:
        if truncate:
            _log("Truncating fact_laps...")
            conn.execute(text("TRUNCATE TABLE fact_laps CASCADE"))

        _log(f"Inserting {len(df_laps)} laps in batches of {batch_size}...")

        # Insert in batches for better performance
        for i in range(0, len(df_laps), batch_size):
            batch = df_laps.iloc[i : i + batch_size]
            batch.to_sql(
                "fact_laps",
                con=conn,
                if_exists="append",
                index=False,
                method="multi",
            )
            _log(f"  Inserted batch {i // batch_size + 1}/{(len(df_laps) - 1) // batch_size + 1} ({len(batch)} rows)")

    _log(f"Loaded {len(df_laps)} laps into fact_laps")

    # Summary statistics
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    COUNT(*) as total_laps,
                    COUNT(DISTINCT session_key) as sessions,
                    COUNT(DISTINCT driver_number) as drivers,
                    MIN(year) as min_year,
                    MAX(year) as max_year,
                    ROUND(AVG(lap_duration)::numeric, 3) as avg_lap_duration,
                    COUNT(CASE WHEN temp IS NULL THEN 1 END) as missing_temp
                FROM fact_laps
            """)
        )
        stats = result.fetchone()
        _log("\nFact table statistics:")
        _log(f"  Total laps: {stats[0]}")
        _log(f"  Unique sessions: {stats[1]}")
        _log(f"  Unique drivers: {stats[2]}")
        _log(f"  Year range: {stats[3]}-{stats[4]}")
        _log(f"  Avg lap duration: {stats[5]}s")
        _log(f"  Missing temperature: {stats[6]} laps")

    return len(df_laps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load fact_laps from ML dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to dataset_ml CSV",
    )
    parser.add_argument("--no-truncate", action="store_true", help="Do not truncate table before loading")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for inserts")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", type=str, default="f1pa_db")
    parser.add_argument("--user", type=str, default="f1pa")
    parser.add_argument("--password", type=str, default="f1pa")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    config = DBConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )

    rows = load_laps(dataset_path, config, truncate=not args.no_truncate, batch_size=args.batch_size)
    _log(f"Load complete: {rows} laps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
