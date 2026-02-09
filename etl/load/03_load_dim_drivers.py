"""
Load dim_drivers dimension table.

Reads drivers data from OpenF1 extract and populates dim_drivers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db_config import DBConfig, create_db_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_EXTRACT_DIR = PROJECT_ROOT / "data" / "extract" / "openf1"
DEFAULT_DRIVERS_PATH = DATA_EXTRACT_DIR / "openf1_drivers_2023_2024_2025.csv"


def _log(msg: str) -> None:
    print(f"[03_load_dim_drivers] {msg}", flush=True)


def load_drivers(drivers_path: Path, config: DBConfig, truncate: bool = True) -> int:
    """
    Load dim_drivers from OpenF1 drivers extract.

    Args:
        drivers_path: Path to drivers CSV
        config: Database configuration
        truncate: Whether to truncate table before loading

    Returns:
        Number of rows inserted
    """
    if not drivers_path.exists():
        raise FileNotFoundError(f"Drivers file not found: {drivers_path}")

    _log(f"Reading drivers: {drivers_path}")
    df = pd.read_csv(drivers_path)

    required_cols = ["driver_number", "full_name"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select relevant columns (align with dim_drivers schema)
    driver_cols = [
        "driver_number",
        "full_name",
        "broadcast_name",
        "name_acronym",
        "first_name",
        "last_name",
        "country_code",
        "team_name",
        "team_colour",
        "headshot_url",
    ]

    existing_cols = [col for col in driver_cols if col in df.columns]
    df_drivers = df[existing_cols].copy()

    # Deduplicate on driver_number (should already be unique from extract)
    initial_count = len(df_drivers)
    df_drivers = df_drivers.drop_duplicates(subset=["driver_number"], keep="first")
    if initial_count != len(df_drivers):
        _log(f"WARNING: Duplicates found: {initial_count} -> {len(df_drivers)}")

    # Sort by driver_number
    df_drivers = df_drivers.sort_values("driver_number").reset_index(drop=True)

    # Handle missing columns
    for col in driver_cols:
        if col not in df_drivers.columns:
            df_drivers[col] = None

    engine = create_db_engine(config)

    with engine.begin() as conn:
        if truncate:
            _log("Truncating dim_drivers...")
            conn.execute(text("TRUNCATE TABLE dim_drivers CASCADE"))

        _log(f"Inserting {len(df_drivers)} drivers...")
        df_drivers.to_sql(
            "dim_drivers",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
        )

    _log(f"Loaded {len(df_drivers)} drivers into dim_drivers")

    # Sample output
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT driver_number, name_acronym, full_name, team_name FROM dim_drivers ORDER BY driver_number LIMIT 5")
        )
        _log("\nSample drivers loaded:")
        for row in result:
            _log(f"  #{row[0]} {row[1]} - {row[2]} ({row[3]})")

    return len(df_drivers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load dim_drivers from OpenF1 extract")
    parser.add_argument(
        "--drivers",
        type=str,
        default=str(DEFAULT_DRIVERS_PATH),
        help="Path to drivers CSV",
    )
    parser.add_argument("--no-truncate", action="store_true", help="Do not truncate table before loading")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", type=str, default="f1pa_db")
    parser.add_argument("--user", type=str, default="f1pa")
    parser.add_argument("--password", type=str, default="f1pa")

    args = parser.parse_args()

    drivers_path = Path(args.drivers)
    if not drivers_path.is_absolute():
        drivers_path = PROJECT_ROOT / drivers_path

    config = DBConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )

    rows = load_drivers(drivers_path, config, truncate=not args.no_truncate)
    _log(f"Load complete: {rows} drivers")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
