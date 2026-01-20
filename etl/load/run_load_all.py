"""
Orchestrator script to run the complete LOAD pipeline.

Executes all load steps in sequence:
1. Initialize schema
2. Load dim_circuits
3. Load dim_drivers
4. Load dim_sessions
5. Load fact_laps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db_config import DBConfig, create_db_engine, test_connection


# Import load modules
sys.path.insert(0, str(Path(__file__).parent))

# Dynamic imports to avoid circular dependencies
def _import_module(name: str):
    """Dynamic import helper."""
    return __import__(name)


def _log(msg: str) -> None:
    print(f"[run_load_all] {msg}", flush=True)


def run_load_pipeline(config: DBConfig, skip_schema: bool = False) -> dict[str, int]:
    """
    Run complete LOAD pipeline.

    Args:
        config: Database configuration
        skip_schema: Skip schema initialization (if already exists)

    Returns:
        Dictionary with row counts per table
    """
    results = {}

    # Test connection first
    _log("Testing database connection...")
    engine = create_db_engine(config)
    if not test_connection(engine):
        raise RuntimeError("Database connection failed. Check PostgreSQL is running.")
    _log("Connection OK")

    # Step 1: Initialize schema
    if not skip_schema:
        _log("\n=== STEP 1: Initialize Schema ===")
        from importlib import import_module
        init_schema_module = import_module("01_init_schema")
        schema_path = Path(__file__).parent / "schema.sql"
        init_schema_module.init_schema(schema_path, config)
        _log("Schema initialized")
    else:
        _log("\n=== STEP 1: Skip Schema (already exists) ===")

    # Step 2: Load dim_circuits
    _log("\n=== STEP 2: Load dim_circuits ===")
    from importlib import import_module
    load_circuits_module = import_module("02_load_dim_circuits")
    sessions_path = Path(__file__).parents[2] / "data" / "transform" / "sessions_scope_2023_2024_2025.csv"
    rows_circuits = load_circuits_module.load_circuits(sessions_path, config, truncate=True)
    results["dim_circuits"] = rows_circuits

    # Step 3: Load dim_drivers
    _log("\n=== STEP 3: Load dim_drivers ===")
    load_drivers_module = import_module("03_load_dim_drivers")
    drivers_path = Path(__file__).parents[2] / "data" / "extract" / "openf1" / "openf1_drivers_2023_2024_2025.csv"
    rows_drivers = load_drivers_module.load_drivers(drivers_path, config, truncate=True)
    results["dim_drivers"] = rows_drivers

    # Step 4: Load dim_sessions
    _log("\n=== STEP 4: Load dim_sessions ===")
    load_sessions_module = import_module("04_load_dim_sessions")
    rows_sessions = load_sessions_module.load_sessions(sessions_path, config, truncate=True)
    results["dim_sessions"] = rows_sessions

    # Step 5: Load fact_laps
    _log("\n=== STEP 5: Load fact_laps ===")
    load_laps_module = import_module("05_load_fact_laps")
    dataset_path = Path(__file__).parents[2] / "data" / "processed" / "dataset_ml_lap_level_2023_2024_2025.csv"
    rows_laps = load_laps_module.load_laps(dataset_path, config, truncate=True, batch_size=5000)
    results["fact_laps"] = rows_laps

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run complete F1PA LOAD pipeline")
    parser.add_argument("--skip-schema", action="store_true", help="Skip schema initialization")
    parser.add_argument("--host", type=str, default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", type=str, default="f1pa_db", help="Database name")
    parser.add_argument("--user", type=str, default="f1pa", help="Database user")
    parser.add_argument("--password", type=str, default="f1pa", help="Database password")

    args = parser.parse_args()

    config = DBConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )

    _log("=" * 80)
    _log("F1PA LOAD PIPELINE")
    _log("=" * 80)
    _log(f"Database: {config.database}@{config.host}:{config.port}")
    _log("")

    try:
        results = run_load_pipeline(config, skip_schema=args.skip_schema)

        _log("\n" + "=" * 80)
        _log("LOAD PIPELINE COMPLETE")
        _log("=" * 80)
        _log("\nSummary:")
        for table, rows in results.items():
            _log(f"  {table}: {rows} rows")

        _log("\nDatabase ready for ML training and API development!")
        return 0

    except Exception as e:
        _log(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
