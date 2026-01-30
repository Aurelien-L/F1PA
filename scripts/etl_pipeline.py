"""
F1PA - Full ETL Pipeline Orchestrator

Executes the complete Extract → Transform → Load pipeline in the correct order,
handling dependencies (e.g., extract_drivers requires Transform step 01).

Usage:
    python run_full_pipeline.py --years 2023 2024 2025
    python run_full_pipeline.py --years 2023 2024 2025 --skip-extract  # Skip Extract if already done
    python run_full_pipeline.py --years 2023 2024 2025 --skip-load     # Skip Load if DB already populated
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def log(msg: str, level: str = "INFO") -> None:
    """Log with timestamp and level."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "[i]",
        "SUCCESS": "[OK]",
        "ERROR": "[ERROR]",
        "STEP": "[>>]",
    }.get(level, "[-]")
    print(f"[{ts}] {prefix} {msg}", flush=True)


def run_command(cmd: list[str], description: str, cwd: Path = PROJECT_ROOT) -> None:
    """Execute a command and handle errors."""
    log(f"{description}", "STEP")
    log(f"Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"FAILED after {elapsed:.1f}s (exit code {result.returncode})", "ERROR")
        raise RuntimeError(f"Pipeline failed at: {description}")

    log(f"Completed in {elapsed:.1f}s", "SUCCESS")


def check_prerequisites() -> None:
    """Check that required tools are available."""
    log("Checking prerequisites...", "STEP")

    # Check Python
    log(f"Python: {sys.version.split()[0]}")

    # Check Docker
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        log(f"Docker: {result.stdout.strip()}")
    except Exception as e:
        log(f"Docker not found: {e}", "ERROR")
        raise RuntimeError("Docker is required for PostgreSQL")

    # Check venv
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    if not venv_python.exists():
        log("Virtual environment not found at .venv/", "ERROR")
        raise RuntimeError("Please create virtual environment first: python -m venv .venv")

    log(f"Using Python: {venv_python}")
    log("Prerequisites OK", "SUCCESS")


def check_data_exists(years: list[int]) -> dict[str, bool]:
    """Check which data files already exist."""
    years_str = "_".join(str(y) for y in years)

    checks = {
        "extract_sessions": (PROJECT_ROOT / "data" / "extract" / "openf1" / f"sessions_openf1_{years[0]}_{years[-1]}.csv").exists(),
        "extract_drivers": (PROJECT_ROOT / "data" / "extract" / "openf1" / f"openf1_drivers_{years_str}.csv").exists(),
        "sessions_scope": (PROJECT_ROOT / "data" / "transform" / f"sessions_scope_{years_str}.csv").exists(),
        "dataset_ml": (PROJECT_ROOT / "data" / "processed" / f"dataset_ml_lap_level_{years_str}.csv").exists(),
    }

    return checks


def run_extract(years: list[int], force: bool = False) -> None:
    """Execute Extract phase."""
    log("=" * 80)
    log("PHASE 1: EXTRACT", "STEP")
    log("=" * 80)

    existing = check_data_exists(years)
    if existing["extract_sessions"] and not force:
        log("Extract data already exists (use --force to re-extract)", "INFO")
        return

    years_args = [str(y) for y in years]

    # Run extract_all orchestrator
    run_command(
        [sys.executable, "-m", "etl.extract.run_extract_all",
         "--years", *years_args,
         "--wiki-sleep", "0.5",
         "--top-n", "15",
         "--purge-raw"],
        "Extract: OpenF1 + Wikipedia + Meteostat"
    )


def run_transform_step_01(years: list[int]) -> None:
    """Execute Transform step 01 (sessions_scope)."""
    years_args = [str(y) for y in years]

    run_command(
        [sys.executable, "etl/transform/01_build_sessions_scope.py",
         "--years", *years_args],
        "Transform Step 01: Build sessions scope"
    )


def run_extract_drivers(years: list[int]) -> None:
    """Execute Extract drivers (depends on Transform step 01)."""
    years_str = "_".join(str(y) for y in years)
    sessions_scope = PROJECT_ROOT / "data" / "transform" / f"sessions_scope_{years_str}.csv"

    if not sessions_scope.exists():
        raise RuntimeError(f"sessions_scope not found: {sessions_scope}")

    run_command(
        [sys.executable, "etl/extract/openf1/extract_drivers.py",
         "--sessions-scope", str(sessions_scope)],
        "Extract: Drivers data from OpenF1"
    )


def run_transform(years: list[int], force: bool = False) -> None:
    """Execute Transform phase."""
    log("=" * 80)
    log("PHASE 2: TRANSFORM", "STEP")
    log("=" * 80)

    existing = check_data_exists(years)

    # Step 01: sessions_scope (required for drivers)
    if not existing["sessions_scope"] or force:
        run_transform_step_01(years)
    else:
        log("sessions_scope already exists", "INFO")

    # Extract drivers (architectural dependency)
    if not existing["extract_drivers"] or force:
        run_extract_drivers(years)
    else:
        log("Drivers data already exists", "INFO")

    # Steps 02-06: Complete Transform pipeline
    if not existing["dataset_ml"] or force:
        years_args = [str(y) for y in years]
        run_command(
            [sys.executable, "etl/transform/run_transform_all.py",
             "--years", *years_args],
            "Transform Steps 02-06: Build ML dataset"
        )
    else:
        log("ML dataset already exists", "INFO")


def run_load(years: list[int], force: bool = False) -> None:
    """Execute Load phase."""
    log("=" * 80)
    log("PHASE 3: LOAD", "STEP")
    log("=" * 80)

    # Check PostgreSQL is running
    log("Checking PostgreSQL container...", "STEP")
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=f1pa_postgres", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if "f1pa_postgres" not in result.stdout:
        log("Starting PostgreSQL container...", "INFO")
        subprocess.run(["docker-compose", "up", "-d", "postgres"], cwd=PROJECT_ROOT, check=True)
        log("Waiting for PostgreSQL to be ready...", "INFO")
        time.sleep(10)
    else:
        log("PostgreSQL already running", "INFO")

    # Initialize schema
    log("Initializing PostgreSQL schema...", "STEP")
    schema_path = PROJECT_ROOT / "etl" / "load" / "schema.sql"
    subprocess.run(
        ["docker", "exec", "-i", "f1pa_postgres", "psql", "-U", "f1pa", "-d", "f1pa_db"],
        stdin=open(schema_path),
        check=True
    )
    log("Schema initialized", "SUCCESS")

    # Load data
    run_command(
        [sys.executable, "etl/load/load_all_docker.py"],
        "Load: Populate PostgreSQL database"
    )


def verify_data_quality(years: list[int]) -> None:
    """Verify data quality and consistency."""
    log("=" * 80)
    log("DATA QUALITY VERIFICATION", "STEP")
    log("=" * 80)

    years_str = "_".join(str(y) for y in years)

    # Check dataset exists
    dataset_path = PROJECT_ROOT / "data" / "processed" / f"dataset_ml_lap_level_{years_str}.csv"
    report_path = dataset_path.with_suffix(".report.json")

    if not dataset_path.exists():
        log(f"Dataset not found: {dataset_path}", "ERROR")
        raise RuntimeError("ML dataset missing")

    # Check report
    if report_path.exists():
        import json
        with open(report_path) as f:
            report = json.load(f)

        log(f"Dataset rows: {report['summary']['n_rows']:,}", "INFO")
        log(f"Dataset cols: {report['summary']['n_cols']}", "INFO")
        log(f"Duplicates: {report['summary']['n_duplicates_key']}", "INFO")
        log(f"Missing target: {report['summary']['missing_target']}", "INFO")

        if report['summary']['n_duplicates_key'] > 0:
            log("WARNING: Duplicates found in dataset", "ERROR")

        if report['summary']['missing_target'] > 0:
            log("WARNING: Missing target values", "ERROR")

    # Check PostgreSQL data
    log("Checking PostgreSQL data...", "STEP")
    result = subprocess.run(
        ["docker", "exec", "f1pa_postgres", "psql", "-U", "f1pa", "-d", "f1pa_db", "-c",
         "SELECT 'dim_circuits' as table, COUNT(*) FROM dim_circuits UNION ALL "
         "SELECT 'dim_drivers', COUNT(*) FROM dim_drivers UNION ALL "
         "SELECT 'dim_sessions', COUNT(*) FROM dim_sessions UNION ALL "
         "SELECT 'fact_laps', COUNT(*) FROM fact_laps;"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        log("PostgreSQL row counts:", "INFO")
        for line in result.stdout.split("\n"):
            if "|" in line and "table" not in line and "---" not in line:
                log(f"  {line.strip()}", "INFO")

    log("Data quality verification complete", "SUCCESS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="F1PA Full ETL Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py --years 2023 2024 2025
  python run_full_pipeline.py --years 2023 2024 2025 --skip-extract
  python run_full_pipeline.py --years 2023 2024 2025 --force
        """
    )
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                        help="Years to process (default: 2023 2024 2025)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip Extract phase (use existing data)")
    parser.add_argument("--skip-transform", action="store_true",
                        help="Skip Transform phase (use existing data)")
    parser.add_argument("--skip-load", action="store_true",
                        help="Skip Load phase (keep existing DB)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-execution even if data exists")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run data quality verification")

    args = parser.parse_args()

    try:
        log("=" * 80)
        log("F1PA - FULL ETL PIPELINE ORCHESTRATOR", "STEP")
        log("=" * 80)
        log(f"Years: {args.years}")
        log(f"Skip Extract: {args.skip_extract}")
        log(f"Skip Transform: {args.skip_transform}")
        log(f"Skip Load: {args.skip_load}")
        log(f"Force: {args.force}")
        log("")

        start_time = time.time()

        # Prerequisites check
        if not args.verify_only:
            check_prerequisites()

        # Show existing data
        existing = check_data_exists(args.years)
        log("Existing data:", "INFO")
        for key, exists in existing.items():
            status = "[OK]" if exists else "[NO]"
            log(f"  {status} {key}", "INFO")
        log("")

        # Verify only mode
        if args.verify_only:
            verify_data_quality(args.years)
            return 0

        # Execute pipeline phases
        if not args.skip_extract:
            run_extract(args.years, force=args.force)
        else:
            log("Skipping Extract phase (--skip-extract)", "INFO")

        if not args.skip_transform:
            run_transform(args.years, force=args.force)
        else:
            log("Skipping Transform phase (--skip-transform)", "INFO")

        if not args.skip_load:
            run_load(args.years, force=args.force)
        else:
            log("Skipping Load phase (--skip-load)", "INFO")

        # Verify data quality
        verify_data_quality(args.years)

        # Success summary
        elapsed = time.time() - start_time
        log("=" * 80)
        log(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes", "SUCCESS")
        log("=" * 80)
        log("Next steps:", "INFO")
        log("  1. Train ML model: python ml/train_model.py", "INFO")
        log("  2. Start API: python api/main.py", "INFO")
        log("  3. Launch UI: streamlit run streamlit/app.py", "INFO")

        return 0

    except Exception as e:
        log(f"Pipeline failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
