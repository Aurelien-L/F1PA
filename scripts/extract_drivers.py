"""
Standalone script to extract drivers data after Transform step 01.

This script handles the architectural dependency:
- extract_drivers.py needs sessions_scope from Transform step 01
- But logically belongs to Extract phase

Usage:
    python run_extract_drivers_standalone.py --years 2023 2024 2025
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[extract_drivers_standalone] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract drivers after Transform step 01")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args()

    years = args.years
    years_str = "_".join(str(y) for y in years)

    # Check if sessions_scope exists
    sessions_scope = PROJECT_ROOT / "data" / "transform" / f"sessions_scope_{years_str}.csv"

    if not sessions_scope.exists():
        log(f"ERROR: sessions_scope not found: {sessions_scope}")
        log("Please run Transform step 01 first:")
        log(f"  python etl/transform/01_build_sessions_scope.py --years {' '.join(map(str, years))}")
        return 1

    log(f"Found sessions_scope: {sessions_scope}")
    log(f"Extracting drivers for years: {years}")

    # Run extract_drivers
    cmd = [
        sys.executable,
        "etl/extract/openf1/extract_drivers.py",
        "--sessions-scope",
        str(sessions_scope),
    ]

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        log("Drivers extraction complete!")
        log(f"Output: data/extract/openf1/openf1_drivers_{years_str}.csv")
        return 0
    else:
        log(f"ERROR: Script failed with code {result.returncode}")
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
