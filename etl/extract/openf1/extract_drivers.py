"""
Extract drivers data from OpenF1 API for all sessions in scope.

This script:
1. Reads sessions_scope CSV (output from Transform step 01)
2. Fetches drivers data for each session_key from OpenF1 API
3. Deduplicates drivers (driver_number as unique key)
4. Exports to CSV: openf1_drivers_<years>.csv

API: https://api.openf1.org/v1/drivers?session_key=<session_key>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_TRANSFORM_DIR = PROJECT_ROOT / "data" / "transform"
DATA_EXTRACT_DIR = PROJECT_ROOT / "data" / "extract" / "openf1"

BASE_URL = "https://api.openf1.org/v1/drivers"


def _log(msg: str) -> None:
    print(f"[extract_drivers] {msg}", flush=True)


def fetch_json(url: str, retries: int = 3, sleep_s: float = 1.0) -> list[dict[str, Any]]:
    """Fetch JSON from URL with retry logic."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch after {retries} retries: {url}\nLast error: {last_err}")


def extract_drivers_for_sessions(sessions_scope_path: Path, sleep_between_calls: float = 0.2) -> pd.DataFrame:
    """
    Fetch drivers data for all sessions in scope.

    Args:
        sessions_scope_path: Path to sessions_scope CSV
        sleep_between_calls: Polite delay between API calls

    Returns:
        DataFrame with all drivers (deduplicated on driver_number)
    """
    if not sessions_scope_path.exists():
        raise FileNotFoundError(f"Sessions scope file not found: {sessions_scope_path}")

    sessions = pd.read_csv(sessions_scope_path)

    if "session_key" not in sessions.columns:
        raise ValueError("sessions_scope CSV must contain 'session_key' column")

    session_keys = sessions["session_key"].dropna().astype(int).unique().tolist()
    _log(f"Found {len(session_keys)} unique sessions to process")

    all_drivers = []

    for idx, session_key in enumerate(session_keys, start=1):
        url = f"{BASE_URL}?session_key={session_key}"
        _log(f"[{idx}/{len(session_keys)}] Fetching {url}")

        try:
            data = fetch_json(url)
            if data:
                all_drivers.extend(data)
                _log(f"  > {len(data)} drivers retrieved")
            else:
                _log(f"  > No drivers found (empty response)")
        except Exception as e:
            _log(f"  > ERROR: {e}")
            continue

        time.sleep(sleep_between_calls)

    if not all_drivers:
        raise RuntimeError("No drivers data retrieved from any session")

    df = pd.DataFrame(all_drivers)
    _log(f"Total drivers records fetched: {len(df)}")

    # Deduplicate on driver_number (keep most recent session data)
    # Priority: keep record with most complete data
    initial_count = len(df)

    # Sort by session_key descending to prioritize recent sessions
    if "session_key" in df.columns:
        df = df.sort_values("session_key", ascending=False)

    # Keep first occurrence per driver_number
    df = df.drop_duplicates(subset=["driver_number"], keep="first")

    _log(f"Deduplication: {initial_count} -> {len(df)} unique drivers")

    # Sort by driver_number for readability
    df = df.sort_values("driver_number").reset_index(drop=True)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract drivers data from OpenF1 API")
    parser.add_argument(
        "--sessions-scope",
        type=str,
        default="data/transform/sessions_scope_2023_2024_2025.csv",
        help="Path to sessions_scope CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default: data/extract/openf1/openf1_drivers_<years>.csv)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep time between API calls in seconds",
    )
    args = parser.parse_args()

    # Resolve paths
    sessions_path = Path(args.sessions_scope)
    if not sessions_path.is_absolute():
        sessions_path = PROJECT_ROOT / sessions_path

    DATA_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract drivers
    df_drivers = extract_drivers_for_sessions(sessions_path, sleep_between_calls=args.sleep)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        # Infer years from sessions_scope filename
        # e.g., sessions_scope_2023_2024_2025.csv → openf1_drivers_2023_2024_2025.csv
        stem = sessions_path.stem  # sessions_scope_2023_2024_2025
        years_part = stem.replace("sessions_scope_", "")
        out_path = DATA_EXTRACT_DIR / f"openf1_drivers_{years_part}.csv"

    # Export
    df_drivers.to_csv(out_path, index=False)
    _log(f"Export OK: {out_path} ({len(df_drivers)} drivers)")

    # Summary
    if "full_name" in df_drivers.columns:
        _log("\nSample drivers:")
        for _, row in df_drivers.head(5).iterrows():
            name = row.get("full_name", "N/A")
            number = row.get("driver_number", "N/A")
            team = row.get("team_name", "N/A")
            _log(f"  #{number} {name} ({team})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
