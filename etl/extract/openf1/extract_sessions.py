from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

from dataclasses import dataclass

@dataclass
class YearStatus:
    year: int
    url: str
    rows: int
    status: str  # OK / EMPTY


BASE_URL = "https://api.openf1.org/v1/sessions"


def fetch_json(url: str, retries: int = 3, sleep_s: float = 1.0) -> list[dict[str, Any]]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch after {retries} retries: {url}\nLast error: {last_err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract OpenF1 sessions for a year range into CSV files")
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--out-dir", default="data/extract/openf1")
    p.add_argument("--sleep", type=float, default=0.2, help="Polite sleep between calls (seconds)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    year_status: list[YearStatus] = []

    for year in args.years:
        url = f"{BASE_URL}?year={year}"
        print(f"GET {url}")

        data = fetch_json(url)
        df = pd.DataFrame(data)

        rows = len(df)
        if rows == 0:
            print(f"OBSERVATION: OpenF1 sessions returned 0 rows for year={year}. "
                f"This suggests the source may not provide data for this year (as observed during extraction).")
            year_status.append(YearStatus(year=year, url=url, rows=rows, status="EMPTY"))
        else:
            year_status.append(YearStatus(year=year, url=url, rows=rows, status="OK"))
            df["year"] = year

        # Sanity: columns expected from docs
        # circuit_key, circuit_short_name, country_name, location, meeting_key, session_key, session_name, session_type, date_start, date_end, gmt_offset, year
        if df.empty:
            print(f"WARNING: empty response for year={year}")
        else:
            df["year"] = year  # ensure it exists and is consistent

        out_path = out_dir / f"sessions_openf1_{year}.csv"
        df.to_csv(out_path, index=False)
        print(f"OK: wrote {out_path} rows={len(df)}")

        all_dfs.append(df)
        time.sleep(args.sleep)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    combined_path = out_dir / "sessions_openf1_2022_2025.csv"
    combined.to_csv(combined_path, index=False)
    print(f"OK: wrote {combined_path} rows={len(combined)}")

    status_path = out_dir / "openf1_year_availability.csv"
    pd.DataFrame([s.__dict__ for s in year_status]).to_csv(status_path, index=False)
    print(f"OK: wrote {status_path}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())