# etl/extract/run_extract_meteostat.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}")


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + args
    log(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orchestrate Meteostat extract: stations.db -> mapping -> hourly downloads")
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--delete-raw", action="store_true", default=True)
    p.add_argument("--purge-raw", action="store_true", help="Purge hourly_raw tree at end")
    p.add_argument("--manifest", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    years = args.years

    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else (root / "data" / "extract" / "meteostat" / f"manifest_meteostat_{years[0]}_{years[-1]}.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # The mapping file is years-aware
    mapping_path = root / "data" / "extract" / "meteostat" / "mapping" / f"circuit_station_mapping_{years[0]}_{years[-1]}.csv"

    log("=== F1PA | Orchestrator | Extract | Meteostat ===")
    log(
        f"years={years} top_n={args.top_n} skip_existing={args.skip_existing} "
        f"delete_raw={args.delete_raw} purge_raw={args.purge_raw}"
    )
    log(f"manifest={manifest_path}")

    started = datetime.now(timezone.utc).isoformat()

    # 1) Download stations database (reproducible, not versioned)
    run_module("etl.extract.meteostat.download_stations_db", [])

    # 2) Build circuit->station mapping (availability-aware)
    run_module(
        "etl.extract.meteostat.build_circuit_station_mapping",
        ["--years", *map(str, years), "--top-n", str(args.top_n)],
    )

    if not mapping_path.exists():
        raise FileNotFoundError(f"Expected mapping file not found: {mapping_path}")

    # 3) Download hourly CSVs by station/year, using the mapping we just produced
    dl_args: list[str] = [
        "--mapping",
        str(mapping_path),
        "--years",
        *map(str, years),
    ]
    if args.skip_existing:
        dl_args.append("--skip-existing")
    if args.delete_raw:
        dl_args.append("--delete-raw")
    if args.purge_raw:
        dl_args.append("--purge-raw")

    run_module("etl.extract.meteostat.download_hourly_by_station", dl_args)

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "step": "extract_meteostat",
        "years": years,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "modules": [
            "etl.extract.meteostat.download_stations_db",
            "etl.extract.meteostat.build_circuit_station_mapping",
            "etl.extract.meteostat.download_hourly_by_station",
        ],
        "outputs": {
            "mapping": str(mapping_path),
            "stations_db": str(root / "data" / "extract" / "meteostat" / "stations" / "stations.db"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("OK: Meteostat extract complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
