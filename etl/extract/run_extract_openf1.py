# etl/extract/run_extract_openf1.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # .../etl/extract/run_extract_openf1.py -> parents[2] = F1PA


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}")


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + args
    log(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orchestrate OpenF1 Extract scripts (sessions + circuits used).")
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--manifest", default=None, help="Optional manifest JSON output path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()

    years = args.years
    manifest_path = Path(args.manifest).resolve() if args.manifest else (root / "data" / "extract" / "openf1" / f"manifest_openf1_{years[0]}_{years[-1]}.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    log("=== F1PA | Orchestrator | Extract | OpenF1 ===")
    log(f"years={years}")
    log(f"manifest={manifest_path}")

    started = datetime.now(timezone.utc).isoformat()

    # 1) Extract sessions (and whatever artifacts your script produces)
    run_module("etl.extract.openf1.extract_sessions", ["--years", *map(str, years)])

    # 2) Build circuits used (based on sessions)
    run_module("etl.extract.openf1.build_circuits_used", ["--years", *map(str, years)])

    # Note: extract_drivers.py NOT included in orchestrator (see etl/extract/README_DRIVERS.md)
    # It requires sessions_scope from Transform step 01 and must be run manually after that step

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "step": "extract_openf1",
        "years": years,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "modules": [
            "etl.extract.openf1.extract_sessions",
            "etl.extract.openf1.build_circuits_used",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("OK: OpenF1 extract complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())