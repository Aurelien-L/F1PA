# etl/extract/run_extract_all.py
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


def run_script(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + args
    log(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Extract pipeline (OpenF1 -> Wikipedia -> Meteostat).")
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--wiki-sleep", type=float, default=1.0)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--purge-raw", action="store_true", default=False)
    p.add_argument("--manifest", default=None, help="Optional manifest JSON output path for the whole run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()

    years = args.years
    manifest_path = Path(args.manifest).resolve() if args.manifest else (root / "data" / "extract" / f"manifest_extract_all_{years[0]}_{years[-1]}.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    log("=== F1PA | Orchestrator | Extract | ALL ===")
    log(f"years={years} wiki_sleep={args.wiki_sleep} top_n={args.top_n} purge_raw={args.purge_raw}")
    log(f"manifest={manifest_path}")

    started = datetime.now(timezone.utc).isoformat()

    # 1) OpenF1
    run_script("etl.extract.run_extract_openf1", ["--years", *map(str, years)])

    # 2) Wikipedia + matching + filter
    run_script("etl.extract.run_extract_wikipedia", ["--years", *map(str, years), "--sleep", str(args.wiki_sleep)])

    # 3) Meteostat
    meteostat_args = ["--years", *map(str, years), "--top-n", str(args.top_n)]
    if args.purge_raw:
        meteostat_args.append("--purge-raw")
    run_script("etl.extract.run_extract_meteostat", meteostat_args)

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "step": "extract_all",
        "years": years,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "orchestrators": [
            "etl.extract.run_extract_openf1",
            "etl.extract.run_extract_wikipedia",
            "etl.extract.run_extract_meteostat",
        ],
        "params": {
            "wiki_sleep": args.wiki_sleep,
            "top_n": args.top_n,
            "purge_raw": args.purge_raw,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("OK: full Extract pipeline complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())