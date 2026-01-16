# etl/extract/run_extract_wikipedia.py
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
    p = argparse.ArgumentParser(description="Orchestrate Wikipedia extract + OpenF1 matching + filtered circuits list.")
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--sleep", type=float, default=1.0, help="Sleep between circuit page requests for Wikipedia")
    p.add_argument("--manifest", default=None, help="Optional manifest JSON output path")
    p.add_argument("--top-n", type=int, default=5, help="Top-N wikipedia candidates per OpenF1 circuit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()

    years = args.years
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else (root / "data" / "extract" / "wikipedia" / f"manifest_wikipedia_{years[0]}_{years[-1]}.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Inputs produced by OpenF1 extract
    openf1_circuits_used = root / "data" / "extract" / "openf1" / "openf1_circuits_used_2022_2025.csv"
    # NOTE: your OpenF1 scripts currently write a 2022_2025 aggregate even if you request 2023-2025
    # If you later change naming to match years, update this single path here.

    # Outputs produced by Wikipedia extract
    wikipedia_circuits = root / "data" / "extract" / "wikipedia" / "circuits_wikipedia_extract.csv"

    # Matching outputs
    candidates_out = root / "data" / "extract" / "matching" / "openf1_wikipedia_match_candidates.csv"
    final_map_out = root / "data" / "extract" / "matching" / "openf1_to_wikipedia_circuit_map.csv"

    log("=== F1PA | Orchestrator | Extract | Wikipedia ===")
    log(f"years={years}")
    log(f"manifest={manifest_path}")

    started = datetime.now(timezone.utc).isoformat()

    # 1) Extract circuits from Wikipedia
    run_module("etl.extract.wikipedia.extract_circuits", ["--sleep", str(args.sleep)])

    # Sanity check: required inputs exist
    if not openf1_circuits_used.exists():
        raise FileNotFoundError(f"Missing OpenF1 circuits used file: {openf1_circuits_used}")
    if not wikipedia_circuits.exists():
        raise FileNotFoundError(f"Missing Wikipedia circuits extract file: {wikipedia_circuits}")

    # 2) Build OpenF1->Wikipedia match candidates (NO --years)
    run_module(
        "etl.extract.matching.build_openf1_wikipedia_candidates",
        [
            "--openf1",
            str(openf1_circuits_used),
            "--wikipedia",
            str(wikipedia_circuits),
            "--out",
            str(candidates_out),
            "--top-n",
            str(args.top_n),
        ],
    )

    # 3) Finalize mapping (NO --years)
    # Assumes your finalize script uses defaults or takes explicit input/output args.
    # If it supports args: --candidates and --out, pass them; otherwise keep it empty.
    run_module(
        "etl.extract.matching.finalize_openf1_wikipedia_mapping",
        [
            "--candidates",
            str(candidates_out),
            "--out",
            str(final_map_out),
        ],
    )

    # 4) Filter Wikipedia circuits to only those used by OpenF1 (NO --years)
    # Prefer passing explicit inputs so this is deterministic.
    run_module(
        "etl.extract.wikipedia.filter_circuits_for_openf1",
        [
            "--mapping",
            str(final_map_out),
            "--wikipedia",
            str(wikipedia_circuits),
            "--out",
            str(root / "data" / "extract" / "wikipedia" / "circuits_wikipedia_filtered_2023_2025.csv"),
        ],
    )

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "step": "extract_wikipedia",
        "years": years,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "modules": [
            "etl.extract.wikipedia.extract_circuits",
            "etl.extract.matching.build_openf1_wikipedia_candidates",
            "etl.extract.matching.finalize_openf1_wikipedia_mapping",
            "etl.extract.wikipedia.filter_circuits_for_openf1",
        ],
        "inputs": {
            "openf1_circuits_used": str(openf1_circuits_used),
            "wikipedia_circuits": str(wikipedia_circuits),
        },
        "outputs": {
            "candidates": str(candidates_out),
            "final_map": str(final_map_out),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("OK: Wikipedia extract + matching complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
