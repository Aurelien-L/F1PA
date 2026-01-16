from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize OpenF1 -> Wikipedia circuit mapping from candidates")
    p.add_argument(
        "--candidates",
        default="data/extract/matching/openf1_wikipedia_match_candidates.csv",
        help="Candidates CSV produced by build_openf1_wikipedia_candidates",
    )
    p.add_argument(
        "--overrides",
        default="etl/extract/matching/openf1_wikipedia_overrides.csv",
        help="Optional overrides CSV with columns: circuit_key, chosen_candidate_rank",
    )
    p.add_argument(
        "--out",
        default="data/extract/matching/openf1_to_wikipedia_circuit_map.csv",
        help="Final mapping output CSV",
    )
    p.add_argument(
        "--fail-on-missing-latlon",
        action="store_true",
        help="Fail if the chosen mapping contains missing latitude/longitude",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cand_path = Path(args.candidates).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cand_path)

    required = {
        "circuit_key",
        "candidate_rank",
        "match_score",
        "openf1_circuit_short_name",
        "openf1_country_name",
        "openf1_location",
        "wikipedia_circuit_name",
        "wikipedia_country",
        "wikipedia_locality",
        "wikipedia_circuit_url",
        "wikipedia_latitude",
        "wikipedia_longitude",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidates file missing required columns: {sorted(missing)}")

    # Default: pick rank 1 per circuit
    choice = df[df["candidate_rank"] == 1].copy()

    # Apply overrides
    if args.overrides:
        ov_path = Path(args.overrides).resolve()
        ov = pd.read_csv(ov_path)

        if not {"circuit_key", "chosen_candidate_rank"}.issubset(set(ov.columns)):
            raise ValueError("Overrides CSV must contain columns: circuit_key, chosen_candidate_rank")

        for _, r in ov.iterrows():
            ck = int(r["circuit_key"])
            rank = int(r["chosen_candidate_rank"])
            alt = df[(df["circuit_key"] == ck) & (df["candidate_rank"] == rank)]
            if alt.empty:
                raise ValueError(f"Override refers to missing candidate: circuit_key={ck} rank={rank}")

            choice = choice[choice["circuit_key"] != ck]
            choice = pd.concat([choice, alt], ignore_index=True)

    # Ensure one row per circuit_key
    choice = (
        choice.sort_values(["circuit_key", "candidate_rank"])
        .drop_duplicates(subset=["circuit_key"], keep="first")
        .reset_index(drop=True)
    )

    # Validate lat/lon presence
    choice["wikipedia_latitude"] = pd.to_numeric(choice["wikipedia_latitude"], errors="coerce")
    choice["wikipedia_longitude"] = pd.to_numeric(choice["wikipedia_longitude"], errors="coerce")
    missing_latlon = choice[choice["wikipedia_latitude"].isna() | choice["wikipedia_longitude"].isna()]
    if not missing_latlon.empty:
        msg = (
            "Chosen mapping contains missing latitude/longitude for:\n"
            + missing_latlon[["circuit_key", "openf1_circuit_short_name", "wikipedia_circuit_name", "candidate_rank"]]
            .to_string(index=False)
        )
        if args.fail_on_missing_latlon:
            raise ValueError(msg)
        else:
            print("WARNING:", msg)

    out_cols = [
        "circuit_key",
        "openf1_circuit_short_name",
        "openf1_country_name",
        "openf1_location",
        "candidate_rank",
        "match_score",
        "wikipedia_circuit_name",
        "wikipedia_country",
        "wikipedia_locality",
        "wikipedia_circuit_url",
        "wikipedia_latitude",
        "wikipedia_longitude",
    ]
    choice[out_cols].to_csv(out_path, index=False)
    print(f"OK: wrote final mapping to {out_path} rows={len(choice)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
