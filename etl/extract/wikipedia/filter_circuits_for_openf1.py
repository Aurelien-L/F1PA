from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter Wikipedia circuits to those used by OpenF1 (2023–2025)"
    )
    p.add_argument(
        "--mapping",
        default="data/extract/matching/openf1_to_wikipedia_circuit_map.csv",
        help="Validated OpenF1 -> Wikipedia circuit mapping",
    )
    p.add_argument(
        "--wikipedia",
        default="data/extract/wikipedia/circuits_wikipedia_extract.csv",
        help="Wikipedia circuits extract (full)",
    )
    p.add_argument(
        "--out",
        default="data/extract/wikipedia/circuits_wikipedia_filtered_2023_2025.csv",
        help="Filtered Wikipedia circuits (OpenF1 only)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    map_path = Path(args.mapping).resolve()
    wiki_path = Path(args.wikipedia).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_map = pd.read_csv(map_path)
    df_wiki = pd.read_csv(wiki_path)

    required_map = {"wikipedia_circuit_name", "wikipedia_circuit_url"}
    required_wiki = {"circuit_name", "circuit_url", "latitude", "longitude"}

    missing_map = required_map - set(df_map.columns)
    missing_wiki = required_wiki - set(df_wiki.columns)

    if missing_map:
        raise ValueError(f"Mapping file missing required columns: {sorted(missing_map)}")
    if missing_wiki:
        raise ValueError(f"Wikipedia file missing required columns: {sorted(missing_wiki)}")

    # Join on circuit_url (most stable key)
    filtered = df_wiki.merge(
        df_map[["wikipedia_circuit_name", "wikipedia_circuit_url"]],
        left_on="circuit_url",
        right_on="wikipedia_circuit_url",
        how="inner",
    )

    # Optional sanity: ensure 1 Wikipedia row per OpenF1 circuit
    if len(filtered) != len(df_map):
        print(
            "WARNING: filtered Wikipedia rows count does not match mapping rows.\n"
            f"mapping={len(df_map)} filtered={len(filtered)}"
        )

    # Clean columns (keep Wikipedia schema intact)
    filtered = filtered[df_wiki.columns]

    filtered.to_csv(out_path, index=False)
    print(f"OK: wrote {out_path} rows={len(filtered)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
