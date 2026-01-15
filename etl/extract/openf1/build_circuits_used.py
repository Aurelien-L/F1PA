from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a deduplicated list of circuits used from OpenF1 sessions extract"
    )
    p.add_argument(
        "--sessions",
        default="data/extract/openf1/sessions_openf1_2022_2025.csv",
        help="Input sessions extract CSV (combined)",
    )
    p.add_argument(
        "--out",
        default="data/extract/openf1/openf1_circuits_used_2022_2025.csv",
        help="Output circuits used CSV",
    )
    p.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional filter on years (e.g. --years 2023 2024 2025). If omitted, uses all rows.",
    )
    return p.parse_args()


def most_frequent(series: pd.Series) -> str | None:
    """
    Returns the most frequent non-null value; if tie, returns the first (stable after sort).
    """
    s = series.dropna().astype(str)
    if s.empty:
        return None
    vc = s.value_counts()
    top = vc.index[0]
    return str(top)


def main() -> int:
    args = parse_args()

    sessions_path = Path(args.sessions).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sessions_path)

    required = {"circuit_key", "circuit_short_name", "country_name", "location"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in sessions extract: {sorted(missing)}")

    # Optional filtering by years, if year column exists (it should, but we stay defensive)
    if args.years is not None and len(args.years) > 0:
        if "year" not in df.columns:
            raise ValueError("Cannot filter by --years because 'year' column is missing in sessions extract.")
        df = df[df["year"].isin(args.years)].copy()

    # Normalize types for grouping
    df["circuit_key"] = pd.to_numeric(df["circuit_key"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["circuit_key", "circuit_short_name"]).copy()

    # Build variants for auditability (e.g., Miami vs Miami Gardens)
    def variants(series: pd.Series) -> str:
        vals = (
            series.dropna()
            .astype(str)
            .map(lambda x: x.strip())
            .loc[lambda s: s.ne("")]
            .unique()
            .tolist()
        )
        vals_sorted = sorted(vals)
        return "|".join(vals_sorted)

    circuits = (
        df.groupby("circuit_key", as_index=False)
        .agg(
            circuit_short_name=("circuit_short_name", most_frequent),
            country_name=("country_name", most_frequent),
            location=("location", most_frequent),
            location_variants=("location", variants),
            years_present=("year", variants) if "year" in df.columns else ("circuit_short_name", lambda s: None),
        )
        .sort_values(["country_name", "circuit_short_name"], na_position="last")
        .reset_index(drop=True)
    )

    # Ensure stable column order
    if "year" in df.columns:
        circuits = circuits[
            ["circuit_key", "circuit_short_name", "country_name", "location", "location_variants", "years_present"]
        ]
    else:
        circuits = circuits[
            ["circuit_key", "circuit_short_name", "country_name", "location", "location_variants"]
        ]

    circuits.to_csv(out_path, index=False)
    print(f"OK: wrote {out_path} rows={len(circuits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
