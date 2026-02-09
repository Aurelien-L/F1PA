# etl/transform/01_build_sessions_scope.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_EXTRACT_DIR = PROJECT_ROOT / "data" / "extract" / "openf1"
DATA_TRANSFORM_DIR = PROJECT_ROOT / "data" / "transform"


BASE_COLS = [
    "year",
    "meeting_key",
    "session_key",
    "session_name",
    "session_type",
    "date_start",
    "date_end",
    "gmt_offset",
    "circuit_key",
    "circuit_short_name",
    "location",
    "country_name",
    "country_code",
    "country_key",
]


def _log(msg: str) -> None:
    print(f"[01_build_sessions_scope] {msg}")


def _read_sessions_for_year(year: int) -> pd.DataFrame:
    """
    Read annual sessions file (sessions_openf1_<year>.csv).
    Fallback: read consolidated file sessions_openf1_2022_2025.csv if present, then filter by year.
    """
    yearly_path = DATA_EXTRACT_DIR / f"sessions_openf1_{year}.csv"
    consolidated_path = DATA_EXTRACT_DIR / "sessions_openf1_2022_2025.csv"

    if yearly_path.exists():
        _log(f"Lecture: {yearly_path}")
        return pd.read_csv(yearly_path)
    if consolidated_path.exists():
        _log(f"Lecture (fallback consolidé): {consolidated_path} puis filtre year=={year}")
        df = pd.read_csv(consolidated_path)
        if "year" not in df.columns:
            raise ValueError(
                "Le fichier consolidé ne contient pas la colonne 'year'. "
                "Impossible de filtrer par année."
            )
        return df[df["year"] == year].copy()

    raise FileNotFoundError(
        f"Aucun fichier sessions trouvé pour {year}. "
        f"Attendu: {yearly_path} ou {consolidated_path}"
    )


def build_sessions_scope(
    years: List[int],
    session_type: str = "Race",
) -> pd.DataFrame:
    """Build sessions scope DataFrame from annual OpenF1 sessions data."""
    dfs = []
    for y in years:
        df_y = _read_sessions_for_year(y)
        dfs.append(df_y)

    df = pd.concat(dfs, ignore_index=True)

    existing_cols = [c for c in BASE_COLS if c in df.columns]
    missing_cols = [c for c in BASE_COLS if c not in df.columns]
    if missing_cols:
        _log(f"Colonnes absentes (toléré): {missing_cols}")

    df = df[existing_cols].copy()

    if "session_type" not in df.columns:
        raise ValueError("Colonne 'session_type' absente : impossible de filtrer sur Race.")

    before = len(df)
    df = df[df["session_type"].astype(str) == session_type].copy()
    _log(f"Filtre session_type=='{session_type}': {before} -> {len(df)} lignes")

    required_keys = ["meeting_key", "session_key", "year"]
    for k in required_keys:
        if k not in df.columns:
            raise ValueError(f"Colonne clé manquante: {k}")

    for col in ["date_start", "date_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    if "date_start" in df.columns:
        df["session_start_hour_utc"] = df["date_start"].dt.floor("H")

    before = len(df)
    df = df.drop_duplicates(subset=["session_key"], keep="first")
    _log(f"Drop duplicates sur session_key: {before} -> {len(df)} lignes")

    if "date_start" in df.columns and "date_end" in df.columns:
        bad = df[(df["date_start"].notna()) & (df["date_end"].notna()) & (df["date_end"] < df["date_start"])]
        if len(bad) > 0:
            _log(f"WARNING: {len(bad)} sessions ont date_end < date_start (elles seront conservées, à investiguer).")

    sort_cols = [c for c in ["year", "meeting_key", "date_start", "session_key"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sessions scope (Race-only) for lap-level Transform.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Années à inclure (ex: 2023 2024 2025)",
    )
    parser.add_argument(
        "--session-type",
        type=str,
        default="Race",
        help="Type de session à conserver (default: Race)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Chemin de sortie CSV (optionnel). Par défaut: data/transform/sessions_scope_<years>.csv",
    )
    args = parser.parse_args()

    years = sorted(set(args.years))
    _log(f"Années demandées: {years}")

    DATA_TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)

    df_scope = build_sessions_scope(years=years, session_type=args.session_type)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        years_str = "_".join(str(y) for y in years)
        out_path = DATA_TRANSFORM_DIR / f"sessions_scope_{years_str}.csv"

    df_scope.to_csv(out_path, index=False)
    _log(f"Export OK: {out_path} ({len(df_scope)} lignes)")

    if "year" in df_scope.columns:
        _log("Répartition par année:")
        counts = df_scope["year"].value_counts().sort_index()
        for y, n in counts.items():
            _log(f"  - {y}: {n} sessions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
