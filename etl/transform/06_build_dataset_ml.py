# etl/transform/06_build_dataset_ml.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_TRANSFORM_DIR = PROJECT_ROOT / "data" / "transform"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_IN_DIR = DATA_TRANSFORM_DIR / "laps_with_weather_by_session"
DEFAULT_OUT_PATH = DATA_PROCESSED_DIR / "dataset_ml_lap_level_2023_2025.csv"


def _log(msg: str) -> None:
    print(f"[06_build_dataset_ml] {msg}")


TARGET_COL = "lap_duration"

ID_COLS = [
    "year",
    "meeting_key",
    "session_key",
    "circuit_key",
    "driver_number",
    "lap_number",
]

CONTEXT_COLS = [
    "session_name",
    "session_type",
    "location",
    "country_name",
    "gmt_offset",
    "date_start_session",
    "date_end_session",
    "lap_hour_utc",
    "station_id",
    "wikipedia_circuit_url",
]

SPORT_COLS = [
    "st_speed",
    "i1_speed",
    "i2_speed",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
]

WEATHER_COLS = [
    "temp",
    "rhum",
    "pres",
    "wspd",
    "wdir",
    "prcp",
    "cldc",
]


def _list_session_files(in_dir: Path, fmt: str) -> List[Path]:
    if not in_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {in_dir}")

    pattern = re.compile(rf"^laps_session_(\d+)\.{fmt}$")
    return [p for p in sorted(in_dir.iterdir()) if p.is_file() and pattern.match(p.name)]


def _load_all(in_dir: Path, fmt: str, limit_sessions: int = 0) -> pd.DataFrame:
    files = _list_session_files(in_dir, fmt=fmt)
    if limit_sessions and limit_sessions > 0:
        files = files[:limit_sessions]

    _log(f"Chargement fichiers: {len(files)}")
    dfs = []
    for p in files:
        if fmt == "csv":
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
        df["__source_file"] = p.name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _ensure_datetime_utc(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)


@dataclass
class QualitySummary:
    n_rows: int
    n_cols: int
    n_duplicates_key: int
    missing_target: int
    missing_weather_any: int
    missing_weather_all: int


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, QualitySummary]:
    if df.empty:
        raise ValueError("Dataset vide : aucun fichier session chargé.")

    # Parsing dates (si présentes)
    _ensure_datetime_utc(df, "date_start_session")
    _ensure_datetime_utc(df, "date_end_session")
    _ensure_datetime_utc(df, "lap_hour_utc")
    _ensure_datetime_utc(df, "weather_hour_utc")

    # Sélection colonnes (si elles existent)
    desired = ID_COLS + CONTEXT_COLS + SPORT_COLS + WEATHER_COLS + [TARGET_COL]
    keep = [c for c in desired if c in df.columns]
    df = df[keep + [c for c in ["__source_file"] if c in df.columns]].copy()

    # Cast types identifiants
    for c in ["year", "meeting_key", "session_key", "circuit_key", "driver_number", "lap_number"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Cast numériques
    for c in SPORT_COLS + WEATHER_COLS + [TARGET_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Contrôles qualité
    # Duplicats sur la clé composite (meeting_key, session_key, driver_number, lap_number)
    key_cols = [c for c in ["meeting_key", "session_key", "driver_number", "lap_number"] if c in df.columns]
    n_duplicates_key = int(df.duplicated(subset=key_cols).sum()) if key_cols else 0

    missing_target = int(df[TARGET_COL].isna().sum()) if TARGET_COL in df.columns else len(df)

    # météo : missing sur au moins une variable / toutes les variables
    weather_present = [c for c in WEATHER_COLS if c in df.columns]
    if weather_present:
        missing_weather_any = int(df[weather_present].isna().any(axis=1).sum())
        missing_weather_all = int(df[weather_present].isna().all(axis=1).sum())
    else:
        missing_weather_any = len(df)
        missing_weather_all = len(df)

    summary = QualitySummary(
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        n_duplicates_key=n_duplicates_key,
        missing_target=missing_target,
        missing_weather_any=missing_weather_any,
        missing_weather_all=missing_weather_all,
    )

    # Filtre final MVP (optionnel mais sain) :
    # - target non nulle
    # - météo présente (au moins une variable météo)
    before = len(df)
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL].notna()].copy()
    if weather_present:
        df = df[df[weather_present].notna().any(axis=1)].copy()
    _log(f"Filtre final (target+météo): {before} -> {len(df)}")

    # Tri pour lisibilité
    sort_cols = [c for c in ["year", "meeting_key", "session_key", "driver_number", "lap_number"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final ML dataset from laps_with_weather_by_session.")
    parser.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR), help="Entrée laps enrichis météo.")
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help="Chemin de sortie dataset final.")
    parser.add_argument("--limit-sessions", type=int, default=0, help="Pour tests: limiter nb sessions.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_all = _load_all(in_dir, fmt=args.format, limit_sessions=args.limit_sessions)

    df_dataset, summary = build_dataset(df_all)

    df_dataset.to_csv(out_path, index=False)
    _log(f"Export dataset ML: {out_path} | rows={len(df_dataset)} cols={df_dataset.shape[1]}")

    # Report qualité
    report = {
        "input_dir": str(in_dir),
        "output_path": str(out_path),
        "summary": summary.__dict__,
        "features": {
            "id_cols": ID_COLS,
            "context_cols": CONTEXT_COLS,
            "sport_cols": SPORT_COLS,
            "weather_cols": WEATHER_COLS,
            "target_col": TARGET_COL,
        },
    }
    report_path = out_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _log(f"Report: {report_path}")

    # Code retour : 0 si dataset non vide
    return 0 if len(df_dataset) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())