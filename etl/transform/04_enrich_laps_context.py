# etl/transform/04_enrich_laps_context.py
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

DEFAULT_LAPS_CLEAN_DIR = DATA_TRANSFORM_DIR / "laps_clean_by_session"
DEFAULT_SCOPE_PATH = DATA_TRANSFORM_DIR / "sessions_scope_2023_2024_2025.csv"
DEFAULT_OUT_DIR = DATA_TRANSFORM_DIR / "laps_with_context_by_session"


def _log(msg: str) -> None:
    print(f"[04_enrich_laps_context] {msg}")


@dataclass
class ContextStats:
    session_key: int
    input_path: str
    output_path: Optional[str]
    n_in: int
    n_out: int
    n_missing_session_meta: int
    n_missing_lap_date_start: int
    ok: bool
    error: Optional[str]


def _list_session_files(in_dir: Path, fmt: str) -> List[Path]:
    if not in_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {in_dir}")

    pattern = re.compile(rf"^laps_session_(\d+)\.{fmt}$")
    return [p for p in sorted(in_dir.iterdir()) if p.is_file() and pattern.match(p.name)]


def _extract_session_key_from_filename(path: Path) -> int:
    m = re.search(r"laps_session_(\d+)\.", path.name)
    if not m:
        raise ValueError(f"Nom de fichier inattendu: {path.name}")
    return int(m.group(1))


def _load_scope(scope_path: Path) -> pd.DataFrame:
    if not scope_path.exists():
        raise FileNotFoundError(f"Scope introuvable: {scope_path}")

    df = pd.read_csv(scope_path)

    if "session_key" not in df.columns:
        raise ValueError("Le scope doit contenir 'session_key'.")

    for col in ["date_start", "date_end", "session_start_hour_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df = df.drop_duplicates(subset=["session_key"], keep="first").copy()

    df = df.set_index("session_key", drop=False)
    return df


def _enrich_one_session(
    df_laps: pd.DataFrame,
    df_scope_idx: pd.DataFrame,
    session_key: int,
) -> tuple[pd.DataFrame, int, int]:
    """
    Enrich lap data with session metadata.
    Returns: enriched DataFrame, count of missing metadata, count of missing lap date_start.
    """
    if session_key not in df_scope_idx.index:
        meta = None
    else:
        meta = df_scope_idx.loc[session_key].to_dict()

    if "date_start" in df_laps.columns:
        df_laps["date_start"] = pd.to_datetime(df_laps["date_start"], errors="coerce", utc=True)
    else:
        df_laps["date_start"] = pd.NaT

    n_missing_lap_date_start = int(df_laps["date_start"].isna().sum())

    df_laps["lap_hour_utc"] = df_laps["date_start"].dt.floor("H")

    n_missing_meta = 0
    if meta is None:
        n_missing_meta = len(df_laps)
        for col in [
            "year",
            "meeting_key",
            "session_name",
            "session_type",
            "date_start_session",
            "date_end_session",
            "gmt_offset",
            "circuit_key",
            "circuit_short_name",
            "location",
            "country_name",
            "country_code",
            "country_key",
        ]:
            if col not in df_laps.columns:
                df_laps[col] = pd.NA
    else:
        mapping = {
            "year": "year",
            "meeting_key": "meeting_key",
            "session_name": "session_name",
            "session_type": "session_type",
            "date_start": "date_start_session",
            "date_end": "date_end_session",
            "gmt_offset": "gmt_offset",
            "circuit_key": "circuit_key",
            "circuit_short_name": "circuit_short_name",
            "location": "location",
            "country_name": "country_name",
            "country_code": "country_code",
            "country_key": "country_key",
        }
        for src, dst in mapping.items():
            if src in meta:
                df_laps[dst] = meta[src]

        if "meeting_key" in df_laps.columns and "meeting_key" in meta and pd.notna(meta["meeting_key"]):
            mism = df_laps["meeting_key"].dropna().astype(int) != int(meta["meeting_key"])
            if mism.any():
                df_laps["meeting_key_mismatch"] = mism
            else:
                df_laps["meeting_key_mismatch"] = False

    df_laps["session_key"] = session_key

    sort_cols = [c for c in ["driver_number", "lap_number"] if c in df_laps.columns]
    if sort_cols:
        df_laps = df_laps.sort_values(sort_cols).reset_index(drop=True)

    return df_laps, n_missing_meta, n_missing_lap_date_start


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich cleaned laps with session context and compute lap_hour_utc.")
    parser.add_argument("--laps-clean-dir", type=str, default=str(DEFAULT_LAPS_CLEAN_DIR), help="Entrée laps nettoyés.")
    parser.add_argument("--scope", type=str, default=str(DEFAULT_SCOPE_PATH), help="CSV sessions_scope_*.csv")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Sortie enrichie (par session).")
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv", help="Format entrée/sortie.")
    parser.add_argument("--overwrite", action="store_true", help="Réécrit les fichiers existants.")
    parser.add_argument("--limit-sessions", type=int, default=0, help="Pour tests: limiter le nb de sessions.")
    args = parser.parse_args()

    laps_clean_dir = Path(args.laps_clean_dir)
    if not laps_clean_dir.is_absolute():
        laps_clean_dir = PROJECT_ROOT / laps_clean_dir

    scope_path = Path(args.scope)
    if not scope_path.is_absolute():
        scope_path = PROJECT_ROOT / scope_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_scope_idx = _load_scope(scope_path)

    files = _list_session_files(laps_clean_dir, fmt=args.format)
    if args.limit_sessions and args.limit_sessions > 0:
        files = files[: args.limit_sessions]

    _log(f"Entrée laps clean: {laps_clean_dir} | fichiers: {len(files)} | format={args.format}")
    _log(f"Scope: {scope_path}")
    _log(f"Sortie: {out_dir}")

    stats: List[ContextStats] = []

    for idx, path in enumerate(files, start=1):
        session_key = _extract_session_key_from_filename(path)
        out_path = out_dir / f"laps_session_{session_key}.{args.format}"

        if out_path.exists() and not args.overwrite:
            _log(f"[{idx}/{len(files)}] session_key={session_key} -> skip (déjà présent)")
            stats.append(
                ContextStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=str(out_path),
                    n_in=0,
                    n_out=0,
                    n_missing_session_meta=0,
                    n_missing_lap_date_start=0,
                    ok=True,
                    error=None,
                )
            )
            continue

        _log(f"[{idx}/{len(files)}] Enrich session_key={session_key} ...")

        try:
            if args.format == "csv":
                df_laps = pd.read_csv(path)
            else:
                df_laps = pd.read_parquet(path)

            n_in = len(df_laps)

            df_out, n_missing_meta, n_missing_lap_date_start = _enrich_one_session(
                df_laps=df_laps,
                df_scope_idx=df_scope_idx,
                session_key=session_key,
            )

            n_out = len(df_out)

            if args.format == "csv":
                df_out.to_csv(out_path, index=False)
            else:
                df_out.to_parquet(out_path, index=False)

            _log(
                f"  -> OK {n_in} -> {n_out} | missing_meta={n_missing_meta} | "
                f"missing_lap_date_start={n_missing_lap_date_start}"
            )

            stats.append(
                ContextStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=str(out_path),
                    n_in=n_in,
                    n_out=n_out,
                    n_missing_session_meta=n_missing_meta,
                    n_missing_lap_date_start=n_missing_lap_date_start,
                    ok=True,
                    error=None,
                )
            )

        except Exception as e:
            _log(f"  -> ERROR session_key={session_key}: {e}")
            stats.append(
                ContextStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=None,
                    n_in=0,
                    n_out=0,
                    n_missing_session_meta=0,
                    n_missing_lap_date_start=0,
                    ok=False,
                    error=str(e),
                )
            )
            continue

    report_df = pd.DataFrame([s.__dict__ for s in stats])
    report_csv = out_dir / "report_laps_context.csv"
    report_df.to_csv(report_csv, index=False)

    manifest = {
        "laps_clean_dir": str(laps_clean_dir),
        "scope_path": str(scope_path),
        "out_dir": str(out_dir),
        "format": args.format,
        "n_files": len(files),
        "n_ok": int((report_df["ok"] == True).sum()) if not report_df.empty else 0,  # noqa: E712
        "n_ko": int((report_df["ok"] == False).sum()) if not report_df.empty else 0,  # noqa: E712
        "report_csv": str(report_csv),
    }
    manifest_path = out_dir / "manifest_context.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log(f"Rapport: {report_csv}")
    _log(f"Manifest: {manifest_path}")

    n_ko = manifest["n_ko"]
    return 0 if n_ko == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
