# etl/transform/03_filter_clean_laps.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_TRANSFORM_DIR = PROJECT_ROOT / "data" / "transform"

DEFAULT_IN_DIR = DATA_TRANSFORM_DIR / "laps_raw_by_session"
DEFAULT_OUT_DIR = DATA_TRANSFORM_DIR / "laps_clean_by_session"


def _log(msg: str) -> None:
    print(f"[03_filter_clean_laps] {msg}")


@dataclass
class CleanStats:
    session_key: int
    input_path: str
    output_path: Optional[str]
    n_in: int
    n_out: int
    removed_null_target: int
    removed_nonpositive_target: int
    removed_pit_out: int
    removed_outliers: int
    q_low: float
    q_high: float
    low_threshold: Optional[float]
    high_threshold: Optional[float]
    ok: bool
    error: Optional[str]


def _list_session_files(in_dir: Path, fmt: str) -> List[Path]:
    if not in_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {in_dir}")

    pattern = re.compile(rf"^laps_session_(\d+)\.{fmt}$")
    files = []
    for p in sorted(in_dir.iterdir()):
        if p.is_file() and pattern.match(p.name):
            files.append(p)
    return files


def _extract_session_key_from_filename(path: Path) -> int:
    m = re.search(r"laps_session_(\d+)\.", path.name)
    if not m:
        raise ValueError(f"Nom de fichier inattendu: {path.name}")
    return int(m.group(1))


def _safe_quantile_bounds(series: pd.Series, q_low: float, q_high: float) -> Tuple[Optional[float], Optional[float]]:
    s = series.dropna()
    if len(s) < 20:
        # trop peu de points : on ne fait pas de quantiles => pas de filtre outliers
        return None, None
    low = float(s.quantile(q_low))
    high = float(s.quantile(q_high))
    # sécurité : si low>=high, on désactive
    if low >= high:
        return None, None
    return low, high


def clean_one_session_df(
    df: pd.DataFrame,
    q_low: float,
    q_high: float,
    use_quantiles: bool,
    min_lap_s: Optional[float],
    max_lap_s: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, int], Optional[float], Optional[float]]:
    """
    Applique les règles de nettoyage à un DataFrame de laps (une sesifon).
    Retourne:
      - df_clean
      - compteurs de suppression
      - seuils bas/haut (quantiles) if appliqués
    """
    counters = {
        "removed_null_target": 0,
        "removed_nonpositive_target": 0,
        "removed_pit_out": 0,
        "removed_outliers": 0,
    }

    if "lap_duration" not in df.columns:
        raise ValueError("Colonne 'lap_duration' absente (target).")

    # 1) cible non nulle
    before = len(df)
    df = df[df["lap_duration"].notna()].copy()
    counters["removed_null_target"] += before - len(df)

    # 2) cible strictement positive
    before = len(df)
    df = df[df["lap_duration"] > 0].copy()
    counters["removed_nonpositive_target"] += before - len(df)

    # 3) pit-out laps exclus
    if "is_pit_out_lap" in df.columns:
        before = len(df)
        # is_pit_out_lap peut être bool, 0/1, ou NA => on considère True comme à exclure
        df = df[~(df["is_pit_out_lap"] == True)].copy()  # noqa: E712
        counters["removed_pit_out"] += before - len(df)

    # 4) bornes fixes optionnelles (fallback)
    # utiles si tu veux éviter des extrêmes absurdes même avant quantiles
    if min_lap_s is not None:
        before = len(df)
        df = df[df["lap_duration"] >= float(min_lap_s)].copy()
        # on comptabilise dans outliers (car c'est un filtre de valeurs aberrantes)
        counters["removed_outliers"] += before - len(df)

    if max_lap_s is not None:
        before = len(df)
        df = df[df["lap_duration"] <= float(max_lap_s)].copy()
        counters["removed_outliers"] += before - len(df)

    # 5) filtre outliers par quantiles (recommandé)
    low_thr = None
    high_thr = None
    if use_quantiles:
        low_thr, high_thr = _safe_quantile_bounds(df["lap_duration"], q_low=q_low, q_high=q_high)
        if low_thr is not None and high_thr is not None:
            before = len(df)
            df = df[(df["lap_duration"] >= low_thr) & (df["lap_duration"] <= high_thr)].copy()
            counters["removed_outliers"] += before - len(df)

    # Tri / types minimaux
    sort_cols = [c for c in ["driver_number", "lap_number"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df, counters, low_thr, high_thr


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter & clean OpenF1 laps per session (Race lap-level).")
    parser.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR), help="Dossier d'entrée laps bruts.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Dossier de sortie laps nettoyés.")
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv", help="Format d'entrée/sortie.")
    parser.add_argument(
        "--use-quantiles",
        action="store_true",
        help="Active le filtrage outliers par quantiles (recommandé).",
    )
    parser.add_argument(
        "--q-low",
        type=float,
        default=0.01,
        help="Quantile bas pour outliers (ex: 0.01).",
    )
    parser.add_argument(
        "--q-high",
        type=float,
        default=0.99,
        help="Quantile haut pour outliers (ex: 0.99).",
    )
    parser.add_argument(
        "--min-lap-s",
        type=float,
        default=float("nan"),
        help="Borne basse fixe optionnelle (secondes). NaN = désactivé.",
    )
    parser.add_argument(
        "--max-lap-s",
        type=float,
        default=float("nan"),
        help="Borne haute fixe optionnelle (secondes). NaN = désactivé.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Réécrit les fichiers existants.")
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=0,
        help="Pour tests: limiter le nb de sessions traitées (0 = pas de limite).",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    min_lap_s = None if pd.isna(args.min_lap_s) else float(args.min_lap_s)
    max_lap_s = None if pd.isna(args.max_lap_s) else float(args.max_lap_s)

    if args.use_quantiles:
        if not (0.0 < args.q_low < args.q_high < 1.0):
            raise ValueError("q-low et q-high doivent vérifier 0 < q_low < q_high < 1")

    files = _list_session_files(in_dir, fmt=args.format)
    if args.limit_sessions and args.limit_sessions > 0:
        files = files[: args.limit_sessions]

    _log(f"Entrée: {in_dir} | fichiers: {len(files)} | format={args.format}")
    _log(f"Sortie: {out_dir} | use_quantiles={args.use_quantiles} q=({args.q_low},{args.q_high}) "
         f"min_lap_s={min_lap_s} max_lap_s={max_lap_s}")

    stats: List[CleanStats] = []

    for idx, path in enumerate(files, start=1):
        session_key = _extract_session_key_from_filename(path)
        out_path = out_dir / f"laps_session_{session_key}.{args.format}"

        if out_path.exists() and not args.overwrite:
            _log(f"[{idx}/{len(files)}] session_key={session_key} -> skip (déjà présent)")
            # on ne sait pas n_out sans relire; on log minimal
            stats.append(
                CleanStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=str(out_path),
                    n_in=0,
                    n_out=0,
                    removed_null_target=0,
                    removed_nonpositive_target=0,
                    removed_pit_out=0,
                    removed_outliers=0,
                    q_low=args.q_low,
                    q_high=args.q_high,
                    low_threshold=None,
                    high_threshold=None,
                    ok=True,
                    error=None,
                )
            )
            continue

        _log(f"[{idx}/{len(files)}] Nettoyage session_key={session_key} ...")

        try:
            if args.format == "csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)

            n_in = len(df)

            df_clean, counters, low_thr, high_thr = clean_one_session_df(
                df=df,
                q_low=args.q_low,
                q_high=args.q_high,
                use_quantiles=args.use_quantiles,
                min_lap_s=min_lap_s,
                max_lap_s=max_lap_s,
            )

            n_out = len(df_clean)

            # Export
            if args.format == "csv":
                df_clean.to_csv(out_path, index=False)
            else:
                df_clean.to_parquet(out_path, index=False)

            _log(f"  -> OK {n_in} -> {n_out} (removed={n_in - n_out}) | outliers_thr=({low_thr},{high_thr})")

            stats.append(
                CleanStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=str(out_path),
                    n_in=n_in,
                    n_out=n_out,
                    removed_null_target=counters["removed_null_target"],
                    removed_nonpositive_target=counters["removed_nonpositive_target"],
                    removed_pit_out=counters["removed_pit_out"],
                    removed_outliers=counters["removed_outliers"],
                    q_low=args.q_low,
                    q_high=args.q_high,
                    low_threshold=low_thr,
                    high_threshold=high_thr,
                    ok=True,
                    error=None,
                )
            )

        except Exception as e:
            _log(f"  -> ERROR session_key={session_key}: {e}")
            stats.append(
                CleanStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=None,
                    n_in=0,
                    n_out=0,
                    removed_null_target=0,
                    removed_nonpositive_target=0,
                    removed_pit_out=0,
                    removed_outliers=0,
                    q_low=args.q_low,
                    q_high=args.q_high,
                    low_threshold=None,
                    high_threshold=None,
                    ok=False,
                    error=str(e),
                )
            )
            continue

    # Rapport CSV + manifest JSON
    report_df = pd.DataFrame([s.__dict__ for s in stats])
    report_csv = out_dir / "report_laps_cleaning.csv"
    report_df.to_csv(report_csv, index=False)

    manifest = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "format": args.format,
        "use_quantiles": args.use_quantiles,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "min_lap_s": min_lap_s,
        "max_lap_s": max_lap_s,
        "n_files": len(files),
        "n_ok": int((report_df["ok"] == True).sum()) if not report_df.empty else 0,  # noqa: E712
        "n_ko": int((report_df["ok"] == False).sum()) if not report_df.empty else 0,  # noqa: E712
        "report_csv": str(report_csv),
    }
    manifest_path = out_dir / "manifest_cleaning.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log(f"Rapport: {report_csv}")
    _log(f"Manifest: {manifest_path}")

    # code retour : 0 si tout OK, 2 sinon
    n_ko = manifest["n_ko"]
    return 0 if n_ko == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
