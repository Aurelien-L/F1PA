# etl/transform/run_transform_all.py
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
TRANSFORM_DIR = DATA_DIR / "transform"
PROCESSED_DIR = DATA_DIR / "processed"

# Dossiers intermédiaires Transform (créés par nos scripts)
DIR_LAPS_RAW = TRANSFORM_DIR / "laps_raw_by_session"
DIR_LAPS_CLEAN = TRANSFORM_DIR / "laps_clean_by_session"
DIR_LAPS_CTX = TRANSFORM_DIR / "laps_with_context_by_session"
DIR_LAPS_WEATHER = TRANSFORM_DIR / "laps_with_weather_by_session"


def _log(msg: str) -> None:
    print(f"[run_transform_all] {msg}")


def _run_module(module: str, args: List[str], ok_codes: Optional[List[int]] = None) -> int:
    """
    Exécute: python -m <module> <args...>
    Retourne le code. Erreur seulement si code pas dans ok_codes.
    """
    if ok_codes is None:
        ok_codes = [0]

    cmd = [sys.executable, "-m", module] + args
    _log(f"RUN: {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if p.returncode not in ok_codes:
        raise RuntimeError(f"Echec module {module} (returncode={p.returncode})")

    return p.returncode



def _safe_rmtree(path: Path) -> None:
    """
    Suppression défensive :
    - n'efface que sous data/transform
    - n'efface que si le dossier existe
    """
    if not path.exists():
        return
    # garde-fou: ne supprimer que dans data/transform
    try:
        path.resolve().relative_to(TRANSFORM_DIR.resolve())
    except Exception as e:
        raise RuntimeError(f"Refus de suppression hors data/transform: {path} ({e})")

    _log(f"PURGE: {path}")
    shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Transform steps (lap-level) with optional cleanup.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="Années à inclure (défaut: 2023 2024 2025)",
    )
    parser.add_argument(
        "--scope-output",
        type=str,
        default="",
        help="Chemin de sortie du scope (optionnel). Par défaut: data/transform/sessions_scope_<years>.csv",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Format des fichiers intermédiaires Transform (défaut: csv)",
    )
    parser.add_argument(
        "--use-quantiles",
        action="store_true",
        help="Active le filtrage outliers par quantiles (recommandé).",
    )
    parser.add_argument("--q-low", type=float, default=0.01, help="Quantile bas (défaut: 0.01)")
    parser.add_argument("--q-high", type=float, default=0.99, help="Quantile haut (défaut: 0.99)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Réécrit les sorties existantes à chaque étape.",
    )
    parser.add_argument(
        "--purge-intermediate",
        action="store_true",
        help="Supprime progressivement les dossiers intermédiaires une fois qu'ils ne sont plus nécessaires.",
    )
    parser.add_argument(
        "--keep-weather-sessions",
        action="store_true",
        help="Si purge activée, conserve quand même data/transform/laps_with_weather_by_session (utile debug).",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=0,
        help="Pour tests: limite le nombre de sessions traitées dans les étapes concernées.",
    )
    args = parser.parse_args()

    years = sorted(set(args.years))
    years_str = "_".join(str(y) for y in years)

    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Scope
    scope_path = (
        Path(args.scope_output)
        if args.scope_output
        else (TRANSFORM_DIR / f"sessions_scope_{years_str}.csv")
    )
    if not scope_path.is_absolute():
        scope_path = PROJECT_ROOT / scope_path

    s1_args = ["--years"] + [str(y) for y in years] + ["--session-type", "Race"]
    if args.scope_output:
        s1_args += ["--output", str(scope_path)]
    _run_module("etl.transform.01_build_sessions_scope", s1_args)

    # 2) Extract laps (Transform-side)
    s2_args = [
        "--scope", str(scope_path),
        "--out-dir", str(DIR_LAPS_RAW),
        "--format", args.format,
        "--sleep", "0.2",
    ]
    if args.overwrite:
        s2_args.append("--overwrite")
    if args.limit_sessions and args.limit_sessions > 0:
        s2_args += ["--limit-sessions", str(args.limit_sessions)]
    _run_module("etl.transform.02_extract_openf1_laps", s2_args, ok_codes=[0, 2])

    rc2 = _run_module("etl.transform.02_extract_openf1_laps", s2_args, ok_codes=[0, 2])
    if rc2 == 2:
        _log("WARNING: certaines sessions n'ont pas pu être extraites (voir manifest_laps_extract.json).")


    # 3) Clean laps
    s3_args = [
        "--in-dir", str(DIR_LAPS_RAW),
        "--out-dir", str(DIR_LAPS_CLEAN),
        "--format", args.format,
    ]
    if args.use_quantiles:
        s3_args += ["--use-quantiles", "--q-low", str(args.q_low), "--q-high", str(args.q_high)]
    if args.overwrite:
        s3_args.append("--overwrite")
    if args.limit_sessions and args.limit_sessions > 0:
        s3_args += ["--limit-sessions", str(args.limit_sessions)]
    _run_module("etl.transform.03_filter_clean_laps", s3_args, ok_codes=[0, 2])

    # Purge progressive (raw)
    if args.purge_intermediate:
        _safe_rmtree(DIR_LAPS_RAW)

    # 4) Enrich context
    s4_args = [
        "--laps-clean-dir", str(DIR_LAPS_CLEAN),
        "--scope", str(scope_path),
        "--out-dir", str(DIR_LAPS_CTX),
        "--format", args.format,
    ]
    if args.overwrite:
        s4_args.append("--overwrite")
    if args.limit_sessions and args.limit_sessions > 0:
        s4_args += ["--limit-sessions", str(args.limit_sessions)]
    _run_module("etl.transform.04_enrich_laps_context", s4_args, ok_codes=[0, 2])

    # Purge progressive (clean)
    if args.purge_intermediate:
        _safe_rmtree(DIR_LAPS_CLEAN)

    # 5) Join weather hourly
    s5_args = [
        "--laps-dir", str(DIR_LAPS_CTX),
        "--out-dir", str(DIR_LAPS_WEATHER),
        "--format", args.format,
    ]
    if args.overwrite:
        s5_args.append("--overwrite")
    if args.limit_sessions and args.limit_sessions > 0:
        s5_args += ["--limit-sessions", str(args.limit_sessions)]
    _run_module("etl.transform.05_join_weather_hourly", s5_args, ok_codes=[0, 2])

    # Purge progressive (context)
    if args.purge_intermediate:
        _safe_rmtree(DIR_LAPS_CTX)

    # 6) Build dataset final
    out_dataset = PROCESSED_DIR / f"dataset_ml_lap_level_{years_str}.csv"
    s6_args = [
        "--in-dir", str(DIR_LAPS_WEATHER),
        "--format", args.format,
        "--out", str(out_dataset),
    ]
    if args.limit_sessions and args.limit_sessions > 0:
        s6_args += ["--limit-sessions", str(args.limit_sessions)]
    _run_module("etl.transform.06_build_dataset_ml", s6_args)

    # Purge finale (weather sessions) si demandé
    if args.purge_intermediate and not args.keep_weather_sessions:
        _safe_rmtree(DIR_LAPS_WEATHER)

    _log("Transform terminé avec succès.")
    _log(f"Dataset final: {out_dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
