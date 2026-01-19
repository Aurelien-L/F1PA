# etl/transform/02_extract_openf1_laps.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_TRANSFORM_DIR = PROJECT_ROOT / "data" / "transform"

DEFAULT_SCOPE_PATH = DATA_TRANSFORM_DIR / "sessions_scope_2023_2024_2025.csv"

OPENF1_BASE_URL = "https://api.openf1.org/v1"
LAPS_ENDPOINT = f"{OPENF1_BASE_URL}/laps"


# Champs laps observés / utiles (MVP lap-level)
# On reste permissif : on exporte tout ce que renvoie l'API, mais on pourra ensuite
# sélectionner un sous-ensemble dans le script 03.
SUGGESTED_LAP_COLS = [
    "meeting_key",
    "session_key",
    "driver_number",
    "lap_number",
    "lap_duration",
    "date_start",
    "is_pit_out_lap",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "st_speed",
    "i1_speed",
    "i2_speed",
    # d'autres champs peuvent exister, on ne force pas la liste ici
]


def _log(msg: str) -> None:
    print(f"[02_extract_openf1_laps] {msg}")


def _build_http_session(
    total_retries: int = 5,
    backoff_factor: float = 0.6,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
    timeout_s: int = 60,
) -> Tuple[requests.Session, int]:
    """
    Retourne une session requests configurée avec retry/backoff.
    On retourne aussi le timeout par défaut (en secondes) à utiliser par requête.
    """
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s, timeout_s


@dataclass
class ExtractStats:
    session_key: int
    meeting_key: Optional[int]
    year: Optional[int]
    ok: bool
    n_rows: int
    output_path: Optional[str]
    error: Optional[str]


def _iter_session_keys_from_scope(scope_path: Path) -> List[Dict[str, Any]]:
    if not scope_path.exists():
        raise FileNotFoundError(f"Scope introuvable: {scope_path}")

    df = pd.read_csv(scope_path)
    required = ["session_key"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonne requise absente du scope: {c}")

    # on récupère quelques champs utiles pour logs/partition
    keep = [c for c in ["year", "meeting_key", "session_key", "session_name", "session_type"] if c in df.columns]
    rows = df[keep].to_dict(orient="records")

    # dédoublonnage au cas où
    seen = set()
    uniq_rows = []
    for r in rows:
        sk = int(r["session_key"])
        if sk in seen:
            continue
        seen.add(sk)
        uniq_rows.append(r)
    return uniq_rows


def _openf1_get_all(
    http: requests.Session,
    timeout_s: int,
    endpoint: str,
    params: Dict[str, Any],
    page_size: int = 2000,
    max_pages: int = 50,
    sleep_s: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    OpenF1 peut ne pas supporter limit/offset selon les endpoints.
    Stratégie robuste :
    - 1) Tentative sans pagination (params simples)
    - 2) Si la réponse est une liste non vide, on la retourne telle quelle (MVP)
    - 3) Option future : implémenter une pagination seulement si supportée.
    """
    resp = http.get(endpoint, params=params, timeout=timeout_s)
    if resp.status_code >= 400:
        txt = resp.text[:500] if resp.text else ""
        raise RuntimeError(f"HTTP {resp.status_code} sur {endpoint} params={params} body={txt}")

    try:
        rows = resp.json()
    except Exception as e:
        raise RuntimeError(f"Réponse non JSON: {e}")

    if not isinstance(rows, list):
        raise RuntimeError(f"Format inattendu: attendu list, reçu {type(rows)}")

    # MVP : on ne pagine pas tant qu'on n'a pas besoin
    return rows



def extract_laps_for_session(
    http: requests.Session,
    timeout_s: int,
    session_key: int,
    page_size: int,
    max_pages: int,
    sleep_s: float,
) -> pd.DataFrame:
    params = {"session_key": session_key}
    rows = _openf1_get_all(
        http=http,
        timeout_s=timeout_s,
        endpoint=LAPS_ENDPOINT,
        params=params,
        page_size=page_size,
        max_pages=max_pages,
        sleep_s=sleep_s,
    )
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Aucun lap retourné (réponse vide). Vérifier paramètres API (pagination/endpoint).")

    # Normalisation types minimale (sans sur-transformer)
    # date_start peut être null => on parse en datetime UTC avec coercition
    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)

    # Conversion numérique défensive
    for col in ["lap_number", "driver_number", "meeting_key", "session_key"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3",
                "st_speed", "i1_speed", "i2_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_pit_out_lap" in df.columns:
        # Bool défensif (OpenF1 renvoie normalement bool)
        df["is_pit_out_lap"] = df["is_pit_out_lap"].astype("boolean")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract OpenF1 laps for session_keys from sessions scope (Race-only).")
    parser.add_argument(
        "--scope",
        type=str,
        default=str(DEFAULT_SCOPE_PATH),
        help="Chemin du CSV sessions_scope_*.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DATA_TRANSFORM_DIR / "laps_raw_by_session"),
        help="Dossier de sortie (un fichier par session_key).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Format de sortie (csv recommandé MVP).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=2000,
        help="Taille des pages (si pagination supportée).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=60,
        help="Nombre max de pages par session (sécurité).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Pause (secondes) entre requêtes (throttling).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Si présent, réécrit même si le fichier existe déjà.",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=0,
        help="Pour tests: limiter le nombre de sessions traitées (0 = pas de limite).",
    )
    args = parser.parse_args()

    scope_path = Path(args.scope)
    if not scope_path.is_absolute():
        scope_path = PROJECT_ROOT / scope_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = _iter_session_keys_from_scope(scope_path)
    if args.limit_sessions and args.limit_sessions > 0:
        sessions = sessions[: args.limit_sessions]

    _log(f"Scope: {scope_path} | sessions à traiter: {len(sessions)}")
    _log(f"Sortie: {out_dir} | format={args.format}")

    http, timeout_s = _build_http_session()

    stats: List[ExtractStats] = []
    n_total_rows = 0

    for idx, row in enumerate(sessions, start=1):
        session_key = int(row["session_key"])
        meeting_key = int(row["meeting_key"]) if "meeting_key" in row and pd.notna(row["meeting_key"]) else None
        year = int(row["year"]) if "year" in row and pd.notna(row["year"]) else None

        out_path = out_dir / f"laps_session_{session_key}.{args.format}"
        if out_path.exists() and not args.overwrite:
            # On ne refait pas inutilement
            try:
                if args.format == "csv":
                    n_existing = sum(1 for _ in open(out_path, "r", encoding="utf-8")) - 1
                else:
                    # parquet: lecture rapide
                    n_existing = len(pd.read_parquet(out_path))
            except Exception:
                n_existing = 0

            _log(f"[{idx}/{len(sessions)}] session_key={session_key} déjà présent -> skip ({n_existing} lignes estimées)")
            stats.append(
                ExtractStats(
                    session_key=session_key,
                    meeting_key=meeting_key,
                    year=year,
                    ok=True,
                    n_rows=max(n_existing, 0),
                    output_path=str(out_path),
                    error=None,
                )
            )
            n_total_rows += max(n_existing, 0)
            continue

        _log(f"[{idx}/{len(sessions)}] Extraction laps session_key={session_key} ...")
        t0 = time.time()

        try:
            df_laps = extract_laps_for_session(
                http=http,
                timeout_s=timeout_s,
                session_key=session_key,
                page_size=args.page_size,
                max_pages=args.max_pages,
                sleep_s=args.sleep,
            )

            # On ajoute year/meeting_key si absents dans le payload (utile pour la suite)
            if year is not None and "year" not in df_laps.columns:
                df_laps["year"] = year
            if meeting_key is not None and "meeting_key" not in df_laps.columns:
                df_laps["meeting_key"] = meeting_key
            if "session_key" not in df_laps.columns:
                df_laps["session_key"] = session_key

            # Export
            if args.format == "csv":
                df_laps.to_csv(out_path, index=False)
            else:
                df_laps.to_parquet(out_path, index=False)

            dt = time.time() - t0
            _log(f"  -> OK {len(df_laps)} lignes | {dt:.1f}s | {out_path.name}")

            stats.append(
                ExtractStats(
                    session_key=session_key,
                    meeting_key=meeting_key,
                    year=year,
                    ok=True,
                    n_rows=len(df_laps),
                    output_path=str(out_path),
                    error=None,
                )
            )
            n_total_rows += len(df_laps)

        except Exception as e:
            _log(f"  -> ERROR session_key={session_key}: {e}")
            stats.append(
                ExtractStats(
                    session_key=session_key,
                    meeting_key=meeting_key,
                    year=year,
                    ok=False,
                    n_rows=0,
                    output_path=None,
                    error=str(e),
                )
            )
            # on continue : une session en erreur ne bloque pas tout le pipeline
            continue

    # Manifest / rapport
    manifest_path = out_dir / "manifest_laps_extract.json"
    manifest = {
        "scope_path": str(scope_path),
        "out_dir": str(out_dir),
        "format": args.format,
        "n_sessions": len(sessions),
        "n_rows_total_estimated": n_total_rows,
        "stats": [s.__dict__ for s in stats],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for s in stats if s.ok)
    ko_count = sum(1 for s in stats if not s.ok)
    _log(f"Terminé. Sessions OK={ok_count} | KO={ko_count} | Rows~={n_total_rows}")
    _log(f"Manifest: {manifest_path}")

    # Code retour non bloquant : 0 si tout OK, 2 si au moins une session KO
    return 0 if ko_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
