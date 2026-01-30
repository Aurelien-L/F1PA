# etl/transform/05_join_weather_hourly.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, unquote

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_TRANSFORM_DIR = PROJECT_ROOT / "data" / "transform"
DATA_EXTRACT_DIR = PROJECT_ROOT / "data" / "extract"

DEFAULT_LAPS_CTX_DIR = DATA_TRANSFORM_DIR / "laps_with_context_by_session"
DEFAULT_OUT_DIR = DATA_TRANSFORM_DIR / "laps_with_weather_by_session"

OPENF1_WIKI_MAP_PATH = DATA_EXTRACT_DIR / "matching" / "openf1_to_wikipedia_circuit_map.csv"
WIKI_STATION_MAP_PATH = DATA_EXTRACT_DIR / "meteostat" / "mapping" / "circuit_station_mapping_2023_2025.csv"
METEOSTAT_HOURLY_ROOT = DATA_EXTRACT_DIR / "meteostat" / "hourly"

WEATHER_COLS = ["temp", "rhum", "pres", "wspd", "wdir", "prcp", "cldc"]


def _log(msg: str) -> None:
    print(f"[05_join_weather_hourly] {msg}")


@dataclass
class WeatherJoinStats:
    session_key: int
    input_path: str
    output_path: Optional[str]
    n_in: int
    n_out: int
    circuit_key: Optional[int]
    wikipedia_circuit_url: Optional[str]
    station_id: Optional[str]
    n_missing_station: int
    n_missing_weather: int
    ok: bool
    error: Optional[str]


def _list_session_files(in_dir: Path, fmt: str) -> List[Path]:
    pattern = re.compile(rf"^laps_session_(\d+)\.{fmt}$")
    return [p for p in sorted(in_dir.iterdir()) if p.is_file() and pattern.match(p.name)]


def _extract_session_key(path: Path) -> int:
    m = re.search(r"laps_session_(\d+)\.", path.name)
    if not m:
        raise ValueError(f"Nom inattendu: {path.name}")
    return int(m.group(1))


def _circuit_id_from_wikipedia_url(url: str) -> str:
    """
    Extrait l'identifiant de page Wikipedia (dernier segment du path).
    Ex:
      https://en.wikipedia.org/wiki/Albert_Park_Circuit -> Albert_Park_Circuit
      .../wiki/Aut%C3%B3dromo_Hermanos_Rodr%C3%ADguez -> Autódromo_Hermanos_Rodríguez
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL Wikipedia vide.")
    parsed = urlparse(url)
    path = parsed.path  # /wiki/Albert_Park_Circuit
    if "/wiki/" not in path:
        raise ValueError(f"URL Wikipedia inattendue: {url}")
    page = path.split("/wiki/", 1)[1]
    page = unquote(page)  # decode %C3%B3 etc.
    return page


def _load_openf1_to_wiki_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mapping OpenF1->Wikipedia introuvable: {path}")

    df = pd.read_csv(path)

    if "circuit_key" not in df.columns:
        raise ValueError("Le mapping OpenF1->Wikipedia doit contenir 'circuit_key'.")
    if "wikipedia_circuit_url" not in df.columns:
        raise ValueError("Le mapping OpenF1->Wikipedia doit contenir 'wikipedia_circuit_url'.")

    df["circuit_key"] = pd.to_numeric(df["circuit_key"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["circuit_key"]).copy()

    # circuit_id dérivé
    df["circuit_id"] = df["wikipedia_circuit_url"].apply(_circuit_id_from_wikipedia_url)

    df = df.drop_duplicates(subset=["circuit_key"], keep="first").copy()
    df = df.set_index("circuit_key", drop=False)
    return df


def _load_wiki_to_station_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mapping Wikipedia->Station introuvable: {path}")

    df = pd.read_csv(path)

    # ICI : on s'aligne sur la réalité de tes colonnes
    if "circuit_url" not in df.columns:
        raise ValueError("Le mapping Meteostat doit contenir 'circuit_url' (URL Wikipedia).")
    if "station_id" not in df.columns:
        raise ValueError("Le mapping Meteostat doit contenir 'station_id'.")

    df = df.drop_duplicates(subset=["circuit_url"], keep="first").copy()
    df = df.set_index("circuit_url", drop=False)
    return df


def _find_station_folder(station_id: str) -> Path:
    matches = list(METEOSTAT_HOURLY_ROOT.glob(f"{station_id}__*"))
    if not matches:
        raise FileNotFoundError(f"Aucun dossier hourly pour station_id={station_id} sous {METEOSTAT_HOURLY_ROOT}")
    return matches[0]


def _load_weather_year(station_id: str, year: int) -> pd.DataFrame:
    folder = _find_station_folder(station_id)
    year_path = folder / f"{year}.csv"
    if not year_path.exists():
        raise FileNotFoundError(f"Fichier météo introuvable: {year_path}")

    df = pd.read_csv(year_path)

    for c in ["year", "month", "day", "hour"]:
        if c not in df.columns:
            raise ValueError(f"Colonne temporelle manquante dans Meteostat: {c}")

    df["weather_hour_utc"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        utc=True,
        errors="coerce",
    )

    keep_cols = ["weather_hour_utc"] + [c for c in WEATHER_COLS if c in df.columns]
    return df[keep_cols]


def main() -> int:
    parser = argparse.ArgumentParser(description="Join hourly Meteostat weather to lap-level data.")
    parser.add_argument("--laps-dir", type=str, default=str(DEFAULT_LAPS_CTX_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv")
    parser.add_argument("--openf1-wiki-map", type=str, default=str(OPENF1_WIKI_MAP_PATH))
    parser.add_argument("--wiki-station-map", type=str, default=str(WIKI_STATION_MAP_PATH))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-sessions", type=int, default=0)
    args = parser.parse_args()

    laps_dir = Path(args.laps_dir)
    if not laps_dir.is_absolute():
        laps_dir = PROJECT_ROOT / laps_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    openf1_wiki_path = Path(args.openf1_wiki_map)
    if not openf1_wiki_path.is_absolute():
        openf1_wiki_path = PROJECT_ROOT / openf1_wiki_path

    wiki_station_path = Path(args.wiki_station_map)
    if not wiki_station_path.is_absolute():
        wiki_station_path = PROJECT_ROOT / wiki_station_path

    df_openf1_wiki = _load_openf1_to_wiki_map(openf1_wiki_path)
    df_wiki_station = _load_wiki_to_station_map(wiki_station_path)

    files = _list_session_files(laps_dir, fmt=args.format)
    if args.limit_sessions and args.limit_sessions > 0:
        files = files[: args.limit_sessions]

    _log(f"Laps context: {laps_dir} | sessions: {len(files)}")
    _log(f"OpenF1->Wiki map: {openf1_wiki_path}")
    _log(f"Wiki->Station map: {wiki_station_path}")
    _log(f"Meteostat hourly root: {METEOSTAT_HOURLY_ROOT}")
    _log(f"Sortie: {out_dir}")

    stats: List[WeatherJoinStats] = []

    for idx, path in enumerate(files, start=1):
        session_key = _extract_session_key(path)
        out_path = out_dir / f"laps_session_{session_key}.{args.format}"

        if out_path.exists() and not args.overwrite:
            _log(f"[{idx}/{len(files)}] session_key={session_key} -> skip")
            stats.append(
                WeatherJoinStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=str(out_path),
                    n_in=0,
                    n_out=0,
                    circuit_key=None,
                    wikipedia_circuit_url=None,
                    station_id=None,
                    n_missing_station=0,
                    n_missing_weather=0,
                    ok=True,
                    error=None,
                )
            )
            continue

        _log(f"[{idx}/{len(files)}] Join météo session_key={session_key} ...")

        try:
            df_laps = pd.read_csv(path) if args.format == "csv" else pd.read_parquet(path)
            n_in = len(df_laps)
            if n_in == 0:
                raise RuntimeError("Fichier laps vide (n_in=0).")

            if "circuit_key" not in df_laps.columns:
                raise ValueError("Colonne 'circuit_key' absente des laps.")
            circuit_key = int(df_laps["circuit_key"].iloc[0])

            # circuit_key -> wikipedia URL -> circuit_id
            if circuit_key not in df_openf1_wiki.index:
                raise RuntimeError(f"circuit_key={circuit_key} absent du mapping OpenF1->Wikipedia")

            wiki_url = str(df_openf1_wiki.loc[circuit_key, "wikipedia_circuit_url"])

            if wiki_url not in df_wiki_station.index:
                raise RuntimeError(f"wikipedia_circuit_url='{wiki_url}' absent du mapping Meteostat")

            station_id = str(df_wiki_station.loc[wiki_url, "station_id"])

            if "lap_hour_utc" not in df_laps.columns:
                raise ValueError("Colonne 'lap_hour_utc' absente (clé météo).")
            df_laps["lap_hour_utc"] = pd.to_datetime(df_laps["lap_hour_utc"], errors="coerce", utc=True)

            years = df_laps["lap_hour_utc"].dt.year.dropna().unique().tolist()
            years = [int(y) for y in years]
            if not years:
                raise RuntimeError("Impossible de déterminer l'année depuis lap_hour_utc.")

            df_weather = pd.concat([_load_weather_year(station_id, y) for y in years], ignore_index=True)

            df_join = df_laps.merge(
                df_weather,
                how="left",
                left_on="lap_hour_utc",
                right_on="weather_hour_utc",
            )

            n_missing_weather = int(df_join[WEATHER_COLS].isna().all(axis=1).sum())

            # MVP : exclure laps sans météo
            df_join = df_join[df_join[WEATHER_COLS].notna().any(axis=1)].copy()
            n_out = len(df_join)

            # Traçabilité
            df_join["station_id"] = station_id
            df_join["wikipedia_circuit_url"] = wiki_url

            if args.format == "csv":
                df_join.to_csv(out_path, index=False)
            else:
                df_join.to_parquet(out_path, index=False)

            _log(f"  -> OK {n_in} -> {n_out} | station_id={station_id} | missing_weather={n_missing_weather}")

            stats.append(
                WeatherJoinStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=str(out_path),
                    n_in=n_in,
                    n_out=n_out,
                    circuit_key=circuit_key,
                    wikipedia_circuit_url=wiki_url,
                    station_id=station_id,
                    n_missing_station=0,
                    n_missing_weather=n_missing_weather,
                    ok=True,
                    error=None,
                )
            )

        except Exception as e:
            _log(f"  -> ERROR session_key={session_key}: {e}")
            stats.append(
                WeatherJoinStats(
                    session_key=session_key,
                    input_path=str(path),
                    output_path=None,
                    n_in=0,
                    n_out=0,
                    circuit_key=None,
                    wikipedia_circuit_url=None,
                    station_id=None,
                    n_missing_station=0,
                    n_missing_weather=0,
                    ok=False,
                    error=str(e),
                )
            )

    report_df = pd.DataFrame([s.__dict__ for s in stats])
    report_csv = out_dir / "report_laps_weather_join.csv"
    report_df.to_csv(report_csv, index=False)

    manifest = {
        "laps_context_dir": str(laps_dir),
        "openf1_wiki_map": str(openf1_wiki_path),
        "wiki_station_map": str(wiki_station_path),
        "meteostat_hourly_root": str(METEOSTAT_HOURLY_ROOT),
        "out_dir": str(out_dir),
        "n_sessions": len(files),
        "n_ok": int((report_df["ok"] == True).sum()) if not report_df.empty else 0,  # noqa: E712
        "n_ko": int((report_df["ok"] == False).sum()) if not report_df.empty else 0,  # noqa: E712
        "report_csv": str(report_csv),
    }
    manifest_path = out_dir / "manifest_weather_join.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log(f"Rapport: {report_csv}")
    _log(f"Manifest: {manifest_path}")

    return 0 if manifest["n_ko"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
