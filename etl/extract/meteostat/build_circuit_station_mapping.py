# etl/extract/meteostat/build_circuit_station_mapping.py
from __future__ import annotations

import argparse
import math
import re
import sqlite3
import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


BULK_BASE_URL = "https://data.meteostat.net/hourly"


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}")


def slug_fs(value: str, max_len: int = 40) -> str:
    s = str(value).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        s = "unknown"
    return s[:max_len]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Earth radius
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def http_exists(url: str, timeout: int = 20) -> tuple[bool, int | None]:
    """
    Check if URL exists. Uses HEAD first; falls back to GET on servers which reject HEAD.
    Returns (exists, http_code).
    """
    # HEAD
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "F1PA/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None)
            return (code == 200), code
    except urllib.error.HTTPError as e:
        return False, e.code
    except Exception:
        # fallback GET (range)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "F1PA/1.0", "Range": "bytes=0-0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", None)
                return (code == 200 or code == 206), code
        except urllib.error.HTTPError as e:
            return False, e.code
        except Exception:
            return False, None


@dataclass
class StationCandidate:
    station_id: str
    station_name: str | None
    station_country: str | None
    station_lat: float
    station_lon: float
    station_elevation: float | None
    distance_km: float
    years_ok: list[int]
    years_missing: list[int]


def fetch_station_name(conn: sqlite3.Connection, station_id: str) -> str | None:
    # English preferred; fall back to any
    row = conn.execute(
        "SELECT name FROM names WHERE station = ? AND language = 'en' LIMIT 1",
        (station_id,),
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    row = conn.execute(
        "SELECT name FROM names WHERE station = ? LIMIT 1",
        (station_id,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def query_nearest_stations(conn: sqlite3.Connection, lat: float, lon: float, limit: int) -> pd.DataFrame:
    # Simple bounding box prefilter to keep it fast; then compute exact distance in Python.
    # 1 deg ~ 111km; we start wide enough.
    delta = 5.0
    df = pd.read_sql_query(
        """
        SELECT id, country, latitude, longitude, elevation
        FROM stations
        WHERE latitude BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
        """,
        conn,
        params=(lat - delta, lat + delta, lon - delta, lon + delta),
    )
    if df.empty:
        # fallback: no bbox match, take a larger bbox
        delta = 15.0
        df = pd.read_sql_query(
            """
            SELECT id, country, latitude, longitude, elevation
            FROM stations
            WHERE latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
            """,
            conn,
            params=(lat - delta, lat + delta, lon - delta, lon + delta),
        )

    # compute distance
    df["distance_km"] = df.apply(
        lambda r: haversine_km(lat, lon, float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )
    df = df.sort_values("distance_km").head(limit).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build circuit->station mapping with Meteostat bulk hourly availability checks")
    p.add_argument("--circuits", default="data/extract/wikipedia/circuits_wikipedia_filtered_2023_2025.csv")
    p.add_argument("--stations-db", default="data/extract/meteostat/stations/stations.db")
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--top-n", type=int, default=10, help="How many nearest station candidates to evaluate")
    p.add_argument("--timeout", type=int, default=20, help="HTTP timeout for availability checks")
    p.add_argument("--out", default="data/extract/meteostat/mapping/circuit_station_mapping_2023_2025.csv")
    p.add_argument("--out-candidates", default="data/extract/meteostat/mapping/circuit_station_candidates_2023_2025.csv")
    p.add_argument("--out-decisions", default="data/extract/meteostat/mapping/circuit_station_mapping_decisions_2023_2025.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    circuits_path = Path(args.circuits).resolve()
    db_path = Path(args.stations_db).resolve()

    out_path = Path(args.out).resolve()
    out_candidates_path = Path(args.out_candidates).resolve()
    out_decisions_path = Path(args.out_decisions).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_c = pd.read_csv(circuits_path)

    required = {"circuit_name", "circuit_url", "locality", "country", "latitude", "longitude", "scraped_at_utc"}
    missing = required - set(df_c.columns)
    if missing:
        raise ValueError(f"Circuits file missing columns: {sorted(missing)}")

    years = list(args.years)
    log("=== F1PA | Extract | Meteostat | build_circuit_station_mapping (availability-aware) ===")
    log(f"circuits={len(df_c)} years={years} top_n={args.top_n}")
    log(f"stations_db={db_path}")
    log(f"circuits={circuits_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    all_candidates_rows = []
    decisions_rows = []
    mapping_rows = []

    for _, c in df_c.iterrows():
        circuit_name = str(c["circuit_name"])
        circuit_url = str(c["circuit_url"])
        locality = str(c["locality"])
        country = str(c["country"])
        clat = float(c["latitude"])
        clon = float(c["longitude"])
        scraped_at = str(c["scraped_at_utc"])

        circuit_id = slug_fs(circuit_name, max_len=64)

        nearest_df = query_nearest_stations(conn, clat, clon, limit=args.top_n)
        if nearest_df.empty:
            log(f"WARNING: no stations found near {circuit_name}")
            continue

        candidates: list[StationCandidate] = []
        for rank, r in enumerate(nearest_df.itertuples(index=False), start=1):
            station_id = str(r.id)
            st_lat = float(r.latitude)
            st_lon = float(r.longitude)
            st_country = str(r.country) if r.country is not None else None
            st_elev = float(r.elevation) if r.elevation is not None else None
            dist = float(r.distance_km)

            st_name = fetch_station_name(conn, station_id)

            years_ok = []
            years_missing = []
            for y in years:
                url = f"{BULK_BASE_URL}/{y}/{station_id}.csv.gz"
                ok, code = http_exists(url, timeout=args.timeout)
                if ok:
                    years_ok.append(y)
                else:
                    years_missing.append(y)

            cand = StationCandidate(
                station_id=station_id,
                station_name=st_name,
                station_country=st_country,
                station_lat=st_lat,
                station_lon=st_lon,
                station_elevation=st_elev,
                distance_km=round(dist, 3),
                years_ok=years_ok,
                years_missing=years_missing,
            )
            candidates.append(cand)

            all_candidates_rows.append(
                {
                    "circuit_id": circuit_id,
                    "circuit_name": circuit_name,
                    "country": country,
                    "locality": locality,
                    "circuit_lat": clat,
                    "circuit_lon": clon,
                    "station_rank": rank,
                    "station_id": station_id,
                    "station_name": st_name,
                    "station_country": st_country,
                    "station_lat": st_lat,
                    "station_lon": st_lon,
                    "station_elevation": st_elev,
                    "distance_km": round(dist, 3),
                    "years_ok": "|".join(map(str, years_ok)),
                    "years_missing": "|".join(map(str, years_missing)),
                }
            )

        # Selection rule: first candidate with full coverage
        chosen = None
        for rank, cand in enumerate(candidates, start=1):
            if len(cand.years_missing) == 0:
                chosen = (rank, cand)
                break

        # If none has full coverage, choose best coverage then nearest
        if chosen is None:
            candidates_sorted = sorted(
                enumerate(candidates, start=1),
                key=lambda x: (len(x[1].years_missing), x[1].distance_km),
            )
            chosen = candidates_sorted[0]
            selection_rule = "best_coverage_then_nearest"
        else:
            selection_rule = "nearest_with_full_coverage"

        chosen_rank, chosen_cand = chosen
        coverage_notes = ""
        if chosen_cand.years_missing:
            coverage_notes = f"missing_years={','.join(map(str, chosen_cand.years_missing))}"

        decisions_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": circuit_name,
                "chosen_station_id": chosen_cand.station_id,
                "chosen_station_rank": chosen_rank,
                "selection_rule": selection_rule,
                "chosen_distance_km": chosen_cand.distance_km,
                "years_ok": "|".join(map(str, chosen_cand.years_ok)),
                "years_missing": "|".join(map(str, chosen_cand.years_missing)),
            }
        )

        mapping_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": circuit_name,
                "circuit_url": circuit_url,
                "locality": locality,
                "country": country,
                "circuit_lat": clat,
                "circuit_lon": clon,
                "scraped_at_utc": scraped_at,
                "station_id": chosen_cand.station_id,
                "station_name": chosen_cand.station_name,
                "station_country": chosen_cand.station_country,
                "station_lat": chosen_cand.station_lat,
                "station_lon": chosen_cand.station_lon,
                "station_elevation": chosen_cand.station_elevation,
                "distance_km": chosen_cand.distance_km,
                "selection_rule": selection_rule,
                "coverage_notes": coverage_notes,
            }
        )

        log(
            f"{circuit_name}: chose {chosen_cand.station_id} (rank={chosen_rank}, dist={chosen_cand.distance_km}km, "
            f"missing={chosen_cand.years_missing})"
        )

    conn.close()

    pd.DataFrame(all_candidates_rows).to_csv(out_candidates_path, index=False)
    pd.DataFrame(decisions_rows).to_csv(out_decisions_path, index=False)
    pd.DataFrame(mapping_rows).to_csv(out_path, index=False)

    log(f"OK: wrote candidates: {out_candidates_path}")
    log(f"OK: wrote decisions:  {out_decisions_path}")
    log(f"OK: wrote mapping:    {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
