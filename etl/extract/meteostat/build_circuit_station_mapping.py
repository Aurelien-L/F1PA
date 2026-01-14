# etl/extract/meteostat/build_circuit_station_mapping.py
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance grand cercle (km) entre deux points (lat/lon)."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def slugify(x: str) -> str:
    return (
        str(x)
        .lower()
        .strip()
        .replace("&", "and")
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build circuit -> Meteostat station mapping (nearest station)")
    p.add_argument("--circuits", default="data/extract/wikipedia/circuits_wikipedia_extract.csv")
    p.add_argument("--stations-db", default="data/extract/meteostat/stations/stations.db")
    p.add_argument("--out", default="data/extract/meteostat/mapping/circuit_station_mapping.csv")
    p.add_argument("--max-distance-km", type=float, default=500.0)
    return p.parse_args()


def load_circuits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected = {"circuit_name", "circuit_url", "locality", "country", "latitude", "longitude", "scraped_at_utc"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns in circuits CSV: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )

    # Identifiant technique stable pour le mapping
    df["circuit_id"] = (
        df["circuit_name"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "-", regex=True)
        .str.strip("-")
    )

    return df


def load_stations_with_name(db_path: Path) -> pd.DataFrame:
    """
    Schema constaté:
      - stations(id, country, latitude, longitude, elevation, ...)
      - names(station, name, language)
    On prend le nom EN si possible, sinon fallback.
    """
    conn = sqlite3.connect(str(db_path))

    stations = pd.read_sql_query(
        "SELECT id AS station_id, country AS station_country, "
        "latitude AS station_lat, longitude AS station_lon, elevation AS station_elevation "
        "FROM stations;",
        conn,
    )

    names = pd.read_sql_query(
        "SELECT station AS station_id, name AS station_name, language FROM names;",
        conn,
    )

    conn.close()

    names_en = names[names["language"] == "en"].drop_duplicates(subset=["station_id"], keep="first")
    names_any = names.drop_duplicates(subset=["station_id"], keep="first")

    stations = stations.merge(names_en[["station_id", "station_name"]], on="station_id", how="left")
    stations = stations.merge(
        names_any[["station_id", "station_name"]].rename(columns={"station_name": "station_name_fallback"}),
        on="station_id",
        how="left",
    )
    stations["station_name"] = stations["station_name"].fillna(stations["station_name_fallback"])
    stations = stations.drop(columns=["station_name_fallback"])

    return stations


def main() -> int:
    args = parse_args()

    circuits_csv = Path(args.circuits).resolve()
    db_path = Path(args.stations_db).resolve()
    out_csv = Path(args.out).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    circuits = load_circuits(circuits_csv)
    stations = load_stations_with_name(db_path)

    station_rows = stations.to_dict(orient="records")

    records: list[dict] = []

    for _, c in circuits.iterrows():
        c_lat = float(c["latitude"])
        c_lon = float(c["longitude"])

        best = None  # (distance, station_row)
        for s in station_rows:
            d_km = haversine_km(c_lat, c_lon, float(s["station_lat"]), float(s["station_lon"]))
            if best is None or d_km < best[0]:
                best = (d_km, s)

        d_km, s = best
        notes = ""
        if d_km > float(args.max_distance_km):
            notes = f"Distance>{args.max_distance_km}km"

        records.append(
            {
                # Circuit (source Wikipedia extract)
                "circuit_id": c["circuit_id"],
                "circuit_name": c["circuit_name"],
                "circuit_url": c["circuit_url"],
                "locality": c["locality"],
                "country": c["country"],
                "circuit_lat": c_lat,
                "circuit_lon": c_lon,
                "scraped_at_utc": c["scraped_at_utc"],
                # Station Meteostat (référence)
                "station_id": s["station_id"],
                "station_name": s.get("station_name"),
                "station_country": s.get("station_country"),
                "station_lat": s.get("station_lat"),
                "station_lon": s.get("station_lon"),
                "station_elevation": s.get("station_elevation"),
                # Décision
                "distance_km": round(float(d_km), 3),
                "selection_rule": "nearest",
                "coverage_notes": notes,
            }
        )

    out_df = pd.DataFrame.from_records(records).sort_values(["country", "circuit_name"])
    out_df.to_csv(out_csv, index=False)

    print(f"OK: wrote mapping to {out_csv}")
    print(f"Rows: {len(out_df)} | Unique stations: {out_df['station_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())