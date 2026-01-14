from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def fetch_all(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[tuple]:
    cur = conn.execute(query)
    rows = cur.fetchmany(limit)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/extract/meteostat/stations/stations.db")
    args = p.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"stations.db not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 1) Tables
    tables = fetch_all(conn, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", limit=1000)
    print("Tables:", [r["name"] for r in tables])

    # 2) Sample stations
    print("\nSample stations:")
    rows = fetch_all(
        conn,
        "SELECT id, country, latitude, longitude, elevation FROM stations LIMIT 10;"
    )
    for r in rows:
        print(dict(r))

    # 3) Sample station names
    print("\nSample station names:")
    rows = fetch_all(
        conn,
        "SELECT station, name, language FROM names LIMIT 10;"
    )
    for r in rows:
        print(dict(r))

    # 4) Sample inventory
    print("\nInventory schema (PRAGMA table_info):") # pour vérifier le schéma
    rows = fetch_all(conn, "PRAGMA table_info(inventory);", limit=200)
    for r in rows:
        print(dict(r))

    print("\nSample inventory rows (SELECT *):")
    rows = fetch_all(conn, "SELECT * FROM inventory LIMIT 10;")
    for r in rows:
        print(dict(r))


    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())