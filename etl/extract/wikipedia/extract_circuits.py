# etl/extract/wikipedia/extract_circuits.py
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_URL = "https://en.wikipedia.org/wiki/List_of_Formula_One_circuits"
BASE_WIKI_URL = "https://en.wikipedia.org"

DEFAULT_HEADERS = {
    "User-Agent": "F1PredictiveAssistant/1.0 (Academic project, ETL extract)"
}


def project_root() -> Path:
    # .../F1PA/etl/extract/wikipedia/extract_circuits.py -> parents[3] = F1PA
    return Path(__file__).resolve().parents[3]


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Formula 1 circuits reference data from Wikipedia (name, url, locality, country, lat/lon).")
    p.add_argument("--url", default=DEFAULT_URL, help="Wikipedia page URL listing F1 circuits")
    p.add_argument("--out", default=None, help="Output CSV path (default: data/extract/wikipedia/circuits_wikipedia_extract.csv)")
    p.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between per-circuit page requests")
    p.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds")
    p.add_argument("--max-circuits", type=int, default=None, help="Optional limit for debugging")
    return p.parse_args()


def find_circuits_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    The notebook used tables[1]. Here we make it more robust:
    pick the wikitable which has headers containing both 'Circuit' and ('Location' or 'Country').
    """
    tables = soup.find_all("table", {"class": "wikitable"})
    if not tables:
        return None

    def score_table(tbl: BeautifulSoup) -> int:
        headers = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
        score = 0
        if any("circuit" in h for h in headers):
            score += 2
        if any("location" in h for h in headers):
            score += 1
        if any("country" in h for h in headers):
            score += 1
        return score

    ranked = sorted(((score_table(t), t) for t in tables), key=lambda x: x[0], reverse=True)
    best_score, best_table = ranked[0]
    return best_table if best_score >= 2 else None


def extract_list_from_table(table: BeautifulSoup) -> list[dict]:
    rows = table.find("tbody").find_all("tr")[1:]  # skip header row

    circuits_list: list[dict] = []
    for row in rows:
        cols = row.find_all("td")
        # Notebook expected >= 6 columns
        if len(cols) < 6:
            continue

        circuit_link = cols[0].find("a")
        circuit_name = circuit_link.get_text(strip=True) if circuit_link else None
        circuit_url = (BASE_WIKI_URL + circuit_link["href"]) if circuit_link and circuit_link.get("href") else None

        location_links = cols[4].find_all("a")
        locality = location_links[0].get_text(strip=True) if location_links else None

        country_links = cols[5].find_all("a")
        country = country_links[-1].get_text(strip=True) if country_links else None

        if circuit_name and circuit_url:
            circuits_list.append(
                {
                    "circuit_name": circuit_name,
                    "circuit_url": circuit_url,
                    "locality": locality,
                    "country": country,
                }
            )

    return circuits_list


def extract_coordinates(session: requests.Session, url: str, timeout: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract lat/lon from a Wikipedia page using the 'span.geo' convention (lat;lon).
    """
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None, None

    soup = BeautifulSoup(resp.text, "html.parser")
    geo = soup.select_one("span.geo")
    if geo:
        txt = geo.get_text(strip=True)
        if ";" in txt:
            try:
                lat_str, lon_str = txt.split(";")
                return float(lat_str.strip()), float(lon_str.strip())
            except ValueError:
                return None, None
    return None, None


def main() -> int:
    args = parse_args()

    out_path = Path(args.out).resolve() if args.out else (project_root() / "data" / "extract" / "wikipedia" / "circuits_wikipedia_extract.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("=== F1PA | Extract | Wikipedia | extract_circuits ===")
    log(f"url={args.url}")
    log(f"out={out_path}")
    log(f"sleep={args.sleep}s timeout={args.timeout}s max_circuits={args.max_circuits}")

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    resp = session.get(args.url, timeout=args.timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = find_circuits_table(soup)
    if table is None:
        raise RuntimeError("Could not find a suitable circuits table on the Wikipedia page.")

    circuits_list = extract_list_from_table(table)
    if args.max_circuits is not None:
        circuits_list = circuits_list[: args.max_circuits]

    log(f"Circuits extracted from list page: {len(circuits_list)}")

    df = pd.DataFrame(circuits_list)

    lats: list[Optional[float]] = []
    lons: list[Optional[float]] = []

    for i, row in df.iterrows():
        url = row["circuit_url"]
        lat, lon = extract_coordinates(session, url, timeout=args.timeout)
        lats.append(lat)
        lons.append(lon)

        if args.sleep and i < len(df) - 1:
            time.sleep(args.sleep)

    df["latitude"] = lats
    df["longitude"] = lons
    df["scraped_at_utc"] = datetime.now(timezone.utc).isoformat()

    df.to_csv(out_path, index=False)
    log(f"OK: wrote circuits extract: {out_path}")

    missing_coords = int(df["latitude"].isna().sum() + df["longitude"].isna().sum())
    if missing_coords:
        log(f"WARNING: missing coordinates for some rows (check output).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())