from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


def norm_text(s: str) -> str:
    """Normalize for matching: lowercase, remove accents, keep alnum/spaces."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ratio(a: str, b: str) -> float:
    """Similarity ratio [0..1]."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OpenF1 -> Wikipedia circuit match candidates")
    p.add_argument("--openf1", default="data/extract/openf1/openf1_circuits_used_2023_2025.csv")
    p.add_argument("--wikipedia", default="data/extract/wikipedia/circuits_wikipedia_extract.csv")
    p.add_argument("--out", default="data/extract/matching/openf1_wikipedia_match_candidates.csv")
    p.add_argument("--top-n", type=int, default=5, help="Number of top candidates per OpenF1 circuit")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    openf1_path = Path(args.openf1).resolve()
    wiki_path = Path(args.wikipedia).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_o = pd.read_csv(openf1_path)
    df_w = pd.read_csv(wiki_path)

    # Required columns
    req_o = {"circuit_key", "circuit_short_name", "country_name", "location"}
    req_w = {"circuit_name", "country", "locality", "latitude", "longitude", "circuit_url"}
    missing_o = req_o - set(df_o.columns)
    missing_w = req_w - set(df_w.columns)
    if missing_o:
        raise ValueError(f"OpenF1 file missing columns: {sorted(missing_o)}")
    if missing_w:
        raise ValueError(f"Wikipedia file missing columns: {sorted(missing_w)}")

    # Normalize
    df_o["o_short_norm"] = df_o["circuit_short_name"].map(norm_text)
    df_o["o_loc_norm"] = df_o["location"].map(norm_text)
    df_o["o_country_norm"] = df_o["country_name"].map(norm_text)

    df_w["w_name_norm"] = df_w["circuit_name"].map(norm_text)
    df_w["w_loc_norm"] = df_w["locality"].map(norm_text)
    df_w["w_country_norm"] = df_w["country"].map(norm_text)

    # Pre-split wiki by country to reduce noise (big win)
    wiki_by_country: dict[str, pd.DataFrame] = {}
    for c, g in df_w.groupby("w_country_norm"):
        wiki_by_country[c] = g

    rows = []

    for _, o in df_o.iterrows():
        o_key = int(o["circuit_key"])
        o_short = str(o["circuit_short_name"])
        o_loc = str(o["location"])
        o_country = str(o["country_name"])

        o_short_norm = o["o_short_norm"]
        o_loc_norm = o["o_loc_norm"]
        o_country_norm = o["o_country_norm"]

        # Candidate pool: same country if possible, else fallback to all
        pool = wiki_by_country.get(o_country_norm, df_w)

        scored = []
        for _, w in pool.iterrows():
            # Score components
            s1 = ratio(o_short_norm, w["w_name_norm"])     # short name vs circuit full name
            s2 = ratio(o_loc_norm, w["w_loc_norm"])       # location vs locality
            s3 = ratio(o_short_norm, w["w_loc_norm"])     # sometimes OpenF1 short is city-ish
            s4 = ratio(o_loc_norm, w["w_name_norm"])      # sometimes locality appears in circuit name

            # Weighted score: prioritize circuit name alignment, then location
            score = 0.55 * s1 + 0.25 * s2 + 0.10 * s3 + 0.10 * s4

            scored.append((score, w))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, int(args.top_n))]

        rank = 0
        for score, w in top:
            rank += 1
            rows.append(
                {
                    "circuit_key": o_key,
                    "openf1_circuit_short_name": o_short,
                    "openf1_country_name": o_country,
                    "openf1_location": o_loc,
                    "candidate_rank": rank,
                    "match_score": round(float(score), 4),
                    "wikipedia_circuit_name": w["circuit_name"],
                    "wikipedia_country": w["country"],
                    "wikipedia_locality": w["locality"],
                    "wikipedia_circuit_url": w["circuit_url"],
                    "wikipedia_latitude": w["latitude"],
                    "wikipedia_longitude": w["longitude"],
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["circuit_key", "candidate_rank"])
    out_df.to_csv(out_path, index=False)
    print(f"OK: wrote candidates to {out_path} rows={len(out_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
