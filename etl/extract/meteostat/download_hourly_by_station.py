# etl/extract/meteostat/download_hourly_by_station.py
from __future__ import annotations

import argparse
import gzip
import os
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_URL = "https://data.meteostat.net/hourly"


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


@dataclass
class DownloadResult:
    station_id: str
    station_folder: str
    year: int
    url: str
    status: str  # OK / HTTP_ERROR / ERROR / SKIP
    http_code: int | None
    bytes_gz: int | None
    out_csv: str | None
    error: str | None
    elapsed_s: float


def download_gz(url: str, out_gz: Path, timeout: int = 60) -> tuple[int, int | None]:
    out_gz.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_gz.with_suffix(out_gz.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "F1PA/1.0 (Extract Meteostat hourly)"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        http_code = getattr(resp, "status", None)
        total = 0
        with tmp_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)

    os.replace(tmp_path, out_gz)
    return total, http_code


def gunzip_to_csv(in_gz: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_csv.with_suffix(out_csv.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    with gzip.open(in_gz, "rb") as f_in, tmp_path.open("wb") as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)

    os.replace(tmp_path, out_csv)


def purge_raw_tree(raw_dir: Path) -> tuple[int, int]:
    """
    Delete all *.csv.gz under raw_dir and then delete empty directories.
    Returns (deleted_files_count, deleted_dirs_count).
    """
    if not raw_dir.exists():
        return 0, 0

    deleted_files = 0
    deleted_dirs = 0

    # 1) delete gz files
    for gz_path in raw_dir.rglob("*.csv.gz"):
        try:
            gz_path.unlink()
            deleted_files += 1
        except Exception as e:
            log(f"WARNING: could not delete {gz_path}: {e}")

    # 2) delete empty dirs bottom-up
    # sort by path length descending to remove deepest first
    dirs = sorted([p for p in raw_dir.rglob("*") if p.is_dir()], key=lambda p: len(str(p)), reverse=True)
    for d in dirs:
        try:
            if not any(d.iterdir()):
                d.rmdir()
                deleted_dirs += 1
        except Exception:
            # ignore dirs not empty or locked
            pass

    # 3) if raw_dir itself is now empty, remove it too
    try:
        if raw_dir.exists() and not any(raw_dir.iterdir()):
            raw_dir.rmdir()
            deleted_dirs += 1
    except Exception:
        pass

    return deleted_files, deleted_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Meteostat hourly bulk CSVs for stations and years")
    p.add_argument("--mapping", default="data/extract/meteostat/mapping/circuit_station_mapping.csv")
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--out-dir", default="data/extract/meteostat/hourly")
    p.add_argument("--raw-dir", default="data/extract/meteostat/hourly_raw")
    p.add_argument("--report", default="data/extract/meteostat/hourly_download_report.csv")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--skip-existing", action="store_true", help="Skip download if output CSV already exists")
    p.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete the .csv.gz file immediately after successful decompression (only for OK downloads)",
    )
    p.add_argument(
        "--purge-raw",
        action="store_true",
        help="At end of run, delete any remaining *.csv.gz under raw-dir and remove empty directories",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    mapping_path = Path(args.mapping).resolve()
    out_dir = Path(args.out_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df_map = pd.read_csv(mapping_path)
    if "station_id" not in df_map.columns:
        raise ValueError(f"mapping file must contain 'station_id' column: {mapping_path}")

    required_cols = {"country", "locality", "distance_km", "circuit_name"}
    missing = required_cols - set(df_map.columns)
    if missing:
        raise ValueError(f"mapping file missing columns required for folder naming: {sorted(missing)}")

    df_alias = (
        df_map.copy()
        .assign(distance_km=pd.to_numeric(df_map["distance_km"], errors="coerce"))
        .sort_values(["station_id", "distance_km", "circuit_name"])
        .drop_duplicates(subset=["station_id"], keep="first")
    )

    station_to_folder: dict[str, str] = {}
    for _, r in df_alias.iterrows():
        station_id = str(r["station_id"])
        country = slug_fs(r["country"])
        locality = slug_fs(r["locality"])
        station_to_folder[station_id] = f"{station_id}__{country}__{locality}"

    station_ids = sorted(station_to_folder.keys())
    years = list(args.years)

    log("=== F1PA | Extract | Meteostat | download_hourly_by_station ===")
    log(f"stations={len(station_ids)} years={years}")
    log(f"out_dir={out_dir}")
    log(f"raw_dir={raw_dir}")
    log(f"report={report_path}")
    log(
        f"skip_existing={args.skip_existing} retries={args.retries} timeout={args.timeout} "
        f"delete_raw={args.delete_raw} purge_raw={args.purge_raw}"
    )

    results: list[DownloadResult] = []

    for station_id in station_ids:
        folder = station_to_folder[station_id]

        for year in years:
            url = f"{BASE_URL}/{year}/{station_id}.csv.gz"
            out_csv = out_dir / folder / f"{year}.csv"
            out_gz = raw_dir / folder / f"{year}.csv.gz"

            if args.skip_existing and out_csv.exists():
                results.append(
                    DownloadResult(
                        station_id=station_id,
                        station_folder=folder,
                        year=year,
                        url=url,
                        status="SKIP",
                        http_code=None,
                        bytes_gz=None,
                        out_csv=str(out_csv),
                        error=None,
                        elapsed_s=0.0,
                    )
                )
                continue

            attempt = 0
            start = time.time()
            http_code = None
            bytes_gz = None

            while attempt <= args.retries:
                attempt += 1
                try:
                    log(f"GET {url} -> {folder}/{year}.csv (attempt {attempt}/{args.retries + 1})")

                    bytes_gz, http_code = download_gz(url, out_gz, timeout=args.timeout)
                    gunzip_to_csv(out_gz, out_csv)

                    if args.delete_raw:
                        try:
                            out_gz.unlink(missing_ok=True)
                        except Exception as e:
                            log(f"WARNING: could not delete raw file {out_gz}: {e}")

                    elapsed = time.time() - start
                    results.append(
                        DownloadResult(
                            station_id=station_id,
                            station_folder=folder,
                            year=year,
                            url=url,
                            status="OK",
                            http_code=http_code,
                            bytes_gz=bytes_gz,
                            out_csv=str(out_csv),
                            error=None,
                            elapsed_s=round(elapsed, 3),
                        )
                    )
                    break

                except urllib.error.HTTPError as e:
                    elapsed = time.time() - start
                    msg = f"HTTPError: {e.code} {e.reason}"

                    if e.code == 404:
                        results.append(
                            DownloadResult(
                                station_id=station_id,
                                station_folder=folder,
                                year=year,
                                url=url,
                                status="HTTP_ERROR",
                                http_code=e.code,
                                bytes_gz=None,
                                out_csv=None,
                                error=msg,
                                elapsed_s=round(elapsed, 3),
                            )
                        )
                        break

                    if attempt > args.retries:
                        results.append(
                            DownloadResult(
                                station_id=station_id,
                                station_folder=folder,
                                year=year,
                                url=url,
                                status="HTTP_ERROR",
                                http_code=e.code,
                                bytes_gz=None,
                                out_csv=None,
                                error=msg,
                                elapsed_s=round(elapsed, 3),
                            )
                        )
                        break

                except Exception as e:
                    if attempt > args.retries:
                        elapsed = time.time() - start
                        results.append(
                            DownloadResult(
                                station_id=station_id,
                                station_folder=folder,
                                year=year,
                                url=url,
                                status="ERROR",
                                http_code=http_code,
                                bytes_gz=bytes_gz,
                                out_csv=None,
                                error=f"{type(e).__name__}: {e}",
                                elapsed_s=round(elapsed, 3),
                            )
                        )
                        break

    report_df = pd.DataFrame([r.__dict__ for r in results])
    report_df.to_csv(report_path, index=False)

    ok = int((report_df["status"] == "OK").sum())
    skip = int((report_df["status"] == "SKIP").sum())
    http_err = int((report_df["status"] == "HTTP_ERROR").sum())
    err = int((report_df["status"] == "ERROR").sum())

    log(f"DONE: OK={ok} SKIP={skip} HTTP_ERROR={http_err} ERROR={err}")
    log(f"Report written: {report_path}")

    # Final cleanup option: purge any remaining raw gz (and remove empty dirs)
    if args.purge_raw:
        deleted_files, deleted_dirs = purge_raw_tree(raw_dir)
        log(f"PURGE_RAW: deleted_files={deleted_files} deleted_dirs={deleted_dirs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
