from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from datetime import datetime


DEFAULT_URL = "https://data.meteostat.net/stations.db"


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path, force: bool = False, timeout: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        log(f"SKIP: file already exists: {out_path}")
        log(f"      sha256={sha256_file(out_path)} size={out_path.stat().st_size} bytes")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    log(f"DOWNLOAD: {url}")
    log(f"TO:       {out_path}")

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "F1PA/1.0 (Extract Meteostat stations.db)"
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            if status is not None and status >= 400:
                raise RuntimeError(f"HTTP error: {status}")

            total = 0
            with tmp_path.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)

        os.replace(tmp_path, out_path)

        log(f"OK: downloaded {total} bytes")
        log(f"sha256={sha256_file(out_path)}")

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {e}") from e


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Meteostat stations SQLite database (stations.db)")
    p.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source URL for stations.db",
    )
    p.add_argument(
        "--out",
        default="data/extract/meteostat/stations/stations.db",
        help="Output path for stations.db",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output file already exists",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out_path = Path(args.out).resolve()

    log("=== F1PA | Extract | Meteostat | download_stations_db ===")
    log(f"url={args.url}")
    log(f"out={out_path}")
    log(f"force={args.force}")

    download_file(url=args.url, out_path=out_path, force=args.force, timeout=args.timeout)

    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))