"""Download a reproducible slice of the NGSIM vehicle-trajectory dataset.

NGSIM (Next Generation SIMulation) was collected by the U.S. Federal Highway
Administration using synchronised overhead cameras; every vehicle on the study
section is tracked at 10 Hz. The data is public domain and served through the
U.S. DOT Socrata open-data API, so no registration is required.

Usage:
    python -m src.download_data            # downloads both sites
    python -m src.download_data --site i-80
"""
from __future__ import annotations

import argparse
import io
import time

import pandas as pd
import requests

from . import config as C

COLUMNS = [
    "vehicle_id", "frame_id", "total_frames", "global_time", "local_x",
    "local_y", "v_length", "v_width", "v_class", "v_vel", "v_acc", "lane_id",
    "preceding", "following", "space_headway", "time_headway", "location",
]


def _fetch_page(where: str, offset: int, retries: int = 5) -> pd.DataFrame:
    params = {
        "$select": ",".join(COLUMNS),
        "$where": where,
        "$order": "global_time, vehicle_id",
        "$limit": C.PAGE_SIZE,
        "$offset": offset,
    }
    for attempt in range(retries):
        try:
            r = requests.get(C.NGSIM_API, params=params, timeout=180)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except (requests.RequestException, pd.errors.ParserError) as exc:
            wait = 2 ** attempt
            print(f"  page offset={offset} failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not download page at offset {offset}")


def download_site(site: str) -> pd.DataFrame:
    out = C.DATA_RAW / f"ngsim_{site}.csv"
    if out.exists():
        print(f"[skip] {out.name} already present")
        return pd.read_csv(out)

    win = C.SITES[site]
    t0, t1 = win["t_start_ms"], win["t_start_ms"] + win["duration_ms"]
    where = f"location='{site}' AND global_time >= {t0} AND global_time < {t1}"
    print(f"Downloading NGSIM {site}: global_time [{t0}, {t1})")

    pages, offset = [], 0
    while True:
        page = _fetch_page(where, offset)
        pages.append(page)
        offset += len(page)
        print(f"  {site}: {offset:,} rows")
        if len(page) < C.PAGE_SIZE:
            break

    df = pd.concat(pages, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[done] {out} ({len(df):,} rows, {out.stat().st_size/1e6:.1f} MB)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(C.SITES), default=None)
    args = ap.parse_args()
    for site in ([args.site] if args.site else C.SITES):
        download_site(site)


if __name__ == "__main__":
    main()
