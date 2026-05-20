"""
Download FDIC data files required by GeoRisk-IQ.

Run:
    pip install requests
    python download_fdic_files.py
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode

import requests

BASE = "https://api.fdic.gov/banks"
OUT_DIR = Path(__file__).parent


def fetch_all(
    endpoint: str,
    fields: List[str],
    out_file: str,
    limit: int = 10000,
    filters: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
):
    all_rows = []
    offset = 0

    while True:
        params = {
            "fields": ",".join(fields),
            "limit": limit,
            "offset": offset,
            "format": "json",
        }
        if filters:
            params["filters"] = filters
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order

        url = f"{BASE}/{endpoint}?{urlencode(params)}"
        print(f"Downloading {endpoint} offset={offset}")
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])

        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        offset += limit
        time.sleep(0.25)

    output = {"meta": {"total": len(all_rows), "source": endpoint}, "data": all_rows}
    out_path = OUT_DIR / out_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(all_rows):,} records -> {out_path}")


def maybe_fetch(
    endpoint: str,
    fields: List[str],
    out_file: str,
    skip_existing: bool = False,
    filters: Optional[str] = None,
):
    out_path = OUT_DIR / out_file
    if skip_existing and out_path.exists():
        print(f"Skipping existing file -> {out_path}")
        return
    fetch_all(endpoint, fields, out_file, filters=filters)


def get_latest_financial_report_date() -> str:
    params = {
        "fields": "REPDTE",
        "limit": 1,
        "sort_by": "REPDTE",
        "sort_order": "DESC",
        "format": "json",
    }
    response = requests.get(f"{BASE}/financials", params=params, timeout=30)
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        raise RuntimeError("Could not determine latest FDIC financial reporting date.")
    row = rows[0].get("data", rows[0])
    return str(row["REPDTE"])


def copy_financials_alias() -> None:
    """Keep the legacy FDICFinancials.txt filename in sync."""
    src = OUT_DIR / "FDICFinancialsFull.txt"
    dst = OUT_DIR / "FDICFinancials.txt"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied {src.name} -> {dst.name}")


def main(only: Optional[List[str]] = None, skip_existing: bool = False):
    selected = set(only or ["all"])
    download_all = "all" in selected

    institution_fields = [
        "NAME", "CERT", "STALP", "CITY", "LATITUDE", "LONGITUDE", "ASSET", "OFFICES"
    ]

    if download_all or "institutions" in selected:
        maybe_fetch("institutions", institution_fields, "Geospatiallocation.txt", skip_existing)
        maybe_fetch("institutions", institution_fields, "GeospatialFull.txt", skip_existing)

    if download_all or "financials" in selected:
        report_date = get_latest_financial_report_date()
        print(f"Latest FDIC financial report date: {report_date}")
        maybe_fetch(
            "financials",
            ["CERT", "REPDTE", "STALP", "ASSET", "NETINC", "ROA"],
            "FDICFinancialsFull.txt",
            skip_existing,
            filters=f"REPDTE:{report_date}",
        )
        # Some versions of the project referenced FDICFinancials.txt.
        copy_financials_alias()

    if download_all or "failures" in selected:
        maybe_fetch(
            "failures",
            ["NAME", "CERT", "CITY", "STALP", "FAILDATE", "RESTYPE"],
            "FDICFailure.txt",
            skip_existing,
        )

    print("\nFDIC files complete. HRSAHealthShortage.json must be added separately from HRSA data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FDIC files for GeoRisk-IQ.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["all", "institutions", "financials", "failures"],
        default=["all"],
        help="Download only the selected dataset groups.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not redownload files that already exist.",
    )
    args = parser.parse_args()
    main(only=args.only, skip_existing=args.skip_existing)
