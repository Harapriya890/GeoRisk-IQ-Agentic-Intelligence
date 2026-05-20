"""
fetch_fdic.py

Reads the FDIC institution JSON file downloaded from banks.data.fdic.gov and
returns a clean pandas DataFrame ready for geospatial analysis.
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd


DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "Geospatiallocation.txt")
REQUIRED_COLS = ["NAME", "STALP", "CITY", "LATITUDE", "LONGITUDE"]
OPTIONAL_COLS = ["ASSET", "OFFICES", "CERT", "NETINC", "ROA"]


def _extract_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        rows = raw.get("data", [])
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    records: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("data"), dict):
            records.append(item["data"])
        elif isinstance(item, dict):
            records.append(item)
    return records


def load_fdic_data(json_path: str = DEFAULT_JSON_PATH) -> pd.DataFrame:
    abs_path = os.path.abspath(json_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"FDIC data file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = _extract_records(raw)
    if not records:
        raise ValueError("No records found in FDIC JSON - check the file format.")

    df = pd.DataFrame(records)
    df.columns = [str(c).upper() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"FDIC data is missing required columns: {missing}")

    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    for col in OPTIONAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    total = raw.get("meta", {}).get("total", "unknown") if isinstance(raw, dict) else len(records)
    print(f"Loaded {len(df):,} institutions (API total: {total})")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_fdic_data()
    print(df[["NAME", "CITY", "STALP", "LATITUDE", "LONGITUDE"]].head(10))
    print(f"\nShape: {df.shape}")
    print(f"States represented: {df['STALP'].nunique()}")
