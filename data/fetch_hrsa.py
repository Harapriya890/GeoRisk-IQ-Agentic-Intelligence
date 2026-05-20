"""
fetch_hrsa.py

Load HRSA Health Professional Shortage Area (HPSA) data from local JSON.
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd


DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "HRSAHealthShortage.json")


def _extract_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        rows = raw.get("data", raw.get("records", []))
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


def load_hrsa_data(json_path: str = DEFAULT_JSON_PATH) -> pd.DataFrame:
    abs_path = os.path.abspath(json_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"HRSA data file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = _extract_records(raw)
    df = pd.DataFrame(records)
    df.columns = [str(c).upper() for c in df.columns]

    required = ["LATITUDE", "LONGITUDE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"HRSA data is missing required columns: {missing}")

    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    for col in ["HPSA_SCORE", "SHORTAGE", "POP_BELOW_POVERTY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    print(f"Loaded {len(df):,} HRSA shortage areas")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_hrsa_data()
    print(df.head())
