"""
ground_truth.py

Build real ground-truth labels by combining:
- FDIC active institutions / institution locations -> supply surface
- FDIC bank failures -> real risk events
- FDIC financials -> ROA, NETINC, ASSET per institution/cell
"""

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_fdic import load_fdic_data
from src.h3_utils import build_h3_feature_matrix, compute_density_grid, lat_lng_to_h3

_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_INSTITUTIONS = os.path.join(_ROOT, "Geospatiallocation.txt")
DEFAULT_FAILURES = os.path.join(_ROOT, "FDICFailure.txt")
DEFAULT_FINANCIALS = os.path.join(_ROOT, "FDICFinancialsFull.txt")


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


def _read_json_records(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {os.path.abspath(path)}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = _extract_records(raw)
    df = pd.DataFrame(records)
    df.columns = [str(c).upper() for c in df.columns]
    return df


def load_failures(failure_json: str = DEFAULT_FAILURES) -> pd.DataFrame:
    """Load FDIC bank failure records from local JSON file."""
    df = _read_json_records(failure_json)

    if "FAILDATE" in df.columns:
        df["FAILDATE"] = pd.to_datetime(df["FAILDATE"], errors="coerce")
    elif "FAIL_DATE" in df.columns:
        df["FAILDATE"] = pd.to_datetime(df["FAIL_DATE"], errors="coerce")
    else:
        df["FAILDATE"] = pd.NaT

    df["FAIL_YEAR"] = df["FAILDATE"].dt.year.fillna(0).astype(int)
    print(f"Loaded {len(df):,} bank failure records")
    return df.reset_index(drop=True)


def load_financials(financials_json: str = DEFAULT_FINANCIALS) -> pd.DataFrame:
    """Load FDIC financial data such as ROA, NETINC, ASSET."""
    df = _read_json_records(financials_json)

    for col in ["CERT", "ROA", "NETINC", "ASSET", "LATITUDE", "LONGITUDE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "REPDTE" in df.columns:
        df["REPDTE"] = pd.to_datetime(df["REPDTE"], errors="coerce")
        df = df.sort_values("REPDTE").groupby("CERT", as_index=False).tail(1)

    print(f"Loaded {len(df):,} FDIC financial records")
    return df.reset_index(drop=True)


def _failure_labels_by_cell(failures: pd.DataFrame, institutions: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Map failures to coordinates through CERT if possible, then grid cell labels."""
    fail = failures.copy()
    inst = institutions.copy()

    if "CERT" in fail.columns and "CERT" in inst.columns:
        fail["CERT"] = pd.to_numeric(fail["CERT"], errors="coerce")
        inst["CERT"] = pd.to_numeric(inst["CERT"], errors="coerce")
        fail = fail.merge(
            inst[["CERT", "LATITUDE", "LONGITUDE"]].dropna(subset=["LATITUDE", "LONGITUDE"]),
            on="CERT",
            how="left",
            suffixes=("", "_INST"),
        )

    # Some failure downloads may already contain LATITUDE/LONGITUDE. If not, rows without coords are ignored.
    if "LATITUDE" not in fail.columns or "LONGITUDE" not in fail.columns:
        return pd.DataFrame(columns=["h3_index", "had_failure", "first_failure_year"])

    fail = fail.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    if fail.empty:
        return pd.DataFrame(columns=["h3_index", "had_failure", "first_failure_year"])

    fail["h3_index"] = fail.apply(
        lambda r: lat_lng_to_h3(r["LATITUDE"], r["LONGITUDE"], resolution), axis=1
    )

    labels = fail.groupby("h3_index").agg(
        had_failure=("h3_index", "count"),
        first_failure_year=("FAIL_YEAR", "min"),
    ).reset_index()
    labels["had_failure"] = (labels["had_failure"] > 0).astype(int)
    return labels


def _financial_features_by_cell(financials: pd.DataFrame, institutions: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Create financial features per grid cell."""
    fin = financials.copy()
    inst = institutions.copy()

    if "CERT" in fin.columns and "CERT" in inst.columns:
        fin["CERT"] = pd.to_numeric(fin["CERT"], errors="coerce")
        inst["CERT"] = pd.to_numeric(inst["CERT"], errors="coerce")
        fin = fin.merge(
            inst[["CERT", "LATITUDE", "LONGITUDE"]].dropna(subset=["LATITUDE", "LONGITUDE"]),
            on="CERT",
            how="left",
            suffixes=("", "_INST"),
        )

    if "LATITUDE" not in fin.columns or "LONGITUDE" not in fin.columns:
        return pd.DataFrame(columns=["h3_index", "fin_roa_mean", "fin_netinc_mean", "fin_asset_mean"])

    fin = fin.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    if fin.empty:
        return pd.DataFrame(columns=["h3_index", "fin_roa_mean", "fin_netinc_mean", "fin_asset_mean"])

    fin["h3_index"] = fin.apply(
        lambda r: lat_lng_to_h3(r["LATITUDE"], r["LONGITUDE"], resolution), axis=1
    )

    for col in ["ROA", "NETINC", "ASSET"]:
        if col not in fin.columns:
            fin[col] = 0
        fin[col] = pd.to_numeric(fin[col], errors="coerce").fillna(0)

    return fin.groupby("h3_index").agg(
        fin_roa_mean=("ROA", "mean"),
        fin_netinc_mean=("NETINC", "mean"),
        fin_asset_mean=("ASSET", "mean"),
    ).reset_index()


def build_features_with_labels(
    institutions_json: str = DEFAULT_INSTITUTIONS,
    failures_json: str = DEFAULT_FAILURES,
    financials_json: str = DEFAULT_FINANCIALS,
    resolution: int = 5,
) -> pd.DataFrame:
    """Build spatial + financial feature grid with had_failure labels."""
    institutions = load_fdic_data(institutions_json)
    failures = load_failures(failures_json)
    financials = load_financials(financials_json)

    density_df = compute_density_grid(
        institutions,
        lat_col="LATITUDE",
        lng_col="LONGITUDE",
        resolution=resolution,
        value_col="ASSET" if "ASSET" in institutions.columns else None,
    ).rename(columns={"count": "branch_count", "lat": "center_lat", "lng": "center_lng", "value_sum": "asset_total"})

    density_for_h3 = density_df.rename(columns={"branch_count": "count"})
    feature_df = build_h3_feature_matrix(density_for_h3, k_rings=[1, 2, 3]).rename(
        columns={"count": "branch_count"}
    )

    labels = _failure_labels_by_cell(failures, institutions, resolution)
    fin_features = _financial_features_by_cell(financials, institutions, resolution)

    feature_df = feature_df.merge(labels, on="h3_index", how="left")
    feature_df = feature_df.merge(fin_features, on="h3_index", how="left")

    feature_df["had_failure"] = feature_df["had_failure"].fillna(0).astype(int)
    feature_df["first_failure_year"] = feature_df["first_failure_year"].fillna(0).astype(int)
    for col in ["fin_roa_mean", "fin_netinc_mean", "fin_asset_mean"]:
        if col not in feature_df.columns:
            feature_df[col] = 0
        feature_df[col] = feature_df[col].fillna(0)

    print(f"Feature grid with labels: {len(feature_df):,} cells")
    print(f"Failure cells: {feature_df['had_failure'].sum():,}")
    return feature_df.reset_index(drop=True)


if __name__ == "__main__":
    df = build_features_with_labels()
    print(df.head())
