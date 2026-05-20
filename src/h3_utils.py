"""
h3_utils.py

Geospatial grid utility functions (H3-compatible interface, no h3 library required).

Core idea: Divide any geography into a uniform rectangular grid. Each cell gets
an ID based on rounded lat/lng bins. Count points per cell to produce a density
surface, then do ML on top of that.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_RES_TO_PRECISION: Dict[int, float] = {
    3: 1.0,
    4: 0.5,
    5: 0.2,
    6: 0.1,
    7: 0.05,
    8: 0.025,
    9: 0.01,
}


def _res_step(resolution: int) -> float:
    return _RES_TO_PRECISION.get(resolution, 0.2)


def _snap(value: float, step: float) -> float:
    """Snap a coordinate to the nearest grid step."""
    return round(round(float(value) / step) * step, 10)


def lat_lng_to_h3(lat: float, lng: float, resolution: int = 8) -> str:
    """Convert a lat/lng coordinate to a grid cell ID string."""
    step = _res_step(resolution)
    snapped_lat = _snap(lat, step)
    snapped_lng = _snap(lng, step)
    return f"{snapped_lat:.6f}_{snapped_lng:.6f}"


def h3_to_center(h3_index: str) -> Tuple[float, float]:
    """Return the (lat, lng) center of a grid cell from its ID."""
    parts = h3_index.split("_", 1)
    return float(parts[0]), float(parts[1])


def h3_to_boundary_coords(h3_index: str, resolution: int = 5) -> List[Tuple[float, float]]:
    """Return the 4-corner polygon boundary of a grid cell for mapping."""
    lat, lng = h3_to_center(h3_index)
    step = _res_step(resolution)
    half = step / 2
    return [
        (lat - half, lng - half),
        (lat - half, lng + half),
        (lat + half, lng + half),
        (lat + half, lng - half),
        (lat - half, lng - half),
    ]


def compute_density_grid(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lng_col: str = "longitude",
    resolution: int = 8,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """Convert point locations into a grid density DataFrame."""
    if lat_col not in df.columns or lng_col not in df.columns:
        raise ValueError(f"Missing latitude/longitude columns: {lat_col}, {lng_col}")

    work = df.dropna(subset=[lat_col, lng_col]).copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lng_col] = pd.to_numeric(work[lng_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lng_col])

    work["h3_index"] = work.apply(
        lambda row: lat_lng_to_h3(row[lat_col], row[lng_col], resolution), axis=1
    )

    grouped = work.groupby("h3_index").agg(
        count=pd.NamedAgg(column=lat_col, aggfunc="count")
    )

    if value_col and value_col in work.columns:
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
        grouped["value_sum"] = work.groupby("h3_index")[value_col].sum()

    grouped = grouped.reset_index()
    grouped[["lat", "lng"]] = grouped["h3_index"].apply(
        lambda h: pd.Series(h3_to_center(h))
    )
    return grouped


def get_k_ring(h3_index: str, k: int = 1, resolution: int = 5) -> List[str]:
    """Get all rectangular-grid cells within k steps of a given cell."""
    lat, lng = h3_to_center(h3_index)
    step = _res_step(resolution)
    cells: List[str] = []
    for dlat in range(-k, k + 1):
        for dlng in range(-k, k + 1):
            nb_lat = _snap(lat + dlat * step, step)
            nb_lng = _snap(lng + dlng * step, step)
            cells.append(f"{nb_lat:.6f}_{nb_lng:.6f}")
    return cells


def compute_neighbor_density(density_df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """For each grid cell, compute average density across its k-ring neighbors."""
    density_map = dict(zip(density_df["h3_index"], density_df["count"]))
    neighbor_avg = []
    for h3_idx in density_df["h3_index"]:
        neighbors = get_k_ring(h3_idx, k)
        neighbor_counts = [density_map.get(n, 0) for n in neighbors]
        neighbor_avg.append(float(np.mean(neighbor_counts)))

    out = density_df.copy()
    out["neighbor_avg_density"] = neighbor_avg
    return out


def identify_opportunity_zones(
    density_df: pd.DataFrame,
    low_branch_threshold: Optional[float] = None,
    high_neighbor_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Find cells with low local count but high surrounding density."""
    df = density_df.copy()
    if "neighbor_avg_density" not in df.columns:
        df = compute_neighbor_density(df, k=2)

    if low_branch_threshold is None:
        low_branch_threshold = df["count"].quantile(0.25)
    if high_neighbor_threshold is None:
        high_neighbor_threshold = df["neighbor_avg_density"].quantile(0.75)

    df["opportunity_score"] = df["neighbor_avg_density"] / (df["count"] + 1)
    opportunities = df[
        (df["count"] <= low_branch_threshold)
        & (df["neighbor_avg_density"] >= high_neighbor_threshold)
    ].copy()
    return opportunities.sort_values("opportunity_score", ascending=False)


def build_h3_feature_matrix(
    density_df: pd.DataFrame,
    k_rings: List[int] = [1, 2, 3],
) -> pd.DataFrame:
    """Build an ML feature matrix from the grid density table."""
    if "count" not in density_df.columns:
        raise ValueError("density_df must include a 'count' column")

    density_map = dict(zip(density_df["h3_index"], density_df["count"]))
    features = density_df.copy()

    for k in k_rings:
        col = f"neighbor_avg_k{k}"
        avgs = []
        for h3_idx in features["h3_index"]:
            neighbors = get_k_ring(h3_idx, k)
            neighbor_counts = [density_map.get(n, 0) for n in neighbors]
            avgs.append(float(np.mean(neighbor_counts)))
        features[col] = avgs

    if "neighbor_avg_k1" in features.columns:
        features["density_ratio_k1"] = features["count"] / (features["neighbor_avg_k1"] + 1)
    if "neighbor_avg_k2" in features.columns:
        features["density_ratio_k2"] = features["count"] / (features["neighbor_avg_k2"] + 1)
        low_thresh = features["count"].quantile(0.25)
        features["is_isolated"] = (
            (features["count"] <= low_thresh)
            & (features["neighbor_avg_k2"] <= low_thresh)
        ).astype(int)

    return features


def build_density_map(
    density_df: pd.DataFrame,
    center: Tuple[float, float] = (39.5, -98.35),
    zoom: int = 4,
    column: str = "count",
    title: str = "Branch Density per H3 Cell",
):
    """Build a Folium polygon map of grid density."""
    import folium

    fmap = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    max_val = density_df[column].max() or 1
    min_val = density_df[column].min()

    def get_color(val):
        ratio = (val - min_val) / (max_val - min_val + 1e-9)
        r = int(31 + (0 - 31) * ratio)
        g = int(119 + (68 - 119) * ratio)
        b = int(180 + (153 - 180) * ratio)
        alpha = 0.3 + 0.6 * ratio
        return f"#{r:02x}{g:02x}{b:02x}", alpha

    for _, row in density_df.iterrows():
        boundary = h3_to_boundary_coords(row["h3_index"])
        polygon_coords = [[p[0], p[1]] for p in boundary]
        color, alpha = get_color(row[column])
        lat_c = row.get("lat", row.get("center_lat", 0))
        lng_c = row.get("lng", row.get("center_lng", 0))
        folium.Polygon(
            locations=polygon_coords,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=alpha,
            weight=0.5,
            tooltip=folium.Tooltip(
                f"Cell: {row['h3_index']}<br>"
                f"{column.replace('_', ' ').title()}: {row[column]:.1f}<br>"
                f"Center: ({lat_c:.4f}, {lng_c:.4f})"
            ),
        ).add_to(fmap)

    return fmap
