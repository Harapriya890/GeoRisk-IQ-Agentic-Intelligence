"""
app.py - Streamlit application
From Trucks to Branches: Geospatial AI for Location Intelligence
Interactive demo of H3-based bank branch opportunity scoring using FDIC data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Page config
st.set_page_config(
    page_title="Geospatial AI: Branch Location Intelligence",
    page_icon="📍",
    layout="wide",
)


def stop_app(exit_code: int = 0) -> None:
    """Stop Streamlit, and also stop cleanly when run as plain Python."""
    st.stop()
    raise SystemExit(exit_code)


def build_top_opportunity_map(top_ops: pd.DataFrame, resolution: int):
    from src.h3_utils import h3_to_boundary_coords

    map_df = top_ops.dropna(subset=["center_lat", "center_lng"]).copy()
    map_df["_rank"] = range(1, len(map_df) + 1)
    map_df["_asset_label"] = map_df.get("asset_total", pd.Series(0, index=map_df.index)).map(
        lambda v: f"${float(v):,.0f}K"
    )

    fig = go.Figure()

    for _, row in map_df.iterrows():
        boundary = h3_to_boundary_coords(row["h3_index"], resolution=resolution)
        lats = [lat for lat, _ in boundary]
        lngs = [lng for _, lng in boundary]
        score = float(row["ml_score_pct"])
        alpha = 0.18 + (score / 100.0 * 0.32)
        hover = (
            f"Rank: {int(row['_rank'])}<br>"
            f"H3 Cell: {row['h3_index']}<br>"
            f"Opportunity Score: {score:.1f}<br>"
            f"Current Branches: {int(row['branch_count'])}<br>"
            f"Regional Assets: {row['_asset_label']}"
        )
        fig.add_trace(
            go.Scatter(
                x=lngs,
                y=lats,
                mode="lines",
                fill="toself",
                fillcolor=f"rgba(220, 38, 38, {alpha:.2f})",
                line=dict(color="rgba(127, 29, 29, 0.95)", width=2),
                hoverinfo="text",
                text=[hover] * len(lats),
                showlegend=False,
            )
        )

    marker_sizes = 14 + (map_df["ml_score_pct"] / 100.0 * 18)
    fig.add_trace(
        go.Scatter(
            x=map_df["center_lng"],
            y=map_df["center_lat"],
            mode="markers",
            marker=dict(
                size=marker_sizes + 6,
                color="white",
                opacity=0.95,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=map_df["center_lng"],
            y=map_df["center_lat"],
            mode="markers+text",
            text=map_df["_rank"].astype(str),
            textposition="top center",
            customdata=map_df[["h3_index", "branch_count", "ml_score_pct", "_asset_label"]],
            marker=dict(
                size=marker_sizes,
                color=map_df["ml_score_pct"],
                colorscale="Reds",
                cmin=0,
                cmax=100,
                opacity=0.95,
                colorbar=dict(title="Score"),
                line=dict(color="white", width=1),
            ),
            hovertemplate=(
                "Rank: %{text}<br>"
                "H3 Cell: %{customdata[0]}<br>"
                "Opportunity Score: %{customdata[2]:.1f}<br>"
                "Current Branches: %{customdata[1]}<br>"
                "Regional Assets: %{customdata[3]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Top Opportunity Zones",
        height=620,
        margin=dict(l=0, r=0, t=48, b=0),
        xaxis=dict(
            title="Longitude",
            range=[-126, -66],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.35)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            range=[24, 50],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.35)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="rgb(245, 248, 250)",
        paper_bgcolor="white",
    )
    return fig


def build_top_opportunity_deck(top_ops: pd.DataFrame, resolution: int):
    import pydeck as pdk
    from src.h3_utils import h3_to_boundary_coords

    map_df = top_ops.dropna(subset=["center_lat", "center_lng"]).copy()
    map_df["rank"] = range(1, len(map_df) + 1)
    map_df["rank_label"] = map_df["rank"].astype(str)
    map_df["asset_label"] = map_df.get("asset_total", pd.Series(0, index=map_df.index)).map(
        lambda v: f"${float(v):,.0f}K"
    )
    map_df["polygon"] = map_df["h3_index"].map(
        lambda h: [[lng, lat] for lat, lng in h3_to_boundary_coords(h, resolution=resolution)]
    )
    map_df["fill_r"] = 220
    map_df["fill_g"] = 38
    map_df["fill_b"] = 38
    map_df["fill_a"] = (110 + map_df["ml_score_pct"].clip(0, 100) * 1.1).astype(int)
    map_df["line_r"] = 127
    map_df["line_g"] = 29
    map_df["line_b"] = 29
    map_df["line_a"] = 240
    map_df["marker_radius"] = 14000 + map_df["ml_score_pct"].clip(0, 100) * 220

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=map_df,
        get_polygon="polygon",
        get_fill_color=["fill_r", "fill_g", "fill_b", "fill_a"],
        get_line_color=["line_r", "line_g", "line_b", "line_a"],
        line_width_min_pixels=2,
        filled=True,
        stroked=True,
        pickable=True,
        auto_highlight=True,
    )
    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["center_lng", "center_lat"],
        get_radius="marker_radius",
        get_fill_color=[185, 28, 28, 220],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        pickable=True,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=map_df,
        get_position=["center_lng", "center_lat"],
        get_text="rank_label",
        get_size=16,
        get_color=[255, 255, 255, 255],
        get_angle=0,
        get_text_anchor='"middle"',
        get_alignment_baseline='"center"',
        pickable=False,
    )
    view_state = pdk.ViewState(
        latitude=float(map_df["center_lat"].mean()),
        longitude=float(map_df["center_lng"].mean()),
        zoom=3,
        pitch=35,
        bearing=0,
    )
    return pdk.Deck(
        layers=[polygon_layer, marker_layer, text_layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>Rank {rank}</b><br/>"
                "H3 Cell: {h3_index}<br/>"
                "Opportunity Score: {ml_score_pct}<br/>"
                "Current Branches: {branch_count}<br/>"
                "Regional Assets: {asset_label}"
            ),
            "style": {"backgroundColor": "white", "color": "#111827"},
        },
        map_style="mapbox://styles/mapbox/light-v9",
    )


# Lazy imports (heavy)
@st.cache_resource
def get_pipeline():
    from src.ml_scorer import run_pipeline
    return run_pipeline


@st.cache_data(show_spinner="Loading FDIC data & building H3 grid...")
def load_results(json_path: str, resolution: int, top_n: int):
    run_pipeline = get_pipeline()
    return run_pipeline(json_path=json_path, resolution=resolution, top_n=top_n)


@st.cache_data(show_spinner="Running HRSA shortage prediction pipeline...")
def load_hrsa_results(resolution: int):
    from src.ml_scorer import run_hrsa_pipeline
    return run_hrsa_pipeline(resolution=resolution)


@st.cache_data(show_spinner="Running IEEE validation pipeline (takes ~60s)...")
def load_ieee_results():
    from src.ml_scorer import run_ieee_pipeline
    return run_ieee_pipeline()


# Sidebar
st.sidebar.title("Configuration")

domain = st.sidebar.radio(
    "Domain",
    ["Banking (FDIC)", "Healthcare (HRSA)"],
    help="Banking: FDIC branch opportunity scoring. Healthcare: HRSA shortage prediction.",
)

default_path = os.path.join(os.path.dirname(__file__), "Geospatiallocation.txt")
uploaded = None

if domain == "Banking (FDIC)":
    uploaded = st.sidebar.file_uploader(
        "Upload FDIC JSON file (optional)",
        type=["txt", "json"],
        help="Leave blank to use the default Geospatiallocation.txt",
    )

resolution = st.sidebar.slider(
    "H3 Resolution",
    min_value=3,
    max_value=7,
    value=5,
    help="Lower = larger hexes (coarser). Higher = smaller hexes (finer). Resolution 5 ≈ metro-level.",
)

top_n = st.sidebar.slider("Top N Opportunity Zones", min_value=5, max_value=50, value=20)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data source**: [FDIC BankFind Suite](https://banks.data.fdic.gov) \n"
    "**H3 library**: [Uber H3](https://h3geo.org) \n"
    "**Author**: Geospatial AI Demo"
)


# Handle uploaded file (Banking only)
data_path = default_path

if domain == "Banking (FDIC)" and uploaded:
    tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_upload.txt")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())
    data_path = tmp_path


# ============================================================
# HEALTHCARE DOMAIN (HRSA) - st.stop() skips banking below
# ============================================================
if domain == "Healthcare (HRSA)":
    st.title("GeoRisk-IQ: Healthcare Shortage Prediction (HRSA)")
    st.markdown(
        "> **Cross-domain generalisation.** "
        "The same H3 spatial pipeline applied to HRSA Health Professional Shortage Areas - "
        "proving the framework is not banking-specific."
    )

    try:
        with st.spinner("Running HRSA shortage prediction pipeline..."):
            hrsa = load_hrsa_results(resolution)
    except Exception as _e:
        st.error(f"HRSA pipeline error: {_e}")
        stop_app(1)

    hfdf = hrsa["feature_df"]
    hcomp = hrsa["comparison"]
    hy = hrsa["y"]

    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Shortage Cells Loaded", f"{len(hfdf):,}")
    hc2.metric("High-Priority Cells", f"{int(hy.sum()):,} ({hy.mean() * 100:.0f}%)")
    hc3.metric("Hybrid AUC", hcomp.get("hybrid", {}).get("auc", "N/A"))
    hc4.metric("H3 Resolution", resolution)

    st.markdown("---")

    htab1, htab2, htab3, htab4 = st.tabs(
        ["Model Comparison", "Shortage Map", "Feature Importance", "Raw Data"]
    )

    with htab1:
        st.subheader("Four-Model AUC / F1 Comparison")
        st.markdown("Hybrid model (spatial + domain metrics) vs baselines - same pipeline as banking.")
        comp_rows = [{"Model": k, "AUC": v["auc"], "F1": v["f1"]} for k, v in hcomp.items()]
        comp_df = pd.DataFrame(comp_rows).sort_values("AUC", ascending=False)
        st.dataframe(comp_df, use_container_width=True)
        fig_comp = px.bar(
            comp_df.sort_values("AUC"),
            x="AUC",
            y="Model",
            orientation="h",
            color="AUC",
            color_continuous_scale="Blues",
            title="ROC-AUC by Model (HRSA Healthcare Domain)",
            text="AUC",
        )
        fig_comp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_comp.update_layout(height=320, margin=dict(l=200, r=80, t=50, b=40))
        st.plotly_chart(fig_comp, use_container_width=True)

    with htab2:
        st.subheader("HRSA Shortage Cell Map (pydeck 3D)")
        st.markdown(
            "**Red columns** = high-priority shortage cells (HPSA score >= 14). "
            "Height = number of shortage designations in that H3 cell."
        )
        lat_c = "center_lat" if "center_lat" in hfdf.columns else "lat"
        lng_c = "center_lng" if "center_lng" in hfdf.columns else "lng"
        if lat_c in hfdf.columns:
            try:
                import pydeck as pdk

                mdf = hfdf[[lat_c, lng_c, "shortage_count", "high_shortage"]].copy()
                mdf = mdf.rename(columns={lat_c: "lat", lng_c: "lng"}).dropna(subset=["lat", "lng"])
                mdf["r"] = mdf["high_shortage"].apply(lambda x: 220 if x else 30)
                mdf["g"] = mdf["high_shortage"].apply(lambda x: 30 if x else 140)
                mdf["b"] = mdf["high_shortage"].apply(lambda x: 30 if x else 220)

                hlayer = pdk.Layer(
                    "ColumnLayer",
                    data=mdf,
                    get_position=["lng", "lat"],
                    get_elevation="shortage_count",
                    elevation_scale=30000,
                    radius=25000,
                    get_fill_color=["r", "g", "b", 200],
                    pickable=True,
                    auto_highlight=True,
                )
                hview = pdk.ViewState(latitude=38.5, longitude=-96.5, zoom=3, pitch=45, bearing=0)
                hdeck = pdk.Deck(
                    layers=[hlayer],
                    initial_view_state=hview,
                    tooltip={"text": "Shortage areas: {shortage_count}\nHigh-priority: {high_shortage}"},
                    map_style="mapbox://styles/mapbox/light-v9",
                )
                st.pydeck_chart(hdeck, use_container_width=True)
            except Exception as _pe:
                st.warning(f"pydeck unavailable ({_pe}) - using scatter map.")
                mdf2 = hfdf.dropna(subset=[lat_c, lng_c])
                st.plotly_chart(
                    px.scatter_map(
                        mdf2,
                        lat=lat_c,
                        lon=lng_c,
                        color=mdf2["high_shortage"].astype(str),
                        color_discrete_map={"0": "steelblue", "1": "crimson"},
                        zoom=3,
                        map_style="carto-positron",
                        title="HRSA Shortage Cells (red = high-priority)",
                    ),
                    use_container_width=True,
                )
        else:
            st.info("Coordinates not available.")

    with htab3:
        st.subheader("Feature Importance (Hybrid GBM)")
        if "feature_importance" in hrsa and hrsa["feature_importance"] is not None:
            fi = hrsa["feature_importance"].sort_values("importance", ascending=True)
            st.plotly_chart(
                px.bar(
                    fi,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Reds",
                    title="Feature Importance (HRSA Shortage Prediction)",
                    height=420,
                ),
                use_container_width=True,
            )
        else:
            st.info("Feature importance not available.")

    with htab4:
        st.subheader("Raw HRSA Feature Grid")
        st.dataframe(hfdf.head(500), use_container_width=True)
        csv_h = hfdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download HRSA feature grid (CSV)",
            data=csv_h,
            file_name="hrsa_shortage_grid.csv",
            mime="text/csv",
        )

    stop_app()  # skip banking section below


# ============================================================
# BANKING DOMAIN (FDIC)
# ============================================================
st.title("From Trucks to Branches: Geospatial AI for Location Intelligence")
st.markdown(
    """
> **Same technology. Different domain.**
> H3 hexagonal indexing was pioneered in logistics (Uber, Volvo) for driver/truck density analysis.
> Here we apply the same pipeline to optimize **bank branch network expansion** using FDIC data.
"""
)

# Run pipeline
try:
    with st.spinner("Running ML pipeline..."):
        results = load_results(data_path, resolution, top_n)
except FileNotFoundError:
    st.error(
        "FDIC data file not found. Upload a file via the sidebar or ensure "
        "`Geospatiallocation.txt` is in the project root."
    )
    stop_app(1)
except Exception as e:
    st.error(f"Pipeline error: {e}")
    stop_app(1)


density_df = results["density_df"]
scored_df = results["scored_df"]
top_ops = results["top_opportunities"]
metrics = results["metrics"]

# KPI row
col1, col2, col3, col4 = st.columns(4)
total_institutions = int(density_df["branch_count"].sum()) if "branch_count" in density_df.columns else 0
col1.metric("Institutions Loaded", f"{total_institutions:,}")
col2.metric("H3 Hexes (Resolution)", f"{len(density_df):,} (res {resolution})")
col3.metric("Model R²", metrics.get("r2", "N/A"))
col4.metric("Opportunity Zones", top_n)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Top Opportunities",
        "Hex Density Map",
        "Model Insights",
        "Raw Data",
        "IEEE Results",
        "Agentic AI",
        "Veritas Governance",
        "Cross-Domain",
        "Advanced AI",
    ]
)

# Tab 1: Top Opportunities
with tab1:
    st.subheader(f"Top {top_n} Opportunity Zones")
    st.markdown(
        "Hexes ranked by ML score → **low current supply** in an area **surrounded by high demand**."
    )

    display_cols = ["h3_index", "ml_score_pct", "branch_count"]
    if "center_lat" in top_ops.columns:
        display_cols += ["center_lat", "center_lng"]
    if "asset_total" in top_ops.columns:
        display_cols.append("asset_total")

    display_top = top_ops[[c for c in display_cols if c in top_ops.columns]].copy()
    display_top.columns = [
        c.replace("_", " ")
        .replace("ml score pct", "Opportunity Score (0-100)")
        .replace("branch count", "Current Branches")
        .replace("center lat", "Lat")
        .replace("center lng", "Lng")
        .replace("asset total", "Regional Assets ($K)")
        .replace("h3 index", "H3 Hex ID")
        for c in display_top.columns
    ]

    st.dataframe(display_top, use_container_width=True)

    # Map of top opportunity zones
    if "center_lat" in top_ops.columns and "center_lng" in top_ops.columns:
        st.subheader("Map of Top Opportunity Zones")
        try:
            deck_map = build_top_opportunity_deck(top_ops, resolution)
            st.pydeck_chart(deck_map, use_container_width=True)
        except Exception as _map_err:
            st.warning(f"Background map unavailable ({_map_err}) - showing coordinate map.")
            fig_map = build_top_opportunity_map(top_ops, resolution)
            st.plotly_chart(fig_map, use_container_width=True)


# Tab 2: Hex Density Map
with tab2:
    st.subheader("Branch Density → H3-Style Hexagonal Map (pydeck)")
    st.markdown(
        "Each column represents one H3 cell. **Height** = branch count; "
        "**colour** = opportunity score (blue = low, orange/red = high)."
    )

    lat_col = "center_lat" if "center_lat" in scored_df.columns else "lat"
    lng_col = "center_lng" if "center_lng" in scored_df.columns else "lng"
    bc_col = "branch_count" if "branch_count" in scored_df.columns else "count"

    if lat_col in scored_df.columns:
        try:
            import pydeck as pdk

            map_df = scored_df[[lat_col, lng_col, bc_col, "ml_score_pct"]].copy()
            map_df = map_df.rename(columns={lat_col: "lat", lng_col: "lng", bc_col: "branch_count"})
            map_df = map_df.dropna(subset=["lat", "lng"])

            # Colour: interpolate blue (low score) → red (high score)
            map_df["r"] = (map_df["ml_score_pct"] * 2.55).clip(0, 255).astype(int)
            map_df["g"] = 50
            map_df["b"] = (255 - map_df["ml_score_pct"] * 2.55).clip(0, 255).astype(int)

            layer = pdk.Layer(
                "ColumnLayer",
                data=map_df,
                get_position=["lng", "lat"],
                get_elevation="branch_count",
                elevation_scale=8000,
                radius=25000,
                get_fill_color=["r", "g", "b", 200],
                pickable=True,
                auto_highlight=True,
            )
            view_state = pdk.ViewState(
                latitude=38.5,
                longitude=-96.5,
                zoom=3,
                pitch=45,
                bearing=0,
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Branches: {branch_count}\nOpportunity: {ml_score_pct}"},
                map_style="mapbox://styles/mapbox/light-v9",
            )
            st.pydeck_chart(deck, use_container_width=True)
        except Exception as _pdk_err:
            st.warning(f"pydeck unavailable ({_pdk_err}) → falling back to density map.")
            fig_density = px.density_map(
                scored_df,
                lat=lat_col,
                lon=lng_col,
                z=bc_col,
                radius=15,
                zoom=3,
                map_style="carto-positron",
                color_continuous_scale="Blues",
                title="Branch Density per H3 Cell",
            )
            st.plotly_chart(fig_density, use_container_width=True)
    else:
        st.info("Center coordinates not available → re-run with resolution 6.")


# Tab 3: Model Insights
with tab3:
    st.subheader("Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("Mean Absolute Error (MAE)", metrics.get("mae", "N/A"))
    m2.metric("R² Score", metrics.get("r2", "N/A"))

    st.markdown("### Feature Importance")
    try:
        model = results["model"]
        feature_cols = [
            c
            for c in results["scored_df"].columns
            if c
            not in {
                "h3_index",
                "asset_total",
                "center_lat",
                "center_lng",
                "lat",
                "lng",
                "ml_score",
                "ml_score_pct",
                "branch_count",
            }
        ]
        importances = model.feature_importances_
        n = min(len(importances), len(feature_cols))
        fi_df = pd.DataFrame(
            {"Feature": feature_cols[:n], "Importance": importances[:n]}
        ).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="GradientBoosting Feature Importances",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

    st.markdown("### Branch Count Distribution")
    bc_col = "branch_count" if "branch_count" in scored_df.columns else "count"
    fig_hist = px.histogram(
        scored_df,
        x=bc_col,
        nbins=50,
        title="Distribution of Institutions per H3 Cell",
        labels={bc_col: "Institutions per Hex"},
        color_discrete_sequence=["#1f77b4"],
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# Tab 4: Raw Data
with tab4:
    st.subheader("Scored Hex Grid (all hexes)")
    st.dataframe(scored_df.head(500), use_container_width=True)
    csv = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full scored grid (CSV)",
        data=csv,
        file_name="scored_hex_grid.csv",
        mime="text/csv",
    )


# Tab 5: IEEE Results
with tab5:
    st.subheader("IEEE Validation Results")
    st.markdown(
        "Full IEEE-quality pipeline: 10-fold CV, temporal holdout, bootstrap CI, "
        "ROC & PR curves, ablation study."
    )

    if st.button("Run IEEE Pipeline (takes ~60s)", type="primary"):
        try:
            with st.spinner("Running IEEE pipeline..."):
                ieee = load_ieee_results()
                st.session_state["ieee"] = ieee
        except FileNotFoundError as e:
            st.error(f"Missing IEEE data file: {e}")
            stop_app(1)
        except Exception as e:
            st.error(f"IEEE pipeline error: {e}")
            stop_app(1)

    ieee = st.session_state.get("ieee", None)

    if ieee is None:
        st.info("Click the button above to run the full IEEE validation pipeline.")
    else:
        comp = ieee.get("comparison", {})
        m = ieee.get("metrics", {})
        bci = ieee.get("bootstrap_ci", {})
        ablat = ieee.get("ablation", {})
        fi = ieee.get("feature_importance")

        # KPIs
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Hybrid AUC (CV)", comp.get("hybrid", {}).get("auc", "N/A"))
        ic2.metric("Temporal AUC", m.get("roc_auc", "N/A"))
        ic3.metric("Temporal F1", m.get("f1", "N/A"))
        h_ci = bci.get("hybrid")
        ic4.metric("Bootstrap 95% CI", f"[{h_ci[0]}, {h_ci[1]}]" if h_ci else "N/A")

        pp = ieee.get("permutation_p", {})
        p_val = pp.get("hybrid_vs_spatial")
        sig_label = (
            "p < 0.05 (significant)"
            if p_val and p_val < 0.05
            else (f"p = {p_val} (marginal)" if p_val else "N/A")
        )
        st.info(
            f"Permutation test (hybrid vs spatial, n=10,000): **{sig_label}** | "
            f"Wilcoxon is supplementary only."
        )

        st.markdown("---")
        it1, it2, it3, it4 = st.tabs(
            ["Model Comparison", "ROC / PR Curves", "Feature Importance", "Ablation Study"]
        )

        with it1:
            st.subheader("Four-Way Model Comparison (10-fold CV)")
            rows = [{"Model": k, "AUC": v["auc"], "F1": v["f1"]} for k, v in comp.items()]
            cdf = pd.DataFrame(rows).sort_values("AUC", ascending=False)
            st.dataframe(cdf, use_container_width=True)
            fig_c = px.bar(
                cdf.sort_values("AUC"),
                x="AUC",
                y="Model",
                orientation="h",
                color="AUC",
                color_continuous_scale="Purples",
                title="ROC-AUC by Model (Banking Domain, 10-fold CV)",
                text="AUC",
                height=320,
            )
            fig_c.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_c.update_layout(margin=dict(l=200, r=80, t=50, b=40))
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown("**Temporal holdout test (pre-2010 train / 2010-2020 test):**")
            st.json(
                {
                    "Precision": m.get("precision"),
                    "Recall": m.get("recall"),
                    "F1": m.get("f1"),
                    "ROC-AUC": m.get("roc_auc"),
                }
            )

        with it2:
            st.subheader("ROC Curves (10-fold CV, all models)")
            cv_probas = ieee.get("cv_probas", {})
            if cv_probas:
                from sklearn.metrics import (
                    roc_curve,
                    precision_recall_curve,
                    average_precision_score,
                    roc_auc_score,
                )

                COLORS = {
                    "logistic_regression": "#636EFA",
                    "spatial_only": "#EF553B",
                    "financial_only": "#00CC96",
                    "hybrid": "#AB63FA",
                }
                LABELS = {
                    "logistic_regression": "Logistic Regression",
                    "spatial_only": "Spatial-only GBM",
                    "financial_only": "Financial-only GBM",
                    "hybrid": "Hybrid GBM (ours)",
                }
                fig_roc = go.Figure()
                fig_roc.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=1,
                    line=dict(dash="dash", color="gray", width=1),
                )
                fig_pr = go.Figure()

                for lbl, (yt, yp) in cv_probas.items():
                    fpr, tpr, _ = roc_curve(yt, yp)
                    auc_v = roc_auc_score(yt, yp)
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"{LABELS.get(lbl, lbl)} (AUC={auc_v:.3f})",
                            line=dict(color=COLORS.get(lbl, "gray"), width=3 if lbl == "hybrid" else 1.5),
                        )
                    )
                    prec, rec, _ = precision_recall_curve(yt, yp)
                    ap = average_precision_score(yt, yp)
                    fig_pr.add_trace(
                        go.Scatter(
                            x=rec,
                            y=prec,
                            mode="lines",
                            name=f"{LABELS.get(lbl, lbl)} (AP={ap:.3f})",
                            line=dict(color=COLORS.get(lbl, "gray"), width=3 if lbl == "hybrid" else 1.5),
                        )
                    )

                fig_roc.update_layout(
                    title="ROC Curves",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400,
                )
                st.plotly_chart(fig_roc, use_container_width=True)

                st.subheader("Precision-Recall Curves (10-fold CV, all models)")
                fig_pr.update_layout(
                    title="Precision-Recall Curves",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=400,
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("CV probabilities not available.")

        with it3:
            st.subheader("Feature Importance (Hybrid GBM)")
            if fi is not None:
                imp_col = "mean_abs_shap" if "mean_abs_shap" in fi.columns else "importance"
                fi_s = fi.sort_values(imp_col, ascending=True)
                st.plotly_chart(
                    px.bar(
                        fi_s,
                        x=imp_col,
                        y="feature",
                        orientation="h",
                        color=imp_col,
                        color_continuous_scale="Viridis",
                        title="Feature Importance - Hybrid GBM",
                        height=420,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Feature importance not available.")

        with it4:
            st.subheader("Ablation Study (F1, 5-fold CV)")
            if ablat:
                baseline = ablat.get("full_model", 0)
                abl_rows = [
                    {"Configuration": k, "F1": v, "Delta_F1": round(v - baseline, 4)}
                    for k, v in ablat.items()
                ]
                abl_df = pd.DataFrame(abl_rows)
                st.dataframe(abl_df, use_container_width=True)
                fig_abl = px.bar(
                    abl_df.sort_values("F1"),
                    x="F1",
                    y="Configuration",
                    orientation="h",
                    color="Delta_F1",
                    color_continuous_scale="RdYlGn",
                    title="Ablation: F1 by feature group removed",
                    height=320,
                )
                st.plotly_chart(fig_abl, use_container_width=True)
            else:
                st.info("Ablation results not available.")


# Tab 6: Agentic AI
with tab6:
    st.subheader("Agentic AI Extension")
    st.markdown(
        "Optional architecture for adding CrewAI, LangGraph, LangChain, and "
        "Snowflake to the GeoRisk-IQ pipeline. The base dashboard runs without "
        "these dependencies."
    )

    try:
        from src.agentic_ai import (
            agentic_prompt_template,
            build_langgraph_workflow_outline,
            get_agentic_architecture,
            snowflake_table_plan,
        )
        from src.agentic_runtime import (
            get_crewai_blueprint,
            get_langchain_tools,
            run_langgraph_agentic_workflow,
            run_local_agentic_workflow,
            snowflake_health_check,
        )

        arch = get_agentic_architecture()
        deps = pd.DataFrame(arch["dependencies"])
        agents = pd.DataFrame(arch["agents"])
        workflow = pd.DataFrame(build_langgraph_workflow_outline())
        snowflake_tables = pd.DataFrame(snowflake_table_plan())

        st.markdown("### Optional Dependency Status")
        st.dataframe(deps, use_container_width=True)

        st.markdown("### Agent Roles")
        st.dataframe(agents, use_container_width=True)

        st.markdown("### LangGraph Workflow Outline")
        st.dataframe(workflow, use_container_width=True)

        st.markdown("### Snowflake Production Table Plan")
        st.dataframe(snowflake_tables, use_container_width=True)

        st.markdown("### Narrative Agent Prompt Template")
        st.code(agentic_prompt_template(), language="text")

        st.markdown("### Testable Agentic Runtime")
        c1, c2, c3 = st.columns(3)
        sample_rows = c1.number_input("Sample rows", min_value=100, max_value=5000, value=1000, step=100)
        run_mode = c2.selectbox("Runtime", ["Local deterministic", "LangGraph if installed"])
        execute_full_model = c3.checkbox("Run full ML scorer", value=False)

        if st.button("Run Agentic Workflow", type="primary"):
            with st.spinner("Running agentic workflow..."):
                kwargs = {
                    "data_path": data_path,
                    "resolution": resolution,
                    "top_n": top_n,
                    "sample_rows": int(sample_rows),
                    "execute_model": bool(execute_full_model),
                    "domain": "banking",
                }
                if run_mode == "LangGraph if installed":
                    runtime_result = run_langgraph_agentic_workflow(**kwargs)
                else:
                    runtime_result = run_local_agentic_workflow(**kwargs)
                st.session_state["agentic_runtime_result"] = runtime_result

        if "agentic_runtime_result" in st.session_state:
            st.json(st.session_state["agentic_runtime_result"])

        st.markdown("### LangChain Tool Runtime")
        tools = get_langchain_tools()
        st.write(
            {
                "langchain_tools_available": len(tools),
                "tool_names": [getattr(tool, "name", str(tool)) for tool in tools],
            }
        )

        st.markdown("### CrewAI Blueprint")
        st.json(get_crewai_blueprint())

        st.markdown("### Snowflake Health Check")
        st.json(snowflake_health_check())

        st.info(
            "To enable these optional integrations, install "
            "`pip install -r requirements-agentic.txt` and connect the workflow "
            "to your chosen LLM and Snowflake account."
        )
    except Exception as e:
        st.warning(f"Agentic AI architecture unavailable: {e}")


# Tab 7: Veritas Governance
with tab7:
    st.subheader("Veritas-Style Responsible AI Governance")

    try:
        from src.veritas_governance import (
            build_model_card,
            deployment_controls,
            governance_checklist,
            slice_audit_plan,
            veritas_summary,
        )

        st.markdown(veritas_summary())

        model_card = build_model_card()
        checklist = pd.DataFrame(governance_checklist())
        slices = pd.DataFrame(slice_audit_plan())
        controls = pd.DataFrame(deployment_controls())

        st.markdown("### Model Card")
        st.json(model_card)

        st.markdown("### Governance Checklist")
        st.dataframe(checklist, use_container_width=True)

        st.markdown("### Fairness and Geographic Slice Audit Plan")
        st.dataframe(slices, use_container_width=True)

        st.markdown("### Deployment Controls")
        st.dataframe(controls, use_container_width=True)

        st.info(
            "This tab provides local Veritas-style governance documentation. "
            "It does not indicate third-party certification or regulatory approval."
        )
    except Exception as e:
        st.warning(f"Veritas governance view unavailable: {e}")


# Tab 8: Cross-Domain Strategy
with tab8:
    st.subheader("Cross-Domain Spatial Intelligence")

    try:
        from src.cross_domain_use_cases import (
            cross_domain_summary,
            get_cross_domain_use_cases,
        )

        st.markdown(cross_domain_summary())
        use_cases = pd.DataFrame(get_cross_domain_use_cases())
        st.dataframe(use_cases, use_container_width=True)

        st.info(
            "Banking is implemented in this workspace. The healthcare path has "
            "been tested with downloaded HRSA data in another workspace and can "
            "be transferred here later. Volvo/fleet GPS and driver behavior "
            "tracking are prior-domain inspirations and future connector "
            "extensions until real telemetry or platform data is added."
        )
    except Exception as e:
        st.warning(f"Cross-domain view unavailable: {e}")


# Tab 9: Advanced AI + Responsible AI
with tab9:
    st.subheader("Advanced AI + Responsible AI Architecture")

    try:
        from src.advanced_responsible_ai import (
            advanced_responsible_ai_summary,
            get_advanced_ai_capabilities,
            get_responsible_ai_controls,
        )

        st.markdown(advanced_responsible_ai_summary())

        capabilities = pd.DataFrame(get_advanced_ai_capabilities())
        controls = pd.DataFrame(get_responsible_ai_controls())

        st.markdown("### Advanced AI Capabilities")
        st.dataframe(capabilities, use_container_width=True)

        st.markdown("### Responsible AI Controls")
        st.dataframe(controls, use_container_width=True)

        st.info(
            "The base app remains lightweight. Advanced agentic, RAG, Snowflake, "
            "privacy, and monitoring capabilities are documented as architecture "
            "layers unless explicitly implemented in the current pipeline."
        )
    except Exception as e:
        st.warning(f"Advanced AI view unavailable: {e}")
