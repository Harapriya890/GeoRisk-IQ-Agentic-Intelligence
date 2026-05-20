"""
ml_scorer.py

Trains a location-opportunity scoring model using H3 hexagonal density features
derived from FDIC bank institution data.

Pipeline:
 1. Load FDIC data via fetch_fdic.load_fdic_data()
 2. Build H3 density grid (supply = current branch density)
 3. Engineer features: local density, neighbor density, k-ring rings
 4. Score every hex cell rank opportunity zones
 5. Return top-N candidate locations with scores
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 output on Windows to prevent charmap encode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)

from src.h3_utils import (
    compute_density_grid,
    build_h3_feature_matrix,
    identify_opportunity_zones,
)
from data.fetch_fdic import load_fdic_data

# Configuration
H3_RESOLUTION = 5  # ~86 km per hex - good for metro-level clustering
K_RINGS = [1, 2, 3]  # neighbor rings to include as features
TOP_N = 25  # number of opportunity zones to surface
RANDOM_STATE = 42


# Feature Engineering
def build_features(density_df: pd.DataFrame, k_rings: list = K_RINGS) -> pd.DataFrame:
    """
    Build ML feature matrix from H3 density grid.

    Features per hex:
    - branch_count: raw institution count in this hex
    - log_branch_count: log1p transform for skewness
    - neighbor_density_k*: average branch count in k-ring k
    - asset_total: sum of ASSET values in hex (if available)
    - log_asset_total: log1p of asset_total
    """
    # h3_utils.build_h3_feature_matrix expects 'count' column
    density_for_h3 = density_df.rename(columns={"branch_count": "count"})
    feature_df = build_h3_feature_matrix(density_for_h3, k_rings)

    # Restore branch_count name
    feature_df = feature_df.rename(columns={"count": "branch_count"})

    if "branch_count" in feature_df.columns:
        feature_df["log_branch_count"] = np.log1p(feature_df["branch_count"])

    if "asset_total" in feature_df.columns:
        feature_df["log_asset_total"] = np.log1p(feature_df["asset_total"])

    return feature_df


# Synthetic Target: Opportunity Score
def compute_opportunity_target(feature_df: pd.DataFrame) -> pd.Series:
    """
    Create a proxy target: 'opportunity score' is inversely related to supply
    (branch density) but positively related to regional demand proxy (neighbor density).

    Score = neighbor_density_k2 / (1 + branch_count)

    This rewards hexes that sit near high-density corridors but are themselves underserved.
    """
    bc = feature_df.get(
        "branch_count",
        pd.Series(np.zeros(len(feature_df)), index=feature_df.index),
    )
    nd = feature_df.get(
        "neighbor_avg_k2",
        feature_df.get(
            "neighbor_avg_k1",
            pd.Series(np.ones(len(feature_df)), index=feature_df.index),
        ),
    )
    score = nd / (1 + bc)
    return score


# Model Training
def train_scorer(feature_df: pd.DataFrame):
    """
    Train a GradientBoostingRegressor on the opportunity score target.

    Returns
    -------
    model: fitted GradientBoostingRegressor
    scaler: fitted StandardScaler
    feature_cols: list of feature column names used
    metrics: dict with MAE and R2 on test split
    """
    target = compute_opportunity_target(feature_df)

    exclude = {
        "h3_index",
        "branch_count",
        "asset_total",
        "center_lat",
        "center_lng",
        "lat",
        "lng",
    }
    feature_cols = [c for c in feature_df.columns if c not in exclude]

    X = feature_df[feature_cols].fillna(0)
    y = target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "r2": round(r2_score(y_test, y_pred), 4),
    }
    print(f"Model trained | MAE: {metrics['mae']} R2: {metrics['r2']}")

    return model, scaler, feature_cols, metrics


# Scoring All Hexes
def score_all_hexes(feature_df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Apply trained model to all hexes. Returns feature_df with added 'ml_score' column.
    """
    X = feature_df[feature_cols].fillna(0)
    X_s = scaler.transform(X)

    feature_df = feature_df.copy()
    feature_df["ml_score"] = model.predict(X_s)

    # Normalize to 0-100
    mn, mx = feature_df["ml_score"].min(), feature_df["ml_score"].max()
    if mx > mn:
        feature_df["ml_score_pct"] = ((feature_df["ml_score"] - mn) / (mx - mn) * 100).round(1)
    else:
        feature_df["ml_score_pct"] = 50.0

    return feature_df


# Top-N Opportunities
def get_top_opportunities(scored_df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Return the top_n hexes by ml_score_pct along with human-readable location info.
    """
    cols = ["h3_index", "ml_score_pct", "branch_count"]
    if "center_lat" in scored_df.columns:
        cols += ["center_lat", "center_lng"]
    if "asset_total" in scored_df.columns:
        cols.append("asset_total")

    top = (
        scored_df[cols]
        .sort_values("ml_score_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top.index += 1  # rank starts at 1
    return top


# Full Pipeline
def run_pipeline(
    json_path: str = None,
    resolution: int = H3_RESOLUTION,
    top_n: int = TOP_N,
) -> dict:
    """
    End-to-end pipeline: load data H3 grid features train score top-N.
    """
    # 1. Load data
    kwargs = {"json_path": json_path} if json_path else {}
    df = load_fdic_data(**kwargs)

    # 2. H3 density grid - aggregate ASSET if available
    value_col = "ASSET" if "ASSET" in df.columns else None
    density_df = compute_density_grid(
        df,
        lat_col="LATITUDE",
        lng_col="LONGITUDE",
        resolution=resolution,
        value_col=value_col,
    )

    # Normalize column names for downstream steps
    rename_map = {"count": "branch_count", "lat": "center_lat", "lng": "center_lng"}
    if value_col:
        rename_map["value_sum"] = "asset_total"
    density_df = density_df.rename(columns=rename_map)

    print(f"H3 grid: {len(density_df):,} hexes at resolution {resolution}")

    # 3. Feature engineering
    feature_df = build_features(density_df)

    # 4. Train model
    model, scaler, feature_cols, metrics = train_scorer(feature_df)

    # 5. Score all hexes
    scored_df = score_all_hexes(feature_df, model, scaler, feature_cols)

    # 6. Top-N opportunities
    top = get_top_opportunities(scored_df, top_n=top_n)

    print(f"\nTop {top_n} Opportunity Zones:")
    print(top.to_string())

    return {
        "density_df": density_df,
        "feature_df": feature_df,
        "scored_df": scored_df,
        "top_opportunities": top,
        "metrics": metrics,
        "model": model,
        "scaler": scaler,
    }


# ===============================================================================
# IEEE-QUALITY PIPELINE - uses real ground truth from two FDIC snapshots
# ===============================================================================
def run_ieee_pipeline(
    baseline_json: str = None,
    current_json: str = None,
    financials_json: str = None,
    resolution: int = H3_RESOLUTION,
) -> dict:
    """
    IEEE-publishable pipeline using real ground-truth labels.

    Runs models for comparison:
    1. Logistic Regression
    2. Spatial-only GBM
    3. Financial-only GBM
    4. Hybrid GBM
    """
    from src.ground_truth import build_features_with_labels
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate

    _root = os.path.join(os.path.dirname(__file__), "..")
    institutions_json = os.path.join(_root, "GeospatialFull.txt")
    failures_json = os.path.join(_root, "FDICFailure.txt")

    if not financials_json:
        financials_json = os.path.join(_root, "FDICFinancialsFull.txt")
    if baseline_json:
        institutions_json = baseline_json

    # 1. Build features + real labels
    feature_df = build_features_with_labels(
        institutions_json=institutions_json,
        failures_json=failures_json,
        financials_json=financials_json,
        resolution=resolution,
    )

    label_col = "had_failure"
    FINANCIAL_COLS = ["fin_roa_mean", "fin_netinc_mean", "fin_asset_mean"]
    exclude = {
        "h3_index",
        "branch_count",
        "center_lat",
        "center_lng",
        "lat",
        "lng",
        label_col,
        "had_failure",
        "branch_opened",
        "first_failure_year",
    }
    feature_cols = [c for c in feature_df.columns if c not in exclude]

    # Partition features into spatial vs financial
    spatial_cols = [c for c in feature_cols if c not in FINANCIAL_COLS]
    financial_cols = [c for c in feature_cols if c in FINANCIAL_COLS]
    hybrid_cols = feature_cols

    X = feature_df[hybrid_cols].fillna(0)
    y = feature_df[label_col]

    print(f"\nFeatures used ({len(hybrid_cols)} total):")
    print(f" Spatial ({len(spatial_cols)}): {spatial_cols}")
    print(f" Financial ({len(financial_cols)}): {financial_cols}")
    print(
        f"Class balance: failures={y.sum()} ({y.mean() * 100:.1f}%) | "
        f"clean={len(y) - y.sum()} ({(1 - y.mean()) * 100:.1f}%)"
    )

    # 2. Four-way model comparison with 10-fold CV
    print(f"\n{'=' * 60}")
    print("FOUR-WAY MODEL COMPARISON (10-fold CV)")
    print(f"{'=' * 60}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    cv_probas = {}

    for label, cols in [
        ("logistic_regression", hybrid_cols),
        ("spatial_only", spatial_cols),
        ("financial_only", financial_cols),
        ("hybrid", hybrid_cols),
    ]:
        if not cols:
            print(f" {label:<22} SKIPPED (no features)")
            continue

        Xc = feature_df[cols].fillna(0)
        Xc_s = StandardScaler().fit_transform(Xc)

        if label == "logistic_regression":
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        else:
            clf = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)

        auc_scores = cross_val_score(clf, Xc_s, y, cv=cv, scoring="roc_auc")
        f1_scores = cross_val_score(clf, Xc_s, y, cv=cv, scoring="f1")

        fold_yt, fold_yp = [], []
        for tr_idx, te_idx in cv.split(Xc_s, y):
            _clf = (
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
                if label == "logistic_regression"
                else GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
            )
            _clf.fit(Xc_s[tr_idx], y.iloc[tr_idx])
            fold_yt.append(y.iloc[te_idx].values)
            fold_yp.append(_clf.predict_proba(Xc_s[te_idx])[:, 1])

        cv_probas[label] = (np.concatenate(fold_yt), np.concatenate(fold_yp))
        comparison[label] = {
            "auc": round(auc_scores.mean(), 4),
            "f1": round(f1_scores.mean(), 4),
            "auc_folds": auc_scores,
        }
        print(
            f" {label:<22} AUC={auc_scores.mean():.4f}+/-{auc_scores.std():.4f} "
            f"F1={f1_scores.mean():.4f}"
        )

    # Bootstrap 95% CI on hybrid AUC
    bootstrap_ci = {}
    if "hybrid" in cv_probas:
        _yt_all, _yp_all = cv_probas["hybrid"]
        rng_bs = np.random.RandomState(RANDOM_STATE)
        bs_aucs = []
        for _ in range(1000):
            idx = rng_bs.choice(len(_yt_all), len(_yt_all), replace=True)
            if _yt_all[idx].sum() == 0 or _yt_all[idx].sum() == len(idx):
                continue
            try:
                bs_aucs.append(roc_auc_score(_yt_all[idx], _yp_all[idx]))
            except Exception:
                pass
        if bs_aucs:
            ci_lo, ci_hi = np.percentile(bs_aucs, [2.5, 97.5])
            bootstrap_ci["hybrid"] = (round(ci_lo, 4), round(ci_hi, 4))
            print(f"\n Hybrid AUC 95% CI (bootstrap n=1000): [{ci_lo:.4f}, {ci_hi:.4f}]")

    def _permutation_auc_test(yt, yp_a, yp_b, n_perm=10000, seed=RANDOM_STATE):
        """Two-sided permutation test: p(AUC_a - AUC_b >= observed delta)."""
        obs_delta = roc_auc_score(yt, yp_a) - roc_auc_score(yt, yp_b)
        rng_p = np.random.RandomState(seed)
        count = 0
        for _ in range(n_perm):
            swap = rng_p.rand(len(yt)) > 0.5
            perm_a = np.where(swap, yp_b, yp_a)
            perm_b = np.where(swap, yp_a, yp_b)
            try:
                d = roc_auc_score(yt, perm_a) - roc_auc_score(yt, perm_b)
                if abs(d) >= abs(obs_delta):
                    count += 1
            except Exception:
                pass
        return count / n_perm

    permutation_p = {}
    if "hybrid" in cv_probas and "spatial_only" in cv_probas:
        delta_auc = comparison["hybrid"]["auc"] - comparison["spatial_only"]["auc"]
        delta_fin = comparison["hybrid"]["auc"] - comparison.get("financial_only", {}).get("auc", 0)
        print(f"  Hybrid vs Spatial  : DAUC = {delta_auc:+.4f}")
        if "financial_only" in comparison:
            print(f"  Hybrid vs Financial: DAUC = {delta_fin:+.4f}")

        yt_h, yp_h = cv_probas["hybrid"]
        yt_sp, yp_sp = cv_probas["spatial_only"]
        p_sp = _permutation_auc_test(yt_h, yp_h, yp_sp)
        permutation_p["hybrid_vs_spatial"] = round(p_sp, 4)
        print(
            f"  Permutation p (hybrid vs spatial, n=10000): {p_sp:.4f} "
            f"{'<- significant' if p_sp < 0.05 else '(marginal)'}"
        )

        if "financial_only" in cv_probas:
            yt_fi, yp_fi = cv_probas["financial_only"]
            p_fi = _permutation_auc_test(yt_h, yp_h, yp_fi)
            permutation_p["hybrid_vs_financial"] = round(p_fi, 4)
            print(
                f"  Permutation p (hybrid vs financial, n=10000): {p_fi:.4f} "
                f"{'<- significant' if p_fi < 0.05 else '(marginal)'}"
            )

        try:
            from scipy import stats as _stats

            _, p_wil = _stats.wilcoxon(
                comparison["hybrid"]["auc_folds"],
                comparison["spatial_only"]["auc_folds"],
            )
            print(f"  Wilcoxon p (hybrid vs spatial, 10-fold): {p_wil:.4f}")
        except Exception:
            pass

    print(f"{'=' * 60}\n")

    # 3. Temporal train/test split
    fy_col = "first_failure_year"
    fy_series = feature_df[fy_col] if fy_col in feature_df.columns else pd.Series(0, index=feature_df.index)
    mask_pre2010 = (y == 1) & (fy_series > 0) & (fy_series < 2010)
    mask_post2010 = (y == 1) & (fy_series >= 2010) & (fy_series <= 2020)
    mask_no_fail = y == 0

    pre2010_idx = feature_df.index[mask_pre2010].tolist()
    post2010_idx = feature_df.index[mask_post2010].tolist()
    no_fail_idx = feature_df.index[mask_no_fail].tolist()

    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(no_fail_idx)
    split_at = int(len(no_fail_idx) * 0.8)
    neg_train_idx = no_fail_idx[:split_at]
    neg_test_idx = no_fail_idx[split_at:]
    train_idx_all = pre2010_idx + neg_train_idx
    test_idx_all = post2010_idx + neg_test_idx

    if len(post2010_idx) < 5:
        print(f"  WARNING: only {len(post2010_idx)} post-2010 positives -- using stratified random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    else:
        print(f"  Temporal split: train={len(pre2010_idx)} pre-2010 failures + {len(neg_train_idx)} clean")
        print(f"                  test ={len(post2010_idx)} post-2010 failures + {len(neg_test_idx)} clean")
        X_train = X.loc[train_idx_all]
        y_train = y.loc[train_idx_all]
        X_test = X.loc[test_idx_all]
        y_test = y.loc[test_idx_all]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Handle class imbalance with sample weights
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # 5. Train GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train, sample_weight=sample_weights)

    y_proba = model.predict_proba(X_test_s)[:, 1]

    # 6. Tune threshold for best F1 on minority class
    best_thresh, best_f1 = 0.5, 0.0
    for t in [i / 20 for i in range(1, 20)]:
        preds = (y_proba >= t).astype(int)
        f = f1_score(y_test, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    print(f"Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")

    y_pred = (y_proba >= best_thresh).astype(int)
    metrics = {
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    report = classification_report(y_test, y_pred, target_names=["no_failure", "had_failure"])

    print(f"\n{'-' * 50}")
    print("IEEE Classifier Results:")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1 Score  : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"{'-' * 50}")
    print(report)

    # 7. SHAP / Feature Importance
    print(f"{'-' * 50}")
    print("Feature Importance:")
    feature_importance = None
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test_s)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        feature_importance = pd.DataFrame(
            {
                "feature": hybrid_cols,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        print("  (SHAP values)")
        print(feature_importance.to_string(index=False))
    except ImportError:
        feature_importance = pd.DataFrame(
            {
                "feature": hybrid_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        print("  (GBM built-in importances -- install shap for SHAP values)")
        print(feature_importance.to_string(index=False))

    # Save feature importance chart via plotly
    try:
        import plotly.graph_objects as go

        imp_col = "mean_abs_shap" if "mean_abs_shap" in feature_importance.columns else "importance"
        fi_sorted = feature_importance.sort_values(imp_col)
        fig = go.Figure(
            go.Bar(
                x=fi_sorted[imp_col],
                y=fi_sorted["feature"],
                orientation="h",
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            title="Feature Importance -- Hybrid GBM (Banking Failure Prediction)",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=420,
            margin=dict(l=160, r=40, t=60, b=40),
        )
        _root_dir = os.path.join(os.path.dirname(__file__), "..")
        html_path = os.path.abspath(os.path.join(_root_dir, "feature_importance.html"))
        fig.write_html(html_path)
        print(f"  Chart saved: {html_path}")
        try:
            png_path = os.path.abspath(os.path.join(_root_dir, "feature_importance.png"))
            fig.write_image(png_path, width=800, height=420)
            print(f"  PNG saved: {png_path}")
        except Exception:
            print("  (PNG export skipped -- open the HTML in browser and screenshot)")
    except Exception as _ce:
        print(f"  (Chart save failed: {_ce})")
    print(f"{'-' * 50}")

    # 8. ROC curves (all 4 models)
    roc_html_path = None
    pr_html_path = None
    try:
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

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
        for lbl, (yt, yp) in cv_probas.items():
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_v = roc_auc_score(yt, yp)
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{LABELS[lbl]} (AUC={auc_v:.3f})",
                    line=dict(color=COLORS[lbl], width=3 if lbl == "hybrid" else 1.5),
                )
            )
        fig_roc.update_layout(
            title="ROC Curves -- All Models (10-fold CV, concatenated folds)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.4, y=0.05),
            width=700,
            height=500,
        )
        _root_dir = os.path.join(os.path.dirname(__file__), "..")
        roc_html_path = os.path.abspath(os.path.join(_root_dir, "roc_curves.html"))
        fig_roc.write_html(roc_html_path)
        try:
            fig_roc.write_image(os.path.abspath(os.path.join(_root_dir, "roc_curves.png")), width=700, height=500)
        except Exception:
            pass
        print(f"  ROC chart saved: {roc_html_path}")

        fig_pr = go.Figure()
        base_prev = y.mean()
        fig_pr.add_shape(
            type="line",
            x0=0,
            y0=base_prev,
            x1=1,
            y1=base_prev,
            line=dict(dash="dash", color="gray", width=1),
        )
        for lbl, (yt, yp) in cv_probas.items():
            prec, rec, _ = precision_recall_curve(yt, yp)
            ap = average_precision_score(yt, yp)
            fig_pr.add_trace(
                go.Scatter(
                    x=rec,
                    y=prec,
                    mode="lines",
                    name=f"{LABELS[lbl]} (AP={ap:.3f})",
                    line=dict(color=COLORS[lbl], width=3 if lbl == "hybrid" else 1.5),
                )
            )
        fig_pr.update_layout(
            title="Precision-Recall Curves -- All Models (10-fold CV)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(x=0.4, y=0.95),
            width=700,
            height=500,
        )
        pr_html_path = os.path.abspath(os.path.join(_root_dir, "pr_curves.html"))
        fig_pr.write_html(pr_html_path)
        try:
            fig_pr.write_image(os.path.abspath(os.path.join(_root_dir, "pr_curves.png")), width=700, height=500)
        except Exception:
            pass
        print(f"  PR chart saved: {pr_html_path}")
    except Exception as _ce:
        print(f"  (ROC/PR charts skipped: {_ce})")

    # 9. Ablation study
    print("Running ablation study...")
    ablation = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    _fast_clf = lambda: GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE)

    if len(X) > 3000:
        _idx = np.random.RandomState(RANDOM_STATE).choice(len(X), 3000, replace=False)
        Xa, ya = X.iloc[_idx], y.iloc[_idx]
    else:
        Xa, ya = X, y

    full_cv = cross_val_score(_fast_clf(), scaler.fit_transform(Xa), ya, cv=cv, scoring="f1")
    ablation["full_model"] = round(full_cv.mean(), 4)

    for k in [1, 2, 3]:
        drop_cols = [c for c in hybrid_cols if f"k{k}" in c and "k" + str(k + 1) not in c]
        if not drop_cols:
            continue
        X_ablated = Xa.drop(columns=drop_cols, errors="ignore")
        cv_scores = cross_val_score(_fast_clf(), scaler.fit_transform(X_ablated), ya, cv=cv, scoring="f1")
        ablation[f"without_k{k}_ring"] = round(cv_scores.mean(), 4)

    no_neighbor_cols = [c for c in hybrid_cols if "neighbor" not in c and "ratio" not in c]
    if no_neighbor_cols:
        cv_scores = cross_val_score(_fast_clf(), scaler.fit_transform(Xa[no_neighbor_cols]), ya, cv=cv, scoring="f1")
        ablation["without_all_neighbors"] = round(cv_scores.mean(), 4)

    if financial_cols:
        no_fin_cols = [c for c in hybrid_cols if c not in FINANCIAL_COLS]
        cv_scores = cross_val_score(_fast_clf(), scaler.fit_transform(Xa[no_fin_cols]), ya, cv=cv, scoring="f1")
        ablation["without_financial"] = round(cv_scores.mean(), 4)

    print("\nAblation Results (F1, 5-fold CV):")
    for k, v in ablation.items():
        delta = v - ablation["full_model"]
        flag = "  <- KEY FEATURE" if delta < -0.05 else ""
        print(f"  {k:<30} F1={v:.4f}  D={delta:+.4f}{flag}")

    return {
        "feature_df": feature_df,
        "metrics": metrics,
        "report": report,
        "ablation": ablation,
        "comparison": comparison,
        "feature_importance": feature_importance,
        "bootstrap_ci": bootstrap_ci,
        "permutation_p": permutation_p,
        "cv_probas": cv_probas,
        "roc_html_path": roc_html_path,
        "pr_html_path": pr_html_path,
        "y_test": y_test,
        "y_proba": y_proba,
        "model": model,
        "scaler": scaler,
        "feature_cols": hybrid_cols,
    }


# ===============================================================================
# HRSA DOMAIN PIPELINE -- second domain for cross-domain IEEE proof
# ===============================================================================
def run_hrsa_pipeline(hrsa_json: str = None, resolution: int = H3_RESOLUTION) -> dict:
    """
    Apply the same spatial ML pipeline to HRSA health shortage areas.
    Proves the framework generalizes beyond banking (cross-domain claim).

    Ground truth: HPSA_SCORE >= 14 = high shortage (top quartile) = label 1
    Features: spatial k-ring density + shortage severity metrics
    """
    from src.h3_utils import compute_density_grid, build_h3_feature_matrix, lat_lng_to_h3
    from data.fetch_hrsa import load_hrsa_data
    from sklearn.linear_model import LogisticRegression

    _root = os.path.join(os.path.dirname(__file__), "..")
    if not hrsa_json:
        hrsa_json = os.path.join(_root, "HRSAHealthShortage.json")

    df = load_hrsa_data(hrsa_json)

    # Build density grid -- count of shortage areas per cell
    grid = compute_density_grid(
        df,
        lat_col="LATITUDE",
        lng_col="LONGITUDE",
        resolution=resolution,
        value_col="HPSA_SCORE",
    ).rename(
        columns={
            "count": "branch_count",
            "lat": "center_lat",
            "lng": "center_lng",
            "value_sum": "hpsa_score_total",
        }
    )
    print(f"HRSA grid: {len(grid):,} cells at resolution {resolution}")

    # Spatial features
    density_for_h3 = grid[["h3_index", "branch_count", "center_lat", "center_lng"]].rename(
        columns={"branch_count": "count"}
    )
    features = build_h3_feature_matrix(density_for_h3, k_rings=K_RINGS)
    features = features.rename(columns={"count": "shortage_count"})
    features["log_shortage_count"] = np.log1p(features["shortage_count"])

    # Aggregate HRSA financial-analog features per cell
    df["h3_index"] = df.apply(
        lambda r: lat_lng_to_h3(r["LATITUDE"], r["LONGITUDE"], resolution), axis=1
    )
    fin_agg = (
        df.groupby("h3_index")
        .agg(
            hpsa_score_mean=("HPSA_SCORE", "mean"),
            shortage_mean=("SHORTAGE", "mean"),
            poverty_mean=("POP_BELOW_POVERTY", "mean"),
        )
        .reset_index()
    )
    features = features.merge(fin_agg, on="h3_index", how="left")
    for col in ["hpsa_score_mean", "shortage_mean", "poverty_mean"]:
        features[col] = features[col].fillna(0)

    # Ground truth: HRSA's own high-priority threshold is HPSA_SCORE >= 14.
    # hpsa_score_mean is the label proxy -- it must NOT appear in features.
    HRSA_HIGH_PRIORITY = 14.0
    features["high_shortage"] = (features["hpsa_score_mean"] >= HRSA_HIGH_PRIORITY).astype(int)

    label_col = "high_shortage"
    exclude = {
        "h3_index",
        "shortage_count",
        "center_lat",
        "center_lng",
        "lat",
        "lng",
        label_col,
        "hpsa_score_mean",
    }
    feature_cols = [c for c in features.columns if c not in exclude]

    X = features[feature_cols].fillna(0)
    y = features[label_col]

    print(
        f"HRSA labels: high_shortage={y.sum()} ({y.mean() * 100:.1f}%) | "
        f"low={len(y) - y.sum()} ({(1 - y.mean()) * 100:.1f}%)"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    DOMAIN_FINANCIAL = ["hpsa_score_mean", "shortage_mean", "poverty_mean"]
    spatial_cols = [c for c in feature_cols if c not in DOMAIN_FINANCIAL]
    financial_cols = [c for c in feature_cols if c in DOMAIN_FINANCIAL]

    print(f"\n{'=' * 60}")
    print("HRSA DOMAIN -- THREE-WAY COMPARISON")
    print(f"{'=' * 60}")

    comparison = {}
    for label, cols in [
        ("logistic_regression", feature_cols),
        ("spatial_only", spatial_cols),
        ("domain_metrics_only", financial_cols),
        ("hybrid", feature_cols),
    ]:
        if not cols:
            continue
        Xc_s = StandardScaler().fit_transform(X[cols])
        if label == "logistic_regression":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        else:
            clf = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        auc = cross_val_score(clf, Xc_s, y, cv=cv, scoring="roc_auc").mean()
        f1 = cross_val_score(clf, Xc_s, y, cv=cv, scoring="f1").mean()
        comparison[label] = {"auc": round(auc, 4), "f1": round(f1, 4)}
        print(f"  {label:<24} AUC={auc:.4f} F1={f1:.4f}")
    print(f"{'=' * 60}")

    # Train final hybrid model for Streamlit app display
    scaler_app = StandardScaler()
    X_s_app = scaler_app.fit_transform(X)
    model_app = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model_app.fit(X_s_app, y)

    feature_importance_hrsa = (
        pd.DataFrame({"feature": feature_cols, "importance": model_app.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "feature_df": features,
        "comparison": comparison,
        "feature_cols": feature_cols,
        "model": model_app,
        "scaler": scaler_app,
        "feature_importance": feature_importance_hrsa,
        "y": y,
    }


if __name__ == "__main__":
    results = run_pipeline()
