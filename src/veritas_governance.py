"""
veritas_governance.py

Veritas-style responsible AI and model governance helpers for GeoRisk-IQ.

This module does not depend on a third-party Veritas package. It provides a
lightweight governance layer inspired by common model validation practices:
model cards, fairness/slice testing plans, explainability checks, data lineage,
and deployment controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class GovernanceCheck:
    area: str
    check: str
    evidence: str
    status: str


@dataclass(frozen=True)
class SliceAudit:
    slice_name: str
    rationale: str
    suggested_metrics: List[str]


def build_model_card() -> Dict[str, object]:
    """Return a concise model card for the GeoRisk-IQ project."""
    return {
        "model_name": "GeoRisk-IQ spatial-financial opportunity and risk models",
        "intended_use": (
            "Research and decision support for branch opportunity screening, "
            "spatial risk validation, and public-data geospatial analytics."
        ),
        "not_intended_for": (
            "Automated credit decisions, regulatory supervision, consumer "
            "eligibility decisions, or production use without independent model "
            "risk review."
        ),
        "data_sources": [
            "FDIC institution locations",
            "FDIC historical failure records",
            "FDIC financial snapshots",
            "Optional HRSA shortage-area records for cross-domain testing",
        ],
        "model_families": [
            "Gradient boosting regressor for opportunity scoring",
            "Gradient boosting classifier for failure-risk validation",
            "Logistic regression and ablation baselines",
        ],
        "primary_metrics": ["MAE", "R2", "ROC-AUC", "F1", "precision", "recall"],
        "explainability": [
            "Feature importance",
            "Optional SHAP attribution when shap is installed",
            "Ablation by spatial and financial feature groups",
        ],
        "known_limits": [
            "Grid resolution can affect spatial conclusions",
            "Historical failure geocoding may be incomplete",
            "Latest-quarter financial snapshots do not replace lagged time-series validation",
            "Opportunity target is a proxy, not observed branch profitability",
        ],
    }


def governance_checklist() -> List[Dict[str, str]]:
    """Return Veritas-style governance controls for the current project."""
    checks = [
        GovernanceCheck(
            area="Data lineage",
            check="Document source files, download method, reporting date, and row counts.",
            evidence="README, download_fdic_files.py, data loaders",
            status="implemented",
        ),
        GovernanceCheck(
            area="Data quality",
            check="Validate required columns and drop records without valid coordinates.",
            evidence="data/fetch_fdic.py and data/fetch_hrsa.py",
            status="implemented",
        ),
        GovernanceCheck(
            area="Model validation",
            check="Compare spatial-only, financial-only, hybrid, and baseline models.",
            evidence="src/ml_scorer.py IEEE validation pipeline",
            status="implemented",
        ),
        GovernanceCheck(
            area="Temporal robustness",
            check="Evaluate a time-based holdout where historical dates are available.",
            evidence="pre-2010 train / 2010-2020 test in IEEE pipeline",
            status="implemented",
        ),
        GovernanceCheck(
            area="Explainability",
            check="Report feature importance and SHAP values when available.",
            evidence="Model Insights and IEEE Results tabs",
            status="implemented",
        ),
        GovernanceCheck(
            area="Fairness and geographic equity",
            check="Audit performance and score distributions across states, regions, rural/urban bands, and underserved-area indicators.",
            evidence="recommended next validation step",
            status="planned",
        ),
        GovernanceCheck(
            area="Human oversight",
            check="Keep outputs as decision-support signals and require human review before strategic action.",
            evidence="model card limitations",
            status="planned",
        ),
        GovernanceCheck(
            area="Monitoring",
            check="Track data refresh date, score drift, feature drift, and validation metric changes across runs.",
            evidence="recommended production control",
            status="planned",
        ),
    ]
    return [asdict(check) for check in checks]


def slice_audit_plan() -> List[Dict[str, object]]:
    """Return recommended fairness and robustness slices for GeoRisk-IQ."""
    slices = [
        SliceAudit(
            slice_name="State or region",
            rationale="Banking markets, failure history, and branch density vary by geography.",
            suggested_metrics=["AUC by state/region", "mean score", "false positive rate", "recall"],
        ),
        SliceAudit(
            slice_name="Urban vs rural cells",
            rationale="Spatial resolution and branch-density meaning differ between dense metros and rural markets.",
            suggested_metrics=["score distribution", "calibration", "top-N representation"],
        ),
        SliceAudit(
            slice_name="Low-density vs high-density banking markets",
            rationale="Opportunity scoring intentionally rewards low local density near stronger surrounding markets.",
            suggested_metrics=["precision@K", "score stability", "feature attribution"],
        ),
        SliceAudit(
            slice_name="High-asset vs low-asset cells",
            rationale="Financial aggregates can dominate model behavior if not monitored.",
            suggested_metrics=["AUC", "F1", "feature contribution", "score drift"],
        ),
        SliceAudit(
            slice_name="Historical failure era",
            rationale="Crisis-period failures may differ from post-crisis failures.",
            suggested_metrics=["temporal AUC", "recall", "calibration by period"],
        ),
    ]
    return [asdict(item) for item in slices]


def deployment_controls() -> List[Dict[str, str]]:
    """Return suggested controls before production use."""
    return [
        {
            "control": "Human-in-the-loop review",
            "description": "Require analyst review for any branch expansion or risk-monitoring action.",
        },
        {
            "control": "Data refresh logging",
            "description": "Record source file date, reporting period, row counts, and excluded rows for every run.",
        },
        {
            "control": "Model drift monitoring",
            "description": "Compare score distributions, feature distributions, and validation metrics across refreshes.",
        },
        {
            "control": "Geographic equity review",
            "description": "Check whether opportunity rankings systematically underrepresent specific regions or community types.",
        },
        {
            "control": "Use-case boundary",
            "description": "Document that outputs are not automated credit, eligibility, or regulatory decisions.",
        },
    ]


def veritas_summary() -> str:
    """Return a short responsible-AI summary for the dashboard."""
    return (
        "GeoRisk-IQ includes a Veritas-style governance layer for transparent "
        "model documentation, validation evidence, explainability, geographic "
        "slice audits, and human oversight. It is a local governance framework, "
        "not a claim of certification or regulatory approval."
    )

