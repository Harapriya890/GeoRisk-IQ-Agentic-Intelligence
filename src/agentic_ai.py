"""
agentic_ai.py

Optional agentic-AI integration layer for GeoRisk-IQ.

The core GeoRisk-IQ pipeline is deterministic Python/Streamlit code. This
module documents and exposes extension points for teams that want to add
CrewAI, LangGraph, LangChain, and Snowflake-backed agent workflows without
making those heavier dependencies mandatory for the base app.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Dict, List


OPTIONAL_PACKAGES = {
    "crewai": "CrewAI multi-agent orchestration",
    "langgraph": "LangGraph stateful workflow graph",
    "langchain": "LangChain LLM/tool abstractions",
    "snowflake.connector": "Snowflake data warehouse connector",
}


@dataclass(frozen=True)
class AgentRole:
    name: str
    framework: str
    purpose: str
    inputs: List[str]
    outputs: List[str]


AGENT_ROLES = [
    AgentRole(
        name="Data Ingestion Agent",
        framework="LangChain tool wrapper",
        purpose=(
            "Loads FDIC and HRSA files through data.fetch_fdic and "
            "data.fetch_hrsa, then reports missing fields or coordinate issues."
        ),
        inputs=["Geospatiallocation.txt", "FDICFinancialsFull.txt", "FDICFailure.txt", "HRSAHealthShortage.json"],
        outputs=["Clean pandas DataFrames", "data quality notes"],
    ),
    AgentRole(
        name="Spatial Feature Agent",
        framework="LangGraph node",
        purpose=(
            "Runs the H3-style grid and neighborhood feature functions from "
            "src.h3_utils before passing feature matrices to the model layer."
        ),
        inputs=["clean institution or shortage records", "H3 resolution"],
        outputs=["density grid", "neighbor features", "opportunity candidates"],
    ),
    AgentRole(
        name="Risk Modeling Agent",
        framework="CrewAI specialist agent",
        purpose=(
            "Runs src.ml_scorer opportunity scoring and IEEE validation, then "
            "summarizes ROC-AUC, F1, temporal holdout, and feature importance."
        ),
        inputs=["spatial-financial feature matrix", "failure labels"],
        outputs=["model metrics", "ranked risk/opportunity explanations"],
    ),
    AgentRole(
        name="Snowflake Analytics Agent",
        framework="Snowflake + LangChain SQL agent",
        purpose=(
            "Optional production path for querying FDIC/HRSA tables in "
            "Snowflake instead of local JSON files."
        ),
        inputs=["Snowflake connection", "FDIC/HRSA warehouse tables"],
        outputs=["SQL result sets", "auditable analytics summaries"],
    ),
    AgentRole(
        name="Executive Narrative Agent",
        framework="CrewAI report writer",
        purpose=(
            "Converts model outputs and map summaries into human-readable "
            "branch strategy, risk monitoring, or public-data research notes."
        ),
        inputs=["top cells", "metrics", "feature importance"],
        outputs=["dashboard narrative", "Medium/LinkedIn summary"],
    ),
]


def get_optional_dependency_status() -> Dict[str, bool]:
    """Return whether optional agentic-AI packages are importable."""
    return {pkg: _is_package_available(pkg) for pkg in OPTIONAL_PACKAGES}


def _is_package_available(package: str) -> bool:
    """Safely check optional packages, including nested modules."""
    try:
        return find_spec(package) is not None
    except ModuleNotFoundError:
        return False


def get_agentic_architecture() -> Dict[str, object]:
    """Return a serializable architecture summary for docs or Streamlit."""
    return {
        "status": "optional extension",
        "description": (
            "CrewAI, LangGraph, LangChain, and Snowflake can be layered on top "
            "of GeoRisk-IQ to orchestrate data checks, spatial feature runs, "
            "model validation, SQL-backed analytics, and narrative reporting."
        ),
        "dependencies": [
            {
                "package": package,
                "purpose": purpose,
                "installed": installed,
            }
            for package, purpose in OPTIONAL_PACKAGES.items()
            for installed in [_is_package_available(package)]
        ],
        "agents": [asdict(role) for role in AGENT_ROLES],
    }


def build_langgraph_workflow_outline() -> List[Dict[str, str]]:
    """Describe the proposed LangGraph state machine for future execution."""
    return [
        {
            "node": "load_data",
            "implementation": "data.fetch_fdic.load_fdic_data / data.fetch_hrsa.load_hrsa_data",
            "next": "quality_gate",
        },
        {
            "node": "quality_gate",
            "implementation": "validate required columns, coordinate coverage, and numeric fields",
            "next": "build_spatial_features",
        },
        {
            "node": "build_spatial_features",
            "implementation": "src.h3_utils.compute_density_grid and build_h3_feature_matrix",
            "next": "run_models",
        },
        {
            "node": "run_models",
            "implementation": "src.ml_scorer.run_pipeline or run_ieee_pipeline",
            "next": "summarize_outputs",
        },
        {
            "node": "summarize_outputs",
            "implementation": "CrewAI/LangChain report agent using metrics and ranked cells",
            "next": "publish_dashboard",
        },
    ]


def snowflake_table_plan() -> List[Dict[str, str]]:
    """Return a suggested Snowflake schema for productionizing GeoRisk-IQ."""
    return [
        {
            "table": "FDIC_INSTITUTIONS",
            "purpose": "Institution-level locations and attributes loaded from FDIC BankFind files.",
        },
        {
            "table": "FDIC_FINANCIALS",
            "purpose": "Quarterly financial snapshots keyed by certificate identifier and report date.",
        },
        {
            "table": "FDIC_FAILURES",
            "purpose": "Historical failure events used to build validation labels.",
        },
        {
            "table": "GEORISK_CELL_FEATURES",
            "purpose": "Materialized H3-style spatial-financial feature matrix.",
        },
        {
            "table": "GEORISK_MODEL_OUTPUTS",
            "purpose": "Opportunity scores, risk scores, validation metrics, and run metadata.",
        },
    ]


def agentic_prompt_template() -> str:
    """Prompt template for a future GeoRisk-IQ narrative/reporting agent."""
    return (
        "You are the GeoRisk-IQ agentic analytics assistant. Use the supplied "
        "FDIC/HRSA metrics, spatial cells, model validation results, and feature "
        "importance values to produce a concise, evidence-grounded location "
        "intelligence summary. Distinguish observed model outputs from business "
        "recommendations, and never claim regulatory approval or production "
        "supervisory use."
    )
