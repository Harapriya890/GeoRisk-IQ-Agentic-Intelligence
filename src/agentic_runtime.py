"""
agentic_runtime.py

Executable Agentic AI runtime for GeoRisk-IQ.

The runtime is intentionally dependency-safe:
- It always supports a deterministic local workflow.
- If LangGraph is installed, it can execute the same workflow as a graph.
- If LangChain is installed, it exposes pipeline functions as tools.
- If Snowflake credentials and connector are available, it can run a simple
  warehouse health query.

No LLM key is required to test this module.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from data.fetch_fdic import load_fdic_data
from src.agentic_ai import (
    agentic_prompt_template,
    build_langgraph_workflow_outline,
    get_optional_dependency_status,
    snowflake_table_plan,
)
from src.h3_utils import compute_density_grid


@dataclass
class AgentStepResult:
    agent: str
    status: str
    message: str
    outputs: Dict[str, Any]


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def data_ingestion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load a small FDIC summary and validate coordinate readiness."""
    data_path = state.get("data_path")
    sample_rows = int(state.get("sample_rows", 1000))
    result = AgentStepResult(
        agent="Data Ingestion Agent",
        status="started",
        message="Loading FDIC institution data and checking required fields.",
        outputs={},
    )

    try:
        df = load_fdic_data(data_path) if data_path else load_fdic_data()
        total_rows = len(df)
        sample = df.head(sample_rows).copy()
        coord_ready = {"LATITUDE", "LONGITUDE"}.issubset(set(sample.columns))
        result.status = "completed"
        result.message = "FDIC data loaded and coordinate fields validated."
        result.outputs = {
            "total_rows": int(total_rows),
            "sample_rows": int(len(sample)),
            "columns": list(sample.columns),
            "coordinate_ready": bool(coord_ready),
            "states": int(sample["STALP"].nunique()) if "STALP" in sample.columns else None,
        }
        state["_fdic_sample"] = sample
    except Exception as exc:
        result.status = "failed"
        result.message = f"Data ingestion failed: {exc}"
        result.outputs = {"error": str(exc)}

    state.setdefault("steps", []).append(asdict(result))
    return state


def spatial_feature_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a lightweight density grid from the sampled data."""
    result = AgentStepResult(
        agent="Spatial Feature Agent",
        status="started",
        message="Building H3-style spatial density features from the data sample.",
        outputs={},
    )

    try:
        sample = state.get("_fdic_sample")
        if sample is None or len(sample) == 0:
            raise ValueError("No FDIC sample available from Data Ingestion Agent.")

        resolution = int(state.get("resolution", 5))
        lat_col = "LATITUDE" if "LATITUDE" in sample.columns else "latitude"
        lng_col = "LONGITUDE" if "LONGITUDE" in sample.columns else "longitude"
        value_col = "ASSET" if "ASSET" in sample.columns else None
        density = compute_density_grid(
            sample,
            lat_col=lat_col,
            lng_col=lng_col,
            resolution=resolution,
            value_col=value_col,
        )
        state["_density_sample"] = density

        branch_col = "branch_count" if "branch_count" in density.columns else "count"
        result.status = "completed"
        result.message = "Spatial density grid built successfully."
        result.outputs = {
            "resolution": resolution,
            "cell_count": int(len(density)),
            "branch_count_sum": int(density[branch_col].sum()) if branch_col in density.columns else None,
            "has_centers": bool({"center_lat", "center_lng"}.issubset(set(density.columns))),
        }
    except Exception as exc:
        result.status = "failed"
        result.message = f"Spatial feature build failed: {exc}"
        result.outputs = {"error": str(exc)}

    state.setdefault("steps", []).append(asdict(result))
    return state


def risk_modeling_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize whether model execution is available or run the full scorer."""
    result = AgentStepResult(
        agent="Risk Modeling Agent",
        status="started",
        message="Checking model execution mode.",
        outputs={},
    )

    try:
        if state.get("execute_model"):
            from src.ml_scorer import run_pipeline

            pipeline = run_pipeline(
                json_path=state.get("data_path"),
                resolution=int(state.get("resolution", 5)),
                top_n=int(state.get("top_n", 10)),
            )
            metrics = pipeline.get("metrics", {})
            top_ops = pipeline.get("top_opportunities", pd.DataFrame())
            result.outputs = {
                "mode": "full_pipeline",
                "metrics": metrics,
                "top_opportunity_count": int(len(top_ops)),
            }
            result.message = "Full GeoRisk-IQ scoring pipeline executed."
        else:
            density = state.get("_density_sample")
            if density is None:
                raise ValueError("No spatial density sample available.")

            branch_col = "branch_count" if "branch_count" in density.columns else "count"
            branch_values = density[branch_col] if branch_col in density.columns else pd.Series(dtype=float)
            result.outputs = {
                "mode": "lightweight_agent_check",
                "mean_branches_per_cell": _safe_float(branch_values.mean()),
                "max_branches_in_cell": _safe_float(branch_values.max()),
                "model_note": "Set execute_model=True to run the full ML scorer.",
            }
            result.message = "Lightweight model-readiness summary generated."

        result.status = "completed"
    except Exception as exc:
        result.status = "failed"
        result.message = f"Risk modeling step failed: {exc}"
        result.outputs = {"error": str(exc)}

    state.setdefault("steps", []).append(asdict(result))
    return state


def governance_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Attach responsible-AI governance artifacts."""
    result = AgentStepResult(
        agent="Governance Agent",
        status="started",
        message="Attaching model governance and responsible-AI controls.",
        outputs={},
    )

    try:
        from src.veritas_governance import (
            build_model_card,
            deployment_controls,
            governance_checklist,
            slice_audit_plan,
        )

        result.status = "completed"
        result.message = "Governance artifacts attached."
        result.outputs = {
            "model_card": build_model_card(),
            "check_count": len(governance_checklist()),
            "slice_audit_count": len(slice_audit_plan()),
            "deployment_control_count": len(deployment_controls()),
        }
    except Exception as exc:
        result.status = "failed"
        result.message = f"Governance step failed: {exc}"
        result.outputs = {"error": str(exc)}

    state.setdefault("steps", []).append(asdict(result))
    return state


def executive_narrative_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a deterministic executive summary from previous steps."""
    failed = [step for step in state.get("steps", []) if step.get("status") == "failed"]
    completed = [step for step in state.get("steps", []) if step.get("status") == "completed"]
    domain = state.get("domain", "banking")

    if failed:
        narrative = (
            f"GeoRisk-IQ agentic workflow for {domain} completed with "
            f"{len(failed)} issue(s). Review failed agent outputs before using results."
        )
    else:
        narrative = (
            f"GeoRisk-IQ agentic workflow for {domain} completed successfully. "
            "The run validated data readiness, built spatial density features, "
            "generated model-readiness outputs, and attached governance artifacts."
        )

    result = AgentStepResult(
        agent="Executive Narrative Agent",
        status="completed",
        message="Narrative summary generated.",
        outputs={
            "summary": narrative,
            "completed_agents": len(completed),
            "failed_agents": len(failed),
            "prompt_template": agentic_prompt_template(),
        },
    )
    state.setdefault("steps", []).append(asdict(result))
    state["summary"] = narrative
    return state


def run_local_agentic_workflow(
    data_path: Optional[str] = None,
    resolution: int = 5,
    top_n: int = 10,
    sample_rows: int = 1000,
    execute_model: bool = False,
    domain: str = "banking",
) -> Dict[str, Any]:
    """Run the deterministic local agent workflow."""
    state: Dict[str, Any] = {
        "domain": domain,
        "data_path": data_path,
        "resolution": resolution,
        "top_n": top_n,
        "sample_rows": sample_rows,
        "execute_model": execute_model,
        "steps": [],
        "runtime": "local",
    }
    for node in [
        data_ingestion_agent,
        spatial_feature_agent,
        risk_modeling_agent,
        governance_agent,
        executive_narrative_agent,
    ]:
        state = node(state)
    state.pop("_fdic_sample", None)
    state.pop("_density_sample", None)
    return state


def run_langgraph_agentic_workflow(**kwargs: Any) -> Dict[str, Any]:
    """Run the same workflow through LangGraph if it is installed."""
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:
        fallback = run_local_agentic_workflow(**kwargs)
        fallback["runtime"] = "local_fallback"
        fallback["langgraph_error"] = str(exc)
        return fallback

    def _start_state(_: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain": kwargs.get("domain", "banking"),
            "data_path": kwargs.get("data_path"),
            "resolution": kwargs.get("resolution", 5),
            "top_n": kwargs.get("top_n", 10),
            "sample_rows": kwargs.get("sample_rows", 1000),
            "execute_model": kwargs.get("execute_model", False),
            "steps": [],
            "runtime": "langgraph",
        }

    graph = StateGraph(dict)
    graph.add_node("start", _start_state)
    graph.add_node("data_ingestion", data_ingestion_agent)
    graph.add_node("spatial_features", spatial_feature_agent)
    graph.add_node("risk_modeling", risk_modeling_agent)
    graph.add_node("governance", governance_agent)
    graph.add_node("narrative", executive_narrative_agent)
    graph.set_entry_point("start")
    graph.add_edge("start", "data_ingestion")
    graph.add_edge("data_ingestion", "spatial_features")
    graph.add_edge("spatial_features", "risk_modeling")
    graph.add_edge("risk_modeling", "governance")
    graph.add_edge("governance", "narrative")
    graph.add_edge("narrative", END)

    app = graph.compile()
    result = app.invoke({})
    result.pop("_fdic_sample", None)
    result.pop("_density_sample", None)
    return result


def get_langchain_tools() -> List[Any]:
    """Return LangChain tools if LangChain is installed, otherwise an empty list."""
    try:
        from langchain.tools import tool
    except Exception:
        return []

    @tool
    def georisk_dependency_status() -> Dict[str, bool]:
        """Return optional GeoRisk-IQ dependency status."""
        return get_optional_dependency_status()

    @tool
    def georisk_workflow_outline() -> List[Dict[str, str]]:
        """Return the GeoRisk-IQ LangGraph workflow outline."""
        return build_langgraph_workflow_outline()

    @tool
    def georisk_snowflake_table_plan() -> List[Dict[str, str]]:
        """Return the suggested GeoRisk-IQ Snowflake table plan."""
        return snowflake_table_plan()

    return [georisk_dependency_status, georisk_workflow_outline, georisk_snowflake_table_plan]


def get_crewai_blueprint() -> Dict[str, Any]:
    """Return a CrewAI-ready role/task blueprint without requiring an LLM key."""
    return {
        "note": "This blueprint is deterministic and does not instantiate CrewAI agents.",
        "agents": [
            {"role": "Data Ingestion Agent", "goal": "Validate source files and coordinate readiness."},
            {"role": "Spatial Feature Agent", "goal": "Build H3-style density and neighborhood features."},
            {"role": "Risk Modeling Agent", "goal": "Run scoring and validation workflows."},
            {"role": "Governance Agent", "goal": "Attach model cards, slice audits, and controls."},
            {"role": "Executive Narrative Agent", "goal": "Summarize findings for decision makers."},
        ],
        "tasks": [
            "Load and validate data.",
            "Build spatial features.",
            "Run lightweight or full modeling workflow.",
            "Attach responsible-AI governance artifacts.",
            "Generate an executive narrative.",
        ],
    }


def snowflake_health_check() -> Dict[str, Any]:
    """Run a Snowflake SELECT 1 check when credentials and connector exist."""
    required_env = [
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        return {
            "status": "not_configured",
            "message": "Snowflake credentials are not configured.",
            "missing_env": missing,
        }

    try:
        import snowflake.connector
    except Exception as exc:
        return {
            "status": "connector_missing",
            "message": "snowflake-connector-python is not installed.",
            "error": str(exc),
        }

    conn = None
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        value = cur.fetchone()[0]
        cur.close()
        return {"status": "ok", "query": "SELECT 1", "result": int(value)}
    except Exception as exc:
        return {"status": "failed", "message": "Snowflake health check failed.", "error": str(exc)}
    finally:
        if conn is not None:
            conn.close()
