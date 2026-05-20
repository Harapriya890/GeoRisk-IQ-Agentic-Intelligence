import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.agentic_runtime import (  # noqa: E402
    get_crewai_blueprint,
    get_langchain_tools,
    run_local_agentic_workflow,
    snowflake_health_check,
)


def test_agentic_blueprint_has_agents():
    blueprint = get_crewai_blueprint()
    assert len(blueprint["agents"]) >= 3
    assert "tasks" in blueprint


def test_langchain_tools_safe_without_dependency():
    tools = get_langchain_tools()
    assert isinstance(tools, list)


def test_snowflake_health_check_safe_without_credentials():
    result = snowflake_health_check()
    assert result["status"] in {"not_configured", "connector_missing", "ok", "failed"}


def test_local_agentic_workflow_runs_on_sample():
    data_path = os.path.join(ROOT, "Geospatiallocation.txt")
    result = run_local_agentic_workflow(
        data_path=data_path,
        resolution=5,
        top_n=5,
        sample_rows=100,
        execute_model=False,
    )
    assert result["runtime"] == "local"
    assert "summary" in result
    assert len(result["steps"]) >= 5

