# GeoRisk-IQ

Agentic AI geospatial intelligence application for bank branch opportunity scoring, FDIC failure risk validation, and HRSA shortage-area cross-domain testing.

GeoRisk-IQ solves a location-dependent decision problem: organizations often know what happened, but not where the surrounding spatial context changes the interpretation. The project converts records with coordinates into spatial cells, neighborhood features, opportunity/risk scores, dashboards, and governance artifacts.

The idea is based on cross-domain systems thinking from fleet GPS tracking and driver behavior analytics, plus prior experience architecting Agentic AI workflows with tools such as LangGraph and LangChain. Volvo/fleet-style GPS work involves routes, movement, dwell-time hotspots, harsh braking, speeding, idle time, service gaps, and operational risk. GeoRisk-IQ applies the same spatial intelligence pattern to banking failure-risk validation, branch opportunity scoring, and healthcare shortage-area analysis.

Potential users:

- Banks, credit unions, fintech lenders, branch strategy teams, and financial-risk teams can use the pattern for branch expansion screening, market saturation analysis, underserved-area discovery, spatial risk monitoring, and branch network optimization.
- Healthcare systems, clinic planners, public-health teams, HRSA-style shortage-area analysts, and access-to-care researchers can use the pattern for provider shortage analysis, facility placement, rural/urban access planning, and service-gap dashboards.

## Project Structure

```text
GeoRisk-IQ/
├── app.py
├── download_fdic_files.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── h3_utils.py
│   ├── ml_scorer.py
│   └── ground_truth.py
└── data/
    ├── __init__.py
    ├── fetch_fdic.py
    └── fetch_hrsa.py
```

## Required data files in project root

```text
Geospatiallocation.txt
GeospatialFull.txt
FDICFinancialsFull.txt
FDICFinancials.txt
FDICFailure.txt
HRSAHealthShortage.json
```

FDIC files can be downloaded with:

```bash
python download_fdic_files.py
```

HRSAHealthShortage.json is not from FDIC. Add it manually from HRSA source in the same root folder.

## Run

```bash
pip install -r requirements.txt
python download_fdic_files.py
streamlit run app.py
```

## Notes

- `app.py` is the Streamlit UI.
- `src/agentic_ai.py` documents optional CrewAI, LangGraph, LangChain, and Snowflake agentic-AI extension points.
- `src/advanced_responsible_ai.py` maps advanced AI capabilities and responsible AI controls.
- `src/veritas_governance.py` provides Veritas-style responsible AI governance artifacts.
- `src/cross_domain_use_cases.py` documents healthcare, fleet GPS, driver behavior, banking operations, and cross-platform extension paths.
- `src/ml_scorer.py` contains the ML pipeline functions.
- `src/h3_utils.py` contains grid/density utilities.
- `src/ground_truth.py` builds failure labels for IEEE-style validation.

## Agentic AI Application Architecture

GeoRisk-IQ is positioned as an Agentic AI application architecture with a base Streamlit app that continues to work without heavy agent dependencies. The deterministic spatial ML workflow runs with `requirements.txt`; the optional agentic layer documents how LangGraph, LangChain, CrewAI, Snowflake, and governance agents can orchestrate data ingestion, spatial feature engineering, model validation, reporting, and controls.

- CrewAI multi-agent orchestration
- LangGraph stateful workflow graphs
- LangChain tool and LLM abstractions
- Snowflake-backed FDIC/HRSA/future telemetry analytics tables
- Veritas-style governance artifacts and responsible-AI reporting

Proposed agent responsibilities:

- Data Ingestion Agent: validates FDIC, HRSA, or future telemetry files
- Spatial Feature Agent: builds H3-style grid, density, and neighborhood features
- Risk Modeling Agent: runs scoring, validation, and model comparison
- Snowflake Analytics Agent: queries production-scale warehouse tables
- Executive Narrative Agent: creates decision-ready summaries from maps, metrics, and feature importance
- Governance Agent pattern: supports model cards, slice audits, controls, and human review

The base app does not require these packages. To experiment with the optional agentic packages:

```bash
pip install -r requirements-agentic.txt
streamlit run app.py
```

Then open the **Agentic AI** tab in the dashboard.

You can also test the executable agentic runtime from the command line:

```bash
python -c "from src.agentic_runtime import run_local_agentic_workflow; print(run_local_agentic_workflow(sample_rows=100, execute_model=False)['summary'])"
python -c "from src.agentic_runtime import snowflake_health_check; print(snowflake_health_check())"
```

If `pytest` is installed:

```bash
python -m pytest tests/test_agentic_runtime.py
```

Without `pytest`, the same assertions can be run directly:

```bash
python tests/test_agentic_runtime.py
```

## Advanced AI + Responsible AI

GeoRisk-IQ includes an advanced AI and responsible AI architecture map covering:

- Agentic AI orchestration with CrewAI, LangGraph, and LangChain
- Geospatial machine learning and H3-style spatial feature engineering
- Explainable AI with feature importance and optional SHAP
- Snowflake warehouse-backed analytics for future production workflows
- RAG/narrative intelligence for analyst-ready reporting
- Cross-domain reuse across banking, healthcare, fleet, driver behavior, and platform analytics
- Veritas-style model cards, governance checklists, slice audits, and deployment controls
- Future monitoring for data drift, score drift, metric drift, privacy, and security controls

Open the **Advanced AI** tab in the dashboard. The base app remains lightweight; advanced agentic, RAG, Snowflake, monitoring, and privacy/security capabilities are documented as architecture layers unless explicitly implemented in the current pipeline.

## Veritas-Style Governance

GeoRisk-IQ also includes a lightweight Veritas-style responsible AI governance layer:

- Model card
- Data lineage and validation checklist
- Explainability and ablation evidence
- Fairness and geographic slice audit plan
- Human oversight and deployment controls

Open the **Veritas Governance** tab in the dashboard. This is local model-governance documentation, not third-party certification or regulatory approval.

## Cross-Domain Platform Strategy

GeoRisk-IQ is designed as a reusable spatial intelligence pattern:

1. Load domain records with coordinates and timestamps.
2. Map observations to H3-style spatial cells or domain-specific route/service segments.
3. Engineer local density, neighboring activity, isolation, competition, risk, and utilization features.
4. Train or validate domain-specific opportunity, shortage, risk, safety, or service-improvement scores.
5. Govern outputs with responsible-AI controls.

Implemented domains:

- Banking: FDIC branch opportunity scoring and failure-risk validation
- Healthcare: HRSA shortage-area spatial validation has been tested with downloaded data in a separate workspace and can be transferred into this repository

Extension blueprints:

- Volvo/fleet GPS-style truck tracking: route efficiency, dwell-time hotspots, corridor gaps, and service reliability
- Driver behavior tracking: harsh braking, speeding, idle time, fatigue proxies, safety scoring, and coaching priority
- Banking system improvement: branch service gaps, ATM usage, customer access, and operational stress scoring
- Cross-platform analytics: finance, healthcare, logistics, mobility, insurance, and public-service spatial intelligence

The project demonstrates cross-domain systems thinking: the same spatial pattern used for banking has already been tested on healthcare data in another workspace. Volvo/fleet and driver-behavior examples reflect prior fleet GPS and operational tracking experience and remain architecture blueprints until connected to real telemetry data or an official platform API.
