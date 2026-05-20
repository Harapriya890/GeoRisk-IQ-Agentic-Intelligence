# GeoRisk-IQ: Using Geospatial AI to Score Bank Branch Opportunity and Validate Failure Risk

**Subtitle:** A practical look at how FDIC data, spatial indexing, and machine learning can help banks reason about local market opportunity and risk.

**GitHub:** https://github.com/<your-username>/GeoRisk-IQ

## The Problem

Bank branch strategy is still deeply geographic.

A branch is not just a financial asset on a balance sheet. It sits inside a local market: near customers, competitors, underserved communities, growth corridors, and sometimes regions where historical failures reveal structural fragility. Traditional bank-risk models often focus on institution-level financial ratios such as return on assets, net income, and asset size. Those signals matter, but they miss a simple question:

**Where is the bank operating?**

GeoRisk-IQ was built around that question. It combines FDIC banking records with spatial feature engineering and machine learning to support two related tasks:

1. Identifying potential branch opportunity zones.
2. Validating whether spatial market structure adds signal to bank failure-risk modeling.

The result is a reproducible geospatial AI pipeline and Streamlit dashboard for financial location intelligence.

More explicitly, GeoRisk-IQ solves a location-dependent decision problem: organizations often have records, financial indicators, events, or service activity, but they do not have a clean way to convert those records into local and neighboring spatial context. In banking, this makes it hard to see which areas are underserved, saturated, or historically associated with higher failure risk. In healthcare, it makes it hard to compare provider shortages, access gaps, and nearby service availability. In fleet operations, the same pattern appears in route risk, dwell-time hotspots, unsafe driving behavior, and service reliability.

My own path into this problem came from prior Volvo/fleet GPS tracking and driver behavior analytics work. Fleet systems force you to think spatially: vehicles move through routes, create GPS traces, stop in hotspots, idle in certain zones, and generate safety signals such as harsh braking, speeding, cornering, and fatigue proxies. That experience shaped the systems-thinking behind GeoRisk-IQ. I recognized that the same spatial intelligence pattern could be applied to banking failures, branch opportunity, and healthcare shortage-area analysis.

## What GeoRisk-IQ Does

GeoRisk-IQ maps bank institution records into spatial cells, engineers neighborhood-level density features, joins financial indicators, and trains machine-learning models over the resulting spatial-financial feature set.

At a high level, the workflow is:

1. Load FDIC institution, financial, and failure datasets.
2. Clean and normalize coordinates, certificate identifiers, and financial fields.
3. Aggregate institutions into H3-style spatial grid cells.
4. Create local and neighborhood density features.
5. Build opportunity scores for underserved but strategically connected cells.
6. Train and compare failure-risk validation models.
7. Expose the outputs through an interactive Streamlit dashboard.

The project also includes a healthcare cross-domain path using HRSA shortage-area data. I have already tested that healthcare data in a separate workspace, and the same code/data pattern can be transferred into this repository later. This is the larger idea behind GeoRisk-IQ: spatial systems thinking can move across domains. The same architecture can also be extended to fleet GPS tracking, driver behavior analytics, banking operations improvement, and other cross-platform spatial intelligence use cases when appropriate telemetry or platform data is available.

## Why Spatial Context Matters

Consider two banks with similar financial ratios.

One operates in a dense, competitive corridor with many nearby institutions. Another sits in a thinly served area with few surrounding branches. A traditional model may treat them as similar. GeoRisk-IQ treats their local context as different.

The spatial features include:

- Local branch density inside each cell.
- Neighboring branch density across nearby cells.
- Competition and isolation ratios.
- Multi-ring neighborhood structure.
- Cell-level financial aggregates such as return on assets, net income, and assets.

This makes geography more than a visualization layer. It becomes part of the model.

## Data Sources

The current reproducible run uses public FDIC datasets:

- **FDIC institution records:** 27,832 mapped institution records.
- **FDIC failure records:** historical bank failure events.
- **FDIC financial records:** latest available financial snapshot used for cell-level aggregates.

The paper version of the project reports 6,086 generated spatial cells, with 1,403 cells receiving at least one historical failure label in the validation setup.

All of this is implemented in a lightweight Python project:

- `app.py` powers the Streamlit interface.
- `download_fdic_files.py` downloads required FDIC files.
- `data/fetch_fdic.py` loads and cleans FDIC institution JSON records.
- `data/fetch_hrsa.py` loads and normalizes HRSA shortage-area records.
- `src/agentic_ai.py` documents optional CrewAI, LangGraph, LangChain, and Snowflake agentic-AI extension points.
- `src/advanced_responsible_ai.py` maps advanced AI capabilities and responsible AI controls.
- `src/veritas_governance.py` provides Veritas-style responsible AI governance artifacts.
- `src/cross_domain_use_cases.py` documents healthcare, fleet GPS, driver behavior, banking operations, and cross-platform extension paths.
- `src/ml_scorer.py` contains the machine-learning pipeline.
- `src/h3_utils.py` contains grid and density utilities.
- `src/ground_truth.py` builds failure labels for validation.

## How the Code Is Organized

The GeoRisk-IQ codebase is intentionally split into small layers so the project can work both as a research pipeline and as an interactive demo.

**Data ingestion:**  
`data/fetch_fdic.py` reads the FDIC institution file, extracts records from the downloaded JSON structure, validates required fields, converts latitude and longitude to numeric values, drops unmappable rows, and coerces optional financial fields such as assets, offices, certificate number, net income, and return on assets.

`data/fetch_hrsa.py` follows the same pattern for HRSA Health Professional Shortage Area data. It loads local JSON, extracts records, validates coordinates, and normalizes shortage-related fields such as HPSA score, shortage count, and poverty indicators. This mirrors the FDIC loader so the same spatial pipeline can be reused across domains.

**Spatial feature engineering:**  
`src/h3_utils.py` converts latitude and longitude into H3-style grid cells, computes branch density by cell, builds neighborhood density features, estimates cell centers and boundaries, and identifies high-opportunity areas based on local scarcity and surrounding market activity.

**Machine learning:**  
`src/ml_scorer.py` is the main modeling layer. It builds feature matrices, creates opportunity targets, trains scoring models, ranks top opportunity cells, runs the IEEE-style validation pipeline, and applies the same spatial logic to the HRSA healthcare shortage experiment.

**Ground truth construction:**  
`src/ground_truth.py` joins FDIC institutions, failures, and financial records. It builds labeled spatial cells by linking failures through certificate identifiers where possible and aggregates financial indicators at the cell level.

**Application layer:**  
`app.py` turns the pipeline into an interactive Streamlit dashboard. It supports the FDIC banking workflow, the HRSA healthcare workflow, top opportunity maps, density maps, model-performance views, downloadable grids, and IEEE validation outputs.

**Agentic AI extension:**  
`src/agentic_ai.py` adds an Agentic AI application architecture layer for teams that want to connect GeoRisk-IQ to modern agent frameworks. This design is based on my prior experience with Agentic AI workflows using tools such as LangGraph and LangChain. The base Streamlit app still works without installing the agentic packages; the agentic layer shows how the same deterministic pipeline can be orchestrated by specialist agents. CrewAI can coordinate agents for ingestion, spatial feature engineering, model validation, governance, and narrative reporting. LangGraph can express the workflow as a state machine from data loading to dashboard publishing. LangChain can wrap pipeline functions as tools for LLM-assisted analysis. Snowflake can serve as the production data warehouse for FDIC, HRSA, future telemetry, spatial-feature, and model-output tables.

The proposed agent roles are:

- Data Ingestion Agent for validating FDIC, HRSA, or future telemetry files.
- Spatial Feature Agent for building H3-style grid, density, and neighborhood features.
- Risk Modeling Agent for opportunity scoring, failure-risk validation, and model comparison.
- Snowflake Analytics Agent for warehouse-backed analytics.
- Executive Narrative Agent for converting maps, metrics, and feature importance into decision-ready summaries.
- Governance Agent pattern for model cards, fairness slice audits, responsible-AI controls, and human review.

The base Streamlit app does not require these packages. The optional dependencies are isolated in `requirements-agentic.txt`, and the dashboard includes an Agentic AI tab that shows the proposed application architecture, dependency status, LangGraph workflow outline, Snowflake table plan, and reporting-agent prompt template.

**Advanced AI and Responsible AI layer:**  
`src/advanced_responsible_ai.py` brings the full architecture together. It documents advanced AI capabilities such as agentic orchestration, geospatial machine learning, explainable AI, Snowflake-backed analytics, future RAG/narrative intelligence, cross-domain reuse, and MLOps monitoring. It also documents responsible AI controls such as model cards, data lineage, fairness and geographic slice audits, explainability, human-in-the-loop review, use-case boundaries, privacy/security guardrails, and drift reporting. The dashboard includes an Advanced AI tab so reviewers can see which capabilities are implemented now and which are future enterprise extensions.

**Veritas-style governance:**  
`src/veritas_governance.py` adds responsible AI documentation for the project. It creates a model card, governance checklist, fairness and geographic slice audit plan, deployment controls, and use-case boundaries. This is especially important because GeoRisk-IQ works with financial-location signals. The dashboard includes a Veritas Governance tab to show these artifacts clearly. It is not a claim of regulatory approval or external certification; it is a transparent model-governance layer for review and communication.

**Cross-domain platform layer:**  
`src/cross_domain_use_cases.py` makes the platform strategy explicit. Banking is implemented in this repository. The HRSA healthcare validation has already been tested in another workspace and can be transferred here later. Fleet GPS and driver behavior analytics are documented as extension blueprints: the same grid-and-neighborhood logic can be applied to Volvo/fleet GPS-style truck telematics, route traces, dwell-time hotspots, harsh braking, speeding, idle time, fatigue proxies, safety scoring, and coaching priority. For banking operations, the same pattern can support branch service-gap detection, ATM usage analysis, customer-access improvement, and operational stress scoring. The fleet and driver-behavior use cases require real telemetry or platform data before they become production integrations.

That structure makes GeoRisk-IQ easy to extend. A new domain can reuse the same pattern: write a loader, map observations to cells, create domain-specific labels, and run the spatial machine-learning pipeline.

## Cross-Domain Potential

GeoRisk-IQ is best understood as a reusable spatial intelligence framework. The domain changes, but the pattern stays consistent:

1. Load records with coordinates, timestamps, and domain-specific attributes.
2. Map events or assets to H3-style cells, route segments, or service areas.
3. Engineer local and neighboring density features.
4. Train a domain-specific score or validate against historical outcomes.
5. Visualize and govern the results.

In banking, that means branch opportunity and failure-risk validation. In healthcare, it means shortage-area prioritization. In fleet and logistics, it could mean GPS-based route efficiency, dwell-time hotspots, and corridor risk. In driver behavior analytics, it could mean safety scoring from harsh braking, speeding, idle time, and fatigue indicators. In banking operations, it could mean service-gap detection, ATM-demand analysis, and branch-system improvement.

The important distinction is that the FDIC banking path is implemented in this repository, while the HRSA healthcare path has been tested in another workspace and is transfer-ready. Fleet GPS, Volvo-style telematics, driver behavior, and broader platform analytics are documented extension paths ready for future data connectors.

## Modeling Approach

GeoRisk-IQ evaluates four modeling views:

1. **Logistic regression baseline**
2. **Spatial-only gradient boosting**
3. **Financial-only gradient boosting**
4. **Hybrid spatial-financial gradient boosting**

This comparison is important because the goal is not simply to build a high-performing model. The goal is to test whether spatial market structure adds useful signal beyond financial data alone.

The validation protocol includes:

- 10-fold stratified cross-validation.
- Temporal holdout validation.
- Bootstrap confidence intervals.
- Permutation testing.
- Ablation analysis.
- Feature-importance analysis.

## Results

In the IEEE-style paper draft, GeoRisk-IQ reports strong validation performance for the hybrid spatial-financial model:

- **Opportunity scoring:** mean absolute error of 0.0081 and R2 of 0.9991 against a spatial opportunity proxy.
- **Failure-risk validation:** 10-fold cross-validation ROC-AUC of 0.8341.
- **Bootstrap 95% confidence interval:** [0.8226, 0.8454].
- **Hybrid vs. spatial-only:** +0.0062 AUC improvement, permutation p = 0.0013.
- **Hybrid vs. financial-only:** +0.2312 AUC improvement, p < 0.0001.

The key takeaway is not just that the hybrid model performs well. It is that spatial neighborhood structure appears to carry measurable, interpretable signal for financial location intelligence.

## The Dashboard

GeoRisk-IQ includes an interactive Streamlit dashboard for exploring:

- Top opportunity zones.
- Branch-density maps.
- H3-style spatial cells.
- Model comparison metrics.
- Temporal validation outputs.
- Feature-importance views.
- Banking and HRSA healthcare domain modes.

This is useful because geospatial models are easier to trust when users can inspect the underlying geography. A ranked cell is more meaningful when you can see its position, surrounding density, and relationship to nearby markets.

## What Makes This Project Interesting

Three ideas make GeoRisk-IQ especially useful:

**1. Geography is modeled directly.**  
The project does not stop at plotting branches on a map. It engineers spatial features that become part of the predictive model.

**2. The model is validated, not just visualized.**  
GeoRisk-IQ compares spatial-only, financial-only, and hybrid models to quantify the contribution of each signal type.

**3. The pipeline is portable.**  
The HRSA healthcare shortage experiment shows that the same spatial machine-learning approach can be reused for other location-based public datasets.

## Practical Use Cases

GeoRisk-IQ can support several financial analytics workflows:

- Branch expansion screening.
- Market saturation analysis.
- Underserved-area discovery.
- Spatial risk monitoring.
- Competitive-density analysis.
- Public-data experimentation for financial AI research.

It is not a production credit or supervisory model. It is a research and decision-support framework that shows how public spatial data can be converted into interpretable machine-learning signals.

Potential banking users include regional banks, community banks, credit unions, fintech lenders, bank strategy teams, branch network planners, and financial-risk teams. They can use the framework to evaluate where branch access is thin, where competition is dense, where historical risk patterns cluster, and where financial ratios should be interpreted together with local market geography.

Potential healthcare users include hospital systems, clinic planners, public-health agencies, HRSA-style shortage-area analysts, rural-health teams, and access-to-care researchers. They can use the same pattern to identify underserved service areas, compare local demand with nearby provider availability, prioritize facility placement, and build spatial dashboards for public-health decision support. I have already tested HRSA healthcare shortage-area data in another workspace, and that workflow can be transferred into this repository later.

## Limitations and Future Work

There are several clear next steps:

- Validate against native H3 hexagons where available.
- Add finer spatial resolutions for dense urban markets.
- Use lagged historical financial time series instead of only the latest financial snapshot.
- Improve historical failure geocoding where branch-level coordinates are unavailable.
- Add demographic, macroeconomic, and census-area features.
- Expand model explainability through SHAP and local cell-level narratives.
- Run Veritas-style slice audits across states, regions, rural/urban cells, asset bands, and historical failure eras.
- Transfer the verified HRSA healthcare workflow from the separate workspace into this repository.
- Add fleet GPS and driver behavior connectors for route traces, harsh braking, speeding, idle time, dwell-time hotspots, and service reliability analysis.
- Add Snowflake tables for FDIC, HRSA, future telemetry, model outputs, audit logs, and governance reports.
- Add agent-generated strategy briefs using CrewAI, LangGraph, and LangChain.
- Add drift monitoring across data refreshes, geography, model scores, and feature distributions.

These improvements would make the framework stronger for real-world market planning and risk-monitoring scenarios.

## Try It

The project is available on GitHub:

https://github.com/<your-username>/GeoRisk-IQ

Basic setup:

```bash
pip install -r requirements.txt
python download_fdic_files.py
streamlit run app.py
```

Required FDIC files can be downloaded with the included script. HRSA shortage-area data must be added separately if you want to run the healthcare cross-domain mode.

Optional agentic setup:

```bash
pip install -r requirements-agentic.txt
streamlit run app.py
```

Optional agentic stack: CrewAI, LangGraph, LangChain, and Snowflake.

Responsible AI layer: Veritas-style model card, validation checklist, fairness slice plan, and deployment controls.

Advanced AI layer: agentic orchestration, explainability, future RAG/narrative intelligence, Snowflake analytics, MLOps monitoring, privacy/security controls, and responsible AI governance.

Cross-domain layer: implemented for banking, healthcare-tested in another workspace, with extension blueprints for fleet GPS tracking, driver behavior analytics, banking operations improvement, and broader platform intelligence.

## Closing Thought

GeoRisk-IQ started from a simple premise: banking decisions happen in places.

When we combine financial indicators with local market geography, we get a richer view of opportunity and risk. For banks, fintech researchers, and public-data builders, that opens the door to more transparent and spatially aware financial AI.
