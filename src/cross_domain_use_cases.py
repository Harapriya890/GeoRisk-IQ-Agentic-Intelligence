"""
cross_domain_use_cases.py

Cross-domain positioning for the GeoRisk-IQ spatial intelligence pipeline.

GeoRisk-IQ started with banking and FDIC validation, but the same pattern can
work anywhere records have latitude/longitude, time, local density, neighboring
activity, and outcome labels. This module documents those target domains for
the dashboard, README, and publishing material.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class UseCase:
    domain: str
    example_data: str
    spatial_unit: str
    target_or_score: str
    business_value: str
    current_status: str


USE_CASES = [
    UseCase(
        domain="Banking",
        example_data="FDIC institution, financial, and failure records",
        spatial_unit="H3-style market cells",
        target_or_score="Branch opportunity score and failure-risk validation label",
        business_value="Branch expansion screening, market saturation analysis, and spatial risk monitoring",
        current_status="implemented",
    ),
    UseCase(
        domain="Healthcare",
        example_data="HRSA Health Professional Shortage Area records tested in a separate workspace",
        spatial_unit="H3-style service-area cells",
        target_or_score="High-shortage area label and shortage opportunity score",
        business_value="Shortage-area prioritization, access planning, and cross-domain validation",
        current_status="pipeline present; external workspace validation completed",
    ),
    UseCase(
        domain="Fleet and logistics",
        example_data="Volvo/fleet GPS-style truck telematics, route traces, stops, and delivery events",
        spatial_unit="Road-network or H3-style mobility cells",
        target_or_score="Route efficiency, dwell-time hotspots, service gaps, and risk zones",
        business_value="Fleet utilization, corridor planning, dispatch optimization, and service reliability",
        current_status="extension blueprint",
    ),
    UseCase(
        domain="Driver behavior",
        example_data="Telematics events such as harsh braking, speeding, cornering, idle time, and fatigue proxies",
        spatial_unit="Route segment, depot zone, or H3-style behavior cell",
        target_or_score="Driver safety score, behavior-risk score, and coaching priority",
        business_value="Driver coaching, safety monitoring, insurance analytics, and operational improvement",
        current_status="extension blueprint",
    ),
    UseCase(
        domain="Banking system improvement",
        example_data="Branch traffic, ATM usage, service incidents, local demand, and spatial complaint patterns",
        spatial_unit="Branch catchment or H3-style customer-service cells",
        target_or_score="Service gap score, operational stress score, and improvement priority",
        business_value="Better branch placement, improved customer access, and operational risk reduction",
        current_status="extension blueprint",
    ),
    UseCase(
        domain="Cross-platform spatial intelligence",
        example_data="Any platform data with coordinates, timestamps, local counts, and measurable outcomes",
        spatial_unit="Reusable H3-style grid or domain-specific spatial segment",
        target_or_score="Domain-specific opportunity, shortage, risk, safety, or utilization score",
        business_value="Reusable analytics layer across finance, healthcare, mobility, logistics, and public services",
        current_status="architecture pattern",
    ),
]


def get_cross_domain_use_cases() -> List[Dict[str, str]]:
    """Return cross-domain use cases as dashboard-friendly dictionaries."""
    return [asdict(use_case) for use_case in USE_CASES]


def cross_domain_summary() -> str:
    """Return a concise summary of the platform strategy."""
    return (
        "GeoRisk-IQ is designed as a reusable spatial intelligence pattern: "
        "load domain records, map them to spatial cells, engineer local and "
        "neighbor features, train or validate domain-specific scores, and "
        "govern the outputs with responsible-AI controls. Banking is "
        "implemented in this workspace, healthcare has been validated in a "
        "separate workspace and is transfer-ready, and fleet GPS/driver "
        "behavior remains a future telemetry connector extension."
    )
