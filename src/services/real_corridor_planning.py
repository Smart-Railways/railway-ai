"""
Real Corridor Planning Service — Phase 5.9

End-to-end pipeline that connects real RailKit section pressure
to the existing decision engine and CP-SAT optimizer.

Architecture:
    RailKit captures
        → SectionEvidenceBuilder (5.7)
        → RailKitSectionPressureBuilder (5.8)
        → section pressure dict
        → PlanningService.plan_with_validated_mapping() (existing)
        → optimized block plan

This module orchestrates the full real-data pipeline without
modifying any existing module. It composes existing components.
"""

import pandas as pd

from src.features.section_pressure_builder import (
    RailKitSectionPressureBuilder,
)
from src.services.planning_service import PlanningService


class RealCorridorPlanningService:
    """
    Orchestrates real-data enriched maintenance planning.

    Combines:
        - Real section pressure from RailKit (Phase 5.8)
        - Maintenance decision scoring (existing)
        - CP-SAT block optimization (existing)

    This service ONLY uses sections with verified evidence.
    Unknown sections get no artificial pressure.
    """

    def __init__(
        self,
        pressure_builder: RailKitSectionPressureBuilder = None,
        planning_service: PlanningService = None,
    ):
        self.pressure_builder = (
            pressure_builder or RailKitSectionPressureBuilder()
        )
        self.planning_service = (
            planning_service or PlanningService()
        )

    def plan_with_real_pressure(
        self,
        worklist: pd.DataFrame,
        pressure_override: dict = None,
    ) -> dict:
        """
        Generate an optimized block plan using real section pressure.

        Args:
            worklist: Maintenance task DataFrame with required columns.
            pressure_override: Optional dict to override computed pressure.
                             If provided, uses this instead of computing
                             from RailKit captures.

        Returns:
            dict with:
                "plan": optimized schedule DataFrame
                "section_pressure": {section_id: pressure} used
                "pressure_report": DataFrame with detailed pressure info
                "sections_with_pressure": list of sections that got real pressure
                "sections_without_pressure": list of worklist sections without evidence
                "pipeline_status": summary string
        """
        # Step 1: Get section pressure
        if pressure_override is not None:
            section_pressure = pressure_override
            pressure_report = pd.DataFrame([
                {"section_id": k, "operational_pressure": v, "source": "override"}
                for k, v in section_pressure.items()
            ])
        else:
            section_pressure = self.pressure_builder.get_pressure_dict()
            pressure_report = self.pressure_builder.get_pressure_report()

        # Step 2: Identify coverage
        worklist_sections = set(worklist["section_id"].unique())
        sections_with_pressure = sorted(
            worklist_sections & set(section_pressure.keys())
        )
        sections_without_pressure = sorted(
            worklist_sections - set(section_pressure.keys())
        )

        # Step 3: Plan using validated mapping
        plan = self.planning_service.plan_with_validated_mapping(
            worklist=worklist,
            section_pressure=section_pressure,
        )

        # Step 4: Determine pipeline status
        total = len(worklist_sections)
        covered = len(sections_with_pressure)

        if covered == total:
            status = "FULL_COVERAGE"
        elif covered > 0:
            status = f"PARTIAL_COVERAGE ({covered}/{total} sections)"
        else:
            status = "NO_REAL_PRESSURE (baseline mode)"

        return {
            "plan": plan,
            "section_pressure": section_pressure,
            "pressure_report": pressure_report,
            "sections_with_pressure": sections_with_pressure,
            "sections_without_pressure": sections_without_pressure,
            "pipeline_status": status,
        }

    def score_with_real_pressure(
        self,
        worklist: pd.DataFrame,
        pressure_override: dict = None,
    ) -> dict:
        """
        Score maintenance tasks with real section pressure (without optimization).

        Returns:
            dict with:
                "scored": scored DataFrame
                "section_pressure": {section_id: pressure} used
                "sections_with_pressure": list
                "sections_without_pressure": list
        """
        if pressure_override is not None:
            section_pressure = pressure_override
        else:
            section_pressure = self.pressure_builder.get_pressure_dict()

        df = worklist.copy()
        if "section_id" not in df.columns:
            df["section_id"] = "UNKNOWN"

        worklist_sections = set(df["section_id"].unique())
        sections_with_pressure = sorted(
            worklist_sections & set(section_pressure.keys())
        )
        sections_without_pressure = sorted(
            worklist_sections - set(section_pressure.keys())
        )

        # Apply section pressure to worklist
        df["railkit_operational_pressure"] = (
            df["section_id"]
            .map(section_pressure)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        scored = self.planning_service.decision_engine.transform(df)

        return {
            "scored": scored,
            "section_pressure": section_pressure,
            "sections_with_pressure": sections_with_pressure,
            "sections_without_pressure": sections_without_pressure,
        }
