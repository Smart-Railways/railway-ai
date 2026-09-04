"""
Railway ML Engine — Services Layer

Implements the official service entry point for Railway-AI:
- health(): System readiness, versions, component health
- predict(): Maintenance priority, failure probabilities, and operational risk estimation
- generate_block_plan(): Optimization block allocation (6-hr, weekly, monthly, rolling, dynamic)
"""

import os
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from src.decision.maintenance_decision_engine import MaintenanceDecisionEngine
from src.optimization.block_optimizer import BlockOptimizer
from src.optimization.multi_horizon_planner import MultiHorizonPlanner
from src.features.section_pressure_builder import RailKitSectionPressureBuilder
from src.services.planning_service import PlanningService
from src.services.real_corridor_planning import RealCorridorPlanningService
from src.models.failure_predictor import FailureRiskPredictor


class RailwayMLEngine:
    """
    Unified ML & Optimization Service Engine for Railway-AI.
    Combines calibrated risk scoring, real RailKit operational features,
    and CP-SAT multi-horizon block optimization.
    """

    def __init__(
        self,
        decision_engine: Optional[MaintenanceDecisionEngine] = None,
        block_optimizer: Optional[BlockOptimizer] = None,
        multi_horizon_planner: Optional[MultiHorizonPlanner] = None,
        planning_service: Optional[PlanningService] = None,
        real_corridor_planner: Optional[RealCorridorPlanningService] = None,
        failure_predictor: Optional[FailureRiskPredictor] = None,
    ):
        self.version = "0.1.0"
        self.decision_engine = decision_engine or MaintenanceDecisionEngine()
        self.block_optimizer = block_optimizer or BlockOptimizer()
        self.multi_horizon_planner = multi_horizon_planner or MultiHorizonPlanner()
        self.failure_predictor = failure_predictor or FailureRiskPredictor()
        self.planning_service = planning_service or PlanningService(
            decision_engine=self.decision_engine,
            optimizer=self.block_optimizer
        )
        self.real_corridor_planner = real_corridor_planner or RealCorridorPlanningService(
            planning_service=self.planning_service
        )

    def health(self) -> Dict[str, Any]:
        """
        Health and diagnostics endpoint.
        """
        # Check presence of key models and real data directories
        has_models = self.failure_predictor.is_available
        has_real_evidence = os.path.exists("data/processed_real/railkit_section_mapping_evidence.csv")

        return {
            "status": "ok",
            "service": "railway-ml-engine",
            "version": self.version,
            "components": {
                "decision_engine": "ready",
                "failure_predictor": "ready" if has_models else "fallback",
                "block_optimizer": "ready",
                "multi_horizon_planner": "ready",
                "planning_service": "ready",
                "real_corridor_planner": "ready",
            },
            "artifacts": {
                "calibrated_xgboost_available": has_models,
                "real_section_evidence_available": has_real_evidence,
            }
        }

    def predict(
        self,
        input_data: Union[pd.DataFrame, Dict[str, Any], list],
        apply_real_pressure: bool = True
    ) -> pd.DataFrame:
        """
        Predicts maintenance decision scores, ranking, and urgency categories.
        Optionally applies real RailKit section operational pressure.
        Integrates calibrated ML failure risk prediction when raw features are present.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a pandas DataFrame, dict, or list of dicts")

        if df.empty:
            return pd.DataFrame()

        # Step 1: Run Calibrated ML Failure Risk Predictor if telemetry features exist
        # or condition score needs estimation
        if self.failure_predictor.is_available:
            df = self.failure_predictor.predict_risk(df)

        # Step 2: Incorporate live RailKit operational pressure & multi-objective decision scoring
        if apply_real_pressure:
            res = self.real_corridor_planner.score_with_real_pressure(df)
            return res["scored"]
        else:
            return self.decision_engine.transform(df)

    def generate_block_plan(
        self,
        input_data: Union[pd.DataFrame, list],
        horizon_type: str = "daily",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates an optimized maintenance block plan.
        Supports:
        - "daily" / "single": 6-hour standard prototype horizon
        - "weekly": 7-day multi-block schedule
        - "monthly": 30-day multi-block schedule
        - "rolling": lookahead horizon with locked committed days
        - "dynamic": dynamic rescheduling on delay disruptions
        """
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a pandas DataFrame or list of dicts")

        if df.empty:
            return {"plan": pd.DataFrame(), "status": "EMPTY_INPUT"}

        # Normalize required scoring fields
        scored = self.predict(df, apply_real_pressure=kwargs.get("apply_real_pressure", True))

        if horizon_type in ("daily", "single", "6h"):
            block_windows = kwargs.get("block_windows", None)
            plan = self.block_optimizer.optimize(scored, block_windows=block_windows)
            return {
                "horizon_type": "daily_6h",
                "plan": plan,
                "tasks_scheduled": len(plan),
                "status": "OPTIMAL" if not plan.empty else "NO_FEASIBLE_SCHEDULE"
            }

        elif horizon_type == "weekly":
            plan = self.multi_horizon_planner.plan_weekly(scored)
            return {
                "horizon_type": "weekly_7d",
                "plan": plan,
                "tasks_scheduled": len(plan),
                "status": "OPTIMAL" if not plan.empty else "NO_FEASIBLE_SCHEDULE"
            }

        elif horizon_type == "monthly":
            plan = self.multi_horizon_planner.plan_monthly(scored)
            return {
                "horizon_type": "monthly_30d",
                "plan": plan,
                "tasks_scheduled": len(plan),
                "status": "OPTIMAL" if not plan.empty else "NO_FEASIBLE_SCHEDULE"
            }

        elif horizon_type == "rolling":
            res = self.multi_horizon_planner.plan_rolling_horizon(
                tasks=scored,
                horizon_days=kwargs.get("horizon_days", 7),
                committed_days=kwargs.get("committed_days", 2),
                current_committed_plan=kwargs.get("current_committed_plan", None)
            )
            return {
                "horizon_type": "rolling_horizon",
                "plan": res["full_plan"],
                "committed_plan": res["committed_plan"],
                "tentative_plan": res["tentative_plan"],
                "tasks_scheduled": len(res["full_plan"]),
                "status": "OPTIMAL" if not res["full_plan"].empty else "NO_FEASIBLE_SCHEDULE"
            }

        elif horizon_type == "dynamic":
            current_plan = kwargs.get("current_plan", pd.DataFrame())
            disrupted_sections = kwargs.get("disrupted_sections", {})
            current_slot = kwargs.get("current_slot", 0)
            rescheduled = self.multi_horizon_planner.reschedule_dynamically(
                current_plan=current_plan,
                disrupted_sections=disrupted_sections,
                worklist=scored,
                current_slot=current_slot,
                planning_days=kwargs.get("planning_days", 7)
            )
            return {
                "horizon_type": "dynamic_rescheduling",
                "plan": rescheduled,
                "tasks_scheduled": len(rescheduled),
                "disrupted_sections": list(disrupted_sections.keys()),
                "status": "RESCHEDULED" if not rescheduled.empty else "NO_FEASIBLE_SCHEDULE"
            }

        else:
            raise ValueError(f"Unknown horizon_type: {horizon_type}")
