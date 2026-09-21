"""
Railway ML Engine — Services Layer

Implements the official service entry point for Railway-AI:
- health(): System readiness, versions, component health
- predict(): Maintenance priority, failure probabilities, and operational risk estimation
- generate_block_plan(): Optimization block allocation (6-hr, weekly, monthly, rolling, dynamic)
"""

import os
import json
from pathlib import Path
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
        lifecycle_manager: Optional[Any] = None,
    ):
        self.version = "0.1.0"
        self.decision_engine = decision_engine or MaintenanceDecisionEngine()
        self.block_optimizer = block_optimizer or BlockOptimizer()
        self.multi_horizon_planner = multi_horizon_planner or MultiHorizonPlanner()
        self.failure_predictor = failure_predictor or FailureRiskPredictor()
        self.planning_service = planning_service or PlanningService(
            decision_engine=self.decision_engine,
            optimizer=self.block_optimizer,
            lifecycle_manager=lifecycle_manager,
        )
        self.real_corridor_planner = real_corridor_planner or RealCorridorPlanningService(
            planning_service=self.planning_service
        )

    @property
    def lifecycle_manager(self):
        """Access the MaintenanceLifecycleManager."""
        return self.planning_service.get_lifecycle_manager()

    def health(self) -> Dict[str, Any]:
        """
        Health and diagnostics endpoint.
        """
        # Check presence of key models and real data directories
        has_models = self.failure_predictor.is_available
        evidence_path = Path(__file__).resolve().parents[2] / "data/processed_real/railkit_section_mapping_evidence.csv"
        has_real_evidence = os.path.exists("data/processed_real/railkit_section_mapping_evidence.csv") or evidence_path.exists()

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
                "lifecycle_manager": "ready",
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

        # Step 1: Run Calibrated ML Failure Risk Predictor if asset features exist
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

    def recommend_windows(
        self,
        input_data: Union[pd.DataFrame, list],
        block_windows: Optional[list] = None,
        max_recommendations: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates ranked candidate block window recommendations per maintenance task.

        Phase 2 Semantics:
        CP-SAT is strictly a recommendation engine. It produces at most max_recommendations
        (default 3) ranked feasible block windows per task (labeled rank 1, 2, 3).
        Final maintenance slot selection is made by the human operator.
        """
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a pandas DataFrame or list of dicts")

        if df.empty:
            return self.block_optimizer.optimize_with_recommendations(
                pd.DataFrame(),
                block_windows=block_windows,
                max_recommendations=max_recommendations,
            )

        return self.planning_service.recommend_windows(
            df,
            block_windows=block_windows,
            max_recommendations=max_recommendations,
        )

    def evaluate_for_backend(
        self,
        input_data: Union[pd.DataFrame, list, dict],
        block_windows: Optional[list] = None,
        apply_real_pressure: bool = True,
        max_recommendations: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Standardized Pre-Backend-Integration Contract Endpoint.

        Evaluates incoming maintenance tasks and produces a clean,
        pure-JSON-serializable payload adhering strictly to the 4 frozen
        output categories:
        1. RISK: Calibrated failure probability and 30-day failure classification.
        2. PRIORITY: 5-factor priority score, priority category, and score breakdown.
        3. EXPLANATION: Transparent rule-based reason codes and breakdown summary.
        4. SCHEDULING: Up to 3 ranked feasible block window recommendations with
           cross-department coordination and human-in-the-loop invariants.

        Excludes all misleading heuristic confidence metrics.
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
            return {
                "status": "EMPTY_INPUT",
                "engine_version": self.version,
                "tasks_evaluated": 0,
                "results": [],
                "metrics": {
                    "tasks_count": 0,
                    "critical_tasks_count": 0,
                    "high_tasks_count": 0,
                    "medium_tasks_count": 0,
                    "low_tasks_count": 0,
                    "tasks_with_recommendations": 0,
                    "infeasible_tasks": [],
                },
            }

        # Normalize required task identity & scheduling fields
        if "task_id" not in df.columns:
            df["task_id"] = [f"TASK-{i+1:03d}" for i in range(len(df))]
        else:
            df["task_id"] = df["task_id"].astype(str)

        if "section_id" not in df.columns:
            df["section_id"] = "NDL-MTJ-01"
        else:
            df["section_id"] = df["section_id"].astype(str)

        if "department" not in df.columns:
            df["department"] = "ENGINEERING"
        else:
            df["department"] = df["department"].astype(str)

        if "estimated_duration" not in df.columns:
            df["estimated_duration"] = 60
        else:
            df["estimated_duration"] = pd.to_numeric(df["estimated_duration"], errors="coerce").fillna(60).astype(int)

        if "required_manpower" not in df.columns:
            df["required_manpower"] = 2
        else:
            df["required_manpower"] = pd.to_numeric(df["required_manpower"], errors="coerce").fillna(2).astype(int)

        if "predicted_delay_minutes" not in df.columns:
            df["predicted_delay_minutes"] = 0.0
        else:
            df["predicted_delay_minutes"] = pd.to_numeric(df["predicted_delay_minutes"], errors="coerce").fillna(0.0).astype(float)

        # Step 1 & 2: Risk prediction & Multi-objective Priority scoring
        scored = self.predict(df, apply_real_pressure=apply_real_pressure)

        # Step 3: CP-SAT Block Window Recommendations
        rec_res = self.recommend_windows(
            scored,
            block_windows=block_windows,
            max_recommendations=max_recommendations,
        )
        recommendations_map = rec_res.get("recommendations", {})

        results = []
        for idx, row in scored.iterrows():
            task_id = str(row["task_id"])
            raw_recs = recommendations_map.get(task_id, [])

            # Category 1: RISK
            failure_prob = round(float(row.get("failure_probability", 0.0)), 4)
            threshold = float(getattr(self.failure_predictor, "threshold", 0.0120))
            pred_fail_30d = bool(
                row.get("predicted_failure_30d", False)
                or (failure_prob >= threshold)
            )
            risk_data = {
                "failure_probability": failure_prob,
                "predicted_failure_30d": pred_fail_30d,
            }

            # Category 2: PRIORITY
            bd_raw = row.get("score_breakdown")
            if isinstance(bd_raw, str):
                try:
                    bd = json.loads(bd_raw)
                except Exception:
                    bd = {}
            elif isinstance(bd_raw, dict):
                bd = bd_raw
            else:
                bd = {}

            score_breakdown = {
                "failure_risk_factor": round(float(bd.get("failure_risk_factor", row.get("failure_risk_factor", 0.0))), 4),
                "criticality_factor": round(float(bd.get("criticality_factor", row.get("criticality_factor", 0.0))), 4),
                "urgency_factor": round(float(bd.get("urgency_factor", row.get("urgency_factor", 0.0))), 4),
                "overdue_factor": round(float(bd.get("overdue_factor", row.get("overdue_factor", 0.0))), 4),
                "operational_factor": round(float(bd.get("operational_factor", row.get("operational_factor", 0.0))), 4),
                "total_score": round(float(bd.get("total", row.get("maintenance_decision_score", 0.0))), 4),
            }

            priority_score = round(float(row.get("maintenance_decision_score", 0.0)), 4)
            priority_cat = str(row.get("decision_category", "LOW"))

            priority_data = {
                "priority_score": priority_score,
                "priority_category": priority_cat,
                "score_breakdown": score_breakdown,
            }

            # Category 3: EXPLANATION
            reasons_str = str(row.get("decision_reasons", ""))
            reason_tags = [tag.strip() for tag in reasons_str.split(";") if tag.strip()]
            if not reason_tags:
                reason_tags = ["ROUTINE_INSPECTION"]

            explanation_summary = (
                f"Task {task_id} prioritized as {priority_cat} (score: {priority_score:.4f}) "
                f"driven by: {', '.join(reason_tags)}."
            )

            explanation_data = {
                "reason_tags": reason_tags,
                "score_breakdown": score_breakdown,
                "explanation_summary": explanation_summary,
            }

            # Category 4: SCHEDULING
            clean_recs = []
            for r in raw_recs:
                clean_recs.append({
                    "rank": int(r["rank"]),
                    "block_id": str(r["block_id"]),
                    "start_slot": int(r["start_slot"]),
                    "end_slot": int(r["end_slot"]),
                    "duration_minutes": int(r["duration_minutes"]),
                    "required_manpower": int(r.get("required_manpower", row.get("required_manpower", 2))),
                    "feasibility_status": str(r.get("feasibility_status", "FEASIBLE")),
                    "recommendation_reasons": [str(x) for x in r.get("recommendation_reasons", [])],
                    "coordination": {
                        "is_coordinated": bool(r.get("coordination", {}).get("is_coordinated", False)),
                        "coordination_departments": [str(d) for d in r.get("coordination", {}).get("coordination_departments", [])],
                        "coordination_reason": str(r.get("coordination", {}).get("coordination_reason", "No cross-department coordination")),
                    },
                    "selection_status": "RECOMMENDED",
                    "human_confirmation_required": True,
                    "is_booked": False,
                    "booking_status": "UNBOOKED",
                })

            scheduling_data = {
                "recommended_windows": clean_recs,
                "total_recommendations": len(clean_recs),
                "has_feasible_window": len(clean_recs) > 0,
            }

            results.append({
                "task_id": task_id,
                "section_id": str(row.get("section_id", "UNKNOWN")),
                "department": str(row.get("department", "ENGINEERING")),
                "decision_rank": int(row.get("decision_rank", idx + 1)),
                "risk": risk_data,
                "priority": priority_data,
                "explanation": explanation_data,
                "scheduling": scheduling_data,
            })

        metrics = {
            "tasks_count": len(results),
            "critical_tasks_count": sum(1 for r in results if r["priority"]["priority_category"] == "CRITICAL"),
            "high_tasks_count": sum(1 for r in results if r["priority"]["priority_category"] == "HIGH"),
            "medium_tasks_count": sum(1 for r in results if r["priority"]["priority_category"] == "MEDIUM"),
            "low_tasks_count": sum(1 for r in results if r["priority"]["priority_category"] == "LOW"),
            "tasks_with_recommendations": sum(1 for r in results if r["scheduling"]["total_recommendations"] > 0),
            "infeasible_tasks": [r["task_id"] for r in results if r["scheduling"]["total_recommendations"] == 0],
        }

        return {
            "status": "SUCCESS",
            "engine_version": self.version,
            "tasks_evaluated": len(results),
            "results": results,
            "metrics": metrics,
        }
