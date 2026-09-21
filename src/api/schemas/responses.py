"""
Pydantic response models for the Railway-AI ML API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service identifier")
    model_available: bool = Field(..., description="Whether calibrated XGBoost model is loaded")
    engine_version: str = Field(..., description="ML engine version")
    components: dict = Field(..., description="Component readiness map")
    artifacts: dict = Field(..., description="Artifact availability map")


# ---------------------------------------------------------------------------
# Evaluation response — matches schemas/ml_response.schema.json
# ---------------------------------------------------------------------------

class Coordination(BaseModel):
    """Cross-department coordination metadata."""

    is_coordinated: bool
    coordination_departments: list[str]
    coordination_reason: str


class RecommendedWindow(BaseModel):
    """Single CP-SAT recommended block window."""

    rank: int = Field(..., ge=1, le=3)
    block_id: str
    start_slot: int = Field(..., ge=0)
    end_slot: int = Field(..., ge=0)
    duration_minutes: int = Field(..., ge=1)
    required_manpower: int = Field(..., ge=1)
    feasibility_status: str
    recommendation_reasons: list[str]
    coordination: Coordination
    selection_status: str  # const "RECOMMENDED"
    human_confirmation_required: bool  # const True
    is_booked: bool  # const False
    booking_status: str  # const "UNBOOKED"


class RiskOutput(BaseModel):
    """Category 1: Calibrated failure risk."""

    failure_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_failure_30d: bool


class ScoreBreakdown(BaseModel):
    """5-factor weighted priority breakdown."""

    failure_risk_factor: float = Field(..., ge=0.0, le=1.0)
    criticality_factor: float = Field(..., ge=0.0, le=1.0)
    urgency_factor: float = Field(..., ge=0.0, le=1.0)
    overdue_factor: float = Field(..., ge=0.0, le=1.0)
    operational_factor: float = Field(..., ge=0.0, le=1.0)
    total_score: float = Field(..., ge=0.0, le=1.0)


class PriorityOutput(BaseModel):
    """Category 2: Priority decision score and category."""

    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_category: str  # LOW | MEDIUM | HIGH | CRITICAL
    score_breakdown: ScoreBreakdown


class ExplanationOutput(BaseModel):
    """Category 3: Human-readable decision rationale."""

    reason_tags: list[str]
    score_breakdown: ScoreBreakdown
    explanation_summary: str


class SchedulingOutput(BaseModel):
    """Category 4: CP-SAT scheduling recommendations."""

    recommended_windows: list[RecommendedWindow]
    total_recommendations: int = Field(..., ge=0, le=3)
    has_feasible_window: bool


class TaskResult(BaseModel):
    """Single evaluated task with all 4 frozen output categories."""

    task_id: str
    section_id: str
    department: str
    decision_rank: int = Field(..., ge=1)
    risk: RiskOutput
    priority: PriorityOutput
    explanation: ExplanationOutput
    scheduling: SchedulingOutput


class EvaluationMetrics(BaseModel):
    """Aggregate metrics for the evaluation batch."""

    tasks_count: int = Field(..., ge=0)
    critical_tasks_count: int = Field(..., ge=0)
    high_tasks_count: int = Field(..., ge=0)
    medium_tasks_count: int = Field(..., ge=0)
    low_tasks_count: int = Field(..., ge=0)
    tasks_with_recommendations: int = Field(..., ge=0)
    infeasible_tasks: list[str]


class EvaluateResponse(BaseModel):
    """Full response for POST /api/v1/evaluate.

    Matches schemas/ml_response.schema.json exactly.
    """

    status: str  # SUCCESS | EMPTY_INPUT | ERROR
    engine_version: str
    tasks_evaluated: int = Field(..., ge=0)
    results: list[TaskResult]
    metrics: EvaluationMetrics


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Structured error response — no stack traces or file paths."""

    error: str = Field(..., description="Error code (e.g. INVALID_REQUEST, INTERNAL_ERROR)")
    detail: str = Field(..., description="Human-readable error message")
    status_code: int
