"""
Pydantic request models for Railway-AI ML API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TaskInput(BaseModel):
    """Single maintenance task input for ML evaluation."""

    task_id: str = Field(..., description="Unique task identifier")
    section_id: str = Field(
        default="NDL-MTJ-01",
        description="Railway corridor section identifier",
    )
    department: str = Field(
        default="ENGINEERING",
        description="Responsible department (ENGINEERING, S&T, TRACTION)",
    )
    estimated_duration: int = Field(
        default=60, ge=1, description="Estimated duration in minutes"
    )
    required_manpower: int = Field(
        default=2, ge=1, description="Required personnel count"
    )
    predicted_delay_minutes: float = Field(
        default=0.0, ge=0.0, description="Predicted delay in minutes"
    )

    # Asset telemetry — the 7 production features (optional)
    asset_age_years: Optional[float] = Field(
        default=None, description="Asset age in years"
    )
    condition_score: Optional[float] = Field(
        default=None, description="Asset condition score"
    )
    criticality: Optional[float] = Field(
        default=None, description="Asset criticality (1-10)"
    )
    usage_factor: Optional[float] = Field(
        default=None, description="Asset usage factor"
    )
    historical_failure_count: Optional[int] = Field(
        default=None, ge=0, description="Count of historical failures"
    )
    historical_downtime_hours: Optional[float] = Field(
        default=None, ge=0.0, description="Total historical downtime in hours"
    )
    days_since_last_failure: Optional[float] = Field(
        default=None, ge=0.0, description="Days since last recorded failure"
    )

    # Precomputed alternatives (when raw telemetry is unavailable)
    failure_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Precomputed failure probability"
    )
    urgency_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Precomputed urgency score"
    )
    overdue_days: Optional[float] = Field(
        default=None, description="Days overdue for maintenance"
    )
    railkit_operational_pressure: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized RailKit operational pressure",
    )


class BlockWindow(BaseModel):
    """Available traffic block window for scheduling."""

    block_id: str = Field(..., description="Block window identifier")
    start_slot: int = Field(..., ge=0, description="Start time slot (inclusive)")
    end_slot: int = Field(..., ge=1, description="End time slot (exclusive)")


class EvaluateRequest(BaseModel):
    """Request body for POST /api/v1/evaluate."""

    tasks: list[TaskInput] = Field(
        ..., min_length=1, description="Maintenance tasks to evaluate"
    )
    block_windows: Optional[list[BlockWindow]] = Field(
        default=None, description="Available block windows (defaults to engine defaults)"
    )
    max_recommendations: int = Field(
        default=3, ge=1, le=3, description="Max scheduling recommendations per task"
    )


class RecommendWindowsRequest(BaseModel):
    """Request body for POST /api/v1/recommend-windows."""

    tasks: list[TaskInput] = Field(
        ..., min_length=1, description="Tasks to schedule"
    )
    block_windows: Optional[list[BlockWindow]] = Field(
        default=None, description="Available block windows"
    )
    max_recommendations: int = Field(
        default=3, ge=1, le=3, description="Max recommendations per task"
    )
