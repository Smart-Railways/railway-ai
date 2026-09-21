"""
Data models and schemas for Railway-AI Maintenance Lifecycle Management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from src.lifecycle.clock import to_aware_datetime


class MaintenanceStatus(str, Enum):
    """
    Formal maintenance lifecycle states.

    Target lifecycle:
        GENERATED -> UPCOMING -> ACTIVE -> COMPLETED

    Missed/rescheduled path:
        UPCOMING / ACTIVE -> MISSED -> RESCHEDULED -> UPCOMING -> COMPLETED
    """
    GENERATED = "GENERATED"
    UPCOMING = "UPCOMING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    MISSED = "MISSED"
    RESCHEDULED = "RESCHEDULED"
    CANCELLED = "CANCELLED"


@dataclass
class MaintenanceWindow:
    """
    Represents a specific physical or scheduled block window.
    Supports windows that span across midnight.
    """
    window_id: str
    start_time: datetime
    end_time: datetime
    block_id: Optional[str] = None
    duration_minutes: Optional[int] = None

    def __post_init__(self):
        self.start_time = to_aware_datetime(self.start_time)
        self.end_time = to_aware_datetime(self.end_time)

        if self.end_time <= self.start_time:
            raise ValueError(
                f"Window end_time ({self.end_time.isoformat()}) must be strictly after start_time ({self.start_time.isoformat()})"
            )

        if self.duration_minutes is None:
            self.duration_minutes = int((self.end_time - self.start_time).total_seconds() // 60)

        if self.block_id is None:
            self.block_id = self.window_id

    @property
    def crosses_midnight(self) -> bool:
        """Returns True if the window starts on one calendar date and ends on a later date."""
        return self.end_time.date() > self.start_time.date()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "block_id": self.block_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "crosses_midnight": self.crosses_midnight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaintenanceWindow":
        w_id = data.get("window_id") or data.get("block_id") or "UNKNOWN_WINDOW"
        s_time = data.get("start_time") or data.get("window_start")
        e_time = data.get("end_time") or data.get("window_end")
        if not s_time or not e_time:
            raise ValueError(f"Window dict requires start_time and end_time: {data}")
        return cls(
            window_id=str(w_id),
            start_time=s_time,
            end_time=e_time,
            block_id=str(data.get("block_id", w_id)),
            duration_minutes=data.get("duration_minutes"),
        )


@dataclass
class LifecycleEvent:
    """
    Append-only record of a discrete lifecycle state transition or event.
    """
    event_id: str
    task_id: str
    section_id: str
    status: MaintenanceStatus
    timestamp: datetime
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    missed_at: Optional[datetime] = None
    overdue_duration_seconds: float = 0.0
    reschedule_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "section_id": self.section_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "missed_at": self.missed_at.isoformat() if self.missed_at else None,
            "overdue_duration_seconds": max(0.0, float(self.overdue_duration_seconds)),
            "reschedule_count": self.reschedule_count,
            "details": self.details,
        }


@dataclass
class TaskLifecycleRecord:
    """
    Represents the active state and historical linkages for a maintenance task.
    Exposes Section 14 structured output.
    """
    task_id: str
    section_id: str
    status: MaintenanceStatus
    current_window: Optional[MaintenanceWindow] = None
    original_window: Optional[MaintenanceWindow] = None
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    reschedule_count: int = 0
    completed_at: Optional[datetime] = None
    missed_at: Optional[datetime] = None
    overdue_duration_seconds: float = 0.0
    last_evaluated_at: Optional[datetime] = None

    def to_dict(self, current_timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Produces the standardized structured dictionary required by Phase 3 Section 14.
        """
        curr_ts = current_timestamp or self.last_evaluated_at or datetime.now().astimezone()
        curr_ts = to_aware_datetime(curr_ts)

        return {
            "task_id": self.task_id,
            "section_id": self.section_id,
            "status": self.status.value,
            "window_start": self.current_window.start_time.isoformat() if self.current_window else None,
            "window_end": self.current_window.end_time.isoformat() if self.current_window else None,
            "current_timestamp": curr_ts.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "missed_at": self.missed_at.isoformat() if self.missed_at else None,
            "overdue_duration": max(0.0, float(self.overdue_duration_seconds)),
            "reschedule_count": self.reschedule_count,
            "original_window": self.original_window.to_dict() if self.original_window else None,
            "current_window": self.current_window.to_dict() if self.current_window else None,
        }
