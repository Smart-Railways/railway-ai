"""
Railway-AI Maintenance Lifecycle Package.

Provides centralized timezone-aware clock abstractions, domain models,
and the state machine manager for real-time maintenance lifecycle tracking.
"""

from src.lifecycle.clock import (
    Clock,
    MockClock,
    SystemClock,
    get_default_clock,
    set_default_clock,
    to_aware_datetime,
)
from src.lifecycle.models import (
    LifecycleEvent,
    MaintenanceStatus,
    MaintenanceWindow,
    TaskLifecycleRecord,
)
from src.lifecycle.manager import MaintenanceLifecycleManager

__all__ = [
    "Clock",
    "SystemClock",
    "MockClock",
    "get_default_clock",
    "set_default_clock",
    "to_aware_datetime",
    "MaintenanceStatus",
    "MaintenanceWindow",
    "LifecycleEvent",
    "TaskLifecycleRecord",
    "MaintenanceLifecycleManager",
]
