"""
Maintenance Lifecycle Manager & State Machine for Railway-AI.

Lifecycle states:
- GENERATED -> UPCOMING -> ACTIVE -> COMPLETED
- UPCOMING / ACTIVE -> MISSED -> RESCHEDULED -> UPCOMING -> COMPLETED
- Human-in-the-loop confirmation enforcement (no auto-booking)
- Timezone-aware evaluation via centralized Clock
- Append-only event log with idempotency
- Operational view separation (active vs historical)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uuid

from src.lifecycle.clock import Clock, get_default_clock, to_aware_datetime
from src.lifecycle.models import (
    LifecycleEvent,
    MaintenanceStatus,
    MaintenanceWindow,
    TaskLifecycleRecord,
)


class MaintenanceLifecycleManager:
    """
    State machine and lifecycle coordinator for railway maintenance tasks.
    """

    def __init__(self, clock: Optional[Clock] = None):
        self.clock: Clock = clock or get_default_clock()
        self._tasks: Dict[str, TaskLifecycleRecord] = {}
        self._events: List[LifecycleEvent] = []

    def _generate_event_id(self) -> str:
        return f"EVT-{uuid.uuid4().hex[:12].upper()}"

    def _emit_event(
        self,
        task: TaskLifecycleRecord,
        status: MaintenanceStatus,
        timestamp: datetime,
        details: Optional[Dict[str, Any]] = None,
    ) -> LifecycleEvent:
        """
        Emits an append-only lifecycle event.
        Guarantees idempotency: duplicate events for the same status and window are suppressed.
        """
        # Idempotency check against the task's most recent event
        task_events = [e for e in self._events if e.task_id == task.task_id]
        if task_events:
            last_event = task_events[-1]
            if last_event.status == status:
                # If window boundaries match, suppress duplicate event
                curr_w_start = task.current_window.start_time if task.current_window else None
                curr_w_end = task.current_window.end_time if task.current_window else None
                if last_event.window_start == curr_w_start and last_event.window_end == curr_w_end:
                    return last_event

        w_start = task.current_window.start_time if task.current_window else None
        w_end = task.current_window.end_time if task.current_window else None

        event = LifecycleEvent(
            event_id=self._generate_event_id(),
            task_id=task.task_id,
            section_id=task.section_id,
            status=status,
            timestamp=timestamp,
            window_start=w_start,
            window_end=w_end,
            completed_at=task.completed_at,
            missed_at=task.missed_at,
            overdue_duration_seconds=max(0.0, float(task.overdue_duration_seconds)),
            reschedule_count=task.reschedule_count,
            details=details or {},
        )
        self._events.append(event)
        return event

    def register_recommendations(
        self,
        task_id: str,
        section_id: str,
        recommendations: List[Dict[str, Any]],
    ) -> TaskLifecycleRecord:
        """
        Registers candidate recommendations produced by Phase 2.

        Human-in-the-loop semantics:
        - Initial status is GENERATED.
        - current_window is None (NO automatic booking).
        - A human operator must explicitly confirm a window.
        """
        now = self.clock.now()

        # Enforce human-in-the-loop checks on input recommendations
        for rec in recommendations:
            if rec.get("selection_status") != "RECOMMENDED":
                rec["selection_status"] = "RECOMMENDED"
            rec["human_confirmation_required"] = True
            rec["is_booked"] = False
            rec["booking_status"] = "UNBOOKED"

        record = TaskLifecycleRecord(
            task_id=task_id,
            section_id=section_id,
            status=MaintenanceStatus.GENERATED,
            current_window=None,
            original_window=None,
            recommendations=list(recommendations),
            reschedule_count=0,
            completed_at=None,
            missed_at=None,
            overdue_duration_seconds=0.0,
            last_evaluated_at=now,
        )
        self._tasks[task_id] = record
        self._emit_event(record, MaintenanceStatus.GENERATED, now, details={"recommendation_count": len(recommendations)})
        return record

    def confirm_window(
        self,
        task_id: str,
        window: Union[MaintenanceWindow, Dict[str, Any]],
    ) -> TaskLifecycleRecord:
        """
        Explicit human/operator action confirming a maintenance window.

        Transitions:
        - GENERATED -> UPCOMING (or ACTIVE if window is currently active)
        - MISSED -> RESCHEDULED -> UPCOMING (or ACTIVE)
        - UPCOMING -> UPCOMING (window update)
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in lifecycle manager")

        task = self._tasks[task_id]

        if task.status == MaintenanceStatus.COMPLETED:
            raise ValueError(f"Cannot confirm window for already completed task '{task_id}'")

        if isinstance(window, dict):
            m_win = MaintenanceWindow.from_dict(window)
        elif isinstance(window, MaintenanceWindow):
            m_win = window
        else:
            raise TypeError(f"window must be MaintenanceWindow or dict, got {type(window)}")

        now = self.clock.now()
        task.last_evaluated_at = now

        # Case 1: Rescheduling a MISSED task
        if task.status == MaintenanceStatus.MISSED:
            task.reschedule_count += 1
            task.status = MaintenanceStatus.RESCHEDULED
            self._emit_event(
                task,
                MaintenanceStatus.RESCHEDULED,
                now,
                details={
                    "reschedule_count": task.reschedule_count,
                    "previous_window": task.current_window.to_dict() if task.current_window else None,
                    "new_window": m_win.to_dict(),
                },
            )
            task.current_window = m_win
            task.overdue_duration_seconds = 0.0

            # Then transition immediately to UPCOMING or ACTIVE depending on current time
            if now < m_win.start_time:
                task.status = MaintenanceStatus.UPCOMING
                self._emit_event(task, MaintenanceStatus.UPCOMING, now)
            elif now <= m_win.end_time:
                task.status = MaintenanceStatus.ACTIVE
                self._emit_event(task, MaintenanceStatus.ACTIVE, now)
            else:
                task.status = MaintenanceStatus.MISSED
                task.missed_at = m_win.end_time
                task.overdue_duration_seconds = max(0.0, (now - m_win.end_time).total_seconds())
                self._emit_event(task, MaintenanceStatus.MISSED, now)
            return task

        # Case 2: Initial confirmation from GENERATED
        if task.status == MaintenanceStatus.GENERATED:
            task.original_window = m_win
            task.current_window = m_win

            if now < m_win.start_time:
                task.status = MaintenanceStatus.UPCOMING
                self._emit_event(task, MaintenanceStatus.UPCOMING, now)
            elif now <= m_win.end_time:
                task.status = MaintenanceStatus.ACTIVE
                self._emit_event(task, MaintenanceStatus.ACTIVE, now)
            else:
                task.status = MaintenanceStatus.MISSED
                task.missed_at = m_win.end_time
                task.overdue_duration_seconds = max(0.0, (now - m_win.end_time).total_seconds())
                self._emit_event(task, MaintenanceStatus.MISSED, now)
            return task

        # Case 3: Slot modification for UPCOMING task
        if task.status == MaintenanceStatus.UPCOMING:
            task.current_window = m_win
            if now < m_win.start_time:
                task.status = MaintenanceStatus.UPCOMING
                self._emit_event(task, MaintenanceStatus.UPCOMING, now)
            elif now <= m_win.end_time:
                task.status = MaintenanceStatus.ACTIVE
                self._emit_event(task, MaintenanceStatus.ACTIVE, now)
            else:
                task.status = MaintenanceStatus.MISSED
                task.missed_at = m_win.end_time
                task.overdue_duration_seconds = max(0.0, (now - m_win.end_time).total_seconds())
                self._emit_event(task, MaintenanceStatus.MISSED, now)
            return task

        raise ValueError(f"Cannot confirm window for task in state '{task.status.value}'")

    def evaluate(self, task_id: Optional[str] = None) -> List[TaskLifecycleRecord]:
        """
        Evaluates current lifecycle state against centralized clock.now().
        If task_id is None, evaluates all registered tasks.
        """
        now = self.clock.now()
        target_tasks = [self._tasks[task_id]] if task_id else list(self._tasks.values())
        results = []

        for task in target_tasks:
            task.last_evaluated_at = now

            # COMPLETED is terminal: completion takes precedence over missed status
            if task.status == MaintenanceStatus.COMPLETED:
                results.append(task)
                continue

            # GENERATED: Waiting for human confirmation; time cannot change this state
            if task.status == MaintenanceStatus.GENERATED:
                results.append(task)
                continue

            if task.current_window is None:
                results.append(task)
                continue

            w_start = task.current_window.start_time
            w_end = task.current_window.end_time

            # State: UPCOMING
            if task.status == MaintenanceStatus.UPCOMING:
                if now > w_end:
                    task.status = MaintenanceStatus.MISSED
                    task.missed_at = w_end
                    task.overdue_duration_seconds = max(0.0, (now - w_end).total_seconds())
                    self._emit_event(task, MaintenanceStatus.MISSED, now)
                elif now >= w_start:
                    task.status = MaintenanceStatus.ACTIVE
                    self._emit_event(task, MaintenanceStatus.ACTIVE, now)

            # State: ACTIVE
            elif task.status == MaintenanceStatus.ACTIVE:
                if now > w_end:
                    task.status = MaintenanceStatus.MISSED
                    task.missed_at = w_end
                    task.overdue_duration_seconds = max(0.0, (now - w_end).total_seconds())
                    self._emit_event(task, MaintenanceStatus.MISSED, now)

            # State: MISSED (idempotent update of overdue duration)
            elif task.status == MaintenanceStatus.MISSED:
                task.overdue_duration_seconds = max(0.0, (now - w_end).total_seconds())
                # Repeated evaluation does not emit a duplicate MISSED event!

            results.append(task)

        return results

    def complete_task(
        self,
        task_id: str,
        completion_time: Optional[Union[datetime, str]] = None,
    ) -> TaskLifecycleRecord:
        """
        Explicitly marks a maintenance task as COMPLETED.

        - Completion must be explicit (never inferred just because window ended).
        - Idempotent: repeated calls do not duplicate events.
        - Terminal: a completed task can never later become MISSED.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in lifecycle manager")

        task = self._tasks[task_id]

        if task.status == MaintenanceStatus.COMPLETED:
            # Idempotent: already completed, return as-is
            return task

        now = self.clock.now()
        comp_dt = to_aware_datetime(completion_time, default_tz=self.clock.now().tzinfo) if completion_time else now

        task.status = MaintenanceStatus.COMPLETED
        task.completed_at = comp_dt
        task.last_evaluated_at = now
        self._emit_event(task, MaintenanceStatus.COMPLETED, comp_dt)
        return task

    def attach_new_recommendations(
        self,
        task_id: str,
        recommendations: List[Dict[str, Any]],
    ) -> TaskLifecycleRecord:
        """
        Attaches new candidate recommendations to a MISSED task.

        Critical Rule:
        Generating recommendations does NOT mean the task has been rescheduled.
        Status remains MISSED until a human selects a window.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in lifecycle manager")

        task = self._tasks[task_id]

        for rec in recommendations:
            if rec.get("selection_status") != "RECOMMENDED":
                rec["selection_status"] = "RECOMMENDED"
            rec["human_confirmation_required"] = True
            rec["is_booked"] = False
            rec["booking_status"] = "UNBOOKED"

        task.recommendations = list(recommendations)
        # Status remains MISSED!
        return task

    # ------------------------------------------------------------------
    # Query & View Methods (Separating Active/Upcoming vs History)
    # ------------------------------------------------------------------

    def get_active_tasks(self) -> List[TaskLifecycleRecord]:
        """
        Operational view: returns tasks that are currently UPCOMING or ACTIVE.
        COMPLETED and MISSED tasks disappear from this operational view.
        """
        return [
            t for t in self._tasks.values()
            if t.status in (MaintenanceStatus.UPCOMING, MaintenanceStatus.ACTIVE)
        ]

    def get_tasks_by_status(self, status: MaintenanceStatus) -> List[TaskLifecycleRecord]:
        """Returns all tasks currently in the given status."""
        return [t for t in self._tasks.values() if t.status == status]

    def get_all_tasks(self) -> List[TaskLifecycleRecord]:
        """Returns all registered tasks."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Optional[TaskLifecycleRecord]:
        """Retrieves a task by ID."""
        return self._tasks.get(task_id)

    def get_task_history(self, task_id: str) -> List[LifecycleEvent]:
        """Returns the append-only event history for a specific task."""
        return [e for e in self._events if e.task_id == task_id]

    def get_section_history(self, section_id: str) -> List[LifecycleEvent]:
        """Returns the append-only event history for a specific corridor section."""
        return [e for e in self._events if e.section_id == section_id]

    def get_all_events(self) -> List[LifecycleEvent]:
        """Returns the complete append-only lifecycle event log."""
        return list(self._events)

    def get_structured_record(self, task_id: str) -> Dict[str, Any]:
        """
        Returns the standardized structured dictionary required by Phase 3 Section 14.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in lifecycle manager")
        return self._tasks[task_id].to_dict(current_timestamp=self.clock.now())
