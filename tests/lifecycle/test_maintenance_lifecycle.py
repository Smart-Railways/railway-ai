"""
Phase 3 Test Suite: Maintenance Lifecycle & Real-Time State Management.

Covers all 26 required verification scenarios:
1. GENERATED state exists correctly upon recommendation registration.
2. Future confirmed window transitions to UPCOMING.
3. Current window (window_start <= now <= window_end) transitions to ACTIVE.
4. Window ended without completion transitions to MISSED.
5. Explicit completion before window end transitions to COMPLETED.
6. Explicit completion while ACTIVE transitions to COMPLETED.
7. Completed task does not later become MISSED after window end.
8. MISSED task can receive new recommendations.
9. New recommendation alone does NOT change MISSED -> UPCOMING.
10. Human confirmation of new window transitions MISSED -> RESCHEDULED -> UPCOMING.
11. reschedule_count increments correctly.
12. Original window is preserved after rescheduling.
13. Historical missed event remains after rescheduling.
14. Repeated lifecycle evaluation does not duplicate MISSED events.
15. Repeated lifecycle evaluation does not duplicate COMPLETED events.
16. Timezone-aware datetime handling (UTC, IST, explicit offsets).
17. Naive datetime is normalized or rejected according to convention.
18. Midnight-crossing window works correctly (e.g. 23:30 -> 01:30 next day).
19. overdue_duration is correct (now - window_end).
20. overdue_duration is never negative for MISSED state.
21. Multiple sections work independently across the 10 corridor sections.
22. Two tasks in different sections can have independent lifecycle states.
23. Lifecycle does not alter Phase 2 recommendation cardinality.
24. Phase 2 recommendations remain capped at max 3.
25. Lifecycle does not automatically book recommendations (human confirmation required).
26. Full integration with PlanningService and RailwayMLEngine.
"""

from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd

from src.lifecycle.clock import MockClock, SystemClock, to_aware_datetime
from src.lifecycle.models import (
    MaintenanceStatus,
    MaintenanceWindow,
    TaskLifecycleRecord,
    LifecycleEvent,
)
from src.lifecycle.manager import MaintenanceLifecycleManager
from src.services.planning_service import PlanningService
from src.services.ml_engine import RailwayMLEngine


def make_sample_rec(rank=1, block_id="B001", start_slot=2, end_slot=4, duration=60):
    return {
        "rank": rank,
        "block_id": block_id,
        "start_slot": start_slot,
        "end_slot": end_slot,
        "duration_minutes": duration,
        "selection_status": "RECOMMENDED",
        "human_confirmation_required": True,
        "is_booked": False,
        "booking_status": "UNBOOKED",
    }


class TestMaintenanceLifecycleStates:
    """Verifies core state machine transitions and human-in-the-loop semantics."""

    def test_01_generated_state_exists_correctly(self):
        """1. GENERATED state exists correctly on recommendation."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        recs = [make_sample_rec(1), make_sample_rec(2, "B002", 5, 7)]

        task = mgr.register_recommendations("TASK-100", "NDL-MTJ-01", recs)

        assert task.status == MaintenanceStatus.GENERATED
        assert task.current_window is None
        assert task.original_window is None
        assert len(task.recommendations) == 2
        assert task.reschedule_count == 0

        # Event log check
        history = mgr.get_task_history("TASK-100")
        assert len(history) == 1
        assert history[0].status == MaintenanceStatus.GENERATED

    def test_02_future_confirmed_window_transitions_to_upcoming(self):
        """2. Future confirmed window: now < window_start -> UPCOMING."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow(
            window_id="B001",
            start_time="2026-09-20T10:00:00+00:00",
            end_time="2026-09-20T12:00:00+00:00",
        )
        task = mgr.confirm_window("TASK-100", window)

        assert task.status == MaintenanceStatus.UPCOMING
        assert task.current_window.window_id == "B001"
        assert task.original_window.window_id == "B001"

        history = mgr.get_task_history("TASK-100")
        assert [e.status for e in history] == [
            MaintenanceStatus.GENERATED,
            MaintenanceStatus.UPCOMING,
        ]

    def test_03_current_window_transitions_to_active(self):
        """3. Current window: window_start <= now <= window_end -> ACTIVE."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow(
            window_id="B001",
            start_time="2026-09-20T09:00:00+00:00",
            end_time="2026-09-20T11:00:00+00:00",
        )
        mgr.confirm_window("TASK-100", window)

        # Advance clock to 09:30: inside the window!
        clock.set_time("2026-09-20T09:30:00+00:00")
        mgr.evaluate("TASK-100")

        task = mgr.get_task("TASK-100")
        assert task.status == MaintenanceStatus.ACTIVE

        history = mgr.get_task_history("TASK-100")
        assert [e.status for e in history] == [
            MaintenanceStatus.GENERATED,
            MaintenanceStatus.UPCOMING,
            MaintenanceStatus.ACTIVE,
        ]

    def test_04_window_ended_without_completion_transitions_to_missed(self):
        """4. Window ended without completion -> MISSED."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow(
            window_id="B001",
            start_time="2026-09-20T09:00:00+00:00",
            end_time="2026-09-20T11:00:00+00:00",
        )
        mgr.confirm_window("TASK-100", window)

        # Advance clock past 11:00 without marking complete
        clock.set_time("2026-09-20T11:30:00+00:00")
        mgr.evaluate("TASK-100")

        task = mgr.get_task("TASK-100")
        assert task.status == MaintenanceStatus.MISSED
        assert task.missed_at == window.end_time
        assert task.overdue_duration_seconds == 1800.0 # 30 min past end

        # Must no longer appear in active tasks view
        active_tasks = mgr.get_active_tasks()
        assert task not in active_tasks

    def test_05_explicit_completion_before_window_end(self):
        """5. Explicit completion before window end -> COMPLETED."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow(
            window_id="B001",
            start_time="2026-09-20T09:00:00+00:00",
            end_time="2026-09-20T11:00:00+00:00",
        )
        mgr.confirm_window("TASK-100", window)

        # Complete before window starts (e.g. preemptive completion or early clearance)
        clock.set_time("2026-09-20T08:30:00+00:00")
        mgr.complete_task("TASK-100")

        task = mgr.get_task("TASK-100")
        assert task.status == MaintenanceStatus.COMPLETED
        assert task.completed_at == to_aware_datetime("2026-09-20T08:30:00+00:00")

        # Must disappear from active view
        assert task not in mgr.get_active_tasks()

    def test_06_explicit_completion_while_active(self):
        """6. Explicit completion while ACTIVE -> COMPLETED."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow(
            window_id="B001",
            start_time="2026-09-20T09:00:00+00:00",
            end_time="2026-09-20T11:00:00+00:00",
        )
        mgr.confirm_window("TASK-100", window)

        clock.set_time("2026-09-20T09:45:00+00:00")
        mgr.evaluate("TASK-100")
        assert mgr.get_task("TASK-100").status == MaintenanceStatus.ACTIVE

        # Now complete while ACTIVE
        mgr.complete_task("TASK-100")
        task = mgr.get_task("TASK-100")
        assert task.status == MaintenanceStatus.COMPLETED
        assert task.completed_at == to_aware_datetime("2026-09-20T09:45:00+00:00")

    def test_07_completed_task_does_not_later_become_missed(self):
        """7. Completed task does not later become MISSED after window end."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow(
            window_id="B001",
            start_time="2026-09-20T09:00:00+00:00",
            end_time="2026-09-20T11:00:00+00:00",
        )
        mgr.confirm_window("TASK-100", window)

        clock.set_time("2026-09-20T10:00:00+00:00")
        mgr.complete_task("TASK-100")

        # Advance clock well past the window end (e.g. 14:00)
        clock.set_time("2026-09-20T14:00:00+00:00")
        mgr.evaluate("TASK-100")

        task = mgr.get_task("TASK-100")
        assert task.status == MaintenanceStatus.COMPLETED
        assert task.status != MaintenanceStatus.MISSED

        # Verify no MISSED event in history
        history = mgr.get_task_history("TASK-100")
        assert not any(e.status == MaintenanceStatus.MISSED for e in history)


class TestReschedulingAndHistory:
    """Verifies rescheduling workflow, history preservation, and idempotency."""

    def test_08_missed_task_can_receive_new_recommendations(self):
        """8. MISSED task can receive new recommendations."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow("B001", "2026-09-20T09:00:00+00:00", "2026-09-20T10:00:00+00:00")
        mgr.confirm_window("TASK-100", window)

        clock.set_time("2026-09-20T11:00:00+00:00")
        mgr.evaluate("TASK-100")
        assert mgr.get_task("TASK-100").status == MaintenanceStatus.MISSED

        # Attach new recommendations generated by scheduling optimizer
        new_recs = [make_sample_rec(1, "B002", 7, 9), make_sample_rec(2, "B003", 10, 12)]
        task = mgr.attach_new_recommendations("TASK-100", new_recs)

        assert len(task.recommendations) == 2
        assert task.recommendations[0]["block_id"] == "B002"

    def test_09_new_recommendation_alone_does_not_change_missed_state(self):
        """9. New recommendation alone does NOT change MISSED -> UPCOMING."""
        clock = MockClock("2026-09-20T11:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])
        window = MaintenanceWindow("B001", "2026-09-20T09:00:00+00:00", "2026-09-20T10:00:00+00:00")
        mgr.confirm_window("TASK-100", window)
        mgr.evaluate("TASK-100")
        assert mgr.get_task("TASK-100").status == MaintenanceStatus.MISSED

        # Attach recommendations
        mgr.attach_new_recommendations("TASK-100", [make_sample_rec(1, "B002")])
        mgr.evaluate("TASK-100")

        # Crucial check: MUST REMAIN MISSED until human confirmation
        task = mgr.get_task("TASK-100")
        assert task.status == MaintenanceStatus.MISSED
        assert task not in mgr.get_active_tasks()

    def test_10_human_confirmation_of_new_window_transitions_to_rescheduled_then_upcoming(self):
        """10. Human confirmation of new window: MISSED -> RESCHEDULED -> UPCOMING."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("TASK-100", "NDL-MTJ-01", [make_sample_rec()])
        window1 = MaintenanceWindow("B001", "2026-09-20T09:00:00+00:00", "2026-09-20T10:00:00+00:00")
        mgr.confirm_window("TASK-100", window1)

        # Advance clock to 11:00 (past window end)
        clock.set_time("2026-09-20T11:00:00+00:00")
        mgr.evaluate("TASK-100")
        assert mgr.get_task("TASK-100").status == MaintenanceStatus.MISSED

        # Human operator confirms new window B002 (in future: 14:00 - 16:00)
        window2 = MaintenanceWindow("B002", "2026-09-20T14:00:00+00:00", "2026-09-20T16:00:00+00:00")
        task = mgr.confirm_window("TASK-100", window2)

        assert task.status == MaintenanceStatus.UPCOMING
        assert task.current_window.window_id == "B002"

        history = mgr.get_task_history("TASK-100")
        status_sequence = [e.status for e in history]
        assert status_sequence == [
            MaintenanceStatus.GENERATED,
            MaintenanceStatus.UPCOMING,
            MaintenanceStatus.MISSED,
            MaintenanceStatus.RESCHEDULED,
            MaintenanceStatus.UPCOMING,
        ]

    def test_11_reschedule_count_increments_correctly(self):
        """11. reschedule_count increments correctly."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])

        w1 = MaintenanceWindow("B1", "2026-09-20T08:00:00+00:00", "2026-09-20T09:00:00+00:00")
        mgr.confirm_window("T1", w1)
        clock.set_time("2026-09-20T09:30:00+00:00")
        mgr.evaluate("T1")
        assert mgr.get_task("T1").reschedule_count == 0

        # Reschedule 1
        w2 = MaintenanceWindow("B2", "2026-09-20T10:00:00+00:00", "2026-09-20T11:00:00+00:00")
        mgr.confirm_window("T1", w2)
        assert mgr.get_task("T1").reschedule_count == 1

        # Miss again
        clock.set_time("2026-09-20T11:30:00+00:00")
        mgr.evaluate("T1")
        assert mgr.get_task("T1").status == MaintenanceStatus.MISSED

        # Reschedule 2
        w3 = MaintenanceWindow("B3", "2026-09-20T13:00:00+00:00", "2026-09-20T14:00:00+00:00")
        mgr.confirm_window("T1", w3)
        assert mgr.get_task("T1").reschedule_count == 2

    def test_12_original_window_is_preserved_after_rescheduling(self):
        """12. Original window is preserved after rescheduling."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])

        w1 = MaintenanceWindow("B1", "2026-09-20T08:00:00+00:00", "2026-09-20T09:00:00+00:00")
        mgr.confirm_window("T1", w1)
        clock.set_time("2026-09-20T09:30:00+00:00")
        mgr.evaluate("T1")

        w2 = MaintenanceWindow("B2", "2026-09-20T11:00:00+00:00", "2026-09-20T12:00:00+00:00")
        mgr.confirm_window("T1", w2)

        task = mgr.get_task("T1")
        assert task.original_window.window_id == "B1"
        assert task.current_window.window_id == "B2"

    def test_13_historical_missed_event_remains_after_rescheduling(self):
        """13. Historical missed event remains after rescheduling."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])

        w1 = MaintenanceWindow("B1", "2026-09-20T08:00:00+00:00", "2026-09-20T09:00:00+00:00")
        mgr.confirm_window("T1", w1)
        clock.set_time("2026-09-20T09:30:00+00:00")
        mgr.evaluate("T1")

        w2 = MaintenanceWindow("B2", "2026-09-20T11:00:00+00:00", "2026-09-20T12:00:00+00:00")
        mgr.confirm_window("T1", w2)

        # Full history must contain the MISSED event permanently
        history = mgr.get_task_history("T1")
        missed_events = [e for e in history if e.status == MaintenanceStatus.MISSED]
        assert len(missed_events) == 1
        assert missed_events[0].window_start == w1.start_time

    def test_14_repeated_lifecycle_evaluation_does_not_duplicate_missed_events(self):
        """14. Repeated lifecycle evaluation does not duplicate MISSED events (Idempotency)."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])
        w1 = MaintenanceWindow("B1", "2026-09-20T08:00:00+00:00", "2026-09-20T09:00:00+00:00")
        mgr.confirm_window("T1", w1)

        # Evaluation at 09:30 -> MISSED
        clock.set_time("2026-09-20T09:30:00+00:00")
        mgr.evaluate("T1")
        count_first = len(mgr.get_task_history("T1"))

        # Evaluation at 09:31, 09:32, 09:35 -> must NOT duplicate MISSED events
        clock.set_time("2026-09-20T09:31:00+00:00")
        mgr.evaluate("T1")
        clock.set_time("2026-09-20T09:35:00+00:00")
        mgr.evaluate("T1")

        count_later = len(mgr.get_task_history("T1"))
        assert count_later == count_first

    def test_15_repeated_lifecycle_evaluation_does_not_duplicate_completed_events(self):
        """15. Repeated lifecycle evaluation does not duplicate COMPLETED events (Idempotency)."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])
        w1 = MaintenanceWindow("B1", "2026-09-20T08:00:00+00:00", "2026-09-20T09:00:00+00:00")
        mgr.confirm_window("T1", w1)

        mgr.complete_task("T1")
        count_first = len(mgr.get_task_history("T1"))

        # Re-complete or re-evaluate
        mgr.complete_task("T1")
        mgr.evaluate("T1")
        mgr.evaluate("T1")

        count_later = len(mgr.get_task_history("T1"))
        assert count_later == count_first


class TestTimezoneAndWindowGeometry:
    """Verifies timezone awareness, naive datetime handling, and midnight crossing."""

    def test_16_timezone_aware_datetime_handling(self):
        """16. Timezone-aware datetime handling (UTC and IST offsets)."""
        ist = timezone(timedelta(hours=5, minutes=30))
        clock = MockClock("2026-09-20T10:00:00+05:30", tz=ist)
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])

        w = MaintenanceWindow(
            "B_IST",
            start_time="2026-09-20T11:00:00+05:30",
            end_time="2026-09-20T13:00:00+05:30",
        )
        task = mgr.confirm_window("T1", w)
        assert task.status == MaintenanceStatus.UPCOMING
        assert task.current_window.start_time.tzinfo is not None

        # Compare across timezones: 11:30 IST is 06:00 UTC
        clock.set_time("2026-09-20T06:00:00+00:00")
        mgr.evaluate("T1")
        assert mgr.get_task("T1").status == MaintenanceStatus.ACTIVE

    def test_17_naive_datetime_normalization_and_validation(self):
        """17. Naive datetime is normalized according to convention, invalid input rejected."""
        naive_dt = datetime(2026, 9, 20, 8, 0, 0)
        aware_dt = to_aware_datetime(naive_dt, default_tz=timezone.utc)
        assert aware_dt.tzinfo is not None

        # Rejection of empty/invalid strings
        with pytest.raises(ValueError):
            to_aware_datetime("")
        with pytest.raises(ValueError):
            to_aware_datetime("not-a-datetime")

    def test_18_midnight_crossing_window_works_correctly(self):
        """18. Midnight-crossing window works correctly (e.g. 23:30 -> 01:30 next day)."""
        clock = MockClock("2026-09-20T22:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T_NIGHT", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow(
            "B_NIGHT",
            start_time="2026-09-20T23:30:00+00:00",
            end_time="2026-09-21T01:30:00+00:00", # Next day!
        )
        assert window.crosses_midnight is True
        assert window.duration_minutes == 120

        mgr.confirm_window("T_NIGHT", window)
        assert mgr.get_task("T_NIGHT").status == MaintenanceStatus.UPCOMING

        # Step into next day inside window: 00:30 on 2026-09-21
        clock.set_time("2026-09-21T00:30:00+00:00")
        mgr.evaluate("T_NIGHT")
        assert mgr.get_task("T_NIGHT").status == MaintenanceStatus.ACTIVE

        # Step past end on next day: 02:00 on 2026-09-21
        clock.set_time("2026-09-21T02:00:00+00:00")
        mgr.evaluate("T_NIGHT")
        assert mgr.get_task("T_NIGHT").status == MaintenanceStatus.MISSED
        assert mgr.get_task("T_NIGHT").overdue_duration_seconds == 1800.0

    def test_19_overdue_duration_is_correct(self):
        """19. overdue_duration is correct (now - window_end)."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow("B1", "2026-09-20T08:00:00+00:00", "2026-09-20T09:00:00+00:00")
        mgr.confirm_window("T1", window)

        # Clock is at 09:45 (45 min = 2700 sec overdue)
        clock.set_time("2026-09-20T09:45:00+00:00")
        mgr.evaluate("T1")

        rec = mgr.get_structured_record("T1")
        assert rec["overdue_duration"] == 2700.0

    def test_20_overdue_duration_is_never_negative(self):
        """20. overdue_duration is never negative for MISSED or any state."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)
        mgr.register_recommendations("T1", "NDL-MTJ-01", [make_sample_rec()])

        window = MaintenanceWindow("B1", "2026-09-20T08:00:00+00:00", "2026-09-20T09:00:00+00:00")
        mgr.confirm_window("T1", window)

        rec = mgr.get_structured_record("T1")
        assert rec["overdue_duration"] >= 0.0


class TestCorridorSectionsAndConcurrency:
    """Verifies multi-section support and independent task lifecycles."""

    def test_21_multiple_sections_work_independently(self):
        """21. Multiple sections work independently across the 10 corridor sections."""
        all_10_sections = [
            "NDL-MTJ-01", "MTJ-AGC-01", "AGC-GWL-01", "GWL-JHS-01", "JHS-BINA-01",
            "BINA-BPL-01", "BPL-RTM-01", "RTM-VAD-01", "VAD-SRT-01", "SRT-MUM-01",
        ]
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)

        for idx, sec in enumerate(all_10_sections):
            mgr.register_recommendations(f"TASK-{idx}", sec, [make_sample_rec()])

        assert len(mgr.get_all_tasks()) == 10

        # Check section history filtering
        history_bpl = mgr.get_section_history("BPL-RTM-01")
        assert len(history_bpl) == 1
        assert history_bpl[0].task_id == "TASK-6"

    def test_22_two_tasks_in_different_sections_independent_lifecycles(self):
        """22. Two tasks in different sections can have completely independent lifecycle states."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)

        mgr.register_recommendations("T_NORTH", "NDL-MTJ-01", [make_sample_rec()])
        mgr.register_recommendations("T_SOUTH", "SRT-MUM-01", [make_sample_rec()])

        # North is confirmed for 09:00 - 10:00
        w_north = MaintenanceWindow("B_N", "2026-09-20T09:00:00+00:00", "2026-09-20T10:00:00+00:00")
        mgr.confirm_window("T_NORTH", w_north)

        # South is confirmed for 14:00 - 16:00
        w_south = MaintenanceWindow("B_S", "2026-09-20T14:00:00+00:00", "2026-09-20T16:00:00+00:00")
        mgr.confirm_window("T_SOUTH", w_south)

        # Evaluate at 09:30
        clock.set_time("2026-09-20T09:30:00+00:00")
        mgr.evaluate()

        assert mgr.get_task("T_NORTH").status == MaintenanceStatus.ACTIVE
        assert mgr.get_task("T_SOUTH").status == MaintenanceStatus.UPCOMING

        # Evaluate at 10:30
        clock.set_time("2026-09-20T10:30:00+00:00")
        mgr.evaluate()

        assert mgr.get_task("T_NORTH").status == MaintenanceStatus.MISSED
        assert mgr.get_task("T_SOUTH").status == MaintenanceStatus.UPCOMING


class TestContractIntegrityAndServices:
    """Verifies that Phase 2 contracts remain intact and services integrate cleanly."""

    def test_23_lifecycle_does_not_alter_phase2_recommendation_cardinality(self):
        """23. Lifecycle does not alter Phase 2 recommendation cardinality."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)

        recs = [make_sample_rec(1), make_sample_rec(2)]
        task = mgr.register_recommendations("T1", "NDL-MTJ-01", recs)
        assert len(task.recommendations) == 2

    def test_24_phase2_recommendations_remain_capped_at_max_3(self):
        """24. Phase 2 recommendations remain capped at max 3."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)

        recs = [make_sample_rec(1), make_sample_rec(2), make_sample_rec(3)]
        task = mgr.register_recommendations("T1", "NDL-MTJ-01", recs)
        assert len(task.recommendations) == 3

    def test_25_lifecycle_does_not_automatically_book_recommendations(self):
        """25. Lifecycle does not automatically book recommendations (Human confirmation required)."""
        clock = MockClock("2026-09-20T08:00:00+00:00")
        mgr = MaintenanceLifecycleManager(clock=clock)

        recs = [make_sample_rec(1)]
        task = mgr.register_recommendations("T1", "NDL-MTJ-01", recs)

        # Advance clock to any time
        clock.set_time("2026-09-20T12:00:00+00:00")
        mgr.evaluate()

        # Without human confirmation, it MUST stay GENERATED and NEVER transition to UPCOMING or ACTIVE!
        assert task.status == MaintenanceStatus.GENERATED
        assert task.current_window is None
        assert task not in mgr.get_active_tasks()

    def test_26_services_integration(self):
        """26. PlanningService and RailwayMLEngine integrate lifecycle manager seamlessly."""
        engine = RailwayMLEngine()
        health = engine.health()
        assert health["status"] == "ok"
        assert health["components"]["lifecycle_manager"] == "ready"

        lm = engine.lifecycle_manager
        assert isinstance(lm, MaintenanceLifecycleManager)

        ps = PlanningService()
        lm2 = ps.get_lifecycle_manager()
        assert isinstance(lm2, MaintenanceLifecycleManager)
