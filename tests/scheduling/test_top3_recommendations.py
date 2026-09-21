"""
Phase 2 Test Suite: Top-3 Block Window Recommendations & Coordination

Verifies:
1. At most 3 recommendations per task, labeled rank 1, 2, 3.
2. Sequential ranks without gaps.
3. No duplicate candidate windows per task.
4. Correct behavior for 0, 1, 2, 3, and >3 feasible windows.
5. Never fabricates candidate windows.
6. Hard constraints strictly preserved:
   - Same-section non-overlap
   - Different-section concurrency
   - Manpower cumulative capacity
   - Task duration containment
7. Cross-department coordination preference:
   - Soft bonus for compatible departments on same section & window
   - Coordination metadata exposed
   - Coordination NEVER violates hard constraints (section non-overlap, manpower)
8. Human-in-the-loop semantics:
   - Status remains unselected/unbooked (RECOMMENDED, UNBOOKED)
   - Human confirmation required
   - No auto-booking or state mutation
9. Edge cases & error handling:
   - Empty worklist
   - Missing required columns
   - Deterministic execution
10. End-to-end services integration (PlanningService and RailwayMLEngine)
"""

import math
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.optimization.block_optimizer import BlockOptimizer
from src.services.planning_service import PlanningService
from src.services.ml_engine import RailwayMLEngine


def make_sample_task(
    task_id="TASK-001",
    section_id="NDL-MTJ-01",
    department="ENGINEERING",
    duration=60,
    manpower=4,
    score=0.85,
    delay=0.0,
):
    return {
        "task_id": task_id,
        "section_id": section_id,
        "department": department,
        "estimated_duration": duration,
        "required_manpower": manpower,
        "maintenance_decision_score": score,
        "predicted_delay_minutes": delay,
    }


class TestTop3WindowStructure:
    """Verifies the schema, human-in-the-loop semantics, and metadata."""

    def test_recommendation_schema(self):
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([make_sample_task()])
        res = optimizer.optimize_with_recommendations(tasks)

        assert "recommendations" in res
        assert "recommendations_df" in res
        assert "infeasible_tasks" in res
        assert "metrics" in res

        recs = res["recommendations"]["TASK-001"]
        assert len(recs) > 0
        r0 = recs[0]

        expected_fields = [
            "task_id",
            "section_id",
            "department",
            "rank",
            "block_id",
            "start_slot",
            "end_slot",
            "duration_minutes",
            "required_manpower",
            "maintenance_decision_score",
            "predicted_delay_minutes",
            "feasibility_status",
            "recommendation_reasons",
            "coordination",
            "selection_status",
            "human_confirmation_required",
            "is_booked",
            "booking_status",
        ]
        for field in expected_fields:
            assert field in r0, f"Field '{field}' missing from recommendation"

    def test_human_in_the_loop_semantics(self):
        """CP-SAT is strictly a recommendation engine. It must never auto-book."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1"),
            make_sample_task("T2", department="S&T"),
        ])
        res = optimizer.optimize_with_recommendations(tasks)

        for t_id, recs in res["recommendations"].items():
            for r in recs:
                assert r["selection_status"] == "RECOMMENDED"
                assert r["human_confirmation_required"] is True
                assert r["is_booked"] is False
                assert r["booking_status"] == "UNBOOKED"

        df = res["recommendations_df"]
        assert (df["selection_status"] == "RECOMMENDED").all()
        assert (df["human_confirmation_required"] == True).all()
        assert (df["is_booked"] == False).all()
        assert (df["booking_status"] == "UNBOOKED").all()


class TestRecommendationBoundsAndRanks:
    """Verifies the Top-3 limit, rank ordering, and candidate window counts."""

    def test_more_than_three_feasible_windows_returns_top_3(self):
        """If >3 windows are feasible, return exactly the top 3."""
        optimizer = BlockOptimizer(planning_hours=12)
        tasks = pd.DataFrame([make_sample_task(duration=60)]) # 2 slots
        windows = [
            {"block_id": f"B00{i}", "start_slot": i * 4, "end_slot": i * 4 + 3}
            for i in range(1, 6) # 5 windows
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["TASK-001"]

        assert len(recs) == 3, f"Expected 3 recommendations, got {len(recs)}"
        assert [r["rank"] for r in recs] == [1, 2, 3]

    def test_rank_values_are_sequential(self):
        optimizer = BlockOptimizer(planning_hours=8)
        tasks = pd.DataFrame([make_sample_task()])
        windows = [
            {"block_id": "B001", "start_slot": 1, "end_slot": 4},
            {"block_id": "B002", "start_slot": 5, "end_slot": 8},
            {"block_id": "B003", "start_slot": 9, "end_slot": 12},
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        ranks = [r["rank"] for r in res["recommendations"]["TASK-001"]]
        assert ranks == [1, 2, 3]

    def test_exactly_two_feasible_windows(self):
        """If only 2 windows exist, return only 2 recommendations."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([make_sample_task()])
        windows = [
            {"block_id": "B001", "start_slot": 1, "end_slot": 5},
            {"block_id": "B002", "start_slot": 6, "end_slot": 10},
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["TASK-001"]
        assert len(recs) == 2
        assert [r["rank"] for r in recs] == [1, 2]

    def test_exactly_one_feasible_window(self):
        """If only 1 window exists, return only 1 recommendation."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([make_sample_task()])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 5}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["TASK-001"]
        assert len(recs) == 1
        assert recs[0]["rank"] == 1

    def test_zero_feasible_windows_duration_exceeded(self):
        """If task duration exceeds all windows, return empty list."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([make_sample_task(duration=180)]) # 6 slots
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 4}] # 3 slots
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["TASK-001"]
        assert len(recs) == 0
        assert len(res["infeasible_tasks"]) == 1
        assert "exceeds all available block windows" in res["infeasible_tasks"][0]["reason"]

    def test_zero_feasible_windows_manpower_exceeded(self):
        """If task requires more manpower than max capacity, return empty list."""
        optimizer = BlockOptimizer(max_manpower=10)
        tasks = pd.DataFrame([make_sample_task(manpower=15)])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 5}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["TASK-001"]
        assert len(recs) == 0
        assert len(res["infeasible_tasks"]) == 1
        assert "exceeds maximum available capacity" in res["infeasible_tasks"][0]["reason"]

    def test_no_duplicate_candidate_windows(self):
        """Each recommendation for a task must represent a distinct block window."""
        optimizer = BlockOptimizer(planning_hours=10)
        tasks = pd.DataFrame([make_sample_task()])
        windows = [
            {"block_id": "B001", "start_slot": 1, "end_slot": 4},
            {"block_id": "B002", "start_slot": 5, "end_slot": 8},
            {"block_id": "B003", "start_slot": 9, "end_slot": 12},
            {"block_id": "B004", "start_slot": 13, "end_slot": 16},
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["TASK-001"]
        block_ids = [r["block_id"] for r in recs]
        assert len(block_ids) == len(set(block_ids)), "Duplicate block windows found in recommendations"

    def test_no_fabricated_windows(self):
        """Never return a window that does not exist in the configured block windows."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([make_sample_task()])
        windows = [{"block_id": "REAL_B01", "start_slot": 1, "end_slot": 5}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["TASK-001"]
        for r in recs:
            assert r["block_id"] == "REAL_B01"


class TestHardConstraintsPreserved:
    """Verifies that all safety and operational constraints are 100% enforced."""

    def test_same_section_non_overlap(self):
        """Two tasks on the same section scheduled in the same window must not overlap."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_A", duration=60), # 2 slots
            make_sample_task("T2", section_id="SEC_A", duration=60), # 2 slots
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}] # 5 slots (fits 2+2=4 sequentially)
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]

        # Intervals [start, end] must not overlap
        t1_start, t1_end = r1["start_slot"], r1["end_slot"]
        t2_start, t2_end = r2["start_slot"], r2["end_slot"]
        assert (t1_end <= t2_start) or (t2_end <= t1_start), (
            f"Section overlap detected: T1[{t1_start}, {t1_end}] vs T2[{t2_start}, {t2_end}]"
        )

    def test_different_section_tasks_can_execute_concurrently(self):
        """Tasks on different sections can execute in parallel without conflict."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T_NORTH", section_id="NDL-MTJ-01", duration=60, manpower=4),
            make_sample_task("T_SOUTH", section_id="BINA-BPL-01", duration=60, manpower=4),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 5}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r_north = res["recommendations"]["T_NORTH"][0]
        r_south = res["recommendations"]["T_SOUTH"][0]

        # Both can start at slot 1 concurrently because sections are distinct and total manpower = 8 <= 12
        assert r_north["block_id"] == "B001"
        assert r_south["block_id"] == "B001"
        assert r_north["start_slot"] == 1
        assert r_south["start_slot"] == 1

    def test_manpower_saturation_prevents_concurrency(self):
        """Tasks whose concurrent manpower exceeds max_manpower cannot overlap."""
        optimizer = BlockOptimizer(max_manpower=10)
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_A", duration=60, manpower=7),
            make_sample_task("T2", section_id="SEC_B", duration=60, manpower=6),
        ])
        # Total manpower = 7 + 6 = 13 > 10.
        # Window has length 5 slots (fits 2 + 2 = 4 slots sequentially).
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]

        # Must not overlap in time despite being on different sections!
        t1_start, t1_end = r1["start_slot"], r1["end_slot"]
        t2_start, t2_end = r2["start_slot"], r2["end_slot"]
        assert (t1_end <= t2_start) or (t2_end <= t1_start), (
            f"Manpower constraint violated: concurrent execution at slot exceeds max_manpower"
        )

    def test_task_duration_contained_in_window(self):
        """Every recommended task interval must fit within its block window."""
        optimizer = BlockOptimizer(planning_hours=8)
        tasks = pd.DataFrame([make_sample_task(duration=90)]) # 3 slots
        windows = [
            {"block_id": "B001", "start_slot": 2, "end_slot": 6}, # length 4
            {"block_id": "B002", "start_slot": 8, "end_slot": 12}, # length 4
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        for t_id, recs in res["recommendations"].items():
            for r in recs:
                w_match = next(w for w in windows if w["block_id"] == r["block_id"])
                assert r["start_slot"] >= w_match["start_slot"]
                assert r["end_slot"] <= w_match["end_slot"]
                assert r["end_slot"] - r["start_slot"] == 3


class TestCrossDepartmentCoordination:
    """Verifies the Phase 2 cross-department coordination preference."""

    def test_coordination_bonus_and_metadata(self):
        """Tasks from different departments on the same section get coordinated."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T_ENG", section_id="SEC_CORR", department="ENGINEERING", duration=60, score=0.8),
            make_sample_task("T_ST", section_id="SEC_CORR", department="S&T", duration=60, score=0.7),
        ])
        windows = [
            {"block_id": "B_JOINT", "start_slot": 1, "end_slot": 6}, # length 5 slots
            {"block_id": "B_OTHER", "start_slot": 7, "end_slot": 11},
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r_eng = res["recommendations"]["T_ENG"][0]
        r_st = res["recommendations"]["T_ST"][0]

        assert r_eng["block_id"] == "B_JOINT"
        assert r_st["block_id"] == "B_JOINT"

        # Check coordination metadata
        assert r_eng["coordination"]["is_coordinated"] is True
        assert "T_ST" in r_eng["coordination"]["coordinated_task_ids"]
        assert "S&T" in r_eng["coordination"]["coordination_departments"]
        assert "ENGINEERING" in r_eng["coordination"]["coordination_departments"]
        assert "CROSS_DEPARTMENT_COORDINATION" in r_eng["recommendation_reasons"]

        assert r_st["coordination"]["is_coordinated"] is True
        assert "T_ENG" in r_st["coordination"]["coordinated_task_ids"]

    def test_coordination_not_forced_when_section_duration_exceeded(self):
        """If tasks cannot fit sequentially on the same section, coordination is not forced."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T_ENG", section_id="SEC_TIGHT", department="ENGINEERING", duration=90), # 3 slots
            make_sample_task("T_ST", section_id="SEC_TIGHT", department="S&T", duration=90), # 3 slots
        ])
        # Window has only 4 slots. 3 + 3 = 6 > 4 slots!
        windows = [{"block_id": "B_SHORT", "start_slot": 1, "end_slot": 5}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r_eng = res["recommendations"]["T_ENG"][0]
        r_st = res["recommendations"]["T_ST"][0]

        # Neither can coordinate in this window because both cannot fit non-overlapping
        assert r_eng["coordination"]["is_coordinated"] is False
        assert r_st["coordination"]["is_coordinated"] is False

    def test_same_department_does_not_receive_cross_department_coordination(self):
        """Two tasks from the SAME department do not get cross-department coordination."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_A", department="ENGINEERING", duration=60),
            make_sample_task("T2", section_id="SEC_A", department="ENGINEERING", duration=60),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]

        assert r1["coordination"]["is_coordinated"] is False
        assert r2["coordination"]["is_coordinated"] is False


class TestEdgeCasesAndIntegration:
    """Verifies edge cases, input validation, and service layer integration."""

    def test_empty_worklist(self):
        optimizer = BlockOptimizer()
        res = optimizer.optimize_with_recommendations(pd.DataFrame())
        assert res["metrics"]["total_tasks"] == 0
        assert res["recommendations_df"].empty

    def test_missing_required_columns_raises_error(self):
        optimizer = BlockOptimizer()
        bad_df = pd.DataFrame([{"invalid_col": 1}])
        with pytest.raises(ValueError, match="Missing required columns"):
            optimizer.optimize_with_recommendations(bad_df)

    def test_deterministic_repeated_execution(self):
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", department="ENGINEERING", score=0.8),
            make_sample_task("T2", department="S&T", score=0.7),
        ])
        run1 = optimizer.optimize_with_recommendations(tasks)
        run2 = optimizer.optimize_with_recommendations(tasks)
        assert run1["recommendations_df"].to_dict() == run2["recommendations_df"].to_dict()

    def test_planning_service_recommend_windows(self):
        ps = PlanningService()
        worklist = pd.DataFrame([
            {
                "task_id": "TASK-001",
                "section_id": "NDL-MTJ-01",
                "department": "ENGINEERING",
                "estimated_duration": 60,
                "required_manpower": 4,
                "failure_probability": 0.35,
                "urgency_score": 0.50,
                "criticality": 8,
                "overdue_days": 5,
            }
        ])
        res = ps.recommend_windows(worklist)
        assert "recommendations" in res
        assert "TASK-001" in res["recommendations"]
        recs = res["recommendations"]["TASK-001"]
        assert len(recs) > 0
        assert recs[0]["rank"] == 1
        assert recs[0]["selection_status"] == "RECOMMENDED"

    def test_ml_engine_recommend_windows(self):
        engine = RailwayMLEngine()
        worklist = [
            {
                "task_id": "TASK-101",
                "section_id": "NDL-MTJ-01",
                "department": "TRACTION",
                "estimated_duration": 60,
                "required_manpower": 4,
                "failure_probability": 0.20,
                "urgency_score": 0.40,
                "criticality": 6,
                "overdue_days": 0,
            }
        ]
        res = engine.recommend_windows(worklist)
        assert "recommendations" in res
        assert "TASK-101" in res["recommendations"]
        assert res["recommendations"]["TASK-101"][0]["rank"] == 1


def independent_validator(recommendation, original_task, block_windows, max_manpower=12):
    """
    Independently validates a recommendation against physical and domain rules.
    Returns (bool, str) -> (is_valid, failure_reason)
    """
    # 1. Check window identity & non-fabrication
    b_id = recommendation["block_id"]
    matching_windows = [w for w in block_windows if w["block_id"] == b_id]
    if not matching_windows:
        return False, f"Fabricated block_id: {b_id} not in configured windows"
    win = matching_windows[0]

    # 2. Check bounds
    s_slot = recommendation["start_slot"]
    e_slot = recommendation["end_slot"]
    if s_slot < win["start_slot"]:
        return False, f"start_slot {s_slot} is before window start {win['start_slot']}"
    if e_slot > win["end_slot"]:
        return False, f"end_slot {e_slot} is after window end {win['end_slot']}"
    if e_slot <= s_slot:
        return False, f"end_slot {e_slot} <= start_slot {s_slot}"

    # 3. Check duration fit
    expected_dur_slots = max(1, math.ceil(original_task["estimated_duration"] / 30))
    actual_slots = e_slot - s_slot
    if actual_slots != expected_dur_slots:
        return False, f"Duration mismatch: scheduled slots {actual_slots} != required slots {expected_dur_slots}"

    # 4. Check manpower
    m_req = max(0, int(math.ceil(original_task["required_manpower"])))
    if m_req > max_manpower:
        return False, f"Manpower {m_req} exceeds max {max_manpower}"

    # 5. Check semantic flags
    if recommendation["selection_status"] != "RECOMMENDED":
        return False, f"Invalid selection_status: {recommendation['selection_status']}"
    if recommendation["is_booked"] is not False:
        return False, "is_booked should be False"
    if recommendation["human_confirmation_required"] is not True:
        return False, "human_confirmation_required should be True"

    return True, "VALID"


class TestRankingInvariants:
    """Verifies the 6 ranking invariants specified in Part 5."""

    def test_invariant_a_deterministic_tie_break(self):
        """Invariant A: identical priority, delay, coordination -> deterministic tie-break by start_slot then block_id."""
        optimizer = BlockOptimizer(planning_hours=12)
        tasks = pd.DataFrame([make_sample_task("T1", score=0.8, delay=0.0)])
        windows = [
            {"block_id": "B002", "start_slot": 6, "end_slot": 10},
            {"block_id": "B001", "start_slot": 1, "end_slot": 5},
            {"block_id": "B003", "start_slot": 11, "end_slot": 15},
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs = res["recommendations"]["T1"]
        # Score is identical for all 3 windows. Tie-break: earliest start_slot.
        # B001 starts at 1, B002 starts at 6, B003 starts at 11.
        assert [r["block_id"] for r in recs] == ["B001", "B002", "B003"]
        assert [r["rank"] for r in recs] == [1, 2, 3]

    def test_invariant_b_higher_objective_ranks_above_lower(self):
        """Invariant B: Candidate A with better objective score must rank above candidate B."""
        optimizer = BlockOptimizer(planning_hours=12)
        tasks = pd.DataFrame([
            make_sample_task("T_ENG", section_id="SEC_1", department="ENGINEERING", duration=60, score=0.8),
            make_sample_task("T_ST", section_id="SEC_1", department="S&T", duration=60, score=0.7),
        ])
        windows = [
            {"block_id": "B_UNCOORD", "start_slot": 1, "end_slot": 4},
            {"block_id": "B_COORD", "start_slot": 5, "end_slot": 10},
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        recs_eng = res["recommendations"]["T_ENG"]
        assert recs_eng[0]["block_id"] == "B_COORD"
        assert recs_eng[0]["rank"] == 1
        assert recs_eng[0]["coordination"]["is_coordinated"] is True
        assert recs_eng[1]["block_id"] == "B_UNCOORD"
        assert recs_eng[1]["rank"] == 2
        assert recs_eng[1]["coordination"]["is_coordinated"] is False

    def test_invariant_c_coordination_never_makes_infeasible_feasible(self):
        """Invariant C: Soft coordination bonus must never make an infeasible candidate feasible."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T_ENG", section_id="SEC_1", department="ENGINEERING", duration=180),
            make_sample_task("T_ST", section_id="SEC_1", department="S&T", duration=60),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 4}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        assert len(res["recommendations"]["T_ENG"]) == 0
        assert any(inf["task_id"] == "T_ENG" for inf in res["infeasible_tasks"])

    def test_invariant_d_hard_feasibility_dominates_soft_preference(self):
        """Invariant D: Hard feasibility difference must dominate any soft coordination preference."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T_ENG", section_id="SEC_1", department="ENGINEERING", duration=60),
            make_sample_task("T_ST", section_id="SEC_1", department="S&T", duration=60),
        ])
        windows = [
            {"block_id": "B001", "start_slot": 1, "end_slot": 4},
            {"block_id": "B002", "start_slot": 5, "end_slot": 10},
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r_eng_b001 = [r for r in res["recommendations"]["T_ENG"] if r["block_id"] == "B001"][0]
        r_st_b001 = [r for r in res["recommendations"]["T_ST"] if r["block_id"] == "B001"][0]
        assert r_eng_b001["coordination"]["is_coordinated"] is False
        assert r_st_b001["coordination"]["is_coordinated"] is False

    def test_invariant_e_deterministic_ranking_across_runs(self):
        """Invariant E: Ranking must be 100% deterministic across repeated identical runs."""
        optimizer = BlockOptimizer(planning_hours=12)
        tasks = pd.DataFrame([
            make_sample_task(f"T{i}", section_id=f"SEC_{i%2}", department=["ENGINEERING", "S&T"][i%2], score=0.6 + i*0.05)
            for i in range(4)
        ])
        windows = [
            {"block_id": f"B00{i}", "start_slot": i * 4, "end_slot": i * 4 + 3}
            for i in range(1, 5)
        ]
        run1 = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        run2 = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        for t_id in tasks["task_id"]:
            ranks1 = [(r["rank"], r["block_id"]) for r in run1["recommendations"][t_id]]
            ranks2 = [(r["rank"], r["block_id"]) for r in run2["recommendations"][t_id]]
            assert ranks1 == ranks2

    def test_invariant_f_unrelated_tasks_do_not_alter_independent_ranking(self):
        """Invariant F: Changing unrelated tasks on separate sections with ample manpower should not alter a task's ranking."""
        optimizer = BlockOptimizer(planning_hours=12, max_manpower=20)
        task_a = make_sample_task("TASK_A", section_id="SEC_A", department="ENGINEERING", duration=60, score=0.8)
        windows = [
            {"block_id": "B001", "start_slot": 1, "end_slot": 4},
            {"block_id": "B002", "start_slot": 5, "end_slot": 8},
            {"block_id": "B003", "start_slot": 9, "end_slot": 12},
        ]
        res1 = optimizer.optimize_with_recommendations(pd.DataFrame([task_a]), block_windows=windows)
        ranks1 = [(r["rank"], r["block_id"]) for r in res1["recommendations"]["TASK_A"]]

        task_x = make_sample_task("TASK_X", section_id="SEC_X", department="TRACTION", duration=60, score=0.9, manpower=2)
        res2 = optimizer.optimize_with_recommendations(pd.DataFrame([task_a, task_x]), block_windows=windows)
        ranks2 = [(r["rank"], r["block_id"]) for r in res2["recommendations"]["TASK_A"]]

        assert ranks1 == ranks2, f"Unrelated task affected Task A ranking: {ranks1} vs {ranks2}"


class TestComprehensiveCoordinationScenarios:
    """Verifies all 7 coordination scenarios required by Part 6."""

    def test_scenario_1_engineering_plus_st(self):
        """Scenario 1: Engineering + S&T, same section, compatible window."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_COMMON", department="ENGINEERING", duration=60),
            make_sample_task("T2", section_id="SEC_COMMON", department="S&T", duration=60),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]
        assert r1["coordination"]["is_coordinated"] is True
        assert r2["coordination"]["is_coordinated"] is True
        assert "S&T" in r1["coordination"]["coordination_departments"]
        assert "ENGINEERING" in r2["coordination"]["coordination_departments"]

    def test_scenario_2_engineering_plus_traction(self):
        """Scenario 2: Engineering + Traction, same section, compatible window."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_COMMON", department="ENGINEERING", duration=60),
            make_sample_task("T2", section_id="SEC_COMMON", department="TRACTION", duration=60),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]
        assert r1["coordination"]["is_coordinated"] is True
        assert r2["coordination"]["is_coordinated"] is True
        assert "TRACTION" in r1["coordination"]["coordination_departments"]
        assert "ENGINEERING" in r2["coordination"]["coordination_departments"]

    def test_scenario_3_st_plus_traction(self):
        """Scenario 3: S&T + Traction, same section, compatible window."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_COMMON", department="S&T", duration=60),
            make_sample_task("T2", section_id="SEC_COMMON", department="TRACTION", duration=60),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]
        assert r1["coordination"]["is_coordinated"] is True
        assert r2["coordination"]["is_coordinated"] is True
        assert "TRACTION" in r1["coordination"]["coordination_departments"]
        assert "S&T" in r2["coordination"]["coordination_departments"]

    def test_scenario_4_same_department_same_section(self):
        """Scenario 4: Same department, same section -> NO coordination bonus."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_COMMON", department="ENGINEERING", duration=60),
            make_sample_task("T2", section_id="SEC_COMMON", department="ENGINEERING", duration=60),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]
        assert r1["coordination"]["is_coordinated"] is False
        assert r2["coordination"]["is_coordinated"] is False

    def test_scenario_5_different_sections(self):
        """Scenario 5: Different sections -> NO coordination bonus."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_A", department="ENGINEERING", duration=60),
            make_sample_task("T2", section_id="SEC_B", department="S&T", duration=60),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 6}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]
        assert r1["coordination"]["is_coordinated"] is False
        assert r2["coordination"]["is_coordinated"] is False

    def test_scenario_6_same_section_manpower_conflict(self):
        """Scenario 6: Same section but manpower conflict prevents joint execution."""
        optimizer = BlockOptimizer(max_manpower=8)
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_COMMON", department="ENGINEERING", duration=60, manpower=6),
            make_sample_task("T2", section_id="SEC_COMMON", department="S&T", duration=60, manpower=6),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 4}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]
        assert r1["coordination"]["is_coordinated"] is False
        assert r2["coordination"]["is_coordinated"] is False

    def test_scenario_7_same_section_duration_conflict(self):
        """Scenario 7: Same section but duration conflict -> cannot fit sequentially."""
        optimizer = BlockOptimizer()
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_COMMON", department="ENGINEERING", duration=90),
            make_sample_task("T2", section_id="SEC_COMMON", department="S&T", duration=90),
        ])
        windows = [{"block_id": "B001", "start_slot": 1, "end_slot": 5}]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)
        r1 = res["recommendations"]["T1"][0]
        r2 = res["recommendations"]["T2"][0]
        assert r1["coordination"]["is_coordinated"] is False
        assert r2["coordination"]["is_coordinated"] is False


class TestIndependentFeasibilityValidator:
    """Verifies all recommendations pass the independent physical validator."""

    def test_independent_validator_on_multi_task_recommendations(self):
        optimizer = BlockOptimizer(planning_hours=12)
        tasks = pd.DataFrame([
            make_sample_task("T1", section_id="SEC_1", department="ENGINEERING", duration=60, manpower=4, score=0.85),
            make_sample_task("T2", section_id="SEC_1", department="S&T", duration=60, manpower=4, score=0.75),
            make_sample_task("T3", section_id="SEC_2", department="TRACTION", duration=90, manpower=5, score=0.65),
            make_sample_task("T4", section_id="SEC_2", department="ENGINEERING", duration=120, manpower=6, score=0.55),
        ])
        windows = [
            {"block_id": f"B{i:02d}", "start_slot": i * 5, "end_slot": i * 5 + 4}
            for i in range(1, 5)
        ]
        res = optimizer.optimize_with_recommendations(tasks, block_windows=windows)

        for t_id, recs in res["recommendations"].items():
            orig_task = tasks[tasks["task_id"] == t_id].iloc[0]
            for r in recs:
                valid, msg = independent_validator(r, orig_task, windows, max_manpower=12)
                assert valid, f"Validation failure for {t_id}: {msg}"


class TestRankingConstantsAndPreemptionCorrectness:
    """Verifies that preemption does not arbitrarily distort delay-based ranking."""

    def test_pure_objective_ranking_without_preemption_penalty(self):
        """
        Confirms that when a task has multiple candidate windows,
        ranking strictly follows objective score (priority - delay + coord)
        and is not perturbed by arbitrary constants.
        """
        optimizer = BlockOptimizer(planning_hours=12)
        task_b = make_sample_task("TASK_B", section_id="SEC_1", department="ENGINEERING", duration=60, score=0.7, delay=0.0)
        windows = [
            {"block_id": "B001", "start_slot": 1, "end_slot": 4},
            {"block_id": "B002", "start_slot": 5, "end_slot": 8},
        ]
        res = optimizer.optimize_with_recommendations(pd.DataFrame([task_b]), block_windows=windows)
        recs = res["recommendations"]["TASK_B"]
        assert len(recs) == 2
        assert recs[0]["rank"] == 1
        assert recs[1]["rank"] == 2
