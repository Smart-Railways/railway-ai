"""
Unit & Integration Tests for RailwayMLEngine and MultiHorizonPlanner (Phase 7 & Phase 8)
"""

import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.services.ml_engine import RailwayMLEngine
from src.optimization.multi_horizon_planner import MultiHorizonPlanner


def make_test_worklist(n=6):
    sections = [
        "NDL-MTJ-01", "MTJ-AGC-01", "AGC-GWL-01",
        "GWL-JHS-01", "JHS-BINA-01", "BINA-BPL-01"
    ]
    departments = ["ENGINEERING", "S&T", "TRACTION"]
    tasks = []
    for i in range(n):
        tasks.append({
            "task_id": f"TEST_TASK_{i+1:03d}",
            "section_id": sections[i % len(sections)],
            "department": departments[i % len(departments)],
            "estimated_duration": 30 + (i * 15),
            "required_manpower": 3 + (i % 3),
            "failure_probability": 0.15 + (i * 0.1),
            "urgency_score": 0.35 + (i * 0.08),
            "criticality": 4 + i,
            "overdue_days": i * 4,
            "traffic_intensity": 0.5 + (i * 0.05),
            "maintenance_decision_score": 0.35 + (i * 0.07),
            "predicted_delay_minutes": 0.0,
        })
    return pd.DataFrame(tasks)


class TestRailwayMLEngine:
    def test_health(self):
        engine = RailwayMLEngine()
        h = engine.health()
        assert h["status"] == "ok"
        assert h["service"] == "railway-ml-engine"
        assert h["version"] == "0.1.0"
        assert "components" in h
        assert h["components"]["block_optimizer"] == "ready"
        assert h["components"]["multi_horizon_planner"] == "ready"

    def test_predict_dataframe(self):
        engine = RailwayMLEngine()
        df = make_test_worklist(4)
        scored = engine.predict(df, apply_real_pressure=True)
        assert not scored.empty
        assert "maintenance_decision_score" in scored.columns
        assert "decision_category" in scored.columns
        assert "decision_rank" in scored.columns

    def test_predict_dict(self):
        engine = RailwayMLEngine()
        single_task = make_test_worklist(1).to_dict(orient="records")[0]
        scored = engine.predict(single_task)
        assert len(scored) == 1
        assert "maintenance_decision_score" in scored.columns

    def test_generate_block_plan_daily(self):
        engine = RailwayMLEngine()
        df = make_test_worklist(4)
        res = engine.generate_block_plan(df, horizon_type="daily")
        assert res["horizon_type"] == "daily_6h"
        assert "plan" in res
        assert not res["plan"].empty
        assert "plan_sequence" in res["plan"].columns

    def test_generate_block_plan_weekly(self):
        engine = RailwayMLEngine()
        df = make_test_worklist(6)
        res = engine.generate_block_plan(df, horizon_type="weekly")
        assert res["horizon_type"] == "weekly_7d"
        assert "plan" in res
        assert not res["plan"].empty
        assert "day" in res["plan"].columns

    def test_generate_block_plan_rolling(self):
        engine = RailwayMLEngine()
        df = make_test_worklist(6)
        res = engine.generate_block_plan(df, horizon_type="rolling", horizon_days=7, committed_days=2)
        assert res["horizon_type"] == "rolling_horizon"
        assert "plan" in res
        assert "committed_plan" in res
        assert "tentative_plan" in res

    def test_generate_block_plan_dynamic(self):
        engine = RailwayMLEngine()
        df = make_test_worklist(6)
        # First generate weekly plan
        initial = engine.generate_block_plan(df, horizon_type="weekly")
        # Then reschedule with a delay disruption on NDL-MTJ-01
        res = engine.generate_block_plan(
            df,
            horizon_type="dynamic",
            current_plan=initial["plan"],
            disrupted_sections={"NDL-MTJ-01": 35.0},
            current_slot=10
        )
        assert res["horizon_type"] == "dynamic_rescheduling"
        assert "plan" in res
        assert "NDL-MTJ-01" in res["disrupted_sections"]


class TestMultiHorizonPlanner:
    def test_generate_daily_block_windows(self):
        windows = MultiHorizonPlanner.generate_daily_block_windows(days=7)
        assert len(windows) == 14  # 2 windows per day * 7 days
        assert windows[0]["block_id"] == "D1_B1"
        assert windows[1]["block_id"] == "D1_B2"
        assert windows[-1]["block_id"] == "D7_B2"

    def test_plan_horizon_monthly(self):
        planner = MultiHorizonPlanner(max_time_seconds=10)
        df = make_test_worklist(6)
        plan = planner.plan_monthly(df)
        assert not plan.empty
        assert (plan["day"] <= 30).all()
        # Verify no section overlap
        for sec, grp in plan.groupby("section_id"):
            slots = sorted(zip(grp["start_slot"], grp["end_slot"]))
            for idx in range(len(slots) - 1):
                assert slots[idx][1] <= slots[idx + 1][0]

    def test_plan_rolling_horizon(self):
        planner = MultiHorizonPlanner(max_time_seconds=10)
        df = make_test_worklist(6)
        res = planner.plan_rolling_horizon(df, horizon_days=7, committed_days=3)
        committed = res["committed_plan"]
        tentative = res["tentative_plan"]
        committed_slots_limit = 3 * MultiHorizonPlanner.SLOTS_PER_DAY
        if not committed.empty:
            assert (committed["start_slot"] < committed_slots_limit).all()
        if not tentative.empty:
            assert (tentative["start_slot"] >= committed_slots_limit).all()
