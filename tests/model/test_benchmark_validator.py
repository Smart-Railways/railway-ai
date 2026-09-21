"""
Unit & Regression Tests for BenchmarkValidator (Phase 8)
"""

import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.evaluation.benchmark_validator import BenchmarkValidator
from tests.scheduling.test_multi_horizon_and_engine import make_test_worklist


class TestBenchmarkValidator:
    def test_simulate_uncoordinated_baseline(self):
        df = make_test_worklist(6)
        res = BenchmarkValidator.simulate_uncoordinated_baseline(df)
        assert "scheduled_tasks_count" in res
        assert "total_decision_score" in res
        assert "plan" in res
        assert res["scheduled_tasks_count"] > 0

    def test_run_benchmark(self):
        validator = BenchmarkValidator()
        df = make_test_worklist(6)
        bench = validator.run_benchmark(df)

        assert "baseline" in bench
        assert "railway_ai" in bench
        assert "comparison" in bench

        # AI should be at least as good as or better than FIFO heuristic
        assert bench["comparison"]["superiority_demonstrated"] is True
        assert bench["railway_ai"]["total_decision_score"] >= bench["baseline"]["total_decision_score"]
