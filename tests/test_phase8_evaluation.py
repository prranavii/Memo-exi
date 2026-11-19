"""
Unit and integration tests for Phase 8: Benchmark and Evaluation Suite.
"""

import pytest
from memo_exi.evaluation.evaluator import BenchmarkRunner


def test_benchmark_runner_execution():
    """Verify benchmark runner produces results across scenarios and baselines."""
    runner = BenchmarkRunner()
    results = runner.run_evaluation(
        scenario_ids=[1, 2, 3, 4],
        baselines=["B0", "B2", "P"],
        ablations=["A1", "A3"],
    )

    assert "results" in results
    assert "ablation_results" in results
    assert "summary_by_system" in results
    assert len(results["results"]) == 4 * 3  # 4 scenarios x 3 baselines
    assert len(results["ablation_results"]) == 2  # 2 ablations

    summary = results["summary_by_system"]
    assert "P" in summary
    assert "B0" in summary
    assert "B2" in summary

    # Proposed system P should outperform standard RAG B2 on historical accuracy
    assert summary["P"]["avg_historical_accuracy"] > summary["B2"]["avg_historical_accuracy"]
    # Proposed system P should have lower stale error rate
    assert summary["P"]["avg_stale_error_rate"] < summary["B2"]["avg_stale_error_rate"]
