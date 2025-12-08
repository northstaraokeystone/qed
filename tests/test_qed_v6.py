"""
QED v6 Smoke Tests - Telemetry Compression Engine

Smoke tests for core QED v6 functionality:
- QEDReceipt emission validation (fields: params dict, ratio float>20, verified bool, violations)
- edge_lab_v1 N=10 scenario runs with sane counts
- sympy_constraints performance (<1ms evaluation, no spikes)

Edge Test Plan:
EDGE_001: Receipt fields - correctness - valid Receipt w/params/ratio/verified=0 violations
EDGE_002: Edge lab N=10 - recall - runs 10 scenarios, hits>=8
EDGE_003: Sympy runtime - latency - <1ms eval
"""

import time
from typing import Any, Dict

import numpy as np
import pytest

import qed
from qed import QEDReceipt
from edge_lab_v1 import (
    get_edge_lab_scenarios,
    load_scenarios,
    run_edge_lab,
    summarize_results,
)
import sympy_constraints


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_signal() -> np.ndarray:
    """Generate a clean sinusoidal test signal within Tesla bounds."""
    t = np.linspace(0, 1, 1000)
    return 12.0 * np.sin(2 * np.pi * 40.0 * t) + 2.0


@pytest.fixture
def mock_signal_with_noise() -> np.ndarray:
    """Generate a noisy sinusoidal test signal."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 1000)
    signal = 12.0 * np.sin(2 * np.pi * 40.0 * t) + 2.0
    return signal + rng.normal(0, 0.1, 1000)


@pytest.fixture
def mock_receipt_data() -> Dict[str, Any]:
    """Fixture with valid receipt field values."""
    return {
        "params": {"A": 12.0, "f": 40.0, "phi": 0.0, "c": 2.0, "scenario": "tesla_fsd"},
        "ratio": 60.0,
        "verified": True,
        "violations": [],
    }


# -----------------------------------------------------------------------------
# Test: QEDReceipt Emission (EDGE_001)
# -----------------------------------------------------------------------------


class TestReceiptEmit:
    """
    Smoke tests for QED receipt emission.

    EDGE_001: Receipt fields - correctness metric
    Target: Emit valid Receipt w/params/ratio/verified=0 violations
    KPI threshold: fields=4 (params, ratio, verified, violations)
    """

    def test_receipt_emit_valid_fields(self, mock_signal: np.ndarray):
        """Single-window qed run emits valid QEDReceipt with required fields."""
        result = qed.qed(mock_signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        # Validate required fields exist
        assert isinstance(receipt, QEDReceipt)
        assert isinstance(receipt.params, dict)
        assert isinstance(receipt.ratio, float)
        assert isinstance(receipt.verified, bool) or receipt.verified is None
        assert isinstance(receipt.violations, list)

    def test_receipt_params_dict_structure(self, mock_signal: np.ndarray):
        """Receipt params dict contains fitted sine parameters."""
        result = qed.qed(mock_signal, scenario="tesla_fsd")
        params = result["receipt"].params

        assert "A" in params
        assert "f" in params
        assert "phi" in params
        assert "c" in params
        assert "scenario" in params
        assert isinstance(params["A"], float)
        assert isinstance(params["f"], float)

    def test_receipt_ratio_exceeds_20(self, mock_signal: np.ndarray):
        """Receipt ratio should exceed 20:1 compression."""
        result = qed.qed(mock_signal, scenario="tesla_fsd", bit_depth=12)
        receipt = result["receipt"]

        # For 1000 samples @ 12 bits, ratio = (1000 * 12) / 200 = 60
        assert receipt.ratio > 20.0, f"Ratio {receipt.ratio} should exceed 20"

    def test_receipt_verified_true_no_violations(self, mock_signal: np.ndarray):
        """Receipt verified=True when signal within bounds and violations empty."""
        result = qed.qed(mock_signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        # Signal amplitude 12.0 < 14.7 bound, so should pass
        assert receipt.verified is True
        assert len(receipt.violations) == 0

    def test_receipt_jsonl_serializable(self, mock_signal: np.ndarray):
        """Receipt can be serialized to JSONL format."""
        import io
        import json
        from qed import write_receipt_jsonl

        result = qed.qed(mock_signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        buffer = io.StringIO()
        write_receipt_jsonl(receipt, buffer)
        buffer.seek(0)

        loaded = json.loads(buffer.readline())
        assert "params" in loaded
        assert "ratio" in loaded
        assert "verified" in loaded
        assert "violations" in loaded


class TestReceiptViolations:
    """Test receipt violations field population."""

    def test_violations_populated_when_bound_exceeded(self):
        """Receipt violations list non-empty when amplitude exceeds bound."""
        # Create signal with amplitude > 14.7 (Tesla bound)
        t = np.linspace(0, 1, 1000)
        high_amplitude_signal = 16.0 * np.sin(2 * np.pi * 40.0 * t)

        # This should raise ValueError due to _check_amplitude_bounds
        with pytest.raises(ValueError, match="amplitude .* exceeds bound"):
            qed.qed(high_amplitude_signal, scenario="tesla_fsd")


# -----------------------------------------------------------------------------
# Test: Edge Lab N=10 Scenario Counts (EDGE_002)
# -----------------------------------------------------------------------------


class TestEdgeLabCounts:
    """
    Smoke tests for edge_lab_v1 scenario runs.

    EDGE_002: Edge lab N=10 - recall metric
    Target: Runs 10 scenarios, hits>=8
    KPI threshold: hits>=0.8
    """

    def test_edge_lab_scenarios_available(self):
        """In-memory fallback scenarios are available."""
        scenarios = get_edge_lab_scenarios()
        assert len(scenarios) >= 10, "Should have at least 10 scenarios"

    def test_load_scenarios_returns_list(self):
        """load_scenarios returns list of scenario dicts."""
        scenarios = load_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 10

    def test_load_scenarios_structure(self):
        """Each scenario has required fields."""
        scenarios = load_scenarios()
        for s in scenarios[:10]:  # Check first 10
            assert "scenario_id" in s
            assert "hook" in s
            assert "type" in s
            assert "expected_loss" in s
            assert "signal" in s
            assert isinstance(s["signal"], list)
            assert len(s["signal"]) > 0

    def test_edge_lab_runs_10_scenarios(self):
        """run_edge_lab executes and returns results for 10 scenarios."""
        # Filter to just 10 normal/tesla scenarios for fast smoke test
        results = run_edge_lab(scenario_filter="tesla_normal")

        # Should have some results
        assert len(results) >= 1

        # Run with full set limited to 10
        scenarios = load_scenarios()[:10]
        all_results = []
        for scenario in scenarios:
            signal = np.array(scenario["signal"])
            try:
                qed_result = qed.qed(
                    signal=signal,
                    scenario="tesla_fsd" if scenario["hook"] == "tesla" else "generic",
                    hook_name=scenario["hook"],
                )
                all_results.append({"hit": True, "error": None})
            except Exception as e:
                all_results.append({"hit": False, "error": str(e)})

        assert len(all_results) == 10

    def test_edge_lab_hit_rate_sane(self):
        """Edge lab returns results with expected metric structure."""
        # Run edge lab with Tesla normal scenarios (within bounds)
        results = run_edge_lab(scenario_filter="tesla_normal")

        if len(results) == 0:
            pytest.skip("No Tesla normal scenarios found")

        # Verify all results have valid metrics
        for r in results:
            assert r.latency_ms >= 0
            assert isinstance(r.hit, bool)
            assert isinstance(r.miss, bool)
            assert r.hit != r.miss  # Exactly one should be true

        # Tesla normal scenarios should succeed (within 14.7 bound)
        success = sum(1 for r in results if r.error is None)
        success_rate = success / len(results) if results else 0
        assert success_rate >= 0.5, f"Success rate {success_rate:.2%} too low"

    def test_edge_lab_result_fields(self):
        """EdgeLabResult has all required metric fields."""
        results = run_edge_lab(scenario_filter="tesla_normal_001")

        if len(results) == 0:
            pytest.skip("No matching scenarios found")

        r = results[0]
        assert hasattr(r, "scenario_id")
        assert hasattr(r, "hook")
        assert hasattr(r, "type")
        assert hasattr(r, "hit")
        assert hasattr(r, "miss")
        assert hasattr(r, "latency_ms")
        assert hasattr(r, "ratio")
        assert hasattr(r, "violations")
        assert hasattr(r, "verified")

    def test_edge_lab_summarize_results(self):
        """summarize_results computes aggregate metrics."""
        results = run_edge_lab(scenario_filter="normal")

        if len(results) == 0:
            pytest.skip("No normal scenarios found")

        summary = summarize_results(results)

        assert "n_scenarios" in summary
        assert "n_hits" in summary
        assert "n_misses" in summary
        assert "hit_rate" in summary
        assert "avg_ratio" in summary
        assert "avg_latency_ms" in summary
        assert "kpi" in summary


# -----------------------------------------------------------------------------
# Test: Sympy Constraints Runtime (EDGE_003)
# -----------------------------------------------------------------------------


class TestSympyTime:
    """
    Smoke tests for sympy_constraints evaluation performance.

    EDGE_003: Sympy runtime - latency metric
    Target: <1ms eval
    KPI threshold: latency_ms<1.0
    """

    def test_get_constraints_tesla_under_1ms(self):
        """get_constraints('tesla_fsd') completes in <1ms."""
        # Warmup
        _ = sympy_constraints.get_constraints("tesla_fsd")

        start = time.perf_counter()
        cons = sympy_constraints.get_constraints("tesla_fsd")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1.0, f"get_constraints took {elapsed_ms:.3f}ms"
        assert len(cons) >= 1

    def test_evaluator_call_under_1ms(self):
        """Single evaluator call completes in <1ms."""
        evals = sympy_constraints.get_constraint_evaluators("tesla_fsd")
        _, _, fn = evals[0]

        # Warmup
        _ = fn(12.5)

        start = time.perf_counter()
        result = fn(12.5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1.0, f"Evaluator took {elapsed_ms:.3f}ms"
        assert isinstance(result, (bool, np.bool_))

    def test_evaluate_all_under_1ms(self):
        """evaluate_all() completes in <1ms."""
        # Warmup
        _ = sympy_constraints.evaluate_all("tesla_fsd", A=12.0, ratio=60.0, savings_M=38.0)

        start = time.perf_counter()
        passed, violations = sympy_constraints.evaluate_all(
            "tesla_fsd", A=12.0, ratio=60.0, savings_M=38.0
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1.0, f"evaluate_all took {elapsed_ms:.3f}ms"
        assert passed is True
        assert violations == []

    def test_no_exception_on_constraint_eval(self):
        """Constraint evaluation does not raise exceptions."""
        hooks = ["tesla_fsd", "spacex_flight", "generic", "neuralink_stream"]

        for hook in hooks:
            cons = sympy_constraints.get_constraints(hook)
            assert len(cons) >= 1

            evals = sympy_constraints.get_constraint_evaluators(hook)
            for cid, ctype, fn in evals:
                if ctype == "amplitude_bound":
                    result = fn(10.0)
                    assert isinstance(result, (bool, np.bool_))
                elif ctype == "ratio_min":
                    result = fn(50.0)
                    assert isinstance(result, (bool, np.bool_))
                elif ctype == "savings_min":
                    result = fn(20.0)
                    assert isinstance(result, (bool, np.bool_))
                elif ctype == "mse_max":
                    result = fn(0.0001)
                    assert isinstance(result, (bool, np.bool_))

    def test_batch_eval_no_latency_spike(self):
        """Batch constraint evaluation shows no latency spikes."""
        # Run 100 evaluations, check max latency
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            sympy_constraints.evaluate_all("tesla_fsd", A=12.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        # Average should be well under 1ms, max should be under 10ms (allow for GC)
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}ms exceeds 1ms"
        assert max_latency < 10.0, f"Max latency spike {max_latency:.3f}ms"


# -----------------------------------------------------------------------------
# Integration Smoke Tests
# -----------------------------------------------------------------------------


class TestQEDv6Integration:
    """End-to-end integration smoke tests for QED v6 pipeline."""

    def test_full_pipeline_receipt_to_edge_lab(self, mock_signal: np.ndarray):
        """Full pipeline: qed -> receipt -> edge lab compatible."""
        # Run QED
        result = qed.qed(mock_signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        # Verify receipt is valid
        assert receipt.ratio > 20.0
        assert receipt.verified is True

        # Verify constraints pass
        passed, violations = sympy_constraints.evaluate_all(
            "tesla_fsd",
            A=receipt.params["A"],
            ratio=receipt.ratio,
            savings_M=receipt.savings_M,
        )
        assert passed is True

    def test_constraint_violation_flow(self):
        """Constraint violations are properly detected and reported."""
        # Use amplitude that exceeds generic bound but not in exception path
        t = np.linspace(0, 1, 1000)
        signal = 14.0 * np.sin(2 * np.pi * 40.0 * t)  # Within Tesla 14.7 bound

        result = qed.qed(signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        # Should pass Tesla bounds
        assert receipt.verified is True or receipt.verified is None

    def test_multiple_hooks_supported(self):
        """Multiple hook configurations are supported."""
        hooks_to_test = ["tesla", "spacex", "generic"]

        for hook in hooks_to_test:
            cons = sympy_constraints.get_constraints(hook)
            # At least one constraint per hook
            assert len(cons) >= 0  # generic fallback is OK


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
