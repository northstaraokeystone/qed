"""
QED v6 Edge Tests - core_telemetry

Edge case tests based on EdgeTestPlan_v1 for high-value/high-risk scenarios.
Focuses on threshold boundaries, degenerate signals, and latency constraints.
"""

import time

import numpy as np
import pytest

import qed
from qed import QEDReceipt, check_constraints


def _make_signal_with_amplitude(
    target_amplitude: float,
    n: int = 1000,
    sample_rate_hz: float = 1000.0,
    frequency_hz: float = 40.0,
    offset: float = 2.0,
) -> np.ndarray:
    """Generate signal with precise target amplitude (no noise for boundary tests)."""
    t = np.arange(n) / sample_rate_hz
    return target_amplitude * np.sin(2.0 * np.pi * frequency_hz * t) + offset


class TestEdge001AmplitudeAtBound:
    """EDGE_001: Amplitude exactly at bound triggers violation."""

    def test_amplitude_just_above_bound_triggers_violation(self):
        """A=14.7001 should trigger amplitude violation."""
        # Create signal with amplitude just above 14.7 bound
        signal = _make_signal_with_amplitude(target_amplitude=14.7001)

        # This should raise ValueError from _check_amplitude_bounds
        with pytest.raises(ValueError) as exc_info:
            qed.qed(signal, scenario="tesla_fsd")

        assert "amplitude" in str(exc_info.value).lower()
        assert "14.7" in str(exc_info.value)

    def test_check_constraints_at_exact_boundary(self):
        """check_constraints with A=14.70001 should flag violation."""
        verified, violations = check_constraints(14.70001, 40.0, "tesla_fsd")
        assert verified is False
        assert len(violations) == 1
        assert violations[0]["constraint_id"] == "amplitude_bound_tesla"

    def test_amplitude_at_boundary_epsilon(self):
        """A=14.7 + 1e-6 should still flag violation."""
        verified, violations = check_constraints(14.7 + 1e-6, 40.0, "tesla_fsd")
        assert verified is False


class TestEdge002AmplitudeBelowBound:
    """EDGE_002: Amplitude just below bound passes."""

    def test_amplitude_just_below_bound_passes(self):
        """A=14.699 should pass constraint check."""
        verified, violations = check_constraints(14.699, 40.0, "tesla_fsd")
        assert verified is True
        assert violations == []

    def test_signal_at_safe_amplitude_returns_receipt(self):
        """Signal with A < 14.7 should return valid receipt."""
        signal = _make_signal_with_amplitude(target_amplitude=14.0)
        result = qed.qed(signal, scenario="tesla_fsd")

        assert "receipt" in result
        receipt = result["receipt"]
        assert receipt.verified is True
        assert receipt.violations == []


class TestEdge003ConstantSignal:
    """EDGE_003: Constant signal (degenerate) handling."""

    def test_constant_signal_no_crash(self):
        """Constant signal should not crash, but may have edge behavior."""
        signal = np.ones(1000) * 5.0

        # This may produce unusual FFT results, but should not crash
        try:
            result = qed.qed(signal, scenario="tesla_fsd")
            # If it succeeds, verify receipt exists
            assert "receipt" in result
            receipt = result["receipt"]
            assert isinstance(receipt, QEDReceipt)
        except ValueError:
            # Amplitude bound violation is acceptable for degenerate signal
            pass

    def test_near_constant_signal_with_tiny_variation(self):
        """Signal with tiny variation should handle gracefully."""
        signal = np.ones(1000) * 5.0
        signal[500] = 5.0001  # Tiny variation

        try:
            result = qed.qed(signal, scenario="tesla_fsd")
            assert "receipt" in result
        except ValueError:
            # Acceptable if amplitude check fails
            pass


class TestEdge004LargeWindowLatency:
    """EDGE_004: Large window 10k samples under 1s."""

    def test_10k_samples_under_1_second(self):
        """10k sample window must complete in under 1000ms."""
        rng = np.random.default_rng(12345)
        t = np.arange(10000) / 1000.0
        signal = 10.0 * np.sin(2.0 * np.pi * 40.0 * t) + rng.normal(0, 0.1, 10000)

        start = time.perf_counter()
        result = qed.qed(signal, scenario="tesla_fsd", sample_rate_hz=1000.0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < 1000.0, f"10k window took {elapsed_ms:.1f}ms, exceeds 1s"
        assert "receipt" in result

    def test_large_window_receipt_valid(self):
        """Large window should produce valid receipt with all fields."""
        rng = np.random.default_rng(54321)
        t = np.arange(10000) / 1000.0
        signal = 8.0 * np.sin(2.0 * np.pi * 50.0 * t) + rng.normal(0, 0.05, 10000)

        result = qed.qed(signal, scenario="tesla_fsd", sample_rate_hz=1000.0)
        receipt = result["receipt"]

        assert receipt.params["A"] > 0
        assert receipt.ratio > 0
        assert receipt.H_bits > 0


class TestEdge005EmptySignal:
    """EDGE_005: Empty signal raises ValueError."""

    def test_empty_signal_raises_value_error(self):
        """Empty numpy array should raise ValueError."""
        signal = np.array([])

        with pytest.raises(ValueError) as exc_info:
            qed.qed(signal, scenario="tesla_fsd")

        assert "non-empty" in str(exc_info.value).lower()

    def test_empty_signal_error_message_clear(self):
        """Error message should clearly indicate the problem."""
        signal = np.array([], dtype=np.float64)

        with pytest.raises(ValueError) as exc_info:
            qed.qed(signal)

        assert "signal" in str(exc_info.value).lower()


class TestEdgeConstraintModuleFallback:
    """Test graceful fallback when sympy_constraints has issues."""

    def test_unknown_scenario_uses_generic_constraints(self):
        """Unknown scenario should fall back to generic constraints."""
        verified, violations = check_constraints(12.0, 40.0, "unknown_scenario")
        # Should use generic constraints (bound=14.7)
        assert verified is True
        assert violations == []

    def test_unknown_scenario_violation_detection(self):
        """Unknown scenario should still detect violations with generic bounds."""
        verified, violations = check_constraints(15.0, 40.0, "unknown_scenario")
        assert verified is False
        assert len(violations) == 1


class TestEdgeNaNInfHandling:
    """Test NaN and Inf parameter handling."""

    def test_nan_amplitude_flagged(self):
        """NaN amplitude should be flagged as violation."""
        verified, violations = check_constraints(float("nan"), 40.0, "tesla_fsd")
        assert verified is False
        assert violations[0]["constraint_id"] == "finite_params"

    def test_inf_amplitude_flagged(self):
        """Inf amplitude should be flagged as violation."""
        verified, violations = check_constraints(float("inf"), 40.0, "tesla_fsd")
        assert verified is False
        assert violations[0]["constraint_id"] == "finite_params"

    def test_negative_inf_amplitude_flagged(self):
        """Negative Inf amplitude should be flagged as violation."""
        verified, violations = check_constraints(float("-inf"), 40.0, "tesla_fsd")
        assert verified is False

    def test_nan_frequency_flagged(self):
        """NaN frequency should be flagged as violation."""
        verified, violations = check_constraints(10.0, float("nan"), "tesla_fsd")
        assert verified is False
        assert violations[0]["constraint_id"] == "finite_params"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
