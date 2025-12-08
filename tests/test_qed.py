"""
QED v6 Baseline Tests - core_telemetry

Tests for QEDReceipt, JSONL writer, constraint checking, and receipt generation.
"""

import io
import json
import time

import numpy as np
import pytest

import qed
from qed import QEDReceipt, check_constraints, write_receipt_jsonl


def _make_signal(
    n: int = 1000,
    sample_rate_hz: float = 1000.0,
    amplitude: float = 12.0,
    frequency_hz: float = 40.0,
    noise_sigma: float = 0.1,
    offset: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic test signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate_hz
    clean = amplitude * np.sin(2.0 * np.pi * frequency_hz * t) + offset
    noise = rng.normal(0.0, noise_sigma, n)
    return clean + noise


class TestQEDReturnsReceipt:
    """Test that qed() returns receipt key with QEDReceipt."""

    def test_qed_returns_receipt_key(self):
        """Verify receipt key exists in qed() return dict."""
        signal = _make_signal()
        result = qed.qed(signal, scenario="tesla_fsd")
        assert "receipt" in result
        assert isinstance(result["receipt"], QEDReceipt)

    def test_receipt_has_required_fields(self):
        """Verify receipt has all required fields."""
        signal = _make_signal()
        result = qed.qed(signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        assert hasattr(receipt, "ts")
        assert hasattr(receipt, "window_id")
        assert hasattr(receipt, "params")
        assert hasattr(receipt, "ratio")
        assert hasattr(receipt, "H_bits")
        assert hasattr(receipt, "recall")
        assert hasattr(receipt, "savings_M")
        assert hasattr(receipt, "verified")
        assert hasattr(receipt, "violations")
        assert hasattr(receipt, "trace")

    def test_receipt_params_contains_fitted_values(self):
        """Verify params dict contains A, f, phi, c, scenario."""
        signal = _make_signal()
        result = qed.qed(signal, scenario="tesla_fsd")
        params = result["receipt"].params

        assert "A" in params
        assert "f" in params
        assert "phi" in params
        assert "c" in params
        assert "scenario" in params
        assert params["scenario"] == "tesla_fsd"

    def test_receipt_is_immutable(self):
        """Verify QEDReceipt is frozen (immutable)."""
        signal = _make_signal()
        result = qed.qed(signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        with pytest.raises(Exception):  # FrozenInstanceError
            receipt.ratio = 999.0


class TestReceiptJSONLRoundtrip:
    """Test JSONL writer and roundtrip serialization."""

    def test_write_receipt_jsonl_basic(self):
        """Test basic JSONL write operation."""
        signal = _make_signal()
        result = qed.qed(signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        buffer = io.StringIO()
        write_receipt_jsonl(receipt, buffer)

        buffer.seek(0)
        line = buffer.readline()
        assert line.endswith("\n")

    def test_receipt_jsonl_roundtrip(self):
        """Test write and read back receipt as JSON."""
        signal = _make_signal()
        result = qed.qed(signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        buffer = io.StringIO()
        write_receipt_jsonl(receipt, buffer)

        buffer.seek(0)
        loaded = json.loads(buffer.readline())

        assert loaded["ratio"] == receipt.ratio
        assert loaded["H_bits"] == receipt.H_bits
        assert loaded["recall"] == receipt.recall
        assert loaded["params"]["scenario"] == "tesla_fsd"

    def test_jsonl_multiple_receipts(self):
        """Test writing multiple receipts to same buffer."""
        buffer = io.StringIO()

        for i in range(3):
            signal = _make_signal(seed=42 + i)
            result = qed.qed(signal, scenario="tesla_fsd")
            write_receipt_jsonl(result["receipt"], buffer)

        buffer.seek(0)
        lines = buffer.readlines()
        assert len(lines) == 3

        for line in lines:
            loaded = json.loads(line)
            assert "ratio" in loaded
            assert "receipt" not in loaded  # receipt should be flattened


class TestCheckConstraintsNoSympy:
    """Test constraint checking fallback when sympy_constraints unavailable."""

    def test_check_constraints_with_sympy_module(self):
        """Test constraint checking with sympy_constraints available."""
        # Since sympy_constraints module exists, this should return valid result
        verified, violations = check_constraints(12.0, 40.0, "tesla_fsd")
        # With sympy_constraints present and A=12.0 < 14.7, should pass
        assert verified is True
        assert violations == []

    def test_check_constraints_violation_detected(self):
        """Test that violations are detected when amplitude exceeds bound."""
        # A=15.0 exceeds tesla_fsd bound of 14.7
        verified, violations = check_constraints(15.0, 40.0, "tesla_fsd")
        assert verified is False
        assert len(violations) == 1
        assert violations[0]["constraint_id"] == "amplitude_bound_tesla"
        assert violations[0]["value"] == 15.0
        assert violations[0]["bound"] == 14.7

    def test_check_constraints_nan_handling(self):
        """Test that NaN params are flagged as violations."""
        verified, violations = check_constraints(float("nan"), 40.0, "tesla_fsd")
        assert verified is False
        assert len(violations) == 1
        assert violations[0]["constraint_id"] == "finite_params"

    def test_check_constraints_inf_handling(self):
        """Test that Inf params are flagged as violations."""
        verified, violations = check_constraints(float("inf"), 40.0, "tesla_fsd")
        assert verified is False
        assert len(violations) == 1
        assert violations[0]["constraint_id"] == "finite_params"


class TestQEDLatency:
    """Test latency requirements for qed() with receipt generation."""

    def test_qed_latency_under_50ms(self):
        """Verify qed() completes in under 50ms for 1000-sample signal."""
        signal = _make_signal(n=1000)

        start = time.perf_counter()
        _ = qed.qed(signal, scenario="tesla_fsd")
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < 50.0, f"Latency {elapsed_ms:.2f}ms exceeds 50ms"

    def test_qed_latency_under_1s_for_large_window(self):
        """Verify qed() completes in under 1s for 10k-sample signal."""
        signal = _make_signal(n=10000)

        start = time.perf_counter()
        _ = qed.qed(signal, scenario="tesla_fsd")
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < 1000.0, f"Latency {elapsed_ms:.2f}ms exceeds 1000ms"


class TestReceiptViolationsField:
    """Test that violations field is properly populated."""

    def test_receipt_has_violations_field_empty(self):
        """Test violations field exists and is empty list for valid signal."""
        signal = _make_signal(amplitude=12.0)  # Within bounds
        result = qed.qed(signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        assert receipt.violations == []
        assert receipt.verified is True

    def test_receipt_verified_status(self):
        """Test that verified status matches violations state."""
        signal = _make_signal(amplitude=12.0)
        result = qed.qed(signal, scenario="tesla_fsd")
        receipt = result["receipt"]

        # No violations means verified=True
        assert receipt.verified is True
        assert len(receipt.violations) == 0


class TestBackwardCompatibility:
    """Test that v5 API remains unchanged."""

    def test_existing_return_keys_unchanged(self):
        """Verify original return keys still present."""
        signal = _make_signal()
        result = qed.qed(signal, scenario="tesla_fsd")

        # v5 keys must still exist
        assert "ratio" in result
        assert "H_bits" in result
        assert "recall" in result
        assert "savings_M" in result
        assert "trace" in result

    def test_ratio_value_unchanged(self):
        """Verify ratio calculation matches v5 behavior."""
        signal = _make_signal(n=1000)
        result = qed.qed(signal, scenario="tesla_fsd", bit_depth=12)

        # ratio = (1000 * 12) / 200 = 60.0
        assert abs(result["ratio"] - 60.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
