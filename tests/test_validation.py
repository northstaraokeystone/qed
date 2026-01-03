"""
tests/test_validation.py - Reconstruction Validation Tests

CLAUDEME v3.1 Compliant: All tests have assertions.

SLO Thresholds:
    - Altitude: <1% or <500m
    - Velocity: <5% or <50 m/s
"""

from pathlib import Path

import numpy as np
import pytest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generalized.validation import (
    compute_rmse,
    compute_mae,
    compute_max_error,
    compute_relative_error,
    compute_metrics,
    check_slo_altitude,
    check_slo_velocity,
    check_slo_acceleration,
    validate_against_slo,
    quick_validate,
    validate_compression_roundtrip,
    ValidationResult,
    SLO_ALTITUDE_ABS_M,
    SLO_ALTITUDE_REL,
    SLO_VELOCITY_ABS_MPS,
    SLO_VELOCITY_REL,
)


class TestRMSE:
    """Test RMSE computation."""

    def test_zero_error(self):
        """Zero RMSE for identical arrays."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rmse = compute_rmse(arr, arr)
        assert rmse == 0.0

    def test_known_rmse(self):
        """Compute RMSE for known values."""
        pred = np.array([1.0, 2.0, 3.0])
        actual = np.array([2.0, 3.0, 4.0])  # Each off by 1

        rmse = compute_rmse(pred, actual)
        assert abs(rmse - 1.0) < 0.001

    def test_rmse_with_variance(self):
        """RMSE is sqrt of mean squared error."""
        pred = np.array([0.0, 0.0, 0.0, 0.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0])

        # MSE = (1 + 4 + 9 + 16) / 4 = 7.5
        # RMSE = sqrt(7.5) = 2.739
        rmse = compute_rmse(pred, actual)
        assert abs(rmse - np.sqrt(7.5)) < 0.001


class TestMAE:
    """Test MAE computation."""

    def test_zero_error(self):
        """Zero MAE for identical arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        mae = compute_mae(arr, arr)
        assert mae == 0.0

    def test_known_mae(self):
        """Compute MAE for known values."""
        pred = np.array([0.0, 0.0, 0.0, 0.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0])

        # MAE = (1 + 2 + 3 + 4) / 4 = 2.5
        mae = compute_mae(pred, actual)
        assert abs(mae - 2.5) < 0.001


class TestMaxError:
    """Test max error computation."""

    def test_max_error(self):
        """Find maximum absolute error."""
        pred = np.array([0.0, 0.0, 0.0])
        actual = np.array([1.0, 5.0, 2.0])

        max_err = compute_max_error(pred, actual)
        assert max_err == 5.0


class TestRelativeError:
    """Test relative error computation."""

    def test_relative_error(self):
        """Compute relative error correctly."""
        pred = np.array([100.0, 100.0, 100.0])
        actual = np.array([110.0, 110.0, 110.0])  # 10 off

        rmse = compute_rmse(pred, actual)  # = 10
        mean_actual = np.mean(np.abs(actual))  # = 110

        rel_error = compute_relative_error(pred, actual)
        expected = rmse / mean_actual

        assert abs(rel_error - expected) < 0.001

    def test_zero_actual_mean(self):
        """Zero actual mean returns 0 relative error."""
        pred = np.array([1.0, 1.0, 1.0])
        actual = np.array([0.0, 0.0, 0.0])

        rel_error = compute_relative_error(pred, actual)
        assert rel_error == 0.0


class TestComputeMetrics:
    """Test combined metrics computation."""

    def test_all_metrics(self):
        """Compute all metrics at once."""
        predicted = {
            "altitude": np.array([100.0, 90.0, 80.0]),
            "velocity": np.array([500.0, 400.0, 300.0]),
            "acceleration": np.array([10.0, 20.0, 30.0]),
        }

        actual = {
            "altitude": np.array([105.0, 95.0, 85.0]),
            "velocity": np.array([510.0, 410.0, 310.0]),
            "acceleration": np.array([11.0, 21.0, 31.0]),
        }

        metrics = compute_metrics(predicted, actual)

        assert "rmse_altitude_m" in metrics
        assert "rmse_velocity_mps" in metrics
        assert "rmse_acceleration_mps2" in metrics
        assert "relative_error_altitude" in metrics


class TestSLOChecks:
    """Test SLO threshold checks."""

    def test_altitude_slo_pass_absolute(self):
        """Altitude passes SLO via absolute threshold."""
        # 400m error, 2% relative - passes because absolute < 500m
        passes = check_slo_altitude(rmse=400.0, relative_error=0.02)
        assert passes is True

    def test_altitude_slo_pass_relative(self):
        """Altitude passes SLO via relative threshold."""
        # 600m error, 0.5% relative - passes because relative < 1%
        passes = check_slo_altitude(rmse=600.0, relative_error=0.005)
        assert passes is True

    def test_altitude_slo_fail(self):
        """Altitude fails SLO when both thresholds exceeded."""
        # 600m error, 2% relative - fails both
        passes = check_slo_altitude(rmse=600.0, relative_error=0.02)
        assert passes is False

    def test_velocity_slo_pass(self):
        """Velocity passes SLO."""
        # 40 m/s error - passes absolute
        passes = check_slo_velocity(rmse=40.0, relative_error=0.1)
        assert passes is True

    def test_velocity_slo_fail(self):
        """Velocity fails SLO."""
        # 60 m/s, 10% - fails both
        passes = check_slo_velocity(rmse=60.0, relative_error=0.1)
        assert passes is False

    def test_acceleration_slo(self):
        """Acceleration SLO check."""
        passes = check_slo_acceleration(relative_error=0.05)
        assert passes is True

        fails = check_slo_acceleration(relative_error=0.15)
        assert fails is False


class TestValidateAgainstSLO:
    """Test full SLO validation."""

    def test_perfect_match(self):
        """Perfect reconstruction passes SLO."""
        altitude = np.array([80000.0, 75000.0, 70000.0])
        velocity = np.array([5000.0, 4500.0, 4000.0])

        predicted = {"altitude": altitude, "velocity": velocity}
        actual = {"altitude": altitude, "velocity": velocity}

        result, receipt = validate_against_slo(predicted, actual)

        assert result.slo_overall_pass is True
        assert result.rmse_altitude_m < 1.0
        assert receipt["slo_pass"] is True

    def test_small_error_passes(self):
        """Small reconstruction error passes SLO."""
        altitude_true = np.array([80000.0, 75000.0, 70000.0])
        velocity_true = np.array([5000.0, 4500.0, 4000.0])

        # Add small error
        altitude_pred = altitude_true + 100  # 100m error
        velocity_pred = velocity_true + 10  # 10 m/s error

        predicted = {"altitude": altitude_pred, "velocity": velocity_pred}
        actual = {"altitude": altitude_true, "velocity": velocity_true}

        result, receipt = validate_against_slo(predicted, actual)

        assert result.slo_altitude_pass is True  # 100m < 500m
        assert result.slo_velocity_pass is True  # 10 m/s < 50 m/s
        assert result.slo_overall_pass is True

    def test_large_error_fails(self):
        """Large reconstruction error fails SLO."""
        altitude_true = np.array([80000.0, 75000.0, 70000.0])
        velocity_true = np.array([5000.0, 4500.0, 4000.0])

        # Add large error - must exceed BOTH absolute AND relative thresholds
        # Altitude: need > 500m AND > 1%
        # 1% of ~75000 = 750m, so 1000m error gives ~1.3% relative
        altitude_pred = altitude_true + 1000  # 1000m error > 500m AND ~1.3% > 1%

        # Velocity: need > 50 m/s AND > 5%
        # 5% of ~4500 = 225 m/s, so use 300 m/s error
        velocity_pred = velocity_true + 300  # 300 m/s > 50 m/s AND ~6.7% > 5%

        predicted = {"altitude": altitude_pred, "velocity": velocity_pred}
        actual = {"altitude": altitude_true, "velocity": velocity_true}

        result, receipt = validate_against_slo(predicted, actual)

        # Both should fail (exceed both absolute and relative thresholds)
        assert result.slo_altitude_pass is False
        assert result.slo_velocity_pass is False
        assert result.slo_overall_pass is False

    def test_anomalies_detected(self):
        """Anomalies are detected and listed."""
        altitude_true = np.array([80000.0, 75000.0, 70000.0])
        velocity_true = np.array([5000.0, 4500.0, 4000.0])

        # Very large error
        altitude_pred = altitude_true + 2000
        velocity_pred = velocity_true + 200

        predicted = {"altitude": altitude_pred, "velocity": velocity_pred}
        actual = {"altitude": altitude_true, "velocity": velocity_true}

        result, receipt = validate_against_slo(predicted, actual)

        assert len(result.anomalies_detected) > 0
        assert "altitude_error_exceeds_slo" in result.anomalies_detected or \
               "velocity_error_exceeds_slo" in result.anomalies_detected

    def test_custom_thresholds(self):
        """Custom SLO thresholds work."""
        altitude = np.array([80000.0, 75000.0, 70000.0])
        velocity = np.array([5000.0, 4500.0, 4000.0])

        # Add error that exceeds BOTH custom absolute AND custom relative thresholds
        # 200m error on ~75000m mean = ~0.27% relative, which is < 1%
        # So need to also set strict relative threshold
        altitude_pred = altitude + 200

        predicted = {"altitude": altitude_pred, "velocity": velocity}
        actual = {"altitude": altitude, "velocity": velocity}

        # Strict thresholds: 100m absolute AND 0.1% relative
        # 200m > 100m AND 0.27% > 0.1%, so should fail both
        result, _ = validate_against_slo(
            predicted, actual,
            custom_thresholds={"altitude_abs_m": 100.0, "altitude_rel": 0.001}
        )

        # Should fail with stricter thresholds
        assert result.slo_altitude_pass is False


class TestQuickValidate:
    """Test quick validation function."""

    def test_quick_validate(self):
        """Quick validate returns metrics dict."""
        altitude_pred = np.array([100.0, 90.0, 80.0])
        velocity_pred = np.array([500.0, 400.0, 300.0])
        altitude_true = np.array([105.0, 95.0, 85.0])
        velocity_true = np.array([510.0, 410.0, 310.0])

        metrics = quick_validate(
            altitude_pred, velocity_pred,
            altitude_true, velocity_true
        )

        assert "rmse_altitude_m" in metrics
        assert "rmse_velocity_mps" in metrics
        assert metrics["rmse_altitude_m"] == 5.0
        assert metrics["rmse_velocity_mps"] == 10.0


class TestCompressionRoundtrip:
    """Test compression roundtrip validation."""

    def test_lossless_roundtrip(self):
        """Lossless roundtrip is detected."""
        original = {
            "altitude": np.array([100.0, 90.0, 80.0]),
            "velocity": np.array([500.0, 400.0, 300.0]),
        }

        # Perfect decompression
        decompressed = {
            "altitude": original["altitude"].copy(),
            "velocity": original["velocity"].copy(),
        }

        is_lossless, receipt = validate_compression_roundtrip(decompressed, original)

        assert is_lossless is True

    def test_lossy_roundtrip(self):
        """Lossy roundtrip is detected."""
        original = {
            "altitude": np.array([100.0, 90.0, 80.0]),
            "velocity": np.array([500.0, 400.0, 300.0]),
        }

        # Lossy decompression
        decompressed = {
            "altitude": original["altitude"] + 10,
            "velocity": original["velocity"] + 5,
        }

        is_lossless, receipt = validate_compression_roundtrip(decompressed, original)

        assert is_lossless is False


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_result_structure(self):
        """ValidationResult has expected fields."""
        result = ValidationResult(
            rmse_altitude_m=100.0,
            rmse_velocity_mps=10.0,
            rmse_acceleration_mps2=1.0,
            mae_altitude_m=80.0,
            mae_velocity_mps=8.0,
            mae_acceleration_mps2=0.8,
            max_error_altitude_m=200.0,
            max_error_velocity_mps=20.0,
            max_error_acceleration_mps2=2.0,
            relative_error_altitude=0.001,
            relative_error_velocity=0.002,
            relative_error_acceleration=0.01,
            slo_altitude_pass=True,
            slo_velocity_pass=True,
            slo_acceleration_pass=True,
            slo_overall_pass=True,
            anomalies_detected=(),
            ground_truth_hash="hash1",
            reconstruction_hash="hash2",
        )

        assert result.rmse_altitude_m == 100.0
        assert result.slo_overall_pass is True
        assert len(result.anomalies_detected) == 0
