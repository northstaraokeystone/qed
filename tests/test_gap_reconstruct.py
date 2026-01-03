"""
tests/test_gap_reconstruct.py - Gap Reconstruction Tests

CLAUDEME v3.1 Compliant: All tests have assertions.
SLO: error < 5% for gaps <= 60s
"""

from pathlib import Path

import numpy as np
import pytest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spaceflight.telemetry_ingest import TelemetryStream
from src.spaceflight.physics_discovery import (
    integrate_drag_trajectory,
    discover_law,
    PhysicsLaw,
)
from src.spaceflight.gap_reconstruct import (
    StateVector,
    ReconstructionResult,
    reconstruct_gap,
    reconstruct_forward,
    reconstruct_backward,
    reconstruct_bidirectional,
    validate_reconstruction,
    compute_confidence,
    get_error_slo,
)


class TestStateVector:
    """Test StateVector dataclass."""

    def test_create_state_vector(self):
        """Create valid state vector."""
        state = StateVector(
            time=100.0,
            altitude=80000.0,
            velocity=5000.0,
            acceleration=30.0,
        )

        assert state.time == 100.0
        assert state.altitude == 80000.0
        assert state.velocity == 5000.0


class TestConfidenceScoring:
    """Test confidence score computation."""

    def test_short_gap_high_confidence(self):
        """Short gaps (<30s) have high confidence."""
        confidence = compute_confidence(20.0)
        assert confidence >= 0.95, f"20s gap should have >0.95 confidence, got {confidence}"

    def test_medium_gap_medium_confidence(self):
        """Medium gaps (30-90s) have medium confidence."""
        confidence = compute_confidence(60.0)
        assert 0.80 <= confidence <= 0.95, f"60s gap confidence should be 0.80-0.95"

    def test_long_gap_low_confidence(self):
        """Long gaps (>90s) have low confidence."""
        confidence = compute_confidence(120.0)
        assert confidence <= 0.80, f"120s gap should have <0.80 confidence"

    def test_convergence_error_reduces_confidence(self):
        """High convergence error reduces confidence."""
        conf_no_error = compute_confidence(30.0, convergence_error=None)
        conf_with_error = compute_confidence(30.0, convergence_error=0.2)

        assert conf_with_error < conf_no_error


class TestErrorSLO:
    """Test error SLO thresholds."""

    def test_60s_slo(self):
        """60s gap has 5% error SLO."""
        slo = get_error_slo(60.0)
        assert slo == 0.05

    def test_120s_slo(self):
        """120s gap has 10% error SLO."""
        slo = get_error_slo(120.0)
        assert slo == 0.10


class TestReconstructForward:
    """Test forward reconstruction."""

    def test_forward_produces_output(self):
        """Forward reconstruction produces trajectory."""
        # Create a simple physics law
        law = PhysicsLaw(
            law_type="ballistic",
            equation="test",
            parameters={
                "Cd": 1.5,
                "mass_kg": 100000.0,
                "area_m2": 300.0,
                "lift_to_drag": 0.0,
            },
            compression_ratio=20.0,
            residual_rms=100.0,
            applicability_range={},
            mission_id="test",
            payload_hash="test_hash",
        )

        pre_gap = StateVector(
            time=0.0,
            altitude=80000.0,
            velocity=5000.0,
            acceleration=0.0,
        )

        time, alt, vel, acc = reconstruct_forward(
            pre_gap_state=pre_gap,
            gap_duration=30.0,
            law=law,
            sample_rate_hz=1.0,
        )

        assert len(time) > 0
        assert len(alt) == len(time)
        assert len(vel) == len(time)
        assert alt[-1] < pre_gap.altitude  # Should descend


class TestReconstructBidirectional:
    """Test bidirectional reconstruction."""

    def test_bidirectional_blends(self):
        """Bidirectional reconstruction blends forward and backward."""
        law = PhysicsLaw(
            law_type="ballistic",
            equation="test",
            parameters={
                "Cd": 1.5,
                "mass_kg": 100000.0,
                "area_m2": 300.0,
                "lift_to_drag": 0.0,
            },
            compression_ratio=20.0,
            residual_rms=100.0,
            applicability_range={},
            mission_id="test",
            payload_hash="test_hash",
        )

        # Generate consistent pre/post states
        time_full = np.linspace(0, 60, 61)
        alt_full, vel_full = integrate_drag_trajectory(
            initial_altitude=80000.0,
            initial_velocity=5000.0,
            time_span=time_full,
            Cd=1.5,
            mass=100000.0,
            area=300.0,
            lift_to_drag=0.0,
        )

        pre_gap = StateVector(
            time=0.0,
            altitude=float(alt_full[0]),
            velocity=float(vel_full[0]),
            acceleration=0.0,
        )

        post_gap = StateVector(
            time=60.0,
            altitude=float(alt_full[-1]),
            velocity=float(vel_full[-1]),
            acceleration=0.0,
        )

        time, alt, vel, acc, conv_error = reconstruct_bidirectional(
            pre_gap_state=pre_gap,
            post_gap_state=post_gap,
            law=law,
            sample_rate_hz=1.0,
        )

        # Check output
        assert len(time) > 0
        assert conv_error >= 0  # Convergence error is non-negative


class TestReconstructGap:
    """Test main gap reconstruction function."""

    def test_reconstruct_gap_forward(self):
        """Reconstruct gap with forward method."""
        law = PhysicsLaw(
            law_type="ballistic",
            equation="test",
            parameters={
                "Cd": 1.5,
                "mass_kg": 100000.0,
                "area_m2": 300.0,
                "lift_to_drag": 0.0,
            },
            compression_ratio=20.0,
            residual_rms=100.0,
            applicability_range={},
            mission_id="test",
            payload_hash="test_hash",
        )

        pre_gap = StateVector(
            time=0.0,
            altitude=80000.0,
            velocity=5000.0,
            acceleration=0.0,
        )

        post_gap = StateVector(
            time=60.0,
            altitude=60000.0,
            velocity=3000.0,
            acceleration=0.0,
        )

        result, receipt = reconstruct_gap(
            pre_gap_state=pre_gap,
            post_gap_state=post_gap,
            law=law,
            method="forward",
        )

        # Check result structure
        assert isinstance(result, ReconstructionResult)
        assert result.gap_duration == 60.0
        assert result.method == "forward"
        assert 0.0 <= result.confidence_score <= 1.0

        # Check receipt
        assert receipt["receipt_type"] == "gap_reconstruction"
        assert "reconstruction_hash" in receipt

    def test_reconstruct_gap_bidirectional(self):
        """Reconstruct gap with bidirectional method."""
        law = PhysicsLaw(
            law_type="ballistic",
            equation="test",
            parameters={
                "Cd": 1.5,
                "mass_kg": 100000.0,
                "area_m2": 300.0,
                "lift_to_drag": 0.0,
            },
            compression_ratio=20.0,
            residual_rms=100.0,
            applicability_range={},
            mission_id="test",
            payload_hash="test_hash",
        )

        # Use consistent states from integration
        time_full = np.linspace(0, 60, 61)
        alt_full, vel_full = integrate_drag_trajectory(
            initial_altitude=80000.0,
            initial_velocity=5000.0,
            time_span=time_full,
            Cd=1.5,
            mass=100000.0,
            area=300.0,
            lift_to_drag=0.0,
        )

        pre_gap = StateVector(
            time=0.0,
            altitude=float(alt_full[0]),
            velocity=float(vel_full[0]),
            acceleration=0.0,
        )

        post_gap = StateVector(
            time=60.0,
            altitude=float(alt_full[-1]),
            velocity=float(vel_full[-1]),
            acceleration=0.0,
        )

        result, receipt = reconstruct_gap(
            pre_gap_state=pre_gap,
            post_gap_state=post_gap,
            law=law,
            method="bidirectional",
        )

        assert result.method == "bidirectional"


class TestValidateReconstruction:
    """Test reconstruction validation."""

    def test_validate_perfect_reconstruction(self):
        """Perfect reconstruction has zero error."""
        # Ground truth
        altitude = np.array([80000, 75000, 70000, 65000, 60000], dtype=float)
        velocity = np.array([5000, 4500, 4000, 3500, 3000], dtype=float)

        # Create result with same values
        result = ReconstructionResult(
            gap_start_time=0.0,
            gap_end_time=40.0,
            gap_duration=40.0,
            time=np.arange(5, dtype=float),
            altitude=altitude.copy(),
            velocity=velocity.copy(),
            acceleration=np.zeros(5),
            method="forward",
            confidence_score=0.95,
            estimated_error=0.0,
            validation_available=False,
            actual_error=None,
            pre_gap_state_hash="test",
            post_gap_state_hash="test",
            physics_law_hash="test",
            reconstruction_hash="test",
        )

        metrics, receipt = validate_reconstruction(
            result=result,
            ground_truth_altitude=altitude,
            ground_truth_velocity=velocity,
        )

        # Perfect match should have zero RMSE
        assert metrics["altitude_rmse_m"] < 1.0
        assert metrics["velocity_rmse_mps"] < 1.0

    def test_validate_with_error(self):
        """Validation detects reconstruction error."""
        # Ground truth
        altitude_true = np.array([80000, 75000, 70000, 65000, 60000], dtype=float)
        velocity_true = np.array([5000, 4500, 4000, 3500, 3000], dtype=float)

        # Reconstructed with error
        altitude_recon = altitude_true + 500  # 500m error
        velocity_recon = velocity_true + 50  # 50 m/s error

        result = ReconstructionResult(
            gap_start_time=0.0,
            gap_end_time=40.0,
            gap_duration=40.0,
            time=np.arange(5, dtype=float),
            altitude=altitude_recon,
            velocity=velocity_recon,
            acceleration=np.zeros(5),
            method="forward",
            confidence_score=0.95,
            estimated_error=0.0,
            validation_available=False,
            actual_error=None,
            pre_gap_state_hash="test",
            post_gap_state_hash="test",
            physics_law_hash="test",
            reconstruction_hash="test",
        )

        metrics, receipt = validate_reconstruction(
            result=result,
            ground_truth_altitude=altitude_true,
            ground_truth_velocity=velocity_true,
        )

        # Should detect the error
        assert metrics["altitude_rmse_m"] > 400
        assert metrics["velocity_rmse_mps"] > 40
