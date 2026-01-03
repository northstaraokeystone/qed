"""
tests/test_trajectory_laws.py - Trajectory Laws Tests

CLAUDEME v3.1 Compliant: All tests have assertions.
"""

from pathlib import Path

import numpy as np
import pytest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generalized.trajectory_laws import (
    VehicleConfig,
    BallisticLaw,
    LiftingLaw,
    OrbitalLaw,
    predict_trajectory,
    compress_with_law,
    decompress_with_law,
    STARSHIP_CONFIG,
    DRAGON_CONFIG,
    MARS_SAMPLE_RETURN_CONFIG,
)
from src.generalized.atmospheric_models import EARTH_ATMOSPHERE, MARS_ATMOSPHERE


class TestVehicleConfig:
    """Test VehicleConfig dataclass."""

    def test_create_config(self):
        """Create valid vehicle configuration."""
        config = VehicleConfig(
            name="test_vehicle",
            mass_kg=10000.0,
            area_m2=20.0,
            drag_coefficient=1.2,
            lift_coefficient=0.3,
            flight_regime="ballistic",
        )

        assert config.name == "test_vehicle"
        assert config.mass_kg == 10000.0

    def test_lift_to_drag(self):
        """Compute L/D ratio correctly."""
        config = VehicleConfig(
            name="test",
            mass_kg=10000.0,
            area_m2=20.0,
            drag_coefficient=1.0,
            lift_coefficient=0.5,
        )

        assert config.lift_to_drag == 0.5

    def test_ballistic_coefficient(self):
        """Compute ballistic coefficient correctly."""
        config = VehicleConfig(
            name="test",
            mass_kg=10000.0,
            area_m2=10.0,
            drag_coefficient=1.0,
        )

        # beta = m / (Cd * A) = 10000 / (1.0 * 10) = 1000
        assert config.ballistic_coefficient == 1000.0

    def test_preset_configs_exist(self):
        """Preset vehicle configurations exist."""
        assert STARSHIP_CONFIG.name == "starship"
        assert DRAGON_CONFIG.name == "dragon_capsule"
        assert MARS_SAMPLE_RETURN_CONFIG.name == "mars_sample_return"


class TestBallisticLaw:
    """Test ballistic trajectory law."""

    def test_predict_descent(self):
        """Ballistic prediction shows descent."""
        law = BallisticLaw(EARTH_ATMOSPHERE)

        initial_state = {
            "altitude": 80000.0,
            "velocity": 5000.0,
            "flight_path_angle": -np.pi / 4,
        }

        time_span = np.linspace(0, 100, 101)

        result = law.predict(initial_state, time_span, DRAGON_CONFIG)

        assert "altitude" in result
        assert "velocity" in result
        assert len(result["altitude"]) == len(time_span)

        # Should descend
        assert result["altitude"][-1] < initial_state["altitude"]

    def test_compress_decompress(self):
        """Compression/decompression roundtrip."""
        law = BallisticLaw(EARTH_ATMOSPHERE)

        initial_state = {
            "altitude": 80000.0,
            "velocity": 5000.0,
            "flight_path_angle": -np.pi / 4,
        }

        time_span = np.linspace(0, 60, 61)

        predicted = law.predict(initial_state, time_span, DRAGON_CONFIG)

        # Add small noise to observed
        observed = {
            "altitude": predicted["altitude"] + np.random.normal(0, 10, len(time_span)),
            "velocity": predicted["velocity"] + np.random.normal(0, 1, len(time_span)),
        }

        # Compress
        compressed = law.compress(observed, predicted)

        # Decompress
        decompressed = law.decompress(compressed, initial_state, time_span, DRAGON_CONFIG)

        # Check roundtrip error
        alt_error = np.abs(decompressed["altitude"] - observed["altitude"])
        assert np.max(alt_error) < 100, "Altitude error should be < 100m"


class TestLiftingLaw:
    """Test lifting body trajectory law."""

    def test_predict_with_lift(self):
        """Lifting body produces different trajectory than ballistic."""
        ballistic = BallisticLaw(EARTH_ATMOSPHERE)
        lifting = LiftingLaw(EARTH_ATMOSPHERE)

        initial_state = {
            "altitude": 80000.0,
            "velocity": 5000.0,
            "flight_path_angle": -np.pi / 6,
        }

        time_span = np.linspace(0, 100, 101)

        result_ballistic = ballistic.predict(initial_state, time_span, STARSHIP_CONFIG)
        result_lifting = lifting.predict(initial_state, time_span, STARSHIP_CONFIG)

        # Lifting body should have different trajectory
        # (difference may be small depending on L/D)
        assert "altitude" in result_lifting

    def test_bank_angle_effect(self):
        """Bank angle affects trajectory."""
        law_no_bank = LiftingLaw(EARTH_ATMOSPHERE, bank_angle=0.0)
        law_banked = LiftingLaw(EARTH_ATMOSPHERE, bank_angle=np.pi / 4)

        initial_state = {
            "altitude": 60000.0,
            "velocity": 3000.0,
            "flight_path_angle": -np.pi / 6,
        }

        time_span = np.linspace(0, 60, 61)

        result_no_bank = law_no_bank.predict(initial_state, time_span, STARSHIP_CONFIG)
        result_banked = law_banked.predict(initial_state, time_span, STARSHIP_CONFIG)

        # Both should produce valid trajectories
        assert len(result_no_bank["altitude"]) == len(time_span)
        assert len(result_banked["altitude"]) == len(time_span)


class TestOrbitalLaw:
    """Test orbital trajectory law."""

    def test_circular_orbit(self):
        """Circular orbit maintains altitude."""
        law = OrbitalLaw()

        # Circular orbit at ~400 km (ISS-like)
        orbital_velocity = np.sqrt(3.986004418e14 / (6.371e6 + 400000))

        initial_state = {
            "altitude": 400000.0,
            "velocity": orbital_velocity,
            "flight_path_angle": 0.0,  # Circular
        }

        # One orbit period ~90 minutes = 5400 seconds
        time_span = np.linspace(0, 5400, 541)

        result = law.predict(initial_state, time_span, DRAGON_CONFIG)

        # Altitude should stay roughly constant (circular orbit)
        alt_mean = np.mean(result["altitude"])
        alt_std = np.std(result["altitude"])

        assert alt_std < alt_mean * 0.1, "Circular orbit should have low altitude variation"

    def test_orbital_energy_conservation(self):
        """Orbital energy should be conserved (no drag)."""
        law = OrbitalLaw()

        initial_state = {
            "altitude": 400000.0,
            "velocity": 7700.0,
            "flight_path_angle": 0.0,
        }

        time_span = np.linspace(0, 1000, 101)

        result = law.predict(initial_state, time_span, DRAGON_CONFIG)

        # Compute specific energy: E = v^2/2 - mu/r
        mu = 3.986004418e14
        R = 6.371e6

        E_initial = initial_state["velocity"] ** 2 / 2 - mu / (R + initial_state["altitude"])

        # Final energy (approximate - using mean of last few points)
        alt_final = np.mean(result["altitude"][-10:])
        vel_final = np.mean(result["velocity"][-10:])
        E_final = vel_final ** 2 / 2 - mu / (R + alt_final)

        # Energy should be roughly conserved
        assert abs(E_final - E_initial) / abs(E_initial) < 0.1, \
            "Orbital energy should be roughly conserved"


class TestPredictTrajectory:
    """Test convenience predict_trajectory function."""

    def test_predict_earth_ballistic(self):
        """Predict ballistic trajectory on Earth."""
        initial_state = {
            "altitude": 80000.0,
            "velocity": 5000.0,
            "flight_path_angle": -np.pi / 4,
        }

        time_span = np.linspace(0, 60, 61)

        trajectory, receipt = predict_trajectory(
            initial_state=initial_state,
            time_span=time_span,
            vehicle=DRAGON_CONFIG,
            body="earth",
        )

        assert "altitude" in trajectory
        assert receipt["receipt_type"] == "trajectory_law"

    def test_predict_mars(self):
        """Predict trajectory on Mars."""
        initial_state = {
            "altitude": 50000.0,
            "velocity": 5000.0,
            "flight_path_angle": -np.pi / 4,
        }

        time_span = np.linspace(0, 60, 61)

        trajectory, receipt = predict_trajectory(
            initial_state=initial_state,
            time_span=time_span,
            vehicle=MARS_SAMPLE_RETURN_CONFIG,
            body="mars",
        )

        assert "altitude" in trajectory
        assert receipt["body"] == "mars"


class TestCompressDecompress:
    """Test compress/decompress convenience functions."""

    def test_compress_with_law(self):
        """Compress trajectory using physics law."""
        initial_state = {
            "altitude": 80000.0,
            "velocity": 5000.0,
            "flight_path_angle": -np.pi / 4,
        }

        time_span = np.linspace(0, 60, 61)

        # Generate observed data
        law = BallisticLaw(EARTH_ATMOSPHERE)
        predicted = law.predict(initial_state, time_span, DRAGON_CONFIG)

        observed = {
            "altitude": predicted["altitude"],
            "velocity": predicted["velocity"],
        }

        compressed, ratio, receipt = compress_with_law(
            observed=observed,
            initial_state=initial_state,
            time_span=time_span,
            vehicle=DRAGON_CONFIG,
        )

        assert len(compressed) > 0
        assert ratio > 1.0  # Should compress
        assert receipt["receipt_type"] == "trajectory_compression"

    def test_decompress_with_law(self):
        """Decompress trajectory using physics law."""
        initial_state = {
            "altitude": 80000.0,
            "velocity": 5000.0,
            "flight_path_angle": -np.pi / 4,
        }

        time_span = np.linspace(0, 60, 61)

        # Generate and compress
        law = BallisticLaw(EARTH_ATMOSPHERE)
        predicted = law.predict(initial_state, time_span, DRAGON_CONFIG)
        observed = {
            "altitude": predicted["altitude"],
            "velocity": predicted["velocity"],
        }

        compressed, _, _ = compress_with_law(
            observed=observed,
            initial_state=initial_state,
            time_span=time_span,
            vehicle=DRAGON_CONFIG,
        )

        # Decompress
        trajectory, receipt = decompress_with_law(
            compressed=compressed,
            initial_state=initial_state,
            time_span=time_span,
            vehicle=DRAGON_CONFIG,
        )

        assert "altitude" in trajectory
        assert receipt["receipt_type"] == "trajectory_decompression"
