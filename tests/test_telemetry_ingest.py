"""
tests/test_telemetry_ingest.py - Telemetry Ingest Tests

CLAUDEME v3.1 Compliant: All tests have assertions.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spaceflight.telemetry_ingest import (
    ingest_telemetry,
    normalize_units,
    detect_discontinuities,
    TelemetryStream,
    ingest_ift6_sample,
)


class TestNormalizeUnits:
    """Test unit normalization."""

    def test_km_to_m(self):
        """Convert kilometers to meters."""
        result = normalize_units(10.0, "km")
        assert result == 10000.0, f"10 km should be 10000 m, got {result}"

    def test_ft_to_m(self):
        """Convert feet to meters."""
        result = normalize_units(1.0, "ft")
        assert abs(result - 0.3048) < 0.0001, f"1 ft should be ~0.3048 m"

    def test_g_to_mps2(self):
        """Convert g's to m/s^2."""
        result = normalize_units(1.0, "g")
        assert abs(result - 9.80665) < 0.0001, f"1 g should be ~9.8 m/s^2"

    def test_array_conversion(self):
        """Convert numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = normalize_units(arr, "km")
        assert np.allclose(result, [1000, 2000, 3000])

    def test_unknown_unit_raises(self):
        """Unknown unit raises ValueError."""
        with pytest.raises(ValueError):
            normalize_units(1.0, "unknown_unit")


class TestDiscontinuityDetection:
    """Test gap detection in time series."""

    def test_no_gaps(self):
        """No gaps in regular time series."""
        time = np.arange(0, 10, 1.0)
        gaps = detect_discontinuities(time, expected_dt=1.0)
        assert len(gaps) == 0, f"Should find no gaps, found {len(gaps)}"

    def test_single_gap(self):
        """Detect single gap in time series."""
        time = np.array([0, 1, 2, 5, 6, 7])  # Gap between 2 and 5
        gaps = detect_discontinuities(time, expected_dt=1.0)
        assert len(gaps) == 1, f"Should find 1 gap, found {len(gaps)}"
        assert gaps[0] == 3, f"Gap should be at index 3, got {gaps[0]}"

    def test_multiple_gaps(self):
        """Detect multiple gaps."""
        time = np.array([0, 1, 5, 6, 10, 11])  # Gaps at 2 and 5
        gaps = detect_discontinuities(time, expected_dt=1.0)
        assert len(gaps) == 2, f"Should find 2 gaps, found {len(gaps)}"


class TestIngestTelemetry:
    """Test telemetry ingestion."""

    def test_ingest_csv(self):
        """Ingest CSV telemetry file."""
        # Use the sample IFT-6 data
        csv_path = Path(__file__).parent.parent / "data" / "ift6_telemetry.csv"

        if csv_path.exists():
            stream, receipt = ingest_telemetry(
                source=csv_path,
                mission_id="ift6_test",
                vehicle_type="starship",
                altitude_unit="km",
                velocity_unit="km/s",
                acceleration_unit="g",
                acceleration_col="g_load",
            )

            # Check stream structure
            assert stream.mission_id == "ift6_test"
            assert stream.vehicle_type == "starship"
            assert len(stream.time) > 0, "Should have time data"
            assert len(stream.altitude) == len(stream.time), "Altitude length mismatch"
            assert len(stream.velocity) == len(stream.time), "Velocity length mismatch"

            # Check receipt
            assert receipt["receipt_type"] == "telemetry_ingest"
            assert "raw_data_hash" in receipt
            assert receipt["slo_pass"] is True  # Latency should be < 100ms

    def test_ingest_dict(self):
        """Ingest from dictionary."""
        data = {
            "time": np.array([0, 1, 2, 3, 4]),
            "altitude": np.array([100, 90, 80, 70, 60]),  # km
            "velocity": np.array([7, 6.5, 6, 5.5, 5]),  # km/s
            "acceleration": np.array([1, 1.5, 2, 2.5, 3]),  # g
        }

        stream, receipt = ingest_telemetry(
            source=data,
            mission_id="dict_test",
            altitude_unit="km",
            velocity_unit="km/s",
            acceleration_unit="g",
        )

        # Check normalization to SI
        assert stream.altitude[0] == 100000.0, "Altitude should be in meters"
        assert stream.velocity[0] == 7000.0, "Velocity should be in m/s"
        assert abs(stream.acceleration[0] - 9.80665) < 0.01, "Accel should be in m/s^2"

    def test_ingest_latency_slo(self):
        """Verify ingest latency SLO check."""
        data = {
            "time": np.arange(100),
            "altitude": np.random.rand(100) * 100,
            "velocity": np.random.rand(100) * 10,
            "acceleration": np.random.rand(100),
        }

        stream, receipt = ingest_telemetry(
            source=data,
            mission_id="latency_test",
            altitude_unit="km",
            velocity_unit="km/s",
            acceleration_unit="g",
        )

        assert "latency_ms" in receipt
        assert receipt["latency_ms"] >= 0


class TestTelemetryStream:
    """Test TelemetryStream dataclass."""

    def test_stream_creation(self):
        """Create valid telemetry stream."""
        stream = TelemetryStream(
            mission_id="test",
            vehicle_type="capsule",
            flight_phase="reentry",
            epoch="2024-01-01T00:00:00Z",
            time=np.array([0, 1, 2]),
            altitude=np.array([100, 90, 80]),
            velocity=np.array([7, 6, 5]),
            acceleration=np.array([1, 2, 3]),
            sample_rate_hz=1.0,
            duration_seconds=2.0,
        )

        assert stream.mission_id == "test"
        assert len(stream.time) == 3

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths raise error."""
        with pytest.raises(ValueError):
            TelemetryStream(
                mission_id="test",
                vehicle_type="capsule",
                flight_phase="reentry",
                epoch="2024-01-01T00:00:00Z",
                time=np.array([0, 1, 2]),
                altitude=np.array([100, 90]),  # Wrong length
                velocity=np.array([7, 6, 5]),
                acceleration=np.array([1, 2, 3]),
            )


class TestIFT6Sample:
    """Test IFT-6 sample ingestion."""

    def test_ift6_sample_exists(self):
        """Verify IFT-6 sample data file exists."""
        csv_path = Path(__file__).parent.parent / "data" / "ift6_telemetry.csv"
        assert csv_path.exists(), f"IFT-6 sample not found at {csv_path}"

    def test_ift6_sample_format(self):
        """Verify IFT-6 sample has correct format."""
        csv_path = Path(__file__).parent.parent / "data" / "ift6_telemetry.csv"

        if csv_path.exists():
            with open(csv_path) as f:
                header = f.readline().strip()
                assert "time" in header
                assert "altitude" in header
                assert "velocity" in header
