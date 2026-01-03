"""
src/spaceflight/telemetry_ingest.py - Telemetry Ingestion Module

Ingest spaceflight telemetry from multiple sources into standardized format.
CLAUDEME v3.1 Compliant: All functions emit receipts.

Inputs:
    - Raw telemetry data (CSV, JSON, or video OCR output)
    - Source metadata (mission ID, vehicle type, phase)
    - Sampling rate and units

Outputs:
    - Standardized telemetry stream (time, altitude, velocity, acceleration)
    - ingest_receipt with dual-hash of raw data
    - Metadata receipt with units, coordinate frames, epoch

Receipt: telemetry_ingest_receipt
SLO: ingest_latency < 100ms
"""

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from receipts import dual_hash, emit_receipt, StopRule


# =============================================================================
# CONSTANTS
# =============================================================================

# SI unit conversion factors
UNIT_CONVERSIONS = {
    # Length to meters
    "km": 1000.0,
    "m": 1.0,
    "ft": 0.3048,
    "mi": 1609.34,
    "nmi": 1852.0,
    # Velocity to m/s
    "km/s": 1000.0,
    "m/s": 1.0,
    "ft/s": 0.3048,
    "mph": 0.44704,
    "kts": 0.514444,
    "mach": 343.0,  # Approximate at sea level
    # Acceleration to m/s^2
    "m/s^2": 1.0,
    "g": 9.80665,
    "ft/s^2": 0.3048,
    # Angle to radians
    "deg": np.pi / 180.0,
    "rad": 1.0,
}

# Default parameters for common telemetry streams
TELEMETRY_DEFAULTS = {
    "ift6": {
        "sample_rate_hz": 1.0,
        "altitude_unit": "km",
        "velocity_unit": "km/s",
        "acceleration_unit": "g",
    },
    "dragon": {
        "sample_rate_hz": 10.0,
        "altitude_unit": "km",
        "velocity_unit": "m/s",
        "acceleration_unit": "g",
    },
    "starship": {
        "sample_rate_hz": 10.0,
        "altitude_unit": "km",
        "velocity_unit": "km/s",
        "acceleration_unit": "g",
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class TelemetryStream:
    """
    Standardized telemetry stream in SI units.

    All values are in SI units:
        - time: seconds from epoch
        - altitude: meters
        - velocity: m/s
        - acceleration: m/s^2
        - attitude: radians (if available)
    """
    mission_id: str
    vehicle_type: str
    flight_phase: str
    epoch: str  # ISO8601 timestamp

    # Core arrays (all in SI units)
    time: np.ndarray  # seconds from epoch
    altitude: np.ndarray  # meters
    velocity: np.ndarray  # m/s
    acceleration: np.ndarray  # m/s^2

    # Optional arrays
    attitude: Optional[np.ndarray] = None  # radians (pitch, yaw, roll)

    # Metadata
    sample_rate_hz: float = 1.0
    duration_seconds: float = 0.0
    raw_data_hash: str = ""
    discontinuities: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Validate array shapes."""
        n = len(self.time)
        if len(self.altitude) != n or len(self.velocity) != n or len(self.acceleration) != n:
            raise ValueError("All telemetry arrays must have the same length")
        if self.attitude is not None and len(self.attitude) != n:
            raise ValueError("Attitude array must have same length as other arrays")


# =============================================================================
# UNIT CONVERSION
# =============================================================================

def normalize_units(
    value: Union[float, np.ndarray],
    from_unit: str,
    to_base: str = "si"
) -> Union[float, np.ndarray]:
    """
    Convert value from given unit to SI base unit.

    Args:
        value: Scalar or array to convert
        from_unit: Source unit (e.g., "km", "ft", "g")
        to_base: Target base ("si" only for now)

    Returns:
        Converted value in SI units

    Raises:
        ValueError: If unit is unknown
    """
    if from_unit not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown unit: {from_unit}. Known: {list(UNIT_CONVERSIONS.keys())}")

    factor = UNIT_CONVERSIONS[from_unit]
    return value * factor


# =============================================================================
# DISCONTINUITY DETECTION
# =============================================================================

def detect_discontinuities(
    time_array: np.ndarray,
    expected_dt: float,
    tolerance: float = 0.5
) -> List[int]:
    """
    Detect discontinuities (gaps) in time series.

    Args:
        time_array: Array of timestamps
        expected_dt: Expected time step between samples
        tolerance: Fraction of expected_dt to allow as variation

    Returns:
        List of indices where discontinuities occur
    """
    if len(time_array) < 2:
        return []

    dt = np.diff(time_array)
    threshold = expected_dt * (1 + tolerance)

    gaps = np.where(dt > threshold)[0]
    return list(gaps + 1)  # Return index after gap


# =============================================================================
# CSV INGESTION
# =============================================================================

def _read_csv_telemetry(
    path: Path,
    time_col: str = "time",
    altitude_col: str = "altitude",
    velocity_col: str = "velocity",
    acceleration_col: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Read telemetry columns from CSV file.

    Args:
        path: Path to CSV file
        time_col: Column name for time
        altitude_col: Column name for altitude
        velocity_col: Column name for velocity
        acceleration_col: Column name for acceleration (optional)

    Returns:
        Dict with arrays for each column
    """
    data = {
        "time": [],
        "altitude": [],
        "velocity": [],
        "acceleration": [],
    }

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate columns exist
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row")

        required = [time_col, altitude_col, velocity_col]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Required column '{col}' not found in CSV. Found: {reader.fieldnames}")

        for row in reader:
            try:
                data["time"].append(float(row[time_col]))
                data["altitude"].append(float(row[altitude_col]))
                data["velocity"].append(float(row[velocity_col]))

                if acceleration_col and acceleration_col in reader.fieldnames:
                    data["acceleration"].append(float(row[acceleration_col]))
                else:
                    data["acceleration"].append(0.0)
            except (TypeError, ValueError):
                continue  # Skip malformed rows

    return {k: np.array(v) for k, v in data.items()}


# =============================================================================
# JSON INGESTION
# =============================================================================

def _read_json_telemetry(
    path: Path,
    data_key: str = "telemetry",
) -> Dict[str, np.ndarray]:
    """
    Read telemetry from JSON file.

    Expected format:
    {
        "telemetry": [
            {"time": 0.0, "altitude": 1000, "velocity": 100, "acceleration": 3.0},
            ...
        ]
    }
    """
    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)

    records = content.get(data_key, content if isinstance(content, list) else [])

    data = {
        "time": [],
        "altitude": [],
        "velocity": [],
        "acceleration": [],
    }

    for rec in records:
        data["time"].append(float(rec.get("time", 0)))
        data["altitude"].append(float(rec.get("altitude", 0)))
        data["velocity"].append(float(rec.get("velocity", 0)))
        data["acceleration"].append(float(rec.get("acceleration", 0)))

    return {k: np.array(v) for k, v in data.items()}


# =============================================================================
# MAIN INGEST FUNCTION
# =============================================================================

def ingest_telemetry(
    source: Union[str, Path, Dict[str, Any]],
    mission_id: str,
    vehicle_type: str = "starship",
    flight_phase: str = "reentry",
    tenant_id: str = "default",
    # Unit specifications
    altitude_unit: str = "km",
    velocity_unit: str = "km/s",
    acceleration_unit: str = "g",
    sample_rate_hz: Optional[float] = None,
    # Column mappings (for CSV)
    time_col: str = "time",
    altitude_col: str = "altitude",
    velocity_col: str = "velocity",
    acceleration_col: Optional[str] = None,
) -> Tuple[TelemetryStream, Dict[str, Any]]:
    """
    Ingest spaceflight telemetry into standardized format.

    Supports:
        - CSV files (path ending in .csv)
        - JSON files (path ending in .json)
        - Dict with numpy arrays

    Args:
        source: Path to file or dict with arrays
        mission_id: Unique mission identifier (e.g., "ift6")
        vehicle_type: Vehicle type (e.g., "starship", "dragon")
        flight_phase: Flight phase (e.g., "reentry", "launch", "landing")
        tenant_id: Tenant ID for receipt
        altitude_unit: Input altitude unit
        velocity_unit: Input velocity unit
        acceleration_unit: Input acceleration unit
        sample_rate_hz: Override sample rate (auto-detected if None)
        time_col: CSV column name for time
        altitude_col: CSV column name for altitude
        velocity_col: CSV column name for velocity
        acceleration_col: CSV column name for acceleration

    Returns:
        Tuple of (TelemetryStream, ingest_receipt)

    Raises:
        StopRule: If ingest fails SLO thresholds
    """
    start_time = time.time()

    # Load raw data
    if isinstance(source, dict):
        raw_data = source
        raw_bytes = json.dumps({k: v.tolist() for k, v in source.items()}, sort_keys=True).encode()
    elif isinstance(source, (str, Path)):
        path = Path(source)
        raw_bytes = path.read_bytes()

        if path.suffix.lower() == ".csv":
            raw_data = _read_csv_telemetry(
                path,
                time_col=time_col,
                altitude_col=altitude_col,
                velocity_col=velocity_col,
                acceleration_col=acceleration_col,
            )
        elif path.suffix.lower() == ".json":
            raw_data = _read_json_telemetry(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    raw_hash = dual_hash(raw_bytes)

    # Extract arrays
    time_arr = raw_data["time"]
    alt_raw = raw_data["altitude"]
    vel_raw = raw_data["velocity"]
    acc_raw = raw_data["acceleration"]

    if len(time_arr) == 0:
        receipt = emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "ingest_empty",
            "delta": -1,
            "action": "halt",
        })
        raise StopRule(f"Empty telemetry data for {mission_id}")

    # Normalize units to SI
    altitude = normalize_units(alt_raw, altitude_unit)
    velocity = normalize_units(vel_raw, velocity_unit)
    acceleration = normalize_units(acc_raw, acceleration_unit)

    # Detect sample rate if not provided
    if sample_rate_hz is None:
        if len(time_arr) > 1:
            dt_median = np.median(np.diff(time_arr))
            sample_rate_hz = 1.0 / dt_median if dt_median > 0 else 1.0
        else:
            sample_rate_hz = 1.0

    # Detect discontinuities
    expected_dt = 1.0 / sample_rate_hz
    discontinuities = detect_discontinuities(time_arr, expected_dt)

    # Create epoch (first timestamp)
    epoch = datetime.now(timezone.utc).isoformat()

    # Calculate duration
    duration_seconds = float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0

    # Build telemetry stream
    stream = TelemetryStream(
        mission_id=mission_id,
        vehicle_type=vehicle_type,
        flight_phase=flight_phase,
        epoch=epoch,
        time=time_arr,
        altitude=altitude,
        velocity=velocity,
        acceleration=acceleration,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
        raw_data_hash=raw_hash,
        discontinuities=tuple(discontinuities),
    )

    # Check SLO: ingest latency < 100ms
    latency_ms = (time.time() - start_time) * 1000

    # Emit ingest receipt
    receipt = emit_receipt("telemetry_ingest", {
        "tenant_id": tenant_id,
        "mission_id": mission_id,
        "vehicle_type": vehicle_type,
        "data_source": str(source) if isinstance(source, (str, Path)) else "dict",
        "raw_data_hash": raw_hash,
        "normalized_units": {
            "altitude": "m",
            "velocity": "m/s",
            "acceleration": "m/s^2",
        },
        "sampling_rate_hz": sample_rate_hz,
        "duration_seconds": duration_seconds,
        "n_samples": len(time_arr),
        "n_discontinuities": len(discontinuities),
        "latency_ms": latency_ms,
        "slo_pass": latency_ms < 100,
    })

    if latency_ms >= 100:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "ingest_latency",
            "baseline": 100.0,
            "delta": latency_ms - 100.0,
            "classification": "violation",
            "action": "alert",
        })

    return stream, receipt


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def ingest_ift6_sample(path: Union[str, Path], tenant_id: str = "default") -> Tuple[TelemetryStream, Dict[str, Any]]:
    """
    Convenience function to ingest IFT-6 style telemetry.

    IFT-6 format:
        - time: seconds from T+0
        - altitude: km
        - velocity: km/s (or m/s)
        - g_load: g's
    """
    return ingest_telemetry(
        source=path,
        mission_id="ift6",
        vehicle_type="starship",
        flight_phase="reentry",
        tenant_id=tenant_id,
        altitude_unit="km",
        velocity_unit="km/s",
        acceleration_unit="g",
        time_col="time",
        altitude_col="altitude",
        velocity_col="velocity",
        acceleration_col="g_load",
    )
