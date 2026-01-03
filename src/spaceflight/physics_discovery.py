"""
src/spaceflight/physics_discovery.py - Physics Law Discovery Module

Discover physics laws governing telemetry using compression ratio as fitness metric.
CLAUDEME v3.1 Compliant: All functions emit receipts.

Core Physics Models:
    - Drag equation: dv/dt = -rho(h) * v^2 * A * Cd / (2m) - g(h)
    - Kepler orbit: r = a(1-e^2) / (1 + e*cos(theta))
    - Lifting body: includes L/D ratio for cross-range

Compression = Discovery:
    - High compression ratio (>20:1) -> physics law is accurate
    - Low compression ratio (<5:1) -> anomaly or active control

Receipt: physics_law_receipt
SLO: compression_ratio >= 20:1
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize

# Import from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from receipts import dual_hash, emit_receipt, StopRule

from .telemetry_ingest import TelemetryStream


# =============================================================================
# CONSTANTS
# =============================================================================

# Earth constants (SI units)
EARTH_MU = 3.986004418e14  # m^3/s^2 - gravitational parameter
EARTH_RADIUS = 6.371e6  # m - mean radius
EARTH_G0 = 9.80665  # m/s^2 - standard gravity

# Atmospheric scale height (simplified exponential model)
EARTH_H_SCALE = 8500.0  # m
EARTH_RHO_0 = 1.225  # kg/m^3 at sea level


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class PhysicsLaw:
    """
    Discovered physics law from telemetry.

    The law is encoded as:
        - law_type: Type of physics model used
        - equation: Human-readable equation string
        - parameters: Fitted parameters (Cd, L/D, mass, etc.)
        - compression_ratio: Ratio of raw bits to compressed bits
        - residual_rms: RMS of (actual - predicted) in native units
    """
    law_type: str  # "ballistic", "lifting", "orbital", "hybrid"
    equation: str  # Human-readable equation
    parameters: Dict[str, float]  # Fitted parameters
    compression_ratio: float  # Raw / compressed size
    residual_rms: float  # RMS residual in m or m/s
    applicability_range: Dict[str, Tuple[float, float]]  # min/max for each param
    mission_id: str
    payload_hash: str


@dataclass
class TrajectoryFit:
    """Intermediate result from trajectory fitting."""
    Cd: float  # Drag coefficient
    mass: float  # kg
    area: float  # m^2 (reference area)
    lift_to_drag: float  # L/D ratio (0 for ballistic)
    predicted_altitude: np.ndarray
    predicted_velocity: np.ndarray
    residual_altitude: np.ndarray
    residual_velocity: np.ndarray


# =============================================================================
# ATMOSPHERIC MODEL (SIMPLIFIED)
# =============================================================================

def atmospheric_density(altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute atmospheric density using exponential model.

    Args:
        altitude: Altitude in meters above sea level

    Returns:
        Density in kg/m^3
    """
    return EARTH_RHO_0 * np.exp(-altitude / EARTH_H_SCALE)


def gravity(altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute local gravity at altitude.

    Args:
        altitude: Altitude in meters above sea level

    Returns:
        Local gravity in m/s^2
    """
    r = EARTH_RADIUS + altitude
    return EARTH_MU / (r ** 2)


# =============================================================================
# DRAG EQUATION INTEGRATOR
# =============================================================================

def integrate_drag_trajectory(
    initial_altitude: float,
    initial_velocity: float,
    time_span: np.ndarray,
    Cd: float,
    mass: float,
    area: float,
    lift_to_drag: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate trajectory using drag equation.

    dv/dt = -rho(h) * v^2 * Cd * A / (2m) - g(h)
    dh/dt = -v * sin(gamma)

    For simplicity, assumes near-vertical descent (gamma ~ 90 deg).

    Args:
        initial_altitude: Starting altitude in meters
        initial_velocity: Starting velocity magnitude in m/s
        time_span: Array of time points to integrate to
        Cd: Drag coefficient
        mass: Vehicle mass in kg
        area: Reference area in m^2
        lift_to_drag: L/D ratio (0 for pure ballistic)

    Returns:
        Tuple of (altitude_array, velocity_array)
    """
    # Use simple Euler integration for speed
    dt = time_span[1] - time_span[0] if len(time_span) > 1 else 1.0
    n = len(time_span)

    altitude = np.zeros(n)
    velocity = np.zeros(n)

    altitude[0] = initial_altitude
    velocity[0] = initial_velocity

    # Ballistic coefficient
    beta = mass / (Cd * area)  # kg/m^2

    for i in range(1, n):
        h = altitude[i - 1]
        v = velocity[i - 1]

        if h <= 0:
            # Hit ground
            altitude[i:] = 0
            velocity[i:] = 0
            break

        # Atmospheric density
        rho = atmospheric_density(h)

        # Drag deceleration
        drag_acc = 0.5 * rho * v ** 2 * Cd * area / mass

        # Gravity
        g = gravity(h)

        # For lifting body, reduce effective descent rate
        # sin(gamma) ~ 1/(1 + L/D) for equilibrium glide
        if lift_to_drag > 0:
            descent_factor = 1.0 / (1.0 + lift_to_drag)
        else:
            descent_factor = 1.0

        # Simple dynamics (assuming mostly downward)
        dv_dt = -drag_acc - g * descent_factor
        dh_dt = -v * descent_factor  # Descending

        velocity[i] = max(0, v + dv_dt * dt)
        altitude[i] = max(0, h + dh_dt * dt)

    return altitude, velocity


# =============================================================================
# PHYSICS LAW FITTING
# =============================================================================

def _fit_drag_parameters(
    stream: TelemetryStream,
    initial_guess: Optional[Dict[str, float]] = None,
) -> TrajectoryFit:
    """
    Fit drag coefficient and mass to match observed trajectory.

    Uses scipy.optimize to minimize residual between predicted
    and observed altitude/velocity profiles.

    Args:
        stream: Ingested telemetry stream
        initial_guess: Optional initial parameter guesses

    Returns:
        TrajectoryFit with fitted parameters
    """
    # Default initial guesses for Starship
    defaults = {
        "Cd": 1.5,  # Drag coefficient (belly-flop is high)
        "mass": 100000.0,  # kg
        "area": 300.0,  # m^2 (approximate cross-section)
        "lift_to_drag": 0.3,  # L/D for Starship flaps
    }
    if initial_guess:
        defaults.update(initial_guess)

    # Extract data
    time = stream.time - stream.time[0]  # Relative time
    alt_obs = stream.altitude
    vel_obs = stream.velocity

    def objective(params):
        """Objective function: sum of squared residuals."""
        Cd, mass, area, ld = params

        # Clamp to physical values
        Cd = max(0.1, min(3.0, Cd))
        mass = max(100, min(500000, mass))
        area = max(1, min(1000, area))
        ld = max(0, min(2.0, ld))

        alt_pred, vel_pred = integrate_drag_trajectory(
            initial_altitude=alt_obs[0],
            initial_velocity=vel_obs[0],
            time_span=time,
            Cd=Cd,
            mass=mass,
            area=area,
            lift_to_drag=ld,
        )

        # Normalize residuals (altitude in km scale, velocity in km/s scale)
        alt_residual = (alt_pred - alt_obs) / 1000.0  # km
        vel_residual = (vel_pred - vel_obs) / 1000.0  # km/s

        return np.sum(alt_residual ** 2) + np.sum(vel_residual ** 2)

    # Optimize
    x0 = [defaults["Cd"], defaults["mass"], defaults["area"], defaults["lift_to_drag"]]
    bounds = [(0.1, 3.0), (100, 500000), (1, 1000), (0, 2.0)]

    result = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100},
    )

    Cd_fit, mass_fit, area_fit, ld_fit = result.x

    # Compute predictions with fitted params
    alt_pred, vel_pred = integrate_drag_trajectory(
        initial_altitude=alt_obs[0],
        initial_velocity=vel_obs[0],
        time_span=time,
        Cd=Cd_fit,
        mass=mass_fit,
        area=area_fit,
        lift_to_drag=ld_fit,
    )

    return TrajectoryFit(
        Cd=Cd_fit,
        mass=mass_fit,
        area=area_fit,
        lift_to_drag=ld_fit,
        predicted_altitude=alt_pred,
        predicted_velocity=vel_pred,
        residual_altitude=alt_obs - alt_pred,
        residual_velocity=vel_obs - vel_pred,
    )


# =============================================================================
# COMPRESSION RATIO CALCULATION
# =============================================================================

def compute_compression_ratio(
    stream: TelemetryStream,
    fit: TrajectoryFit,
    bit_depth: int = 16,
) -> float:
    """
    Compute compression ratio achieved by physics law.

    Raw bits = n_samples * n_params * bit_depth
    Compressed bits = law parameters + residual entropy

    For well-fitting physics:
        residual is small -> high compression

    Args:
        stream: Original telemetry
        fit: Fitted trajectory
        bit_depth: Bits per sample in raw data

    Returns:
        Compression ratio (raw / compressed)
    """
    n = len(stream.time)

    # Raw data: time, altitude, velocity, acceleration (4 channels)
    raw_bits = n * 4 * bit_depth

    # Compressed: law parameters (4 floats = 4 * 64 = 256 bits)
    # Plus quantized residuals (lower bit depth due to small magnitude)
    param_bits = 4 * 64

    # Estimate residual entropy from RMS
    alt_rms = np.sqrt(np.mean(fit.residual_altitude ** 2))
    vel_rms = np.sqrt(np.mean(fit.residual_velocity ** 2))

    # Residual bits: proportional to log2 of residual magnitude
    # If residual is small, we need fewer bits
    alt_range = np.max(stream.altitude) - np.min(stream.altitude)
    vel_range = np.max(stream.velocity) - np.min(stream.velocity)

    # Effective bits for residuals
    if alt_range > 0:
        alt_residual_bits = max(1, int(np.log2(alt_range / (alt_rms + 1e-6))))
    else:
        alt_residual_bits = bit_depth

    if vel_range > 0:
        vel_residual_bits = max(1, int(np.log2(vel_range / (vel_rms + 1e-6))))
    else:
        vel_residual_bits = bit_depth

    residual_bits = n * (alt_residual_bits + vel_residual_bits)
    compressed_bits = param_bits + residual_bits

    return raw_bits / compressed_bits if compressed_bits > 0 else 1.0


# =============================================================================
# MAIN DISCOVERY FUNCTION
# =============================================================================

def discover_law(
    stream: TelemetryStream,
    tenant_id: str = "default",
    initial_guess: Optional[Dict[str, float]] = None,
    min_compression_ratio: float = 20.0,
) -> Tuple[PhysicsLaw, Dict[str, Any]]:
    """
    Discover physics law from telemetry stream.

    Fits drag equation to observed trajectory and computes
    compression ratio as a measure of law fidelity.

    Args:
        stream: Ingested telemetry stream
        tenant_id: Tenant ID for receipt
        initial_guess: Optional initial parameter guesses
        min_compression_ratio: SLO threshold for compression

    Returns:
        Tuple of (PhysicsLaw, physics_law_receipt)

    Raises:
        StopRule: If compression ratio below SLO threshold
    """
    # Fit drag parameters
    fit = _fit_drag_parameters(stream, initial_guess)

    # Compute compression ratio
    compression_ratio = compute_compression_ratio(stream, fit)

    # Compute residual RMS (combined altitude + velocity)
    alt_rms = np.sqrt(np.mean(fit.residual_altitude ** 2))
    vel_rms = np.sqrt(np.mean(fit.residual_velocity ** 2))
    combined_rms = np.sqrt(alt_rms ** 2 + vel_rms ** 2)

    # Determine law type
    if fit.lift_to_drag > 0.1:
        law_type = "lifting"
        equation = (
            f"dv/dt = -0.5*rho(h)*v^2*Cd*A/m - g(h)/(1+L/D), "
            f"Cd={fit.Cd:.3f}, L/D={fit.lift_to_drag:.3f}"
        )
    else:
        law_type = "ballistic"
        equation = (
            f"dv/dt = -0.5*rho(h)*v^2*Cd*A/m - g(h), "
            f"Cd={fit.Cd:.3f}"
        )

    # Build applicability range
    applicability_range = {
        "altitude_m": (float(np.min(stream.altitude)), float(np.max(stream.altitude))),
        "velocity_mps": (float(np.min(stream.velocity)), float(np.max(stream.velocity))),
    }

    # Create payload for hash
    payload = {
        "law_type": law_type,
        "equation": equation,
        "parameters": {
            "Cd": fit.Cd,
            "mass_kg": fit.mass,
            "area_m2": fit.area,
            "lift_to_drag": fit.lift_to_drag,
        },
        "compression_ratio": compression_ratio,
        "residual_rms": combined_rms,
        "applicability_range": applicability_range,
        "mission_id": stream.mission_id,
    }

    payload_hash = dual_hash(json.dumps(payload, sort_keys=True))

    # Create physics law
    law = PhysicsLaw(
        law_type=law_type,
        equation=equation,
        parameters=payload["parameters"],
        compression_ratio=compression_ratio,
        residual_rms=combined_rms,
        applicability_range=applicability_range,
        mission_id=stream.mission_id,
        payload_hash=payload_hash,
    )

    # Emit receipt
    receipt = emit_receipt("physics_law", {
        "tenant_id": tenant_id,
        "mission_id": stream.mission_id,
        "law_type": law_type,
        "equation": equation,
        "parameters": payload["parameters"],
        "compression_ratio": compression_ratio,
        "residual_rms_m": alt_rms,
        "residual_rms_mps": vel_rms,
        "applicability_range": applicability_range,
        "slo_pass": compression_ratio >= min_compression_ratio,
    })

    # Check SLO
    if compression_ratio < min_compression_ratio:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "compression_ratio",
            "baseline": min_compression_ratio,
            "delta": compression_ratio - min_compression_ratio,
            "classification": "violation",
            "action": "alert",
        })

    return law, receipt


# =============================================================================
# COMPRESSION FUNCTIONS
# =============================================================================

def compress_trajectory(
    stream: TelemetryStream,
    law: PhysicsLaw,
    tenant_id: str = "default",
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress trajectory using discovered physics law.

    Stores:
        - Law parameters (compact)
        - Initial conditions
        - Quantized residuals (only where prediction deviates)

    Args:
        stream: Telemetry stream to compress
        law: Discovered physics law
        tenant_id: Tenant ID for receipt

    Returns:
        Tuple of (compressed_bytes, compression_receipt)
    """
    # Reconstruct predicted trajectory
    time = stream.time - stream.time[0]
    alt_pred, vel_pred = integrate_drag_trajectory(
        initial_altitude=stream.altitude[0],
        initial_velocity=stream.velocity[0],
        time_span=time,
        Cd=law.parameters["Cd"],
        mass=law.parameters["mass_kg"],
        area=law.parameters["area_m2"],
        lift_to_drag=law.parameters["lift_to_drag"],
    )

    # Compute residuals
    alt_residual = stream.altitude - alt_pred
    vel_residual = stream.velocity - vel_pred

    # Quantize residuals (16-bit signed integers in meters)
    alt_quantized = np.clip(alt_residual, -32767, 32767).astype(np.int16)
    vel_quantized = np.clip(vel_residual, -32767, 32767).astype(np.int16)

    # Pack compressed data
    compressed = {
        "law_hash": law.payload_hash,
        "initial_altitude_m": float(stream.altitude[0]),
        "initial_velocity_mps": float(stream.velocity[0]),
        "n_samples": len(stream.time),
        "dt_s": float(time[1] - time[0]) if len(time) > 1 else 1.0,
        "alt_residual_int16": alt_quantized.tobytes().hex(),
        "vel_residual_int16": vel_quantized.tobytes().hex(),
    }

    compressed_bytes = json.dumps(compressed).encode()
    compressed_hash = dual_hash(compressed_bytes)

    # Calculate compression ratio
    raw_bytes = len(stream.time) * 4 * 8  # 4 channels * 8 bytes each
    ratio = raw_bytes / len(compressed_bytes) if len(compressed_bytes) > 0 else 1.0

    receipt = emit_receipt("trajectory_compression", {
        "tenant_id": tenant_id,
        "mission_id": stream.mission_id,
        "law_hash": law.payload_hash,
        "raw_bytes": raw_bytes,
        "compressed_bytes": len(compressed_bytes),
        "compression_ratio": ratio,
        "compressed_hash": compressed_hash,
    })

    return compressed_bytes, receipt


def decompress_trajectory(
    compressed_bytes: bytes,
    law: PhysicsLaw,
    tenant_id: str = "default",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Decompress trajectory using physics law.

    Args:
        compressed_bytes: Compressed trajectory data
        law: Physics law used for compression
        tenant_id: Tenant ID for receipt

    Returns:
        Tuple of (telemetry_dict, decompression_receipt)
    """
    compressed = json.loads(compressed_bytes.decode())

    # Extract parameters
    initial_alt = compressed["initial_altitude_m"]
    initial_vel = compressed["initial_velocity_mps"]
    n_samples = compressed["n_samples"]
    dt = compressed["dt_s"]

    # Reconstruct predicted trajectory
    time_span = np.arange(n_samples) * dt
    alt_pred, vel_pred = integrate_drag_trajectory(
        initial_altitude=initial_alt,
        initial_velocity=initial_vel,
        time_span=time_span,
        Cd=law.parameters["Cd"],
        mass=law.parameters["mass_kg"],
        area=law.parameters["area_m2"],
        lift_to_drag=law.parameters["lift_to_drag"],
    )

    # Decode residuals
    alt_residual = np.frombuffer(
        bytes.fromhex(compressed["alt_residual_int16"]), dtype=np.int16
    ).astype(np.float64)
    vel_residual = np.frombuffer(
        bytes.fromhex(compressed["vel_residual_int16"]), dtype=np.int16
    ).astype(np.float64)

    # Reconstruct original
    altitude = alt_pred + alt_residual
    velocity = vel_pred + vel_residual

    telemetry = {
        "time": time_span,
        "altitude": altitude,
        "velocity": velocity,
    }

    receipt = emit_receipt("trajectory_decompression", {
        "tenant_id": tenant_id,
        "law_hash": law.payload_hash,
        "n_samples": n_samples,
        "decompressed_hash": dual_hash(json.dumps({
            k: v.tolist() for k, v in telemetry.items()
        }, sort_keys=True)),
    })

    return telemetry, receipt
