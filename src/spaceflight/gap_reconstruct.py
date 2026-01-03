"""
src/spaceflight/gap_reconstruct.py - Gap Reconstruction Module

Reconstruct missing telemetry during simulated or real blackout gaps
using discovered physics laws.

CLAUDEME v3.1 Compliant: All functions emit receipts.

Reconstruction Methods:
    - Forward: Integrate from pre-gap state
    - Backward: Integrate backward from post-gap state
    - Bidirectional: Both directions, meet in middle, compare

Confidence Scoring:
    - Short gaps (<30s): high confidence (>0.95)
    - Medium gaps (30-90s): medium confidence (0.80-0.95)
    - Long gaps (>90s): low confidence (<0.80)

Receipt: reconstruction_receipt
SLO: error < 5% for gaps <= 60s
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from receipts import dual_hash, emit_receipt, merkle, StopRule

from .physics_discovery import (
    PhysicsLaw,
    integrate_drag_trajectory,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Confidence thresholds based on gap duration
CONFIDENCE_THRESHOLDS = {
    30.0: 0.95,   # < 30s: high confidence
    90.0: 0.80,   # 30-90s: medium confidence
    float("inf"): 0.60,  # > 90s: low confidence
}

# Error thresholds for SLO (relative error)
ERROR_SLO = {
    60.0: 0.05,   # <= 60s: < 5% error
    120.0: 0.10,  # <= 120s: < 10% error
    float("inf"): 0.20,  # > 120s: < 20% error
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class StateVector:
    """State at a specific time point."""
    time: float  # seconds
    altitude: float  # meters
    velocity: float  # m/s
    acceleration: float  # m/s^2


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of gap reconstruction."""
    gap_start_time: float
    gap_end_time: float
    gap_duration: float

    # Reconstructed data
    time: np.ndarray
    altitude: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    # Quality metrics
    method: str  # "forward", "backward", "bidirectional"
    confidence_score: float  # 0.0 - 1.0
    estimated_error: float  # relative error estimate

    # Validation (if post-gap data available)
    validation_available: bool
    actual_error: Optional[float]

    # Provenance
    pre_gap_state_hash: str
    post_gap_state_hash: str
    physics_law_hash: str
    reconstruction_hash: str


# =============================================================================
# BACKWARD INTEGRATION
# =============================================================================

def integrate_backward(
    final_altitude: float,
    final_velocity: float,
    time_span: np.ndarray,
    Cd: float,
    mass: float,
    area: float,
    lift_to_drag: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate trajectory backward in time.

    Uses time-reversal of drag equation.

    Args:
        final_altitude: Ending altitude in meters
        final_velocity: Ending velocity magnitude in m/s
        time_span: Array of time points (reversed internally)
        Cd: Drag coefficient
        mass: Vehicle mass in kg
        area: Reference area in m^2
        lift_to_drag: L/D ratio

    Returns:
        Tuple of (altitude_array, velocity_array) in forward time order
    """
    # Reverse time for backward integration
    dt = time_span[1] - time_span[0] if len(time_span) > 1 else 1.0
    n = len(time_span)

    # Start from end
    altitude = np.zeros(n)
    velocity = np.zeros(n)
    altitude[-1] = final_altitude
    velocity[-1] = final_velocity

    from .physics_discovery import atmospheric_density, gravity

    for i in range(n - 2, -1, -1):
        h = altitude[i + 1]
        v = velocity[i + 1]

        if h <= 0:
            altitude[:i + 1] = 0
            velocity[:i + 1] = 0
            break

        # Atmospheric density
        rho = atmospheric_density(h)

        # Drag deceleration (reversed: adds velocity going backward)
        drag_acc = 0.5 * rho * v ** 2 * Cd * area / mass

        # Gravity
        g = gravity(h)

        # Descent factor for lifting bodies
        if lift_to_drag > 0:
            descent_factor = 1.0 / (1.0 + lift_to_drag)
        else:
            descent_factor = 1.0

        # Reverse dynamics
        dv_dt = drag_acc + g * descent_factor  # Add back energy
        dh_dt = v * descent_factor  # Ascending backward

        velocity[i] = v - dv_dt * dt  # Subtract the change
        altitude[i] = h - dh_dt * dt

    return altitude, velocity


# =============================================================================
# RECONSTRUCTION METHODS
# =============================================================================

def reconstruct_forward(
    pre_gap_state: StateVector,
    gap_duration: float,
    law: PhysicsLaw,
    sample_rate_hz: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct gap by forward integration from pre-gap state.

    Args:
        pre_gap_state: State vector just before gap
        gap_duration: Duration of gap in seconds
        law: Physics law for integration
        sample_rate_hz: Output sample rate

    Returns:
        Tuple of (time, altitude, velocity, acceleration)
    """
    n_samples = max(2, int(gap_duration * sample_rate_hz))
    time_span = np.linspace(0, gap_duration, n_samples)

    alt, vel = integrate_drag_trajectory(
        initial_altitude=pre_gap_state.altitude,
        initial_velocity=pre_gap_state.velocity,
        time_span=time_span,
        Cd=law.parameters["Cd"],
        mass=law.parameters["mass_kg"],
        area=law.parameters["area_m2"],
        lift_to_drag=law.parameters["lift_to_drag"],
    )

    # Compute acceleration from velocity derivative
    if len(vel) > 1:
        acc = np.gradient(vel, time_span)
    else:
        acc = np.zeros_like(vel)

    return time_span + pre_gap_state.time, alt, vel, acc


def reconstruct_backward(
    post_gap_state: StateVector,
    gap_duration: float,
    law: PhysicsLaw,
    sample_rate_hz: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct gap by backward integration from post-gap state.

    Args:
        post_gap_state: State vector just after gap
        gap_duration: Duration of gap in seconds
        law: Physics law for integration
        sample_rate_hz: Output sample rate

    Returns:
        Tuple of (time, altitude, velocity, acceleration)
    """
    n_samples = max(2, int(gap_duration * sample_rate_hz))
    time_span = np.linspace(0, gap_duration, n_samples)

    alt, vel = integrate_backward(
        final_altitude=post_gap_state.altitude,
        final_velocity=post_gap_state.velocity,
        time_span=time_span,
        Cd=law.parameters["Cd"],
        mass=law.parameters["mass_kg"],
        area=law.parameters["area_m2"],
        lift_to_drag=law.parameters["lift_to_drag"],
    )

    # Compute acceleration
    if len(vel) > 1:
        acc = np.gradient(vel, time_span)
    else:
        acc = np.zeros_like(vel)

    # Adjust time to gap start
    gap_start_time = post_gap_state.time - gap_duration
    return time_span + gap_start_time, alt, vel, acc


def reconstruct_bidirectional(
    pre_gap_state: StateVector,
    post_gap_state: StateVector,
    law: PhysicsLaw,
    sample_rate_hz: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Reconstruct gap using bidirectional integration.

    Integrates forward from pre-gap and backward from post-gap,
    then blends results with linear weight transition at midpoint.

    Args:
        pre_gap_state: State vector just before gap
        post_gap_state: State vector just after gap
        law: Physics law for integration
        sample_rate_hz: Output sample rate

    Returns:
        Tuple of (time, altitude, velocity, acceleration, convergence_error)
    """
    gap_duration = post_gap_state.time - pre_gap_state.time

    # Forward integration
    t_fwd, alt_fwd, vel_fwd, acc_fwd = reconstruct_forward(
        pre_gap_state, gap_duration, law, sample_rate_hz
    )

    # Backward integration
    t_bwd, alt_bwd, vel_bwd, acc_bwd = reconstruct_backward(
        post_gap_state, gap_duration, law, sample_rate_hz
    )

    # Compute convergence error at midpoint
    mid_idx = len(t_fwd) // 2
    alt_error = abs(alt_fwd[mid_idx] - alt_bwd[mid_idx])
    vel_error = abs(vel_fwd[mid_idx] - vel_bwd[mid_idx])

    # Normalize errors
    alt_range = max(abs(pre_gap_state.altitude), abs(post_gap_state.altitude), 1)
    vel_range = max(abs(pre_gap_state.velocity), abs(post_gap_state.velocity), 1)
    convergence_error = 0.5 * (alt_error / alt_range + vel_error / vel_range)

    # Blend using linear weights
    n = len(t_fwd)
    weights_fwd = np.linspace(1.0, 0.0, n)
    weights_bwd = 1.0 - weights_fwd

    altitude = weights_fwd * alt_fwd + weights_bwd * alt_bwd
    velocity = weights_fwd * vel_fwd + weights_bwd * vel_bwd
    acceleration = weights_fwd * acc_fwd + weights_bwd * acc_bwd

    return t_fwd, altitude, velocity, acceleration, convergence_error


# =============================================================================
# CONFIDENCE CALCULATION
# =============================================================================

def compute_confidence(
    gap_duration: float,
    convergence_error: Optional[float] = None,
) -> float:
    """
    Compute confidence score based on gap duration and convergence.

    Args:
        gap_duration: Duration of gap in seconds
        convergence_error: Optional bidirectional convergence error

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Base confidence from duration
    for threshold, base_confidence in sorted(CONFIDENCE_THRESHOLDS.items()):
        if gap_duration <= threshold:
            base = base_confidence
            break
    else:
        base = 0.60

    # Reduce confidence if convergence error is high
    if convergence_error is not None:
        # High convergence error reduces confidence
        error_penalty = min(0.3, convergence_error)
        return max(0.0, base - error_penalty)

    return base


def get_error_slo(gap_duration: float) -> float:
    """Get error SLO threshold for given gap duration."""
    for threshold, slo in sorted(ERROR_SLO.items()):
        if gap_duration <= threshold:
            return slo
    return 0.20


# =============================================================================
# MAIN RECONSTRUCTION FUNCTION
# =============================================================================

def reconstruct_gap(
    pre_gap_state: StateVector,
    post_gap_state: Optional[StateVector],
    law: PhysicsLaw,
    tenant_id: str = "default",
    sample_rate_hz: float = 1.0,
    method: str = "auto",
) -> Tuple[ReconstructionResult, Dict[str, Any]]:
    """
    Reconstruct missing telemetry during a gap.

    Args:
        pre_gap_state: State vector just before gap
        post_gap_state: State vector just after gap (None for forward-only)
        law: Physics law for reconstruction
        tenant_id: Tenant ID for receipt
        sample_rate_hz: Output sample rate
        method: Reconstruction method ("forward", "backward", "bidirectional", "auto")

    Returns:
        Tuple of (ReconstructionResult, reconstruction_receipt)

    Raises:
        StopRule: If reconstruction fails
    """
    # Compute gap duration
    if post_gap_state is not None:
        gap_duration = post_gap_state.time - pre_gap_state.time
    else:
        gap_duration = 60.0  # Default 60s for forward-only

    if gap_duration <= 0:
        raise StopRule(f"Invalid gap duration: {gap_duration}")

    # Hash pre-gap state
    pre_gap_data = {
        "time": pre_gap_state.time,
        "altitude": pre_gap_state.altitude,
        "velocity": pre_gap_state.velocity,
        "acceleration": pre_gap_state.acceleration,
    }
    pre_gap_hash = dual_hash(json.dumps(pre_gap_data, sort_keys=True))

    # Hash post-gap state
    if post_gap_state is not None:
        post_gap_data = {
            "time": post_gap_state.time,
            "altitude": post_gap_state.altitude,
            "velocity": post_gap_state.velocity,
            "acceleration": post_gap_state.acceleration,
        }
        post_gap_hash = dual_hash(json.dumps(post_gap_data, sort_keys=True))
    else:
        post_gap_hash = dual_hash(b"no_post_gap")

    # Select method
    if method == "auto":
        if post_gap_state is not None:
            method = "bidirectional"
        else:
            method = "forward"

    # Perform reconstruction
    convergence_error = None

    if method == "forward":
        t, alt, vel, acc = reconstruct_forward(
            pre_gap_state, gap_duration, law, sample_rate_hz
        )
    elif method == "backward":
        if post_gap_state is None:
            raise StopRule("Backward reconstruction requires post-gap state")
        t, alt, vel, acc = reconstruct_backward(
            post_gap_state, gap_duration, law, sample_rate_hz
        )
    elif method == "bidirectional":
        if post_gap_state is None:
            raise StopRule("Bidirectional reconstruction requires post-gap state")
        t, alt, vel, acc, convergence_error = reconstruct_bidirectional(
            pre_gap_state, post_gap_state, law, sample_rate_hz
        )
    else:
        raise StopRule(f"Unknown reconstruction method: {method}")

    # Compute confidence
    confidence = compute_confidence(gap_duration, convergence_error)

    # Estimate error (use convergence error or duration-based estimate)
    if convergence_error is not None:
        estimated_error = convergence_error
    else:
        # Estimate based on duration and law residual
        estimated_error = min(0.5, law.residual_rms / 1000.0 * gap_duration / 60.0)

    # Validate against post-gap if available
    validation_available = post_gap_state is not None and method in ["forward", "backward"]
    actual_error = None

    if validation_available and method == "forward":
        # Check predicted end state vs actual post-gap
        pred_alt = alt[-1]
        pred_vel = vel[-1]
        alt_err = abs(pred_alt - post_gap_state.altitude) / max(abs(post_gap_state.altitude), 1)
        vel_err = abs(pred_vel - post_gap_state.velocity) / max(abs(post_gap_state.velocity), 1)
        actual_error = 0.5 * (alt_err + vel_err)

    # Hash reconstruction
    recon_data = {
        "time": t.tolist(),
        "altitude": alt.tolist(),
        "velocity": vel.tolist(),
        "acceleration": acc.tolist(),
    }
    reconstruction_hash = dual_hash(json.dumps(recon_data, sort_keys=True))

    # Create result
    result = ReconstructionResult(
        gap_start_time=pre_gap_state.time,
        gap_end_time=pre_gap_state.time + gap_duration,
        gap_duration=gap_duration,
        time=t,
        altitude=alt,
        velocity=vel,
        acceleration=acc,
        method=method,
        confidence_score=confidence,
        estimated_error=estimated_error,
        validation_available=validation_available,
        actual_error=actual_error,
        pre_gap_state_hash=pre_gap_hash,
        post_gap_state_hash=post_gap_hash,
        physics_law_hash=law.payload_hash,
        reconstruction_hash=reconstruction_hash,
    )

    # Check SLO
    error_slo = get_error_slo(gap_duration)
    error_to_check = actual_error if actual_error is not None else estimated_error
    slo_pass = error_to_check < error_slo

    # Emit receipt
    receipt = emit_receipt("gap_reconstruction", {
        "tenant_id": tenant_id,
        "mission_id": law.mission_id,
        "gap_start": pre_gap_state.time,
        "gap_end": pre_gap_state.time + gap_duration,
        "gap_duration_seconds": gap_duration,
        "pre_gap_state_hash": pre_gap_hash,
        "post_gap_state_hash": post_gap_hash,
        "physics_law_hash": law.payload_hash,
        "reconstruction_hash": reconstruction_hash,
        "method": method,
        "confidence_score": confidence,
        "estimated_error": estimated_error,
        "actual_error": actual_error,
        "validation_available": validation_available,
        "slo_threshold": error_slo,
        "slo_pass": slo_pass,
    })

    if not slo_pass:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "reconstruction_error",
            "baseline": error_slo,
            "delta": error_to_check - error_slo,
            "classification": "violation",
            "action": "alert",
        })

    return result, receipt


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_reconstruction(
    result: ReconstructionResult,
    ground_truth_altitude: np.ndarray,
    ground_truth_velocity: np.ndarray,
    tenant_id: str = "default",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Validate reconstruction against ground truth.

    Args:
        result: Reconstruction result
        ground_truth_altitude: Actual altitude values
        ground_truth_velocity: Actual velocity values
        tenant_id: Tenant ID for receipt

    Returns:
        Tuple of (metrics_dict, validation_receipt)
    """
    # Ensure arrays match
    n = min(len(result.altitude), len(ground_truth_altitude))

    alt_pred = result.altitude[:n]
    alt_true = ground_truth_altitude[:n]
    vel_pred = result.velocity[:n]
    vel_true = ground_truth_velocity[:n]

    # Compute metrics
    alt_rmse = np.sqrt(np.mean((alt_pred - alt_true) ** 2))
    vel_rmse = np.sqrt(np.mean((vel_pred - vel_true) ** 2))

    alt_mae = np.mean(np.abs(alt_pred - alt_true))
    vel_mae = np.mean(np.abs(vel_pred - vel_true))

    alt_max_error = np.max(np.abs(alt_pred - alt_true))
    vel_max_error = np.max(np.abs(vel_pred - vel_true))

    # Relative errors
    alt_mean = np.mean(np.abs(alt_true)) + 1e-6
    vel_mean = np.mean(np.abs(vel_true)) + 1e-6

    alt_rel_error = alt_rmse / alt_mean
    vel_rel_error = vel_rmse / vel_mean

    metrics = {
        "altitude_rmse_m": float(alt_rmse),
        "velocity_rmse_mps": float(vel_rmse),
        "altitude_mae_m": float(alt_mae),
        "velocity_mae_mps": float(vel_mae),
        "altitude_max_error_m": float(alt_max_error),
        "velocity_max_error_mps": float(vel_max_error),
        "altitude_relative_error": float(alt_rel_error),
        "velocity_relative_error": float(vel_rel_error),
    }

    # Check SLOs
    slo_pass = (
        alt_rmse < 500.0 or alt_rel_error < 0.01
    ) and (
        vel_rmse < 50.0 or vel_rel_error < 0.05
    )

    # Detect anomalies
    anomalies = []
    if alt_rmse >= 500.0 and alt_rel_error >= 0.01:
        anomalies.append("altitude_error_high")
    if vel_rmse >= 50.0 and vel_rel_error >= 0.05:
        anomalies.append("velocity_error_high")

    # Hash ground truth
    ground_truth_hash = dual_hash(json.dumps({
        "altitude": ground_truth_altitude.tolist(),
        "velocity": ground_truth_velocity.tolist(),
    }, sort_keys=True))

    receipt = emit_receipt("reconstruction_validation", {
        "tenant_id": tenant_id,
        "reconstruction_id": result.reconstruction_hash,
        "ground_truth_hash": ground_truth_hash,
        "rmse_altitude": alt_rmse,
        "rmse_velocity": vel_rmse,
        "mae_altitude": alt_mae,
        "mae_velocity": vel_mae,
        "max_error_altitude": alt_max_error,
        "max_error_velocity": vel_max_error,
        "relative_error_altitude": alt_rel_error,
        "relative_error_velocity": vel_rel_error,
        "slo_pass": slo_pass,
        "anomalies_detected": anomalies,
    })

    return metrics, receipt
