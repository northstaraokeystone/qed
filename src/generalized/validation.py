"""
src/generalized/validation.py - Reconstruction Validation Module

Validate reconstruction fidelity against ground truth.
CLAUDEME v3.1 Compliant: All functions emit receipts.

Metrics:
    - RMSE: Root Mean Square Error
    - MAE: Mean Absolute Error
    - Max Error: Maximum absolute error
    - Relative Error: RMSE / mean(actual)

SLO Thresholds:
    - Altitude: <1% or <500m
    - Velocity: <5% or <50 m/s
    - Acceleration: <10%

Receipt: validation_receipt
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

from receipts import dual_hash, emit_receipt, StopRule


# =============================================================================
# SLO THRESHOLDS
# =============================================================================

# Default SLO thresholds
SLO_ALTITUDE_ABS_M = 500.0  # meters
SLO_ALTITUDE_REL = 0.01  # 1%
SLO_VELOCITY_ABS_MPS = 50.0  # m/s
SLO_VELOCITY_REL = 0.05  # 5%
SLO_ACCELERATION_REL = 0.10  # 10%


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ValidationResult:
    """Result of validation against ground truth."""
    # RMSE metrics
    rmse_altitude_m: float
    rmse_velocity_mps: float
    rmse_acceleration_mps2: Optional[float]

    # MAE metrics
    mae_altitude_m: float
    mae_velocity_mps: float
    mae_acceleration_mps2: Optional[float]

    # Max error
    max_error_altitude_m: float
    max_error_velocity_mps: float
    max_error_acceleration_mps2: Optional[float]

    # Relative errors
    relative_error_altitude: float
    relative_error_velocity: float
    relative_error_acceleration: Optional[float]

    # SLO pass/fail
    slo_altitude_pass: bool
    slo_velocity_pass: bool
    slo_acceleration_pass: bool
    slo_overall_pass: bool

    # Anomalies
    anomalies_detected: Tuple[str, ...]

    # Provenance
    ground_truth_hash: str
    reconstruction_hash: str


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_rmse(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    Compute Root Mean Square Error.

    Args:
        predicted: Predicted values
        actual: Actual (ground truth) values

    Returns:
        RMSE value
    """
    n = min(len(predicted), len(actual))
    if n == 0:
        return 0.0

    diff = predicted[:n] - actual[:n]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_mae(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Compute Mean Absolute Error."""
    n = min(len(predicted), len(actual))
    if n == 0:
        return 0.0

    diff = np.abs(predicted[:n] - actual[:n])
    return float(np.mean(diff))


def compute_max_error(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Compute Maximum Absolute Error."""
    n = min(len(predicted), len(actual))
    if n == 0:
        return 0.0

    diff = np.abs(predicted[:n] - actual[:n])
    return float(np.max(diff))


def compute_relative_error(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    Compute Relative Error (RMSE / mean of actual).

    Returns 0 if actual mean is 0.
    """
    n = min(len(predicted), len(actual))
    if n == 0:
        return 0.0

    rmse = compute_rmse(predicted, actual)
    mean_actual = np.mean(np.abs(actual[:n]))

    if mean_actual == 0:
        return 0.0

    return rmse / mean_actual


def compute_metrics(
    predicted: Dict[str, np.ndarray],
    actual: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute all validation metrics for trajectory.

    Args:
        predicted: Dict with altitude, velocity, acceleration arrays
        actual: Dict with altitude, velocity, acceleration arrays

    Returns:
        Dict with all computed metrics
    """
    metrics = {}

    # Altitude metrics
    if "altitude" in predicted and "altitude" in actual:
        pred_alt = predicted["altitude"]
        act_alt = actual["altitude"]
        metrics["rmse_altitude_m"] = compute_rmse(pred_alt, act_alt)
        metrics["mae_altitude_m"] = compute_mae(pred_alt, act_alt)
        metrics["max_error_altitude_m"] = compute_max_error(pred_alt, act_alt)
        metrics["relative_error_altitude"] = compute_relative_error(pred_alt, act_alt)

    # Velocity metrics
    if "velocity" in predicted and "velocity" in actual:
        pred_vel = predicted["velocity"]
        act_vel = actual["velocity"]
        metrics["rmse_velocity_mps"] = compute_rmse(pred_vel, act_vel)
        metrics["mae_velocity_mps"] = compute_mae(pred_vel, act_vel)
        metrics["max_error_velocity_mps"] = compute_max_error(pred_vel, act_vel)
        metrics["relative_error_velocity"] = compute_relative_error(pred_vel, act_vel)

    # Acceleration metrics (if available)
    if "acceleration" in predicted and "acceleration" in actual:
        pred_acc = predicted["acceleration"]
        act_acc = actual["acceleration"]
        metrics["rmse_acceleration_mps2"] = compute_rmse(pred_acc, act_acc)
        metrics["mae_acceleration_mps2"] = compute_mae(pred_acc, act_acc)
        metrics["max_error_acceleration_mps2"] = compute_max_error(pred_acc, act_acc)
        metrics["relative_error_acceleration"] = compute_relative_error(pred_acc, act_acc)

    return metrics


# =============================================================================
# SLO VALIDATION
# =============================================================================

def check_slo_altitude(
    rmse: float,
    relative_error: float,
    slo_abs: float = SLO_ALTITUDE_ABS_M,
    slo_rel: float = SLO_ALTITUDE_REL,
) -> bool:
    """
    Check altitude SLO.

    Passes if EITHER absolute OR relative threshold is met.
    """
    return rmse < slo_abs or relative_error < slo_rel


def check_slo_velocity(
    rmse: float,
    relative_error: float,
    slo_abs: float = SLO_VELOCITY_ABS_MPS,
    slo_rel: float = SLO_VELOCITY_REL,
) -> bool:
    """Check velocity SLO."""
    return rmse < slo_abs or relative_error < slo_rel


def check_slo_acceleration(
    relative_error: float,
    slo_rel: float = SLO_ACCELERATION_REL,
) -> bool:
    """Check acceleration SLO (relative only)."""
    return relative_error < slo_rel


def validate_against_slo(
    predicted: Dict[str, np.ndarray],
    actual: Dict[str, np.ndarray],
    tenant_id: str = "default",
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[ValidationResult, Dict[str, Any]]:
    """
    Validate reconstruction against ground truth and SLO thresholds.

    Args:
        predicted: Reconstructed/predicted trajectory
        actual: Ground truth trajectory
        tenant_id: Tenant ID for receipt
        custom_thresholds: Optional custom SLO thresholds

    Returns:
        Tuple of (ValidationResult, validation_receipt)
    """
    # Compute metrics
    metrics = compute_metrics(predicted, actual)

    # Get thresholds
    if custom_thresholds is None:
        custom_thresholds = {}

    slo_alt_abs = custom_thresholds.get("altitude_abs_m", SLO_ALTITUDE_ABS_M)
    slo_alt_rel = custom_thresholds.get("altitude_rel", SLO_ALTITUDE_REL)
    slo_vel_abs = custom_thresholds.get("velocity_abs_mps", SLO_VELOCITY_ABS_MPS)
    slo_vel_rel = custom_thresholds.get("velocity_rel", SLO_VELOCITY_REL)
    slo_acc_rel = custom_thresholds.get("acceleration_rel", SLO_ACCELERATION_REL)

    # Check altitude SLO
    slo_alt_pass = True
    if "rmse_altitude_m" in metrics:
        slo_alt_pass = check_slo_altitude(
            metrics["rmse_altitude_m"],
            metrics.get("relative_error_altitude", 0),
            slo_alt_abs,
            slo_alt_rel,
        )

    # Check velocity SLO
    slo_vel_pass = True
    if "rmse_velocity_mps" in metrics:
        slo_vel_pass = check_slo_velocity(
            metrics["rmse_velocity_mps"],
            metrics.get("relative_error_velocity", 0),
            slo_vel_abs,
            slo_vel_rel,
        )

    # Check acceleration SLO
    slo_acc_pass = True
    if "relative_error_acceleration" in metrics:
        slo_acc_pass = check_slo_acceleration(
            metrics["relative_error_acceleration"],
            slo_acc_rel,
        )

    # Overall SLO
    slo_overall_pass = slo_alt_pass and slo_vel_pass and slo_acc_pass

    # Detect anomalies
    anomalies = []
    if not slo_alt_pass:
        anomalies.append("altitude_error_exceeds_slo")
    if not slo_vel_pass:
        anomalies.append("velocity_error_exceeds_slo")
    if not slo_acc_pass:
        anomalies.append("acceleration_error_exceeds_slo")

    # Additional anomaly checks
    if metrics.get("rmse_altitude_m", 0) > 1000:
        anomalies.append("altitude_error_very_high")
    if metrics.get("rmse_velocity_mps", 0) > 100:
        anomalies.append("velocity_error_very_high")

    # Hash data
    ground_truth_hash = dual_hash(json.dumps({
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in actual.items()
    }, sort_keys=True))

    reconstruction_hash = dual_hash(json.dumps({
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in predicted.items()
    }, sort_keys=True))

    # Build result (convert numpy bools to Python bools)
    result = ValidationResult(
        rmse_altitude_m=float(metrics.get("rmse_altitude_m", 0.0)),
        rmse_velocity_mps=float(metrics.get("rmse_velocity_mps", 0.0)),
        rmse_acceleration_mps2=float(metrics["rmse_acceleration_mps2"]) if "rmse_acceleration_mps2" in metrics else None,
        mae_altitude_m=float(metrics.get("mae_altitude_m", 0.0)),
        mae_velocity_mps=float(metrics.get("mae_velocity_mps", 0.0)),
        mae_acceleration_mps2=float(metrics["mae_acceleration_mps2"]) if "mae_acceleration_mps2" in metrics else None,
        max_error_altitude_m=float(metrics.get("max_error_altitude_m", 0.0)),
        max_error_velocity_mps=float(metrics.get("max_error_velocity_mps", 0.0)),
        max_error_acceleration_mps2=float(metrics["max_error_acceleration_mps2"]) if "max_error_acceleration_mps2" in metrics else None,
        relative_error_altitude=float(metrics.get("relative_error_altitude", 0.0)),
        relative_error_velocity=float(metrics.get("relative_error_velocity", 0.0)),
        relative_error_acceleration=float(metrics["relative_error_acceleration"]) if "relative_error_acceleration" in metrics else None,
        slo_altitude_pass=bool(slo_alt_pass),
        slo_velocity_pass=bool(slo_vel_pass),
        slo_acceleration_pass=bool(slo_acc_pass),
        slo_overall_pass=bool(slo_overall_pass),
        anomalies_detected=tuple(anomalies),
        ground_truth_hash=ground_truth_hash,
        reconstruction_hash=reconstruction_hash,
    )

    # Emit receipt (convert numpy bools to Python bools for JSON serialization)
    receipt = emit_receipt("reconstruction_validation", {
        "tenant_id": tenant_id,
        "reconstruction_hash": reconstruction_hash,
        "ground_truth_hash": ground_truth_hash,
        "rmse_altitude": metrics.get("rmse_altitude_m", 0.0),
        "rmse_velocity": metrics.get("rmse_velocity_mps", 0.0),
        "rmse_acceleration": metrics.get("rmse_acceleration_mps2"),
        "max_error_altitude": metrics.get("max_error_altitude_m", 0.0),
        "max_error_velocity": metrics.get("max_error_velocity_mps", 0.0),
        "relative_error_altitude": metrics.get("relative_error_altitude", 0.0),
        "relative_error_velocity": metrics.get("relative_error_velocity", 0.0),
        "slo_altitude_pass": bool(slo_alt_pass),
        "slo_velocity_pass": bool(slo_vel_pass),
        "slo_acceleration_pass": bool(slo_acc_pass),
        "slo_pass": bool(slo_overall_pass),
        "anomalies_detected": anomalies,
    })

    # Emit anomaly receipts if SLO fails
    if not slo_overall_pass:
        for anomaly in anomalies:
            emit_receipt("anomaly", {
                "tenant_id": tenant_id,
                "metric": anomaly,
                "baseline": 0.0,
                "delta": 1.0,
                "classification": "violation",
                "action": "alert",
            })

    return result, receipt


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validate(
    predicted_altitude: np.ndarray,
    predicted_velocity: np.ndarray,
    actual_altitude: np.ndarray,
    actual_velocity: np.ndarray,
) -> Dict[str, float]:
    """
    Quick validation without receipts.

    Returns dict with basic metrics.
    """
    return {
        "rmse_altitude_m": compute_rmse(predicted_altitude, actual_altitude),
        "rmse_velocity_mps": compute_rmse(predicted_velocity, actual_velocity),
        "mae_altitude_m": compute_mae(predicted_altitude, actual_altitude),
        "mae_velocity_mps": compute_mae(predicted_velocity, actual_velocity),
        "max_error_altitude_m": compute_max_error(predicted_altitude, actual_altitude),
        "max_error_velocity_mps": compute_max_error(predicted_velocity, actual_velocity),
    }


def validate_compression_roundtrip(
    original: Dict[str, np.ndarray],
    decompressed: Dict[str, np.ndarray],
    tenant_id: str = "default",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that compression/decompression roundtrip is lossless.

    Returns:
        Tuple of (is_lossless, validation_receipt)
    """
    result, receipt = validate_against_slo(decompressed, original, tenant_id)

    # For roundtrip, we want very low error
    is_lossless = (
        result.rmse_altitude_m < 1.0 and  # <1m error
        result.rmse_velocity_mps < 0.1  # <0.1 m/s error
    )

    return is_lossless, receipt
