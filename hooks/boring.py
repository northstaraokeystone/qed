"""
Boring Company TBM hook for QED v5.0.

Normalizes TBM signals into QED space by scaling each channel so the safety threshold
maps to 10.0 internal units (QED's internal recall threshold). Exposes a small CLI:
  - 'demo' for synthetic TBM signals
  - 'from-csv' for real data windowing
"""

from __future__ import annotations

from typing import Any, Dict, List

from shared_anomalies import get_patterns_for_hook

# -----------------------------------------------------------------------------
# Hook metadata for QED v6 edge lab integration
# -----------------------------------------------------------------------------
HOOK_NAME: str = "boring_tbm"
COMPANY: str = "boring"
STREAM_ID: str = "tbm_sensors"

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import qed

_CHANNELS: Dict[str, Dict[str, float | str]] = {
    "torque": {
        "pretty_name": "Cutterhead torque kN·m",
        "sample_rate_hz": 125.0,
        "window_seconds": 10.0,
        "safety_threshold": 4480.0,  # kN·m, 80 percent of 5600
        "dominant_freq_hz": 0.1667,  # 10 RPM
        "noise_sigma": 0.05,  # fraction of amplitude for demo
        "offset": 0.0,
    },
    "vibration": {
        "pretty_name": "Main bearing vibration g-RMS",
        "sample_rate_hz": 12_800.0,
        "window_seconds": 1.0,
        "safety_threshold": 3.5,  # g, alert level
        "dominant_freq_hz": 183.7,
        "noise_sigma": 0.1,
        "offset": 0.0,
    },
    "pressure": {
        "pretty_name": "Thrust pressure bar",
        "sample_rate_hz": 500.0,
        "window_seconds": 20.0,
        "safety_threshold": 315.0,  # bar
        "dominant_freq_hz": 1.0,
        "noise_sigma": 0.05,
        "offset": 250.0,
    },
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _classify_window(
    max_raw: float,
    safety_threshold: float,
    recall: float,
) -> str:
    """Classify a window into NORMAL, OPERATIONAL_LIMIT, DETECTION_FAILURE, or CRITICAL_SAFETY."""
    amp_frac = max_raw / safety_threshold if safety_threshold > 0.0 else 0.0
    if (recall < 0.999) and (amp_frac > 0.9):
        return "CRITICAL_SAFETY"
    if (recall < 0.999) and (amp_frac <= 0.9):
        return "DETECTION_FAILURE"
    if (recall >= 0.999) and (amp_frac > 0.9):
        return "OPERATIONAL_LIMIT"
    return "NORMAL"


def _health_score(
    recall: float,
    max_raw: float,
    safety_threshold: float,
    ratio: float,
    H_bits: float,  # kept for future use
) -> float:
    """
    Return a health score in [0.0, 1.0].

    Simplified version of the research formula:
      health = 0.4*recall
             + 0.3*(1 - min(amp_frac, 1))
             + 0.2*compression_quality
             + 0.1*entropy_stability (assumed 1 here)
    """
    del H_bits  # placeholder, entropy_stability assumed 1.0
    amp_frac = max_raw / safety_threshold if safety_threshold > 0.0 else 0.0
    compression_quality = _clamp((ratio - 44.0) / (65.0 - 44.0), 0.0, 1.0)
    entropy_stability = 1.0
    health = (
        0.4 * recall
        + 0.3 * (1.0 - min(amp_frac, 1.0))
        + 0.2 * compression_quality
        + 0.1 * entropy_stability
    )
    return _clamp(health, 0.0, 1.0)


def _get_channel_meta(channel: str) -> Dict[str, float | str]:
    meta = _CHANNELS.get(channel)
    if meta is None:
        raise ValueError(f"unknown channel: {channel}")
    return meta


def _make_demo_signal(
    channel: str,
    duration_sec: float,
    fault: bool,
    seed: int,
) -> np.ndarray:
    """
    Generate a synthetic TBM signal for the given channel.

    Uses _CHANNELS metadata to set sample_rate_hz, dominant frequency, noise level, and offset.
    If fault=True, injects a short central period where the signal exceeds the safety threshold.
    """
    meta = _get_channel_meta(channel)
    sample_rate_hz = float(meta["sample_rate_hz"])
    safety_threshold = float(meta["safety_threshold"])
    dominant_freq_hz = float(meta["dominant_freq_hz"])
    noise_sigma = float(meta["noise_sigma"])
    offset = float(meta["offset"])
    n = int(sample_rate_hz * duration_sec)
    if n < 256:
        raise ValueError("duration too short, need at least 256 samples")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    A = 0.8 * safety_threshold  # nominal 80 percent load
    clean = A * np.sin(2.0 * np.pi * dominant_freq_hz * t) + offset
    noise = rng.normal(0.0, noise_sigma * A, n)
    raw = clean + noise
    if fault:
        # Inject a fault in the central 10 percent of samples
        fault_width = max(int(0.10 * n), 1)
        start_fault = max((n // 2) - (fault_width // 2), 0)
        end_fault = min(start_fault + fault_width, n)
        raw[start_fault:end_fault] *= 1.1
    return raw


def _read_csv_column(path: Path, column: str) -> np.ndarray:
    """
    Read a single numeric column from a CSV file with a header row.

    Assumes comma-separated, skips rows where the column cannot be parsed as float.
    Raises ValueError if the column name is not found or no numeric values are read.
    """
    values: List[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column '{column}' not found in CSV header")
        for row in reader:
            try:
                values.append(float(row[column]))
            except (ValueError, TypeError):
                continue
    if not values:
        raise ValueError(f"no numeric values found for column '{column}'")
    return np.asarray(values, dtype=np.float64)


def _run_qed_scaled(
    segment_raw: np.ndarray,
    sample_rate_hz: float,
    safety_threshold: float,
) -> Dict[str, float | str]:
    """
    Scale a raw TBM segment into QED space and run qed.qed.

    Returns a dict with numeric fields converted to plain Python floats plus classification helpers.
    """
    if safety_threshold <= 0.0:
        raise ValueError("safety_threshold must be positive")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    scale = 10.0 / safety_threshold
    scaled = segment_raw * scale
    result = qed.qed(
        scaled,
        scenario="tesla_fsd",
        bit_depth=12,
        sample_rate_hz=sample_rate_hz,
    )
    ratio = float(result["ratio"])
    H_bits = float(result["H_bits"])
    recall = float(result["recall"])
    savings_M = float(result["savings_M"])
    trace = str(result["trace"])
    max_raw = float(np.max(np.abs(segment_raw)))
    classification = _classify_window(max_raw, safety_threshold, recall)
    health = _health_score(recall, max_raw, safety_threshold, ratio, H_bits)
    return {
        "ratio": ratio,
        "H_bits": H_bits,
        "recall": recall,
        "savings_M": savings_M,
        "trace": trace,
        "max_raw": max_raw,
        "classification": classification,
        "health": health,
    }


def get_cross_domain_config() -> Dict[str, Any]:
    """
    Return cross-domain integration configuration for Boring Company.

    Boring has no cross-domain mappings (tunnel telemetry is unique).
    """
    return {
        "exports": {},
        "accepts": {},
    }


def get_deployment_config() -> Dict[str, Any]:
    """Return Boring Company-specific QEDConfig defaults for tunneling telemetry."""
    return {
        "hook": "boring",
        "recall_floor": 0.9995,
        "max_fp_rate": 0.005,
        "slo_latency_ms": 200,
        "slo_breach_budget": 0.002,
        "compression_target": 20.0,
        "enabled_patterns": ["PAT_TBM_*", "PAT_STRUCTURAL_*", "PAT_HVAC_*", "PAT_TRANSPORT_*"],
        "regulatory_flags": {"OSHA": True, "MSHA": True, "DOT": True},
        "safety_critical": True,
    }


def get_hardware_profile() -> Dict[str, Any]:
    """Return Boring Company hardware identifiers for mesh_view clustering."""
    return {
        "platform": "tbm_controller",
        "compute_class": "industrial",
        "connectivity": "fiber_wired",
        "storage_type": "ruggedized_ssd",
        "real_time": False,
        "safety_critical": True,
    }


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Return edge lab test scenarios for Boring Company TBM telemetry.

    Each scenario is a dict with:
        - id: unique scenario identifier
        - type: scenario type (spike, step, drift, normal)
        - expected_loss: expected loss threshold (>0.1 for high-loss scenarios)
        - signal: list of float values representing the test signal
        - pattern_id: optional pattern ID from shared_anomalies (None for legacy)

    Returns hand-crafted scenarios plus any patterns from shared_anomalies.
    No cross-domain patterns (tunnel telemetry is unique).
    """
    # Hand-crafted legacy scenarios (pattern_id=None)
    legacy_scenarios = [
        {
            "id": "pressure_surge",
            "type": "spike",
            "expected_loss": 0.17,
            "signal": [320.0] * 1000,  # Exceeds 315 bar threshold
            "pattern_id": None,
        },
        {
            "id": "vibration_exceed",
            "type": "spike",
            "expected_loss": 0.15,
            "signal": [3.8] * 1000,  # Exceeds 3.5g threshold
            "pattern_id": None,
        },
        {
            "id": "torque_ramp",
            "type": "drift",
            "expected_loss": 0.12,
            "signal": [float(i * 5) for i in range(1000)],  # Gradual torque increase
            "pattern_id": None,
        },
        {
            "id": "cutterhead_step",
            "type": "step",
            "expected_loss": 0.14,
            "signal": [200.0] * 500 + [400.0] * 500,  # Sudden load change
            "pattern_id": None,
        },
        {
            "id": "tunnel_normal",
            "type": "normal",
            "expected_loss": 0.03,
            "signal": [250.0] * 1000,  # Nominal boring operation
            "pattern_id": None,
        },
    ]

    # Query shared_anomalies for patterns where "boring" in hooks
    try:
        patterns = get_patterns_for_hook("boring")
    except Exception:
        patterns = []

    # Convert patterns to scenario format
    pattern_scenarios = []
    for p in patterns:
        scenario = {
            "id": f"pattern_{p.pattern_id}",
            "type": p.failure_mode,
            "expected_loss": 1.0 - p.validation_recall if p.validation_recall > 0 else 0.1,
            "signal": p.params.get("signal", [0.0] * 1000),
            "pattern_id": p.pattern_id,
        }
        pattern_scenarios.append(scenario)

    return legacy_scenarios + pattern_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boring Company TBM hook for QED v5.0",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # demo subcommand
    demo = subparsers.add_parser("demo", help="Generate synthetic TBM signals")
    demo.add_argument("--channel", choices=list(_CHANNELS.keys()), default="torque")
    demo.add_argument("--duration-sec", type=float, default=None)
    demo.add_argument("--inject-fault", action="store_true")
    demo.add_argument("--json", action="store_true")

    # from-csv subcommand
    csv_cmd = subparsers.add_parser("from-csv", help="Process CSV TBM data")
    csv_cmd.add_argument("--file", type=Path, required=True)
    csv_cmd.add_argument("--column", required=True)
    csv_cmd.add_argument("--channel", choices=list(_CHANNELS.keys()), required=True)
    csv_cmd.add_argument("--sample-rate-hz", type=float, default=None)
    csv_cmd.add_argument("--window-sec", type=float, default=None)
    csv_cmd.add_argument("--stride-sec", type=float, default=None)
    csv_cmd.add_argument("--jsonl", action="store_true")

    args = parser.parse_args()

    if args.subcommand == "demo":
        meta = _get_channel_meta(args.channel)
        duration_sec = float(args.duration_sec or meta["window_seconds"])
        sample_rate_hz = float(meta["sample_rate_hz"])
        safety_threshold = float(meta["safety_threshold"])

        raw = _make_demo_signal(
            args.channel,
            duration_sec,
            args.inject_fault,
            seed=42,
        )

        result = _run_qed_scaled(
            segment_raw=raw,
            sample_rate_hz=sample_rate_hz,
            safety_threshold=safety_threshold,
        )

        ratio = result["ratio"]
        H_bits = result["H_bits"]
        recall = result["recall"]
        savings_M = result["savings_M"]
        trace = result["trace"]
        max_raw = result["max_raw"]
        classification = result["classification"]
        health = result["health"]

        print(
            f"Channel: {meta['pretty_name']} ({args.channel})\n"
            f"Samples: {len(raw)} at {sample_rate_hz:.1f} Hz ({duration_sec:.2f} s)\n"
            f"QED: ratio={ratio:.1f}, H_bits={H_bits:.0f}, recall={recall:.4f}, "
            f"savings_M={savings_M:.1f}\n"
            f"TBM: max={max_raw:.1f} / {safety_threshold:.1f} "
            f"(amp_frac={max_raw / safety_threshold:.2f})\n"
            f"Status: {classification}, health={health:.2f}, trace={trace}"
        )

        if args.json:
            print(
                json.dumps(
                    {
                        "channel": args.channel,
                        "pretty_name": meta["pretty_name"],
                        "sample_rate_hz": sample_rate_hz,
                        "n_samples": len(raw),
                        "safety_threshold": safety_threshold,
                        "max_raw": max_raw,
                        "ratio": ratio,
                        "H_bits": H_bits,
                        "recall": recall,
                        "savings_M": savings_M,
                        "classification": classification,
                        "health": health,
                        "trace": trace,
                    }
                )
            )

    elif args.subcommand == "from-csv":
        meta = _get_channel_meta(args.channel)
        raw_values = _read_csv_column(args.file, args.column)
        sample_rate_hz = float(args.sample_rate_hz or meta["sample_rate_hz"])
        if sample_rate_hz <= 0.0:
            raise ValueError("sample-rate-hz must be positive")

        window_sec = float(args.window_sec or meta["window_seconds"])
        stride_sec = float(args.stride_sec or (window_sec / 2.0))
        safety_threshold = float(meta["safety_threshold"])

        window_n = int(sample_rate_hz * window_sec)
        stride_n = int(sample_rate_hz * stride_sec)

        if window_n < 256:
            raise ValueError("window too short for QED, need at least 256 samples")
        if stride_n < 1:
            raise ValueError("stride too small, must be at least one sample")

        n_values = len(raw_values)
        if n_values < window_n:
            raise ValueError(
                f"not enough samples ({n_values}) for one window of length {window_n}"
            )

        window_index = 0
        for start in range(0, n_values - window_n + 1, stride_n):
            end = start + window_n
            segment = raw_values[start:end]
            window_start_sec = start / sample_rate_hz

            result = _run_qed_scaled(
                segment_raw=segment,
                sample_rate_hz=sample_rate_hz,
                safety_threshold=safety_threshold,
            )

            ratio = result["ratio"]
            H_bits = result["H_bits"]
            recall = result["recall"]
            savings_M = result["savings_M"]
            trace = result["trace"]
            max_raw = result["max_raw"]
            classification = result["classification"]
            health = result["health"]

            print(
                f"window={window_index} start={window_start_sec:.2f}s len={len(segment)}\n"
                f"  ratio={ratio:.1f} H_bits={H_bits:.0f} recall={recall:.4f} "
                f"savings_M={savings_M:.1f}\n"
                f"  max_raw={max_raw:.1f} threshold={safety_threshold:.1f} "
                f"status={classification} health={health:.2f}"
            )

            if args.jsonl:
                print(
                    json.dumps(
                        {
                            "window_index": window_index,
                            "start_sec": window_start_sec,
                            "channel": args.channel,
                            "ratio": ratio,
                            "H_bits": H_bits,
                            "recall": recall,
                            "savings_M": savings_M,
                            "max_raw": max_raw,
                            "safety_threshold": safety_threshold,
                            "classification": classification,
                            "health": health,
                            "trace": trace,
                        }
                    )
                )

            window_index += 1


if __name__ == "__main__":
    main()
