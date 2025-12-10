"""
Tesla hook for QED v5.0.

Normalizes Tesla vehicle telemetry signals into QED space by scaling each channel so the
primary safety threshold maps to 10.0 in QED space (QED's recall threshold). Exposes a
CLI with "demo" for synthetic signals and "from-csv" for windowed processing of real data.
"""

from typing import Any, Dict, List

from shared_anomalies import get_patterns_for_hook

# -----------------------------------------------------------------------------
# Hook metadata for QED v6 edge lab integration
# -----------------------------------------------------------------------------
HOOK_NAME: str = "tesla_torque"
COMPANY: str = "tesla"
STREAM_ID: str = "can_bus"

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import qed

_TESLA_CHANNELS: Dict[str, Dict[str, Any]] = {
    "steering_torque": {
        "pretty_name": "Steering column torque (Nm)",
        "sample_rate_hz": 1000.0,
        "window_seconds": 1.0,
        "safety_threshold": 65.0,  # Nm, emergency override
        "dominant_freq_hz": 2.5,  # driver correction fundamental
        "noise_fraction": 0.08,
        "baseline": 0.0,
    },
    "brake_pressure": {
        "pretty_name": "Brake system pressure (bar)",
        "sample_rate_hz": 500.0,
        "window_seconds": 2.0,
        "safety_threshold": 180.0,  # bar, emergency braking
        "dominant_freq_hz": 17.333,  # ABS modulation
        "noise_fraction": 0.10,
        "baseline": 2.0,  # nominal low pressure offset
    },
    "lateral_accel": {
        "pretty_name": "Lateral acceleration (m/s^2)",
        "sample_rate_hz": 100.0,
        "window_seconds": 10.0,
        "safety_threshold": 8.5,  # m/s^2, stability control threshold
        "dominant_freq_hz": 0.35,  # lane change fundamental
        "noise_fraction": 0.10,
        "baseline": 0.0,
    },
    "battery_delta_t": {
        "pretty_name": "Battery cell ΔT (degC)",
        "sample_rate_hz": 1.0,
        "window_seconds": 300.0,
        "safety_threshold": 12.0,  # degC, thermal runaway threshold
        "dominant_freq_hz": 0.03333,
        "noise_fraction": 0.05,
        "baseline": 0.0,
    },
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_channel_meta(channel: str) -> Dict[str, Any]:
    meta = _TESLA_CHANNELS.get(channel)
    if meta is None:
        raise ValueError(f"unknown channel: {channel}")
    return meta


def _classify_window(max_raw: float, safety_threshold: float, recall: float) -> str:
    """
    Map amplitude fraction and recall into a five level ladder.

    This reflects the spec hierarchy:
      Level 1 NOMINAL
      Level 2 WATCH
      Level 3 ALERT
      Level 4 CRITICAL
      Level 5 ABORT
    """
    if safety_threshold <= 0.0:
        return "NO_SIGNAL"

    amp_frac = max_raw / safety_threshold if safety_threshold > 0.0 else 0.0

    if recall < 0.95 or amp_frac >= 1.5:
        return "ABORT"
    if recall < 0.999 or amp_frac >= 1.2:
        return "CRITICAL"
    if amp_frac >= 1.0:
        return "ALERT"
    if amp_frac >= 0.8:
        return "WATCH"
    return "NOMINAL"


def _health_score(
    recall: float,
    max_raw: float,
    safety_threshold: float,
    ratio: float,
    H_bits: float,  # reserved for future entropy checks
) -> float:
    """
    One dimensional health index in [0, 1].

    Higher is better. Combines recall, distance from threshold, and compression quality.
    """
    amp_frac = max_raw / safety_threshold if safety_threshold > 0.0 else 0.0
    compression_quality = _clamp((ratio - 44.0) / (65.0 - 44.0), 0.0, 1.0)
    entropy_stability = 1.0

    health = (
        0.35 * recall
        + 0.30 * (1.0 - min(amp_frac / 1.5, 1.0))
        + 0.20 * compression_quality
        + 0.15 * entropy_stability
    )
    return _clamp(health, 0.0, 1.0)


def _make_demo_signal(
    channel: str,
    duration_sec: float,
    inject_event: bool,
    seed: int,
) -> np.ndarray:
    """
    Generate a synthetic Tesla telemetry signal for the given channel.

    Uses metadata to set sample rate, dominant frequency, noise, and baseline.
    If inject_event is true, a central window is boosted above the safety threshold.
    """
    meta = _get_channel_meta(channel)
    fs: float = float(meta["sample_rate_hz"])
    safety_threshold: float = float(meta["safety_threshold"])
    f0: float = float(meta["dominant_freq_hz"])
    noise_fraction: float = float(meta["noise_fraction"])
    baseline: float = float(meta["baseline"])

    n = int(fs * duration_sec)
    if n < 256:
        raise ValueError("duration too short, need at least 256 samples")

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / fs

    # Nominal amplitude at 60 percent of safety threshold.
    A = 0.6 * safety_threshold
    clean = baseline + A * np.sin(2.0 * np.pi * f0 * t)
    noise_sigma = noise_fraction * A
    noise = rng.normal(0.0, noise_sigma, size=n)
    raw = clean + noise

    if inject_event:
        start = n // 4
        end = 3 * n // 4
        # Increase amplitude during event window.
        raw[start:end] *= 1.3

    # Respect non negative physics for some channels.
    if channel in ("brake_pressure", "battery_delta_t"):
        raw = np.maximum(raw, 0.0)

    return raw.astype(np.float64)


def _read_csv_column(path: Path, column: str) -> np.ndarray:
    """
    Read a single numeric column from a CSV file with a header row.

    Assumes comma separated values.
    Skips rows where the column cannot be parsed as float.
    Raises ValueError if the column name is not found or no numeric values are read.
    """
    values = []
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


def get_cross_domain_config() -> Dict[str, Any]:
    """
    Return cross-domain integration configuration for Tesla.

    Tesla is a SOURCE for:
      - battery_thermal → exports to SpaceX
      - comms → exports to Starlink

    Tesla accepts nothing (pure source).
    """
    return {
        "exports": {
            "battery_thermal": ["spacex"],
            "comms": ["starlink"],
        },
        "accepts": {},
    }


def get_deployment_config() -> Dict[str, Any]:
    """Return Tesla-specific QEDConfig defaults for automotive telemetry."""
    return {
        "hook": "tesla",
        "recall_floor": 0.9995,
        "max_fp_rate": 0.005,
        "slo_latency_ms": 50,
        "slo_breach_budget": 0.001,
        "compression_target": 10.0,
        "enabled_patterns": ["PAT_CAN_*", "PAT_BATTERY_*", "PAT_THERMAL_*", "PAT_MOTION_*"],
        "regulatory_flags": {"NHTSA": True, "ISO26262": True},
        "safety_critical": True,
    }


def get_hardware_profile() -> Dict[str, Any]:
    """Return Tesla hardware identifiers for mesh_view clustering."""
    return {
        "platform": "vehicle_ecu",
        "compute_class": "embedded",
        "connectivity": "cellular",
        "storage_type": "flash",
        "real_time": True,
        "safety_critical": True,
    }


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Return edge lab test scenarios for Tesla telemetry.

    Each scenario is a dict with:
        - id: unique scenario identifier
        - type: scenario type (spike, drift, step, normal)
        - expected_loss: expected loss threshold (>0.1 for high-loss scenarios)
        - signal: list of float values representing the test signal
        - pattern_id: optional pattern ID from shared_anomalies (None for legacy)

    Returns hand-crafted scenarios plus any patterns from shared_anomalies.
    """
    # Hand-crafted legacy scenarios (pattern_id=None)
    legacy_scenarios = [
        {
            "id": "torque_spike",
            "type": "spike",
            "expected_loss": 0.15,
            "signal": [14.8] * 1000,  # Above 14.7g safety threshold
            "pattern_id": None,
        },
        {
            "id": "speed_drift",
            "type": "drift",
            "expected_loss": 0.12,
            "signal": list(range(1000)),  # Gradual drift pattern
            "pattern_id": None,
        },
        {
            "id": "brake_anomaly",
            "type": "step",
            "expected_loss": 0.18,
            "signal": [0.0] * 500 + [20.0] * 500,  # Sudden step change
            "pattern_id": None,
        },
        {
            "id": "accel_exceed",
            "type": "spike",
            "expected_loss": 0.20,
            "signal": [15.0] * 1000,  # Exceeds 14.7g bound
            "pattern_id": None,
        },
        {
            "id": "normal_drive",
            "type": "normal",
            "expected_loss": 0.05,
            "signal": [10.0] * 1000,  # Nominal operation
            "pattern_id": None,
        },
    ]

    # Query shared_anomalies for patterns where "tesla" in hooks
    try:
        patterns = get_patterns_for_hook("tesla")
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
        description="Tesla vehicle telemetry hook for QED v5.0",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # demo subcommand
    demo = subparsers.add_parser(
        "demo",
        help="Generate synthetic Tesla telemetry for a single channel",
    )
    demo.add_argument(
        "--channel",
        choices=list(_TESLA_CHANNELS.keys()),
        default="steering_torque",
    )
    demo.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Duration of the synthetic signal in seconds "
        "(defaults to the channel window length)",
    )
    demo.add_argument(
        "--inject-event",
        action="store_true",
        help="Inject a safety relevant event that exceeds the threshold",
    )
    demo.add_argument(
        "--json",
        action="store_true",
        help="Emit a single line JSON summary in addition to human text",
    )

    # from-csv subcommand
    csv_cmd = subparsers.add_parser(
        "from-csv",
        help="Process real Tesla telemetry from a CSV file",
    )
    csv_cmd.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to CSV file with header row",
    )
    csv_cmd.add_argument(
        "--column",
        required=True,
        help="Name of the numeric column to process",
    )
    csv_cmd.add_argument(
        "--channel",
        choices=list(_TESLA_CHANNELS.keys()),
        required=True,
        help="Telemetry channel metadata to apply",
    )
    csv_cmd.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
        help="Override sample rate in Hz (defaults to channel metadata)",
    )
    csv_cmd.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="Window length in seconds (defaults to channel metadata)",
    )
    csv_cmd.add_argument(
        "--stride-sec",
        type=float,
        default=None,
        help="Stride between windows in seconds (defaults to window_sec / 2)",
    )
    csv_cmd.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit JSONL, one record per window, in addition to human text",
    )

    args = parser.parse_args()

    if args.subcommand == "demo":
        meta = _get_channel_meta(args.channel)
        duration_sec = float(args.duration_sec or meta["window_seconds"])
        raw = _make_demo_signal(
            args.channel,
            duration_sec=duration_sec,
            inject_event=args.inject_event,
            seed=42,
        )
        sample_rate_hz = float(meta["sample_rate_hz"])
        safety_threshold = float(meta["safety_threshold"])
        scale = 10.0 / safety_threshold if safety_threshold > 0.0 else 1.0
        scaled = raw * scale

        result = qed.qed(
            scaled,
            scenario="tesla_fsd",
            bit_depth=16,
            sample_rate_hz=sample_rate_hz,
        )

        ratio = float(result.get("ratio", 0.0))
        H_bits = float(result.get("H_bits", 0.0))
        recall = float(result.get("recall", 0.0))
        savings_M = float(result.get("savings_M", 0.0))
        trace = str(result.get("trace", ""))

        max_raw = float(np.max(np.abs(raw))) if raw.size > 0 else 0.0
        classification = _classify_window(max_raw, safety_threshold, recall)
        health = _health_score(recall, max_raw, safety_threshold, ratio, H_bits)

        print(
            f"Channel: {meta['pretty_name']} ({args.channel})\n"
            f"Samples: {len(raw)} at {sample_rate_hz:.1f} Hz ({duration_sec:.3f} s)\n"
            f"QED: ratio={ratio:.1f}, H_bits={H_bits:.1f}, "
            f"recall={recall:.6f}, savings_M={savings_M:.2f}\n"
            f"Tesla: max={max_raw:.3f} / {safety_threshold:.3f} "
            f"(amp_frac={max_raw / safety_threshold if safety_threshold > 0 else 0.0:.3f})\n"
            f"Status: {classification}, health={health:.3f}, trace={trace}"
        )

        if args.json:
            summary = {
                "channel": args.channel,
                "pretty_name": meta["pretty_name"],
                "sample_rate_hz": sample_rate_hz,
                "n_samples": int(len(raw)),
                "duration_sec": float(duration_sec),
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
            print(json.dumps(summary))

    elif args.subcommand == "from-csv":
        meta = _get_channel_meta(args.channel)
        raw_values = _read_csv_column(args.file, args.column)

        sample_rate_hz = float(args.sample_rate_hz or meta["sample_rate_hz"])
        window_sec = float(args.window_sec or meta["window_seconds"])
        stride_sec = float(args.stride_sec or (window_sec / 2.0))
        safety_threshold = float(meta["safety_threshold"])

        window_n = int(sample_rate_hz * window_sec)
        stride_n = int(sample_rate_hz * stride_sec)
        if window_n < 256:
            raise ValueError("window too short for QED, need at least 256 samples")

        n_values = len(raw_values)
        if n_values < window_n:
            raise ValueError(
                f"not enough samples for a single window "
                f"(have {n_values}, need at least {window_n})"
            )

        scale = 10.0 / safety_threshold if safety_threshold > 0.0 else 1.0

        window_index = 0
        for start in range(0, n_values - window_n + 1, stride_n):
            end = start + window_n
            segment = raw_values[start:end]
            if segment.size == 0:
                continue

            max_raw = float(np.max(np.abs(segment)))
            scaled = segment * scale

            result = qed.qed(
                scaled,
                scenario="tesla_fsd",
                bit_depth=16,
                sample_rate_hz=sample_rate_hz,
            )

            ratio = float(result.get("ratio", 0.0))
            H_bits = float(result.get("H_bits", 0.0))
            recall = float(result.get("recall", 0.0))
            savings_M = float(result.get("savings_M", 0.0))
            trace = str(result.get("trace", ""))

            classification = _classify_window(max_raw, safety_threshold, recall)
            health = _health_score(recall, max_raw, safety_threshold, ratio, H_bits)
            window_start_sec = start / sample_rate_hz

            print(
                f"window={window_index} start={window_start_sec:.3f}s len={len(segment)}\n"
                f"  ratio={ratio:.1f} H_bits={H_bits:.1f} recall={recall:.6f} "
                f"savings_M={savings_M:.2f}\n"
                f"  max_raw={max_raw:.3f} threshold={safety_threshold:.3f} "
                f"status={classification} health={health:.3f}"
            )

            if args.jsonl:
                record = {
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
                print(json.dumps(record))

            window_index += 1

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
