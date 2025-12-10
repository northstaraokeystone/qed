"""
Neuralink style hook for QED v5.0.

Normalizes neural telemetry signals into QED space by scaling each channel so the
safety threshold maps to 10.0 in QED space (QED's recall threshold).

CLI:

  python -m hooks.neuralink demo        # synthetic signals
  python -m hooks.neuralink from-csv    # real data windowing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import qed

try:
    from shared_anomalies import get_patterns_for_hook
except ImportError:
    def get_patterns_for_hook(hook_name: str) -> List:
        """Fallback if shared_anomalies module not ready."""
        return []

# -----------------------------------------------------------------------------
# Hook metadata for QED v7 edge lab integration
# -----------------------------------------------------------------------------
HOOK_NAME: str = "neuralink_neural"
COMPANY: str = "neuralink"
STREAM_ID: str = "neural_telemetry"

_NEURAL_CHANNELS: Dict[str, Dict[str, float | str]] = {
    "micro_ecog": {
        "pretty_name": "MicroECoG voltage μV",
        "sample_rate_hz": 30000.0,
        "window_seconds": 1.0,
        "safety_threshold": 1000.0,  # microvolts, epileptiform threshold
        "dominant_freq_hz": 250.0,  # ripple band
        "noise_fraction": 0.1,
        "baseline": 0.0,
    },
    "gamma_power": {
        "pretty_name": "High gamma power μV²/Hz",
        "sample_rate_hz": 1000.0,
        "window_seconds": 1.0,
        "safety_threshold": 80.0,
        "dominant_freq_hz": 80.0,  # gamma band center
        "noise_fraction": 0.1,
        "baseline": 0.0,
    },
    "decoder_conf": {
        "pretty_name": "Decoder confidence percent",
        "sample_rate_hz": 50.0,
        "window_seconds": 50.0,
        "safety_threshold": 70.0,
        "dominant_freq_hz": 0.1,
        "noise_fraction": 0.05,
        "baseline": 80.0,  # typical good performance
    },
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_channel_meta(channel: str) -> Dict[str, float | str]:
    meta = _NEURAL_CHANNELS.get(channel)
    if meta is None:
        raise ValueError(f"unknown channel: {channel}")
    return meta


def _classify_window(
    max_raw: float,
    safety_threshold: float,
    recall: float,
) -> str:
    """
    Map amplitude fraction and QED recall into a simple clinical label.

    Returns one of:
      NO_SIGNAL, NORMAL, WATCH, ALERT, CRITICAL
    """
    if safety_threshold <= 0.0 or max_raw <= 0.0:
        return "NO_SIGNAL"

    amp_frac = max_raw / safety_threshold

    if recall < 0.999 or amp_frac >= 1.2:
        return "CRITICAL"
    if amp_frac >= 1.0:
        return "ALERT"
    if amp_frac >= 0.8:
        return "WATCH"
    return "NORMAL"


def _health_score(
    recall: float,
    max_raw: float,
    safety_threshold: float,
    ratio: float,
    H_bits: float,  # kept for future entropy based refinements
) -> float:
    if safety_threshold > 0.0:
        amp_frac = max_raw / safety_threshold
    else:
        amp_frac = 0.0

    compression_quality = _clamp((ratio - 44.0) / (65.0 - 44.0), 0.0, 1.0)
    entropy_stability = 1.0

    health = (
        0.4 * recall
        + 0.3 * (1.0 - min(amp_frac, 1.0))
        + 0.2 * compression_quality
        + 0.1 * entropy_stability
    )
    return _clamp(health, 0.0, 1.0)


def _make_demo_signal(
    channel: str,
    duration_sec: float,
    inject_event: bool,
    seed: int,
) -> np.ndarray:
    """
    Generate a synthetic neural signal for the given channel.

    Uses _NEURAL_CHANNELS metadata to set sample_rate_hz, dominant frequency,
    noise properties, and baseline. If inject_event is True, injects a central
    burst that clearly exceeds the safety threshold.
    """
    meta = _get_channel_meta(channel)
    sample_rate_hz = float(meta["sample_rate_hz"])
    safety_threshold = float(meta["safety_threshold"])
    dominant_freq_hz = float(meta["dominant_freq_hz"])
    noise_fraction = float(meta["noise_fraction"])
    baseline = float(meta["baseline"])

    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")

    n = int(sample_rate_hz * duration_sec)
    if n < 256:
        raise ValueError("duration too short, need at least 256 samples")

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    # Nominal sub threshold amplitude
    A = 0.6 * safety_threshold
    clean = baseline + A * np.sin(2.0 * np.pi * dominant_freq_hz * t)
    noise_sigma = noise_fraction * A
    noise = rng.normal(0.0, noise_sigma, n)
    raw = clean + noise

    if inject_event:
        # Inject event in the central region and push above threshold
        start_event = n // 2 - n // 20
        end_event = n // 2 + n // 20
        start_event = max(start_event, 0)
        end_event = min(end_event, n)
        if start_event < end_event:
            raw[start_event:end_event] += 0.7 * safety_threshold

    return raw.astype(np.float64)


def _read_csv_column(path: Path, column: str) -> np.ndarray:
    """
    Read a single numeric column from a CSV file with a header row.

    Assumes comma separated values.
    Skips rows where the column cannot be parsed as float.
    Raises ValueError if the column name is not found or no numeric values are read.
    """
    values: list[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column '{column}' not found in CSV header")
        for row in reader:
            raw_val = row.get(column)
            if raw_val is None:
                continue
            try:
                values.append(float(raw_val))
            except ValueError:
                continue

    if not values:
        raise ValueError(f"no numeric values found for column '{column}'")

    return np.asarray(values, dtype=np.float64)


def _run_qed_scaled(
    segment_raw: np.ndarray,
    sample_rate_hz: float,
    safety_threshold: float,
) -> Dict[str, Any]:
    """
    Scale raw neural units so safety_threshold maps to 10.0 in QED space,
    run qed.qed, and return a flat dict with QED metrics plus classification.

    If the segment has no signal (max absolute value is zero), returns a
    "NO_SIGNAL" result without calling qed.qed.
    """
    if segment_raw.size == 0:
        raise ValueError("segment_raw must be non empty")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if safety_threshold <= 0.0:
        raise ValueError("safety_threshold must be positive")

    max_raw = float(np.max(np.abs(segment_raw)))
    if max_raw == 0.0:
        return {
            "ratio": 0.0,
            "H_bits": 0.0,
            "recall": 1.0,
            "savings_M": 0.0,
            "trace": "neuralink_no_signal",
            "max_raw": 0.0,
            "safety_threshold": float(safety_threshold),
            "classification": "NO_SIGNAL",
            "health": 0.0,
        }

    scale = 10.0 / safety_threshold
    scaled = segment_raw.astype(np.float64) * scale

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

    classification = _classify_window(max_raw, safety_threshold, recall)
    health = _health_score(recall, max_raw, safety_threshold, ratio, H_bits)

    return {
        "ratio": ratio,
        "H_bits": H_bits,
        "recall": recall,
        "savings_M": savings_M,
        "trace": trace,
        "max_raw": max_raw,
        "safety_threshold": float(safety_threshold),
        "classification": classification,
        "health": health,
    }


def get_cross_domain_config() -> Dict[str, Any]:
    """
    Return cross-domain integration configuration for Neuralink.

    Neuralink has no cross-domain mappings - neural stream telemetry
    is a unique physics domain with no exports or accepts.
    """
    return {
        "exports": {},
        "accepts": {},
    }


def get_deployment_config() -> Dict[str, Any]:
    """Return Neuralink-specific QEDConfig defaults for medical-grade neural telemetry."""
    return {
        "hook": "neuralink",
        "recall_floor": 0.999999,
        "max_fp_rate": 0.0001,
        "slo_latency_ms": 5,
        "slo_breach_budget": 0.00001,
        "compression_target": 5.0,
        "enabled_patterns": ["PAT_NEURAL_*", "PAT_IMPLANT_*", "PAT_POWER_*", "PAT_THERMAL_*"],
        "regulatory_flags": {"FDA": True, "HIPAA": True, "ISO13485": True, "IEC62304": True},
        "safety_critical": True,
    }


def get_hardware_profile() -> Dict[str, Any]:
    """Return Neuralink hardware identifiers for mesh_view clustering."""
    return {
        "platform": "implant_asic",
        "compute_class": "ultra_low_power",
        "connectivity": "wireless_implant",
        "storage_type": "none",
        "real_time": True,
        "safety_critical": True,
    }


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Return edge lab test scenarios for Neuralink neural telemetry.

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
            "id": "ecog_spike",
            "type": "spike",
            "expected_loss": 0.15,
            "signal": [1100.0] * 1000,  # Above 1000μV epileptiform threshold
            "pattern_id": None,
        },
        {
            "id": "gamma_surge",
            "type": "spike",
            "expected_loss": 0.12,
            "signal": [90.0] * 1000,  # Above 80μV²/Hz gamma threshold
            "pattern_id": None,
        },
        {
            "id": "decoder_drift",
            "type": "drift",
            "expected_loss": 0.18,
            "signal": list(range(100, 0, -1)) * 10,  # Decoder confidence degradation
            "pattern_id": None,
        },
        {
            "id": "neural_baseline",
            "type": "normal",
            "expected_loss": 0.05,
            "signal": [500.0] * 1000,  # Nominal neural activity
            "pattern_id": None,
        },
    ]

    # Query shared_anomalies for patterns where "neuralink" in hooks
    try:
        patterns = get_patterns_for_hook("neuralink")
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
        description="Neuralink style neural telemetry hook for QED v5.0",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    demo = subparsers.add_parser(
        "demo",
        help="Generate synthetic neural signals and run QED on a single window",
    )
    demo.add_argument(
        "--channel",
        choices=list(_NEURAL_CHANNELS.keys()),
        default="micro_ecog",
    )
    demo.add_argument("--duration-sec", type=float, default=None)
    demo.add_argument("--inject-event", action="store_true")
    demo.add_argument("--json", action="store_true")

    csv_cmd = subparsers.add_parser(
        "from-csv",
        help="Process CSV neural data in sliding windows",
    )
    csv_cmd.add_argument("--file", type=Path, required=True)
    csv_cmd.add_argument("--column", required=True)
    csv_cmd.add_argument(
        "--channel",
        choices=list(_NEURAL_CHANNELS.keys()),
        required=True,
    )
    csv_cmd.add_argument("--sample-rate-hz", type=float, default=None)
    csv_cmd.add_argument("--window-sec", type=float, default=None)
    csv_cmd.add_argument("--stride-sec", type=float, default=None)
    csv_cmd.add_argument("--jsonl", action="store_true")

    args = parser.parse_args()

    if args.subcommand == "demo":
        meta = _get_channel_meta(args.channel)
        sample_rate_hz = float(meta["sample_rate_hz"])
        safety_threshold = float(meta["safety_threshold"])
        window_seconds = float(meta["window_seconds"])

        duration_sec = (
            float(args.duration_sec)
            if args.duration_sec is not None
            else window_seconds
        )

        raw = _make_demo_signal(
            channel=args.channel,
            duration_sec=duration_sec,
            inject_event=bool(args.inject_event),
            seed=42424242,
        )

        stats = _run_qed_scaled(raw, sample_rate_hz, safety_threshold)

        ratio = stats["ratio"]
        H_bits = stats["H_bits"]
        recall = stats["recall"]
        savings_M = stats["savings_M"]
        trace = stats["trace"]
        max_raw = stats["max_raw"]
        classification = stats["classification"]
        health = stats["health"]

        print(
            f"Channel: {meta['pretty_name']} ({args.channel})\n"
            f"Samples: {raw.size} at {sample_rate_hz:.1f} Hz "
            f"({duration_sec:.3f} s)\n"
            f"QED: ratio={ratio:.1f}, H_bits={H_bits:.0f}, "
            f"recall={recall:.4f}, savings_M={savings_M:.2f}\n"
            f"Neural: max={max_raw:.1f} / {safety_threshold} "
            f"(amp_frac={max_raw / safety_threshold:.2f})\n"
            f"Status: {classification}, health={health:.2f}, trace={trace}"
        )

        if args.json:
            payload: Dict[str, Any] = {
                "channel": args.channel,
                "pretty_name": meta["pretty_name"],
                "sample_rate_hz": sample_rate_hz,
                "n_samples": int(raw.size),
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
            print(json.dumps(payload))

    elif args.subcommand == "from-csv":
        meta = _get_channel_meta(args.channel)
        raw_values = _read_csv_column(args.file, args.column)

        default_sample_rate = float(meta["sample_rate_hz"])
        default_window = float(meta["window_seconds"])
        safety_threshold = float(meta["safety_threshold"])

        sample_rate_hz = (
            float(args.sample_rate_hz)
            if args.sample_rate_hz is not None
            else default_sample_rate
        )
        window_sec = (
            float(args.window_sec) if args.window_sec is not None else default_window
        )
        stride_sec = (
            float(args.stride_sec) if args.stride_sec is not None else window_sec / 2.0
        )

        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")

        window_n = int(sample_rate_hz * window_sec)
        stride_n = int(sample_rate_hz * stride_sec)

        if window_n < 256:
            raise ValueError("window too short for QED, need at least 256 samples")
        if stride_n <= 0:
            raise ValueError("stride_sec must produce at least one sample per step")

        n_values = int(raw_values.size)
        if n_values < window_n:
            raise ValueError(
                f"not enough samples for one window: have {n_values}, "
                f"need at least {window_n}",
            )

        window_index = 0
        start = 0
        while start + window_n <= n_values:
            end = start + window_n
            segment = raw_values[start:end]
            stats = _run_qed_scaled(segment, sample_rate_hz, safety_threshold)

            ratio = stats["ratio"]
            H_bits = stats["H_bits"]
            recall = stats["recall"]
            savings_M = stats["savings_M"]
            trace = stats["trace"]
            max_raw = stats["max_raw"]
            classification = stats["classification"]
            health = stats["health"]
            start_sec = start / sample_rate_hz

            print(
                f"window={window_index} start={start_sec:.2f}s len={segment.size}\n"
                f"  ratio={ratio:.1f} H_bits={H_bits:.0f} "
                f"recall={recall:.4f} savings_M={savings_M:.2f}\n"
                f"  max_raw={max_raw:.1f} threshold={safety_threshold} "
                f"status={classification} health={health:.2f} trace={trace}"
            )

            if args.jsonl:
                payload: Dict[str, Any] = {
                    "window_index": window_index,
                    "start_sec": start_sec,
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
                print(json.dumps(payload))

            window_index += 1
            start += stride_n

    else:
        parser.error(f"unknown subcommand: {args.subcommand!r}")


if __name__ == "__main__":
    main()
