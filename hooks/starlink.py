"""
Starlink hook for QED v5.0.

Normalizes Starlink satellite constellation telemetry signals into QED space by scaling
each channel so the safety threshold maps to 10.0 internal units (QED's recall threshold).
Exposes a CLI with:
  - 'demo' for synthetic satellite signals
  - 'from-csv' for windowed processing of real telemetry data
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import qed
from shared_anomalies import get_patterns_for_hook

# -----------------------------------------------------------------------------
# Hook metadata for QED v6 edge lab integration
# -----------------------------------------------------------------------------
HOOK_NAME: str = "starlink_link"
COMPANY: str = "starlink"
STREAM_ID: str = "satellite_bus"

_STARLINK_CHANNELS: Dict[str, Dict[str, Any]] = {
    "link_drop_rate": {
        "pretty_name": "Link drop rate (events/sec)",
        "sample_rate_hz": 100.0,
        "window_seconds": 10.0,
        "safety_threshold": 0.1,  # drops per second
        "dominant_freq_hz": 0.05,
        "noise_fraction": 0.08,
        "baseline": 0.01,
    },
    "beam_power": {
        "pretty_name": "Beam transmit power (dBm)",
        "sample_rate_hz": 1000.0,
        "window_seconds": 1.0,
        "safety_threshold": 33.0,  # dBm max
        "dominant_freq_hz": 10.0,
        "noise_fraction": 0.05,
        "baseline": 27.0,
    },
    "orbital_drift": {
        "pretty_name": "Orbital position drift (m)",
        "sample_rate_hz": 10.0,
        "window_seconds": 60.0,
        "safety_threshold": 500.0,  # meters
        "dominant_freq_hz": 0.01,
        "noise_fraction": 0.10,
        "baseline": 0.0,
    },
    "thermal_delta": {
        "pretty_name": "Thermal differential (degC)",
        "sample_rate_hz": 1.0,
        "window_seconds": 300.0,
        "safety_threshold": 45.0,  # degC
        "dominant_freq_hz": 0.001,
        "noise_fraction": 0.05,
        "baseline": 20.0,
    },
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_channel_meta(channel: str) -> Dict[str, Any]:
    meta = _STARLINK_CHANNELS.get(channel)
    if meta is None:
        raise ValueError(f"unknown channel: {channel}")
    return meta


def _classify_window(
    max_raw: float,
    safety_threshold: float,
    recall: float,
) -> str:
    """
    Safety classification for Starlink telemetry.

    Levels:
      NOMINAL: recall >= 0.999 and amp < 0.8 * threshold
      WATCH:   recall >= 0.999 and 0.8 <= amp/threshold < 1.0
      ALERT:   recall >= 0.999 and 1.0 <= amp/threshold < 1.2
      CRITICAL: recall >= 0.95 and 1.2 <= amp/threshold < 1.5
      ABORT:   recall < 0.95 or amp/threshold >= 1.5
    """
    if safety_threshold <= 0.0:
        return "NO_SIGNAL"

    amp_frac = max_raw / safety_threshold

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
    H_bits: float,
) -> float:
    """
    Aggregate health score in [0, 1] for Starlink telemetry.
    """
    del H_bits  # Reserved for entropy checks
    if safety_threshold <= 0.0:
        amp_frac = 0.0
    else:
        amp_frac = max_raw / safety_threshold

    compression_quality = _clamp((ratio - 44.0) / (65.0 - 44.0), 0.0, 1.0)
    entropy_stability = 1.0

    health = (
        0.35 * recall
        + 0.30 * (1.0 - min(amp_frac / 1.5, 1.0))
        + 0.20 * compression_quality
        + 0.15 * entropy_stability
    )
    return _clamp(health, 0.0, 1.0)


def _normalize_segment(
    segment: np.ndarray,
    baseline: float,
    safety_threshold: float,
) -> np.ndarray:
    """
    Map raw engineering units into QED units.

    safety_threshold is mapped to 10.0. Values are clipped to +/-14.7.
    """
    if safety_threshold <= 0:
        raise ValueError("safety_threshold must be positive for normalization")

    scale = 10.0 / safety_threshold
    scaled = (segment.astype(np.float64) - baseline) * scale
    return np.clip(scaled, -14.7, 14.7)


def _make_demo_signal(
    channel: str,
    duration_sec: float,
    inject_event: bool,
    seed: int,
) -> np.ndarray:
    """
    Generate a synthetic Starlink telemetry signal for the given channel.
    """
    meta = _get_channel_meta(channel)
    sample_rate_hz = float(meta["sample_rate_hz"])
    safety_threshold = float(meta["safety_threshold"])
    dominant_freq_hz = float(meta["dominant_freq_hz"])
    noise_fraction = float(meta["noise_fraction"])
    baseline = float(meta["baseline"])

    n = int(sample_rate_hz * duration_sec)
    if n < 256:
        raise ValueError("duration too short, need at least 256 samples")

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    # Keep nominal amplitude below threshold
    if safety_threshold > baseline:
        amp_span = safety_threshold - baseline
    else:
        amp_span = safety_threshold
    amplitude = 0.6 * amp_span

    clean = baseline + amplitude * np.sin(2.0 * np.pi * dominant_freq_hz * t)
    noise_sigma = noise_fraction * amplitude
    noise = rng.normal(0.0, noise_sigma, size=n)
    raw = clean + noise

    if inject_event:
        start = n // 4
        end = 3 * n // 4
        raw[start:end] *= 1.4  # Push above safety threshold

    return raw.astype(np.float64)


def _read_csv_column(path: Path, column: str) -> np.ndarray:
    """
    Read a single numeric column from a CSV file with a header row.
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


def get_cross_domain_config() -> Dict[str, Any]:
    """
    Return cross-domain integration configuration for Starlink.

    Starlink is a TARGET for:
      - comms from Tesla

    Starlink exports nothing.
    """
    return {
        "exports": {},
        "accepts": {
            "comms": "tesla",
        },
    }


def get_deployment_config() -> Dict[str, Any]:
    """Return Starlink-specific QEDConfig defaults for satellite fleet telemetry."""
    return {
        "hook": "starlink",
        "recall_floor": 0.999,
        "max_fp_rate": 0.01,
        "slo_latency_ms": 100,
        "slo_breach_budget": 0.005,
        "compression_target": 50.0,
        "enabled_patterns": ["PAT_ORBITAL_*", "PAT_LINK_*", "PAT_THERMAL_*", "PAT_POWER_*"],
        "regulatory_flags": {"FCC": True, "ITU": True},
        "safety_critical": False,
    }


def get_hardware_profile() -> Dict[str, Any]:
    """Return Starlink hardware identifiers for mesh_view clustering."""
    return {
        "platform": "satellite_bus",
        "compute_class": "embedded_leo",
        "connectivity": "laser_mesh",
        "storage_type": "minimal_flash",
        "real_time": False,
        "safety_critical": False,
    }


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Return edge lab test scenarios for Starlink telemetry.

    Each scenario is a dict with:
        - id: unique scenario identifier
        - type: scenario type (spike, step, drift, normal)
        - expected_loss: expected loss threshold (>0.1 for high-loss scenarios)
        - signal: list of float values representing the test signal
        - pattern_id: optional pattern ID from shared_anomalies (None for legacy)

    Returns hand-crafted scenarios plus any patterns from shared_anomalies,
    including cross-domain patterns from Tesla comms.
    """
    # Hand-crafted legacy scenarios (pattern_id=None)
    legacy_scenarios = [
        {
            "id": "link_drop_burst",
            "type": "spike",
            "expected_loss": 0.16,
            "signal": [0.15] * 1000,  # Exceeds 0.1 drop rate threshold
            "pattern_id": None,
        },
        {
            "id": "beam_fail",
            "type": "step",
            "expected_loss": 0.18,
            "signal": [27.0] * 500 + [35.0] * 500,  # Beam power spike
            "pattern_id": None,
        },
        {
            "id": "orbital_drift_exceed",
            "type": "drift",
            "expected_loss": 0.14,
            "signal": [float(i * 0.6) for i in range(1000)],  # Drift to 600m
            "pattern_id": None,
        },
        {
            "id": "thermal_runaway",
            "type": "spike",
            "expected_loss": 0.17,
            "signal": [50.0] * 1000,  # Exceeds 45 degC threshold
            "pattern_id": None,
        },
        {
            "id": "constellation_normal",
            "type": "normal",
            "expected_loss": 0.03,
            "signal": [0.02] * 1000,  # Nominal link operation
            "pattern_id": None,
        },
    ]

    # Query shared_anomalies for patterns where "starlink" in hooks
    try:
        patterns = get_patterns_for_hook("starlink")
    except Exception:
        patterns = []

    # Also include cross-domain patterns from Tesla comms
    cross_domain_config = get_cross_domain_config()
    for domain, source in cross_domain_config.get("accepts", {}).items():
        try:
            source_patterns = get_patterns_for_hook(source)
            for p in source_patterns:
                if p.physics_domain == domain and "starlink" in p.cross_domain_targets:
                    patterns.append(p)
        except Exception:
            pass

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
        description="Starlink satellite telemetry hook for QED v5.0",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # demo subcommand
    demo = subparsers.add_parser(
        "demo",
        help="Generate synthetic Starlink telemetry for a single channel",
    )
    demo.add_argument(
        "--channel",
        choices=list(_STARLINK_CHANNELS.keys()),
        default="link_drop_rate",
    )
    demo.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Duration of the synthetic signal in seconds",
    )
    demo.add_argument(
        "--inject-event",
        action="store_true",
        help="Inject a safety relevant event that exceeds the threshold",
    )
    demo.add_argument(
        "--json",
        action="store_true",
        help="Emit a single line JSON summary",
    )

    # from-csv subcommand
    csv_cmd = subparsers.add_parser(
        "from-csv",
        help="Process real Starlink telemetry from a CSV file",
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
        choices=list(_STARLINK_CHANNELS.keys()),
        required=True,
        help="Telemetry channel metadata to apply",
    )
    csv_cmd.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
        help="Override sample rate in Hz",
    )
    csv_cmd.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="Window length in seconds",
    )
    csv_cmd.add_argument(
        "--stride-sec",
        type=float,
        default=None,
        help="Stride between windows in seconds",
    )
    csv_cmd.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit JSONL, one record per window",
    )

    args = parser.parse_args()

    if args.subcommand == "demo":
        meta = _get_channel_meta(args.channel)
        duration_sec = float(args.duration_sec or meta["window_seconds"])
        sample_rate_hz = float(meta["sample_rate_hz"])
        safety_threshold = float(meta["safety_threshold"])
        baseline = float(meta["baseline"])

        raw = _make_demo_signal(
            args.channel,
            duration_sec=duration_sec,
            inject_event=args.inject_event,
            seed=42,
        )

        scaled = _normalize_segment(raw, baseline, safety_threshold)

        result = qed.qed(
            scaled,
            scenario="starlink_flow",
            bit_depth=12,
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
            f"Starlink: max={max_raw:.4f} / {safety_threshold:.4f} "
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
        baseline = float(meta["baseline"])

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

        window_index = 0
        for start in range(0, n_values - window_n + 1, stride_n):
            end = start + window_n
            segment = raw_values[start:end]
            if segment.size == 0:
                continue

            max_raw = float(np.max(np.abs(segment)))
            scaled = _normalize_segment(segment, baseline, safety_threshold)

            result = qed.qed(
                scaled,
                scenario="starlink_flow",
                bit_depth=12,
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
                f"  max_raw={max_raw:.4f} threshold={safety_threshold:.4f} "
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
