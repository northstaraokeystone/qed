"""
SpaceX hook for QED v5.0.

Maps SpaceX flight and test telemetry into QED space by scaling each channel so
its safety threshold lands at 10.0 internal units, with headroom up to about
±14.7. Provides:
  - 'demo' for synthetic, fault-injectable signals
  - 'from-csv' for windowed analysis of recorded telemetry
"""

from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Hook metadata for QED v6 edge lab integration
# -----------------------------------------------------------------------------
HOOK_NAME: str = "spacex_vibe"
COMPANY: str = "spacex"
STREAM_ID: str = "merlin_sensor"

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import qed

_SPACEX_CHANNELS: Dict[str, Dict[str, Any]] = {
    "chamber_pressure": {
        "pretty_name": "Raptor 3 chamber pressure (kPa)",
        "sample_rate_hz": 50_000.0,
        "window_seconds": 1.0,
        "safety_threshold": 30_000.0,  # kPa
        "dominant_freq_hz": 2_347.0,  # first acoustic mode
        "noise_fraction": 0.05,
        "baseline": 20_000.0,
    },
    "heat_shield_temp": {
        "pretty_name": "Starship heat shield panel temperature (degC)",
        "sample_rate_hz": 1_000.0,
        "window_seconds": 10.0,
        "safety_threshold": 1_485.0,  # degC
        "dominant_freq_hz": 0.5,  # thermal cycle
        "noise_fraction": 0.10,
        "baseline": 1_100.0,
    },
    "grid_fin_force": {
        "pretty_name": "Grid fin actuator force (kN)",
        "sample_rate_hz": 2_000.0,
        "window_seconds": 1.0,
        "safety_threshold": 1_020.0,  # kN
        "dominant_freq_hz": 8.5,  # fin natural frequency
        "noise_fraction": 0.05,
        "baseline": 0.0,
    },
    "ullage_pressure": {
        "pretty_name": "LOX tank ullage pressure (bar)",
        "sample_rate_hz": 500.0,
        "window_seconds": 10.0,
        "safety_threshold": 5.10,  # bar high side
        "dominant_freq_hz": 0.35,  # slosh
        "noise_fraction": 0.10,
        "baseline": 3.5,
    },
    "methane_valve_pos": {
        "pretty_name": "Raptor methane valve position (percent)",
        "sample_rate_hz": 1_000.0,
        "window_seconds": 1.0,
        "safety_threshold": 92.5,  # percent
        "dominant_freq_hz": 10.0,  # control update
        "noise_fraction": 0.05,
        "baseline": 80.0,
    },
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_channel_meta(channel: str) -> Dict[str, Any]:
    meta = _SPACEX_CHANNELS.get(channel)
    if meta is None:
        raise ValueError(f"unknown channel: {channel}")
    return meta


def _classify_window(
    max_raw: float,
    safety_threshold: float,
    recall: float,
) -> str:
    """
    Safety ladder:

      NOMINAL: recall >= 0.999 and amp < 0.8 * threshold
      WATCH:   recall >= 0.999 and 0.8 * threshold <= amp < 1.0 * threshold
      ALERT:   recall >= 0.999 and 1.0 * threshold <= amp < 1.2 * threshold
      CRITICAL: recall >= 0.95 and 1.2 * threshold <= amp < 1.5 * threshold
      ABORT:   recall < 0.95 or amp >= 1.5 * threshold

    Where amp is max_raw.
    """
    if safety_threshold <= 0 or max_raw <= 0:
        return "NO_SIGNAL"

    amp_frac = max_raw / safety_threshold

    if recall < 0.95 or amp_frac >= 1.5:
        return "ABORT"
    if recall >= 0.95 and amp_frac >= 1.2:
        return "CRITICAL"
    if recall >= 0.999 and amp_frac >= 1.0:
        return "ALERT"
    if recall >= 0.999 and amp_frac >= 0.8:
        return "WATCH"
    return "NOMINAL"


def _health_score(
    recall: float,
    max_raw: float,
    safety_threshold: float,
    ratio: float,
    H_bits: float,  # pylint: disable=unused-argument
) -> float:
    """
    Compressed health scalar in [0, 1]:

      0.4 * recall
      + 0.3 * (1 - amp_frac)
      + 0.2 * compression_quality
      + 0.1 * entropy_stability (stubbed to 1.0)

    Where amp_frac is clipped to [0, 1] and compression_quality maps ratio
    from [44, 65] into [0, 1].
    """
    if safety_threshold > 0:
        amp_frac = max_raw / safety_threshold
    else:
        amp_frac = 0.0

    amp_term = 1.0 - min(amp_frac, 1.0)
    compression_quality = _clamp((ratio - 44.0) / (65.0 - 44.0), 0.0, 1.0)
    entropy_stability = 1.0

    health = (
        0.4 * recall
        + 0.3 * amp_term
        + 0.2 * compression_quality
        + 0.1 * entropy_stability
    )
    return _clamp(health, 0.0, 1.0)


def _normalize_segment(
    segment: np.ndarray,
    baseline: float,
    safety_threshold: float,
) -> np.ndarray:
    """
    Map raw engineering units into QED units.

    safety_threshold is mapped to 10.0. Values are clipped to ±14.7 to leave
    headroom above the safety limit.
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
    Generate a synthetic SpaceX telemetry signal for the given channel.

    Uses sinusoidal carrier at the dominant frequency, plus Gaussian noise,
    with an optional central fault region that pushes above the safety
    threshold.
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

    # Keep nominal amplitude clearly below threshold taking baseline into account.
    if safety_threshold > baseline:
        amp_span = safety_threshold - baseline
    else:
        amp_span = safety_threshold
    amplitude = 0.7 * amp_span

    clean = baseline + amplitude * np.sin(2.0 * np.pi * dominant_freq_hz * t)
    noise_sigma = noise_fraction * amplitude
    noise = rng.normal(0.0, noise_sigma, size=n)
    raw = clean + noise

    if inject_event:
        # Push the middle half of the window over the safety threshold.
        start = n // 4
        end = 3 * n // 4
        raw[start:end] *= 1.3

    return raw.astype(np.float64)


def _read_csv_column(path: Path, column: str) -> np.ndarray:
    """
    Read a single numeric column from a CSV file with a header row.

    Skips rows that cannot be parsed as float. Raises ValueError if the
    column is not found or no numeric values are parsed.
    """
    values = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column '{column}' not found in CSV header")
        for row in reader:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                continue

    if not values:
        raise ValueError(f"no numeric values found for column '{column}'")

    return np.asarray(values, dtype=np.float64)


def validate_preflight(config: Dict[str, Any]) -> None:
    """
    Placeholder for pre flight Monte Carlo validation.

    This would run thousands of synthetic trajectories through QED to
    estimate detection probability and latency distributions. Kept as a
    stub so the hook stays local and lightweight.
    """
    raise NotImplementedError(
        f"pre flight validation not implemented in spacex hook: {config}"
    )


def analyze_postflight(config: Dict[str, Any]) -> None:
    """
    Placeholder for post flight batch analysis.

    This would stream mission telemetry from archival storage, window it
    through QED, and emit engineering reports. Not implemented here.
    """
    raise NotImplementedError(
        f"post flight analysis not implemented in spacex hook: {config}"
    )


def _run_demo(args: argparse.Namespace) -> None:
    meta = _get_channel_meta(args.channel)
    sample_rate_hz = float(meta["sample_rate_hz"])
    safety_threshold = float(meta["safety_threshold"])
    baseline = float(meta["baseline"])
    duration_sec = float(args.duration_sec or meta["window_seconds"])

    raw = _make_demo_signal(
        channel=args.channel,
        duration_sec=duration_sec,
        inject_event=bool(args.inject_event),
        seed=42,
    )

    scaled = _normalize_segment(raw, baseline, safety_threshold)

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

    max_raw = float(np.max(np.abs(raw)))
    amp_frac = max_raw / safety_threshold if safety_threshold > 0 else 0.0
    classification = _classify_window(max_raw, safety_threshold, recall)
    health = _health_score(recall, max_raw, safety_threshold, ratio, H_bits)

    print(
        f"Channel: {meta['pretty_name']} ({args.channel})\n"
        f"Samples: {len(raw)} at {sample_rate_hz:.1f} Hz ({duration_sec:.3f} s)\n"
        f"QED: ratio={ratio:.1f}, H_bits={H_bits:.0f}, "
        f"recall={recall:.4f}, savings_M={savings_M:.2f}\n"
        f"Signal: max={max_raw:.3f} / {safety_threshold:.3f} "
        f"(amp_frac={amp_frac:.3f})\n"
        f"Status: {classification}, health={health:.3f}, trace={trace}"
    )

    if args.json:
        payload = {
            "channel": args.channel,
            "pretty_name": meta["pretty_name"],
            "sample_rate_hz": sample_rate_hz,
            "n_samples": len(raw),
            "duration_sec": duration_sec,
            "safety_threshold": safety_threshold,
            "baseline": baseline,
            "max_raw": max_raw,
            "amp_frac": amp_frac,
            "ratio": ratio,
            "H_bits": H_bits,
            "recall": recall,
            "savings_M": savings_M,
            "classification": classification,
            "health": health,
            "trace": trace,
        }
        print(json.dumps(payload))


def _run_from_csv(args: argparse.Namespace) -> None:
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
    if stride_n <= 0:
        raise ValueError("stride must be positive")

    n_values = len(raw_values)
    window_index = 0

    for start in range(0, n_values - window_n + 1, stride_n):
        end = start + window_n
        segment = raw_values[start:end]

        max_raw = float(np.max(np.abs(segment)))
        amp_frac = max_raw / safety_threshold if safety_threshold > 0 else 0.0

        scaled = _normalize_segment(segment, baseline, safety_threshold)

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
        window_start_sec = start / sample_rate_hz

        print(
            f"window={window_index} start={window_start_sec:.3f}s len={len(segment)}\n"
            f"  ratio={ratio:.1f} H_bits={H_bits:.0f} recall={recall:.4f} "
            f"savings_M={savings_M:.2f}\n"
            f"  max_raw={max_raw:.3f} threshold={safety_threshold:.3f} "
            f"amp_frac={amp_frac:.3f} status={classification} "
            f"health={health:.3f}"
        )

        if args.jsonl:
            payload = {
                "window_index": window_index,
                "start_sec": window_start_sec,
                "channel": args.channel,
                "ratio": ratio,
                "H_bits": H_bits,
                "recall": recall,
                "savings_M": savings_M,
                "max_raw": max_raw,
                "amp_frac": amp_frac,
                "safety_threshold": safety_threshold,
                "baseline": baseline,
                "classification": classification,
                "health": health,
                "trace": trace,
            }
            print(json.dumps(payload))

        window_index += 1


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Return edge lab test scenarios for SpaceX telemetry.

    Each scenario is a dict with:
        - id: unique scenario identifier
        - type: scenario type (spike, step, drift, normal)
        - expected_loss: expected loss threshold (>0.1 for high-loss scenarios)
        - signal: list of float values representing the test signal

    Returns 5 scenarios including high-loss edge cases for ROI validation.
    """
    import numpy as np_local

    return [
        {
            "id": "vibe_exceed",
            "type": "spike",
            "expected_loss": 0.16,
            "signal": [0.6] * 1000,  # Exceeds 0.5mm vibe threshold
        },
        {
            "id": "thrust_anomaly",
            "type": "step",
            "expected_loss": 0.14,
            "signal": [0.0] * 500 + [0.4] * 500,  # Sudden thrust change
        },
        {
            "id": "orbit_drift",
            "type": "drift",
            "expected_loss": 0.11,
            "signal": list(np_local.linspace(0, 0.5, 1000)),  # Gradual drift to threshold
        },
        {
            "id": "chamber_spike",
            "type": "spike",
            "expected_loss": 0.19,
            "signal": [21.0] * 1000,  # Exceeds 20.0 amplitude bound
        },
        {
            "id": "launch_normal",
            "type": "normal",
            "expected_loss": 0.04,
            "signal": [15.0] * 1000,  # Nominal flight telemetry
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpaceX telemetry hook for QED v5.0",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    demo = subparsers.add_parser("demo", help="Generate synthetic SpaceX signals")
    demo.add_argument(
        "--channel", choices=list(_SPACEX_CHANNELS.keys()), default="chamber_pressure"
    )
    demo.add_argument("--duration-sec", type=float, default=None)
    demo.add_argument("--inject-event", action="store_true")
    demo.add_argument("--json", action="store_true")

    csv_cmd = subparsers.add_parser("from-csv", help="Process CSV telemetry data")
    csv_cmd.add_argument("--file", type=Path, required=True)
    csv_cmd.add_argument("--column", required=True)
    csv_cmd.add_argument(
        "--channel", choices=list(_SPACEX_CHANNELS.keys()), required=True
    )
    csv_cmd.add_argument("--sample-rate-hz", type=float, default=None)
    csv_cmd.add_argument("--window-sec", type=float, default=None)
    csv_cmd.add_argument("--stride-sec", type=float, default=None)
    csv_cmd.add_argument("--jsonl", action="store_true")

    args = parser.parse_args()

    if args.subcommand == "demo":
        _run_demo(args)
    elif args.subcommand == "from-csv":
        _run_from_csv(args)
    else:
        raise ValueError(f"unknown subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
