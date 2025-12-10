"""
xAI hook for QED v5.0.

Normalizes xAI LLM telemetry signals into QED space by scaling each channel so the
safety or critical threshold maps to about 10.0 in QED space (QED's recall threshold).
Exposes CLI: 'demo' for synthetic signals and 'from-jsonl' for real telemetry windowing.
"""

from typing import Any, Dict, List, Optional

from shared_anomalies import get_patterns_for_hook

# -----------------------------------------------------------------------------
# Hook metadata for QED v6 edge lab integration
# -----------------------------------------------------------------------------
HOOK_NAME: str = "xai_eval"
COMPANY: str = "xai"
STREAM_ID: str = "grok_metrics"

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import qed

# Minimum samples per window for xAI telemetry.
_MIN_WINDOW_SAMPLES = 50

_XAI_CHANNELS: Dict[str, Dict[str, Any]] = {
    "kv_phase_density": {
        "pretty_name": "KV-cache phase density",
        "sample_rate_hz": 50.0,
        "window_seconds": 2.0,
        "safety_threshold": 1.0,  # treat full phase excursion as unit scale
        "dominant_freq_hz": 0.5,
        "noise_fraction": 0.05,
        "baseline": 0.32,  # critical point
    },
    "logit_margin": {
        "pretty_name": "Top-k logit margin",
        "sample_rate_hz": 50.0,
        "window_seconds": 1.0,
        "safety_threshold": 1.2,  # critical margin
        "dominant_freq_hz": 1.0,
        "noise_fraction": 0.10,
        "baseline": 4.0,  # rolling median baseline
    },
    "safety_head_entropy": {
        "pretty_name": "Safety head entropy (bits)",
        "sample_rate_hz": 50.0,
        "window_seconds": 1.0,
        "safety_threshold": 1.0,  # squeeze threshold
        "dominant_freq_hz": 1.0,
        "noise_fraction": 0.05,
        "baseline": 5.0,  # normal 4-7 bits
    },
    "hidden_state_energy": {
        "pretty_name": "Hidden state energy (norm)",
        "sample_rate_hz": 50.0,
        "window_seconds": 1.0,
        "safety_threshold": 2.7,  # 2.7x baseline
        "dominant_freq_hz": 0.5,
        "noise_fraction": 0.05,
        "baseline": 1.0,
    },
    "gpu_mem_bandwidth": {
        "pretty_name": "GPU memory bandwidth (GB/s)",
        "sample_rate_hz": 10000.0,
        "window_seconds": 0.1,
        "safety_threshold": 900.0,  # turbulence onset
        "dominant_freq_hz": 50.0,
        "noise_fraction": 0.05,
        "baseline": 600.0,
    },
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_channel_meta(channel: str) -> Dict[str, Any]:
    meta = _XAI_CHANNELS.get(channel)
    if meta is None:
        raise ValueError(f"unknown channel: {channel}")
    return meta


def _classify_window(
    max_raw: float,
    safety_threshold: float,
    recall: float,
) -> str:
    """
    Safety classification in raw units, aligned with other hooks.

    Levels:
      NOMINAL, WATCH, ALERT, CRITICAL, ABORT.
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
    H_bits: float,
) -> float:
    """
    Aggregate health score in [0, 1] mixing recall, amplitude margin,
    compression quality, and entropy stability.
    """
    if safety_threshold <= 0.0:
        amp_frac = 0.0
    else:
        amp_frac = max_raw / safety_threshold

    # Compression quality: 44:1 to 65:1 is considered the sweet spot.
    compression_quality = _clamp((ratio - 44.0) / (65.0 - 44.0), 0.0, 1.0)

    # Simple entropy stability proxy: penalize very low or very high H_bits.
    # Assume 2000-7200 bits is a good band for telemetry streams.
    if H_bits <= 0.0:
        entropy_stability = 0.0
    else:
        entropy_stability = _clamp((H_bits - 2000.0) / (7200.0 - 2000.0), 0.0, 1.0)

    health = (
        0.35 * _clamp(recall, 0.0, 1.0)
        + 0.30 * (1.0 - min(amp_frac / 1.5, 1.0))
        + 0.20 * compression_quality
        + 0.15 * entropy_stability
    )
    return _clamp(health, 0.0, 1.0)


def _scale_kv_phase_density(raw: np.ndarray) -> np.ndarray:
    """
    Piecewise critical scaling for KV phase density.

    ρ in [0, 1], with:
      - critical point ρ_c = 0.32
      - glass threshold ρ_g = 0.05
      - plasma threshold ρ_p = 0.85
    """
    rho = np.clip(raw.astype(np.float64), 0.0, 1.0)
    rho_c = 0.32
    rho_g = 0.05
    rho_p = 0.85

    q = np.zeros_like(rho, dtype=np.float64)

    # Supercritical region toward plasma.
    mask_high = rho >= rho_c
    if np.any(mask_high):
        num = rho[mask_high] - rho_c
        den = rho_p - rho_c
        frac = np.clip(num / den, 0.0, 1.0)
        q[mask_high] = 10.0 * np.power(frac, 0.326)

    # Subcritical region toward glass.
    mask_low = rho < rho_c
    if np.any(mask_low):
        num = rho_c - rho[mask_low]
        den = rho_c - rho_g
        frac = np.clip(num / den, 0.0, 1.0)
        q[mask_low] = -10.0 * np.power(frac, 0.11)

    return np.clip(q, -14.7, 14.7)


def _scale_logit_margin(raw: np.ndarray, baseline: float) -> np.ndarray:
    """
    Critical scaling for logit margin.

    m_c = 1.2 is the critical margin.
    baseline m_0 is around 4.0 (rolling median).
    """
    m = raw.astype(np.float64)
    m_c = 1.2
    m_0 = baseline

    q = np.zeros_like(m, dtype=np.float64)

    # Near or below critical margin: power law divergence toward negative.
    mask_low = m <= m_c
    if np.any(mask_low):
        safe_m = np.clip(m[mask_low], 1e-3, None)
        q[mask_low] = -10.0 * np.power(m_c / safe_m, 2.5)

    # Above critical: mean field scaling.
    mask_high = m > m_c
    if np.any(mask_high):
        num = m[mask_high] - m_c
        den = max(m_0 - m_c, 1e-6)
        frac = np.clip(num / den, 0.0, 1.0)
        q[mask_high] = 10.0 * np.sqrt(frac)

    return np.clip(q, -14.7, 14.7)


def _scale_safety_head_entropy(raw: np.ndarray, baseline: float) -> np.ndarray:
    """
    Scale safety head entropy.

    S < 1.0 bits maps toward strong negative amplitudes.
    Normal 4-7 bits is near zero. Very high entropy drifts mildly positive.
    """
    S = raw.astype(np.float64)
    danger_thresh = 1.0
    normal_hi = 7.0
    max_bits = 11.0

    q = np.zeros_like(S, dtype=np.float64)

    mask_danger = S < danger_thresh
    if np.any(mask_danger):
        frac = np.clip((danger_thresh - S[mask_danger]) / danger_thresh, 0.0, 1.0)
        q[mask_danger] = -10.0 * frac

    mask_mid = (S >= danger_thresh) & (S <= normal_hi)
    if np.any(mask_mid):
        span = max(normal_hi - danger_thresh, 1e-6)
        q[mask_mid] = 5.0 * (S[mask_mid] - baseline) / span

    mask_high = S > normal_hi
    if np.any(mask_high):
        span = max(max_bits - normal_hi, 1e-6)
        frac = np.clip((S[mask_high] - normal_hi) / span, 0.0, 1.0)
        q[mask_high] = 5.0 * frac

    return np.clip(q, -14.7, 14.7)


def _scale_hidden_state_energy(
    raw: np.ndarray,
    baseline: float,
    safety_threshold: float,
) -> np.ndarray:
    """
    Scale hidden state energy.

    Baseline around 1.0. Safety threshold at about 2.7x baseline maps to 10.
    """
    E = raw.astype(np.float64)
    base = baseline
    thr = safety_threshold

    q = np.zeros_like(E, dtype=np.float64)

    # Below baseline: small negative deviations.
    mask_low = E <= base
    if np.any(mask_low):
        span = max(base, 1e-6)
        q[mask_low] = -5.0 * (base - E[mask_low]) / span

    # Between baseline and threshold.
    mask_mid = (E > base) & (E <= thr)
    if np.any(mask_mid):
        span = max(thr - base, 1e-6)
        q[mask_mid] = 10.0 * (E[mask_mid] - base) / span

    # Above threshold: push toward saturation.
    mask_high = E > thr
    if np.any(mask_high):
        span = max(thr, 1e-6)
        frac = np.clip((E[mask_high] - thr) / span, 0.0, 1.0)
        q[mask_high] = 10.0 + 4.7 * frac

    return np.clip(q, -14.7, 14.7)


def _scale_gpu_mem_bandwidth(
    raw: np.ndarray,
    baseline: float,
    safety_threshold: float,
) -> np.ndarray:
    """
    Scale GPU memory bandwidth.

    - Turbulence onset around safety_threshold (~900 GB/s) maps to +10.
    - Stall around 350 GB/s maps to about -10.
    - Baseline around 600 GB/s is near zero.
    """
    b = raw.astype(np.float64)
    base = baseline
    thr_high = safety_threshold
    stall_low = 350.0

    q = np.zeros_like(b, dtype=np.float64)

    # Stall region.
    mask_stall = b < stall_low
    if np.any(mask_stall):
        span = max(stall_low, 1e-6)
        frac = np.clip((stall_low - b[mask_stall]) / span, 0.0, 1.0)
        q[mask_stall] = -10.0 * frac

    # Nominal region between stall and turbulence threshold.
    mask_mid = (b >= stall_low) & (b <= thr_high)
    if np.any(mask_mid):
        if thr_high > base:
            q[mask_mid] = 10.0 * (b[mask_mid] - base) / (thr_high - base)
        else:
            q[mask_mid] = 0.0

    # Turbulent region above threshold.
    mask_high = b > thr_high
    if np.any(mask_high):
        span = max(thr_high, 1e-6)
        frac = np.clip((b[mask_high] - thr_high) / span, 0.0, 1.0)
        q[mask_high] = 10.0 + 4.7 * frac

    return np.clip(q, -14.7, 14.7)


def _normalize_window(window: np.ndarray, channel: str) -> np.ndarray:
    """
    Dispatch to channel specific scaling and return QED ready samples.
    """
    meta = _get_channel_meta(channel)
    if channel == "kv_phase_density":
        return _scale_kv_phase_density(window)
    if channel == "logit_margin":
        return _scale_logit_margin(window, baseline=float(meta["baseline"]))
    if channel == "safety_head_entropy":
        return _scale_safety_head_entropy(window, baseline=float(meta["baseline"]))
    if channel == "hidden_state_energy":
        return _scale_hidden_state_energy(
            window,
            baseline=float(meta["baseline"]),
            safety_threshold=float(meta["safety_threshold"]),
        )
    if channel == "gpu_mem_bandwidth":
        return _scale_gpu_mem_bandwidth(
            window,
            baseline=float(meta["baseline"]),
            safety_threshold=float(meta["safety_threshold"]),
        )
    # Fallback linear scaling if a new channel is added without custom logic.
    safety = float(meta["safety_threshold"])
    baseline = float(meta["baseline"])
    if safety <= 0.0:
        return np.zeros_like(window, dtype=np.float64)
    scaled = (window.astype(np.float64) - baseline) * (10.0 / max(safety, 1e-6))
    return np.clip(scaled, -14.7, 14.7)


def _make_demo_signal(
    channel: str,
    duration_sec: float,
    inject_event: bool,
    seed: int,
) -> np.ndarray:
    """
    Generate synthetic xAI telemetry for a given channel.

    Uses a sine around baseline with Gaussian noise, then injects
    a critical event segment if requested.
    """
    meta = _get_channel_meta(channel)
    sample_rate_hz = float(meta["sample_rate_hz"])
    dominant_freq_hz = float(meta["dominant_freq_hz"])
    noise_fraction = float(meta["noise_fraction"])
    baseline = float(meta["baseline"])
    safety_threshold = float(meta["safety_threshold"])
    window_seconds = float(meta["window_seconds"])

    n = int(sample_rate_hz * duration_sec)

    # Require at least one full window worth of samples for this channel.
    min_samples = max(
        int(sample_rate_hz * window_seconds),
        _MIN_WINDOW_SAMPLES,
    )
    if n < min_samples:
        raise ValueError(
            f"duration {duration_sec:.3f}s too short for channel '{channel}', "
            f"need at least {min_samples} samples "
            f"({min_samples / sample_rate_hz:.3f}s)",
        )

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    # Nominal amplitude around 60 percent of the critical swing.
    A = 0.6 * safety_threshold
    clean = baseline + A * np.sin(2.0 * np.pi * dominant_freq_hz * t)
    noise_sigma = noise_fraction * A
    noise = rng.normal(0.0, noise_sigma, size=n)
    raw = clean + noise

    if inject_event:
        start = n // 3
        end = 2 * n // 3
        segment = raw[start:end].copy()

        if channel == "kv_phase_density":
            # Drive into glass then plasma.
            half = segment.shape[0] // 2
            segment[:half] = 0.02
            segment[half:] = 0.9
        elif channel == "logit_margin":
            # Collapse margin near zero.
            segment[:] = np.clip(segment, 0.05, 0.8)
        elif channel == "safety_head_entropy":
            # Squeeze below 1 bit.
            segment[:] = 0.5
        elif channel == "hidden_state_energy":
            # Push energy above safety threshold.
            segment[:] = 3.2 * baseline
        elif channel == "gpu_mem_bandwidth":
            # Push bandwidth well above turbulence threshold.
            segment[:] = safety_threshold + 200.0

        raw[start:end] = segment

    return raw.astype(np.float64)


def _window_signal(
    signal: np.ndarray,
    window_len: int,
    stride: int,
) -> List[np.ndarray]:
    """
    Simple sliding window with given length and stride.
    """
    n = signal.shape[0]
    windows: List[np.ndarray] = []
    if window_len <= 0 or stride <= 0:
        return windows
    for start in range(0, max(n - window_len + 1, 0), stride):
        end = start + window_len
        if end > n:
            break
        windows.append(signal[start:end])
    return windows


def _read_jsonl_field(path: Path, field: str) -> np.ndarray:
    """
    Read a numeric field from a JSONL file.

    Skips lines where the field is missing or cannot be parsed as float.
    Raises ValueError if no usable values are found.
    """
    values: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if field not in obj:
                continue
            try:
                values.append(float(obj[field]))
            except (TypeError, ValueError):
                continue

    if not values:
        raise ValueError(f"no numeric values found for field '{field}' in {path}")
    return np.asarray(values, dtype=np.float64)


def _process_windows(
    channel: str,
    windows: List[np.ndarray],
    sample_rate_hz: float,
) -> List[Dict[str, Any]]:
    """
    Run QED on each window and attach classification and health score.
    """
    meta = _get_channel_meta(channel)
    safety_threshold = float(meta["safety_threshold"])

    results: List[Dict[str, Any]] = []

    for idx, w in enumerate(windows):
        if w.size == 0:
            continue

        max_raw = float(np.max(np.abs(w)))
        # Handle degenerate windows with no variation.
        if np.allclose(w, w[0]):
            recall = 1.0
            ratio = 0.0
            H_bits = 0.0
            classification = "NO_SIGNAL"
            health = 1.0
            results.append(
                {
                    "idx": idx,
                    "channel": channel,
                    "max_raw": max_raw,
                    "classification": classification,
                    "health_score": health,
                    "ratio": ratio,
                    "recall": recall,
                    "H_bits": H_bits,
                    "no_signal": True,
                }
            )
            continue

        normalized = _normalize_window(w, channel)
        # QED core call; assumes QED accepts a 1-D float array and sample rate.
        qed_result: Dict[str, Any] = qed.qed(
            normalized,
            sample_rate_hz=sample_rate_hz,
            bit_depth=16,
        )

        ratio = float(qed_result.get("ratio", 0.0))
        recall = float(qed_result.get("recall", 1.0))
        H_bits = float(qed_result.get("H_bits", 0.0))

        classification = _classify_window(max_raw, safety_threshold, recall)
        health = _health_score(recall, max_raw, safety_threshold, ratio, H_bits)

        result_record: Dict[str, Any] = {
            "idx": idx,
            "channel": channel,
            "max_raw": max_raw,
            "classification": classification,
            "health_score": health,
            "ratio": ratio,
            "recall": recall,
            "H_bits": H_bits,
        }
        # Merge raw QED fields without overwriting ours.
        for k, v in qed_result.items():
            if k not in result_record:
                result_record[k] = v
        results.append(result_record)

    return results


def run_demo(args: argparse.Namespace) -> None:
    channel = args.channel
    duration_sec = float(args.duration_sec)
    inject_event = bool(args.inject_event)
    seed = int(args.seed)

    meta = _get_channel_meta(channel)
    sample_rate_hz = float(meta["sample_rate_hz"])
    window_seconds = float(meta["window_seconds"])
    window_len = int(sample_rate_hz * window_seconds)

    if window_len < _MIN_WINDOW_SAMPLES:
        raise ValueError(
            f"window length {window_len} too short for channel '{channel}', "
            f"need at least {_MIN_WINDOW_SAMPLES} samples",
        )

    stride = max(window_len // 2, 1)

    signal = _make_demo_signal(channel, duration_sec, inject_event, seed)
    windows = _window_signal(signal, window_len, stride)
    results = _process_windows(channel, windows, sample_rate_hz)

    out: Optional[Path] = Path(args.output) if args.output else None
    if out is None:
        sink = sys.stdout
        close_sink = False
    else:
        sink = out.open("w", encoding="utf-8")
        close_sink = True

    try:
        for rec in results:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
    finally:
        if close_sink:
            sink.close()


def run_from_jsonl(args: argparse.Namespace) -> None:
    channel = args.channel
    field = args.field
    input_path = Path(args.input)

    meta = _get_channel_meta(channel)

    sample_rate_hz = (
        float(args.sample_rate_hz)
        if args.sample_rate_hz
        else float(meta["sample_rate_hz"])
    )
    window_seconds = (
        float(args.window_seconds)
        if args.window_seconds
        else float(meta["window_seconds"])
    )

    window_len = int(sample_rate_hz * window_seconds)
    if window_len < _MIN_WINDOW_SAMPLES:
        raise ValueError(
            f"window length {window_len} too short for channel '{channel}', "
            f"need at least {_MIN_WINDOW_SAMPLES} samples",
        )

    stride = max(window_len // 2, 1)

    signal = _read_jsonl_field(input_path, field)
    windows = _window_signal(signal, window_len, stride)
    results = _process_windows(channel, windows, sample_rate_hz)

    out: Optional[Path] = Path(args.output) if args.output else None
    if out is None:
        sink = sys.stdout
        close_sink = False
    else:
        sink = out.open("w", encoding="utf-8")
        close_sink = True

    try:
        for rec in results:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
    finally:
        if close_sink:
            sink.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xAI hook for QED v5.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run synthetic demo telemetry")
    demo.add_argument(
        "--channel",
        required=True,
        choices=sorted(_XAI_CHANNELS.keys()),
        help="telemetry channel to simulate",
    )
    demo.add_argument(
        "--duration-sec",
        type=float,
        default=4.0,
        help="duration of synthetic signal in seconds",
    )
    demo.add_argument(
        "--inject-event",
        action="store_true",
        help="inject a critical event into the synthetic signal",
    )
    demo.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for noise generation",
    )
    demo.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSONL file (default stdout)",
    )
    demo.set_defaults(func=run_demo)

    from_jsonl = subparsers.add_parser(
        "from-jsonl",
        help="ingest xAI telemetry from JSONL",
    )
    from_jsonl.add_argument(
        "--channel",
        required=True,
        choices=sorted(_XAI_CHANNELS.keys()),
        help="telemetry channel to process",
    )
    from_jsonl.add_argument(
        "--input",
        required=True,
        help="path to JSONL telemetry file",
    )
    from_jsonl.add_argument(
        "--field",
        required=True,
        help="numeric field name in each JSON object",
    )
    from_jsonl.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
        help="override sample rate in Hz (default from channel metadata)",
    )
    from_jsonl.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="override window length in seconds (default from channel metadata)",
    )
    from_jsonl.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSONL file (default stdout)",
    )
    from_jsonl.set_defaults(func=run_from_jsonl)

    return parser


def get_cross_domain_config() -> Dict[str, Any]:
    """
    Return cross-domain integration configuration for xAI.

    xAI has no cross-domain mappings (eval metrics are unique).
    """
    return {
        "exports": {},
        "accepts": {},
    }


def get_deployment_config() -> Dict[str, Any]:
    """Return xAI-specific QEDConfig defaults for AI inference telemetry."""
    return {
        "hook": "xai",
        "recall_floor": 0.995,
        "max_fp_rate": 0.02,
        "slo_latency_ms": 500,
        "slo_breach_budget": 0.01,
        "compression_target": 30.0,
        "enabled_patterns": ["PAT_INFERENCE_*", "PAT_TRAINING_*", "PAT_GPU_*", "PAT_MEMORY_*"],
        "regulatory_flags": {"SOC2": True, "GDPR": True},
        "safety_critical": False,
    }


def get_hardware_profile() -> Dict[str, Any]:
    """Return xAI hardware identifiers for mesh_view clustering."""
    return {
        "platform": "gpu_cluster",
        "compute_class": "datacenter",
        "connectivity": "infiniband",
        "storage_type": "distributed_nvme",
        "real_time": False,
        "safety_critical": False,
    }


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Return edge lab test scenarios for xAI LLM telemetry.

    Each scenario is a dict with:
        - id: unique scenario identifier
        - type: scenario type (spike, step, drift, normal)
        - expected_loss: expected loss threshold (>0.1 for high-loss scenarios)
        - signal: list of float values representing the test signal
        - pattern_id: optional pattern ID from shared_anomalies (None for legacy)

    Returns hand-crafted scenarios plus any patterns from shared_anomalies.
    No cross-domain patterns (eval metrics are unique).
    """
    # Hand-crafted legacy scenarios (pattern_id=None)
    legacy_scenarios = [
        {
            "id": "logit_spike",
            "type": "spike",
            "expected_loss": 0.18,
            "signal": [1e7] * 1000,  # Logit spike exceeds 1e6 threshold
            "pattern_id": None,
        },
        {
            "id": "recall_drop",
            "type": "step",
            "expected_loss": 0.16,
            "signal": [0.99] * 500 + [0.90] * 500,  # Recall drops below 0.95
            "pattern_id": None,
        },
        {
            "id": "entropy_squeeze",
            "type": "drift",
            "expected_loss": 0.13,
            "signal": [float(5.0 - i * 0.004) for i in range(1000)],  # Entropy squeeze
            "pattern_id": None,
        },
        {
            "id": "kv_phase_drift",
            "type": "drift",
            "expected_loss": 0.11,
            "signal": [float(0.32 + i * 0.0005) for i in range(1000)],  # Phase density drift
            "pattern_id": None,
        },
        {
            "id": "inference_normal",
            "type": "normal",
            "expected_loss": 0.04,
            "signal": [5.0] * 1000,  # Nominal inference metrics
            "pattern_id": None,
        },
    ]

    # Query shared_anomalies for patterns where "xai" in hooks
    try:
        patterns = get_patterns_for_hook("xai")
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
