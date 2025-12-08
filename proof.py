from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

import qed


def _make_signal(
    n: int,
    sample_rate_hz: float,
    amplitude: float,
    frequency_hz: float,
    noise_sigma: float,
    offset: float,
    phase_rad: float,
    seed: int,
) -> np.ndarray:
    """
    Build a synthetic 1D signal:
      signal(t) = A * sin(2*pi*f*t + phase) + offset + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate_hz
    clean = amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad) + offset
    noise = rng.normal(0.0, noise_sigma, n)
    return clean + noise


def _count_events(signal: np.ndarray, threshold: float) -> int:
    """Count samples above a safety threshold."""
    return int((signal > threshold).sum())


def _normalized_rms_error(raw: np.ndarray, recon: np.ndarray) -> float:
    """Compute normalized RMS error between raw and reconstructed signals."""
    num = np.sqrt(np.mean((raw - recon) ** 2))
    den = np.sqrt(np.mean(raw**2))
    if den == 0.0:
        return 0.0
    return float(num / den)


def _deterministic_check(signal: np.ndarray, sample_rate_hz: float) -> bool:
    """Return True if qed() is deterministic for this signal (v5 keys only)."""
    out1 = qed.qed(
        signal, scenario="tesla_fsd", bit_depth=12, sample_rate_hz=sample_rate_hz
    )
    out2 = qed.qed(
        signal, scenario="tesla_fsd", bit_depth=12, sample_rate_hz=sample_rate_hz
    )
    # Compare only v5 keys (ratio, H_bits, recall, savings_M, trace content)
    # v6 receipt has timestamp/uuid that will differ
    v5_keys = ["ratio", "H_bits", "recall", "savings_M"]
    for key in v5_keys:
        if out1[key] != out2[key]:
            return False
    # Trace should be deterministic except for any timestamp
    if out1["trace"].split()[0:3] != out2["trace"].split()[0:3]:
        return False
    return True


def run_proof(seed: int = 42424242) -> None:
    gates: Dict[str, bool] = {}
    failed: List[str] = []

    # Signal A - Tesla steering torque (primary validation)
    sample_rate_a = 1000.0
    signal_a = _make_signal(
        n=1000,
        sample_rate_hz=sample_rate_a,
        amplitude=12.347,
        frequency_hz=40.0,
        noise_sigma=0.147,
        offset=2.1,
        phase_rad=np.pi / 4.0,
        seed=seed,
    )

    t0 = time.perf_counter()
    out_a = qed.qed(
        signal_a,
        scenario="tesla_fsd",
        bit_depth=12,
        sample_rate_hz=sample_rate_a,
    )
    latency_a_ms = (time.perf_counter() - t0) * 1000.0

    ratio_a = float(out_a["ratio"])
    H_bits_a = float(out_a["H_bits"])
    recall_a = float(out_a["recall"])
    savings_m_a = float(out_a["savings_M"])
    trace_a = str(out_a["trace"])

    # Reconstruction from fitted parameters
    A_a, f_a, phi_a, c_a = qed._fit_dominant_sine(signal_a, sample_rate_a)
    t_a = np.arange(signal_a.size) / sample_rate_a
    recon_a = A_a * np.sin(2.0 * np.pi * f_a * t_a + phi_a) + c_a
    nrmse_a = _normalized_rms_error(signal_a, recon_a)
    events_a = _count_events(signal_a, threshold=10.0)

    # G1 - Compression ratio band (Signal A)
    gates["G1_ratio"] = 57.0 <= ratio_a <= 63.0

    # G2 - Shannon information band (Signal A)
    gates["G2_entropy"] = 7150.0 <= H_bits_a <= 7250.0

    # G3a - Safety recall with events (Signal A)
    gates["G3_recall_with_events"] = (events_a >= 50) and (recall_a >= 0.9985)

    # G4 - ROI band (Signal A)
    gates["G4_roi"] = (abs(ratio_a - 60.0) < 1.0) and (37.8 <= savings_m_a <= 38.2)

    # G6 - Reconstruction fidelity (Signal A)
    gates["G6_reconstruction"] = nrmse_a <= 0.05

    # G7 - Determinism (Signal A)
    gates["G7_determinism"] = _deterministic_check(
        signal_a, sample_rate_hz=sample_rate_a
    )

    # G8 - Latency (Signal A, in milliseconds)
    gates["G8_latency_ms"] = latency_a_ms <= 50.0

    # Signal B - Boring vibration (bound should pass)
    sample_rate_b = 2048.0
    signal_b = _make_signal(
        n=1024,
        sample_rate_hz=sample_rate_b,
        amplitude=14.697,
        frequency_hz=927.4,
        noise_sigma=0.01,
        offset=0.0,
        phase_rad=0.0,
        seed=seed + 1,
    )
    try:
        _ = qed.qed(
            signal_b,
            scenario="tesla_fsd",
            bit_depth=12,
            sample_rate_hz=sample_rate_b,
        )
        gates["G5_bound_pass"] = True
    except ValueError:
        gates["G5_bound_pass"] = False

    # Signal C - Neuralink style low amplitude (no events)
    sample_rate_c = 10_000.0
    signal_c = _make_signal(
        n=1000,
        sample_rate_hz=sample_rate_c,
        amplitude=1.0,
        frequency_hz=250.0,
        noise_sigma=0.2,
        offset=8.0,
        phase_rad=0.0,
        seed=seed + 2,
    )
    out_c = qed.qed(
        signal_c,
        scenario="tesla_fsd",
        bit_depth=12,
        sample_rate_hz=sample_rate_c,
    )
    recall_c = float(out_c["recall"])
    events_c = _count_events(signal_c, threshold=10.0)
    gates["G3_recall_no_events"] = (events_c == 0) and (recall_c == 1.0)

    # Signal D - SpaceX thrust oscillation (bound should fail)
    sample_rate_d = 1000.0
    signal_d = _make_signal(
        n=1000,
        sample_rate_hz=sample_rate_d,
        amplitude=14.703,
        frequency_hz=55.0,
        noise_sigma=0.01,
        offset=0.0,
        phase_rad=0.0,
        seed=seed + 3,
    )
    try:
        _ = qed.qed(
            signal_d,
            scenario="tesla_fsd",
            bit_depth=12,
            sample_rate_hz=sample_rate_d,
        )
        gates["G5_bound_fail"] = False
    except ValueError as exc:
        gates["G5_bound_fail"] = "amplitude" in str(exc).lower()

    # Evaluate gates
    for name, ok in gates.items():
        if not ok:
            failed.append(name)

    if failed:
        raise AssertionError(f"Failed gates: {failed}")

    # Success summary
    print("QED v5 proof gates passed")
    print(
        "Signal A: "
        f"ratio={ratio_a:.1f}, "
        f"H_bits={H_bits_a:.0f}, "
        f"recall={recall_a:.4f}, "
        f"savings_M={savings_m_a:.2f}, "
        f"nrmse={nrmse_a:.4f}, "
        f"latency_ms={latency_a_ms:.2f}, "
        f"trace={trace_a}"
    )


if __name__ == "__main__":
    run_proof()
