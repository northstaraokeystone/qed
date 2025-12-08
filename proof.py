"""
QED v6/v7 Proof CLI Harness

CLI tool for validating QED telemetry compression and safety guarantees.
Provides subcommands for:
  - replay: Load edge_lab_sample.jsonl, run qed per scenario, collect metrics
  - sympy_suite: Get constraints per hook, verify, log violations
  - summarize: Output hits/misses/violations/ROI to JSON
  - gates: Run legacy v5 gate checks (synthetic signals)

v7 subcommands:
  - run-sims: Run pattern simulations via edge_lab_v2
  - recall-floor: Compute Clopper-Pearson exact recall lower bound
  - pattern-report: Display pattern library with sorting/filtering
  - clarity-audit: Process receipts through ClarityClean adapter

What to prove:
  - Recall >= 99.67% (95% CI on 900 anomalies)
  - Precision > 95%
  - ROI $38M (fleet calc)
  - Violations = 0 on normals
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


import qed
import sympy_constraints

# v7 imports
from scipy.stats import beta
from edge_lab_v2 import run_pattern_sims
from shared_anomalies import load_library
from clarity_clean_adapter import process_receipts

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# --- KPI Thresholds ---
KPI_RECALL_THRESHOLD = 0.9967  # 99.67% recall CI
KPI_PRECISION_THRESHOLD = 0.95  # 95% precision
KPI_ROI_TARGET_M = 38.0  # $38M target ROI
KPI_ROI_TOLERANCE_M = 0.5  # +/- $0.5M tolerance
KPI_LATENCY_MS = 50.0  # Max latency per window


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
    v5_keys = ["ratio", "H_bits", "recall", "savings_M"]
    for key in v5_keys:
        if out1[key] != out2[key]:
            return False
    if out1["trace"].split()[0:3] != out2["trace"].split()[0:3]:
        return False
    return True


# --- Replay Subcommand ---


def replay(
    jsonl_path: str,
    scenario: str = "tesla_fsd",
    bit_depth: int = 12,
    sample_rate_hz: float = 1000.0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Replay scenarios from a JSONL file through qed.py.

    Each line in the JSONL file should have:
      - "signal": list of float values OR
      - "params": dict with keys to generate synthetic signal

    Returns list of result dicts with qed output + metadata.
    """
    results: List[Dict[str, Any]] = []
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    with path.open("r") as f:
        lines = f.readlines()

    for idx, line in enumerate(tqdm(lines, desc="Replaying scenarios")):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"Skipping line {idx}: JSON decode error: {e}")
            continue

        if "signal" in data:
            signal = np.array(data["signal"], dtype=np.float64)
        elif "params" in data:
            params = data["params"]
            signal = _make_signal(
                n=params.get("n", 1000),
                sample_rate_hz=params.get("sample_rate_hz", sample_rate_hz),
                amplitude=params.get("amplitude", 12.0),
                frequency_hz=params.get("frequency_hz", 40.0),
                noise_sigma=params.get("noise_sigma", 0.1),
                offset=params.get("offset", 2.0),
                phase_rad=params.get("phase_rad", 0.0),
                seed=params.get("seed", idx),
            )
        else:
            if verbose:
                print(f"Skipping line {idx}: no 'signal' or 'params' key")
            continue

        line_scenario = data.get("scenario", scenario)
        line_bit_depth = data.get("bit_depth", bit_depth)
        line_sample_rate = data.get("sample_rate_hz", sample_rate_hz)
        expected_label = data.get("label", None)

        t0 = time.perf_counter()
        try:
            out = qed.qed(
                signal,
                scenario=line_scenario,
                bit_depth=line_bit_depth,
                sample_rate_hz=line_sample_rate,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error = None
        except ValueError as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            out = None
            error = str(e)

        result: Dict[str, Any] = {
            "idx": idx,
            "scenario": line_scenario,
            "n_samples": len(signal),
            "latency_ms": latency_ms,
            "error": error,
        }

        if out is not None:
            result["ratio"] = out["ratio"]
            result["H_bits"] = out["H_bits"]
            result["recall"] = out["recall"]
            result["savings_M"] = out["savings_M"]
            result["trace"] = out["trace"]

            receipt = out["receipt"]
            result["verified"] = receipt.verified
            result["violations"] = receipt.violations
            result["params"] = receipt.params

        if expected_label is not None:
            result["expected_label"] = expected_label
            if out is not None:
                events = _count_events(signal, threshold=10.0)
                if expected_label == "anomaly":
                    result["is_hit"] = events > 0 and out["recall"] >= 0.95
                else:
                    result["is_hit"] = receipt.verified is True
            else:
                result["is_hit"] = expected_label == "anomaly"

        results.append(result)

    return results


# --- Sympy Suite Subcommand ---


def sympy_suite(
    hook: str,
    test_amplitudes: Optional[List[float]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run sympy constraint suite for a given hook/scenario.

    Gets constraints from sympy_constraints module and verifies them
    against test amplitudes (or default range).

    Returns dict with violations, passes, and constraint details.
    """
    constraints = sympy_constraints.get_constraints(hook)

    if test_amplitudes is None:
        test_amplitudes = [float(a) for a in np.arange(0.0, 25.5, 0.5)]

    violations: List[Dict[str, Any]] = []
    passes: List[Dict[str, Any]] = []
    total_tests = 0

    for constraint in tqdm(constraints, desc=f"Checking constraints for {hook}"):
        constraint_id = constraint.get("id", "unknown")
        constraint_type = constraint.get("type", "amplitude_bound")
        bound = constraint.get("bound", float("inf"))
        description = constraint.get("description", "")

        # Only test amplitude_bound constraints with amplitude values
        if constraint_type != "amplitude_bound":
            continue

        for A in test_amplitudes:
            total_tests += 1
            exceeds_bound = abs(A) > bound

            if exceeds_bound:
                violations.append(
                    {
                        "constraint_id": constraint_id,
                        "amplitude": A,
                        "bound": bound,
                        "description": description,
                    }
                )
            else:
                passes.append(
                    {
                        "constraint_id": constraint_id,
                        "amplitude": A,
                        "bound": bound,
                    }
                )

    verified_at_bound, violations_at_bound = qed.check_constraints(
        constraints[0]["bound"] if constraints else 14.7,
        40.0,
        hook,
    )

    return {
        "hook": hook,
        "constraints": constraints,
        "total_tests": total_tests,
        "n_violations": len(violations),
        "n_passes": len(passes),
        "violations": violations if verbose else violations[:10],
        "verified_at_bound": verified_at_bound,
        "violations_at_bound": violations_at_bound,
    }


# --- Summarize Subcommand ---


def summarize(
    results: List[Dict[str, Any]],
    fleet_size: int = 2_000_000,
) -> Dict[str, Any]:
    """
    Summarize replay results into KPI metrics.

    Computes:
      - hits/misses (recall/precision proxy)
      - violations count
      - ROI estimate ($M)
      - Confidence intervals where applicable

    Returns JSON-serializable summary dict.
    """
    if not results:
        return {
            "n_scenarios": 0,
            "hits": 0,
            "misses": 0,
            "recall": 0.0,
            "precision": 0.0,
            "violations": 0,
            "roi_M": 0.0,
            "kpi_pass": False,
        }

    n_scenarios = len(results)
    n_errors = sum(1 for r in results if r.get("error") is not None)

    labeled = [r for r in results if "expected_label" in r]
    anomalies = [r for r in labeled if r["expected_label"] == "anomaly"]
    normals = [r for r in labeled if r["expected_label"] == "normal"]

    if anomalies:
        hits_anomaly = sum(1 for r in anomalies if r.get("is_hit", False))
        recall = hits_anomaly / len(anomalies)
    else:
        hits_anomaly = 0
        recall = 1.0

    if normals:
        false_positives = sum(1 for r in normals if not r.get("is_hit", True))
        true_positives = hits_anomaly
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 1.0
        )
    else:
        precision = 1.0

    total_violations = sum(
        len(r.get("violations", [])) for r in results if r.get("error") is None
    )

    valid_results = [r for r in results if r.get("error") is None]
    if valid_results:
        avg_savings_M = np.mean([r.get("savings_M", 0.0) for r in valid_results])
        roi_M = float(avg_savings_M)
    else:
        roi_M = 0.0

    latencies = [r.get("latency_ms", 0.0) for r in results]
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    max_latency_ms = float(np.max(latencies)) if latencies else 0.0

    if anomalies:
        n = len(anomalies)
        p = recall
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        recall_ci_lower = max(0.0, center - spread)
        recall_ci_upper = min(1.0, center + spread)
    else:
        recall_ci_lower = recall_ci_upper = recall

    kpi_recall_pass = recall >= KPI_RECALL_THRESHOLD
    kpi_precision_pass = precision >= KPI_PRECISION_THRESHOLD
    kpi_roi_pass = abs(roi_M - KPI_ROI_TARGET_M) <= KPI_ROI_TOLERANCE_M
    kpi_violations_pass = total_violations == 0 or len(normals) == 0
    kpi_pass = all(
        [kpi_recall_pass, kpi_precision_pass, kpi_roi_pass, kpi_violations_pass]
    )

    return {
        "n_scenarios": n_scenarios,
        "n_errors": n_errors,
        "n_anomalies": len(anomalies),
        "n_normals": len(normals),
        "hits": hits_anomaly,
        "misses": len(anomalies) - hits_anomaly,
        "recall": round(recall, 6),
        "recall_ci_95": [round(recall_ci_lower, 6), round(recall_ci_upper, 6)],
        "precision": round(precision, 6),
        "violations": total_violations,
        "roi_M": round(roi_M, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "max_latency_ms": round(max_latency_ms, 2),
        "kpi": {
            "recall_pass": kpi_recall_pass,
            "precision_pass": kpi_precision_pass,
            "roi_pass": kpi_roi_pass,
            "violations_pass": kpi_violations_pass,
            "all_pass": kpi_pass,
        },
        "kpi_thresholds": {
            "recall": KPI_RECALL_THRESHOLD,
            "precision": KPI_PRECISION_THRESHOLD,
            "roi_target_M": KPI_ROI_TARGET_M,
            "roi_tolerance_M": KPI_ROI_TOLERANCE_M,
        },
    }


# --- Legacy Gates Subcommand ---


def run_proof(seed: int = 42424242) -> Dict[str, Any]:
    """
    Run legacy v5 gate checks with synthetic signals.

    Returns dict with gate results and metrics.
    """
    gates: Dict[str, bool] = {}
    failed: List[str] = []

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

    A_a, f_a, phi_a, c_a = qed._fit_dominant_sine(signal_a, sample_rate_a)
    t_a = np.arange(signal_a.size) / sample_rate_a
    recon_a = A_a * np.sin(2.0 * np.pi * f_a * t_a + phi_a) + c_a
    nrmse_a = _normalized_rms_error(signal_a, recon_a)
    events_a = _count_events(signal_a, threshold=10.0)

    gates["G1_ratio"] = 57.0 <= ratio_a <= 63.0
    gates["G2_entropy"] = 7150.0 <= H_bits_a <= 7250.0
    gates["G3_recall_with_events"] = (events_a >= 50) and (recall_a >= 0.9985)
    gates["G4_roi"] = (abs(ratio_a - 60.0) < 1.0) and (37.8 <= savings_m_a <= 38.2)
    gates["G6_reconstruction"] = nrmse_a <= 0.05
    gates["G7_determinism"] = _deterministic_check(
        signal_a, sample_rate_hz=sample_rate_a
    )
    gates["G8_latency_ms"] = latency_a_ms <= 50.0

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

    for name, ok in gates.items():
        if not ok:
            failed.append(name)

    return {
        "gates": gates,
        "failed": failed,
        "all_pass": len(failed) == 0,
        "metrics": {
            "ratio": ratio_a,
            "H_bits": H_bits_a,
            "recall": recall_a,
            "savings_M": savings_m_a,
            "nrmse": nrmse_a,
            "latency_ms": latency_a_ms,
            "trace": trace_a,
        },
    }


# --- Generate Sample JSONL ---


def generate_edge_lab_sample(
    output_path: str,
    n_anomalies: int = 900,
    n_normals: int = 100,
    seed: int = 42,
) -> None:
    """
    Generate edge_lab_sample.jsonl with labeled anomaly/normal scenarios.

    Creates synthetic signals with known labels for validation testing.
    """
    rng = np.random.default_rng(seed)
    path = Path(output_path)

    with path.open("w") as f:
        for i in tqdm(range(n_anomalies), desc="Generating anomalies"):
            amplitude = rng.uniform(11.0, 14.5)
            frequency = rng.uniform(20.0, 100.0)
            noise = rng.uniform(0.05, 0.2)

            record = {
                "id": f"anomaly_{i:04d}",
                "label": "anomaly",
                "params": {
                    "n": 1000,
                    "sample_rate_hz": 1000.0,
                    "amplitude": float(amplitude),
                    "frequency_hz": float(frequency),
                    "noise_sigma": float(noise),
                    "offset": float(rng.uniform(0.0, 5.0)),
                    "phase_rad": float(rng.uniform(0.0, 2 * np.pi)),
                    "seed": seed + i,
                },
                "scenario": "tesla_fsd",
            }
            f.write(json.dumps(record) + "\n")

        for i in tqdm(range(n_normals), desc="Generating normals"):
            amplitude = rng.uniform(2.0, 8.0)
            frequency = rng.uniform(20.0, 100.0)
            noise = rng.uniform(0.01, 0.1)

            record = {
                "id": f"normal_{i:04d}",
                "label": "normal",
                "params": {
                    "n": 1000,
                    "sample_rate_hz": 1000.0,
                    "amplitude": float(amplitude),
                    "frequency_hz": float(frequency),
                    "noise_sigma": float(noise),
                    "offset": float(rng.uniform(0.0, 3.0)),
                    "phase_rad": float(rng.uniform(0.0, 2 * np.pi)),
                    "seed": seed + n_anomalies + i,
                },
                "scenario": "tesla_fsd",
            }
            f.write(json.dumps(record) + "\n")

    print(f"Generated {n_anomalies + n_normals} scenarios to {output_path}")


# --- v7 Subcommands ---


def run_sims(
    receipts_dir: str = "receipts/",
    patterns_path: str = "data/shared_anomalies.jsonl",
    n_per_hook: int = 1000,
    output: str = "data/sim_results.json",
) -> Dict[str, Any]:
    """
    Run pattern simulations via edge_lab_v2.

    Calls run_pattern_sims() with progress tracking and writes results to JSON.
    Returns summary dict with n_tests, aggregate_recall, aggregate_fp_rate.
    """
    # Load patterns for progress tracking
    patterns = load_library(patterns_path)

    # Run simulations with progress
    results = run_pattern_sims(
        receipts_dir=receipts_dir,
        patterns_path=patterns_path,
        n_per_hook=n_per_hook,
        progress_callback=lambda: tqdm(patterns, desc="Running pattern sims"),
    )

    # Write results to output file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Compute summary
    n_tests = results.get("n_tests", 0)
    aggregate_recall = results.get("aggregate_recall", 0.0)
    aggregate_fp_rate = results.get("aggregate_fp_rate", 0.0)

    return {
        "n_tests": n_tests,
        "aggregate_recall": aggregate_recall,
        "aggregate_fp_rate": aggregate_fp_rate,
        "output_path": str(output_path),
    }


def recall_floor(
    sim_results_path: Optional[str] = "data/sim_results.json",
    confidence: float = 0.95,
    n_tests: Optional[int] = None,
    n_misses: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Clopper-Pearson exact recall lower bound.

    Uses scipy.stats.beta.ppf for exact binomial confidence interval.
    Formula: beta.ppf(alpha/2, k, n-k+1) where k=successes, n=total.

    Returns dict with recall_floor, confidence, n_tests, n_misses.
    """
    # Get n_tests and n_misses from sim_results or overrides
    if n_tests is None or n_misses is None:
        if sim_results_path is None:
            raise ValueError(
                "Must provide either sim_results_path or both n_tests and n_misses"
            )
        with open(sim_results_path, "r") as f:
            sim_data = json.load(f)

        if n_tests is None:
            n_tests = sim_data.get("n_tests", 0)
        if n_misses is None:
            n_misses = sim_data.get("n_misses", 0)

    # Compute Clopper-Pearson exact lower bound
    # k = number of successes (hits), n = total tests
    n_successes = n_tests - n_misses
    alpha = 1.0 - confidence

    if n_successes == 0:
        # Lower bound is 0 when no successes
        lower_bound = 0.0
    else:
        # Clopper-Pearson exact lower bound
        # beta.ppf(alpha/2, k, n-k+1) gives lower bound for proportion k/n
        lower_bound = float(beta.ppf(alpha / 2, n_successes, n_misses + 1))

    return {
        "recall_floor": lower_bound,
        "confidence": confidence,
        "n_tests": n_tests,
        "n_misses": n_misses,
        "n_successes": n_successes,
    }


def pattern_report(
    patterns_path: str = "data/shared_anomalies.jsonl",
    sort_by: str = "dollar_value",
    exploit_only: bool = False,
    output_format: str = "table",
) -> List[Dict[str, Any]]:
    """
    Load and display pattern library with sorting and filtering.

    Loads patterns via shared_anomalies.load_library(), sorts by selected field,
    and optionally filters for exploit_grade=true patterns only.

    Returns list of pattern dicts.
    """
    patterns = load_library(patterns_path)

    # Filter if exploit_only
    if exploit_only:
        patterns = [p for p in patterns if p.get("exploit_grade", False)]

    # Sort by selected field (descending)
    sort_key_map = {
        "dollar_value": lambda p: p.get("dollar_value", 0),
        "recall": lambda p: p.get("recall", 0),
        "exploit_grade": lambda p: (1 if p.get("exploit_grade", False) else 0),
    }
    sort_fn = sort_key_map.get(sort_by, sort_key_map["dollar_value"])
    patterns = sorted(patterns, key=sort_fn, reverse=True)

    # Output in requested format
    if output_format == "json":
        print(json.dumps(patterns, indent=2))
    elif output_format == "table":
        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="Pattern Report")

            table.add_column("pattern_id", style="cyan", no_wrap=True)
            table.add_column("physics_domain", style="green")
            table.add_column("failure_mode", style="yellow")
            table.add_column("dollar_value", justify="right", style="magenta")
            table.add_column("recall", justify="right", style="blue")
            table.add_column("fp_rate", justify="right", style="red")
            table.add_column("exploit_grade", justify="center", style="bold")

            for p in patterns:
                pattern_id = str(p.get("pattern_id", ""))[:20]  # truncated
                physics_domain = str(p.get("physics_domain", ""))
                failure_mode = str(p.get("failure_mode", ""))
                dollar_value = f"${p.get('dollar_value', 0):,.0f}"
                recall_val = f"{p.get('recall', 0):.4f}"
                fp_rate = f"{p.get('fp_rate', 0):.4f}"
                exploit = "Yes" if p.get("exploit_grade", False) else "No"

                table.add_row(
                    pattern_id,
                    physics_domain,
                    failure_mode,
                    dollar_value,
                    recall_val,
                    fp_rate,
                    exploit,
                )

            console.print(table)
        else:
            # Fallback to simple text table
            header = (
                f"{'pattern_id':<20} {'physics_domain':<15} {'failure_mode':<20} "
                f"{'dollar_value':>12} {'recall':>8} {'fp_rate':>8} {'exploit':>8}"
            )
            print(header)
            print("-" * len(header))
            for p in patterns:
                pattern_id = str(p.get("pattern_id", ""))[:20]
                physics_domain = str(p.get("physics_domain", ""))[:15]
                failure_mode = str(p.get("failure_mode", ""))[:20]
                dollar_value = p.get("dollar_value", 0)
                recall_val = p.get("recall", 0)
                fp_rate = p.get("fp_rate", 0)
                exploit = "Yes" if p.get("exploit_grade", False) else "No"
                print(
                    f"{pattern_id:<20} {physics_domain:<15} {failure_mode:<20} "
                    f"{dollar_value:>12,.0f} {recall_val:>8.4f} {fp_rate:>8.4f} {exploit:>8}"
                )

    return patterns


def clarity_audit(
    receipts_path: str,
    output_corpus: Optional[str] = None,
    output_receipt: str = "data/clarity_receipts.jsonl",
) -> Dict[str, Any]:
    """
    Process receipts through ClarityClean adapter.

    Calls clarity_clean_adapter.process_receipts() and emits ClarityCleanReceipt
    to JSONL output.

    Returns summary dict with token_count, anomaly_density, noise_ratio, corpus_hash.
    """
    # Process receipts
    result = process_receipts(
        receipts_path=receipts_path,
        output_corpus=output_corpus,
    )

    # Extract ClarityCleanReceipt
    receipt = result.get("receipt", {})

    # Write receipt to JSONL
    output_path = Path(output_receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(receipt) + "\n")

    # Return summary
    return {
        "token_count": receipt.get("token_count", 0),
        "anomaly_density": receipt.get("anomaly_density", 0.0),
        "noise_ratio": receipt.get("noise_ratio", 0.0),
        "corpus_hash": receipt.get("corpus_hash", ""),
        "output_receipt": str(output_path),
        "output_corpus": output_corpus,
    }


# --- CLI Main ---


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QED v6/v7 Proof CLI Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python proof.py gates                    # Run legacy v5 gate checks
  python proof.py replay sample.jsonl      # Replay scenarios from JSONL
  python proof.py sympy_suite tesla_fsd    # Check constraints for hook
  python proof.py generate --output edge_lab_sample.jsonl

v7 Commands:
  python proof.py run-sims                 # Run pattern simulations
  python proof.py recall-floor             # Compute recall lower bound
  python proof.py pattern-report           # Display pattern library
  python proof.py clarity-audit --receipts-path receipts.jsonl
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    gates_parser = subparsers.add_parser(
        "gates",
        help="Run legacy v5 gate checks with synthetic signals",
    )
    gates_parser.add_argument(
        "--seed",
        type=int,
        default=42424242,
        help="Random seed for signal generation",
    )
    gates_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay scenarios from JSONL through qed.py",
    )
    replay_parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to JSONL file with scenarios",
    )
    replay_parser.add_argument(
        "--scenario",
        type=str,
        default="tesla_fsd",
        help="Default scenario if not specified in JSONL",
    )
    replay_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    replay_parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    suite_parser = subparsers.add_parser(
        "sympy_suite",
        help="Run sympy constraint suite for a hook",
    )
    suite_parser.add_argument(
        "hook",
        type=str,
        help="Hook/scenario name (e.g., tesla_fsd, spacex_flight)",
    )
    suite_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include all violations in output",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate edge_lab_sample.jsonl test data",
    )
    generate_parser.add_argument(
        "--output",
        type=str,
        default="edge_lab_sample.jsonl",
        help="Output JSONL file path",
    )
    generate_parser.add_argument(
        "--anomalies",
        type=int,
        default=900,
        help="Number of anomaly scenarios (default: 900)",
    )
    generate_parser.add_argument(
        "--normals",
        type=int,
        default=100,
        help="Number of normal scenarios (default: 100)",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize replay results to KPI metrics",
    )
    summarize_parser.add_argument(
        "results_json",
        type=str,
        help="Path to JSON file with replay results",
    )

    # --- v7 Subcommands ---

    run_sims_parser = subparsers.add_parser(
        "run-sims",
        help="Run pattern simulations via edge_lab_v2",
    )
    run_sims_parser.add_argument(
        "--receipts-dir",
        type=str,
        default="receipts/",
        help="Directory containing receipt files (default: receipts/)",
    )
    run_sims_parser.add_argument(
        "--patterns-path",
        type=str,
        default="data/shared_anomalies.jsonl",
        help="Path to patterns JSONL file (default: data/shared_anomalies.jsonl)",
    )
    run_sims_parser.add_argument(
        "--n-per-hook",
        type=int,
        default=1000,
        help="Number of simulations per hook (default: 1000)",
    )
    run_sims_parser.add_argument(
        "--output",
        type=str,
        default="data/sim_results.json",
        help="Output JSON file for results (default: data/sim_results.json)",
    )

    recall_floor_parser = subparsers.add_parser(
        "recall-floor",
        help="Compute Clopper-Pearson exact recall lower bound",
    )
    recall_floor_parser.add_argument(
        "--sim-results",
        type=str,
        default="data/sim_results.json",
        help="Path to simulation results JSON (default: data/sim_results.json)",
    )
    recall_floor_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )
    recall_floor_parser.add_argument(
        "--n-tests",
        type=int,
        default=None,
        help="Override n_tests from sim results",
    )
    recall_floor_parser.add_argument(
        "--n-misses",
        type=int,
        default=None,
        help="Override n_misses from sim results",
    )

    pattern_report_parser = subparsers.add_parser(
        "pattern-report",
        help="Display pattern library with sorting and filtering",
    )
    pattern_report_parser.add_argument(
        "--patterns-path",
        type=str,
        default="data/shared_anomalies.jsonl",
        help="Path to patterns JSONL file (default: data/shared_anomalies.jsonl)",
    )
    pattern_report_parser.add_argument(
        "--sort-by",
        type=str,
        choices=["dollar_value", "recall", "exploit_grade"],
        default="dollar_value",
        help="Sort field (default: dollar_value)",
    )
    pattern_report_parser.add_argument(
        "--exploit-only",
        action="store_true",
        help="Show only exploit_grade=true patterns",
    )
    pattern_report_parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        dest="output_format",
        help="Output format (default: table)",
    )

    clarity_audit_parser = subparsers.add_parser(
        "clarity-audit",
        help="Process receipts through ClarityClean adapter",
    )
    clarity_audit_parser.add_argument(
        "--receipts-path",
        type=str,
        required=True,
        help="Path to receipts file (required)",
    )
    clarity_audit_parser.add_argument(
        "--output-corpus",
        type=str,
        default=None,
        help="Path to write cleaned corpus (optional)",
    )
    clarity_audit_parser.add_argument(
        "--output-receipt",
        type=str,
        default="data/clarity_receipts.jsonl",
        help="Path to write ClarityCleanReceipt JSONL (default: data/clarity_receipts.jsonl)",
    )

    args = parser.parse_args()

    if args.command == "gates":
        result = run_proof(seed=args.seed)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["all_pass"]:
                print("QED v6 proof gates passed")
                metrics = result["metrics"]
                print(
                    f"Signal A: "
                    f"ratio={metrics['ratio']:.1f}, "
                    f"H_bits={metrics['H_bits']:.0f}, "
                    f"recall={metrics['recall']:.4f}, "
                    f"savings_M={metrics['savings_M']:.2f}, "
                    f"nrmse={metrics['nrmse']:.4f}, "
                    f"latency_ms={metrics['latency_ms']:.2f}, "
                    f"trace={metrics['trace']}"
                )
            else:
                print(f"FAILED gates: {result['failed']}")
                return 1

    elif args.command == "replay":
        results = replay(
            args.jsonl_path,
            scenario=args.scenario,
            verbose=args.verbose,
        )
        summary = summarize(results)

        if args.output:
            output_data = {"results": results, "summary": summary}
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(summary, indent=2))

        if not summary["kpi"]["all_pass"]:
            return 1

    elif args.command == "sympy_suite":
        result = sympy_suite(args.hook, verbose=args.verbose)
        print(json.dumps(result, indent=2))

        if result["n_violations"] > 0 and not args.verbose:
            print(
                f"\nNote: {result['n_violations']} violations found. "
                "Use --verbose for details."
            )

    elif args.command == "generate":
        generate_edge_lab_sample(
            args.output,
            n_anomalies=args.anomalies,
            n_normals=args.normals,
            seed=args.seed,
        )

    elif args.command == "summarize":
        with open(args.results_json, "r") as f:
            data = json.load(f)

        results = data.get("results", data)
        summary = summarize(results)
        print(json.dumps(summary, indent=2))

        if not summary["kpi"]["all_pass"]:
            return 1

    # --- v7 Command Handlers ---

    elif args.command == "run-sims":
        result = run_sims(
            receipts_dir=args.receipts_dir,
            patterns_path=args.patterns_path,
            n_per_hook=args.n_per_hook,
            output=args.output,
        )
        print(
            f"Simulation complete: n_tests={result['n_tests']}, "
            f"aggregate_recall={result['aggregate_recall']:.4f}, "
            f"aggregate_fp_rate={result['aggregate_fp_rate']:.4f}"
        )
        print(f"Results written to {result['output_path']}")

    elif args.command == "recall-floor":
        result = recall_floor(
            sim_results_path=args.sim_results,
            confidence=args.confidence,
            n_tests=args.n_tests,
            n_misses=args.n_misses,
        )
        confidence_pct = result["confidence"] * 100
        print(
            f"Recall floor: {result['recall_floor']:.4f} at {confidence_pct:.0f}% confidence "
            f"({result['n_tests']} tests, {result['n_misses']} misses)"
        )

    elif args.command == "pattern-report":
        pattern_report(
            patterns_path=args.patterns_path,
            sort_by=args.sort_by,
            exploit_only=args.exploit_only,
            output_format=args.output_format,
        )

    elif args.command == "clarity-audit":
        result = clarity_audit(
            receipts_path=args.receipts_path,
            output_corpus=args.output_corpus,
            output_receipt=args.output_receipt,
        )
        print(
            f"ClarityClean audit complete:\n"
            f"  token_count: {result['token_count']}\n"
            f"  anomaly_density: {result['anomaly_density']:.4f}\n"
            f"  noise_ratio: {result['noise_ratio']:.4f}\n"
            f"  corpus_hash: {result['corpus_hash']}"
        )
        print(f"Receipt written to {result['output_receipt']}")
        if result["output_corpus"]:
            print(f"Corpus written to {result['output_corpus']}")

    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
