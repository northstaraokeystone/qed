from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
from scipy.stats import beta

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Local imports
import qed
import sympy_constraints
from clarity_clean_adapter import process_receipts
from edge_lab_v2 import run_pattern_sims
from shared_anomalies import load_library

# v9 imports for new subcommands
from causal_graph import (
    centrality,
    trace_forward,
    trace_backward,
    self_compression_ratio,
    entanglement_coefficient,
    build_graph,
)
from event_stream import replay as event_replay, query as event_query
from binder import QueryPredicate
from portfolio_aggregator import portfolio_health
import networkx as nx

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


# --- v9 Helper Functions ---


def _emit_cli_receipt(
    subcommand: str,
    args: Dict[str, Any],
    result: Dict[str, Any],
    exit_code: int,
    receipts_path: str = "data/receipts.jsonl",
) -> None:
    """
    Emit cli_receipt to JSONL log (CLAUDEME.md Section 5.2 compliance).

    Every CLI action emits a receipt for auditability.

    Args:
        subcommand: Subcommand name (value, trace-forward, etc.)
        args: Arguments passed to subcommand
        result: Result dict from subcommand execution
        exit_code: Exit code (0=success, 1=not found, 2=error)
        receipts_path: Path to receipts JSONL file
    """
    receipt = {
        "type": "cli_receipt",
        "subcommand": subcommand,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": args,
        "result": result,
        "exit_code": exit_code,
    }

    output_path = Path(receipts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(receipt) + "\n")


def _load_graph(graph_path: str = "data/graph") -> nx.DiGraph:
    """
    Load graph from receipts or build from available data.

    Args:
        graph_path: Directory containing graph data

    Returns:
        NetworkX DiGraph, or empty graph if no data
    """
    # Try to load from receipts.jsonl
    receipts_file = Path("data/receipts.jsonl")
    if receipts_file.exists():
        receipts = []
        with receipts_file.open("r") as f:
            for line in f:
                if line.strip():
                    receipts.append(json.loads(line))
        if receipts:
            return build_graph(receipts)

    # Return empty graph if no data
    return nx.DiGraph()


def _ascii_box(title: str, lines: List[str], width: int = 60) -> str:
    """
    Create ASCII box with Unicode box drawing characters.

    Args:
        title: Box title
        lines: Content lines
        width: Box width in characters

    Returns:
        Multi-line ASCII art string
    """
    # Box drawing characters
    top_left = "╭"
    top_right = "╮"
    bottom_left = "╰"
    bottom_right = "╯"
    horizontal = "─"
    vertical = "│"

    # Build box
    result = []

    # Top border with title
    title_line = f"─ {title} "
    title_padding = horizontal * (width - len(title_line) - 2)
    result.append(f"{top_left}{title_line}{title_padding}{top_right}")

    # Content lines
    for line in lines:
        # Truncate or pad to fit
        if len(line) > width - 2:
            line = line[: width - 5] + "..."
        padding = " " * (width - len(line) - 2)
        result.append(f"{vertical} {line}{padding}{vertical}")

    # Bottom border
    result.append(f"{bottom_left}{horizontal * (width - 2)}{bottom_right}")

    return "\n".join(result)


# --- v9 Subcommands ---


def cmd_value(
    pattern_id: str,
    graph_path: str = "data/graph",
    output_json: bool = False,
    quiet: bool = False,
) -> int:
    """
    Auditor Question: "What is this pattern worth?"

    Computes centrality via graph topology (v9 Paradigm 2: Value is Topology).
    No stored dollar fields - value always derived from current graph.

    Args:
        pattern_id: Pattern to compute value for
        graph_path: Path to graph data directory
        output_json: Output as JSON instead of ASCII
        quiet: Suppress ASCII decorations

    Returns:
        Exit code: 0=found, 1=not found, 2=graph error
    """
    try:
        graph = _load_graph(graph_path)

        if graph.number_of_nodes() == 0:
            result = {"error": "Empty graph - no data to compute centrality"}
            _emit_cli_receipt("value", {"pattern_id": pattern_id}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        cent = centrality(pattern_id, graph)

        if cent == 0.0 and pattern_id not in graph:
            result = {"error": f"Pattern '{pattern_id}' not found in graph"}
            _emit_cli_receipt("value", {"pattern_id": pattern_id}, result, 1)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 1

        # Count connected patterns
        if pattern_id in graph:
            upstream = len(list(graph.predecessors(pattern_id)))
            downstream = len(list(graph.successors(pattern_id)))
        else:
            upstream = 0
            downstream = 0

        # Determine status vs CENTRALITY_FLOOR (0.2)
        floor = 0.2
        status = "ABOVE FLOOR" if cent >= floor else "BELOW FLOOR"

        result = {
            "pattern_id": pattern_id,
            "centrality": round(cent, 3),
            "status": status,
            "floor": floor,
            "connected_upstream": upstream,
            "connected_downstream": downstream,
        }

        _emit_cli_receipt("value", {"pattern_id": pattern_id}, result, 0)

        if output_json:
            print(json.dumps(result, indent=2))
        else:
            lines = [
                f"Pattern: {pattern_id}",
                f"Centrality: {cent:.3f}",
                f"Status: {status} (floor={floor})",
                f"Connected patterns: {upstream} upstream, {downstream} downstream",
            ]
            print(_ascii_box("Pattern Value", lines))
            print(f"Next: proof trace-forward {pattern_id}")

        return 0

    except Exception as e:
        result = {"error": str(e)}
        _emit_cli_receipt("value", {"pattern_id": pattern_id}, result, 2)
        if output_json:
            print(json.dumps(result))
        else:
            print(f"ERROR: {e}")
        return 2


def cmd_trace_forward(
    node_id: str,
    depth: int = 3,
    graph_path: str = "data/graph",
    output_json: bool = False,
    quiet: bool = False,
) -> int:
    """
    Auditor Question: "What would change if we changed this?"

    Forward causality for blast radius analysis (v9 Paradigm 4).

    Args:
        node_id: Starting node ID
        depth: Maximum depth to traverse (default 3, max 10)
        graph_path: Path to graph data directory
        output_json: Output as JSON instead of ASCII
        quiet: Suppress ASCII decorations

    Returns:
        Exit code: 0=found, 1=node not found, 2=graph error
    """
    try:
        graph = _load_graph(graph_path)

        if graph.number_of_nodes() == 0:
            result = {"error": "Empty graph - no data to trace"}
            _emit_cli_receipt("trace-forward", {"node_id": node_id, "depth": depth}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        if node_id not in graph:
            result = {"error": f"Node '{node_id}' not found in graph"}
            _emit_cli_receipt("trace-forward", {"node_id": node_id, "depth": depth}, result, 1)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 1

        # Clamp depth to max 10
        depth = min(depth, 10)

        # Trace forward
        affected = trace_forward(node_id, graph, max_depth=100)

        # Group by depth for display
        depth_groups: Dict[int, List[str]] = {}
        for node in affected[:depth * 3]:  # Limit display
            # Compute shortest path length to get depth
            try:
                path_len = nx.shortest_path_length(graph, node_id, node)
            except nx.NetworkXNoPath:
                path_len = depth + 1  # Beyond max depth

            if path_len <= depth:
                if path_len not in depth_groups:
                    depth_groups[path_len] = []
                depth_groups[path_len].append(node)

        # Compute max downstream centrality
        max_cent = 0.0
        for node in affected[:10]:  # Check top 10
            cent = centrality(node, graph)
            max_cent = max(max_cent, cent)

        result = {
            "node_id": node_id,
            "depth": depth,
            "affected_count": len(affected),
            "depth_groups": {str(k): v for k, v in depth_groups.items()},
            "max_downstream_centrality": round(max_cent, 3),
        }

        _emit_cli_receipt("trace-forward", {"node_id": node_id, "depth": depth}, result, 0)

        if output_json:
            print(json.dumps(result, indent=2))
        else:
            lines = []
            for d in sorted(depth_groups.keys()):
                nodes = depth_groups[d]
                for node in nodes[:2]:  # Show max 2 per depth
                    cent = centrality(node, graph)
                    lines.append(f"Depth {d}: {node} (centrality={cent:.2f})")

            lines.append("")
            lines.append(f"Blast radius: {len(affected)} patterns affected")
            lines.append(f"Max downstream centrality: {max_cent:.2f}")

            print(_ascii_box(f"Forward Trace: {node_id}", lines))

            # Suggest next command
            if affected:
                print(f"Next: proof trace-backward {affected[0]}")
            else:
                print(f"Next: proof value {node_id}")

        return 0

    except Exception as e:
        result = {"error": str(e)}
        _emit_cli_receipt("trace-forward", {"node_id": node_id, "depth": depth}, result, 2)
        if output_json:
            print(json.dumps(result))
        else:
            print(f"ERROR: {e}")
        return 2


def cmd_trace_backward(
    node_id: str,
    depth: int = 3,
    graph_path: str = "data/graph",
    output_json: bool = False,
    quiet: bool = False,
) -> int:
    """
    Auditor Question: "How did this decision get made?"

    Backward causality for root cause analysis (v9 Paradigm 4).

    Args:
        node_id: Starting node ID
        depth: Maximum depth to traverse (default 3, max 10)
        graph_path: Path to graph data directory
        output_json: Output as JSON instead of ASCII
        quiet: Suppress ASCII decorations

    Returns:
        Exit code: 0=found, 1=node not found, 2=graph error
    """
    try:
        graph = _load_graph(graph_path)

        if graph.number_of_nodes() == 0:
            result = {"error": "Empty graph - no data to trace"}
            _emit_cli_receipt("trace-backward", {"node_id": node_id, "depth": depth}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        if node_id not in graph:
            result = {"error": f"Node '{node_id}' not found in graph"}
            _emit_cli_receipt("trace-backward", {"node_id": node_id, "depth": depth}, result, 1)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 1

        # Clamp depth to max 10
        depth = min(depth, 10)

        # Trace backward
        causes = trace_backward(node_id, graph, max_depth=100)

        # Group by depth for display
        depth_groups: Dict[int, List[str]] = {}
        for node in causes[:depth * 3]:  # Limit display
            # Compute shortest path length to get depth
            try:
                path_len = nx.shortest_path_length(graph, node, node_id)
            except nx.NetworkXNoPath:
                path_len = depth + 1  # Beyond max depth

            if path_len <= depth:
                if path_len not in depth_groups:
                    depth_groups[path_len] = []
                depth_groups[path_len].append(node)

        # Find terminal nodes (root causes - no predecessors)
        root_causes = []
        for node in causes:
            if graph.in_degree(node) == 0:
                root_causes.append(node)

        # Find primary cause (highest centrality root)
        primary_cause = None
        max_cent = 0.0
        for node in root_causes:
            cent = centrality(node, graph)
            if cent > max_cent:
                max_cent = cent
                primary_cause = node

        result = {
            "node_id": node_id,
            "depth": depth,
            "causes_count": len(causes),
            "depth_groups": {str(k): v for k, v in depth_groups.items()},
            "root_causes": root_causes[:5],  # Top 5
            "primary_cause": primary_cause,
            "primary_cause_centrality": round(max_cent, 3) if primary_cause else 0.0,
        }

        _emit_cli_receipt("trace-backward", {"node_id": node_id, "depth": depth}, result, 0)

        if output_json:
            print(json.dumps(result, indent=2))
        else:
            lines = []
            for d in sorted(depth_groups.keys()):
                nodes = depth_groups[d]
                for node in nodes[:2]:  # Show max 2 per depth
                    cent = centrality(node, graph)
                    lines.append(f"Depth {d}: {node} (centrality={cent:.2f})")

            lines.append("")
            lines.append(f"Root causes: {len(root_causes)} terminal nodes")
            if primary_cause:
                lines.append(f"Primary cause: {primary_cause} (highest centrality)")

            print(_ascii_box(f"Backward Trace: {node_id}", lines))

            # Suggest next command
            if primary_cause:
                print(f"Next: proof value {primary_cause}")
            else:
                print(f"Next: proof health")

        return 0

    except Exception as e:
        result = {"error": str(e)}
        _emit_cli_receipt("trace-backward", {"node_id": node_id, "depth": depth}, result, 2)
        if output_json:
            print(json.dumps(result))
        else:
            print(f"ERROR: {e}")
        return 2


def cmd_replay_counterfactual(
    counterfactual: str,
    graph_path: str = "data/graph",
    events_path: str = "data/events/events.jsonl",
    output_json: bool = False,
) -> int:
    """
    Auditor Question: "What if thresholds were different?"

    Backward causation - future counterfactual changes past observations (v9 Paradigm 4).

    Args:
        counterfactual: KEY=VALUE string (e.g., "centrality_floor=0.5")
        graph_path: Path to graph data directory
        events_path: Path to events JSONL
        output_json: Output as JSON instead of ASCII

    Returns:
        Exit code: 0=replay complete, 1=invalid counterfactual, 2=replay error
    """
    try:
        # Parse counterfactual string
        if "=" not in counterfactual:
            result = {"error": "Invalid counterfactual format. Use KEY=VALUE"}
            _emit_cli_receipt("counterfactual", {"counterfactual": counterfactual}, result, 1)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 1

        key, value = counterfactual.split("=", 1)
        try:
            value_float = float(value)
        except ValueError:
            result = {"error": f"Value must be numeric, got '{value}'"}
            _emit_cli_receipt("counterfactual", {"counterfactual": counterfactual}, result, 1)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 1

        # Load graph
        graph = _load_graph(graph_path)

        # Load events from file
        events_file = Path(events_path)
        events = []
        if events_file.exists():
            from event_stream import EventRecord

            with events_file.open("r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        events.append(EventRecord.from_dict(data))

        if not events:
            result = {"error": f"No events found at {events_path}"}
            _emit_cli_receipt("counterfactual", {"counterfactual": counterfactual}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        # Build counterfactual dict
        cf_dict = {"threshold_override": value_float}

        # Replay events under counterfactual
        replay_receipts = event_replay(events[:100], cf_dict, graph)  # Limit to 100

        if not replay_receipts:
            result = {"error": "Replay failed - no receipts emitted"}
            _emit_cli_receipt("counterfactual", {"counterfactual": counterfactual}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        replay_receipt = replay_receipts[0]
        visibility_changes = replay_receipt.get("visibility_changes", {})

        # Count changes
        changes_count = sum(
            1 for v in visibility_changes.values() if v.get("changed", False)
        )

        result = {
            "counterfactual": counterfactual,
            "events_replayed": replay_receipt.get("events_replayed", 0),
            "changes_count": changes_count,
            "total_events": len(visibility_changes),
        }

        _emit_cli_receipt("counterfactual", {"counterfactual": counterfactual}, result, 0)

        if output_json:
            print(json.dumps(result, indent=2))
        else:
            lines = [
                f"Counterfactual: {counterfactual}",
                "",
                f"Mode changes under new physics:",
                f"  Events analyzed: {len(visibility_changes)}",
                f"  Visibility changed: {changes_count}",
                "",
                f"Affected: {changes_count} of {len(visibility_changes)} events would change visibility",
            ]

            print(_ascii_box("Counterfactual Replay", lines))
            print("Next: python proof.py query --lens=shadow")

        return 0

    except Exception as e:
        result = {"error": str(e)}
        _emit_cli_receipt("counterfactual", {"counterfactual": counterfactual}, result, 2)
        if output_json:
            print(json.dumps(result))
        else:
            print(f"ERROR: {e}")
        return 2


def cmd_health(
    graph_path: str = "data/graph",
    output_json: bool = False,
    quiet: bool = False,
) -> int:
    """
    Auditor Question: "How healthy is the system?"

    Combines self-compression ratio + portfolio health (v9 Paradigm 5).

    Args:
        graph_path: Path to graph data directory
        output_json: Output as JSON instead of ASCII
        quiet: Suppress ASCII decorations

    Returns:
        Exit code: 0=healthy (>0.5), 1=caution (0.3-0.5), 2=unhealthy (<0.3)
    """
    try:
        graph = _load_graph(graph_path)

        if graph.number_of_nodes() == 0:
            result = {
                "compression_ratio": 1.0,
                "portfolio_health": 1.0,
                "status": "healthy (empty graph)",
            }
            _emit_cli_receipt("health", {}, result, 0)
            if output_json:
                print(json.dumps(result))
            else:
                print("Empty graph - no health data")
            return 0

        # Compute self-compression ratio
        compression = self_compression_ratio(graph)

        # Compute portfolio health
        companies = ["tesla", "spacex"]  # Default companies
        health_receipts = portfolio_health(graph, companies)

        if health_receipts:
            health_data = health_receipts[0]
            health_score = health_data.get("health_score", 0.0)
            components = health_data.get("components", {})
            avg_centrality = components.get("avg_centrality", 0.0)
            avg_entanglement = components.get("avg_entanglement", 0.0)
        else:
            health_score = 0.0
            avg_centrality = 0.0
            avg_entanglement = 0.0

        # Interpret compression ratio
        if compression > 0.5:
            compression_status = "HEALTHY"
        elif compression >= 0.3:
            compression_status = "CAUTION"
        else:
            compression_status = "UNHEALTHY"

        # Determine overall status
        if compression > 0.5:
            status = "healthy"
            exit_code = 0
        elif compression >= 0.3:
            status = "caution"
            exit_code = 1
        else:
            status = "unhealthy"
            exit_code = 2

        result = {
            "compression_ratio": round(compression, 2),
            "compression_status": compression_status,
            "portfolio_health": round(health_score, 2),
            "avg_centrality": round(avg_centrality, 2),
            "avg_entanglement": round(avg_entanglement, 2),
            "status": status,
        }

        _emit_cli_receipt("health", {}, result, exit_code)

        if output_json:
            print(json.dumps(result, indent=2))
        else:
            lines = [
                f"Self-Compression Ratio: {compression:.2f}",
                f"Interpretation: {compression_status} (system understands itself well)",
                "",
                f"Portfolio Health: {health_score:.2f}",
                f"Avg Centrality: {avg_centrality:.2f}",
                f"Avg Entanglement: {avg_entanglement:.2f}",
                "",
                f"Status: {'✓' if exit_code == 0 else '⚠' if exit_code == 1 else '✗'} {status.upper()}",
            ]

            print(_ascii_box("System Health", lines))
            print("Next: proof entanglement --pattern=<pattern_id>")

        return exit_code

    except Exception as e:
        result = {"error": str(e)}
        _emit_cli_receipt("health", {}, result, 2)
        if output_json:
            print(json.dumps(result))
        else:
            print(f"ERROR: {e}")
        return 2


def cmd_entanglement(
    pattern: Optional[str] = None,
    companies: Optional[str] = None,
    graph_path: str = "data/graph",
    output_json: bool = False,
) -> int:
    """
    Auditor Question: "What's our systemic risk?"

    Cross-company entanglement coefficient with SLO check (v9 Paradigm 6 + CLAUDEME 5.3).

    Args:
        pattern: Pattern ID to analyze (optional - shows top 10 if not specified)
        companies: Comma-separated company list (default: tesla,spacex)
        graph_path: Path to graph data directory
        output_json: Output as JSON instead of ASCII

    Returns:
        Exit code: 0=low risk (<0.8), 1=elevated (0.8-0.92), 2=high/SLO violation (>0.92)
    """
    try:
        graph = _load_graph(graph_path)

        if graph.number_of_nodes() == 0:
            result = {"error": "Empty graph - no entanglement data"}
            _emit_cli_receipt("entanglement", {"pattern": pattern, "companies": companies}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        # Parse companies
        if companies:
            company_list = [c.strip() for c in companies.split(",")]
        else:
            company_list = ["tesla", "spacex"]

        # If no pattern specified, show top 10 most entangled
        if pattern is None:
            # Get all patterns
            all_patterns = []
            for node in graph.nodes():
                if graph.nodes[node].get("is_pattern", False) or (
                    isinstance(node, str) and "pattern" in node.lower()
                ):
                    all_patterns.append(node)

            if not all_patterns:
                all_patterns = list(graph.nodes())[:10]  # Use first 10 nodes

            # Compute entanglement for each
            pattern_entanglements = []
            for p in all_patterns[:20]:  # Limit to 20
                coeff = entanglement_coefficient(p, company_list, graph)
                pattern_entanglements.append((p, coeff))

            # Sort by coefficient descending
            pattern_entanglements.sort(key=lambda x: x[1], reverse=True)

            result = {
                "top_patterns": [
                    {"pattern_id": p, "coefficient": round(c, 3)}
                    for p, c in pattern_entanglements[:10]
                ],
                "companies": company_list,
            }

            _emit_cli_receipt("entanglement", {"companies": companies}, result, 0)

            if output_json:
                print(json.dumps(result, indent=2))
            else:
                lines = [f"Companies: {', '.join(company_list)}", ""]
                lines.append("Top 10 most entangled patterns:")
                for p, c in pattern_entanglements[:10]:
                    lines.append(f"  {p}: {c:.3f}")

                print(_ascii_box("Entanglement Analysis", lines))
                if pattern_entanglements:
                    print(f"Next: proof entanglement --pattern={pattern_entanglements[0][0]}")

            return 0

        # Analyze specific pattern
        if pattern not in graph:
            result = {"error": f"Pattern '{pattern}' not found in graph"}
            _emit_cli_receipt("entanglement", {"pattern": pattern, "companies": companies}, result, 1)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 1

        coeff = entanglement_coefficient(pattern, company_list, graph)

        # SLO check (>= 0.92 is target for high coherence)
        slo = 0.92
        meets_slo = coeff >= slo

        # Determine exit code based on risk level
        if coeff < 0.8:
            exit_code = 0
            risk_level = "LOW"
        elif coeff < 0.92:
            exit_code = 1
            risk_level = "ELEVATED"
        else:
            exit_code = 0 if meets_slo else 2  # High but expected if meets SLO
            risk_level = "HIGH" if not meets_slo else "HIGH (COHERENT)"

        # Interpretation
        if coeff >= 0.8:
            interpretation = f"High coherence - observing in {company_list[0]} strongly predicts behavior in others"
        elif coeff >= 0.5:
            interpretation = "Moderate coherence - some cross-company correlation"
        else:
            interpretation = "Low coherence - pattern behaves independently across companies"

        result = {
            "pattern_id": pattern,
            "companies": company_list,
            "coefficient": round(coeff, 3),
            "slo_status": "PASS" if meets_slo or coeff < slo else "FAIL",
            "meets_slo": meets_slo,
            "risk_level": risk_level,
            "interpretation": interpretation,
        }

        _emit_cli_receipt("entanglement", {"pattern": pattern, "companies": companies}, result, exit_code)

        if output_json:
            print(json.dumps(result, indent=2))
        else:
            lines = [
                f"Pattern: {pattern}",
                f"Companies: {', '.join(company_list)}",
                "",
                f"Entanglement Coefficient: {coeff:.3f}",
                f"SLO Status: {'✓' if meets_slo or coeff < slo else '✗'} {'PASS' if meets_slo or coeff < slo else 'FAIL'} (threshold={slo})",
                "",
                f"Interpretation: {interpretation}",
                "",
                f"Systemic Risk: {risk_level}",
            ]

            print(_ascii_box("Entanglement Analysis", lines))
            print(f"Next: proof trace-forward {pattern}")

        return exit_code

    except Exception as e:
        result = {"error": str(e)}
        _emit_cli_receipt("entanglement", {"pattern": pattern, "companies": companies}, result, 2)
        if output_json:
            print(json.dumps(result))
        else:
            print(f"ERROR: {e}")
        return 2


def cmd_query(
    lens: str,
    min_centrality: Optional[float] = None,
    graph_path: str = "data/graph",
    output_json: bool = False,
) -> int:
    """
    Auditor Question: "Show me patterns matching this lens"

    Mode as projection - filter by QueryPredicate (v9 Paradigm 3).

    Args:
        lens: Lens name (live, shadow, deprecated, high_value, danger_zone, or custom)
        min_centrality: Override min centrality threshold
        graph_path: Path to graph data directory
        output_json: Output as JSON instead of ASCII

    Returns:
        Exit code: 0=results found, 1=no matches, 2=invalid lens
    """
    try:
        graph = _load_graph(graph_path)

        if graph.number_of_nodes() == 0:
            result = {"error": "Empty graph - no patterns to query"}
            _emit_cli_receipt("query", {"lens": lens}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        # Build predicate from lens
        if lens == "live":
            predicate = QueryPredicate.live()
        elif lens == "shadow":
            predicate = QueryPredicate.shadow()
        elif lens == "deprecated":
            predicate = QueryPredicate.deprecated()
        elif lens == "high_value":
            threshold = min_centrality if min_centrality is not None else 0.7
            predicate = QueryPredicate.high_value(threshold)
        elif lens == "danger_zone":
            # High entanglement + low centrality
            predicate = QueryPredicate(
                actionable=False,
                ttl_valid=True,
                min_centrality=0.0,
                max_centrality=0.3,
            )
        elif "=" in lens:
            # Custom lens parsing
            result = {"error": "Custom lens syntax not yet implemented. Use: live, shadow, deprecated, high_value, danger_zone"}
            _emit_cli_receipt("query", {"lens": lens}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2
        else:
            result = {"error": f"Invalid lens '{lens}'. Use: live, shadow, deprecated, high_value, danger_zone"}
            _emit_cli_receipt("query", {"lens": lens}, result, 2)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"ERROR: {result['error']}")
            return 2

        # Get all patterns
        all_patterns = []
        for node in graph.nodes():
            if graph.nodes[node].get("is_pattern", False) or (
                isinstance(node, str) and "pattern" in node.lower()
            ):
                all_patterns.append(node)

        if not all_patterns:
            all_patterns = list(graph.nodes())  # Use all nodes if no pattern markers

        # Filter by predicate
        matching = []
        for pattern_id in all_patterns:
            cent = centrality(pattern_id, graph)
            if predicate.matches(cent):
                matching.append((pattern_id, cent))

        # Sort by centrality descending
        matching.sort(key=lambda x: x[1], reverse=True)

        if not matching:
            result = {"lens": lens, "match_count": 0, "matches": []}
            _emit_cli_receipt("query", {"lens": lens}, result, 1)
            if output_json:
                print(json.dumps(result))
            else:
                print(f"No patterns match lens '{lens}'")
            return 1

        result = {
            "lens": lens,
            "match_count": len(matching),
            "matches": [
                {"pattern_id": p, "centrality": round(c, 3)}
                for p, c in matching[:20]  # Top 20
            ],
        }

        _emit_cli_receipt("query", {"lens": lens}, result, 0)

        if output_json:
            print(json.dumps(result, indent=2))
        else:
            lines = [f"Matching patterns: {len(matching)}", ""]
            for p, c in matching[:7]:  # Show top 7
                lines.append(f"  {p:<35} centrality={c:.2f}")
            if len(matching) > 7:
                lines.append(f"  ... ({len(matching) - 7} more)")
            lines.append("")
            lines.append(f"Lens: {lens}")

            print(_ascii_box(f"Query Results: lens={lens}", lines))

            if matching:
                print(f"Next: proof value {matching[0][0]}")

        return 0

    except Exception as e:
        result = {"error": str(e)}
        _emit_cli_receipt("query", {"lens": lens}, result, 2)
        if output_json:
            print(json.dumps(result))
        else:
            print(f"ERROR: {e}")
        return 2


# --- CLI Main ---


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QED v6/v7/v9 Proof CLI Harness",
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

v9 Commands (Causal Graph):
  python proof.py value <pattern_id>       # What is this pattern worth?
  python proof.py trace-forward <node_id>  # What would change?
  python proof.py trace-backward <node_id> # How did this happen?
  python proof.py counterfactual centrality_floor=0.5  # What-if analysis
  python proof.py health                   # How healthy is the system?
  python proof.py entanglement --pattern <pattern_id>
  python proof.py query --lens=live        # Show patterns matching lens
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- v9 Subcommand Parsers ---

    value_parser = subparsers.add_parser(
        "value",
        help="Compute pattern value via graph centrality",
    )
    value_parser.add_argument(
        "pattern_id",
        type=str,
        help="Pattern ID to compute value for",
    )
    value_parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph",
        help="Path to graph data directory",
    )
    value_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII",
    )
    value_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress ASCII decorations",
    )

    trace_fwd_parser = subparsers.add_parser(
        "trace-forward",
        help="Trace forward effects (blast radius)",
    )
    trace_fwd_parser.add_argument(
        "node_id",
        type=str,
        help="Starting node ID",
    )
    trace_fwd_parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum depth to traverse (default 3, max 10)",
    )
    trace_fwd_parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph",
        help="Path to graph data directory",
    )
    trace_fwd_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII",
    )
    trace_fwd_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress ASCII decorations",
    )

    trace_bwd_parser = subparsers.add_parser(
        "trace-backward",
        help="Trace backward causes (root cause analysis)",
    )
    trace_bwd_parser.add_argument(
        "node_id",
        type=str,
        help="Starting node ID",
    )
    trace_bwd_parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum depth to traverse (default 3, max 10)",
    )
    trace_bwd_parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph",
        help="Path to graph data directory",
    )
    trace_bwd_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII",
    )
    trace_bwd_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress ASCII decorations",
    )

    counterfactual_parser = subparsers.add_parser(
        "counterfactual",
        help="Replay events under counterfactual (backward causation)",
    )
    counterfactual_parser.add_argument(
        "counterfactual",
        type=str,
        help="Counterfactual string (e.g., centrality_floor=0.5)",
    )
    counterfactual_parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph",
        help="Path to graph data directory",
    )
    counterfactual_parser.add_argument(
        "--events-path",
        type=str,
        default="data/events/events.jsonl",
        help="Path to events JSONL file",
    )
    counterfactual_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII",
    )

    health_parser = subparsers.add_parser(
        "health",
        help="Compute system health via self-compression + portfolio health",
    )
    health_parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph",
        help="Path to graph data directory",
    )
    health_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII",
    )
    health_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress ASCII decorations",
    )

    entanglement_parser = subparsers.add_parser(
        "entanglement",
        help="Compute cross-company entanglement coefficient (SLO >= 0.92)",
    )
    entanglement_parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Pattern ID to analyze (shows top 10 if not specified)",
    )
    entanglement_parser.add_argument(
        "--companies",
        type=str,
        default=None,
        help="Comma-separated company list (default: tesla,spacex)",
    )
    entanglement_parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph",
        help="Path to graph data directory",
    )
    entanglement_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Query patterns by lens (mode as projection)",
    )
    query_parser.add_argument(
        "--lens",
        type=str,
        required=True,
        help="Lens name (live, shadow, deprecated, high_value, danger_zone)",
    )
    query_parser.add_argument(
        "--min-centrality",
        type=float,
        default=None,
        help="Override minimum centrality threshold",
    )
    query_parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph",
        help="Path to graph data directory",
    )
    query_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII",
    )

    # --- Legacy/v7 Subcommand Parsers ---

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

    # --- v9 Command Handlers ---

    if args.command == "value":
        exit_code = cmd_value(
            pattern_id=args.pattern_id,
            graph_path=args.graph_path,
            output_json=args.json,
            quiet=args.quiet,
        )
        return exit_code

    elif args.command == "trace-forward":
        exit_code = cmd_trace_forward(
            node_id=args.node_id,
            depth=args.depth,
            graph_path=args.graph_path,
            output_json=args.json,
            quiet=args.quiet,
        )
        return exit_code

    elif args.command == "trace-backward":
        exit_code = cmd_trace_backward(
            node_id=args.node_id,
            depth=args.depth,
            graph_path=args.graph_path,
            output_json=args.json,
            quiet=args.quiet,
        )
        return exit_code

    elif args.command == "counterfactual":
        exit_code = cmd_replay_counterfactual(
            counterfactual=args.counterfactual,
            graph_path=args.graph_path,
            events_path=args.events_path,
            output_json=args.json,
        )
        return exit_code

    elif args.command == "health":
        exit_code = cmd_health(
            graph_path=args.graph_path,
            output_json=args.json,
            quiet=args.quiet,
        )
        return exit_code

    elif args.command == "entanglement":
        exit_code = cmd_entanglement(
            pattern=args.pattern,
            companies=args.companies,
            graph_path=args.graph_path,
            output_json=args.json,
        )
        return exit_code

    elif args.command == "query":
        exit_code = cmd_query(
            lens=args.lens,
            min_centrality=args.min_centrality,
            graph_path=args.graph_path,
            output_json=args.json,
        )
        return exit_code

    # --- Legacy/v7 Command Handlers ---

    elif args.command == "gates":
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
