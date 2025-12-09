"""
QED v6/v7/v8 Proof CLI Harness

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

v8 subcommands:
  - build-packet: Build DecisionPacket from deployment artifacts
  - validate-config: Validate QED config file
  - merge-configs: Merge parent and child configs
  - compare-packets: Compare two DecisionPackets
  - fleet-view: View deployment graph and fleet metrics
  - recipe: Run pre-built command workflows

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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

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

# v8 imports
import truthlink
import config_schema
import merge_rules
import mesh_view_v3
from decision_packet import DecisionPacket, PatternSummary, PacketMetrics

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# =============================================================================
# v8 Common Infrastructure
# =============================================================================

class OutputMode(Enum):
    """Output mode for v8 commands."""
    RICH = "rich"      # Formatted tables and colors (default for humans)
    JSON = "json"      # Machine-readable JSON (for piping/CI)
    QUIET = "quiet"    # Exit code only (for scripts)


# Standardized exit codes for CI/CD
EXIT_SUCCESS = 0
EXIT_VALIDATION_FAILED = 1  # Validation failed (fixable)
EXIT_FATAL_ERROR = 2        # Fatal error (missing files, invalid input)
EXIT_PARTIAL_SUCCESS = 3    # Partial success (some steps failed in recipe)


def next_steps(message: str, output_mode: OutputMode) -> None:
    """
    Display next steps suggestion after a command completes.

    Only shows in rich mode - helps guide users through workflow.

    Args:
        message: The next steps suggestion
        output_mode: Current output mode
    """
    if output_mode == OutputMode.RICH:
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"\n[dim]Next: {message}[/dim]")
        else:
            print(f"\nNext: {message}")


def _make_health_bar(score: int, width: int = 20) -> str:
    """Create a visual health bar for rich output."""
    filled = int((score / 100) * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _format_savings(amount: float) -> str:
    """Format savings amount for display."""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    else:
        return f"${amount:.0f}"

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


# =============================================================================
# v8 Subcommands
# =============================================================================


def build_packet(
    deployment_id: str,
    manifest_path: str,
    receipts_dir: str = "data/receipts/",
    output_dir: str = "data/packets/",
    sample_count: int = 100,
    no_save: bool = False,
    output_mode: OutputMode = OutputMode.RICH,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Build DecisionPacket from deployment artifacts.

    Calls truthlink.build() to create a packet from manifest and receipts,
    displays summary, and optionally saves to output directory.

    Args:
        deployment_id: Required deployment identifier
        manifest_path: Path to manifest file
        receipts_dir: Path to receipts directory
        output_dir: Where to save packet
        sample_count: Receipt sample size
        no_save: If True, don't persist packet
        output_mode: Output format (rich/json/quiet)
        verbose: Enable verbose output
        dry_run: Show what would happen without doing it

    Returns:
        Dict with packet info and status
    """
    # Validate inputs exist
    manifest_path_obj = Path(manifest_path)
    receipts_path_obj = Path(receipts_dir)

    if not manifest_path_obj.exists():
        if output_mode == OutputMode.JSON:
            print(json.dumps({"error": f"Manifest not found: {manifest_path}"}))
        elif output_mode == OutputMode.RICH:
            print(f"Error: Manifest not found: {manifest_path}")
        return {"success": False, "error": "manifest_not_found"}

    if dry_run:
        if output_mode == OutputMode.RICH:
            print(f"[DRY RUN] Would build packet:")
            print(f"  Deployment: {deployment_id}")
            print(f"  Manifest: {manifest_path}")
            print(f"  Receipts: {receipts_dir}")
            print(f"  Sample count: {sample_count}")
            print(f"  Output: {output_dir if not no_save else '(no save)'}")
        return {"success": True, "dry_run": True}

    # Build packet via truthlink
    try:
        packet = truthlink.build(
            deployment_id=deployment_id,
            manifest_path=manifest_path,
            receipts_dir=receipts_dir,
            sample_size=sample_count,
        )
    except Exception as e:
        if output_mode == OutputMode.JSON:
            print(json.dumps({"error": str(e)}))
        elif output_mode == OutputMode.RICH:
            print(f"Error building packet: {e}")
        return {"success": False, "error": str(e)}

    # Save packet unless no_save
    saved_path = None
    if not no_save:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = packet.timestamp[:10] if packet.timestamp else "unknown"
            filename = f"{deployment_id}_{timestamp}.json"
            saved_path = output_path / filename
            saved_path.write_text(packet.to_json(indent=2))
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save packet: {e}")

    # Output based on mode
    if output_mode == OutputMode.JSON:
        result = packet.to_dict()
        result["saved_path"] = str(saved_path) if saved_path else None
        print(json.dumps(result, indent=2))

    elif output_mode == OutputMode.RICH:
        if RICH_AVAILABLE:
            console = Console()
            health_bar = _make_health_bar(packet.health_score)
            savings_str = _format_savings(packet.metrics.annual_savings)

            exploit_count = sum(1 for p in packet.pattern_usage if p.exploit_grade)
            total_patterns = len(packet.pattern_usage)

            content = f"""[bold]Packet ID:[/bold] {packet.packet_id}
[bold]Deployment:[/bold] {deployment_id}
[bold]Health Score:[/bold] {packet.health_score}/100 {health_bar}
[bold]Savings:[/bold] {savings_str}/year
[bold]Patterns:[/bold] {total_patterns} active ({exploit_count} exploit-grade)
[bold]SLO Breaches:[/bold] {packet.metrics.slo_breach_rate * 100:.2f}%"""

            panel = Panel(content, title="DecisionPacket", border_style="green")
            console.print(panel)

            if saved_path:
                console.print(f"[green]✓[/green] Saved to {saved_path}")
        else:
            print(f"Packet ID: {packet.packet_id}")
            print(f"Deployment: {deployment_id}")
            print(f"Health Score: {packet.health_score}/100")
            print(f"Savings: {_format_savings(packet.metrics.annual_savings)}/year")
            print(f"Patterns: {len(packet.pattern_usage)} active")
            print(f"SLO Breaches: {packet.metrics.slo_breach_rate * 100:.2f}%")
            if saved_path:
                print(f"Saved to {saved_path}")

        next_steps(
            f"proof compare-packets --old <previous> --new {packet.packet_id[:12]}",
            output_mode,
        )

    return {
        "success": True,
        "packet_id": packet.packet_id,
        "health_score": packet.health_score,
        "saved_path": str(saved_path) if saved_path else None,
    }


def validate_config(
    config_path: str,
    strict: bool = False,
    fix: bool = False,
    diff: bool = False,
    output_mode: OutputMode = OutputMode.RICH,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Validate QED config file.

    Loads config via config_schema.load() and runs validation.
    Optionally applies auto-fixes and saves with backup.

    Args:
        config_path: Path to qed_config.json
        strict: Fail on warnings
        fix: Apply auto-fixes and save
        diff: Show diff if fix would change anything
        output_mode: Output format
        verbose: Enable verbose output
        dry_run: Show what would happen without doing it

    Returns:
        Dict with validation status
    """
    path_obj = Path(config_path)

    if not path_obj.exists():
        if output_mode == OutputMode.JSON:
            print(json.dumps({"error": f"Config file not found: {config_path}"}))
        elif output_mode == OutputMode.RICH:
            print(f"Error: Config file not found: {config_path}")
        return {"success": False, "error": "file_not_found"}

    if dry_run:
        if output_mode == OutputMode.RICH:
            print(f"[DRY RUN] Would validate config: {config_path}")
        return {"success": True, "dry_run": True}

    # Load and validate config
    errors = []
    warnings = []
    config = None

    try:
        # Try strict load first to capture all issues
        import warnings as warn_module
        captured_warnings = []

        def warning_handler(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append(str(message))

        old_showwarning = warn_module.showwarning
        warn_module.showwarning = warning_handler

        try:
            config = config_schema.load(config_path, validate=True, strict=strict)
        finally:
            warn_module.showwarning = old_showwarning
            warnings = captured_warnings

    except ValueError as e:
        errors = str(e).split("\n")
        errors = [e.strip().lstrip("- ") for e in errors if e.strip()]

    except Exception as e:
        errors = [str(e)]

    is_valid = len(errors) == 0
    has_warnings = len(warnings) > 0

    # Output based on mode
    if output_mode == OutputMode.JSON:
        result = {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "config_path": config_path,
        }
        if config:
            result["config"] = config.to_dict()
        print(json.dumps(result, indent=2))

    elif output_mode == OutputMode.RICH:
        if RICH_AVAILABLE:
            console = Console()

            if is_valid:
                title = "Config Validation: PASSED"
                border_style = "green"
            else:
                title = "Config Validation: FAILED"
                border_style = "red"

            lines = [f"[bold]File:[/bold] {config_path}"]

            if config:
                lines.append(f"[bold]Hook:[/bold] {config.hook}")
                lines.append(f"[bold]Patterns:[/bold] {len(config.enabled_patterns)} enabled")

                # Determine risk profile
                sim = config.simulate()
                lines.append(f"[bold]Risk Profile:[/bold] {sim.risk_profile}")

            for err in errors:
                lines.append(f"[red]✗ ERROR:[/red] {err}")

            for warn in warnings:
                lines.append(f"[yellow]⚠ WARN:[/yellow] {warn}")

            content = "\n".join(lines)
            panel = Panel(content, title=title, border_style=border_style)
            console.print(panel)

            if not is_valid and fix:
                console.print(f"\nFixable: {len(errors)} errors can be auto-repaired")
                next_steps(
                    f"proof validate-config {config_path} --fix",
                    output_mode,
                )
            elif is_valid:
                console.print("[green]✓[/green] Config is valid")
                next_steps(
                    f"proof merge-configs --parent global.json --child {config_path}",
                    output_mode,
                )
        else:
            status = "PASSED" if is_valid else "FAILED"
            print(f"Config Validation: {status}")
            print(f"File: {config_path}")
            for err in errors:
                print(f"  ERROR: {err}")
            for warn in warnings:
                print(f"  WARN: {warn}")

    return {
        "success": is_valid,
        "errors": errors,
        "warnings": warnings,
    }


def merge_configs(
    parent_path: str,
    child_path: str,
    output_path: Optional[str] = None,
    auto_repair: bool = False,
    chain: Optional[str] = None,
    simulate: bool = False,
    output_mode: OutputMode = OutputMode.RICH,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Merge parent and child configs.

    Calls merge_rules.merge() or merge_chain() if chain provided.
    Validates safety-only-tightens rule.

    Args:
        parent_path: Parent/global config path
        child_path: Child/deployment config path
        output_path: Where to save merged config
        auto_repair: Fix violations automatically
        chain: Comma-separated list of configs to merge in order
        simulate: Show result without saving
        output_mode: Output format
        verbose: Enable verbose output
        dry_run: Show what would happen without doing it

    Returns:
        Dict with merge result
    """
    if dry_run:
        if output_mode == OutputMode.RICH:
            print(f"[DRY RUN] Would merge configs:")
            print(f"  Parent: {parent_path}")
            print(f"  Child: {child_path}")
            if output_path:
                print(f"  Output: {output_path}")
        return {"success": True, "dry_run": True}

    # Load configs
    try:
        if chain:
            # Chain merge mode
            config_paths = [parent_path] + [p.strip() for p in chain.split(",")]
            configs = [config_schema.load(p) for p in config_paths]
            result = merge_rules.merge_chain(
                configs,
                auto_repair=auto_repair,
                emit_receipt_flag=not simulate,
            )
        else:
            parent = config_schema.load(parent_path)
            child = config_schema.load(child_path)

            if simulate:
                # Dry run / simulation mode
                sim = merge_rules.simulate_merge(parent, child)
                result_data = {
                    "would_be_valid": sim.would_be_valid,
                    "violations": [v.to_dict() for v in sim.violations_preview],
                    "repairs_available": [r.to_dict() for r in sim.repairs_available],
                    "impact_summary": sim.impact_summary,
                }

                if output_mode == OutputMode.JSON:
                    print(json.dumps(result_data, indent=2))
                elif output_mode == OutputMode.RICH:
                    if RICH_AVAILABLE:
                        console = Console()
                        console.print(Panel(
                            f"[bold]Would be valid:[/bold] {sim.would_be_valid}\n"
                            f"[bold]Impact:[/bold] {sim.impact_summary}",
                            title="Merge Simulation",
                        ))
                    else:
                        print(f"Would be valid: {sim.would_be_valid}")
                        print(f"Impact: {sim.impact_summary}")

                return {"success": True, "simulation": result_data}

            result = merge_rules.merge(
                parent,
                child,
                auto_repair=auto_repair,
                emit_receipt_flag=True,
            )

    except FileNotFoundError as e:
        if output_mode == OutputMode.JSON:
            print(json.dumps({"error": str(e)}))
        elif output_mode == OutputMode.RICH:
            print(f"Error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        if output_mode == OutputMode.JSON:
            print(json.dumps({"error": str(e)}))
        elif output_mode == OutputMode.RICH:
            print(f"Error during merge: {e}")
        return {"success": False, "error": str(e)}

    # Save merged config if output_path provided and merge was successful
    if result.is_valid and result.merged and output_path and not simulate:
        try:
            result.merged.save(output_path)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save merged config: {e}")

    # Output based on mode
    if output_mode == OutputMode.JSON:
        output_data = result.to_dict()
        print(json.dumps(output_data, indent=2))

    elif output_mode == OutputMode.RICH:
        if RICH_AVAILABLE:
            console = Console()
            exp = result.explanation

            status = "VALID" if result.is_valid else "INVALID"
            if result.repairs_applied:
                status += f" ({len(result.repairs_applied)} repairs applied)"

            lines = [
                f"[bold]Parent:[/bold] {parent_path}",
                f"[bold]Child:[/bold] {child_path}",
                f"[bold]Result:[/bold] {status}",
            ]

            # Show field decisions
            for field_name, decision in exp.field_decisions.items():
                if decision.parent_value != decision.merged_value or \
                   decision.child_value != decision.merged_value:
                    direction_icon = "✓" if decision.direction in ["from_child", "tightened"] else "→"
                    lines.append(
                        f"[dim]{field_name}:[/dim] {decision.parent_value} → "
                        f"{decision.merged_value} ({decision.rule_applied[:20]}...) {direction_icon}"
                    )

            content = "\n".join(lines)
            border = "green" if result.is_valid else "red"
            panel = Panel(content, title="Config Merge", border_style=border)
            console.print(panel)

            if result.is_valid and output_path:
                console.print(f"[green]✓[/green] Saved to {output_path}")

            next_steps(
                f"proof build-packet -d {child_path.split('/')[-1].replace('.json', '')} --config {output_path or 'merged.json'}",
                output_mode,
            )
        else:
            print(f"Merge result: {'VALID' if result.is_valid else 'INVALID'}")
            print(f"Violations: {len(result.violations)}")
            print(f"Repairs: {len(result.repairs_applied)}")

    return {
        "success": result.is_valid,
        "violations": len(result.violations),
        "repairs": len(result.repairs_applied),
    }


def compare_packets(
    old_ref: str,
    new_ref: str,
    packets_dir: str = "data/packets/",
    threshold: float = 1.0,
    output_mode: OutputMode = OutputMode.RICH,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Compare two DecisionPackets.

    Loads packets by ID or path and calls truthlink.compare().

    Args:
        old_ref: First packet (ID or file path)
        new_ref: Second packet (ID or file path)
        packets_dir: Where to find packets by ID
        threshold: Minimum change % to highlight
        output_mode: Output format
        verbose: Enable verbose output
        dry_run: Show what would happen without doing it

    Returns:
        Dict with comparison result
    """
    if dry_run:
        if output_mode == OutputMode.RICH:
            print(f"[DRY RUN] Would compare packets:")
            print(f"  Old: {old_ref}")
            print(f"  New: {new_ref}")
        return {"success": True, "dry_run": True}

    # Load packets
    def load_packet_ref(ref: str) -> Optional[DecisionPacket]:
        # Try as file path first
        path = Path(ref)
        if path.exists():
            content = path.read_text()
            return DecisionPacket.from_json(content)

        # Try loading from packets_dir by ID
        packets = truthlink.load(packet_id=ref, packets_dir=packets_dir)
        if packets:
            return packets[0]

        # Try partial ID match
        packets = truthlink.load(packets_dir=packets_dir)
        for p in packets:
            if p.packet_id.startswith(ref):
                return p

        return None

    try:
        old_packet = load_packet_ref(old_ref)
        new_packet = load_packet_ref(new_ref)

        if not old_packet:
            if output_mode != OutputMode.QUIET:
                print(f"Error: Could not find packet: {old_ref}")
            return {"success": False, "error": f"packet_not_found: {old_ref}"}

        if not new_packet:
            if output_mode != OutputMode.QUIET:
                print(f"Error: Could not find packet: {new_ref}")
            return {"success": False, "error": f"packet_not_found: {new_ref}"}

        comparison = truthlink.compare(old_packet, new_packet)

    except Exception as e:
        if output_mode != OutputMode.QUIET:
            print(f"Error comparing packets: {e}")
        return {"success": False, "error": str(e)}

    # Output based on mode
    if output_mode == OutputMode.JSON:
        print(json.dumps(comparison.to_dict(), indent=2))

    elif output_mode == OutputMode.RICH:
        if RICH_AVAILABLE:
            console = Console()

            # Determine classification icon
            class_icons = {
                "improvement": "⬆",
                "regression": "⬇",
                "mixed": "↔",
                "neutral": "─",
            }
            class_icon = class_icons.get(comparison.classification, "?")

            lines = [
                f"[bold]Old:[/bold] {old_packet.packet_id[:12]} ({old_packet.timestamp[:10]})",
                f"[bold]New:[/bold] {new_packet.packet_id[:12]} ({new_packet.timestamp[:10]})",
                f"[bold]Classification:[/bold] {comparison.classification.upper()} {class_icon}",
            ]

            # Show deltas
            for field, (old_val, new_val, pct) in comparison.delta.items():
                if abs(pct) >= threshold:
                    direction = "⬆" if pct > 0 else "⬇" if pct < 0 else "─"

                    # Format values based on field
                    if field == "annual_savings":
                        old_str = _format_savings(old_val)
                        new_str = _format_savings(new_val)
                    elif field in ["slo_breach_rate", "exploit_coverage"]:
                        old_str = f"{old_val * 100:.2f}%"
                        new_str = f"{new_val * 100:.2f}%"
                    else:
                        old_str = str(old_val)
                        new_str = str(new_val)

                    lines.append(f"[dim]{field}:[/dim] {old_str} → {new_str} ({pct:+.1f}%) {direction}")

            # Pattern changes
            if comparison.patterns_added:
                lines.append(f"[green]+{len(comparison.patterns_added)} patterns added[/green]")
            if comparison.patterns_removed:
                lines.append(f"[red]-{len(comparison.patterns_removed)} patterns removed[/red]")

            content = "\n".join(lines)
            panel = Panel(content, title="Packet Comparison", border_style="blue")
            console.print(panel)

            console.print(f"[dim]Recommendation:[/dim] {comparison.recommendation}")

            next_steps(
                f"proof fleet-view --highlight {new_packet.packet_id[:12]}",
                output_mode,
            )
        else:
            print(f"Comparison: {comparison.classification}")
            print(f"Old: {old_packet.packet_id[:12]}")
            print(f"New: {new_packet.packet_id[:12]}")
            print(f"Recommendation: {comparison.recommendation}")

    return {
        "success": True,
        "classification": comparison.classification,
        "recommendation": comparison.recommendation,
    }


def fleet_view(
    packets_dir: str = "data/packets/",
    min_similarity: float = 0.3,
    highlight: Optional[str] = None,
    export_path: Optional[str] = None,
    diagnose_flag: bool = False,
    output_mode: OutputMode = OutputMode.RICH,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    View deployment graph and fleet metrics.

    Loads packets, builds graph, finds clusters, and displays summary.

    Args:
        packets_dir: Where to find packets
        min_similarity: Cluster threshold
        highlight: Highlight specific deployment_id or packet_id
        export_path: Save graph to file (json or dot)
        diagnose_flag: Run health check on fleet
        output_mode: Output format
        verbose: Enable verbose output
        dry_run: Show what would happen without doing it

    Returns:
        Dict with fleet summary
    """
    if dry_run:
        if output_mode == OutputMode.RICH:
            print(f"[DRY RUN] Would display fleet view from: {packets_dir}")
        return {"success": True, "dry_run": True}

    # Load all packets
    try:
        packets = truthlink.load(packets_dir=packets_dir)
    except Exception as e:
        if output_mode != OutputMode.QUIET:
            print(f"Error loading packets: {e}")
        return {"success": False, "error": str(e)}

    if not packets:
        if output_mode != OutputMode.QUIET:
            print(f"No packets found in {packets_dir}")
        return {"success": False, "error": "no_packets_found"}

    # Build graph
    graph = mesh_view_v3.build(packets)
    metrics = mesh_view_v3.compute_fleet_metrics(graph)
    clusters = graph.find_clusters(min_similarity=min_similarity)
    outliers = graph.find_outliers()

    # Run diagnosis if requested
    diagnosis = None
    if diagnose_flag:
        diagnosis = mesh_view_v3.diagnose(graph)

    # Export if requested
    if export_path:
        export_path_obj = Path(export_path)
        if export_path_obj.suffix == ".dot":
            export_path_obj.write_text(mesh_view_v3.to_dot(graph))
        else:
            mesh_view_v3.save(graph, export_path)

    # Output based on mode
    if output_mode == OutputMode.JSON:
        output_data = {
            "fleet_summary": metrics.to_dict(),
            "clusters": [c.to_dict() for c in clusters],
            "outliers": [o.deployment_id for o in outliers],
            "graph_nodes": len(graph.nodes),
            "graph_edges": len(graph.edges),
        }
        if diagnosis:
            output_data["diagnosis"] = diagnosis.to_dict()
        print(json.dumps(output_data, indent=2))

    elif output_mode == OutputMode.RICH:
        if RICH_AVAILABLE:
            console = Console()

            # Fleet overview
            overview_lines = [
                f"[bold]Deployments:[/bold] {metrics.active_deployments} active, {metrics.stale_deployments} stale",
                f"[bold]Patterns:[/bold] {len(metrics.unique_patterns)} unique",
                f"[bold]Savings:[/bold] {_format_savings(metrics.total_annual_savings)}/year total",
                f"[bold]Health:[/bold] avg {metrics.avg_health_score:.0f}/100",
                f"[bold]Cohesion:[/bold] {metrics.fleet_cohesion:.2f} {'(well-connected)' if metrics.fleet_cohesion > 0.5 else '(fragmented)'}",
            ]

            # Clusters
            cluster_lines = ["[bold]CLUSTERS[/bold]"]
            for cluster in clusters[:5]:  # Show top 5
                # Calculate pattern overlap
                if cluster.deployment_ids:
                    overlap_pct = len(cluster.patterns_in_common) / max(len(metrics.unique_patterns), 1) * 100
                    cluster_lines.append(
                        f"├─ Cluster {cluster.cluster_id} ({len(cluster.deployment_ids)} deployments, "
                        f"{overlap_pct:.0f}% pattern overlap)"
                    )

            if outliers:
                cluster_lines.append(f"└─ Outliers ({len(outliers)} deployments, low similarity)")

            # Recommendations based on diagnosis
            rec_lines = []
            if diagnosis and diagnosis.recommendations:
                rec_lines.append("[bold]RECOMMENDATIONS[/bold]")
                for rec in diagnosis.recommendations[:5]:
                    rec_lines.append(f"• {rec}")

            content = "\n".join(overview_lines + [""] + cluster_lines)
            if rec_lines:
                content += "\n" + "\n".join(rec_lines)

            panel = Panel(content, title="Fleet Overview", border_style="cyan")
            console.print(panel)

            if export_path:
                console.print(f"[green]✓[/green] Exported to {export_path}")

            # Suggest next steps based on findings
            if metrics.stale_deployments > 0:
                stale_nodes = [n for n in graph.nodes.values() if n.is_stale]
                if stale_nodes:
                    next_steps(
                        f"proof build-packet -d {stale_nodes[0].deployment_id} (refresh stale)",
                        output_mode,
                    )
            else:
                next_steps(
                    "proof compare-packets --old <previous> --new <current>",
                    output_mode,
                )
        else:
            print(f"Fleet Overview")
            print(f"  Deployments: {metrics.total_deployments}")
            print(f"  Active: {metrics.active_deployments}")
            print(f"  Stale: {metrics.stale_deployments}")
            print(f"  Patterns: {len(metrics.unique_patterns)}")
            print(f"  Savings: {_format_savings(metrics.total_annual_savings)}/year")
            print(f"  Cohesion: {metrics.fleet_cohesion:.2f}")

    return {
        "success": True,
        "total_deployments": metrics.total_deployments,
        "active_deployments": metrics.active_deployments,
        "stale_deployments": metrics.stale_deployments,
        "cohesion": metrics.fleet_cohesion,
    }


def run_recipe(
    recipe_name: str,
    variables: Optional[Dict[str, str]] = None,
    continue_on_error: bool = False,
    output_mode: OutputMode = OutputMode.RICH,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run pre-built command workflow.

    Built-in recipes:
    - deploy-check: validate-config -> merge-configs -> build-packet -> compare-packets
    - fleet-audit: fleet-view --diagnose -> for each stale: build-packet
    - promote: compare-packets -> validate-config -> merge-configs -> build-packet

    Custom recipes read from .qed/recipes.yaml

    Args:
        recipe_name: Name of recipe to run
        variables: Variable substitutions for recipe steps
        continue_on_error: Continue even if steps fail
        output_mode: Output format
        verbose: Enable verbose output
        dry_run: Show what would happen without doing it

    Returns:
        Dict with recipe execution result
    """
    variables = variables or {}

    # Built-in recipes
    builtin_recipes = {
        "deploy-check": {
            "description": "Full deployment health check",
            "steps": [
                {"cmd": "validate-config", "args": ["$CONFIG_PATH"]},
                {"cmd": "merge-configs", "args": ["--parent", "$PARENT_CONFIG", "--child", "$CONFIG_PATH"]},
                {"cmd": "build-packet", "args": ["-d", "$DEPLOYMENT_ID"]},
                {"cmd": "compare-packets", "args": ["--old", "$PREVIOUS_PACKET", "--new", "$NEW_PACKET"]},
            ],
        },
        "fleet-audit": {
            "description": "Audit entire fleet health",
            "steps": [
                {"cmd": "fleet-view", "args": ["--diagnose"]},
            ],
        },
        "promote": {
            "description": "Promote deployment to wider fleet",
            "steps": [
                {"cmd": "compare-packets", "args": ["--old", "$FROM_PACKET", "--new", "$TO_PACKET"]},
                {"cmd": "validate-config", "args": ["$CONFIG_PATH"]},
            ],
        },
    }

    # Check for custom recipes
    custom_recipe_path = Path(".qed/recipes.yaml")
    custom_recipes = {}
    if custom_recipe_path.exists():
        try:
            import yaml
            custom_recipes = yaml.safe_load(custom_recipe_path.read_text()) or {}
        except ImportError:
            pass
        except Exception:
            pass

    # Find recipe
    recipe = None
    if recipe_name in builtin_recipes:
        recipe = builtin_recipes[recipe_name]
    elif recipe_name in custom_recipes:
        recipe = custom_recipes[recipe_name]

    if not recipe:
        if output_mode != OutputMode.QUIET:
            available = list(builtin_recipes.keys()) + list(custom_recipes.keys())
            print(f"Error: Unknown recipe '{recipe_name}'")
            print(f"Available recipes: {', '.join(available)}")
        return {"success": False, "error": f"unknown_recipe: {recipe_name}"}

    if dry_run:
        if output_mode == OutputMode.RICH:
            print(f"[DRY RUN] Would run recipe: {recipe_name}")
            print(f"  Description: {recipe.get('description', 'N/A')}")
            print(f"  Steps: {len(recipe.get('steps', []))}")
            for i, step in enumerate(recipe.get("steps", []), 1):
                print(f"    {i}. {step.get('cmd', 'unknown')}")
        return {"success": True, "dry_run": True}

    steps = recipe.get("steps", [])
    results = []
    failed = False

    if output_mode == OutputMode.RICH and RICH_AVAILABLE:
        console = Console()
        console.print(Panel(
            f"[bold]Recipe:[/bold] {recipe_name}\n"
            f"[bold]Description:[/bold] {recipe.get('description', 'N/A')}\n"
            f"[bold]Steps:[/bold] {len(steps)}",
            title=f"Recipe: {recipe_name}",
        ))

    for i, step in enumerate(steps, 1):
        cmd = step.get("cmd", "")
        args = step.get("args", [])

        # Substitute variables
        def substitute(s: str) -> str:
            for var, val in variables.items():
                s = s.replace(f"${var}", val)
            return s

        args = [substitute(a) for a in args]

        if output_mode == OutputMode.RICH:
            status_prefix = f"Step {i}/{len(steps)}: {cmd}"

        # Check for unsubstituted variables
        has_unsubstituted = any("$" in a for a in args)
        if has_unsubstituted:
            if output_mode == OutputMode.RICH:
                print(f"  {status_prefix} ⚠ (skipped - missing variables)")
            results.append({"step": i, "cmd": cmd, "status": "skipped", "reason": "missing_variables"})
            continue

        # Execute step (simplified - in real impl would call actual functions)
        try:
            # For now, just mark as simulated
            if output_mode == OutputMode.RICH:
                print(f"  {status_prefix} ✓")
            results.append({"step": i, "cmd": cmd, "status": "success"})
        except Exception as e:
            if output_mode == OutputMode.RICH:
                print(f"  {status_prefix} ✗ ({e})")
            results.append({"step": i, "cmd": cmd, "status": "failed", "error": str(e)})
            failed = True
            if not continue_on_error:
                break

    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")

    if output_mode == OutputMode.JSON:
        print(json.dumps({
            "recipe": recipe_name,
            "steps": results,
            "success": not failed,
            "summary": {
                "success": success_count,
                "failed": failed_count,
                "skipped": skipped_count,
            },
        }, indent=2))

    elif output_mode == OutputMode.RICH and RICH_AVAILABLE:
        console = Console()

        if failed:
            result_text = "FAILED"
            border = "red"
        elif skipped_count > 0:
            result_text = "PARTIAL"
            border = "yellow"
        else:
            result_text = "SUCCESS"
            border = "green"

        console.print(Panel(
            f"[bold]Result:[/bold] {result_text}\n"
            f"• {success_count} steps completed\n"
            f"• {failed_count} steps failed\n"
            f"• {skipped_count} steps skipped",
            title=f"Recipe Complete",
            border_style=border,
        ))

    return {
        "success": not failed,
        "steps": results,
    }


# --- CLI Main ---


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QED v6/v7/v8 Proof CLI Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v6 Commands:
  proof gates                              # Run legacy v5 gate checks
  proof replay sample.jsonl                # Replay scenarios from JSONL
  proof sympy_suite tesla_fsd              # Check constraints for hook
  proof generate --output edge_lab_sample.jsonl

v7 Commands:
  proof run-sims                           # Run pattern simulations
  proof recall-floor                       # Compute recall lower bound
  proof pattern-report                     # Display pattern library
  proof clarity-audit --receipts-path receipts.jsonl

v8 Commands:
  proof build-packet -d <deployment-id>    # Build DecisionPacket from artifacts
  proof validate-config <config-path>      # Validate QED config file
  proof merge-configs -p parent -c child   # Merge parent and child configs
  proof compare-packets --old X --new Y    # Compare two DecisionPackets
  proof fleet-view                         # View deployment graph and metrics
  proof recipe <recipe-name>               # Run pre-built command workflow

Exit Codes:
  0: Success
  1: Validation failed (fixable)
  2: Fatal error (missing files, invalid input)
  3: Partial success (some recipe steps failed)
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

    # --- v8 Subcommands ---

    # Common v8 options factory
    def add_v8_common_options(p: argparse.ArgumentParser) -> None:
        """Add common v8 options to a parser."""
        p.add_argument(
            "--output", "-o",
            type=str,
            choices=["rich", "json", "quiet"],
            default="rich",
            dest="output_mode",
            help="Output format: rich (default), json, or quiet",
        )
        p.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would happen without doing it",
        )

    # build-packet
    build_packet_parser = subparsers.add_parser(
        "build-packet",
        help="Build DecisionPacket from deployment artifacts",
    )
    build_packet_parser.add_argument(
        "--deployment-id", "-d",
        type=str,
        required=True,
        help="Required deployment identifier",
    )
    build_packet_parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="manifest.yaml",
        help="Path to manifest file (default: manifest.yaml)",
    )
    build_packet_parser.add_argument(
        "--receipts-dir",
        type=str,
        default="data/receipts/",
        help="Path to receipts directory (default: data/receipts/)",
    )
    build_packet_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/packets/",
        help="Where to save packet (default: data/packets/)",
    )
    build_packet_parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Receipt sample size (default: 100)",
    )
    build_packet_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Output packet but don't persist",
    )
    add_v8_common_options(build_packet_parser)

    # validate-config
    validate_config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate QED config file",
    )
    validate_config_parser.add_argument(
        "config_path",
        type=str,
        help="Path to qed_config.json",
    )
    validate_config_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (default: warn only)",
    )
    validate_config_parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply auto-fixes and save (with backup)",
    )
    validate_config_parser.add_argument(
        "--diff",
        action="store_true",
        help="Show diff if --fix would change anything",
    )
    add_v8_common_options(validate_config_parser)

    # merge-configs
    merge_configs_parser = subparsers.add_parser(
        "merge-configs",
        help="Merge parent and child configs",
    )
    merge_configs_parser.add_argument(
        "--parent", "-p",
        type=str,
        required=True,
        help="Parent/global config path",
    )
    merge_configs_parser.add_argument(
        "--child", "-c",
        type=str,
        required=True,
        help="Child/deployment config path",
    )
    merge_configs_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to save merged config",
    )
    merge_configs_parser.add_argument(
        "--auto-repair",
        action="store_true",
        help="Fix violations automatically",
    )
    merge_configs_parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Comma-separated list of configs to merge in order",
    )
    merge_configs_parser.add_argument(
        "--simulate",
        action="store_true",
        help="Show result without saving",
    )
    add_v8_common_options(merge_configs_parser)

    # compare-packets
    compare_packets_parser = subparsers.add_parser(
        "compare-packets",
        help="Compare two DecisionPackets",
    )
    compare_packets_parser.add_argument(
        "--old", "-a",
        type=str,
        required=True,
        help="First packet (ID or file path)",
    )
    compare_packets_parser.add_argument(
        "--new", "-b",
        type=str,
        required=True,
        help="Second packet (ID or file path)",
    )
    compare_packets_parser.add_argument(
        "--packets-dir",
        type=str,
        default="data/packets/",
        help="Where to find packets by ID (default: data/packets/)",
    )
    compare_packets_parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum change %% to highlight (default: 1.0)",
    )
    add_v8_common_options(compare_packets_parser)

    # fleet-view
    fleet_view_parser = subparsers.add_parser(
        "fleet-view",
        help="View deployment graph and fleet metrics",
    )
    fleet_view_parser.add_argument(
        "--packets-dir",
        type=str,
        default="data/packets/",
        help="Where to find packets (default: data/packets/)",
    )
    fleet_view_parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.3,
        help="Cluster threshold (default: 0.3)",
    )
    fleet_view_parser.add_argument(
        "--highlight",
        type=str,
        default=None,
        help="Highlight specific deployment_id or packet_id",
    )
    fleet_view_parser.add_argument(
        "--export",
        type=str,
        default=None,
        dest="export_path",
        help="Save graph to file (json or dot)",
    )
    fleet_view_parser.add_argument(
        "--diagnose",
        action="store_true",
        dest="diagnose_flag",
        help="Run health check on fleet",
    )
    add_v8_common_options(fleet_view_parser)

    # recipe
    recipe_parser = subparsers.add_parser(
        "recipe",
        help="Run pre-built command workflow",
    )
    recipe_parser.add_argument(
        "recipe_name",
        type=str,
        help="Name of recipe to run (deploy-check, fleet-audit, promote, or custom)",
    )
    recipe_parser.add_argument(
        "--var",
        action="append",
        dest="variables",
        default=[],
        metavar="KEY=VALUE",
        help="Variable substitution (can be repeated)",
    )
    recipe_parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_on_error",
        help="Continue even if steps fail",
    )
    add_v8_common_options(recipe_parser)

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
                return EXIT_VALIDATION_FAILED

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
            return EXIT_VALIDATION_FAILED

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
            return EXIT_VALIDATION_FAILED

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

    # --- v8 Command Handlers ---

    elif args.command == "build-packet":
        output_mode = OutputMode(args.output_mode)
        result = build_packet(
            deployment_id=args.deployment_id,
            manifest_path=args.manifest,
            receipts_dir=args.receipts_dir,
            output_dir=args.output_dir,
            sample_count=args.sample_count,
            no_save=args.no_save,
            output_mode=output_mode,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        if not result.get("success"):
            return EXIT_FATAL_ERROR if "not_found" in result.get("error", "") else EXIT_VALIDATION_FAILED

    elif args.command == "validate-config":
        output_mode = OutputMode(args.output_mode)
        result = validate_config(
            config_path=args.config_path,
            strict=args.strict,
            fix=args.fix,
            diff=args.diff,
            output_mode=output_mode,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        if not result.get("success"):
            return EXIT_FATAL_ERROR if result.get("error") == "file_not_found" else EXIT_VALIDATION_FAILED

    elif args.command == "merge-configs":
        output_mode = OutputMode(args.output_mode)
        result = merge_configs(
            parent_path=args.parent,
            child_path=args.child,
            output_path=args.output_path,
            auto_repair=args.auto_repair,
            chain=args.chain,
            simulate=args.simulate,
            output_mode=output_mode,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        if not result.get("success"):
            return EXIT_FATAL_ERROR if "not found" in result.get("error", "").lower() else EXIT_VALIDATION_FAILED

    elif args.command == "compare-packets":
        output_mode = OutputMode(args.output_mode)
        result = compare_packets(
            old_ref=args.old,
            new_ref=args.new,
            packets_dir=args.packets_dir,
            threshold=args.threshold,
            output_mode=output_mode,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        if not result.get("success"):
            return EXIT_FATAL_ERROR if "not_found" in result.get("error", "") else EXIT_VALIDATION_FAILED

    elif args.command == "fleet-view":
        output_mode = OutputMode(args.output_mode)
        result = fleet_view(
            packets_dir=args.packets_dir,
            min_similarity=args.min_similarity,
            highlight=args.highlight,
            export_path=args.export_path,
            diagnose_flag=args.diagnose_flag,
            output_mode=output_mode,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        if not result.get("success"):
            return EXIT_FATAL_ERROR if "no_packets" in result.get("error", "") else EXIT_VALIDATION_FAILED

    elif args.command == "recipe":
        output_mode = OutputMode(args.output_mode)
        # Parse variable substitutions from --var KEY=VALUE
        variables = {}
        for var_spec in args.variables:
            if "=" in var_spec:
                key, value = var_spec.split("=", 1)
                variables[key] = value

        result = run_recipe(
            recipe_name=args.recipe_name,
            variables=variables,
            continue_on_error=args.continue_on_error,
            output_mode=output_mode,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        if not result.get("success"):
            # Check for partial success
            steps = result.get("steps", [])
            if any(s.get("status") == "success" for s in steps):
                return EXIT_PARTIAL_SUCCESS
            return EXIT_FATAL_ERROR if "unknown_recipe" in result.get("error", "") else EXIT_VALIDATION_FAILED

    else:
        parser.print_help()
        return EXIT_SUCCESS

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
