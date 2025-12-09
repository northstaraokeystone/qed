"""
QED v8 Proof CLI Harness

CLI tool for validating QED telemetry compression and safety guarantees.
v8 uses Click framework with rich output for improved UX.

v8 subcommands (new):
  - build-packet: Build DecisionPacket from manifest/receipts
  - validate-config: Validate QEDConfig with auto-repair
  - merge-configs: Merge parent/child configs with safety enforcement
  - compare-packets: Compare two DecisionPackets
  - fleet-view: Display fleet topology and health

Legacy subcommands (preserved):
  - gates: Run legacy v5 gate checks (synthetic signals)
  - replay: Load edge_lab_sample.jsonl, run qed per scenario
  - sympy-suite: Get constraints per hook, verify, log violations
  - generate: Generate edge_lab_sample.jsonl test data
  - run-sims: Run pattern simulations via edge_lab_v2
  - recall-floor: Compute Clopper-Pearson exact recall lower bound
  - pattern-report: Display pattern library with sorting/filtering
  - clarity-audit: Process receipts through ClarityClean adapter

Exit Codes:
  0 = Success
  1 = Validation/comparison issue (actionable)
  2 = Fatal error (missing file, bad input)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
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
from decision_packet import DecisionPacket

# Rich imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    pass  # rich.progress imports removed (unused)
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# --- Console Setup ---
console = Console() if RICH_AVAILABLE else None


def print_success(message: str) -> None:
    """Print success message with green checkmark."""
    if console:
        console.print(f"[green]✓[/green] {message}")
    else:
        print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print error message with red X."""
    if console:
        console.print(f"[red]✗[/red] {message}")
    else:
        print(f"✗ {message}")


def print_warning(message: str) -> None:
    """Print warning message with yellow triangle."""
    if console:
        console.print(f"[yellow]⚠[/yellow] {message}")
    else:
        print(f"⚠ {message}")


def print_next(command: str) -> None:
    """Print suggested next command."""
    if console:
        console.print(f"\n[dim]Next:[/dim] [cyan]{command}[/cyan]")
    else:
        print(f"\nNext: {command}")


# --- KPI Thresholds ---
KPI_RECALL_THRESHOLD = 0.9967  # 99.67% recall CI
KPI_PRECISION_THRESHOLD = 0.95  # 95% precision
KPI_ROI_TARGET_M = 38.0  # $38M target ROI
KPI_ROI_TOLERANCE_M = 0.5  # +/- $0.5M tolerance
KPI_LATENCY_MS = 50.0  # Max latency per window


# =============================================================================
# Legacy Helper Functions (preserved from v6/v7)
# =============================================================================

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
    """Build a synthetic 1D signal."""
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
    """Return True if qed() is deterministic for this signal."""
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


# =============================================================================
# Legacy Functions (exported for tests)
# =============================================================================

def replay(
    jsonl_path: str,
    scenario: str = "tesla_fsd",
    bit_depth: int = 12,
    sample_rate_hz: float = 1000.0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Replay scenarios from a JSONL file through qed.py."""
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


def sympy_suite(
    hook: str,
    test_amplitudes: Optional[List[float]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run sympy constraint suite for a given hook/scenario."""
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

        if constraint_type != "amplitude_bound":
            continue

        for A in test_amplitudes:
            total_tests += 1
            exceeds_bound = abs(A) > bound

            if exceeds_bound:
                violations.append({
                    "constraint_id": constraint_id,
                    "amplitude": A,
                    "bound": bound,
                    "description": description,
                })
            else:
                passes.append({
                    "constraint_id": constraint_id,
                    "amplitude": A,
                    "bound": bound,
                })

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


def summarize(
    results: List[Dict[str, Any]],
    fleet_size: int = 2_000_000,
) -> Dict[str, Any]:
    """Summarize replay results into KPI metrics."""
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


def run_proof(seed: int = 42424242) -> Dict[str, Any]:
    """Run legacy v5 gate checks with synthetic signals."""
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
    gates["G7_determinism"] = _deterministic_check(signal_a, sample_rate_hz=sample_rate_a)
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
        _ = qed.qed(signal_b, scenario="tesla_fsd", bit_depth=12, sample_rate_hz=sample_rate_b)
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
    out_c = qed.qed(signal_c, scenario="tesla_fsd", bit_depth=12, sample_rate_hz=sample_rate_c)
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
        _ = qed.qed(signal_d, scenario="tesla_fsd", bit_depth=12, sample_rate_hz=sample_rate_d)
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


def generate_edge_lab_sample(
    output_path: str,
    n_anomalies: int = 900,
    n_normals: int = 100,
    seed: int = 42,
) -> None:
    """Generate edge_lab_sample.jsonl with labeled anomaly/normal scenarios."""
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


def run_sims(
    receipts_dir: str = "receipts/",
    patterns_path: str = "data/shared_anomalies.jsonl",
    n_per_hook: int = 1000,
    output: str = "data/sim_results.json",
) -> Dict[str, Any]:
    """Run pattern simulations via edge_lab_v2."""
    patterns = load_library(patterns_path)
    results = run_pattern_sims(
        receipts_dir=receipts_dir,
        patterns_path=patterns_path,
        n_per_hook=n_per_hook,
        progress_callback=lambda: tqdm(patterns, desc="Running pattern sims"),
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

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
    """Compute Clopper-Pearson exact recall lower bound."""
    if n_tests is None or n_misses is None:
        if sim_results_path is None:
            raise ValueError("Must provide either sim_results_path or both n_tests and n_misses")
        with open(sim_results_path, "r") as f:
            sim_data = json.load(f)
        if n_tests is None:
            n_tests = sim_data.get("n_tests", 0)
        if n_misses is None:
            n_misses = sim_data.get("n_misses", 0)

    n_successes = n_tests - n_misses
    alpha = 1.0 - confidence

    if n_successes == 0:
        lower_bound = 0.0
    else:
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
    """Load and display pattern library with sorting and filtering."""
    patterns = load_library(patterns_path)

    if exploit_only:
        patterns = [p for p in patterns if p.get("exploit_grade", False)]

    sort_key_map = {
        "dollar_value": lambda p: p.get("dollar_value", 0),
        "recall": lambda p: p.get("recall", 0),
        "exploit_grade": lambda p: (1 if p.get("exploit_grade", False) else 0),
    }
    sort_fn = sort_key_map.get(sort_by, sort_key_map["dollar_value"])
    patterns = sorted(patterns, key=sort_fn, reverse=True)

    if output_format == "json":
        print(json.dumps(patterns, indent=2))
    elif output_format == "table" and RICH_AVAILABLE:
        table = Table(title="Pattern Report")
        table.add_column("pattern_id", style="cyan", no_wrap=True)
        table.add_column("physics_domain", style="green")
        table.add_column("failure_mode", style="yellow")
        table.add_column("dollar_value", justify="right", style="magenta")
        table.add_column("recall", justify="right", style="blue")
        table.add_column("fp_rate", justify="right", style="red")
        table.add_column("exploit_grade", justify="center", style="bold")

        for p in patterns:
            pattern_id = str(p.get("pattern_id", ""))[:20]
            physics_domain = str(p.get("physics_domain", ""))
            failure_mode = str(p.get("failure_mode", ""))
            dollar_value = f"${p.get('dollar_value', 0):,.0f}"
            recall_val = f"{p.get('recall', 0):.4f}"
            fp_rate = f"{p.get('fp_rate', 0):.4f}"
            exploit = "Yes" if p.get("exploit_grade", False) else "No"
            table.add_row(pattern_id, physics_domain, failure_mode, dollar_value, recall_val, fp_rate, exploit)

        console.print(table)
    else:
        header = f"{'pattern_id':<20} {'physics_domain':<15} {'failure_mode':<20} {'dollar_value':>12} {'recall':>8} {'fp_rate':>8} {'exploit':>8}"
        print(header)
        print("-" * len(header))
        for p in patterns:
            print(f"{str(p.get('pattern_id', ''))[:20]:<20} {str(p.get('physics_domain', ''))[:15]:<15} {str(p.get('failure_mode', ''))[:20]:<20} {p.get('dollar_value', 0):>12,.0f} {p.get('recall', 0):>8.4f} {p.get('fp_rate', 0):>8.4f} {'Yes' if p.get('exploit_grade', False) else 'No':>8}")

    return patterns


def clarity_audit(
    receipts_path: str,
    output_corpus: Optional[str] = None,
    output_receipt: str = "data/clarity_receipts.jsonl",
) -> Dict[str, Any]:
    """Process receipts through ClarityClean adapter."""
    result = process_receipts(receipts_path=receipts_path, output_corpus=output_corpus)
    receipt = result.get("receipt", {})

    output_path = Path(output_receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(receipt) + "\n")

    return {
        "token_count": receipt.get("token_count", 0),
        "anomaly_density": receipt.get("anomaly_density", 0.0),
        "noise_ratio": receipt.get("noise_ratio", 0.0),
        "corpus_hash": receipt.get("corpus_hash", ""),
        "output_receipt": str(output_path),
        "output_corpus": output_corpus,
    }


# =============================================================================
# Click CLI
# =============================================================================

@click.group()
@click.version_option(version="8.0.0", prog_name="proof")
def cli():
    """
    QED v8 Proof CLI - Validate deployments, configs, and fleet health.

    \b
    v8 Commands:
      build-packet    Build DecisionPacket from manifest/receipts
      validate-config Validate QEDConfig with optional auto-repair
      merge-configs   Merge parent/child configs with safety enforcement
      compare-packets Compare two DecisionPackets
      fleet-view      Display fleet topology and health

    \b
    Legacy Commands:
      gates           Run legacy v5 gate checks
      replay          Replay scenarios from JSONL
      sympy-suite     Check constraints for hook
      generate        Generate edge_lab_sample.jsonl
      run-sims        Run pattern simulations
      recall-floor    Compute recall lower bound
      pattern-report  Display pattern library
      clarity-audit   Process receipts through ClarityClean
    """
    pass


# =============================================================================
# v8 Commands
# =============================================================================

@cli.command("build-packet")
@click.option("--deployment-id", "-d", required=True, help="Deployment identifier")
@click.option("--manifest", "-m", default="data/manifests/", help="Manifest path")
@click.option("--receipts", "-r", default="receipts/", help="Receipts directory")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich", help="Output format")
@click.option("--save", "-s", default="data/packets/", help="Save packets directory")
def build_packet_cmd(deployment_id: str, manifest: str, receipts: str, output: str, save: str):
    """
    Build DecisionPacket from manifest and receipts.

    Calls truthlink.build() to create a packet, saves to JSONL.

    \b
    Exit codes:
      0 = success
      2 = manifest not found
    """
    manifest_path = Path(manifest)

    # Find manifest file
    if manifest_path.is_dir():
        # Look for manifest matching deployment_id
        candidates = list(manifest_path.glob(f"*{deployment_id}*.json"))
        if not candidates:
            candidates = list(manifest_path.glob("*.json"))
        if not candidates:
            print_error(f"No manifest found in {manifest}")
            sys.exit(2)
        manifest_file = str(candidates[0])
    else:
        manifest_file = str(manifest_path)
        if not manifest_path.exists():
            print_error(f"Manifest not found: {manifest}")
            sys.exit(2)

    try:
        packet = truthlink.build(
            deployment_id=deployment_id,
            manifest_path=manifest_file,
            receipts_dir=receipts,
        )

        # Save packet
        truthlink.save(packet, output_dir=save)
        timestamp = datetime.now().strftime("%Y%m%d")
        save_path = f"{save}{deployment_id}_{timestamp}.jsonl"

        if output == "json":
            click.echo(packet.to_json(indent=2))
        else:
            # Rich output
            if RICH_AVAILABLE:
                # Build health bar
                health = packet.health_score
                bar_filled = int(health / 5)
                bar_empty = 20 - bar_filled
                health_bar = "█" * bar_filled + "░" * bar_empty

                # Format savings
                savings = packet.metrics.annual_savings
                if savings >= 1_000_000:
                    savings_str = f"${savings / 1_000_000:,.0f},000"
                else:
                    savings_str = f"${savings:,.0f}"

                # Count exploit patterns
                exploit_count = sum(1 for p in packet.pattern_usage if p.exploit_grade)
                total_patterns = len(packet.pattern_usage)

                content = f"""[bold]packet_id:[/bold]       {packet.packet_id}
[bold]health_score:[/bold]    {health}/100 {health_bar}
[bold]annual_savings:[/bold]  {savings_str}
[bold]patterns:[/bold]        {total_patterns} active ({exploit_count} exploit-grade)
[bold]slo_breach_rate:[/bold] {packet.metrics.slo_breach_rate * 100:.2f}%
[bold]exploit_coverage:[/bold] {packet.exploit_coverage * 100:.0f}%"""

                panel = Panel(
                    content,
                    title=f"DecisionPacket: {deployment_id}",
                    box=box.ROUNDED,
                    border_style="green",
                )
                console.print(panel)
                print_success(f"Saved: {save_path}")
                print_next(f"proof compare-packets -a <previous_id> -b {packet.packet_id}")
            else:
                click.echo(f"Packet ID: {packet.packet_id}")
                click.echo(f"Health Score: {packet.health_score}/100")
                click.echo(f"Annual Savings: ${packet.metrics.annual_savings:,.0f}")

        sys.exit(0)

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(2)
    except Exception as e:
        print_error(f"Failed to build packet: {e}")
        sys.exit(2)


@cli.command("validate-config")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--fix", is_flag=True, help="Auto-repair and save")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich", help="Output format")
def validate_config_cmd(config_path: str, fix: bool, output: str):
    """
    Validate QEDConfig file.

    Calls config_schema.load() with validation.

    \b
    Exit codes:
      0 = valid
      1 = invalid but fixable
      2 = fatal error
    """
    try:
        # Try strict validation first
        try:
            config = config_schema.load(config_path, validate=True, strict=True)
            is_valid = True
            errors = []
            warnings = []
        except ValueError as e:
            is_valid = False
            errors = str(e).split("\n")
            # Try non-strict to get warnings
            try:
                import warnings as warn_module
                with warn_module.catch_warnings(record=True) as w:
                    warn_module.simplefilter("always")
                    config = config_schema.load(config_path, validate=True, strict=False)
                    warnings = [str(warning.message) for warning in w]
            except Exception:
                config = None
                warnings = []

        if output == "json":
            result = {
                "valid": is_valid,
                "file": config_path,
                "errors": errors if not is_valid else [],
                "warnings": warnings,
            }
            if config:
                result["config"] = config.to_dict()
            click.echo(json.dumps(result, indent=2))
        else:
            # Rich output
            if RICH_AVAILABLE:
                if is_valid:
                    # Get config details
                    sim = config.simulate()
                    content = f"""[bold]File:[/bold] {config_path}
[bold]Hook:[/bold] {config.hook}         [bold]Patterns:[/bold] {len(config.enabled_patterns)} enabled
[bold]recall_floor:[/bold] {config.recall_floor}    [bold]max_fp_rate:[/bold] {config.max_fp_rate}
[bold]Risk Profile:[/bold] {sim.risk_profile}"""

                    panel = Panel(
                        content,
                        title="[green]Config Validation: PASSED[/green]",
                        box=box.ROUNDED,
                        border_style="green",
                    )
                    console.print(panel)
                    print_next(f"proof merge-configs -p global.json -c {config_path}")
                else:
                    lines = [f"[bold]File:[/bold] {config_path}"]
                    for err in errors:
                        if err.strip() and not err.startswith("Config validation"):
                            lines.append(f"[red]✗[/red] {err.strip().lstrip('- ')}")
                    for warn in warnings:
                        if "QEDConfig:" in warn:
                            warn = warn.replace("QEDConfig: ", "")
                        lines.append(f"[yellow]⚠[/yellow] {warn}")

                    content = "\n".join(lines)
                    panel = Panel(
                        content,
                        title="[red]Config Validation: FAILED[/red]",
                        box=box.ROUNDED,
                        border_style="red",
                    )
                    console.print(panel)

                    fixable = len([e for e in errors if "recall_floor" in e or "max_fp_rate" in e]) > 0
                    if fixable:
                        click.echo(f"Fixable errors: {len(errors)}   Run with --fix to repair")
                        print_next(f"proof validate-config {config_path} --fix")

                    if fix and config:
                        config.save(config_path)
                        print_success(f"Repaired and saved: {config_path}")
            else:
                if is_valid:
                    click.echo(f"Config Validation: PASSED - {config_path}")
                else:
                    click.echo(f"Config Validation: FAILED - {config_path}")
                    for err in errors:
                        click.echo(f"  - {err}")

        sys.exit(0 if is_valid else 1)

    except FileNotFoundError:
        print_error(f"Config file not found: {config_path}")
        sys.exit(2)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        sys.exit(2)


@cli.command("merge-configs")
@click.option("--parent", "-p", required=True, type=click.Path(exists=True), help="Parent config path")
@click.option("--child", "-c", required=True, type=click.Path(exists=True), help="Child config path")
@click.option("--save", "-s", type=click.Path(), help="Save merged config path")
@click.option("--auto-repair", is_flag=True, help="Fix violations automatically")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich", help="Output format")
def merge_configs_cmd(parent: str, child: str, save: Optional[str], auto_repair: bool, output: str):
    """
    Merge parent and child configs with safety enforcement.

    Uses merge_rules.merge() with safety-only-tightens rule.

    \b
    Exit codes:
      0 = valid merge
      1 = violations (repairable)
      2 = fatal error
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parent_config = config_schema.load(parent, validate=True, strict=False)
            child_config = config_schema.load(child, validate=True, strict=False)

        result = merge_rules.merge(
            parent_config,
            child_config,
            auto_repair=auto_repair,
            emit_receipt_flag=False,
        )

        if output == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            # Rich output
            if RICH_AVAILABLE:
                parent_name = Path(parent).name
                child_name = Path(child).name

                if result.is_valid:
                    merged = result.merged
                    lines = [f"[bold]Parent:[/bold] {parent_name} → [bold]Child:[/bold] {child_name}"]

                    # Show field changes
                    for field, decision in result.explanation.field_decisions.items():
                        if field in ["recall_floor", "max_fp_rate"]:
                            if decision.parent_value != decision.child_value:
                                direction = "tightened" if decision.direction == "from_parent" else "unchanged"
                                lines.append(f"[bold]{field}:[/bold]  {decision.parent_value} → {decision.merged_value} ({direction}) [green]✓[/green]")

                    # Pattern changes
                    if result.explanation.patterns_kept or result.explanation.patterns_removed:
                        kept = len(result.explanation.patterns_kept)
                        lines.append(f"[bold]patterns:[/bold]      {kept} (intersection) [green]✓[/green]")

                    # Regulatory flags
                    if merged.regulatory_flags:
                        flags = " ".join([f"{k}=✓" for k, v in merged.regulatory_flags.items() if v])
                        lines.append(f"[bold]regulatory:[/bold]    {flags} (combined)")

                    lines.append("")
                    lines.append(f"[bold]Direction:[/bold] {result.explanation.safety_direction.upper()}")

                    content = "\n".join(lines)
                    panel = Panel(
                        content,
                        title="[green]Config Merge: VALID[/green]",
                        box=box.ROUNDED,
                        border_style="green",
                    )
                    console.print(panel)

                    if save and merged:
                        merged.save(save)
                        print_success(f"Saved merged config: {save}")

                    print_next(f"proof build-packet -d {child_config.deployment_id}")
                else:
                    lines = [f"[bold]Parent:[/bold] {parent_name} → [bold]Child:[/bold] {child_name}"]

                    for v in result.violations:
                        lines.append(f"[red]✗[/red] {v.field}: {v.parent_value} → {v.child_value} (LOOSENED - blocked)")

                    content = "\n".join(lines)
                    panel = Panel(
                        content,
                        title="[red]Config Merge: VIOLATION[/red]",
                        box=box.ROUNDED,
                        border_style="red",
                    )
                    console.print(panel)
                    click.echo("Safety cannot loosen. Run with --auto-repair to tighten child.")
                    print_next(f"proof merge-configs -p {parent} -c {child} --auto-repair")
            else:
                if result.is_valid:
                    click.echo("Config Merge: VALID")
                else:
                    click.echo("Config Merge: VIOLATION")
                    for v in result.violations:
                        click.echo(f"  - {v}")

        sys.exit(0 if result.is_valid else 1)

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(2)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        sys.exit(2)


@cli.command("compare-packets")
@click.option("--old", "-a", required=True, help="Old packet ID or path")
@click.option("--new", "-b", required=True, help="New packet ID or path")
@click.option("--packets-dir", default="data/packets/", help="Packets directory")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich", help="Output format")
def compare_packets_cmd(old: str, new: str, packets_dir: str, output: str):
    """
    Compare two DecisionPackets.

    Calls truthlink.compare() for structured diff.

    \b
    Exit codes:
      0 = improvement/neutral
      1 = regression
      2 = packet not found
    """
    try:
        # Load packets
        def load_packet(ref: str) -> Optional[DecisionPacket]:
            # Try as file path first
            if Path(ref).exists():
                with open(ref) as f:
                    return DecisionPacket.from_json(f.read())

            # Try loading from packets dir by ID
            packets = truthlink.load(packet_id=ref, packets_dir=packets_dir)
            if packets:
                return packets[0]

            # Try partial match
            all_packets = truthlink.load(packets_dir=packets_dir)
            for p in all_packets:
                if p.packet_id.startswith(ref):
                    return p
            return None

        old_packet = load_packet(old)
        new_packet = load_packet(new)

        if not old_packet:
            print_error(f"Old packet not found: {old}")
            sys.exit(2)
        if not new_packet:
            print_error(f"New packet not found: {new}")
            sys.exit(2)

        comparison = truthlink.compare(old_packet, new_packet)

        if output == "json":
            click.echo(json.dumps(comparison.to_dict(), indent=2))
        else:
            # Rich output
            if RICH_AVAILABLE:
                # Format dates
                try:
                    old_date = old_packet.timestamp[:10]
                    new_date = new_packet.timestamp[:10]
                except Exception:
                    old_date = "unknown"
                    new_date = "unknown"

                classification_emoji = "⬆" if comparison.classification == "improvement" else ("⬇" if comparison.classification == "regression" else "↔")

                lines = [
                    f"[bold]Old:[/bold] {old_packet.packet_id[:8]} ({old_date})",
                    f"[bold]New:[/bold] {new_packet.packet_id[:8]} ({new_date})",
                    f"[bold]Classification:[/bold] {comparison.classification.upper()} {classification_emoji}",
                ]

                # Delta table
                for field, (old_val, new_val, pct) in comparison.delta.items():
                    if field == "health_score":
                        emoji = "⬆" if pct > 0 else ("⬇" if pct < 0 else "")
                        lines.append(f"[bold]{field}:[/bold]     {old_val} → {new_val}      {pct:+.1f}%  {emoji}")
                    elif field == "annual_savings":
                        old_m = f"${old_val/1_000_000:.1f}M" if old_val >= 1_000_000 else f"${old_val:,.0f}"
                        new_m = f"${new_val/1_000_000:.1f}M" if new_val >= 1_000_000 else f"${new_val:,.0f}"
                        emoji = "⬆" if pct > 0 else ("⬇" if pct < 0 else "")
                        lines.append(f"[bold]{field}:[/bold]   {old_m} → {new_m}  {pct:+.1f}%  {emoji}")
                    elif field == "slo_breach_rate":
                        emoji = "⬆" if pct < 0 else ("⬇" if pct > 0 else "")  # Lower is better
                        lines.append(f"[bold]{field}:[/bold]  {old_val*100:.2f}% → {new_val*100:.2f}%  {pct:+.1f}%  {emoji}")
                    elif field == "exploit_coverage":
                        emoji = "⬆" if pct > 0 else ("⬇" if pct < 0 else "")
                        lines.append(f"[bold]{field}:[/bold] {old_val*100:.0f}% → {new_val*100:.0f}%    {pct:+.1f}%  {emoji}")

                # Pattern changes
                if comparison.patterns_added or comparison.patterns_removed:
                    added = len(comparison.patterns_added)
                    removed = len(comparison.patterns_removed)
                    pattern_str = ""
                    if added:
                        pattern_str += f"+{added} added"
                        if comparison.patterns_added:
                            pattern_str += f" ({comparison.patterns_added[0]})"
                    if removed:
                        if added:
                            pattern_str += ", "
                        pattern_str += f"-{removed} removed"
                    lines.append(f"[bold]Patterns:[/bold] {pattern_str}")

                content = "\n".join(lines)
                panel = Panel(
                    content,
                    title="Packet Comparison",
                    box=box.ROUNDED,
                    border_style="cyan",
                )
                console.print(panel)
                click.echo(f"Recommendation: {comparison.recommendation}")
                print_next(f"proof fleet-view --highlight {new_packet.packet_id[:8]}")
            else:
                click.echo(f"Classification: {comparison.classification}")
                click.echo(comparison.narration)

        is_regression = comparison.classification == "regression"
        sys.exit(1 if is_regression else 0)

    except Exception as e:
        print_error(f"Error comparing packets: {e}")
        sys.exit(2)


@cli.command("fleet-view")
@click.option("--packets-dir", default="data/packets/", help="Packets directory")
@click.option("--highlight", "-h", "highlight_id", help="Highlight deployment ID")
@click.option("--diagnose", "run_diagnose", is_flag=True, help="Run fleet health check")
@click.option("--export", "-e", "export_path", type=click.Path(), help="Export graph JSON")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich", help="Output format")
def fleet_view_cmd(packets_dir: str, highlight_id: Optional[str], run_diagnose: bool, export_path: Optional[str], output: str):
    """
    Display fleet topology and health overview.

    Uses mesh_view_v3.build() and diagnose().

    \b
    Exit codes:
      0 = healthy
      1 = issues found
      2 = no packets
    """
    try:
        # Load packets
        packets = truthlink.load(packets_dir=packets_dir)

        if not packets:
            print_error(f"No packets found in {packets_dir}")
            sys.exit(2)

        # Build graph
        graph = mesh_view_v3.build(packets)
        metrics = mesh_view_v3.compute_fleet_metrics(graph)

        if export_path:
            mesh_view_v3.save(graph, export_path)
            print_success(f"Exported graph to {export_path}")

        if output == "json":
            result = {
                "fleet_metrics": metrics.to_dict(),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
            }
            if run_diagnose:
                diagnosis = mesh_view_v3.diagnose(graph)
                result["diagnosis"] = diagnosis.to_dict()
            click.echo(json.dumps(result, indent=2))
        else:
            # Rich output
            if RICH_AVAILABLE:
                # Fleet summary
                lines = [
                    f"[bold]Deployments:[/bold] {metrics.active_deployments} active, {metrics.stale_deployments} stale",
                    f"[bold]Total Savings:[/bold] ${metrics.total_annual_savings/1_000_000:.1f}M/year",
                    f"[bold]Avg Health:[/bold] {metrics.avg_health_score:.0f}/100      [bold]Fleet Cohesion:[/bold] {metrics.fleet_cohesion:.2f}",
                ]

                # Clusters
                clusters = graph.find_clusters(min_similarity=0.3)
                lines.append("")
                lines.append("[bold]CLUSTERS[/bold]")

                # Group by hook
                hook_clusters: Dict[str, List] = {}
                for cluster in clusters:
                    if cluster.deployment_ids:
                        first_node = graph.nodes.get(cluster.deployment_ids[0])
                        hook = first_node.hook if first_node else "unknown"
                        if hook not in hook_clusters:
                            hook_clusters[hook] = []
                        hook_clusters[hook].append(cluster)

                for hook, hook_cluster_list in hook_clusters.items():
                    total_in_hook = sum(len(c.deployment_ids) for c in hook_cluster_list)
                    avg_sim = sum(c.avg_similarity for c in hook_cluster_list) / len(hook_cluster_list) if hook_cluster_list else 0
                    bar_len = int(avg_sim * 16)
                    bar = "█" * bar_len + "░" * (16 - bar_len)
                    lines.append(f"  {hook.capitalize()} ({total_in_hook})     {bar} {avg_sim*100:.0f}% overlap")

                # Outliers
                outliers = graph.find_outliers(threshold=0.1)
                if outliers:
                    lines.append(f"  Outliers ({len(outliers)})  {'░' * 16} <30% similarity")

                # Actions needed (from diagnosis)
                if run_diagnose:
                    diagnosis = mesh_view_v3.diagnose(graph)
                    if diagnosis.warnings or diagnosis.issues:
                        lines.append("")
                        lines.append("[bold]ACTIONS NEEDED[/bold]")
                        for issue in diagnosis.issues[:3]:
                            lines.append(f"  • {issue}")
                        for warning in diagnosis.warnings[:2]:
                            lines.append(f"  • {warning}")

                content = "\n".join(lines)
                panel = Panel(
                    content,
                    title="Fleet Overview",
                    box=box.ROUNDED,
                    border_style="blue",
                )
                console.print(panel)

                # Suggest next action
                if metrics.stale_deployments > 0:
                    stale_nodes = [n for n in graph.nodes.values() if n.is_stale]
                    if stale_nodes:
                        print_next(f"proof build-packet -d {stale_nodes[0].deployment_id}")
            else:
                click.echo(f"Fleet: {metrics.total_deployments} deployments")
                click.echo(f"Total Savings: ${metrics.total_annual_savings:,.0f}/year")
                click.echo(f"Avg Health: {metrics.avg_health_score:.0f}/100")

        # Exit code based on health
        if run_diagnose:
            diagnosis = mesh_view_v3.diagnose(graph)
            sys.exit(0 if diagnosis.is_healthy else 1)
        else:
            sys.exit(0)

    except Exception as e:
        print_error(f"Error loading fleet: {e}")
        sys.exit(2)


# =============================================================================
# Legacy Commands
# =============================================================================

@cli.command("gates")
@click.option("--seed", type=int, default=42424242, help="Random seed")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON")
def gates_cmd(seed: int, json_out: bool):
    """Run legacy v5 gate checks with synthetic signals."""
    result = run_proof(seed=seed)

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        if result["all_pass"]:
            print_success("QED v6 proof gates passed")
            metrics = result["metrics"]
            click.echo(
                f"Signal A: ratio={metrics['ratio']:.1f}, H_bits={metrics['H_bits']:.0f}, "
                f"recall={metrics['recall']:.4f}, savings_M={metrics['savings_M']:.2f}"
            )
        else:
            print_error(f"FAILED gates: {result['failed']}")
            sys.exit(1)


@cli.command("replay")
@click.argument("jsonl_path", type=click.Path(exists=True))
@click.option("--scenario", default="tesla_fsd", help="Default scenario")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("--output-file", "-o", type=click.Path(), help="Output JSON file")
def replay_cmd(jsonl_path: str, scenario: str, verbose: bool, output_file: Optional[str]):
    """Replay scenarios from JSONL through qed.py."""
    results = replay(jsonl_path, scenario=scenario, verbose=verbose)
    summary = summarize(results)

    if output_file:
        output_data = {"results": results, "summary": summary}
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print_success(f"Results written to {output_file}")
    else:
        click.echo(json.dumps(summary, indent=2))

    if not summary["kpi"]["all_pass"]:
        sys.exit(1)


@cli.command("sympy-suite")
@click.argument("hook")
@click.option("--verbose", is_flag=True, help="Include all violations")
def sympy_suite_cmd(hook: str, verbose: bool):
    """Run sympy constraint suite for a hook."""
    result = sympy_suite(hook, verbose=verbose)
    click.echo(json.dumps(result, indent=2))

    if result["n_violations"] > 0 and not verbose:
        click.echo(f"\nNote: {result['n_violations']} violations found. Use --verbose for details.")


@cli.command("generate")
@click.option("--output", "-o", default="edge_lab_sample.jsonl", help="Output file")
@click.option("--anomalies", type=int, default=900, help="Number of anomalies")
@click.option("--normals", type=int, default=100, help="Number of normals")
@click.option("--seed", type=int, default=42, help="Random seed")
def generate_cmd(output: str, anomalies: int, normals: int, seed: int):
    """Generate edge_lab_sample.jsonl test data."""
    generate_edge_lab_sample(output, n_anomalies=anomalies, n_normals=normals, seed=seed)


@cli.command("run-sims")
@click.option("--receipts-dir", default="receipts/", help="Receipts directory")
@click.option("--patterns-path", default="data/shared_anomalies.jsonl", help="Patterns file")
@click.option("--n-per-hook", type=int, default=1000, help="Simulations per hook")
@click.option("--output", "-o", default="data/sim_results.json", help="Output file")
def run_sims_cmd(receipts_dir: str, patterns_path: str, n_per_hook: int, output: str):
    """Run pattern simulations via edge_lab_v2."""
    result = run_sims(receipts_dir, patterns_path, n_per_hook, output)
    click.echo(f"Simulation complete: n_tests={result['n_tests']}, aggregate_recall={result['aggregate_recall']:.4f}")
    print_success(f"Results written to {result['output_path']}")


@cli.command("recall-floor")
@click.option("--sim-results", default="data/sim_results.json", help="Simulation results file")
@click.option("--confidence", type=float, default=0.95, help="Confidence level")
@click.option("--n-tests", type=int, default=None, help="Override n_tests")
@click.option("--n-misses", type=int, default=None, help="Override n_misses")
def recall_floor_cmd(sim_results: str, confidence: float, n_tests: Optional[int], n_misses: Optional[int]):
    """Compute Clopper-Pearson exact recall lower bound."""
    result = recall_floor(sim_results, confidence, n_tests, n_misses)
    click.echo(f"Recall floor: {result['recall_floor']:.4f} at {result['confidence']*100:.0f}% confidence")


@cli.command("pattern-report")
@click.option("--patterns-path", default="data/shared_anomalies.jsonl", help="Patterns file")
@click.option("--sort-by", type=click.Choice(["dollar_value", "recall", "exploit_grade"]), default="dollar_value")
@click.option("--exploit-only", is_flag=True, help="Show only exploit-grade patterns")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table")
def pattern_report_cmd(patterns_path: str, sort_by: str, exploit_only: bool, output_format: str):
    """Display pattern library with sorting and filtering."""
    pattern_report(patterns_path, sort_by, exploit_only, output_format)


@cli.command("clarity-audit")
@click.option("--receipts-path", required=True, help="Receipts file path")
@click.option("--output-corpus", default=None, help="Output corpus path")
@click.option("--output-receipt", default="data/clarity_receipts.jsonl", help="Receipt output")
def clarity_audit_cmd(receipts_path: str, output_corpus: Optional[str], output_receipt: str):
    """Process receipts through ClarityClean adapter."""
    result = clarity_audit(receipts_path, output_corpus, output_receipt)
    click.echo("ClarityClean audit complete:")
    click.echo(f"  token_count: {result['token_count']}")
    click.echo(f"  anomaly_density: {result['anomaly_density']:.4f}")
    click.echo(f"  noise_ratio: {result['noise_ratio']:.4f}")
    print_success(f"Receipt written to {result['output_receipt']}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point for proof CLI."""
    try:
        cli()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
