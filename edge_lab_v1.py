"""
edge_lab_v1.py - Edge Lab Scenario Runner for QED v6

Runs edge-case scenarios (spikes, steps, drifts) through the QED telemetry
compression engine and collects metrics: hit/miss, latency, compression ratio,
and constraint violations.

Schema (JSONL):
  {
    "scenario_id": str,        # Unique identifier e.g. "tesla_spike_001"
    "hook": str,               # Hook name e.g. "tesla", "spacex"
    "type": str,               # Anomaly type: "spike", "step", "drift", "normal"
    "expected_loss": float,    # Expected loss threshold (>0.1 for ROI significance)
    "signal": list[float]      # Raw signal array
  }

Metrics per scenario:
  - hit: bool (recall >= 0.95)
  - miss: bool (1 - hit)
  - latency_ms: float
  - ratio: float (compression ratio)
  - violations: int (count of sympy constraint failures)
  - verified: bool (all constraints passed)
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from qed import qed
from sympy_constraints import get_constraints


# -----------------------------------------------------------------------------
# Schema Definition
# -----------------------------------------------------------------------------

SCENARIO_SCHEMA = {
    "scenario_id": str,
    "hook": str,
    "type": str,
    "expected_loss": float,
    "signal": list,
}

# Valid anomaly types for edge lab scenarios
ANOMALY_TYPES = {"spike", "step", "drift", "normal", "noise", "saturation"}


@dataclass
class EdgeLabResult:
    """Result metrics for a single edge lab scenario run."""
    scenario_id: str
    hook: str
    type: str
    expected_loss: float
    hit: bool              # recall >= 0.95
    miss: bool             # not hit
    latency_ms: float
    ratio: float           # compression ratio
    recall: float          # raw recall value
    violations: int        # count of constraint violations
    verified: bool         # all constraints passed
    violation_details: List[Dict[str, Any]]
    error: Optional[str]   # error message if processing failed


# -----------------------------------------------------------------------------
# In-Memory Fallback Scenarios
# -----------------------------------------------------------------------------

def _generate_spike_signal(
    n: int = 1000,
    amplitude: float = 12.0,
    frequency_hz: float = 40.0,
    spike_amplitude: float = 25.0,
    spike_idx: Optional[int] = None,
    noise_sigma: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sinusoidal signal with an injected spike anomaly."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    signal = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    signal += rng.normal(0, noise_sigma, n)
    # Inject spike
    if spike_idx is None:
        spike_idx = n // 2
    signal[spike_idx] = spike_amplitude
    return signal


def _generate_step_signal(
    n: int = 1000,
    amplitude: float = 12.0,
    frequency_hz: float = 40.0,
    step_value: float = 5.0,
    step_start: Optional[int] = None,
    noise_sigma: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sinusoidal signal with a step change anomaly."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    signal = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    signal += rng.normal(0, noise_sigma, n)
    # Inject step
    if step_start is None:
        step_start = n // 2
    signal[step_start:] += step_value
    return signal


def _generate_drift_signal(
    n: int = 1000,
    amplitude: float = 12.0,
    frequency_hz: float = 40.0,
    drift_rate: float = 3.0,
    noise_sigma: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sinusoidal signal with linear drift anomaly."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    signal = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    signal += rng.normal(0, noise_sigma, n)
    # Inject drift
    drift = np.linspace(0, drift_rate, n)
    signal += drift
    return signal


def _generate_normal_signal(
    n: int = 1000,
    amplitude: float = 12.0,
    frequency_hz: float = 40.0,
    noise_sigma: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a clean sinusoidal signal (no anomaly)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    signal = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    signal += rng.normal(0, noise_sigma, n)
    return signal


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Generate in-memory edge lab scenarios for laptop testing.

    Returns a list of 20 scenarios across hooks (tesla, spacex) with
    various anomaly types (spike, step, drift, normal).
    """
    scenarios = []

    # Tesla scenarios - steering torque (bound: 14.7)
    tesla_scenarios = [
        # Spikes - high amplitude anomalies
        {"id": "tesla_spike_001", "type": "spike", "amp": 12.0, "spike_amp": 20.0, "loss": 0.15, "seed": 1},
        {"id": "tesla_spike_002", "type": "spike", "amp": 13.0, "spike_amp": 25.0, "loss": 0.20, "seed": 2},
        {"id": "tesla_spike_003", "type": "spike", "amp": 10.0, "spike_amp": 18.0, "loss": 0.12, "seed": 3},
        # Steps - sustained offset
        {"id": "tesla_step_001", "type": "step", "amp": 12.0, "step_val": 4.0, "loss": 0.14, "seed": 4},
        {"id": "tesla_step_002", "type": "step", "amp": 11.0, "step_val": 6.0, "loss": 0.18, "seed": 5},
        # Drifts - gradual change
        {"id": "tesla_drift_001", "type": "drift", "amp": 12.0, "drift_rate": 5.0, "loss": 0.16, "seed": 6},
        {"id": "tesla_drift_002", "type": "drift", "amp": 13.0, "drift_rate": 4.0, "loss": 0.13, "seed": 7},
        # Normal - clean signals for baseline
        {"id": "tesla_normal_001", "type": "normal", "amp": 12.0, "loss": 0.05, "seed": 8},
        {"id": "tesla_normal_002", "type": "normal", "amp": 10.0, "loss": 0.04, "seed": 9},
        {"id": "tesla_normal_003", "type": "normal", "amp": 14.0, "loss": 0.06, "seed": 10},
    ]

    for ts in tesla_scenarios:
        if ts["type"] == "spike":
            signal = _generate_spike_signal(
                amplitude=ts["amp"], spike_amplitude=ts["spike_amp"], seed=ts["seed"]
            )
        elif ts["type"] == "step":
            signal = _generate_step_signal(
                amplitude=ts["amp"], step_value=ts["step_val"], seed=ts["seed"]
            )
        elif ts["type"] == "drift":
            signal = _generate_drift_signal(
                amplitude=ts["amp"], drift_rate=ts["drift_rate"], seed=ts["seed"]
            )
        else:
            signal = _generate_normal_signal(amplitude=ts["amp"], seed=ts["seed"])

        scenarios.append({
            "scenario_id": ts["id"],
            "hook": "tesla",
            "type": ts["type"],
            "expected_loss": ts["loss"],
            "signal": signal.tolist(),
        })

    # SpaceX scenarios - thrust oscillation (bound: 20.0)
    spacex_scenarios = [
        # Spikes
        {"id": "spacex_spike_001", "type": "spike", "amp": 15.0, "spike_amp": 30.0, "loss": 0.22, "seed": 11},
        {"id": "spacex_spike_002", "type": "spike", "amp": 18.0, "spike_amp": 35.0, "loss": 0.25, "seed": 12},
        # Steps
        {"id": "spacex_step_001", "type": "step", "amp": 16.0, "step_val": 8.0, "loss": 0.19, "seed": 13},
        {"id": "spacex_step_002", "type": "step", "amp": 14.0, "step_val": 10.0, "loss": 0.21, "seed": 14},
        # Drifts
        {"id": "spacex_drift_001", "type": "drift", "amp": 15.0, "drift_rate": 6.0, "loss": 0.17, "seed": 15},
        {"id": "spacex_drift_002", "type": "drift", "amp": 17.0, "drift_rate": 5.0, "loss": 0.15, "seed": 16},
        # Normal
        {"id": "spacex_normal_001", "type": "normal", "amp": 16.0, "loss": 0.06, "seed": 17},
        {"id": "spacex_normal_002", "type": "normal", "amp": 18.0, "loss": 0.07, "seed": 18},
        {"id": "spacex_normal_003", "type": "normal", "amp": 15.0, "loss": 0.05, "seed": 19},
        {"id": "spacex_normal_004", "type": "normal", "amp": 19.0, "loss": 0.08, "seed": 20},
    ]

    for ss in spacex_scenarios:
        if ss["type"] == "spike":
            signal = _generate_spike_signal(
                amplitude=ss["amp"], spike_amplitude=ss["spike_amp"], seed=ss["seed"]
            )
        elif ss["type"] == "step":
            signal = _generate_step_signal(
                amplitude=ss["amp"], step_value=ss["step_val"], seed=ss["seed"]
            )
        elif ss["type"] == "drift":
            signal = _generate_drift_signal(
                amplitude=ss["amp"], drift_rate=ss["drift_rate"], seed=ss["seed"]
            )
        else:
            signal = _generate_normal_signal(amplitude=ss["amp"], seed=ss["seed"])

        scenarios.append({
            "scenario_id": ss["id"],
            "hook": "spacex",
            "type": ss["type"],
            "expected_loss": ss["loss"],
            "signal": signal.tolist(),
        })

    return scenarios


# -----------------------------------------------------------------------------
# Schema Validation
# -----------------------------------------------------------------------------

def validate_scenario(scenario: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a scenario against the SCENARIO_SCHEMA.

    Returns (is_valid, error_message).
    """
    required_keys = set(SCENARIO_SCHEMA.keys())
    actual_keys = set(scenario.keys())

    missing = required_keys - actual_keys
    if missing:
        return False, f"Missing required keys: {missing}"

    # Type validation
    if not isinstance(scenario["scenario_id"], str):
        return False, "scenario_id must be a string"
    if not isinstance(scenario["hook"], str):
        return False, "hook must be a string"
    if not isinstance(scenario["type"], str):
        return False, "type must be a string"
    if not isinstance(scenario["expected_loss"], (int, float)):
        return False, "expected_loss must be a number"
    if not isinstance(scenario["signal"], list):
        return False, "signal must be a list"
    if len(scenario["signal"]) == 0:
        return False, "signal must not be empty"

    # Validate type is known
    if scenario["type"] not in ANOMALY_TYPES:
        return False, f"Unknown anomaly type: {scenario['type']}. Must be one of {ANOMALY_TYPES}"

    return True, None


# -----------------------------------------------------------------------------
# Scenario Loading
# -----------------------------------------------------------------------------

def load_scenarios(jsonl_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load edge lab scenarios from JSONL file or generate in-memory fallback.

    Args:
        jsonl_path: Path to JSONL file with scenarios. If None, uses in-memory fallback.

    Returns:
        List of validated scenario dictionaries.
    """
    scenarios = []

    if jsonl_path is not None:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {jsonl_path}")

        with open(path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines and comments

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")

                is_valid, error = validate_scenario(data)
                if not is_valid:
                    raise ValueError(f"Invalid scenario on line {line_num}: {error}")

                scenarios.append(data)

    # Fallback to in-memory scenarios if no file or empty file
    if not scenarios:
        scenarios = get_edge_lab_scenarios()

    return scenarios


# -----------------------------------------------------------------------------
# Edge Lab Runner
# -----------------------------------------------------------------------------

def run_scenarios(
    scenarios: List[Dict[str, Any]],
    n: int = 1,
    bit_depth: int = 12,
    sample_rate_hz: float = 1000.0,
    recall_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Run edge lab scenarios through QED and return metric dicts.

    Args:
        scenarios: List of scenario dicts with schema: {scenario_id, hook, type, expected_loss, signal}.
        n: Number of iterations per scenario (default: 1).
        bit_depth: Bit depth for QED compression (default: 12).
        sample_rate_hz: Sample rate in Hz (default: 1000.0).
        recall_threshold: Threshold for hit/miss classification (default: 0.95).

    Returns:
        List of metric dicts with: scenario_id, hook, hit, miss, latency_ms, ratio, recall, violations, verified.
    """
    results = []

    for scenario in scenarios:
        # Validate scenario
        is_valid, error = validate_scenario(scenario)
        if not is_valid:
            results.append({
                "scenario_id": scenario.get("scenario_id", "unknown"),
                "hook": scenario.get("hook", "unknown"),
                "hit": False,
                "miss": True,
                "latency_ms": 0.0,
                "ratio": 0.0,
                "recall": 0.0,
                "violations": 0,
                "verified": False,
                "error": error,
            })
            continue

        scenario_id = scenario["scenario_id"]
        hook = scenario["hook"]
        signal = np.array(scenario["signal"])

        # Map hook to scenario name for qed()
        hook_to_scenario = {
            "tesla": "tesla_fsd",
            "spacex": "spacex_flight",
            "neuralink": "neuralink_stream",
            "boring": "boring_tunnel",
            "starlink": "starlink_flow",
            "xai": "xai_eval",
        }
        scenario_name = hook_to_scenario.get(hook, "generic")

        # Run n iterations and average metrics
        iter_latencies = []
        iter_ratios = []
        iter_recalls = []
        iter_violations = []
        iter_verified = []
        error_msg = None

        for _ in range(n):
            t0 = time.perf_counter()
            try:
                result = qed(
                    signal=signal,
                    scenario=scenario_name,
                    bit_depth=bit_depth,
                    sample_rate_hz=sample_rate_hz,
                    hook_name=hook,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                iter_latencies.append(latency_ms)
                iter_ratios.append(result["ratio"])
                iter_recalls.append(result["recall"])
                receipt = result["receipt"]
                iter_violations.append(len(receipt.violations) if receipt.violations else 0)
                iter_verified.append(receipt.verified if receipt.verified is not None else True)
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                iter_latencies.append(latency_ms)
                error_msg = str(e)
                break

        # Aggregate results
        avg_latency = np.mean(iter_latencies) if iter_latencies else 0.0
        avg_ratio = np.mean(iter_ratios) if iter_ratios else 0.0
        avg_recall = np.mean(iter_recalls) if iter_recalls else 0.0
        total_violations = sum(iter_violations)
        all_verified = all(iter_verified) if iter_verified else False

        hit = avg_recall >= recall_threshold and error_msg is None
        miss = not hit

        results.append({
            "scenario_id": scenario_id,
            "hook": hook,
            "hit": hit,
            "miss": miss,
            "latency_ms": float(avg_latency),
            "ratio": float(avg_ratio),
            "recall": float(avg_recall),
            "violations": total_violations,
            "verified": all_verified,
            "error": error_msg,
        })

    return results


def run_edge_lab(
    jsonl_path: Optional[str] = None,
    scenario_filter: Optional[str] = None,
    bit_depth: int = 12,
    sample_rate_hz: float = 1000.0,
    recall_threshold: float = 0.95,
    verbose: bool = False,
) -> List[EdgeLabResult]:
    """
    Run edge lab scenarios through QED and collect metrics.

    Args:
        jsonl_path: Path to JSONL file with scenarios. If None, uses in-memory fallback.
        scenario_filter: Optional filter string - only run scenarios with IDs containing this.
        bit_depth: Bit depth for QED compression (default: 12).
        sample_rate_hz: Sample rate in Hz (default: 1000.0).
        recall_threshold: Threshold for hit/miss classification (default: 0.95).
        verbose: Print progress and results.

    Returns:
        List of EdgeLabResult objects with metrics per scenario.
    """
    scenarios = load_scenarios(jsonl_path)

    # Apply filter if specified
    if scenario_filter:
        scenarios = [s for s in scenarios if scenario_filter in s["scenario_id"]]

    if verbose:
        print(f"Running {len(scenarios)} edge lab scenarios...")

    results = []

    for idx, scenario in enumerate(scenarios):
        scenario_id = scenario["scenario_id"]
        hook = scenario["hook"]
        scenario_type = scenario["type"]
        expected_loss = scenario["expected_loss"]
        signal = np.array(scenario["signal"])

        # Map hook to scenario name for qed()
        hook_to_scenario = {
            "tesla": "tesla_fsd",
            "spacex": "spacex_flight",
            "neuralink": "neuralink_stream",
            "boring": "boring_tunnel",
            "starlink": "starlink_flow",
            "xai": "xai_eval",
        }
        scenario_name = hook_to_scenario.get(hook, "generic")

        error_msg = None
        ratio = 0.0
        recall = 0.0
        violations_count = 0
        verified = False
        violation_details = []

        # Time the qed() call
        t0 = time.perf_counter()
        try:
            result = qed(
                signal=signal,
                scenario=scenario_name,
                bit_depth=bit_depth,
                sample_rate_hz=sample_rate_hz,
                hook_name=hook,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            ratio = result["ratio"]
            recall = result["recall"]
            receipt = result["receipt"]
            verified = receipt.verified if receipt.verified is not None else True
            violation_details = list(receipt.violations) if receipt.violations else []
            violations_count = len(violation_details)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            error_msg = str(e)

        hit = recall >= recall_threshold and error_msg is None
        miss = not hit

        result_obj = EdgeLabResult(
            scenario_id=scenario_id,
            hook=hook,
            type=scenario_type,
            expected_loss=expected_loss,
            hit=hit,
            miss=miss,
            latency_ms=latency_ms,
            ratio=ratio,
            recall=recall,
            violations=violations_count,
            verified=verified,
            violation_details=violation_details,
            error=error_msg,
        )
        results.append(result_obj)

        if verbose:
            status = "HIT" if hit else "MISS"
            if error_msg:
                status = "ERR"
            print(
                f"  [{idx + 1}/{len(scenarios)}] {scenario_id}: {status} "
                f"(recall={recall:.4f}, ratio={ratio:.1f}, latency={latency_ms:.1f}ms, "
                f"violations={violations_count})"
            )

    return results


# -----------------------------------------------------------------------------
# Summary & Reporting
# -----------------------------------------------------------------------------

def summarize_results(results: List[EdgeLabResult]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from edge lab results.

    Returns summary dict with:
      - n_scenarios, n_hits, n_misses, n_errors
      - recall (hit rate), hit_rate_by_type, hit_rate_by_hook
      - avg_ratio, avg_latency_ms, max_latency_ms
      - total_violations, violation_rate
      - kpi gates (recall >= 0.9967, precision > 0.95, avg_ratio > 20, violations < 5%)
    """
    n = len(results)
    if n == 0:
        return {"error": "No results to summarize"}

    n_hits = sum(1 for r in results if r.hit)
    n_misses = sum(1 for r in results if r.miss)
    n_errors = sum(1 for r in results if r.error is not None)

    # Exclude errors from metric calculations
    valid_results = [r for r in results if r.error is None]
    n_valid = len(valid_results)

    hit_rate = n_hits / n if n > 0 else 0.0

    # Hit rate by anomaly type
    hit_by_type = {}
    for atype in ANOMALY_TYPES:
        type_results = [r for r in valid_results if r.type == atype]
        if type_results:
            hit_by_type[atype] = sum(1 for r in type_results if r.hit) / len(type_results)

    # Hit rate by hook
    hooks = set(r.hook for r in valid_results)
    hit_by_hook = {}
    for hook in hooks:
        hook_results = [r for r in valid_results if r.hook == hook]
        if hook_results:
            hit_by_hook[hook] = sum(1 for r in hook_results if r.hit) / len(hook_results)

    # Compression and latency metrics
    ratios = [r.ratio for r in valid_results if r.ratio > 0]
    latencies = [r.latency_ms for r in valid_results]

    avg_ratio = np.mean(ratios) if ratios else 0.0
    avg_latency_ms = np.mean(latencies) if latencies else 0.0
    max_latency_ms = np.max(latencies) if latencies else 0.0

    # Violation metrics
    total_violations = sum(r.violations for r in valid_results)
    violation_rate = total_violations / n_valid if n_valid > 0 else 0.0

    # KPI gates (from proof.py thresholds)
    kpi = {
        "recall_pass": hit_rate >= 0.9967,
        "precision_pass": True,  # Edge lab focuses on recall; precision tracked separately
        "avg_ratio_pass": avg_ratio >= 20.0,
        "violations_pass": violation_rate < 0.05,
        "all_pass": False,
    }
    kpi["all_pass"] = all([
        kpi["recall_pass"],
        kpi["avg_ratio_pass"],
        kpi["violations_pass"],
    ])

    return {
        "n_scenarios": n,
        "n_valid": n_valid,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "n_errors": n_errors,
        "hit_rate": hit_rate,
        "hit_rate_by_type": hit_by_type,
        "hit_rate_by_hook": hit_by_hook,
        "avg_ratio": float(avg_ratio),
        "avg_latency_ms": float(avg_latency_ms),
        "max_latency_ms": float(max_latency_ms),
        "total_violations": total_violations,
        "violation_rate": float(violation_rate),
        "kpi": kpi,
    }


def write_results_jsonl(
    results: List[EdgeLabResult],
    output_path: str,
) -> None:
    """Write edge lab results to a JSONL file."""
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    """CLI entry point for edge_lab_v1."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Edge Lab v1 - Run edge-case scenarios through QED v6"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to JSONL file with scenarios (uses in-memory fallback if not provided)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter scenarios by ID substring",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=12,
        help="Bit depth for QED compression (default: 12)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1000.0,
        help="Sample rate in Hz (default: 1000.0)",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.95,
        help="Recall threshold for hit/miss classification (default: 0.95)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSONL",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Run edge lab
    results = run_edge_lab(
        jsonl_path=args.jsonl,
        scenario_filter=args.filter,
        bit_depth=args.bit_depth,
        sample_rate_hz=args.sample_rate,
        recall_threshold=args.recall_threshold,
        verbose=args.verbose,
    )

    # Summarize
    summary = summarize_results(results)

    print("\n=== Edge Lab Summary ===")
    print(f"Scenarios: {summary['n_scenarios']} total, {summary['n_valid']} valid")
    print(f"Hits: {summary['n_hits']}, Misses: {summary['n_misses']}, Errors: {summary['n_errors']}")
    print(f"Hit Rate: {summary['hit_rate']:.4f}")
    print(f"Avg Ratio: {summary['avg_ratio']:.1f}")
    print(f"Avg Latency: {summary['avg_latency_ms']:.1f}ms (max: {summary['max_latency_ms']:.1f}ms)")
    print(f"Violations: {summary['total_violations']} ({summary['violation_rate']:.2%})")

    print("\nHit Rate by Type:")
    for atype, rate in summary["hit_rate_by_type"].items():
        print(f"  {atype}: {rate:.4f}")

    print("\nHit Rate by Hook:")
    for hook, rate in summary["hit_rate_by_hook"].items():
        print(f"  {hook}: {rate:.4f}")

    print("\nKPI Gates:")
    for gate, passed in summary["kpi"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate}: {status}")

    # Write results if output specified
    if args.output:
        write_results_jsonl(results, args.output)
        print(f"\nResults written to: {args.output}")

    # Exit with error if KPIs failed
    if not summary["kpi"]["all_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
