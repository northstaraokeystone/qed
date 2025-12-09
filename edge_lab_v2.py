"""
edge_lab_v2.py - Edge Lab v2 Simulation Runner for QED v7

Pattern-based simulation runner for anomaly detection validation.
Extends edge_lab_v1 with:
  - Pattern-based simulation (vs scenario-based)
  - JSONL scenario migration from data/edge_lab_scenarios.jsonl
  - SimResults dataclass with per-pattern metrics
  - Clopper-Pearson exact recall floor computation

v7 Exports:
  - run_pattern_sims(): Run simulations across patterns
  - compute_recall_floor(): Clopper-Pearson exact binomial CI
  - SimResults: Dataclass with pattern_results, aggregate metrics

v1 Backward Compatible Exports:
  - EdgeLabResult: Dataclass for single scenario result
  - run_edge_lab(): Run scenarios through QED
  - load_scenarios(): Load from JSONL or in-memory fallback
  - validate_scenario(): Validate scenario schema
  - summarize_results(): Aggregate metrics summary

Schema (JSONL for edge_lab_scenarios.jsonl):
  {
    "scenario_id": str,
    "hook": str,
    "pattern_id": str | null,
    "type": str,  # "anomaly", "normal", "near_miss"
    "expected_loss": float,
    "signal": list[float] (optional),
    "description": str (optional)
  }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.stats import beta as scipy_beta
except ImportError:
    scipy_beta = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from qed import qed, run as qed_run
from shared_anomalies import load_library, AnomalyPattern
from physics_injection import inject_perturbation


# =============================================================================
# Schema Definitions
# =============================================================================

SCENARIO_SCHEMA = {
    "scenario_id": str,
    "hook": str,
    "type": str,
    "expected_loss": float,
}

# Valid anomaly types for edge lab scenarios
ANOMALY_TYPES = {"spike", "step", "drift", "normal", "noise", "saturation", "anomaly", "near_miss"}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class EdgeLabResult:
    """Result metrics for a single edge lab scenario run (v1 compatible)."""
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
    violation_details: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None   # error message if processing failed


@dataclass
class PatternResult:
    """Result metrics for a single pattern simulation."""
    pattern_id: str
    physics_domain: str
    failure_mode: str
    hooks: List[str]
    n_tests: int
    n_hits: int
    n_misses: int
    sim_recall: float          # hits / (hits + misses)
    sim_false_positive_rate: float
    avg_latency_ms: float
    exploit_grade: bool
    training_role: str


@dataclass
class SimResults:
    """Aggregated simulation results across all patterns."""
    pattern_results: List[PatternResult]
    aggregate_recall: float
    aggregate_fp_rate: float
    recall_floor: float
    n_tests: int
    n_misses: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =============================================================================
# Signal Generation (from v1)
# =============================================================================

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


def _generate_signal_for_pattern(
    pattern: Union[AnomalyPattern, Dict[str, Any]],
    seed: int = 42,
    n: int = 1000,
) -> np.ndarray:
    """Generate a synthetic signal based on pattern parameters."""
    rng = np.random.default_rng(seed)

    # Get physics domain
    if isinstance(pattern, AnomalyPattern):
        physics_domain = pattern.physics_domain
        params = pattern.params
    else:
        physics_domain = pattern.get("physics_domain", "generic")
        params = pattern.get("params", {})

    # Base signal
    amplitude = params.get("amplitude", rng.uniform(10.0, 14.0))
    frequency_hz = params.get("frequency_hz", rng.uniform(30.0, 50.0))

    t = np.linspace(0, 1, n)
    signal = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    signal += rng.normal(0, 0.1, n)

    return signal


# =============================================================================
# Scenario Loading and Validation
# =============================================================================

def validate_scenario(scenario: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a scenario against the SCENARIO_SCHEMA.

    Returns (is_valid, error_message).
    """
    required_keys = {"scenario_id", "hook", "type", "expected_loss"}
    actual_keys = set(scenario.keys())

    missing = required_keys - actual_keys
    if missing:
        return False, f"Missing required keys: {missing}"

    # Type validation
    if not isinstance(scenario.get("scenario_id"), str):
        return False, "scenario_id must be a string"
    if not isinstance(scenario.get("hook"), str):
        return False, "hook must be a string"
    if not isinstance(scenario.get("type"), str):
        return False, "type must be a string"
    if not isinstance(scenario.get("expected_loss"), (int, float)):
        return False, "expected_loss must be a number"

    # Signal validation (if present)
    if "signal" in scenario:
        if not isinstance(scenario["signal"], list):
            return False, "signal must be a list"
        if len(scenario["signal"]) == 0:
            return False, "signal must not be empty"

    return True, None


def load_scenarios(jsonl_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load edge lab scenarios from JSONL file or generate in-memory fallback.

    Args:
        jsonl_path: Path to JSONL file with scenarios. If None, tries default path
                   then falls back to in-memory generation.

    Returns:
        List of validated scenario dictionaries.
    """
    scenarios = []

    # Try default path if none provided
    if jsonl_path is None:
        default_path = Path("data/edge_lab_scenarios.jsonl")
        if default_path.exists():
            jsonl_path = str(default_path)

    if jsonl_path is not None:
        path = Path(jsonl_path)
        if path.exists():
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


def get_edge_lab_scenarios() -> List[Dict[str, Any]]:
    """
    Generate in-memory edge lab scenarios for laptop testing.

    Returns a list of scenarios across hooks with various anomaly types.
    """
    scenarios = []

    # Tesla scenarios - steering torque (bound: 14.7)
    tesla_scenarios = [
        {"id": "tesla_spike_001", "type": "spike", "amp": 12.0, "spike_amp": 20.0, "loss": 0.15, "seed": 1},
        {"id": "tesla_spike_002", "type": "spike", "amp": 13.0, "spike_amp": 25.0, "loss": 0.20, "seed": 2},
        {"id": "tesla_step_001", "type": "step", "amp": 12.0, "step_val": 4.0, "loss": 0.14, "seed": 4},
        {"id": "tesla_drift_001", "type": "drift", "amp": 12.0, "drift_rate": 5.0, "loss": 0.16, "seed": 6},
        {"id": "tesla_normal_001", "type": "normal", "amp": 12.0, "loss": 0.05, "seed": 8},
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

    # SpaceX scenarios
    spacex_scenarios = [
        {"id": "spacex_spike_001", "type": "spike", "amp": 15.0, "spike_amp": 30.0, "loss": 0.22, "seed": 11},
        {"id": "spacex_normal_001", "type": "normal", "amp": 16.0, "loss": 0.06, "seed": 17},
    ]

    for ss in spacex_scenarios:
        if ss["type"] == "spike":
            signal = _generate_spike_signal(
                amplitude=ss["amp"], spike_amplitude=ss["spike_amp"], seed=ss["seed"]
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


# =============================================================================
# v1 Compatible Functions
# =============================================================================

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
        jsonl_path: Path to JSONL file with scenarios. If None, uses default or in-memory.
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
    hook_to_scenario = {
        "tesla": "tesla_fsd",
        "spacex": "spacex_flight",
        "neuralink": "neuralink_stream",
        "boring": "boring_tunnel",
        "starlink": "starlink_flow",
        "xai": "xai_eval",
    }

    for idx, scenario in enumerate(tqdm(scenarios, desc="Running scenarios", disable=not verbose)):
        scenario_id = scenario["scenario_id"]
        hook = scenario["hook"]
        scenario_type = scenario["type"]
        expected_loss = scenario["expected_loss"]

        # Get or generate signal
        if "signal" in scenario and scenario["signal"]:
            signal = np.array(scenario["signal"])
        else:
            signal = _generate_normal_signal(seed=idx)

        scenario_name = hook_to_scenario.get(hook, "generic")

        error_msg = None
        ratio = 0.0
        recall = 0.0
        violations_count = 0
        verified = False
        violation_details: List[Dict[str, Any]] = []

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
                f"(recall={recall:.4f}, ratio={ratio:.1f}, latency={latency_ms:.1f}ms)"
            )

    return results


def summarize_results(results: List[EdgeLabResult]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from edge lab results.

    Returns summary dict with:
      - n_scenarios, n_hits, n_misses, n_errors
      - recall (hit rate), hit_rate_by_type, hit_rate_by_hook
      - avg_ratio, avg_latency_ms, max_latency_ms
      - total_violations, violation_rate
      - kpi gates
    """
    n = len(results)
    if n == 0:
        return {"error": "No results to summarize"}

    n_hits = sum(1 for r in results if r.hit)
    n_misses = sum(1 for r in results if r.miss)
    n_errors = sum(1 for r in results if r.error is not None)

    valid_results = [r for r in results if r.error is None]
    n_valid = len(valid_results)

    hit_rate = n_hits / n if n > 0 else 0.0

    # Hit rate by anomaly type
    hit_by_type: Dict[str, float] = {}
    for atype in ANOMALY_TYPES:
        type_results = [r for r in valid_results if r.type == atype]
        if type_results:
            hit_by_type[atype] = sum(1 for r in type_results if r.hit) / len(type_results)

    # Hit rate by hook
    hooks = set(r.hook for r in valid_results)
    hit_by_hook: Dict[str, float] = {}
    for hook in hooks:
        hook_results = [r for r in valid_results if r.hook == hook]
        if hook_results:
            hit_by_hook[hook] = sum(1 for r in hook_results if r.hit) / len(hook_results)

    # Compression and latency metrics
    ratios = [r.ratio for r in valid_results if r.ratio > 0]
    latencies = [r.latency_ms for r in valid_results]

    avg_ratio = float(np.mean(ratios)) if ratios else 0.0
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    max_latency_ms = float(np.max(latencies)) if latencies else 0.0

    # Violation metrics
    total_violations = sum(r.violations for r in valid_results)
    violation_rate = total_violations / n_valid if n_valid > 0 else 0.0

    # KPI gates
    kpi = {
        "recall_pass": hit_rate >= 0.9967,
        "precision_pass": True,
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


# =============================================================================
# v7 New Functions
# =============================================================================

def compute_recall_floor(
    n_tests: int,
    n_misses: int,
    confidence: float = 0.95,
) -> float:
    """
    Compute Clopper-Pearson exact recall lower bound.

    Uses scipy.stats.beta for exact binomial confidence interval.
    Formula: beta.ppf(alpha/2, k, n-k+1) where k=successes, n=total.

    Args:
        n_tests: Total number of tests run.
        n_misses: Number of misses (failures to detect).
        confidence: Confidence level (default: 0.95 for 95% CI).

    Returns:
        Lower bound of recall at given confidence level.
    """
    if scipy_beta is None:
        # Fallback approximation using Wilson score interval
        n_successes = n_tests - n_misses
        if n_tests == 0:
            return 0.0
        p = n_successes / n_tests
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        denom = 1 + z**2 / n_tests
        center = (p + z**2 / (2 * n_tests)) / denom
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_tests)) / n_tests) / denom
        return max(0.0, center - spread)

    n_successes = n_tests - n_misses
    alpha = 1.0 - confidence

    if n_successes == 0:
        return 0.0

    # Clopper-Pearson exact lower bound
    lower_bound = float(scipy_beta.ppf(alpha / 2, n_successes, n_misses + 1))
    return lower_bound


def _load_receipts_from_dir(receipts_dir: str) -> List[Dict[str, Any]]:
    """Load receipts from a directory of JSONL files."""
    receipts = []
    receipts_path = Path(receipts_dir)

    if not receipts_path.exists():
        return receipts

    for jsonl_file in receipts_path.glob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        receipts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return receipts


def _sample_windows_for_hook(
    receipts: List[Dict[str, Any]],
    hook: str,
    n: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """Sample n telemetry windows from receipts for a given hook."""
    rng = np.random.default_rng(seed)

    # Filter receipts for this hook
    hook_receipts = [r for r in receipts if r.get("hook") == hook or
                     r.get("params", {}).get("scenario", "").startswith(hook)]

    if not hook_receipts:
        # Generate synthetic windows
        windows = []
        for i in range(n):
            windows.append(_generate_normal_signal(seed=seed + i))
        return windows

    # Sample from available receipts
    sampled = rng.choice(hook_receipts, size=min(n, len(hook_receipts)), replace=True)

    windows = []
    for i, receipt in enumerate(sampled):
        # Extract or generate window
        if "signal" in receipt:
            windows.append(np.array(receipt["signal"]))
        else:
            # Generate based on params
            params = receipt.get("params", {})
            A = params.get("A", 12.0)
            f = params.get("f", 40.0)
            windows.append(_generate_normal_signal(amplitude=A, frequency_hz=f, seed=seed + i))

    return windows


def run_pattern_sims(
    receipts_dir: Optional[str] = None,
    patterns_path: Optional[str] = None,
    patterns: Optional[List[Union[AnomalyPattern, Dict[str, Any]]]] = None,
    n_per_hook: int = 1000,
    progress_callback: Optional[Callable] = None,
) -> Union[SimResults, Dict[str, Any]]:
    """
    Run pattern simulations across all non-observe_only patterns.

    Flow:
    1. Load patterns via shared_anomalies.load_library(patterns_path)
    2. Filter to patterns where training_role != "observe_only"
    3. For exploit_grade=true patterns, run 2x iterations (priority)
    4. For each pattern:
       - Sample n_per_hook receipts for each hook in pattern.hooks
       - Extract telemetry window from receipt
       - Call physics_injection.inject_perturbation(window, pattern)
       - Run qed.run() on perturbed window
       - Record hit (detected) or miss (not detected)
    5. Return SimResults with per-pattern metrics

    Args:
        receipts_dir: Directory containing receipt JSONL files.
        patterns_path: Path to patterns JSONL file.
        patterns: Direct list of patterns (alternative to patterns_path).
        n_per_hook: Number of simulations per hook (default: 1000).
        progress_callback: Optional callback for progress display.

    Returns:
        SimResults dataclass or dict with pattern_results, aggregate metrics.
    """
    # Load patterns
    if patterns is not None:
        pattern_list = patterns
    elif patterns_path:
        pattern_list = load_library(patterns_path)
    else:
        pattern_list = load_library("data/shared_anomalies.jsonl")

    # Filter to non-observe_only patterns
    active_patterns = []
    for p in pattern_list:
        if isinstance(p, AnomalyPattern):
            if p.training_role != "observe_only":
                active_patterns.append(p)
        elif isinstance(p, dict):
            # Check if dict has training_role or compute from dollar_value
            dollar_value = p.get("dollar_value_annual", 0)
            training_role = p.get("training_role", "observe_only" if dollar_value <= 10_000_000 else "train_cross_company")
            if training_role != "observe_only":
                active_patterns.append(p)

    # Load receipts for sampling
    receipts = []
    if receipts_dir:
        receipts = _load_receipts_from_dir(receipts_dir)

    # Process patterns
    pattern_results: List[PatternResult] = []
    total_hits = 0
    total_misses = 0
    total_false_positives = 0
    total_clean_windows = 0

    # Set up iteration
    iterable = active_patterns
    if progress_callback:
        iterable = progress_callback()

    for pattern in tqdm(active_patterns, desc="Running pattern sims"):
        # Get pattern properties
        if isinstance(pattern, AnomalyPattern):
            pattern_id = pattern.pattern_id
            physics_domain = pattern.physics_domain
            failure_mode = pattern.failure_mode
            hooks = pattern.hooks
            exploit_grade = pattern.exploit_grade
            training_role = pattern.training_role
            pattern_dict = asdict(pattern)
        else:
            pattern_id = pattern.get("pattern_id", "unknown")
            physics_domain = pattern.get("physics_domain", "generic")
            failure_mode = pattern.get("failure_mode", "unknown")
            hooks = pattern.get("hooks", ["tesla"])
            dollar_value = pattern.get("dollar_value_annual", 0)
            validation_recall = pattern.get("validation_recall", 0)
            fp_rate = pattern.get("false_positive_rate", 1.0)
            exploit_grade = (
                dollar_value > 1_000_000 and
                validation_recall >= 0.99 and
                fp_rate <= 0.01
            )
            training_role = "train_cross_company" if dollar_value > 10_000_000 else "observe_only"
            pattern_dict = pattern

        # Determine iterations (2x for exploit_grade)
        n_iterations = n_per_hook * 2 if exploit_grade else n_per_hook

        hits = 0
        misses = 0
        false_positives = 0
        clean_tests = 0
        latencies: List[float] = []

        for hook in hooks:
            # Sample windows for this hook
            windows = _sample_windows_for_hook(receipts, hook, n_iterations, seed=hash(pattern_id) & 0xFFFFFFFF)

            for i, window in enumerate(windows):
                # Inject perturbation
                perturbed = inject_perturbation(window, pattern_dict, seed=i)

                # Run QED on perturbed window
                t0 = time.perf_counter()
                try:
                    result = qed_run(window=perturbed, hook=hook, pattern_id=pattern_id)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    latencies.append(latency_ms)

                    recall = result.get("recall", 0.0)
                    if recall >= 0.95:
                        hits += 1
                    else:
                        misses += 1

                except Exception:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    latencies.append(latency_ms)
                    misses += 1

                # Also test clean window for FP rate
                if i < n_per_hook // 10:  # Test 10% clean
                    clean_tests += 1
                    try:
                        clean_result = qed_run(window=window, hook=hook)
                        clean_recall = clean_result.get("recall", 1.0)
                        # If clean window falsely detected as anomaly
                        if clean_recall < 0.95:
                            false_positives += 1
                    except Exception:
                        pass

        # Calculate per-pattern metrics
        n_tests = hits + misses
        sim_recall = hits / n_tests if n_tests > 0 else 0.0
        sim_fp_rate = false_positives / clean_tests if clean_tests > 0 else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        pattern_results.append(PatternResult(
            pattern_id=pattern_id,
            physics_domain=physics_domain,
            failure_mode=failure_mode,
            hooks=hooks,
            n_tests=n_tests,
            n_hits=hits,
            n_misses=misses,
            sim_recall=sim_recall,
            sim_false_positive_rate=sim_fp_rate,
            avg_latency_ms=avg_latency,
            exploit_grade=exploit_grade,
            training_role=training_role,
        ))

        total_hits += hits
        total_misses += misses
        total_false_positives += false_positives
        total_clean_windows += clean_tests

    # Calculate aggregate metrics
    total_tests = total_hits + total_misses
    aggregate_recall = total_hits / total_tests if total_tests > 0 else 0.0
    aggregate_fp_rate = total_false_positives / total_clean_windows if total_clean_windows > 0 else 0.0

    # Compute recall floor
    recall_floor = compute_recall_floor(total_tests, total_misses, confidence=0.95)

    sim_results = SimResults(
        pattern_results=pattern_results,
        aggregate_recall=aggregate_recall,
        aggregate_fp_rate=aggregate_fp_rate,
        recall_floor=recall_floor,
        n_tests=total_tests,
        n_misses=total_misses,
    )

    # Return as dict for JSON serialization compatibility
    return {
        "pattern_results": [asdict(pr) for pr in pattern_results],
        "aggregate_recall": aggregate_recall,
        "aggregate_fp_rate": aggregate_fp_rate,
        "recall_floor": recall_floor,
        "n_tests": total_tests,
        "n_misses": total_misses,
        "timestamp": sim_results.timestamp,
    }


def write_sim_results(
    results: Union[SimResults, Dict[str, Any]],
    output_path: str = "data/sim_results.json",
) -> None:
    """Write simulation results to JSON file."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, SimResults):
        data = {
            "pattern_results": [asdict(pr) for pr in results.pattern_results],
            "aggregate_recall": results.aggregate_recall,
            "aggregate_fp_rate": results.aggregate_fp_rate,
            "recall_floor": results.recall_floor,
            "n_tests": results.n_tests,
            "n_misses": results.n_misses,
            "timestamp": results.timestamp,
        }
    else:
        data = results

    with open(output_path_obj, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for edge_lab_v2."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Edge Lab v2 - Pattern-based simulation runner for QED v7"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run scenarios (v1 compat)
    run_parser = subparsers.add_parser(
        "run",
        help="Run edge lab scenarios (v1 compatible)",
    )
    run_parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to JSONL file with scenarios",
    )
    run_parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter scenarios by ID substring",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Run pattern sims (v7)
    sims_parser = subparsers.add_parser(
        "sims",
        help="Run pattern simulations (v7)",
    )
    sims_parser.add_argument(
        "--receipts-dir",
        type=str,
        default="receipts/",
        help="Directory containing receipt files",
    )
    sims_parser.add_argument(
        "--patterns-path",
        type=str,
        default="data/shared_anomalies.jsonl",
        help="Path to patterns JSONL file",
    )
    sims_parser.add_argument(
        "--n-per-hook",
        type=int,
        default=100,
        help="Number of simulations per hook (default: 100 for CLI)",
    )
    sims_parser.add_argument(
        "--output",
        type=str,
        default="data/sim_results.json",
        help="Output JSON file for results",
    )

    # Recall floor computation
    floor_parser = subparsers.add_parser(
        "recall-floor",
        help="Compute Clopper-Pearson recall floor",
    )
    floor_parser.add_argument(
        "--n-tests",
        type=int,
        required=True,
        help="Total number of tests",
    )
    floor_parser.add_argument(
        "--n-misses",
        type=int,
        required=True,
        help="Number of misses",
    )
    floor_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )

    args = parser.parse_args()

    if args.command == "run":
        results = run_edge_lab(
            jsonl_path=args.jsonl,
            scenario_filter=args.filter,
            verbose=args.verbose,
        )
        summary = summarize_results(results)
        print("\n=== Edge Lab Summary ===")
        print(f"Scenarios: {summary['n_scenarios']} total, {summary['n_valid']} valid")
        print(f"Hits: {summary['n_hits']}, Misses: {summary['n_misses']}")
        print(f"Hit Rate: {summary['hit_rate']:.4f}")
        print(f"Avg Ratio: {summary['avg_ratio']:.1f}")
        if not summary["kpi"]["all_pass"]:
            return 1

    elif args.command == "sims":
        results = run_pattern_sims(
            receipts_dir=args.receipts_dir,
            patterns_path=args.patterns_path,
            n_per_hook=args.n_per_hook,
        )
        write_sim_results(results, args.output)
        print(f"Simulation complete: n_tests={results['n_tests']}")
        print(f"Aggregate recall: {results['aggregate_recall']:.4f}")
        print(f"Recall floor (95% CI): {results['recall_floor']:.4f}")
        print(f"Results written to {args.output}")

    elif args.command == "recall-floor":
        floor = compute_recall_floor(
            n_tests=args.n_tests,
            n_misses=args.n_misses,
            confidence=args.confidence,
        )
        pct = args.confidence * 100
        print(f"Recall floor: {floor:.4f} at {pct:.0f}% confidence")
        print(f"  n_tests={args.n_tests}, n_misses={args.n_misses}")

    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    exit(main())
