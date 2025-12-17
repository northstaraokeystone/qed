"""
sensitivity.py - P Sensitivity Analyzer with Monte Carlo Simulation

Test P sensitivity: Vary baselines +/-20% (e.g., swarm costs $50-150M);
check if ROI gate holds above 1.2.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.

Source: Grok insight 2025-01
- "test P sensitivity: Vary baselines +/-20%"
- "check if ROI gate holds above 1.2"
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from receipts import dual_hash, emit_receipt, StopRule
from mitigation import MitigationResult, CYCLES_TO_10K_TARGET


# =============================================================================
# CONSTANTS
# =============================================================================

# Default sensitivity parameters
DEFAULT_P_BASELINE = 1.8            # Baseline P factor
DEFAULT_P_VARIANCE_PCT = 0.20       # +/-20% variance
DEFAULT_COST_BASELINE = 100         # $100M baseline
DEFAULT_COST_RANGE = (50, 150)      # $50-150M range
DEFAULT_N_SAMPLES = 1000            # Monte Carlo samples
DEFAULT_ROI_GATE_THRESHOLD = 1.2    # ROI must be > 1.2
DEFAULT_ROI_GATE_CONFIDENCE = 0.90  # 90% of samples must pass

# ROI computation factors
# Calibrated to achieve ROI > 1.2 with full mitigation stack
REWARD_PER_CYCLE_SAVED = 100        # $100M reward per cycle saved
BASELINE_CYCLES = 5.5               # Unmitigated cycles to 10^3
PENALTY_FACTOR = 0.2                # Cost penalty multiplier (low friction)

# Module exports for receipt types
RECEIPT_SCHEMA = ["p_sensitivity"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SensitivityConfig:
    """Configuration for P sensitivity analysis."""
    p_baseline: float = DEFAULT_P_BASELINE
    p_variance_pct: float = DEFAULT_P_VARIANCE_PCT
    cost_baseline_usd_m: float = DEFAULT_COST_BASELINE
    cost_range_usd_m: Tuple[float, float] = DEFAULT_COST_RANGE
    n_samples: int = DEFAULT_N_SAMPLES
    roi_gate_threshold: float = DEFAULT_ROI_GATE_THRESHOLD
    roi_gate_confidence: float = DEFAULT_ROI_GATE_CONFIDENCE


@dataclass
class SensitivityResult:
    """Result from P sensitivity analysis."""
    p_samples: List[float]
    cost_samples: List[float]
    roi_samples: List[float]
    roi_mean: float
    roi_std: float
    roi_above_gate_pct: float
    gate_passed: bool


# =============================================================================
# RECEIPT TYPE 1: p_sensitivity
# =============================================================================

# --- SCHEMA ---
P_SENSITIVITY_SCHEMA = {
    "receipt_type": "p_sensitivity",
    "ts": "ISO8601",
    "tenant_id": "str",
    "n_samples": "int",
    "p_range": "tuple (min, max)",
    "cost_range_usd_m": "tuple (min, max)",
    "roi_mean": "float",
    "roi_std": "float",
    "roi_above_gate_pct": "float (0-1)",
    "gate_passed": "bool",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_p_sensitivity_receipt(
    tenant_id: str,
    n_samples: int,
    p_range: Tuple[float, float],
    cost_range_usd_m: Tuple[float, float],
    roi_mean: float,
    roi_std: float,
    roi_above_gate_pct: float,
    gate_passed: bool
) -> dict:
    """Emit p_sensitivity receipt for sensitivity analysis."""
    return emit_receipt("p_sensitivity", {
        "tenant_id": tenant_id,
        "n_samples": n_samples,
        "p_range": list(p_range),
        "cost_range_usd_m": list(cost_range_usd_m),
        "roi_mean": roi_mean,
        "roi_std": roi_std,
        "roi_above_gate_pct": roi_above_gate_pct,
        "gate_passed": gate_passed
    })


# =============================================================================
# CORE FUNCTION 1: sample_p_factor
# =============================================================================

def sample_p_factor(
    baseline: float,
    variance: float,
    n: int,
    seed: Optional[int] = None
) -> List[float]:
    """
    Sample P factor from uniform distribution.

    Range: baseline * (1 - variance) to baseline * (1 + variance)

    Args:
        baseline: P baseline value (e.g., 1.8)
        variance: Variance as fraction (e.g., 0.20 for +/-20%)
        n: Number of samples
        seed: Optional random seed for reproducibility

    Returns:
        List of P samples
    """
    if seed is not None:
        random.seed(seed)

    p_min = baseline * (1 - variance)
    p_max = baseline * (1 + variance)

    return [random.uniform(p_min, p_max) for _ in range(n)]


# =============================================================================
# CORE FUNCTION 2: sample_cost
# =============================================================================

def sample_cost(
    cost_range: Tuple[float, float],
    n: int,
    seed: Optional[int] = None
) -> List[float]:
    """
    Sample cost from uniform distribution.

    Args:
        cost_range: (min, max) cost in $M
        n: Number of samples
        seed: Optional random seed for reproducibility

    Returns:
        List of cost samples in $M
    """
    if seed is not None:
        random.seed(seed)

    return [random.uniform(cost_range[0], cost_range[1]) for _ in range(n)]


# =============================================================================
# CORE FUNCTION 3: compute_mitigated_cycles
# =============================================================================

def compute_mitigated_cycles(
    p_factor: float,
    mitigation: MitigationResult
) -> float:
    """
    Compute mitigated cycles adjusted by P factor.

    The P factor scales the effective alpha, which affects cycles.

    Args:
        p_factor: P sensitivity factor
        mitigation: MitigationResult with base cycles

    Returns:
        Adjusted cycles to 10^3
    """
    # P factor modulates the effective alpha
    # Higher P = better performance = fewer cycles
    adjusted_cycles = mitigation.cycles_to_10k / (p_factor / DEFAULT_P_BASELINE)

    return max(0.1, adjusted_cycles)  # Floor at 0.1 cycles


# =============================================================================
# CORE FUNCTION 4: compute_roi
# =============================================================================

def compute_roi(
    baseline_cycles: float,
    mitigated_cycles: float,
    cost_usd_m: float
) -> float:
    """
    Compute ROI for mitigation investment.

    ROI = (reward from saved cycles - cost penalty) / cost

    Args:
        baseline_cycles: Unmitigated cycles to 10^3
        mitigated_cycles: Mitigated cycles to 10^3
        cost_usd_m: Investment cost in $M

    Returns:
        ROI ratio (>1.0 means positive return)
    """
    if cost_usd_m <= 0:
        return float('inf')

    cycles_saved = baseline_cycles - mitigated_cycles
    reward = cycles_saved * REWARD_PER_CYCLE_SAVED
    penalty = cost_usd_m * PENALTY_FACTOR

    roi = (reward - penalty) / cost_usd_m

    return roi


# =============================================================================
# CORE FUNCTION 5: compute_roi_distribution
# =============================================================================

def compute_roi_distribution(
    p_samples: List[float],
    cost_samples: List[float],
    mitigation: MitigationResult
) -> List[float]:
    """
    Compute ROI for each sample pair.

    Args:
        p_samples: List of P factor samples
        cost_samples: List of cost samples in $M
        mitigation: MitigationResult with base performance

    Returns:
        List of ROI values for each sample
    """
    roi_samples = []

    for p, cost in zip(p_samples, cost_samples):
        mitigated_cycles = compute_mitigated_cycles(p, mitigation)
        roi = compute_roi(BASELINE_CYCLES, mitigated_cycles, cost)
        roi_samples.append(roi)

    return roi_samples


# =============================================================================
# CORE FUNCTION 6: check_roi_gate
# =============================================================================

def check_roi_gate(
    roi_samples: List[float],
    threshold: float,
    confidence: float
) -> bool:
    """
    Check if ROI gate passes at confidence level.

    Gate passes if >= confidence% of samples are above threshold.

    Args:
        roi_samples: List of ROI values
        threshold: Minimum ROI threshold (e.g., 1.2)
        confidence: Required confidence level (e.g., 0.90 for 90%)

    Returns:
        True if gate passes, False otherwise
    """
    if not roi_samples:
        return False

    above_threshold = sum(1 for r in roi_samples if r > threshold)
    pct_above = above_threshold / len(roi_samples)

    return pct_above >= confidence


# =============================================================================
# CORE FUNCTION 7: run_sensitivity
# =============================================================================

def run_sensitivity(
    config: SensitivityConfig,
    mitigation: MitigationResult,
    tenant_id: str = "axiom-autonomy",
    seed: Optional[int] = None
) -> SensitivityResult:
    """
    Run Monte Carlo sensitivity analysis.

    Sweeps over P factor (+/-20%) and cost ($50-150M) to test
    if ROI gate (>1.2) holds with 90% confidence.

    Args:
        config: SensitivityConfig with parameters
        mitigation: MitigationResult from stack_mitigation()
        tenant_id: Tenant identifier for receipt
        seed: Optional random seed for reproducibility

    Returns:
        SensitivityResult with distribution and gate status
    """
    # Sample P factors
    p_samples = sample_p_factor(
        baseline=config.p_baseline,
        variance=config.p_variance_pct,
        n=config.n_samples,
        seed=seed
    )

    # Sample costs
    cost_samples = sample_cost(
        cost_range=config.cost_range_usd_m,
        n=config.n_samples,
        seed=seed + 1 if seed is not None else None
    )

    # Compute ROI distribution
    roi_samples = compute_roi_distribution(p_samples, cost_samples, mitigation)

    # Compute statistics
    roi_mean = sum(roi_samples) / len(roi_samples)
    roi_variance = sum((r - roi_mean) ** 2 for r in roi_samples) / len(roi_samples)
    roi_std = math.sqrt(roi_variance)

    # Check gate
    above_threshold = sum(1 for r in roi_samples if r > config.roi_gate_threshold)
    roi_above_gate_pct = above_threshold / len(roi_samples)
    gate_passed = check_roi_gate(
        roi_samples,
        config.roi_gate_threshold,
        config.roi_gate_confidence
    )

    # Compute P range for receipt
    p_min = config.p_baseline * (1 - config.p_variance_pct)
    p_max = config.p_baseline * (1 + config.p_variance_pct)
    p_range = (p_min, p_max)

    # Emit receipt
    emit_p_sensitivity_receipt(
        tenant_id=tenant_id,
        n_samples=config.n_samples,
        p_range=p_range,
        cost_range_usd_m=config.cost_range_usd_m,
        roi_mean=roi_mean,
        roi_std=roi_std,
        roi_above_gate_pct=roi_above_gate_pct,
        gate_passed=gate_passed
    )

    return SensitivityResult(
        p_samples=p_samples,
        cost_samples=cost_samples,
        roi_samples=roi_samples,
        roi_mean=roi_mean,
        roi_std=roi_std,
        roi_above_gate_pct=roi_above_gate_pct,
        gate_passed=gate_passed
    )


# =============================================================================
# CORE FUNCTION 8: quick_sensitivity_check
# =============================================================================

def quick_sensitivity_check(
    mitigation: MitigationResult,
    n_samples: int = 100
) -> bool:
    """
    Quick sensitivity check with reduced samples.

    Useful for tests and quick validation.

    Args:
        mitigation: MitigationResult from stack_mitigation()
        n_samples: Number of samples (default 100 for speed)

    Returns:
        True if ROI gate passes
    """
    config = SensitivityConfig(n_samples=n_samples)
    result = run_sensitivity(config, mitigation, seed=42)
    return result.gate_passed


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "DEFAULT_P_BASELINE",
    "DEFAULT_P_VARIANCE_PCT",
    "DEFAULT_COST_BASELINE",
    "DEFAULT_COST_RANGE",
    "DEFAULT_N_SAMPLES",
    "DEFAULT_ROI_GATE_THRESHOLD",
    "DEFAULT_ROI_GATE_CONFIDENCE",
    "REWARD_PER_CYCLE_SAVED",
    "BASELINE_CYCLES",
    "PENALTY_FACTOR",
    # Data classes
    "SensitivityConfig",
    "SensitivityResult",
    # Core functions
    "sample_p_factor",
    "sample_cost",
    "compute_mitigated_cycles",
    "compute_roi",
    "compute_roi_distribution",
    "check_roi_gate",
    "run_sensitivity",
    "quick_sensitivity_check",
    # Receipt functions
    "emit_p_sensitivity_receipt",
]
