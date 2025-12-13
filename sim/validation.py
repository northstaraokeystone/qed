"""
sim/validation.py - Conservation and Validation Functions

Entropy conservation checks and bound validation.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

from typing import List, Tuple

from entropy import emit_receipt

from .constants import (
    TOLERANCE_FLOOR, TOLERANCE_CEILING, ENTROPY_HISTORY_WINDOW
)
from .types_state import SimState
from .measurement import measure_state, measure_observation_cost


def compute_tolerance(state: SimState) -> Tuple[float, dict]:
    """
    Derive tolerance from system's own measurement precision.

    Args:
        state: Current SimState

    Returns:
        Tuple of (tolerance, factors_dict)
    """
    # Entropy variance factor
    if len(state.entropy_trace) >= 2:
        recent_window = state.entropy_trace[-ENTROPY_HISTORY_WINDOW:]
        if len(recent_window) >= 2:
            deltas = [recent_window[i] - recent_window[i-1]
                     for i in range(1, len(recent_window))]
            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
                entropy_variance_factor = variance ** 0.5
            else:
                entropy_variance_factor = 0.0
        else:
            entropy_variance_factor = 0.0
    else:
        entropy_variance_factor = 0.0

    # Population churn factor
    n_active = len(state.active_patterns)
    if n_active > 0:
        total_churn = (state.births_this_cycle +
                      state.deaths_this_cycle +
                      state.superposition_transitions_this_cycle)
        population_churn_factor = total_churn / n_active
    else:
        population_churn_factor = 0.0

    # Wound rate factor
    if len(state.wound_history) >= 20:
        wound_baseline = len(state.wound_history) / max(state.cycle, 1)
    elif state.cycle > 0:
        wound_baseline = len(state.wound_history) / state.cycle
    else:
        wound_baseline = 0.1

    if wound_baseline > 0:
        wound_rate_factor = state.wounds_this_cycle / wound_baseline
    else:
        wound_rate_factor = 0.0

    # Fitness uncertainty factor
    if state.active_patterns:
        fitness_vars = [p.get("fitness_var", 0.1) for p in state.active_patterns]
        fitness_uncertainty_factor = sum(fitness_vars) / len(fitness_vars)
    else:
        fitness_uncertainty_factor = 0.1

    # Compute tolerance
    base = 0.05
    raw = base * (1.0 + entropy_variance_factor + population_churn_factor +
                  wound_rate_factor + fitness_uncertainty_factor)

    tolerance = max(TOLERANCE_FLOOR, min(raw, TOLERANCE_CEILING))

    factors = {
        "entropy_variance_factor": entropy_variance_factor,
        "population_churn_factor": population_churn_factor,
        "wound_rate_factor": wound_rate_factor,
        "fitness_uncertainty_factor": fitness_uncertainty_factor,
        "base": base,
        "raw_value": raw,
        "tolerance": tolerance,
        "was_clamped": raw != tolerance,
        "clamp_direction": (
            "floor" if raw < TOLERANCE_FLOOR else
            "ceiling" if raw > TOLERANCE_CEILING else
            "none"
        )
    }

    return tolerance, factors


def emit_tolerance_receipt(tolerance: float, factors: dict, cycle: int) -> dict:
    """Emit tolerance_measurement receipt."""
    return emit_receipt("tolerance_measurement", {
        "tenant_id": "simulation",
        "cycle": cycle,
        "tolerance": tolerance,
        "entropy_variance_factor": factors["entropy_variance_factor"],
        "population_churn_factor": factors["population_churn_factor"],
        "wound_rate_factor": factors["wound_rate_factor"],
        "fitness_uncertainty_factor": factors["fitness_uncertainty_factor"],
        "raw_value": factors["raw_value"],
        "was_clamped": factors["was_clamped"],
        "clamp_direction": factors["clamp_direction"]
    })


def check_chaos_state(tolerance: float, factors: dict, cycle: int,
                     state: SimState) -> bool:
    """Check if system is in chaos state."""
    if tolerance >= TOLERANCE_CEILING:
        receipt = emit_receipt("anomaly", {
            "tenant_id": "simulation",
            "cycle": cycle,
            "metric": "tolerance",
            "baseline": TOLERANCE_CEILING,
            "delta": tolerance - TOLERANCE_CEILING,
            "classification": "violation",
            "action": "escalate"
        })
        state.receipt_ledger.append(receipt)
        return True

    return False


def validate_conservation(state: SimState) -> bool:
    """
    Validate 2nd law using observer paradigm.

    Conservation: balance = H_boundary + H_observation + H_delta >= -tolerance

    Args:
        state: Current SimState

    Returns:
        bool: True if valid, False if violated
    """
    tolerance, factors = compute_tolerance(state)

    tolerance_receipt = emit_tolerance_receipt(tolerance, factors, state.cycle)
    state.receipt_ledger.append(tolerance_receipt)

    in_chaos = check_chaos_state(tolerance, factors, state.cycle, state)

    H_start = state.H_previous if state.cycle > 0 else state.H_genesis
    H_end = measure_state(state.receipt_ledger, state.vacuum_floor)
    H_delta = H_end - H_start
    balance = state.H_boundary_this_cycle + measure_observation_cost(state.operations_this_cycle) + H_delta

    is_valid = balance >= -tolerance

    if not is_valid:
        receipt = emit_receipt("sim_violation", {
            "cycle": state.cycle,
            "violation_type": "observer_conservation",
            "balance": balance,
            "tolerance": tolerance,
            "H_boundary": state.H_boundary_this_cycle,
            "H_observation": measure_observation_cost(state.operations_this_cycle),
            "H_delta": H_delta,
            "tenant_id": "simulation"
        })
        state.receipt_ledger.append(receipt)
        state.violations.append({
            "cycle": state.cycle,
            "type": "observer_conservation_violation",
            "balance": balance,
            "tolerance": tolerance
        })

    return is_valid


def detect_hidden_risk(state: SimState) -> List[str]:
    """
    Detect patterns with hidden risk.

    Args:
        state: Current SimState

    Returns:
        List of flagged pattern_ids
    """
    flagged = []

    for pattern in state.active_patterns:
        fitness = pattern.get("fitness", 0.0)
        if fitness > 0.8:
            flagged.append(pattern["pattern_id"])

    return flagged


def emit_entropy_state_receipt(state: SimState, cycle: int, H_start: float,
                               H_end: float, H_observation: float, balance: float) -> dict:
    """
    Emit entropy_state receipt at end of each cycle.

    Args:
        state: Current SimState
        cycle: Current cycle number
        H_start: Entropy at cycle start
        H_end: Entropy at cycle end
        H_observation: Observation cost this cycle
        balance: Conservation balance

    Returns:
        Receipt dict
    """
    H_delta = H_end - H_start

    tolerance, _ = compute_tolerance(state)

    if balance >= -tolerance:
        balance_status = "conserved"
    else:
        balance_status = "violation"

    return emit_receipt("entropy_state", {
        "tenant_id": "simulation",
        "cycle": cycle,
        "H_start": H_start,
        "H_end": H_end,
        "H_delta": H_delta,
        "H_boundary": state.H_boundary_this_cycle,
        "H_observation": H_observation,
        "operations_count": state.operations_this_cycle,
        "balance": balance,
        "balance_status": balance_status,
        "H_genesis": state.H_genesis
    })
