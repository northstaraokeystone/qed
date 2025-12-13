"""
sim/vacuum.py - Vacuum Fluctuation and Virtual Pattern Functions

Functions for vacuum fluctuation, spontaneous emergence, and Hawking flux.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import random
from typing import List, Optional

from entropy import emit_receipt
from autocatalysis import autocatalysis_check

from .constants import (
    PLANCK_ENTROPY_BASE, VACUUM_VARIANCE, GENESIS_THRESHOLD,
    VIRTUAL_LIFESPAN, HAWKING_COEFFICIENT, FLUX_WINDOW,
    CRITICALITY_ALERT_THRESHOLD, CRITICALITY_PHASE_TRANSITION,
    ALERT_COOLDOWN_CYCLES, PatternState
)
from .state import SimState
from .measurement import measure_boundary_crossing


def vacuum_fluctuation() -> float:
    """
    Generate fluctuating zero-point energy.

    Vacuum isn't static - it fluctuates. This replaces the static PLANCK_ENTROPY
    with a time-varying floor that follows quantum field theory principles.

    Formula: PLANCK_ENTROPY_BASE * (1 + random.gauss(0, VACUUM_VARIANCE))
    Clamped to minimum PLANCK_ENTROPY_BASE * 0.5

    Returns:
        float: Fluctuating vacuum floor entropy
    """
    fluctuation = PLANCK_ENTROPY_BASE * (1.0 + random.gauss(0, VACUUM_VARIANCE))
    return max(fluctuation, PLANCK_ENTROPY_BASE * 0.5)


def attempt_spontaneous_emergence(state: SimState, H_observation: float) -> Optional[dict]:
    """
    Attempt observer-induced pattern genesis from vacuum.

    High observation cost can spark pattern emergence from superposition.
    This is the core of observer-induced genesis: the observer creates, not just measures.

    Args:
        state: Current SimState (mutated in place if emergence occurs)
        H_observation: Observation cost this cycle

    Returns:
        Receipt dict if emergence occurred, None otherwise
    """
    if H_observation <= GENESIS_THRESHOLD:
        return None

    if len(state.superposition_patterns) == 0:
        return None

    # Select pattern from superposition (weighted by fitness)
    weights = []
    for pattern in state.superposition_patterns:
        fitness = pattern.get("fitness_mean", pattern.get("fitness", 0.5))
        weights.append(fitness)

    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0] * len(state.superposition_patterns)
        total_weight = len(state.superposition_patterns)

    weights = [w / total_weight for w in weights]

    selected_pattern = random.choices(state.superposition_patterns, weights=weights, k=1)[0]

    # Remove from superposition, add to virtual
    state.superposition_patterns.remove(selected_pattern)
    selected_pattern["state"] = PatternState.VIRTUAL.value
    selected_pattern["virtual_lifespan"] = VIRTUAL_LIFESPAN
    state.virtual_patterns.append(selected_pattern)

    state.emergence_count_this_cycle += 1
    state.observer_wake_count += 1

    receipt = emit_receipt("spontaneous_emergence", {
        "tenant_id": "simulation",
        "triggering_observation_cost": H_observation,
        "emerged_pattern_id": selected_pattern["pattern_id"],
        "source_state": "SUPERPOSITION",
        "destination_state": "VIRTUAL",
        "cycle": state.cycle
    })
    state.receipt_ledger.append(receipt)

    return receipt


def process_virtual_patterns(state: SimState) -> List[str]:
    """
    Process VIRTUAL patterns - decay or survive based on re-observation.

    Virtual patterns are ephemeral. They need re-observation to survive,
    otherwise they collapse back to SUPERPOSITION.

    Args:
        state: Current SimState (mutated in place)

    Returns:
        List of collapsed pattern IDs
    """
    collapsed_ids = []
    to_remove = []

    for i, pattern in enumerate(state.virtual_patterns):
        pattern["virtual_lifespan"] = pattern.get("virtual_lifespan", VIRTUAL_LIFESPAN) - 1

        was_reobserved = False
        if autocatalysis_check(pattern):
            was_reobserved = True
            pattern["virtual_lifespan"] = VIRTUAL_LIFESPAN

        if pattern["virtual_lifespan"] <= 0 and not was_reobserved:
            pattern["state"] = PatternState.SUPERPOSITION.value
            pattern["virtual_lifespan"] = 0
            state.superposition_patterns.append(pattern)
            to_remove.append(i)
            collapsed_ids.append(pattern["pattern_id"])

            state.collapse_count_this_cycle += 1

            receipt = emit_receipt("virtual_collapse", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "lifespan_at_collapse": 0,
                "destination_state": "SUPERPOSITION",
                "was_reobserved": False,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)

    for i in reversed(to_remove):
        state.virtual_patterns.pop(i)

    return collapsed_ids


def compute_hawking_flux(state: SimState) -> tuple:
    """
    Compute Hawking entropy flux rate over rolling window.

    Args:
        state: Current SimState

    Returns:
        Tuple of (flux, trend)
    """
    state.flux_history.append(state.hawking_emissions_this_cycle)

    if len(state.flux_history) > FLUX_WINDOW * 2:
        state.flux_history = state.flux_history[-FLUX_WINDOW * 2:]

    if len(state.flux_history) < 2:
        return (0.0, "insufficient_data")

    if len(state.flux_history) >= FLUX_WINDOW:
        flux = (state.flux_history[-1] - state.flux_history[-FLUX_WINDOW]) / FLUX_WINDOW
    else:
        deltas = [state.flux_history[i] - state.flux_history[i-1]
                 for i in range(1, len(state.flux_history))]
        flux = sum(deltas) / len(deltas) if deltas else 0.0

    if flux > 0.01:
        trend = "increasing"
    elif flux < -0.01:
        trend = "decreasing"
    else:
        trend = "stable"

    return (flux, trend)


def compute_collapse_rate(state: SimState) -> float:
    """Compute collapse rate for this cycle."""
    return float(state.collapse_count_this_cycle)


def compute_emergence_rate(state: SimState) -> float:
    """Compute emergence rate for this cycle."""
    return float(state.emergence_count_this_cycle)


def compute_system_criticality(state: SimState, cycle: int) -> float:
    """
    Compute system criticality metric.

    Criticality = cumulative_emergences / cycle

    Args:
        state: Current SimState
        cycle: Current cycle number

    Returns:
        float: System criticality (0.0 to ~1.0)
    """
    if cycle == 0:
        return 0.0

    total_emergences = sum(1 for r in state.receipt_ledger
                          if r.get("receipt_type") == "spontaneous_emergence")
    return total_emergences / cycle


def check_criticality_alert(state: SimState, cycle: int, criticality: float) -> Optional[dict]:
    """Check if criticality alert should be emitted."""
    if (criticality > CRITICALITY_ALERT_THRESHOLD and
        not state.criticality_alert_emitted and
        (cycle - state.last_alert_cycle) > ALERT_COOLDOWN_CYCLES):

        state.criticality_alert_emitted = True
        state.last_alert_cycle = cycle

        receipt = emit_receipt("anomaly", {
            "tenant_id": "simulation",
            "cycle": cycle,
            "metric": "criticality",
            "baseline": CRITICALITY_ALERT_THRESHOLD,
            "delta": criticality - CRITICALITY_ALERT_THRESHOLD,
            "classification": "drift",
            "action": "alert"
        })
        state.receipt_ledger.append(receipt)
        return receipt

    if criticality < CRITICALITY_ALERT_THRESHOLD - 0.05:
        state.criticality_alert_emitted = False

    return None


def check_phase_transition(state: SimState, cycle: int, criticality: float, H_end: float) -> Optional[dict]:
    """Check if phase transition has occurred."""
    if criticality >= CRITICALITY_PHASE_TRANSITION and not state.phase_transition_occurred:
        state.phase_transition_occurred = True

        receipt = emit_receipt("phase_transition", {
            "tenant_id": "simulation",
            "cycle": cycle,
            "criticality": criticality,
            "total_emergences": state.observer_wake_count,
            "transition_type": "quantum_leap",
            "entropy_at_transition": H_end
        })
        state.receipt_ledger.append(receipt)
        return receipt

    return None


def estimate_cycles_to_transition(criticality: float, criticality_rate: float) -> int:
    """Estimate cycles until criticality reaches 1.0."""
    if criticality_rate <= 0:
        return -1

    remaining = CRITICALITY_PHASE_TRANSITION - criticality
    cycles = int(remaining / criticality_rate)
    return max(cycles, 0)


def emit_hawking_entropy(state: SimState, pattern: dict) -> float:
    """
    Emit Hawking radiation when pattern crosses boundary.

    Args:
        state: Current SimState (mutated in place)
        pattern: Pattern crossing boundary

    Returns:
        float: Emitted entropy amount
    """
    boundary_entropy = measure_boundary_crossing(pattern)
    emitted = boundary_entropy * HAWKING_COEFFICIENT
    state.hawking_emissions_this_cycle += emitted
    return emitted
