"""
sim/vacuum_fluctuation.py - Vacuum Fluctuation and Emergence

Functions for vacuum fluctuation, spontaneous emergence, and virtual patterns.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import random
from typing import List, Optional

from entropy import emit_receipt
from autocatalysis import autocatalysis_check

from .constants import (
    PLANCK_ENTROPY_BASE, VACUUM_VARIANCE, GENESIS_THRESHOLD,
    VIRTUAL_LIFESPAN, PatternState
)
from .types_state import SimState


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
