"""
sim/dynamics_lifecycle.py - Birth/Death/Selection Dynamics

Autocatalysis detection, selection pressure, and pattern lifecycle.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

from entropy import emit_receipt
from autocatalysis import coherence_score
from population import selection_pressure
from autoimmune import is_self

from .constants import PatternState
from .types_state import SimState


def simulate_autocatalysis(state: SimState) -> None:
    """
    Detect pattern births and deaths via autocatalysis.

    Args:
        state: Current SimState (mutated in place)
    """
    to_remove = []

    for i, pattern in enumerate(state.active_patterns):
        prev_coherence = pattern.get("prev_coherence", coherence_score(pattern))
        current_coherence = coherence_score(pattern)
        pattern["prev_coherence"] = current_coherence

        if prev_coherence < 0.3 and current_coherence >= 0.3:
            receipt = emit_receipt("sim_birth", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "coherence_at_birth": current_coherence,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)
            state.births_this_cycle += 1

        if prev_coherence >= 0.3 and current_coherence < 0.3:
            receipt = emit_receipt("sim_death", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "coherence_at_death": current_coherence,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)
            state.deaths_this_cycle += 1

            if not is_self(pattern):
                to_remove.append(i)
                state.superposition_patterns.append(pattern)
                state.superposition_transitions_this_cycle += 1

    for i in reversed(to_remove):
        state.active_patterns.pop(i)


def simulate_selection(state: SimState) -> None:
    """
    Apply selection pressure via population.py.

    Args:
        state: Current SimState (mutated in place)
    """
    if not state.active_patterns and not state.virtual_patterns:
        return

    if state.active_patterns:
        survivors, superposition = selection_pressure(state.active_patterns, "simulation")
        state.superposition_transitions_this_cycle += len(superposition)
        state.active_patterns = survivors
        state.superposition_patterns.extend(superposition)

    virtual_promoted = []
    virtual_collapsed = []

    if state.virtual_patterns:
        virtual_survivors, virtual_failures = selection_pressure(state.virtual_patterns, "simulation")

        for pattern in virtual_survivors:
            pattern["state"] = PatternState.ACTIVE.value
            pattern["virtual_lifespan"] = 0
            virtual_promoted.append(pattern)

            receipt = emit_receipt("virtual_promotion", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "promoted_to": "ACTIVE",
                "survival_reason": "selection_passed",
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)

        for pattern in virtual_failures:
            pattern["state"] = PatternState.SUPERPOSITION.value
            pattern["virtual_lifespan"] = 0
            virtual_collapsed.append(pattern)

            state.collapse_count_this_cycle += 1

            receipt = emit_receipt("virtual_collapse", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "lifespan_at_collapse": pattern.get("virtual_lifespan", 0),
                "destination_state": "SUPERPOSITION",
                "was_reobserved": False,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)

        state.active_patterns.extend(virtual_promoted)
        state.superposition_patterns.extend(virtual_collapsed)
        state.virtual_patterns = []

    receipt = emit_receipt("selection_event", {
        "tenant_id": "simulation",
        "cycle": state.cycle,
        "survivors": len(state.active_patterns),
        "to_superposition": state.superposition_transitions_this_cycle,
        "virtual_promoted": len(virtual_promoted),
        "virtual_collapsed": len(virtual_collapsed)
    })
    state.receipt_ledger.append(receipt)
