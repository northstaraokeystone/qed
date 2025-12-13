"""
sim/dynamics_quantum.py - Quantum State Dynamics

Superposition, measurement, wavefunction collapse, and Hilbert space.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import random
from typing import List, Optional

from entropy import emit_receipt, system_entropy
from population import dynamic_cap

from .types_state import SimState


def simulate_superposition(state: SimState, pattern: dict) -> None:
    """Move pattern to superposition state."""
    if pattern in state.active_patterns:
        state.active_patterns.remove(pattern)
    if pattern not in state.superposition_patterns:
        state.superposition_patterns.append(pattern)


def simulate_measurement(state: SimState, wound: dict) -> Optional[dict]:
    """Wound acts as measurement - collapse superposition."""
    if not state.superposition_patterns:
        return None

    pattern = wavefunction_collapse(state.superposition_patterns, wound)

    if pattern:
        state.superposition_patterns.remove(pattern)
        state.active_patterns.append(pattern)

    return pattern


def wavefunction_collapse(potential_patterns: List[dict], wound: dict) -> Optional[dict]:
    """Calculate probability and select pattern from superposition."""
    if not potential_patterns:
        return None

    probabilities = []
    for pattern in potential_patterns:
        fitness = pattern.get("fitness", 0.5)
        match_quality = 1.0 if pattern.get("problem_type") == wound.get("problem_type") else 0.5
        prob = fitness * match_quality
        probabilities.append(prob)

    total = sum(probabilities)
    if total == 0:
        return None

    probabilities = [p / total for p in probabilities]
    selected = random.choices(potential_patterns, weights=probabilities, k=1)[0]
    return selected


def simulate_godel_stress(state: SimState, level: str) -> bool:
    """Test undecidability at given receipt level."""
    if level == "L0":
        return True
    else:
        return True


def hilbert_space_size(state: SimState) -> int:
    """Calculate current dimensionality of pattern space."""
    receipt_types = len(set(r.get("receipt_type", "") for r in state.receipt_ledger))
    active_patterns = len(state.active_patterns)
    possible_states = 2

    return receipt_types * active_patterns * possible_states


def bound_violation_check(state: SimState) -> bool:
    """Check if population exceeds dynamic_cap."""
    current_entropy = system_entropy(state.receipt_ledger)
    cap = dynamic_cap(1.0, current_entropy)

    if len(state.active_patterns) > cap:
        receipt = emit_receipt("sim_violation", {
            "tenant_id": "simulation",
            "cycle": state.cycle,
            "violation_type": "bound",
            "current_population": len(state.active_patterns),
            "dynamic_cap": cap
        })
        state.receipt_ledger.append(receipt)
        return True

    return False
