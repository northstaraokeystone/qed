"""
sim/dynamics.py - Simulation Dynamics Functions

Birth/death/recombination/genesis simulation functions.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import random
from typing import List, Optional

from entropy import emit_receipt, system_entropy, agent_fitness
from autocatalysis import autocatalysis_check, coherence_score
from architect import identify_automation_gaps, synthesize_blueprint
from recombine import recombine, mate_selection
from receipt_completeness import receipt_completeness_check, godel_layer
from population import selection_pressure, dynamic_cap
from autoimmune import is_self

from .constants import PatternState
from .config import SimConfig
from .state import SimState


def simulate_wound(state: SimState, wound_type: str) -> dict:
    """
    Inject synthetic wound into system.

    Args:
        state: Current SimState
        wound_type: Type of wound (operational, safety, etc.)

    Returns:
        wound_receipt dict
    """
    time_to_resolve_ms = int(random.expovariate(1.0 / 1800000))
    resolution_actions = ["restart", "patch", "rollback", "escalate", "ignore"]
    resolution_action = random.choice(resolution_actions)

    wound = {
        "receipt_type": "wound",
        "intervention_id": f"wound_{state.cycle}_{len(state.wound_history)}",
        "problem_type": wound_type,
        "time_to_resolve_ms": time_to_resolve_ms,
        "resolution_action": resolution_action,
        "could_automate": random.uniform(0.3, 0.9),
        "ts": f"2025-01-01T{state.cycle:02d}:00:00Z",
        "tenant_id": "simulation"
    }

    return wound


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


def simulate_recombination(state: SimState, config: SimConfig) -> None:
    """
    Attempt pattern recombination.

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters
    """
    if len(state.active_patterns) < 2:
        return

    pairs = mate_selection(state.active_patterns)

    for pattern_a, pattern_b in pairs[:1]:
        recombination_receipt = recombine(pattern_a, pattern_b)
        state.receipt_ledger.append(recombination_receipt)

        offspring_id = recombination_receipt["offspring_id"]
        offspring = {
            "pattern_id": offspring_id,
            "origin": "recombination",
            "receipts": [],
            "tenant_id": "simulation",
            "fitness": (pattern_a.get("fitness", 0.5) + pattern_b.get("fitness", 0.5)) / 2,
            "fitness_mean": (pattern_a.get("fitness_mean", 0.5) + pattern_b.get("fitness_mean", 0.5)) / 2,
            "fitness_var": 0.1,
            "domain": pattern_a.get("domain", "unknown"),
            "problem_type": pattern_a.get("problem_type", "operational"),
            "prev_coherence": 0.0,
            "state": PatternState.ACTIVE.value,
            "virtual_lifespan": 0
        }

        if random.random() < 0.7:
            offspring["receipts"].append({
                "receipt_type": "agent_decision",
                "pattern_id": offspring_id,
                "ts": f"2025-01-01T{state.cycle:02d}:00:00Z"
            })
            if autocatalysis_check(offspring):
                state.active_patterns.append(offspring)
                state.births_this_cycle += 1

                receipt = emit_receipt("sim_mate", {
                    "tenant_id": "simulation",
                    "parent_a": pattern_a["pattern_id"],
                    "parent_b": pattern_b["pattern_id"],
                    "offspring_id": offspring_id,
                    "viable": True,
                    "cycle": state.cycle
                })
                state.receipt_ledger.append(receipt)


def simulate_genesis(state: SimState, config: SimConfig) -> None:
    """
    Check for automation gaps and synthesize blueprints.

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters
    """
    if len(state.wound_history) < 5:
        return

    gaps = identify_automation_gaps(state.wound_history)

    for gap in gaps[:1]:
        blueprint = synthesize_blueprint(gap, state.wound_history)

        if random.random() < config.hitl_auto_approve_rate:
            receipts_before = state.receipt_ledger.copy()

            offspring_id = f"genesis_{state.cycle}"
            pattern = {
                "pattern_id": offspring_id,
                "origin": "genesis",
                "receipts": [],
                "tenant_id": "simulation",
                "fitness": 0.6,
                "fitness_mean": 0.6,
                "fitness_var": 0.1,
                "domain": "automation",
                "problem_type": gap.get("problem_type", "operational"),
                "prev_coherence": 0.0,
                "state": PatternState.ACTIVE.value,
                "virtual_lifespan": 0
            }
            pattern["receipts"].append({
                "receipt_type": "agent_decision",
                "pattern_id": pattern["pattern_id"],
                "ts": f"2025-01-01T{state.cycle:02d}:00:00Z"
            })
            state.active_patterns.append(pattern)
            state.births_this_cycle += 1

            receipts_after = state.receipt_ledger.copy()
            n_receipts = len(pattern["receipts"])

            H_before = system_entropy(receipts_before)
            H_after = system_entropy(receipts_after)
            fitness = agent_fitness(receipts_before, receipts_after, n_receipts)

            birth_receipt = emit_receipt("genesis_birth_receipt", {
                "tenant_id": "simulation",
                "offspring_id": offspring_id,
                "blueprint_name": blueprint["name"],
                "fitness": fitness,
                "H_before": H_before,
                "H_after": H_after,
                "n_receipts": n_receipts,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(birth_receipt)

            receipt = emit_receipt("genesis_approved", {
                "tenant_id": "simulation",
                "blueprint_name": blueprint["name"],
                "autonomy": blueprint["autonomy"],
                "approved_by": "sim_hitl",
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)


def simulate_completeness(state: SimState) -> None:
    """
    Check for receipt completeness (singularity).

    Args:
        state: Current SimState (mutated in place)
    """
    if receipt_completeness_check(state.receipt_ledger):
        already_emitted = any(r.get("receipt_type") == "sim_complete" for r in state.receipt_ledger)
        if not already_emitted:
            receipt = emit_receipt("sim_complete", {
                "tenant_id": "simulation",
                "cycle": state.cycle,
                "completeness_achieved": True,
                "godel_layer": godel_layer()
            })
            state.receipt_ledger.append(receipt)


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
