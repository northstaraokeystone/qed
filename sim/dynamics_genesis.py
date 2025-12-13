"""
sim/dynamics_genesis.py - Wound/Genesis/Completeness Dynamics

Wound injection, recombination, genesis, and completeness checking.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import random

from entropy import emit_receipt, system_entropy, agent_fitness
from autocatalysis import autocatalysis_check
from architect import identify_automation_gaps, synthesize_blueprint
from recombine import recombine, mate_selection
from receipt_completeness import receipt_completeness_check, godel_layer

from .constants import PatternState
from .types_config import SimConfig
from .types_state import SimState


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
