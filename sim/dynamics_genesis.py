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

from .constants import PatternState, DOMAIN_AFFINITY_MATRIX, AFFINITY_STOCHASTIC_VARIANCE
from .types_config import SimConfig
from .types_state import SimState, FitnessDistribution
from .variance import (
    inherit_variance,
    distribution_from_pattern,
    apply_distribution_to_pattern,
)


def get_domain_affinity(domain_a: str, domain_b: str, stochastic: bool = False) -> float:
    """
    Get affinity between two domains from the DOMAIN_AFFINITY_MATRIX.

    Symmetric lookup: (a,b) == (b,a)
    Same domain: returns 1.0 (implicit)

    Args:
        domain_a: First domain name (lowercase)
        domain_b: Second domain name (lowercase)
        stochastic: If True, add random variance (default: False for deterministic)

    Returns:
        float: Affinity value 0.0-1.0 (clamped after optional variance)
    """
    # Same domain = full affinity
    if domain_a == domain_b:
        return 1.0

    # Normalize to lowercase
    a, b = domain_a.lower(), domain_b.lower()

    # Try both orderings for symmetric lookup
    base_affinity = 0.1  # Default for unknown pairs
    if (a, b) in DOMAIN_AFFINITY_MATRIX:
        base_affinity = DOMAIN_AFFINITY_MATRIX[(a, b)]
    elif (b, a) in DOMAIN_AFFINITY_MATRIX:
        base_affinity = DOMAIN_AFFINITY_MATRIX[(b, a)]

    # Apply stochastic variance if enabled
    if stochastic and AFFINITY_STOCHASTIC_VARIANCE > 0:
        noise = random.gauss(0, AFFINITY_STOCHASTIC_VARIANCE)
        affinity = base_affinity + noise
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, affinity))

    return base_affinity


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
    Attempt pattern recombination with fitness variance inheritance.

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters (includes variance_inheritance strategy)
    """
    if len(state.active_patterns) < 2:
        return

    pairs = mate_selection(state.active_patterns)

    for pattern_a, pattern_b in pairs[:1]:
        recombination_receipt = recombine(pattern_a, pattern_b)
        state.receipt_ledger.append(recombination_receipt)

        offspring_id = recombination_receipt["offspring_id"]

        # Extract parent fitness distributions
        # Handle SUPERPOSITION edge case: if one parent is in different state, use live parent only
        parent_a_state = pattern_a.get("state", PatternState.ACTIVE.value)
        parent_b_state = pattern_b.get("state", PatternState.ACTIVE.value)

        parent_distributions = []
        if parent_a_state == PatternState.ACTIVE.value:
            parent_distributions.append(distribution_from_pattern(pattern_a))
        if parent_b_state == PatternState.ACTIVE.value:
            parent_distributions.append(distribution_from_pattern(pattern_b))

        # If both parents are not active (edge case), use both anyway
        if not parent_distributions:
            parent_distributions = [
                distribution_from_pattern(pattern_a),
                distribution_from_pattern(pattern_b)
            ]

        # Apply variance inheritance strategy
        offspring_dist, variance_receipt = inherit_variance(
            parent_distributions=parent_distributions,
            config=config,
            offspring_id=offspring_id,
            cycle=state.cycle
        )
        state.receipt_ledger.append(variance_receipt)

        offspring = {
            "pattern_id": offspring_id,
            "origin": "recombination",
            "receipts": [],
            "tenant_id": "simulation",
            "domain": pattern_a.get("domain", "unknown"),
            "problem_type": pattern_a.get("problem_type", "operational"),
            "prev_coherence": 0.0,
            "state": PatternState.ACTIVE.value,
            "virtual_lifespan": 0
        }

        # Apply inherited fitness distribution to offspring
        apply_distribution_to_pattern(offspring, offspring_dist)

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
                    "cycle": state.cycle,
                    "variance_strategy": config.variance_inheritance,
                    "offspring_lineage_depth": offspring_dist.lineage_depth
                })
                state.receipt_ledger.append(receipt)


def simulate_genesis(state: SimState, config: SimConfig) -> None:
    """
    Check for automation gaps and synthesize blueprints with fitness variance inheritance.

    Genesis patterns are created by ARCHITECT from wound patterns - they have
    no genetic parents, so they use an uninformed prior distribution.

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters (includes variance_inheritance strategy)
    """
    if len(state.wound_history) < 5:
        return

    gaps = identify_automation_gaps(state.wound_history)

    for gap in gaps[:1]:
        blueprint = synthesize_blueprint(gap, state.wound_history)

        if random.random() < config.hitl_auto_approve_rate:
            receipts_before = state.receipt_ledger.copy()

            offspring_id = f"genesis_{state.cycle}"

            # Genesis patterns have no parents - use uninformed prior
            # Apply variance inheritance to determine initial fitness distribution
            genesis_prior = FitnessDistribution(
                mean=0.6,  # Genesis patterns start with modest positive fitness
                variance=0.1,
                n_samples=1,
                lineage_depth=0
            )

            offspring_dist, variance_receipt = inherit_variance(
                parent_distributions=[genesis_prior],
                config=config,
                offspring_id=offspring_id,
                cycle=state.cycle
            )
            state.receipt_ledger.append(variance_receipt)

            pattern = {
                "pattern_id": offspring_id,
                "origin": "genesis",
                "receipts": [],
                "tenant_id": "simulation",
                "domain": "automation",
                "problem_type": gap.get("problem_type", "operational"),
                "prev_coherence": 0.0,
                "state": PatternState.ACTIVE.value,
                "virtual_lifespan": 0
            }

            # Apply inherited fitness distribution to pattern
            apply_distribution_to_pattern(pattern, offspring_dist)

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
                "cycle": state.cycle,
                "variance_strategy": config.variance_inheritance,
                "offspring_lineage_depth": offspring_dist.lineage_depth
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
