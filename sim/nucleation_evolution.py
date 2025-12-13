"""
sim/nucleation_evolution.py - Replication and Differentiation

Crystal replication, hybrid differentiation, and seed evolution.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import math
import random
from typing import Optional

from entropy import emit_receipt

from .constants import (
    BASE_ATTRACTION_STRENGTH, AUTOCATALYSIS_AMPLIFICATION,
    REPLICATION_THRESHOLD, EVOLUTION_RATE, EVOLUTION_WINDOW_SEEDS,
    EFFECT_RESONANCE_TRIGGER, EFFECT_ENTROPY_INCREASE, EFFECT_SYMMETRY_BREAK,
    RESONANCE_DIFFERENTIATION_THRESHOLD, ENTROPY_DIFFERENTIATION_THRESHOLD,
    SYMMETRY_DIFFERENTIATION_THRESHOLD, DIFFERENTIATION_BIAS, ARCHETYPE_TRIGGER_SIZE,
    TUNNELING_CONSTANT, TUNNELING_FLOOR, ENTANGLEMENT_FACTOR, EXPECTED_ARCHITECTS
)
from .types_state import SimState, Seed, Beacon, Counselor, Crystal


def calculate_entanglement_boost(state: SimState) -> float:
    """Calculate entanglement boost for ARCHITECT formation."""
    architect_deficit = max(0, EXPECTED_ARCHITECTS - state.architect_formations)
    boost = architect_deficit * ENTANGLEMENT_FACTOR

    if boost > 0:
        state.entanglement_boosts += 1

    return boost


def check_architect_tunneling(crystal: Crystal, symmetry_ratio: float, state: SimState) -> bool:
    """Check if ARCHITECT can form via quantum tunneling."""
    if symmetry_ratio < TUNNELING_FLOOR:
        return False

    if symmetry_ratio >= SYMMETRY_DIFFERENTIATION_THRESHOLD:
        return False

    barrier = SYMMETRY_DIFFERENTIATION_THRESHOLD - symmetry_ratio
    tunneling_probability = math.exp(-barrier / TUNNELING_CONSTANT)

    if random.random() < tunneling_probability:
        state.tunneling_events += 1
        return True

    return False


def check_hybrid_differentiation(state: SimState, cycle: int) -> Optional[dict]:
    """Check if any HYBRID crystal should differentiate into a specific archetype."""
    entanglement_boost = calculate_entanglement_boost(state)

    for crystal in state.crystals:
        if not crystal.crystallized:
            continue
        if crystal.agent_type != "HYBRID":
            continue
        if len(crystal.members) < ARCHETYPE_TRIGGER_SIZE:
            continue

        total_effects = sum(crystal.effect_distribution.values())
        if total_effects == 0:
            continue

        resonance_count = crystal.effect_distribution.get(EFFECT_RESONANCE_TRIGGER, 0)
        entropy_count = crystal.effect_distribution.get(EFFECT_ENTROPY_INCREASE, 0)
        symmetry_count = crystal.effect_distribution.get(EFFECT_SYMMETRY_BREAK, 0)

        resonance_ratio = resonance_count / total_effects
        entropy_ratio = entropy_count / total_effects
        symmetry_ratio = symmetry_count / total_effects

        adjusted_resonance_threshold = RESONANCE_DIFFERENTIATION_THRESHOLD - DIFFERENTIATION_BIAS
        adjusted_entropy_threshold = ENTROPY_DIFFERENTIATION_THRESHOLD - DIFFERENTIATION_BIAS
        adjusted_symmetry_threshold = SYMMETRY_DIFFERENTIATION_THRESHOLD - DIFFERENTIATION_BIAS - entanglement_boost

        old_type = crystal.agent_type
        new_type = None
        trigger = None
        threshold_value = 0.0
        tunneled = False

        if resonance_ratio > adjusted_resonance_threshold:
            new_type = "SHEPHERD"
            trigger = "resonance"
            threshold_value = resonance_ratio
            crystal.agent_type = new_type
            state.governance_nodes += 1
        elif symmetry_ratio > adjusted_symmetry_threshold:
            new_type = "ARCHITECT"
            trigger = "symmetry"
            threshold_value = symmetry_ratio
            crystal.agent_type = new_type
            state.architect_formations += 1
        elif check_architect_tunneling(crystal, symmetry_ratio, state):
            new_type = "ARCHITECT"
            trigger = "tunneling"
            threshold_value = symmetry_ratio
            tunneled = True
            crystal.agent_type = new_type
            state.architect_formations += 1
        elif entropy_ratio > adjusted_entropy_threshold:
            new_type = "HUNTER"
            trigger = "entropy"
            threshold_value = entropy_ratio
            crystal.agent_type = new_type
            state.hunter_formations += 1
            state.hunter_delays += 1
        elif len(crystal.members) > ARCHETYPE_TRIGGER_SIZE * 2:
            new_type = "ARCHITECT"
            trigger = "size"
            threshold_value = float(len(crystal.members))
            crystal.agent_type = new_type
            state.architect_formations += 1

        if new_type:
            state.hybrid_differentiation_count += 1
            state.hybrid_formations -= 1

            return emit_receipt("archetype_shift", {
                "tenant_id": "simulation",
                "receipt_type": "archetype_shift",
                "crystal_id": crystal.crystal_id,
                "cycle": cycle,
                "from_type": old_type,
                "to_type": new_type,
                "trigger": trigger,
                "threshold_value": threshold_value,
                "resonance_ratio": resonance_ratio,
                "entropy_ratio": entropy_ratio,
                "symmetry_ratio": symmetry_ratio,
                "entanglement_boost": entanglement_boost,
                "tunneled": tunneled,
                "crystal_size": len(crystal.members),
                "effect_distribution": dict(crystal.effect_distribution)
            })

    return None


def evolve_seeds(state: SimState, cycle: int) -> None:
    """Evolve seeds toward successful captures."""
    if cycle % EVOLUTION_WINDOW_SEEDS != 0 or cycle == 0:
        return

    for seed in state.seeds:
        if seed.captures > 0:
            phase_delta = random.gauss(0, 0.1) * EVOLUTION_RATE
            seed.phase += phase_delta
            seed.phase = seed.phase % (2 * math.pi)

            crystal = state.crystals[seed.seed_id]
            if crystal.crystallized:
                seed.resonance_affinity = min(1.0, seed.resonance_affinity + EVOLUTION_RATE * 0.5)
            else:
                seed.resonance_affinity = max(0.0, seed.resonance_affinity - EVOLUTION_RATE * 0.5)


def check_replication(state: SimState, cycle: int) -> Optional[dict]:
    """Check if any crystallized crystal can replicate."""
    for crystal in state.crystals:
        if not crystal.crystallized:
            continue
        if len(crystal.members) < REPLICATION_THRESHOLD:
            continue

        child_exists = any(c.parent_crystal_id == crystal.crystal_id for c in state.crystals)
        if child_exists:
            continue

        child_id = len(state.crystals)
        child_seed_id = len(state.seeds)

        parent_seed = state.seeds[crystal.seed_id]
        child_seed = Seed(
            seed_id=child_seed_id,
            phase=(parent_seed.phase + random.gauss(0, 0.5)) % (2 * math.pi),
            resonance_affinity=max(0.0, min(1.0, parent_seed.resonance_affinity + random.gauss(0, 0.1))),
            direction=random.choice([1, -1]),
            captures=0
        )
        state.seeds.append(child_seed)

        child_beacon = Beacon(
            seed_id=child_seed_id,
            strength=BASE_ATTRACTION_STRENGTH * (1.0 + AUTOCATALYSIS_AMPLIFICATION)
        )
        state.beacons.append(child_beacon)

        child_counselor = Counselor(
            counselor_id=len(state.counselors),
            seed_id=child_seed_id
        )
        state.counselors.append(child_counselor)

        child_generation = crystal.generation + 1

        child_crystal = Crystal(
            crystal_id=child_id,
            seed_id=child_seed_id,
            members=[],
            coherence=0.0,
            crystallized=False,
            birth_cycle=-1,
            agent_type="",
            effect_distribution={
                "ENTROPY_INCREASE": 0,
                "RESONANCE_TRIGGER": 0,
                "SYMMETRY_BREAK": 0
            },
            parent_crystal_id=crystal.crystal_id,
            generation=child_generation
        )
        state.crystals.append(child_crystal)

        state.max_generation = max(state.max_generation, child_generation)
        state.replication_events += 1
        state.total_branches += 1

        emit_receipt("generation", {
            "tenant_id": "simulation",
            "receipt_type": "generation",
            "cycle": cycle,
            "crystal_id": child_id,
            "generation": child_generation,
            "parent_id": crystal.crystal_id,
            "parent_generation": crystal.generation,
            "lineage_depth": child_generation
        })

        return emit_receipt("replication", {
            "tenant_id": "simulation",
            "receipt_type": "replication",
            "cycle": cycle,
            "parent_crystal_id": crystal.crystal_id,
            "parent_archetype": crystal.agent_type,
            "child_crystal_id": child_id,
            "child_archetype": "",
            "parent_captures": len(crystal.members),
            "note": "Child discovers own archetype via self-measurement"
        })

    return None
