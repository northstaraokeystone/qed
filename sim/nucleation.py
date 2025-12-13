"""
sim/nucleation.py - Quantum Nucleation Functions

Active seeding, crystallization, and archetype discovery.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import math
import random
from typing import Optional

from entropy import emit_receipt

from .constants import (
    N_SEEDS, SEED_PHASES, SEED_RESONANCE_AFFINITY, SEED_DIRECTION,
    BASE_ATTRACTION_STRENGTH, CAPTURE_THRESHOLD, TRANSFORM_STRENGTH,
    TUNNELING_THRESHOLD, GROWTH_FACTOR, MAX_GROWTH_BOOST,
    AUTOCATALYSIS_STREAK, SELF_PREDICTION_THRESHOLD, CRYSTALLIZED_BEACON_BOOST,
    AUTOCATALYSIS_AMPLIFICATION, REPLICATION_THRESHOLD, ARCHETYPE_DOMINANCE_THRESHOLD,
    GOVERNANCE_BIAS, ENTROPY_AMPLIFIER, SYMMETRY_BIAS, ARCHITECT_SIZE_TRIGGER,
    HUNTER_SIZE_TRIGGER, EVOLUTION_RATE, EVOLUTION_WINDOW_SEEDS,
    EFFECT_ENTROPY_INCREASE, EFFECT_RESONANCE_TRIGGER, EFFECT_SYMMETRY_BREAK,
    UNIFORM_KICK_DISTRIBUTION, EFFECT_TYPES,
    RESONANCE_DIFFERENTIATION_THRESHOLD, ENTROPY_DIFFERENTIATION_THRESHOLD,
    SYMMETRY_DIFFERENTIATION_THRESHOLD, DIFFERENTIATION_BIAS, ARCHETYPE_TRIGGER_SIZE,
    TUNNELING_CONSTANT, TUNNELING_FLOOR, ENTANGLEMENT_FACTOR, EXPECTED_ARCHITECTS
)
from .state import SimState, Seed, Beacon, Counselor, Crystal


def initialize_nucleation(state: SimState) -> None:
    """Initialize quantum nucleation system with seeds, beacons, counselors, and crystals."""
    for i in range(N_SEEDS):
        seed = Seed(
            seed_id=i,
            phase=SEED_PHASES[i],
            resonance_affinity=SEED_RESONANCE_AFFINITY[i],
            direction=SEED_DIRECTION[i],
            captures=0
        )
        state.seeds.append(seed)

        beacon = Beacon(seed_id=i, strength=BASE_ATTRACTION_STRENGTH)
        state.beacons.append(beacon)

        counselor = Counselor(counselor_id=i, seed_id=i)
        state.counselors.append(counselor)

        crystal = Crystal(crystal_id=i, seed_id=i, members=[], coherence=0.0)
        state.crystals.append(crystal)


def growth_boost(crystal: Crystal) -> float:
    """Calculate compound boost based on crystal size."""
    size = len(crystal.members)
    boost = 1.0 + GROWTH_FACTOR * (size / 10.0)
    return min(boost, MAX_GROWTH_BOOST)


def counselor_score(counselor: Counselor, seed: Seed, kick_phase: float,
                    kick_resonant: bool, kick_direction: int) -> float:
    """Calculate similarity score between counselor's seed and a kick."""
    phase_diff = abs(kick_phase - seed.phase)
    phase_score = (math.cos(phase_diff) + 1.0) / 2.0

    if kick_resonant:
        resonance_score = seed.resonance_affinity
    else:
        resonance_score = 1.0 - seed.resonance_affinity

    if kick_direction == seed.direction:
        direction_score = 1.0
    else:
        direction_score = 0.5

    return phase_score * resonance_score * direction_score


def counselor_compete(state: SimState, kick_receipt: dict, kick_phase: float,
                     kick_resonant: bool, kick_direction: int) -> Optional[tuple]:
    """Counselors compete to capture a kick. Best match wins."""
    best_seed_id = None
    best_similarity = 0.0

    for counselor in state.counselors:
        seed = state.seeds[counselor.seed_id]
        crystal = state.crystals[counselor.seed_id]

        similarity = counselor_score(counselor, seed, kick_phase, kick_resonant, kick_direction)

        if kick_resonant:
            similarity += GOVERNANCE_BIAS
        else:
            interference_type = kick_receipt.get("interference_type", "neutral")
            if interference_type in ("constructive", "destructive"):
                similarity += SYMMETRY_BIAS
            else:
                similarity += ENTROPY_AMPLIFIER

        if crystal.crystallized:
            similarity *= (1.0 + AUTOCATALYSIS_AMPLIFICATION)

        similarity *= growth_boost(crystal)
        similarity = min(similarity, 1.0)

        if similarity > best_similarity:
            best_similarity = similarity
            best_seed_id = counselor.seed_id

    if best_similarity >= CAPTURE_THRESHOLD:
        return (best_seed_id, best_similarity)
    return None


def classify_effect_type(kick_receipt: dict) -> str:
    """Classify effect type of a captured kick for archetype discovery."""
    if UNIFORM_KICK_DISTRIBUTION:
        return random.choice(EFFECT_TYPES)

    if kick_receipt.get("resonance_hit", False):
        return EFFECT_RESONANCE_TRIGGER

    interference_type = kick_receipt.get("interference_type", "neutral")
    if interference_type in ("constructive", "destructive"):
        return EFFECT_SYMMETRY_BREAK

    return EFFECT_ENTROPY_INCREASE


def counselor_capture(state: SimState, seed_id: int, kick_receipt: dict,
                     similarity: float, cycle: int) -> dict:
    """Capture a kick and transform it into a crystal member."""
    seed = state.seeds[seed_id]
    crystal = state.crystals[seed_id]

    tunneled = similarity >= TUNNELING_THRESHOLD
    transformed = random.random() < TRANSFORM_STRENGTH
    effect_type = classify_effect_type(kick_receipt)

    crystal.effect_distribution[effect_type] = crystal.effect_distribution.get(effect_type, 0) + 1
    state.kick_distribution[effect_type] = state.kick_distribution.get(effect_type, 0) + 1

    member = {**kick_receipt, "capture_similarity": similarity, "effect_type": effect_type}
    crystal.members.append(member)

    seed.captures += 1
    state.total_captures += 1

    current_size = len(crystal.members)
    if current_size >= 50 and not crystal.size_50_reached:
        crystal.size_50_reached = True
        state.size_50_count += 1
        emit_receipt("size_threshold", {
            "tenant_id": "simulation",
            "receipt_type": "size_threshold",
            "cycle": cycle,
            "crystal_id": crystal.crystal_id,
            "size": current_size,
            "first_time": True,
            "growth_boost": growth_boost(crystal),
            "generation": crystal.generation
        })

    if len(crystal.members) > 1:
        phases = [m.get("phase", 0.0) for m in crystal.members]
        mean_phase = sum(phases) / len(phases)
        variance = sum((p - mean_phase) ** 2 for p in phases) / len(phases)
        crystal.coherence = max(0.0, 1.0 - variance / (math.pi ** 2))
    else:
        crystal.coherence = 1.0

    return emit_receipt("capture", {
        "tenant_id": "simulation",
        "cycle": cycle,
        "seed_id": seed_id,
        "similarity": similarity,
        "tunneled": tunneled,
        "transformed": transformed,
        "crystal_size": len(crystal.members),
        "coherence": crystal.coherence,
        "effect_type": effect_type,
        "effect_distribution": dict(crystal.effect_distribution)
    })


def discover_archetype(crystal: Crystal) -> tuple:
    """Discover crystal's archetype through self-measurement of effect distribution."""
    dist = dict(crystal.effect_distribution)
    crystal_size = len(crystal.members)

    if crystal_size > ARCHITECT_SIZE_TRIGGER:
        architect_bias = crystal_size * 0.01
        dist[EFFECT_SYMMETRY_BREAK] = dist.get(EFFECT_SYMMETRY_BREAK, 0) + architect_bias

    if crystal_size < HUNTER_SIZE_TRIGGER:
        hunter_bias = (HUNTER_SIZE_TRIGGER - crystal_size) * 0.01
        dist[EFFECT_ENTROPY_INCREASE] = dist.get(EFFECT_ENTROPY_INCREASE, 0) + hunter_bias

    total = sum(dist.values())
    if total == 0:
        return ("HYBRID", 0.0, True)

    dominant_effect = max(dist.keys(), key=lambda k: dist[k])
    dominant_count = dist[dominant_effect]
    dominance_ratio = dominant_count / total

    if dominance_ratio >= ARCHETYPE_DOMINANCE_THRESHOLD:
        archetype_map = {
            EFFECT_ENTROPY_INCREASE: "HUNTER",
            EFFECT_RESONANCE_TRIGGER: "SHEPHERD",
            EFFECT_SYMMETRY_BREAK: "ARCHITECT"
        }
        archetype = archetype_map.get(dominant_effect, "HYBRID")
        return (archetype, dominance_ratio, False)
    else:
        return ("HYBRID", dominance_ratio, True)


def calculate_entanglement_boost(state: SimState) -> float:
    """Calculate entanglement boost for ARCHITECT formation."""
    architect_deficit = max(0, EXPECTED_ARCHITECTS - state.architect_formations)
    boost = architect_deficit * ENTANGLEMENT_FACTOR

    if boost > 0:
        state.entanglement_boosts += 1

    return boost


def check_architect_tunneling(crystal, symmetry_ratio: float, state: SimState) -> bool:
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


def check_crystallization(state: SimState, cycle: int) -> Optional[dict]:
    """Check if any crystal has achieved autocatalysis (birth)."""
    for crystal in state.crystals:
        if crystal.crystallized:
            continue
        if len(crystal.members) < AUTOCATALYSIS_STREAK:
            continue

        recent_members = crystal.members[-AUTOCATALYSIS_STREAK:]
        high_similarity_count = sum(
            1 for m in recent_members
            if m.get("capture_similarity", 0.0) >= SELF_PREDICTION_THRESHOLD
        )

        if high_similarity_count >= AUTOCATALYSIS_STREAK:
            discovered_archetype, dominance_ratio, was_hybrid = discover_archetype(crystal)

            crystal.crystallized = True
            crystal.birth_cycle = cycle
            crystal.agent_type = discovered_archetype

            state.crystals_formed += 1

            if discovered_archetype == "SHEPHERD":
                state.governance_nodes += 1
                emit_receipt("governance_node", {
                    "tenant_id": "simulation",
                    "receipt_type": "governance_node",
                    "crystal_id": crystal.crystal_id,
                    "cycle": cycle,
                    "total_nodes": state.governance_nodes
                })
            elif discovered_archetype == "ARCHITECT":
                state.architect_formations += 1
                emit_receipt("architect_formation", {
                    "tenant_id": "simulation",
                    "receipt_type": "architect_formation",
                    "crystal_id": crystal.crystal_id,
                    "cycle": cycle,
                    "size": len(crystal.members),
                    "effect_distribution": dict(crystal.effect_distribution)
                })
            elif discovered_archetype == "HUNTER":
                state.hunter_formations += 1
            elif discovered_archetype == "HYBRID":
                state.hybrid_formations += 1

            if state.crystals_formed == 1 and discovered_archetype in state.first_capture_distribution:
                state.first_capture_distribution[discovered_archetype] += 1

            if state.crystals_formed == 1:
                for beacon in state.beacons:
                    beacon.strength *= CRYSTALLIZED_BEACON_BOOST

            archetype_discovery_receipt = emit_receipt("archetype_discovery", {
                "tenant_id": "simulation",
                "receipt_type": "archetype_discovery",
                "crystal_id": crystal.crystal_id,
                "cycle": cycle,
                "effect_distribution": dict(crystal.effect_distribution),
                "discovered_archetype": discovered_archetype,
                "dominance_ratio": dominance_ratio,
                "was_hybrid": was_hybrid
            })
            state.receipt_ledger.append(archetype_discovery_receipt)

            return emit_receipt("agent_birth", {
                "tenant_id": "simulation",
                "receipt_type": "agent_birth",
                "agent_type": crystal.agent_type,
                "discovery_method": "self_measurement",
                "effect_distribution": dict(crystal.effect_distribution),
                "dominance_ratio": dominance_ratio,
                "was_hybrid": was_hybrid,
                "birth_cycle": cycle,
                "pattern_size": len(crystal.members),
                "autocatalysis_streak": AUTOCATALYSIS_STREAK,
                "seed_id": crystal.seed_id,
                "coherence": crystal.coherence,
                "first_crystal": state.crystals_formed == 1
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
