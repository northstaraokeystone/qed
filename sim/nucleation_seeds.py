"""
sim/nucleation_seeds.py - Seed Initialization and Counselor Competition

Quantum seed initialization, counselor scoring, and kick capture.
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
    AUTOCATALYSIS_AMPLIFICATION, GOVERNANCE_BIAS, ENTROPY_AMPLIFIER, SYMMETRY_BIAS,
    EFFECT_ENTROPY_INCREASE, EFFECT_RESONANCE_TRIGGER, EFFECT_SYMMETRY_BREAK,
    UNIFORM_KICK_DISTRIBUTION, EFFECT_TYPES
)
from .types_state import SimState, Seed, Beacon, Counselor, Crystal


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
