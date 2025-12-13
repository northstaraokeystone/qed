"""
sim/nucleation_crystals.py - Crystallization and Archetype Discovery

Crystal formation, autocatalysis detection, and archetype discovery.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

from typing import Optional

from entropy import emit_receipt

from .constants import (
    AUTOCATALYSIS_STREAK, SELF_PREDICTION_THRESHOLD, CRYSTALLIZED_BEACON_BOOST,
    ARCHETYPE_DOMINANCE_THRESHOLD, ARCHITECT_SIZE_TRIGGER, HUNTER_SIZE_TRIGGER,
    EFFECT_ENTROPY_INCREASE, EFFECT_RESONANCE_TRIGGER, EFFECT_SYMMETRY_BREAK
)
from .types_state import SimState, Crystal


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
