"""
recombine.py - Sexual Reproduction Module

Implements v12 genetic operations: patterns mate to evolve the species.
Parents unchanged; offspring inherits traits from both. Germline evolution
through proven fitness, not self-modification.

CLAUDEME v3.1 Compliant: Pure functions with emit_receipt() pattern.
"""

import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from entropy import agent_fitness, emit_receipt, dual_hash
from autocatalysis import is_alive

# =============================================================================
# CONSTANTS
# =============================================================================

MUTATION_RATE = 0.01  # 1% chance per receipt
GERMLINE_FITNESS_THRESHOLD = 0.5  # minimum for germline consideration
GERMLINE_DAYS_REQUIRED = 30  # consecutive days above threshold
COMPATIBILITY_THRESHOLD = 0.7  # minimum problem_type similarity

# Module exports for receipt types
RECEIPT_SCHEMA = ["recombination_event", "offspring_created", "germline_update"]


# =============================================================================
# CORE FUNCTION 1: crossover
# =============================================================================

def crossover(receipts_a: List[dict], receipts_b: List[dict]) -> List[dict]:
    """
    50/50 random selection from parent receipt lists at each position.

    Args:
        receipts_a: First parent's receipts
        receipts_b: Second parent's receipts

    Returns:
        list: Combined receipts (length = min(len_a, len_b))
    """
    min_len = min(len(receipts_a), len(receipts_b))
    offspring_receipts = []

    for i in range(min_len):
        chosen = random.choice([receipts_a[i], receipts_b[i]])
        offspring_receipts.append(chosen)

    return offspring_receipts


# =============================================================================
# CORE FUNCTION 2: mutate
# =============================================================================

def mutate(receipts: List[dict]) -> List[dict]:
    """
    Introduce random variation at MUTATION_RATE per receipt.

    Args:
        receipts: Receipt list to mutate

    Returns:
        list: Mutated receipts (preserves structure)
    """
    mutated = []

    for receipt in receipts:
        if random.random() < MUTATION_RATE:
            # Small parameter tweak - copy and adjust numeric fields
            mutated_receipt = receipt.copy()
            for key, val in mutated_receipt.items():
                if isinstance(val, (int, float)) and key not in ["ts", "tenant_id"]:
                    mutated_receipt[key] = val * random.uniform(0.95, 1.05)
            mutated.append(mutated_receipt)
        else:
            mutated.append(receipt)

    return mutated


# =============================================================================
# CORE FUNCTION 3: recombine
# =============================================================================

def recombine(pattern_a: Dict, pattern_b: Dict) -> Dict:
    """
    Sexual recombination of two parent patterns into offspring.

    Args:
        pattern_a: First parent pattern
        pattern_b: Second parent pattern

    Returns:
        dict: recombination_event receipt
    """
    receipts_a = pattern_a.get("receipts", [])
    receipts_b = pattern_b.get("receipts", [])

    # Crossover
    offspring_receipts = crossover(receipts_a, receipts_b)
    crossover_points = list(range(len(offspring_receipts)))

    # Mutate
    mutated_receipts = mutate(offspring_receipts)
    mutation_positions = [i for i, (orig, mut) in enumerate(zip(offspring_receipts, mutated_receipts)) if orig != mut]

    # Create offspring pattern
    offspring_id = dual_hash(str(mutated_receipts))

    return emit_receipt("recombination_event", {
        "tenant_id": pattern_a.get("tenant_id", "default"),
        "offspring_id": offspring_id,
        "parent_a_id": pattern_a.get("pattern_id", "unknown"),
        "parent_b_id": pattern_b.get("pattern_id", "unknown"),
        "crossover_points": crossover_points,
        "mutation_applied": len(mutation_positions) > 0,
        "mutation_positions": mutation_positions
    })


# =============================================================================
# CORE FUNCTION 4: mate_selection
# =============================================================================

def mate_selection(patterns: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    Identify compatible pattern pairs for recombination.

    Criteria:
    - Similar problem_type (>= COMPATIBILITY_THRESHOLD)
    - Different domain
    - Both have positive fitness
    - Affinity >= MIN_AFFINITY_THRESHOLD (Grok validated gate)

    Args:
        patterns: List of pattern dicts with problem_type, domain, fitness

    Returns:
        list: Tuples of (pattern_a, pattern_b) eligible for mating
    """
    # Import here to avoid circular dependency
    from sim.dynamics_genesis import get_domain_affinity
    from sim.constants import MIN_AFFINITY_THRESHOLD

    pairs = []

    for i, pattern_a in enumerate(patterns):
        for pattern_b in patterns[i+1:]:
            # Check different domains
            domain_a = pattern_a.get("domain")
            domain_b = pattern_b.get("domain")

            if domain_a == domain_b:
                continue

            # Check positive fitness
            if pattern_a.get("fitness", 0) <= 0 or pattern_b.get("fitness", 0) <= 0:
                continue

            # Check problem_type similarity (simple string match for now)
            type_a = pattern_a.get("problem_type", "")
            type_b = pattern_b.get("problem_type", "")
            if type_a != type_b:
                continue

            # Check affinity threshold (Grok validated: below 0.48 adds noise > signal)
            affinity = get_domain_affinity(domain_a, domain_b)
            if affinity < MIN_AFFINITY_THRESHOLD:
                # Emit block receipt
                emit_receipt("affinity_threshold_block_receipt", {
                    "tenant_id": pattern_a.get("tenant_id", "default"),
                    "parent_a_id": pattern_a.get("pattern_id", "unknown"),
                    "parent_a_domain": domain_a,
                    "parent_b_id": pattern_b.get("pattern_id", "unknown"),
                    "parent_b_domain": domain_b,
                    "affinity_score": affinity,
                    "threshold": MIN_AFFINITY_THRESHOLD,
                    "blocked_reason": "affinity below threshold: noise > signal"
                })
                continue  # Skip this pair

            # All checks passed - compatible pair
            pairs.append((pattern_a, pattern_b))

    return pairs


# =============================================================================
# CORE FUNCTION 5: germline_contribution
# =============================================================================

def germline_contribution(offspring: Dict, fitness_history: List[float]) -> Optional[Dict]:
    """
    Evaluate if successful offspring should update archetype templates.

    Criteria: fitness > GERMLINE_FITNESS_THRESHOLD for GERMLINE_DAYS_REQUIRED
    consecutive days.

    Args:
        offspring: Offspring pattern dict
        fitness_history: List of daily fitness scores (length >= days)

    Returns:
        germline_update receipt if eligible, None otherwise
    """
    if len(fitness_history) < GERMLINE_DAYS_REQUIRED:
        return None

    # Check last N days all above threshold
    recent_history = fitness_history[-GERMLINE_DAYS_REQUIRED:]
    if all(f > GERMLINE_FITNESS_THRESHOLD for f in recent_history):
        return emit_receipt("germline_update", {
            "tenant_id": offspring.get("tenant_id", "default"),
            "offspring_id": offspring.get("pattern_id", "unknown"),
            "archetype": offspring.get("archetype", "HUNTER"),
            "fitness_at_contribution": recent_history[-1],
            "days_above_threshold": GERMLINE_DAYS_REQUIRED,
            "proposed_changes": offspring.get("traits", {}),
            "requires_hitl": True
        })

    return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "MUTATION_RATE",
    "GERMLINE_FITNESS_THRESHOLD",
    "GERMLINE_DAYS_REQUIRED",
    "COMPATIBILITY_THRESHOLD",
    # Core functions
    "recombine",
    "mate_selection",
    "crossover",
    "mutate",
    "germline_contribution",
]
