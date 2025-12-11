"""
entropy.py - The Fundamental Module

Computes Shannon entropy of receipt streams. Measures agent fitness as entropy
reduction per receipt. Runs Thompson sampling for selection pressure.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# =============================================================================
# CLAUDEME §8 CORE FUNCTIONS - Import from receipts.py (v6 foundation)
# =============================================================================

from receipts import dual_hash, emit_receipt, StopRule


# =============================================================================
# CONSTANTS
# =============================================================================

SURVIVAL_THRESHOLD = 0.0  # Patterns must have positive expected fitness
PLANCK_ENTROPY = 0.001  # Minimum entropy of existence (matches sim.py)

# Module exports for receipt types
RECEIPT_SCHEMA = ["entropy_measurement", "fitness_score", "selection_event"]


# =============================================================================
# CORE FUNCTION 1: system_entropy
# =============================================================================

def system_entropy(receipts: List[dict]) -> float:
    """
    Shannon entropy H = -sum(p(x) * log2(p(x))) over receipt_type distribution.

    Args:
        receipts: List of receipt dicts with 'receipt_type' field

    Returns:
        float: Shannon entropy in bits, never returns 0 (minimum PLANCK_ENTROPY)

    Edge cases:
        - Empty list -> PLANCK_ENTROPY (existence has minimum entropy)
        - Single receipt type -> PLANCK_ENTROPY (mathematical entropy = 0, but floor applied)
        - N types uniformly distributed -> max(Shannon entropy, PLANCK_ENTROPY)

    Note: PLANCK_ENTROPY floor ensures existence has information content.
    This prevents division-by-zero in conservation checks and enforces that
    structure = entropy (information theory).
    """
    if not receipts:
        return PLANCK_ENTROPY

    # Count receipt types
    type_counts = Counter(r.get("receipt_type", "unknown") for r in receipts)

    # Edge case: single receipt type has mathematical entropy = 0, but apply floor
    if len(type_counts) <= 1:
        return PLANCK_ENTROPY

    total = sum(type_counts.values())
    entropy = 0.0

    for count in type_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Apply floor to ensure minimum entropy
    return max(entropy, PLANCK_ENTROPY)


# =============================================================================
# CORE FUNCTION 2: agent_fitness
# =============================================================================

def agent_fitness(receipts_before: List[dict], receipts_after: List[dict],
                  pattern_receipt_count: int) -> float:
    """
    Compute agent fitness as entropy reduction per receipt.

    fitness = (H_before - H_after) / n_receipts

    Args:
        receipts_before: Receipts before pattern acted
        receipts_after: Receipts after pattern acted
        pattern_receipt_count: Number of receipts pattern emitted

    Returns:
        float: Fitness score
            - Positive = pattern reduces uncertainty = good
            - Negative = pattern adds noise = bad
            - Zero = pattern does nothing = useless

    Edge case: If pattern_receipt_count <= 0, return 0.0
    """
    if pattern_receipt_count <= 0:
        return 0.0

    h_before = system_entropy(receipts_before)
    h_after = system_entropy(receipts_after)

    return (h_before - h_after) / pattern_receipt_count


# =============================================================================
# CORE FUNCTION 3: compute_fitness_score
# =============================================================================

def compute_fitness_score(pattern_id: str, receipts_before: List[dict],
                         receipts_after: List[dict], n_receipts: int) -> dict:
    """
    Compute fitness score with instrumentation fields.

    Args:
        pattern_id: Pattern identifier
        receipts_before: Receipts before pattern acted
        receipts_after: Receipts after pattern acted
        n_receipts: Number of receipts pattern emitted

    Returns:
        dict with fields:
            - pattern_id: str
            - H_before: float (entropy before)
            - H_after: float (entropy after)
            - n_receipts: int
            - fitness: float (numeric score)
            - fitness_class: str (good/bad/neutral)
    """
    h_before = system_entropy(receipts_before)
    h_after = system_entropy(receipts_after)
    fitness = agent_fitness(receipts_before, receipts_after, n_receipts)

    # Classify fitness
    if fitness > 0:
        fitness_class = "good"
    elif fitness < 0:
        fitness_class = "bad"
    else:
        fitness_class = "neutral"

    return {
        "pattern_id": pattern_id,
        "H_before": h_before,
        "H_after": h_after,
        "n_receipts": n_receipts,
        "fitness": fitness,
        "fitness_class": fitness_class
    }


# =============================================================================
# CORE FUNCTION 4: cycle_entropy_delta
# =============================================================================

def cycle_entropy_delta(receipts_before: List[dict], receipts_after: List[dict]) -> float:
    """
    Compute entropy delta (reduction) between cycle states.

    delta = H_before - H_after (entropy REDUCTION)

    Args:
        receipts_before: Receipts at cycle start
        receipts_after: Receipts at cycle end

    Returns:
        float: Entropy delta
            - Positive = entropy decreased = healthy
            - Negative = entropy increased = degrading
    """
    h_before = system_entropy(receipts_before)
    h_after = system_entropy(receipts_after)

    return h_before - h_after


# =============================================================================
# CORE FUNCTION 4: selection_pressure
# =============================================================================

def selection_pressure(patterns: List[dict], threshold: float,
                       tenant_id: str) -> Tuple[List[str], dict]:
    """
    Thompson sampling over fitness distributions.

    For each pattern, sample from normal(mean, sqrt(var)). If sampled > threshold,
    survive. Else go to SUPERPOSITION state.

    Args:
        patterns: List of dicts with 'id', 'fitness_mean', 'fitness_var'
        threshold: Survival threshold (use SURVIVAL_THRESHOLD = 0.0)
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        Tuple of (survivors_list, selection_event_receipt)
    """
    survivors = []
    superposition = []

    for pattern in patterns:
        pattern_id = pattern.get("id", "unknown")
        mean = pattern.get("fitness_mean", 0.0)
        var = pattern.get("fitness_var", 0.0)

        # Thompson sampling: sample from normal distribution
        std = math.sqrt(var) if var > 0 else 0.0
        sampled = random.gauss(mean, std) if std > 0 else mean

        if sampled > threshold:
            survivors.append(pattern_id)
        else:
            superposition.append(pattern_id)

    receipt = emit_selection_event(
        tenant_id=tenant_id,
        patterns_evaluated=len(patterns),
        survivors=survivors,
        superposition=superposition,
        survival_threshold=threshold
    )

    return survivors, receipt


# =============================================================================
# RECEIPT TYPE 1: entropy_measurement
# =============================================================================

# --- SCHEMA ---
ENTROPY_MEASUREMENT_SCHEMA = {
    "receipt_type": "entropy_measurement",
    "ts": "ISO8601",
    "tenant_id": "str",  # REQUIRED per CLAUDEME 4.1:137
    "cycle_id": "int",
    "entropy_before": "float",
    "entropy_after": "float",
    "entropy_delta": "float",  # (before - after), positive = healthy
    "receipt_count": "int",
    "entropy_per_receipt": "float",
    "payload_hash": "str"  # dual_hash of measurement data
}


# --- EMIT ---
def emit_entropy_measurement(tenant_id: str, cycle_id: int,
                             receipts_before: List[dict],
                             receipts_after: List[dict]) -> dict:
    """Emit entropy_measurement receipt for system vital signs per cycle."""
    entropy_before = system_entropy(receipts_before)
    entropy_after = system_entropy(receipts_after)
    entropy_delta = entropy_before - entropy_after
    receipt_count = len(receipts_after)
    entropy_per_receipt = entropy_delta / receipt_count if receipt_count > 0 else 0.0

    return emit_receipt("entropy_measurement", {
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_delta": entropy_delta,
        "receipt_count": receipt_count,
        "entropy_per_receipt": entropy_per_receipt
    })


# --- TEST ---
def test_entropy_measurement():
    """Test entropy_measurement receipt emission."""
    receipts_before = [
        {"receipt_type": "a"},
        {"receipt_type": "b"}
    ]
    receipts_after = [
        {"receipt_type": "a"},
        {"receipt_type": "a"},
        {"receipt_type": "a"},
        {"receipt_type": "b"}
    ]
    r = emit_entropy_measurement("test_tenant", 1, receipts_before, receipts_after)
    assert r["receipt_type"] == "entropy_measurement"
    assert r["tenant_id"] == "test_tenant"
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]  # dual_hash format


# --- STOPRULE ---
# Track consecutive negative deltas for stoprule
_consecutive_negative_deltas = {}


def stoprule_entropy_measurement(tenant_id: str, entropy_delta: float) -> None:
    """
    When entropy_delta < 0 for 3+ consecutive cycles, emit anomaly_receipt
    with classification="degradation", action="escalate", then raise StopRule.
    """
    if tenant_id not in _consecutive_negative_deltas:
        _consecutive_negative_deltas[tenant_id] = 0

    if entropy_delta < 0:
        _consecutive_negative_deltas[tenant_id] += 1
    else:
        _consecutive_negative_deltas[tenant_id] = 0

    if _consecutive_negative_deltas[tenant_id] >= 3:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "entropy_delta",
            "baseline": 0.0,
            "delta": entropy_delta,
            "classification": "degradation",
            "action": "escalate"
        })
        _consecutive_negative_deltas[tenant_id] = 0
        raise StopRule(f"Entropy degradation: 3+ consecutive negative deltas for {tenant_id}")


# =============================================================================
# RECEIPT TYPE 2: fitness_score
# =============================================================================

# --- SCHEMA ---
FITNESS_SCORE_SCHEMA = {
    "receipt_type": "fitness_score",
    "ts": "ISO8601",
    "tenant_id": "str",
    "pattern_id": "str",
    "entropy_before": "float",
    "entropy_after": "float",
    "receipt_count": "int",
    "fitness": "float",  # (before - after) / count
    "classification": "str",  # "positive" | "negative" | "zero"
    "payload_hash": "str"
}


# --- EMIT ---
def emit_fitness_score(tenant_id: str, pattern_id: str,
                       receipts_before: List[dict], receipts_after: List[dict],
                       receipt_count: int) -> dict:
    """Emit fitness_score receipt for per-pattern health measurement."""
    entropy_before = system_entropy(receipts_before)
    entropy_after = system_entropy(receipts_after)
    fitness = agent_fitness(receipts_before, receipts_after, receipt_count)

    if fitness > 0:
        classification = "positive"
    elif fitness < 0:
        classification = "negative"
    else:
        classification = "zero"

    return emit_receipt("fitness_score", {
        "tenant_id": tenant_id,
        "pattern_id": pattern_id,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "receipt_count": receipt_count,
        "fitness": fitness,
        "classification": classification
    })


# --- TEST ---
def test_fitness_score():
    """Test fitness_score receipt emission."""
    receipts_before = [
        {"receipt_type": "a"},
        {"receipt_type": "b"}
    ]
    receipts_after = [
        {"receipt_type": "a"}
    ]
    r = emit_fitness_score("test_tenant", "pattern_001",
                           receipts_before, receipts_after, 1)
    assert r["receipt_type"] == "fitness_score"
    assert r["tenant_id"] == "test_tenant"
    assert r["pattern_id"] == "pattern_001"
    assert r["classification"] in ["positive", "negative", "zero"]


# --- STOPRULE ---
def stoprule_fitness_score(tenant_id: str, pattern_id: str, fitness: float) -> None:
    """
    When fitness < 0, emit anomaly_receipt with classification="deviation",
    action="alert". Note: In v10, do NOT raise StopRule - measurement only.
    v11 adds multi-dimensional fitness.
    """
    if fitness < 0:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "fitness",
            "baseline": 0.0,
            "delta": fitness,
            "classification": "deviation",
            "action": "alert",
            "pattern_id": pattern_id
        })
        # v10: measurement only, no StopRule


# =============================================================================
# RECEIPT TYPE 3: selection_event
# =============================================================================

# --- SCHEMA ---
SELECTION_EVENT_SCHEMA = {
    "receipt_type": "selection_event",
    "ts": "ISO8601",
    "tenant_id": "str",
    "patterns_evaluated": "int",
    "survivors": ["str"],  # list of pattern_id strings
    "superposition": ["str"],  # patterns returning to potential
    "survival_threshold": "float",
    "sampling_method": "str",  # "thompson"
    "payload_hash": "str"
}


# --- EMIT ---
def emit_selection_event(tenant_id: str, patterns_evaluated: int,
                         survivors: List[str], superposition: List[str],
                         survival_threshold: float) -> dict:
    """Emit selection_event receipt for Thompson sampling audit trail."""
    return emit_receipt("selection_event", {
        "tenant_id": tenant_id,
        "patterns_evaluated": patterns_evaluated,
        "survivors": survivors,
        "superposition": superposition,
        "survival_threshold": survival_threshold,
        "sampling_method": "thompson"
    })


# --- TEST ---
def test_selection_event():
    """Test selection_event receipt emission."""
    r = emit_selection_event(
        tenant_id="test_tenant",
        patterns_evaluated=5,
        survivors=["p1", "p2"],
        superposition=["p3", "p4", "p5"],
        survival_threshold=0.0
    )
    assert r["receipt_type"] == "selection_event"
    assert r["tenant_id"] == "test_tenant"
    assert r["sampling_method"] == "thompson"
    assert len(r["survivors"]) == 2
    assert len(r["superposition"]) == 3


# --- STOPRULE ---
def stoprule_selection_event(tenant_id: str, survivors: List[str],
                             patterns_evaluated: int) -> None:
    """
    Selection event stoprule - triggers when all patterns fail selection.
    """
    if patterns_evaluated > 0 and len(survivors) == 0:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "selection",
            "baseline": patterns_evaluated,
            "delta": -patterns_evaluated,
            "classification": "degradation",
            "action": "alert"
        })


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "SURVIVAL_THRESHOLD",
    # Core functions
    "system_entropy",
    "agent_fitness",
    "compute_fitness_score",
    "cycle_entropy_delta",
    "selection_pressure",
    # Emit functions
    "emit_entropy_measurement",
    "emit_fitness_score",
    "emit_selection_event",
    # Stoprules
    "stoprule_entropy_measurement",
    "stoprule_fitness_score",
    "stoprule_selection_event",
    # Core utilities
    "emit_receipt",
    "dual_hash",
    "StopRule",
]
