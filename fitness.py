"""
fitness.py - Multi-Dimensional Fitness Module (v11)

Computes multi-dimensional fitness scores to prevent single-metric death spirals.
Implements cohort-balanced review to protect last-of-archetype patterns.
Uses Thompson sampling for exploration vs exploitation in selection.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import json
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# =============================================================================
# CLAUDEME §8 CORE FUNCTIONS - Import from receipts.py (v6 foundation)
# =============================================================================

from receipts import dual_hash, emit_receipt, StopRule


# =============================================================================
# CONSTANTS (from v6 strategy)
# =============================================================================

WEIGHT_ROI = 0.4
WEIGHT_DIVERSITY = 0.3
WEIGHT_STABILITY = 0.2
WEIGHT_RECENCY = 0.1

MIN_ARCHETYPE_DIVERSITY = 1
DEFAULT_SURVIVAL_THRESHOLD = 0.0
RECENCY_DECAY_DAYS = 30

# Module exports for receipt types
RECEIPT_SCHEMA = ["multi_fitness_score", "selection_event", "archetype_protection"]


# =============================================================================
# CORE FUNCTION 1: multi_fitness
# =============================================================================

def multi_fitness(pattern: dict, population: List[dict] = None,
                  tenant_id: str = "default") -> dict:
    """
    Multi-dimensional fitness prevents single-metric death.

    Four components (each 0.0-1.0):
    - roi_contribution: entropy reduction from the pattern
    - diversity_contribution: uniqueness of archetype
    - stability_contribution: inverse of fitness variance
    - recency_bonus: time decay for recent activity

    Weights (fixed, from v6 strategy):
    - roi: 0.4
    - diversity: 0.3
    - stability: 0.2
    - recency: 0.1

    Formula: fitness = 0.4*roi + 0.3*diversity + 0.2*stability + 0.1*recency

    Args:
        pattern: Dict with fields like 'id', 'archetype', 'fitness_mean', 'fitness_var',
                 'last_receipt_ts', 'roi' (optional)
        population: Optional list of all patterns for diversity calculation
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        multi_fitness_score receipt with all component values and final score
    """
    population = population or []
    pattern_id = pattern.get("id", "unknown")

    # Component 1: ROI contribution (normalize to [0,1])
    # Use pattern's roi if available, or fitness_mean as proxy
    raw_roi = pattern.get("roi", pattern.get("fitness_mean", 0.0))
    # Normalize: sigmoid-like mapping, assuming roi in [-2, 2] typical range
    roi_contribution = 1.0 / (1.0 + math.exp(-raw_roi)) if raw_roi != 0 else 0.5
    roi_contribution = max(0.0, min(1.0, roi_contribution))

    # Component 2: Diversity contribution (uniqueness of archetype)
    archetype = pattern.get("archetype", "default")
    if population:
        archetype_count = sum(1 for p in population if p.get("archetype") == archetype)
    else:
        archetype_count = 1
    diversity_contribution = 1.0 / archetype_count if archetype_count > 0 else 1.0
    diversity_contribution = max(0.0, min(1.0, diversity_contribution))

    # Component 3: Stability contribution (inverse of fitness variance)
    fitness_var = pattern.get("fitness_var", 0.0)
    stability_contribution = 1.0 / (1.0 + fitness_var)
    stability_contribution = max(0.0, min(1.0, stability_contribution))

    # Component 4: Recency bonus (time decay)
    last_receipt_ts = pattern.get("last_receipt_ts")
    if last_receipt_ts:
        try:
            if isinstance(last_receipt_ts, str):
                # Parse ISO8601 timestamp
                last_dt = datetime.fromisoformat(last_receipt_ts.replace("Z", "+00:00"))
            else:
                last_dt = last_receipt_ts
            now = datetime.now(timezone.utc)
            days_since = (now - last_dt).total_seconds() / 86400.0
            recency_bonus = math.exp(-days_since / RECENCY_DECAY_DAYS)
        except (ValueError, TypeError):
            recency_bonus = 0.5  # Default if parsing fails
    else:
        recency_bonus = 0.5  # Default if no timestamp
    recency_bonus = max(0.0, min(1.0, recency_bonus))

    # Weighted sum
    score = (
        WEIGHT_ROI * roi_contribution +
        WEIGHT_DIVERSITY * diversity_contribution +
        WEIGHT_STABILITY * stability_contribution +
        WEIGHT_RECENCY * recency_bonus
    )
    score = max(0.0, min(1.0, score))

    return emit_receipt("multi_fitness_score", {
        "tenant_id": tenant_id,
        "pattern_id": pattern_id,
        "score": score,
        "components": {
            "roi": roi_contribution,
            "diversity": diversity_contribution,
            "stability": stability_contribution,
            "recency": recency_bonus
        },
        "weights": {
            "roi": WEIGHT_ROI,
            "diversity": WEIGHT_DIVERSITY,
            "stability": WEIGHT_STABILITY,
            "recency": WEIGHT_RECENCY
        }
    })


# =============================================================================
# CORE FUNCTION 2: cohort_balanced_review
# =============================================================================

def cohort_balanced_review(pattern: dict, population: List[dict],
                           tenant_id: str = "default") -> dict:
    """
    Cannot kill last agent of an archetype.

    Logic:
    - Count how many patterns share pattern['archetype']
    - If count == 1, this is the last of its kind
    - MIN_ARCHETYPE_DIVERSITY = 1 (constant)
    - If killing this pattern would reduce archetype count below MIN_ARCHETYPE_DIVERSITY,
      block the kill

    Args:
        pattern: The pattern being considered for removal
        population: List of all patterns in population
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        archetype_protection receipt with fields: archetype, count_before,
        would_be_last (bool), action (allow|block)
    """
    pattern_id = pattern.get("id", "unknown")
    archetype = pattern.get("archetype", "default")

    # Count patterns with same archetype in population
    count_before = sum(1 for p in population if p.get("archetype") == archetype)

    # Would killing this pattern reduce count below MIN_ARCHETYPE_DIVERSITY?
    count_after = count_before - 1
    would_be_last = count_after < MIN_ARCHETYPE_DIVERSITY

    # Block if would leave archetype below minimum diversity
    action = "block" if would_be_last else "allow"

    return emit_receipt("archetype_protection", {
        "tenant_id": tenant_id,
        "pattern_id": pattern_id,
        "archetype": archetype,
        "count_before": count_before,
        "would_be_last": would_be_last,
        "action": action
    })


# =============================================================================
# CORE FUNCTION 3: thompson_sample
# =============================================================================

def thompson_sample(patterns: List[dict], survival_threshold: float = DEFAULT_SURVIVAL_THRESHOLD,
                    tenant_id: str = "default") -> dict:
    """
    Sample from fitness distributions using Thompson sampling.
    Balances exploration vs exploitation.

    Logic:
    - For each pattern, get fitness_mean and fitness_var
    - Sample from normal distribution: sampled_fitness = normal(mean, sqrt(variance))
    - High variance patterns get explored (might sample high even if mean is low)
    - Low variance patterns are exploited predictably
    - If sampled_fitness > survival_threshold: pattern survives
    - If sampled_fitness <= survival_threshold: pattern goes to SUPERPOSITION state

    SUPERPOSITION meaning: Pattern is not deleted. It becomes dormant/potential.
    Can resurface if conditions change.

    Args:
        patterns: List of dicts with 'id', 'fitness_mean', 'fitness_var'
        survival_threshold: Threshold for survival (default 0.0)
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        selection_event receipt with fields: patterns_evaluated, survivors (list),
        superposition (list), threshold_used
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

        if sampled > survival_threshold:
            survivors.append(pattern_id)
        else:
            superposition.append(pattern_id)

    return emit_receipt("selection_event", {
        "tenant_id": tenant_id,
        "patterns_evaluated": len(patterns),
        "survivors": survivors,
        "superposition": superposition,
        "threshold_used": survival_threshold,
        "sampling_method": "thompson"
    })


# =============================================================================
# RECEIPT TYPE 1: multi_fitness_score
# =============================================================================

# --- SCHEMA ---
MULTI_FITNESS_SCORE_SCHEMA = {
    "receipt_type": "multi_fitness_score",
    "ts": "ISO8601",
    "tenant_id": "str",  # REQUIRED per CLAUDEME 4.1:137
    "pattern_id": "str",
    "score": "float 0.0-1.0",
    "components": {
        "roi": "float 0.0-1.0",
        "diversity": "float 0.0-1.0",
        "stability": "float 0.0-1.0",
        "recency": "float 0.0-1.0"
    },
    "weights": {
        "roi": 0.4,
        "diversity": 0.3,
        "stability": 0.2,
        "recency": 0.1
    },
    "payload_hash": "str"
}


# --- TEST ---
def test_multi_fitness_score():
    """Test multi_fitness_score receipt emission."""
    pattern = {
        "id": "test_pattern",
        "archetype": "explorer",
        "fitness_mean": 0.5,
        "fitness_var": 0.1
    }
    r = multi_fitness(pattern, tenant_id="test_tenant")
    assert r["receipt_type"] == "multi_fitness_score"
    assert r["tenant_id"] == "test_tenant"
    assert 0.0 <= r["score"] <= 1.0
    assert "components" in r
    assert all(k in r["components"] for k in ["roi", "diversity", "stability", "recency"])
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]  # dual_hash format


# --- STOPRULE ---
def stoprule_multi_fitness(tenant_id: str, pattern_id: str, score: float) -> None:
    """
    Multi-fitness stoprule - triggers alert when score drops critically low.
    """
    if score < 0.1:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "multi_fitness",
            "baseline": 0.1,
            "delta": score - 0.1,
            "classification": "deviation",
            "action": "alert",
            "pattern_id": pattern_id
        })


# =============================================================================
# RECEIPT TYPE 2: archetype_protection
# =============================================================================

# --- SCHEMA ---
ARCHETYPE_PROTECTION_SCHEMA = {
    "receipt_type": "archetype_protection",
    "ts": "ISO8601",
    "tenant_id": "str",
    "pattern_id": "str",
    "archetype": "str",
    "count_before": "int",
    "would_be_last": "bool",
    "action": "str",  # "allow" | "block"
    "payload_hash": "str"
}


# --- TEST ---
def test_archetype_protection():
    """Test archetype_protection receipt emission."""
    pattern = {"id": "p1", "archetype": "explorer"}
    population = [
        {"id": "p1", "archetype": "explorer"},
        {"id": "p2", "archetype": "builder"}
    ]
    r = cohort_balanced_review(pattern, population, tenant_id="test_tenant")
    assert r["receipt_type"] == "archetype_protection"
    assert r["tenant_id"] == "test_tenant"
    assert r["action"] in ["allow", "block"]
    assert isinstance(r["would_be_last"], bool)
    assert "payload_hash" in r


# --- STOPRULE ---
def stoprule_archetype_protection(tenant_id: str, archetype: str, action: str) -> None:
    """
    Archetype protection stoprule - logs when protection is triggered.
    """
    if action == "block":
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "archetype_protection",
            "baseline": MIN_ARCHETYPE_DIVERSITY,
            "delta": -1,
            "classification": "deviation",
            "action": "alert",
            "archetype": archetype
        })


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
    "superposition": ["str"],  # patterns in potential state
    "threshold_used": "float",
    "sampling_method": "str",  # "thompson"
    "payload_hash": "str"
}


# --- TEST ---
def test_selection_event():
    """Test selection_event receipt emission."""
    patterns = [
        {"id": "p1", "fitness_mean": 1.0, "fitness_var": 0.01},
        {"id": "p2", "fitness_mean": -1.0, "fitness_var": 0.01}
    ]
    r = thompson_sample(patterns, tenant_id="test_tenant")
    assert r["receipt_type"] == "selection_event"
    assert r["tenant_id"] == "test_tenant"
    assert r["sampling_method"] == "thompson"
    assert isinstance(r["survivors"], list)
    assert isinstance(r["superposition"], list)
    assert "payload_hash" in r


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
    "WEIGHT_ROI",
    "WEIGHT_DIVERSITY",
    "WEIGHT_STABILITY",
    "WEIGHT_RECENCY",
    "MIN_ARCHETYPE_DIVERSITY",
    "DEFAULT_SURVIVAL_THRESHOLD",
    "RECENCY_DECAY_DAYS",
    # Core functions
    "multi_fitness",
    "cohort_balanced_review",
    "thompson_sample",
    # Stoprules
    "stoprule_multi_fitness",
    "stoprule_archetype_protection",
    "stoprule_selection_event",
    # Core utilities
    "emit_receipt",
    "dual_hash",
    "StopRule",
]
