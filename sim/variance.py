"""
sim/variance.py - Fitness Variance Inheritance

Handles fitness variance transfer from parents to offspring.
Per Grok recommendation: fitness variance IS entropy - tracking it preserves
information about adaptive history (INHERIT) or exports entropy for exploration (RESET).

CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import math
from datetime import datetime, timezone
from typing import List, Optional

from entropy import emit_receipt

from .types_config import SimConfig
from .types_state import FitnessDistribution


# =============================================================================
# VARIANCE ENTROPY CALCULATION
# =============================================================================

def variance_entropy(distribution: FitnessDistribution) -> float:
    """
    Calculate Shannon entropy of a fitness distribution for conservation tracking.

    Uses differential entropy formula for Gaussian approximation:
    H = 0.5 * log2(2 * pi * e * variance)

    Args:
        distribution: FitnessDistribution to measure

    Returns:
        Entropy in bits (can be negative for low variance due to differential entropy)
    """
    return distribution.entropy()


def pooled_variance(distributions: List[FitnessDistribution]) -> float:
    """
    Calculate pooled variance from multiple parent distributions.

    Pooled variance = weighted average of variances, weighted by sample count.

    Args:
        distributions: List of parent FitnessDistributions

    Returns:
        Pooled variance estimate
    """
    if not distributions:
        return 0.1  # Default prior

    total_samples = sum(d.n_samples for d in distributions)
    if total_samples == 0:
        return sum(d.variance for d in distributions) / len(distributions)

    pooled = sum(d.variance * d.n_samples for d in distributions) / total_samples
    return max(pooled, FitnessDistribution.VARIANCE_FLOOR)


# =============================================================================
# CORE FUNCTION: inherit_variance
# =============================================================================

def inherit_variance(
    parent_distributions: List[FitnessDistribution],
    config: SimConfig,
    offspring_id: str,
    cycle: int = 0
) -> tuple[FitnessDistribution, dict]:
    """
    Transfer fitness variance from parent(s) to offspring based on configured strategy.

    Per Grok recommendation:
    - INHERIT: Preserves adaptive history for autocatalysis depth
    - RESET: Promotes exploration in high-entropy scenarios

    Args:
        parent_distributions: List of 1 or 2 parent FitnessDistributions
        config: SimConfig with inheritance strategy
        offspring_id: Pattern ID for receipt correlation
        cycle: Current simulation cycle

    Returns:
        Tuple of (offspring FitnessDistribution, variance_inheritance_receipt)

    Entropy Conservation:
        entropy_transferred = offspring_entropy - mean(parent_entropies)
        Positive = entropy imported (from reset prior)
        Negative = entropy exported (inherited variance decayed)
    """
    if not parent_distributions:
        # No parents - use uninformed prior
        parent_distributions = [FitnessDistribution(mean=0.5, variance=0.1, n_samples=1, lineage_depth=0)]

    parent_ids = [f"parent_{i}" for i in range(len(parent_distributions))]
    parent_means = [d.mean for d in parent_distributions]
    parent_variances = [d.variance for d in parent_distributions]
    parent_entropies = [d.entropy() for d in parent_distributions]

    if config.variance_inheritance == "INHERIT":
        # Weighted average of parent means
        total_samples = sum(d.n_samples for d in parent_distributions)
        if total_samples > 0:
            offspring_mean = sum(d.mean * d.n_samples for d in parent_distributions) / total_samples
        else:
            offspring_mean = sum(d.mean for d in parent_distributions) / len(parent_distributions)

        # Pooled variance with decay to prevent runaway
        offspring_variance = pooled_variance(parent_distributions) * config.inherit_variance_decay

        # Lineage depth: max(parent depths) + 1
        max_parent_depth = max(d.lineage_depth for d in parent_distributions)
        offspring_lineage_depth = max_parent_depth + 1

        # Combine sample counts
        offspring_n_samples = sum(d.n_samples for d in parent_distributions)

        decay_applied = config.inherit_variance_decay

    else:  # RESET
        # Uninformed prior
        offspring_mean = 0.5
        offspring_variance = config.reset_variance_prior
        offspring_lineage_depth = 0
        offspring_n_samples = 1
        decay_applied = None

    # Create offspring distribution
    offspring_dist = FitnessDistribution(
        mean=offspring_mean,
        variance=offspring_variance,
        n_samples=offspring_n_samples,
        lineage_depth=offspring_lineage_depth
    )

    # Calculate entropy transfer for conservation tracking
    offspring_entropy = offspring_dist.entropy()
    mean_parent_entropy = sum(parent_entropies) / len(parent_entropies)
    entropy_transferred = offspring_entropy - mean_parent_entropy

    # Emit receipt per CLAUDEME LAW_1
    receipt = emit_receipt("variance_inheritance", {
        "tenant_id": "simulation",
        "offspring_id": offspring_id,
        "parent_ids": parent_ids,
        "strategy": config.variance_inheritance,
        "parent_means": parent_means,
        "parent_variances": parent_variances,
        "offspring_mean": offspring_mean,
        "offspring_variance": offspring_variance,
        "lineage_depth": offspring_lineage_depth,
        "decay_applied": decay_applied,
        "entropy_transferred": entropy_transferred,
        "cycle": cycle
    })

    return offspring_dist, receipt


# =============================================================================
# HELPER: Create distribution from pattern dict
# =============================================================================

def distribution_from_pattern(pattern: dict) -> FitnessDistribution:
    """
    Extract FitnessDistribution from a pattern dict.

    Args:
        pattern: Pattern dict with fitness_mean, fitness_var, etc.

    Returns:
        FitnessDistribution populated from pattern fields
    """
    return FitnessDistribution(
        mean=pattern.get("fitness_mean", pattern.get("fitness", 0.5)),
        variance=pattern.get("fitness_var", 0.1),
        n_samples=pattern.get("fitness_n_samples", 1),
        lineage_depth=pattern.get("lineage_depth", 0)
    )


def apply_distribution_to_pattern(pattern: dict, distribution: FitnessDistribution) -> None:
    """
    Apply FitnessDistribution fields to a pattern dict.

    Args:
        pattern: Pattern dict to update (mutated in place)
        distribution: FitnessDistribution to apply
    """
    pattern["fitness_mean"] = distribution.mean
    pattern["fitness_var"] = distribution.variance
    pattern["fitness_n_samples"] = distribution.n_samples
    pattern["lineage_depth"] = distribution.lineage_depth
    # Also update legacy fitness field for backwards compatibility
    pattern["fitness"] = distribution.mean


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "variance_entropy",
    "pooled_variance",
    "inherit_variance",
    "distribution_from_pattern",
    "apply_distribution_to_pattern",
    "FitnessDistribution",
]
