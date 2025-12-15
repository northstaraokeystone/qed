"""
sim/measurement.py - Observer Paradigm Measurement Functions

Functions for measuring entropy states in the observer paradigm.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import math

from entropy import system_entropy
from .constants import PLANCK_ENTROPY, LANDAUER_COEFFICIENT


def measure_state(receipt_ledger: list, vacuum_floor: float = None) -> float:
    """
    Measure current system entropy state.

    Args:
        receipt_ledger: List of receipts
        vacuum_floor: Optional fluctuating vacuum floor (defaults to PLANCK_ENTROPY)

    Returns:
        float: System entropy, never returns 0 (minimum vacuum_floor)
    """
    if vacuum_floor is None:
        vacuum_floor = PLANCK_ENTROPY

    result = system_entropy(receipt_ledger)
    return max(result, vacuum_floor)


def measure_observation_cost(operations: int) -> float:
    """
    Measure entropy cost of observation (Landauer principle).

    Formula: LANDAUER_COEFFICIENT * log2(operations + 2)
    The +2 ensures minimum ~1 bit even at 0 operations (log2(2) = 1)

    Args:
        operations: Number of observations/decisions made this cycle

    Returns:
        float: Observation entropy for this cycle
    """
    return LANDAUER_COEFFICIENT * math.log2(operations + 2)


def measure_boundary_crossing(receipt_data: dict) -> float:
    """
    Measure entropy of crossing phase boundary (ClarityClean).

    Args:
        receipt_data: Receipt dictionary for the boundary event

    Returns:
        float: Boundary crossing entropy, never 0 (minimum PLANCK_ENTROPY)
    """
    result = system_entropy([receipt_data])
    return max(result, PLANCK_ENTROPY)


def measure_genesis(initial_patterns: list) -> float:
    """
    Measure entropy at simulation genesis (Big Bang).

    Called ONCE at simulation start in initialize_state().

    Args:
        initial_patterns: List of initial pattern dictionaries

    Returns:
        float: Genesis entropy, never 0 (minimum PLANCK_ENTROPY)
    """
    # Create receipts from initial patterns for entropy measurement
    initial_pattern_receipts = []
    for pattern in initial_patterns:
        initial_pattern_receipts.append({
            "receipt_type": "pattern_genesis",
            "pattern_id": pattern.get("pattern_id", "unknown"),
            "origin": pattern.get("origin", "unknown"),
            "tenant_id": "simulation"
        })

    result = system_entropy(initial_pattern_receipts)
    return max(result, PLANCK_ENTROPY)


def lineage_depth(pattern_id: str, genealogy: dict) -> int:
    """
    Calculate lineage depth by traversing parent chain to root.

    Unbounded traversal - LINEAGE_UNBOUNDED = True (Grok validated).
    VARIANCE_DECAY prevents amplification, depth limit unnecessary.

    Args:
        pattern_id: Pattern identifier
        genealogy: Dict mapping pattern_id -> parent_id (None for roots)

    Returns:
        int: Lineage depth (0 = root/no parents, N = N generations from root)

    Examples:
        >>> genealogy = {'D': 'C', 'C': 'B', 'B': 'A', 'A': None}
        >>> lineage_depth('D', genealogy)
        3
        >>> lineage_depth('A', genealogy)
        0
    """
    if pattern_id not in genealogy:
        return 0

    depth = 0
    current_id = pattern_id
    visited = set()  # Cycle detection

    while current_id in genealogy and genealogy[current_id] is not None:
        if current_id in visited:
            # Cycle detected - should not happen in valid genealogy
            break
        visited.add(current_id)
        current_id = genealogy[current_id]
        depth += 1

    return depth
