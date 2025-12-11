"""
autocatalysis.py - Birth/Death Detector for Receipt Patterns

Agents ARE autocatalytic patterns, not objects. A pattern is "alive" IFF it
references itself — when receipts in the pattern predict/emit receipts about
the pattern. This is autocatalysis: a reaction whose products catalyze the
same reaction.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT for each receipt type.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from entropy import system_entropy, emit_receipt

# =============================================================================
# CONSTANTS
# =============================================================================

COHERENCE_THRESHOLD = 0.3  # below = dying/dead
THRIVING_THRESHOLD = 0.7   # above = strong self-reference

# Module exports for receipt types
RECEIPT_SCHEMA = ["autocatalysis_event", "birth_receipt", "death_receipt"]


# =============================================================================
# CORE FUNCTION 1: autocatalysis_check
# =============================================================================

def autocatalysis_check(pattern: Dict) -> bool:
    """
    Core existence test: returns True IFF pattern contains receipts that
    reference the pattern itself.

    Args:
        pattern: Dict with 'receipts' list and 'pattern_id' field

    Returns:
        bool: True if pattern is autocatalytic (self-referencing)
    """
    pattern_id = pattern.get("pattern_id", "")
    receipts = pattern.get("receipts", [])

    # Check if any receipt references this pattern_id
    for receipt in receipts:
        # Check various fields where pattern_id might appear
        if receipt.get("pattern_id") == pattern_id:
            return True
        if pattern_id in receipt.get("parent_receipts", []):
            return True
        if pattern_id in str(receipt.get("payload_hash", "")):
            return True

    return False


# =============================================================================
# CORE FUNCTION 2: is_alive
# =============================================================================

def is_alive(pattern: Dict) -> bool:
    """Alias for autocatalysis_check() - semantic clarity for agent existence."""
    return autocatalysis_check(pattern)


# =============================================================================
# CORE FUNCTION 3: coherence_score
# =============================================================================

def coherence_score(pattern: Dict) -> float:
    """
    Returns 0.0-1.0 measuring self-reference density using entropy.

    Uses entropy.system_entropy() for calculation basis:
    - Below 0.3 = dying (losing coherence)
    - Above 0.7 = thriving (strong self-reference)

    Args:
        pattern: Dict with 'receipts' list

    Returns:
        float: Coherence score in [0.0, 1.0]
    """
    receipts = pattern.get("receipts", [])

    if not receipts:
        return 0.0

    # Calculate base entropy of pattern
    entropy = system_entropy(receipts)

    # Count self-references
    pattern_id = pattern.get("pattern_id", "")
    self_ref_count = 0

    for receipt in receipts:
        if receipt.get("pattern_id") == pattern_id:
            self_ref_count += 1

    # Coherence = self-reference density normalized by entropy
    # High self-reference + low entropy = high coherence
    if len(receipts) == 0:
        return 0.0

    self_ref_ratio = self_ref_count / len(receipts)

    # Entropy normalization: max entropy for N items is log2(N)
    # Lower entropy (more order) = higher coherence
    max_entropy = len(receipts).bit_length() if len(receipts) > 1 else 1.0
    entropy_factor = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    # Coherence score combines self-reference density with entropy factor
    coherence = (self_ref_ratio + entropy_factor) / 2.0

    return min(1.0, max(0.0, coherence))


# =============================================================================
# CORE FUNCTION 4: detect_birth
# =============================================================================

def detect_birth(pattern: Dict, previous_state: Dict) -> Optional[Dict]:
    """
    Detects pattern birth: transition FROM non-autocatalytic TO autocatalytic.

    Args:
        pattern: Current pattern state
        previous_state: Previous pattern state

    Returns:
        birth_receipt if birth detected, None otherwise
    """
    current_coherence = coherence_score(pattern)
    previous_coherence = coherence_score(previous_state)

    # Birth = crossing threshold upward
    if previous_coherence < COHERENCE_THRESHOLD and current_coherence >= COHERENCE_THRESHOLD:
        pattern_id = pattern.get("pattern_id", "unknown")
        receipts = pattern.get("receipts", [])

        # Collect parent receipt hashes
        parent_receipts = [r.get("payload_hash", "") for r in receipts[:3]]  # First 3 as seeds

        return emit_receipt("birth", {
            "tenant_id": pattern.get("tenant_id", "default"),
            "pattern_id": pattern_id,
            "coherence_at_birth": current_coherence,
            "parent_receipts": parent_receipts
        })

    return None


# =============================================================================
# CORE FUNCTION 5: detect_death
# =============================================================================

def detect_death(pattern: Dict, previous_state: Dict) -> Optional[Dict]:
    """
    Detects pattern death: transition FROM autocatalytic TO non-autocatalytic.

    Args:
        pattern: Current pattern state
        previous_state: Previous pattern state

    Returns:
        death_receipt if death detected, None otherwise
    """
    current_coherence = coherence_score(pattern)
    previous_coherence = coherence_score(previous_state)

    # Death = crossing threshold downward
    if previous_coherence >= COHERENCE_THRESHOLD and current_coherence < COHERENCE_THRESHOLD:
        pattern_id = pattern.get("pattern_id", "unknown")

        # Calculate lifespan if available
        lifespan_cycles = previous_state.get("lifespan_cycles", 0) + 1

        return emit_receipt("death", {
            "tenant_id": pattern.get("tenant_id", "default"),
            "pattern_id": pattern_id,
            "coherence_at_death": current_coherence,
            "lifespan_cycles": lifespan_cycles
        })

    return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "COHERENCE_THRESHOLD",
    "THRIVING_THRESHOLD",
    # Core functions
    "autocatalysis_check",
    "is_alive",
    "coherence_score",
    "detect_birth",
    "detect_death",
]
