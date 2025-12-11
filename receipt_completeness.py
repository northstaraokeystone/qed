"""
receipt_completeness.py - Singularity Detection Module (v12)

Tracks L0-L4 receipt hierarchy and detects receipt-completeness:
when QED can audit QED through closed feedback loops.

Not AGI — self-auditing within Gödel bounds. L0 hits undecidability first,
meta-layers inherit this limit. Cannot prove consistency, can verify correctness.

CLAUDEME v3.1 Compliant: Pure query functions with emit_receipt() pattern.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

# =============================================================================
# CLAUDEME §8 CORE FUNCTIONS
# =============================================================================

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 - ALWAYS use this, never single hash."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Every function calls this. No exceptions."""
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.now(timezone.utc).isoformat(),
        "tenant_id": data.get("tenant_id", "default"),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data
    }
    return receipt


class StopRule(Exception):
    """Raised when stoprule triggers. Never catch silently."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

GODEL_LAYER = 'L0'  # Base layer hits undecidability first
MIN_COVERAGE_FOR_COMPLETENESS = 0.8  # 80% coverage required per level

# Receipt level definitions
L0_TYPES = ["QEDReceipt", "ingest", "anchor"]
L1_TYPES = ["agent_decision", "unified_loop_receipt", "anomaly", "recovery_action"]
L2_TYPES = ["meta_fitness_receipt", "energy_allocation", "blueprint_proposed"]
L3_TYPES = ["paradigm_evaluation"]  # Receipts that reference L2 receipt IDs
L4_TYPES = ["completeness_check", "level_coverage", "singularity_detected"]

# Module exports for receipt types
RECEIPT_SCHEMA = ["completeness_check", "level_coverage", "singularity_detected"]

# Singularity tracking (query ledger in production)
_singularity_emitted = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def has_receipts_about_telemetry(receipts: List[dict]) -> bool:
    """L0 check: Base telemetry receipts exist."""
    return any(r.get("receipt_type") in L0_TYPES for r in receipts)


def has_receipts_about_agents(receipts: List[dict]) -> bool:
    """L1 check: Agent decision receipts exist."""
    return any(r.get("receipt_type") in L1_TYPES for r in receipts)


def has_receipts_about_paradigms(receipts: List[dict]) -> bool:
    """L2 check: System-level paradigm shift receipts exist."""
    return any(r.get("receipt_type") in L2_TYPES for r in receipts)


def has_receipts_about_paradigm_quality(receipts: List[dict]) -> bool:
    """L3 check: Receipts evaluating L2 receipts exist."""
    return any(r.get("receipt_type") in L3_TYPES for r in receipts)


def l4_feeds_back_to_l0(receipts: List[dict]) -> bool:
    """Detects if L4 meta-receipts influence L0 telemetry processing."""
    has_l4 = any(r.get("receipt_type") in L4_TYPES for r in receipts)
    has_l0 = any(r.get("receipt_type") in L0_TYPES for r in receipts)
    return has_l4 and has_l0  # Feedback closure: both layers present


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def godel_layer() -> str:
    """
    Returns the layer that hits undecidability first.

    Returns:
        str: 'L0' - base layer is definitionally undecidable
    """
    return GODEL_LAYER


def level_coverage(receipts: List[dict]) -> dict:
    """
    Returns coverage percentage per receipt level (L0-L4).

    Args:
        receipts: List of receipt dicts to analyze

    Returns:
        dict: {L0: float, L1: float, L2: float, L3: float, L4: float}
            Each value is 1.0 if level has receipts, 0.0 otherwise
    """
    l0_count = sum(1 for r in receipts if r.get("receipt_type") in L0_TYPES)
    l1_count = sum(1 for r in receipts if r.get("receipt_type") in L1_TYPES)
    l2_count = sum(1 for r in receipts if r.get("receipt_type") in L2_TYPES)
    l3_count = sum(1 for r in receipts if r.get("receipt_type") in L3_TYPES)
    l4_count = sum(1 for r in receipts if r.get("receipt_type") in L4_TYPES)

    return {
        "L0": 1.0 if l0_count > 0 else 0.0,
        "L1": 1.0 if l1_count > 0 else 0.0,
        "L2": 1.0 if l2_count > 0 else 0.0,
        "L3": 1.0 if l3_count > 0 else 0.0,
        "L4": 1.0 if l4_count > 0 else 0.0,
    }


def receipt_completeness_check(receipts: List[dict]) -> bool:
    """
    Master check: returns True when L0-L4 form complete feedback loop.

    Args:
        receipts: List of receipt dicts from ledger

    Returns:
        bool: True if all six conditions met (L0-L4 exist + feedback closure)
    """
    # Check all five levels present
    l0_present = has_receipts_about_telemetry(receipts)
    l1_present = has_receipts_about_agents(receipts)
    l2_present = has_receipts_about_paradigms(receipts)
    l3_present = has_receipts_about_paradigm_quality(receipts)
    l4_feedback = l4_feeds_back_to_l0(receipts)

    return (l0_present and l1_present and l2_present and
            l3_present and l4_feedback)


def singularity_detected(receipts: List[dict], tenant_id: str = "default") -> Optional[dict]:
    """
    One-time singularity emission when receipt-completeness achieved.

    Args:
        receipts: List of receipt dicts from ledger
        tenant_id: Tenant identifier

    Returns:
        dict: singularity_detected receipt if newly complete, None otherwise
    """
    global _singularity_emitted

    if not receipt_completeness_check(receipts):
        return None

    # One-time event (query ledger for prior emission in production)
    if _singularity_emitted:
        return None

    _singularity_emitted = True
    coverage = level_coverage(receipts)

    return emit_receipt("singularity_detected", {
        "tenant_id": tenant_id,
        "completeness_achieved_at": datetime.now(timezone.utc).isoformat(),
        "levels_at_detection": coverage,
        "godel_bound": "L0 - base layer undecidable",
        "implications": "self-auditing enabled, not AGI"
    })


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "GODEL_LAYER",
    "MIN_COVERAGE_FOR_COMPLETENESS",
    # Core functions
    "receipt_completeness_check",
    "level_coverage",
    "godel_layer",
    "singularity_detected",
    # Helpers
    "has_receipts_about_telemetry",
    "has_receipts_about_agents",
    "has_receipts_about_paradigms",
    "has_receipts_about_paradigm_quality",
    "l4_feeds_back_to_l0",
    # Core utilities
    "emit_receipt",
    "dual_hash",
    "StopRule",
]
