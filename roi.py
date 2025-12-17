"""
roi.py - ROI Gate Assertions

ROI gate assertion with stoprule for decision gates.
"check if ROI gate holds above 1.2"

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.

Source: Grok insight 2025-01
"""

from typing import Optional

from receipts import dual_hash, emit_receipt, StopRule


# =============================================================================
# CONSTANTS
# =============================================================================

# Default gate thresholds
DEFAULT_ROI_THRESHOLD = 1.2       # ROI must exceed this
DEFAULT_CONFIDENCE = 0.90          # Required confidence level

# Module exports for receipt types
RECEIPT_SCHEMA = ["roi_gate"]


# =============================================================================
# RECEIPT TYPE 1: roi_gate
# =============================================================================

# --- SCHEMA ---
ROI_GATE_SCHEMA = {
    "receipt_type": "roi_gate",
    "ts": "ISO8601",
    "tenant_id": "str",
    "roi_score": "float",
    "threshold": "float",
    "passed": "bool",
    "action": "str (proceed|halt)",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_roi_gate_receipt(
    tenant_id: str,
    roi_score: float,
    threshold: float,
    passed: bool,
    action: str
) -> dict:
    """Emit roi_gate receipt for gate decision."""
    return emit_receipt("roi_gate", {
        "tenant_id": tenant_id,
        "roi_score": roi_score,
        "threshold": threshold,
        "passed": passed,
        "action": action
    })


# --- STOPRULE ---
def stoprule_roi_gate_failed(
    roi_score: float,
    threshold: float,
    tenant_id: str = "axiom-autonomy"
) -> None:
    """
    Stoprule for failed ROI gate.

    Emits anomaly receipt and raises StopRule when ROI < threshold.

    Args:
        roi_score: Computed ROI score
        threshold: Required threshold
        tenant_id: Tenant identifier

    Raises:
        StopRule: Always raises when called
    """
    emit_receipt("anomaly", {
        "tenant_id": tenant_id,
        "metric": "roi_gate",
        "baseline": threshold,
        "delta": roi_score - threshold,
        "classification": "roi_gate_failed",
        "action": "halt"
    })

    raise StopRule(
        f"ROI gate failed: {roi_score:.3f} < threshold {threshold:.3f}"
    )


# =============================================================================
# CORE FUNCTION 1: assert_roi_gate
# =============================================================================

def assert_roi_gate(
    roi_score: float,
    threshold: float = DEFAULT_ROI_THRESHOLD,
    tenant_id: str = "axiom-autonomy",
    halt_on_failure: bool = True
) -> bool:
    """
    Assert ROI gate passes.

    Emits roi_gate receipt and optionally raises StopRule on failure.

    Args:
        roi_score: Computed ROI score
        threshold: Required threshold (default 1.2)
        tenant_id: Tenant identifier for receipt
        halt_on_failure: If True, raise StopRule on failure

    Returns:
        bool: True if gate passed

    Raises:
        StopRule: If gate fails and halt_on_failure is True
    """
    passed = roi_score >= threshold
    action = "proceed" if passed else "halt"

    # Emit gate receipt
    emit_roi_gate_receipt(
        tenant_id=tenant_id,
        roi_score=roi_score,
        threshold=threshold,
        passed=passed,
        action=action
    )

    if not passed and halt_on_failure:
        stoprule_roi_gate_failed(roi_score, threshold, tenant_id)

    return passed


# =============================================================================
# CORE FUNCTION 2: check_roi_gate
# =============================================================================

def check_roi_gate(
    roi_score: float,
    threshold: float = DEFAULT_ROI_THRESHOLD
) -> bool:
    """
    Check ROI gate without emitting receipt or halting.

    Simple boolean check for testing and conditionals.

    Args:
        roi_score: Computed ROI score
        threshold: Required threshold (default 1.2)

    Returns:
        bool: True if roi_score >= threshold
    """
    return roi_score >= threshold


# =============================================================================
# CORE FUNCTION 3: compute_roi_margin
# =============================================================================

def compute_roi_margin(
    roi_score: float,
    threshold: float = DEFAULT_ROI_THRESHOLD
) -> float:
    """
    Compute margin above/below ROI threshold.

    Positive margin = above threshold (safe)
    Negative margin = below threshold (at risk)

    Args:
        roi_score: Computed ROI score
        threshold: Required threshold (default 1.2)

    Returns:
        float: Margin (roi_score - threshold)
    """
    return roi_score - threshold


# =============================================================================
# CORE FUNCTION 4: require_roi_gate
# =============================================================================

def require_roi_gate(
    roi_score: float,
    threshold: float = DEFAULT_ROI_THRESHOLD,
    tenant_id: str = "axiom-autonomy"
) -> None:
    """
    Require ROI gate to pass. Always halts on failure.

    Convenience function that always raises on failure.

    Args:
        roi_score: Computed ROI score
        threshold: Required threshold (default 1.2)
        tenant_id: Tenant identifier for receipt

    Raises:
        StopRule: If gate fails
    """
    assert_roi_gate(
        roi_score=roi_score,
        threshold=threshold,
        tenant_id=tenant_id,
        halt_on_failure=True
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "DEFAULT_ROI_THRESHOLD",
    "DEFAULT_CONFIDENCE",
    # Core functions
    "assert_roi_gate",
    "check_roi_gate",
    "compute_roi_margin",
    "require_roi_gate",
    # Receipt functions
    "emit_roi_gate_receipt",
    # Stoprules
    "stoprule_roi_gate_failed",
]
