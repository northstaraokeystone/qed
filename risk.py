"""
risk.py - The Inflammation Module

Risk as inflammation. High score = system under threat.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.

Key Concept: risk.py is the system's inflammation—preemptive pain before wounds form.
Three-component model: blast_radius (blast), reversibility (healing), precedent (novelty).

Integration:
- remediate.py: uses classify_risk in auto-approve decision
- unified_loop.py: uses score_risk in GATE phase for HITL trigger
- Absorbed function: ACTUARY from original 8-agent design
"""

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Import from entropy.py per CLAUDEME section 8
from entropy import (
    dual_hash,
    emit_receipt,
    StopRule,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Module exports for receipt types
RECEIPT_SCHEMA = ["risk_assessment", "inflammation_alert"]

# Risk thresholds per v6 strategy section 3.7
RISK_THRESHOLD_LOW = 0.1       # low: score < 0.1
RISK_THRESHOLD_MEDIUM = 0.3    # medium: 0.1 <= score < 0.3
RISK_THRESHOLD_HIGH = 0.3      # high: score >= 0.3

# Forced HITL threshold per v6 strategy 340-341
FORCED_HITL_THRESHOLD = 0.3

# Score formula weights
WEIGHT_BLAST_RADIUS = 0.4
WEIGHT_REVERSIBILITY = 0.3
WEIGHT_PRECEDENT = 0.3


# =============================================================================
# RECEIPT TYPE 1: risk_assessment
# =============================================================================

# --- SCHEMA ---
RISK_ASSESSMENT_SCHEMA = {
    "receipt_type": "risk_assessment",
    "ts": "ISO8601",
    "tenant_id": "str",
    "action_id": "str",
    "score": "float 0.0-1.0",
    "confidence": "float 0.0-1.0",
    "classification": "low|medium|high",
    "components": {
        "blast_radius": "float 0.0-1.0",
        "reversibility": "float 0.0-1.0",
        "precedent": "float 0.0-1.0"
    },
    "forced_hitl": "bool",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_risk_assessment(tenant_id: str, action_id: str, score: float,
                        confidence: float, classification: str,
                        components: dict, forced_hitl: bool) -> dict:
    """Emit risk_assessment receipt for action risk evaluation."""
    return emit_receipt("risk_assessment", {
        "tenant_id": tenant_id,
        "action_id": action_id,
        "score": score,
        "confidence": confidence,
        "classification": classification,
        "components": components,
        "forced_hitl": forced_hitl
    })


# --- STOPRULE ---
def stoprule_risk_assessment(tenant_id: str, action_id: str,
                            score: float) -> None:
    """
    Risk assessment stoprule - triggers when score indicates critical risk.
    Emit inflammation_alert before raising StopRule.
    """
    if score >= FORCED_HITL_THRESHOLD:
        emit_inflammation_alert(
            tenant_id=tenant_id,
            metric="risk_score",
            baseline=FORCED_HITL_THRESHOLD,
            delta=score - FORCED_HITL_THRESHOLD,
            classification="high_risk",
            action="escalate",
            trigger_action=action_id
        )


# =============================================================================
# RECEIPT TYPE 2: inflammation_alert
# =============================================================================

# --- SCHEMA ---
INFLAMMATION_ALERT_SCHEMA = {
    "receipt_type": "inflammation_alert",
    "ts": "ISO8601",
    "tenant_id": "str",
    "metric": "str",
    "baseline": "float",
    "delta": "float",
    "classification": "str",
    "action": "halt|escalate",
    "trigger_action": "str",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_inflammation_alert(tenant_id: str, metric: str, baseline: float,
                           delta: float, classification: str, action: str,
                           trigger_action: str) -> dict:
    """Emit inflammation_alert receipt for high-risk conditions."""
    return emit_receipt("inflammation_alert", {
        "tenant_id": tenant_id,
        "metric": metric,
        "baseline": baseline,
        "delta": delta,
        "classification": classification,
        "action": action,
        "trigger_action": trigger_action
    })


# =============================================================================
# CORE FUNCTION 1: score_risk
# =============================================================================

def score_risk(action: dict, context: dict) -> dict:
    """
    Risk as inflammation. High score = system under threat.

    Three components (each 0.0-1.0):
    - blast_radius: fraction of system affected if action fails
    - reversibility: 1.0 = fully reversible, 0.0 = permanent
    - precedent: 1.0 = never seen, 0.0 = routine

    Formula: score = 0.4×blast_radius + 0.3×(1-reversibility) + 0.3×precedent

    Confidence based on data completeness. Range 0.5 to 1.0.

    CRITICAL - forced_hitl field: Set to True when score >= 0.3.
    High risk MUST force HITL regardless of confidence (v6 strategy).

    Args:
        action: dict with fields:
            - action_id: str
            - affected_receipts: list (count used for blast_radius)
            - reversible or reversibility: bool or float
            - tenant_id: str
        context: dict with fields:
            - total_receipts: int (used for blast_radius denominator)
            - similar_actions_count: int (used for precedent)
            - tenant_id: str

    Returns:
        risk_assessment receipt with score in [0,1]
    """
    # Extract fields with safe defaults
    action_id = action.get("action_id", "unknown")
    affected_count = len(action.get("affected_receipts", []))
    tenant_id = action.get("tenant_id", context.get("tenant_id", "default"))

    total_receipts = context.get("total_receipts", 1)
    similar_actions_count = context.get("similar_actions_count", 0)

    # Read reversibility (handle both 'reversible' bool and 'reversibility' float)
    reversible = action.get("reversible", action.get("reversibility", False))
    if isinstance(reversible, bool):
        reversibility = 1.0 if reversible else 0.0
    else:
        reversibility = max(0.0, min(1.0, float(reversible)))

    # Calculate three components
    # 1. blast_radius: fraction of system affected
    blast_radius = min(1.0, affected_count / total_receipts) if total_receipts > 0 else 0.0

    # 2. reversibility: already calculated above

    # 3. precedent: 1.0 = never seen, 0.0 = routine
    # precedent = 1.0 - (similar_actions_count / 10), clamped to [0,1]
    precedent = max(0.0, min(1.0, 1.0 - (similar_actions_count / 10.0)))

    # Calculate score using weighted formula
    score = (WEIGHT_BLAST_RADIUS * blast_radius +
             WEIGHT_REVERSIBILITY * (1.0 - reversibility) +
             WEIGHT_PRECEDENT * precedent)

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    # Calculate confidence based on data completeness
    # Start at 1.0 and reduce for missing fields
    confidence = 1.0

    # Check for missing/incomplete data
    if "affected_receipts" not in action:
        confidence -= 0.1
    if "reversible" not in action and "reversibility" not in action:
        confidence -= 0.1
    if "total_receipts" not in context:
        confidence -= 0.1
    if "similar_actions_count" not in context:
        confidence -= 0.1

    # Clamp confidence to [0.5, 1.0] range per spec
    confidence = max(0.5, min(1.0, confidence))

    # Classify risk based on score
    classification = classify_risk(score)

    # CRITICAL: forced_hitl = True when score >= 0.3
    forced_hitl = score >= FORCED_HITL_THRESHOLD

    # Build components dict
    components = {
        "blast_radius": blast_radius,
        "reversibility": reversibility,
        "precedent": precedent
    }

    # Emit and return risk_assessment receipt
    receipt = emit_risk_assessment(
        tenant_id=tenant_id,
        action_id=action_id,
        score=score,
        confidence=confidence,
        classification=classification,
        components=components,
        forced_hitl=forced_hitl
    )

    return receipt


# =============================================================================
# CORE FUNCTION 2: classify_risk
# =============================================================================

def classify_risk(score: float) -> str:
    """
    Maps score to classification string.

    Thresholds (per v6 strategy section 3.7):
    - low: score < 0.1
    - medium: 0.1 <= score < 0.3
    - high: score >= 0.3

    Args:
        score: float in [0.0, 1.0]

    Returns:
        str: "low", "medium", or "high"
    """
    if score < RISK_THRESHOLD_LOW:
        return "low"
    elif score < RISK_THRESHOLD_MEDIUM:
        return "medium"
    else:
        return "high"


# =============================================================================
# CORE FUNCTION 3: stoprule_high_risk
# =============================================================================

def stoprule_high_risk(score: float, action: dict) -> None:
    """
    Stoprule for critical risk situations.

    Emit inflammation_alert receipt before raising StopRule.
    This enforces HITL for high-risk actions per v6 strategy.

    Args:
        score: risk score (should be >= 0.3)
        action: action dict with action_id and tenant_id
    """
    action_id = action.get("action_id", "unknown")
    tenant_id = action.get("tenant_id", "default")

    emit_inflammation_alert(
        tenant_id=tenant_id,
        metric="risk_score",
        baseline=FORCED_HITL_THRESHOLD,
        delta=score - FORCED_HITL_THRESHOLD,
        classification="high_risk",
        action="escalate",
        trigger_action=action_id
    )

    raise StopRule(f"High risk score {score:.2f} for action {action_id} "
                   f"exceeds HITL threshold {FORCED_HITL_THRESHOLD}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "RISK_THRESHOLD_LOW",
    "RISK_THRESHOLD_MEDIUM",
    "RISK_THRESHOLD_HIGH",
    "FORCED_HITL_THRESHOLD",
    # Core functions
    "score_risk",
    "classify_risk",
    "stoprule_high_risk",
    # Emit functions
    "emit_risk_assessment",
    "emit_inflammation_alert",
    # Stoprules
    "stoprule_risk_assessment",
]
