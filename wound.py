"""
wound.py - The Wound Tracking Module (v11)

Tracks manual interventions as automation gaps. Seeds for agent genesis in v12.
"What bleeds, breeds." - wounds become agents.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import json
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

# =============================================================================
# CLAUDEME §8 CORE FUNCTIONS - Import from receipts.py (v6 foundation)
# =============================================================================

from receipts import dual_hash, emit_receipt, StopRule


# =============================================================================
# CONSTANTS
# =============================================================================

# Automation gap thresholds from v6 strategy
AUTOMATION_GAP_OCCURRENCE_THRESHOLD = 5
AUTOMATION_GAP_RESOLVE_THRESHOLD_MS = 1800000  # 30 minutes

# Valid wound types
WOUND_TYPES = ['operational', 'safety', 'performance', 'integration']

# Module exports for receipt types
RECEIPT_SCHEMA = ["wound_receipt", "automation_gap", "healing_record"]


# =============================================================================
# CORE FUNCTION 1: classify_wound
# =============================================================================

def classify_wound(intervention: dict) -> str:
    """
    Categorizes wound by type based on intervention action and context.

    Args:
        intervention: Dict with 'action' and 'context' fields

    Returns:
        str: One of 'operational', 'safety', 'performance', 'integration'

    Categories:
        - operational: routine maintenance, restarts, config changes
        - safety: prevented harm, blocked dangerous action, escalation
        - performance: optimization, tuning, resource adjustment
        - integration: API fixes, data format corrections, compatibility patches
    """
    action = intervention.get("action", "").lower()
    context = intervention.get("context", "").lower()
    combined = action + " " + context

    # Safety keywords - highest priority
    safety_keywords = [
        "block", "prevent", "halt", "stop", "security", "escalate",
        "danger", "harm", "unsafe", "risk", "violation", "threat",
        "breach", "attack", "malicious", "unauthorized"
    ]
    if any(kw in combined for kw in safety_keywords):
        return "safety"

    # Integration keywords
    integration_keywords = [
        "api", "format", "compatibility", "patch", "schema", "protocol",
        "interface", "endpoint", "data format", "conversion", "migration",
        "sync", "connector", "adapter", "mapping", "transform"
    ]
    if any(kw in combined for kw in integration_keywords):
        return "integration"

    # Performance keywords
    performance_keywords = [
        "optimize", "tune", "resource", "memory", "cpu", "latency",
        "throughput", "cache", "scale", "performance", "slow", "speed",
        "bottleneck", "capacity", "quota", "limit"
    ]
    if any(kw in combined for kw in performance_keywords):
        return "performance"

    # Default to operational (routine maintenance, restarts, config)
    return "operational"


# =============================================================================
# CORE FUNCTION 2: could_automate
# =============================================================================

def could_automate(intervention: dict) -> float:
    """
    Returns confidence 0.0-1.0 that this wound pattern could become an agent.

    Args:
        intervention: Dict with 'duration_ms', 'action', 'context' fields

    Returns:
        float: Confidence score 0.0-1.0

    Factors increasing automation confidence:
        - High time_to_resolve_ms (>30 min = human time is expensive)
        - Clear resolution_action (deterministic, not judgment-based)
        - Low risk classification (operational/performance, not safety)

    Factors decreasing automation confidence:
        - Safety-related (requires human judgment)
        - Novel problem (first occurrence, unknown pattern)
        - Complex multi-step resolution
        - High blast radius if wrong
    """
    score = 0.5  # Start neutral

    duration_ms = intervention.get("duration_ms", 0)
    action = intervention.get("action", "").lower()
    context = intervention.get("context", "").lower()

    # Factor 1: Duration - longer = more valuable to automate
    # >30 min (1.8M ms) adds +0.2, <5 min (300K ms) subtracts -0.1
    if duration_ms >= AUTOMATION_GAP_RESOLVE_THRESHOLD_MS:
        score += 0.2
    elif duration_ms >= 900000:  # 15 minutes
        score += 0.1
    elif duration_ms < 300000:  # Less than 5 minutes
        score -= 0.1

    # Factor 2: Wound type classification
    wound_type = classify_wound(intervention)
    if wound_type == "safety":
        score -= 0.3  # Safety requires human judgment
    elif wound_type == "operational":
        score += 0.15  # Routine tasks are automatable
    elif wound_type == "performance":
        score += 0.1  # Performance tuning often automatable

    # Factor 3: Action complexity
    # Simple, deterministic actions are more automatable
    simple_actions = ["restart", "reboot", "clear", "reset", "refresh", "retry"]
    complex_actions = ["investigate", "analyze", "debug", "decide", "judge", "evaluate"]

    if any(sa in action for sa in simple_actions):
        score += 0.1
    if any(ca in action for ca in complex_actions):
        score -= 0.15

    # Factor 4: Context indicators
    # Novel/unknown problems are harder to automate
    novel_indicators = ["first time", "never seen", "unknown", "new issue", "novel"]
    routine_indicators = ["again", "recurring", "repeated", "same as", "usual"]

    if any(ni in context for ni in novel_indicators):
        score -= 0.2
    if any(ri in context for ri in routine_indicators):
        score += 0.15

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))


# =============================================================================
# RECEIPT TYPE 1: wound_receipt
# =============================================================================

# --- SCHEMA ---
WOUND_RECEIPT_SCHEMA = {
    "receipt_type": "wound_receipt",
    "ts": "ISO8601",
    "tenant_id": "str",  # REQUIRED per CLAUDEME 4.1:137
    "intervention_id": "str",
    "problem_type": "str",  # operational|safety|performance|integration
    "time_to_resolve_ms": "int",
    "resolution_action": "str",
    "could_automate": "float",  # 0.0-1.0
    "operator_id": "str",
    "context_hash": "str",
    "payload_hash": "str"  # dual_hash of receipt data
}


# --- EMIT ---
def emit_wound(intervention: dict) -> dict:
    """
    Creates wound_receipt from manual intervention data.

    Args:
        intervention: Dict with required fields:
            - duration_ms: how long the human spent fixing this
            - action: what the human did to resolve
            - context: what triggered the intervention
            - operator_id: who performed the intervention (for audit)

    Returns:
        dict: wound_receipt with all required fields
    """
    # Extract required fields
    duration_ms = intervention.get("duration_ms", 0)
    action = intervention.get("action", "")
    context = intervention.get("context", "")
    operator_id = intervention.get("operator_id", "unknown")
    tenant_id = intervention.get("tenant_id", "default")
    intervention_id = intervention.get("intervention_id", str(uuid.uuid4()))

    # Compute derived fields
    problem_type = classify_wound(intervention)
    automation_confidence = could_automate(intervention)
    context_hash = dual_hash(context)

    return emit_receipt("wound_receipt", {
        "tenant_id": tenant_id,
        "intervention_id": intervention_id,
        "problem_type": problem_type,
        "time_to_resolve_ms": duration_ms,
        "resolution_action": action,
        "could_automate": automation_confidence,
        "operator_id": operator_id,
        "context_hash": context_hash
    })


# --- TEST ---
def test_wound_receipt():
    """Test wound_receipt emission."""
    intervention = {
        "duration_ms": 600000,
        "action": "restarted the service",
        "context": "service was unresponsive",
        "operator_id": "operator_001",
        "tenant_id": "test_tenant"
    }
    r = emit_wound(intervention)
    assert r["receipt_type"] == "wound_receipt"
    assert r["tenant_id"] == "test_tenant"
    assert r["problem_type"] in WOUND_TYPES
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]  # dual_hash format


# --- STOPRULE ---
def stoprule_wound_capture_failure(intervention: dict) -> None:
    """
    Stoprule for when wound capture fails.
    Emits healing_record documenting the failure, then raises StopRule.
    """
    tenant_id = intervention.get("tenant_id", "default")
    wound_id = intervention.get("intervention_id", str(uuid.uuid4()))

    emit_receipt("healing_record", {
        "tenant_id": tenant_id,
        "wound_id": wound_id,
        "healed": False,
        "healing_method": "manual",
        "time_to_heal_ms": 0
    })

    raise StopRule(f"Wound capture failed for intervention: {wound_id}")


# =============================================================================
# RECEIPT TYPE 2: automation_gap
# =============================================================================

# --- SCHEMA ---
AUTOMATION_GAP_SCHEMA = {
    "receipt_type": "automation_gap",
    "ts": "ISO8601",
    "tenant_id": "str",
    "wound_type": "str",
    "occurrence_count": "int",
    "median_resolve_ms": "int",
    "total_human_time_ms": "int",
    "could_automate_avg": "float",
    "is_gap": "bool",
    "recommendation": "str",  # propose_blueprint|monitor|ignore
    "payload_hash": "str"
}


# --- EMIT ---
def is_automation_gap(wounds: List[dict], tenant_id: str = "default") -> dict:
    """
    Determines if a set of wound receipts constitutes an automation gap
    worth proposing to ARCHITECT.

    ARCHITECT threshold (from v6 strategy):
        - occurrences > 5
        - median resolve time > 30 minutes (1,800,000 ms)

    Args:
        wounds: List of wound_receipt dicts
        tenant_id: Tenant identifier

    Returns:
        dict: automation_gap receipt with:
            - is_gap (bool): True if thresholds met
            - occurrence_count: how many times this wound type appeared
            - median_resolve_ms: median time to resolve
            - recommendation: "propose_blueprint" | "monitor" | "ignore"
    """
    if not wounds:
        return emit_receipt("automation_gap", {
            "tenant_id": tenant_id,
            "wound_type": "unknown",
            "occurrence_count": 0,
            "median_resolve_ms": 0,
            "total_human_time_ms": 0,
            "could_automate_avg": 0.0,
            "is_gap": False,
            "recommendation": "ignore"
        })

    # Extract metrics from wound receipts
    occurrence_count = len(wounds)
    resolve_times = [w.get("time_to_resolve_ms", 0) for w in wounds]
    median_resolve_ms = int(statistics.median(resolve_times))
    total_human_time_ms = sum(resolve_times)

    # Average automation confidence
    automation_scores = [w.get("could_automate", 0.5) for w in wounds]
    could_automate_avg = sum(automation_scores) / len(automation_scores)

    # Determine most common wound type
    wound_types = [w.get("problem_type", "operational") for w in wounds]
    wound_type = max(set(wound_types), key=wound_types.count)

    # Check thresholds
    meets_occurrence = occurrence_count > AUTOMATION_GAP_OCCURRENCE_THRESHOLD
    meets_time = median_resolve_ms > AUTOMATION_GAP_RESOLVE_THRESHOLD_MS
    is_gap = meets_occurrence and meets_time

    # Determine recommendation
    if is_gap and could_automate_avg >= 0.5:
        recommendation = "propose_blueprint"
    elif is_gap or (meets_occurrence and could_automate_avg >= 0.6):
        recommendation = "monitor"
    else:
        recommendation = "ignore"

    return emit_receipt("automation_gap", {
        "tenant_id": tenant_id,
        "wound_type": wound_type,
        "occurrence_count": occurrence_count,
        "median_resolve_ms": median_resolve_ms,
        "total_human_time_ms": total_human_time_ms,
        "could_automate_avg": could_automate_avg,
        "is_gap": is_gap,
        "recommendation": recommendation
    })


# --- TEST ---
def test_automation_gap():
    """Test automation_gap receipt emission."""
    wounds = [
        {"time_to_resolve_ms": 2000000, "could_automate": 0.7, "problem_type": "operational"}
        for _ in range(6)
    ]
    r = is_automation_gap(wounds, "test_tenant")
    assert r["receipt_type"] == "automation_gap"
    assert r["tenant_id"] == "test_tenant"
    assert r["is_gap"] is True
    assert r["recommendation"] == "propose_blueprint"


# --- STOPRULE ---
def stoprule_automation_gap(tenant_id: str, is_gap: bool, recommendation: str) -> None:
    """
    Automation gap stoprule - triggers alert when gap identified.
    Does NOT raise StopRule - just emits anomaly for monitoring.
    """
    if is_gap:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "automation_gap",
            "baseline": 0.0,
            "delta": 1.0,
            "classification": "deviation",
            "action": "alert",
            "recommendation": recommendation
        })


# =============================================================================
# RECEIPT TYPE 3: healing_record
# =============================================================================

# --- SCHEMA ---
HEALING_RECORD_SCHEMA = {
    "receipt_type": "healing_record",
    "ts": "ISO8601",
    "tenant_id": "str",
    "wound_id": "str",
    "healed": "bool",
    "healing_method": "str",  # manual|automated|hybrid
    "time_to_heal_ms": "int",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_healing_record(tenant_id: str, wound_id: str, healed: bool,
                        healing_method: str, time_to_heal_ms: int) -> dict:
    """
    Emit healing_record receipt to track wound resolution.

    Args:
        tenant_id: Tenant identifier
        wound_id: ID of the wound being healed
        healed: Whether the wound was successfully healed
        healing_method: One of 'manual', 'automated', 'hybrid'
        time_to_heal_ms: Time taken to heal in milliseconds

    Returns:
        dict: healing_record receipt
    """
    return emit_receipt("healing_record", {
        "tenant_id": tenant_id,
        "wound_id": wound_id,
        "healed": healed,
        "healing_method": healing_method,
        "time_to_heal_ms": time_to_heal_ms
    })


# --- TEST ---
def test_healing_record():
    """Test healing_record receipt emission."""
    r = emit_healing_record(
        tenant_id="test_tenant",
        wound_id="wound_001",
        healed=True,
        healing_method="automated",
        time_to_heal_ms=5000
    )
    assert r["receipt_type"] == "healing_record"
    assert r["tenant_id"] == "test_tenant"
    assert r["healed"] is True
    assert r["healing_method"] == "automated"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "WOUND_TYPES",
    "AUTOMATION_GAP_OCCURRENCE_THRESHOLD",
    "AUTOMATION_GAP_RESOLVE_THRESHOLD_MS",
    # Core functions
    "classify_wound",
    "could_automate",
    "is_automation_gap",
    # Emit functions
    "emit_wound",
    "emit_healing_record",
    # Stoprules
    "stoprule_wound_capture_failure",
    "stoprule_automation_gap",
    # Core utilities
    "emit_receipt",
    "dual_hash",
    "StopRule",
]
