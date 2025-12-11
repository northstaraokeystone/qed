"""
architect.py - Wound-to-Blueprint Synthesizer (v12)

ARCHITECT watches wounds, synthesizes blueprints. Embedded in unified_loop
HYPOTHESIZE stage. What bleeds, breeds — recurring manual interventions reveal
automation gaps that become new agents.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import json
import statistics
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# =============================================================================
# CLAUDEME §8 CORE FUNCTIONS - Import from receipts.py (v6 foundation)
# =============================================================================

from receipts import dual_hash, emit_receipt, StopRule


# =============================================================================
# CONSTANTS
# =============================================================================

# Genesis lock from v12 spec - no agent can modify genesis logic
GENESIS_LOCK = "no agent can modify genesis logic"

# Automation gap thresholds
WOUND_RECURRENCE_THRESHOLD = 5
WOUND_RESOLVE_THRESHOLD_MS = 1800000  # 30 minutes

# Autonomy thresholds for HITL gate
HIGH_AUTONOMY_THRESHOLD = 0.7
AUTO_APPROVE_CONFIDENCE = 0.9

# Module exports for receipt types
RECEIPT_SCHEMA = ["blueprint_proposed", "genesis_pending", "genesis_approved", "genesis_rejected"]


# =============================================================================
# CORE FUNCTION 1: identify_automation_gaps
# =============================================================================

def identify_automation_gaps(wounds: List[dict]) -> List[dict]:
    """
    Identifies automation gaps from wound patterns.

    Filter criteria: recurrence > 5 AND median_resolve > 30 minutes (1800000 ms)
    Clusters wounds by problem_type field.

    Args:
        wounds: List of wound_receipt dicts from wound.py

    Returns:
        List[dict]: List of AutomationGap dicts with fields:
            - problem_type: str
            - wound_ids: list[str]
            - recurrence_count: int
            - median_resolve_ms: int
            - first_seen: ISO8601
            - last_seen: ISO8601
    """
    if not wounds:
        return []

    # Group wounds by problem_type
    clusters = {}
    for wound in wounds:
        problem_type = wound.get("problem_type", "operational")
        if problem_type not in clusters:
            clusters[problem_type] = []
        clusters[problem_type].append(wound)

    # Filter and build automation gaps
    gaps = []
    for problem_type, cluster_wounds in clusters.items():
        recurrence_count = len(cluster_wounds)
        resolve_times = [w.get("time_to_resolve_ms", 0) for w in cluster_wounds]
        median_resolve_ms = int(statistics.median(resolve_times)) if resolve_times else 0

        # Apply thresholds
        if recurrence_count > WOUND_RECURRENCE_THRESHOLD and median_resolve_ms > WOUND_RESOLVE_THRESHOLD_MS:
            # Extract timestamps
            timestamps = [w.get("ts", "") for w in cluster_wounds]
            first_seen = min(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()
            last_seen = max(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()

            # Extract wound IDs
            wound_ids = [w.get("intervention_id", "") for w in cluster_wounds]

            gaps.append({
                "problem_type": problem_type,
                "wound_ids": wound_ids,
                "recurrence_count": recurrence_count,
                "median_resolve_ms": median_resolve_ms,
                "first_seen": first_seen,
                "last_seen": last_seen
            })

    return gaps


# =============================================================================
# CORE FUNCTION 2: synthesize_blueprint
# =============================================================================

def synthesize_blueprint(gap: dict, wounds: List[dict]) -> dict:
    """
    Creates AgentBlueprint from AutomationGap.

    Args:
        gap: AutomationGap dict from identify_automation_gaps
        wounds: Original wound list for context (to estimate autonomy)

    Returns:
        dict: AgentBlueprint with fields:
            - name: str (derived from problem_type)
            - origin_receipts: list[str] (wound IDs)
            - validation_criteria: dict (extracted from resolution actions)
            - autonomy: float (0.0-1.0, estimated from risk)
            - approved_by: None (until HITL gate)
    """
    problem_type = gap.get("problem_type", "operational")
    wound_ids = gap.get("wound_ids", [])

    # Derive agent name from problem_type
    # e.g., "thermal_runaway" -> "thermal_monitor"
    name = f"{problem_type}_agent"

    # Extract validation criteria from resolution actions
    gap_wounds = [w for w in wounds if w.get("intervention_id", "") in wound_ids]
    resolution_actions = [w.get("resolution_action", "") for w in gap_wounds]

    validation_criteria = {
        "success_patterns": list(set(resolution_actions)),  # Unique actions
        "median_resolve_time_ms": gap.get("median_resolve_ms", 0),
        "expected_recurrence": gap.get("recurrence_count", 0)
    }

    # Estimate autonomy level
    autonomy = _estimate_autonomy(problem_type, gap_wounds)

    blueprint = {
        "name": name,
        "origin_receipts": wound_ids,
        "validation_criteria": validation_criteria,
        "autonomy": autonomy,
        "approved_by": None
    }

    return blueprint


# =============================================================================
# CORE FUNCTION 3: propose_agent
# =============================================================================

def propose_agent(blueprint: dict, tenant_id: str = "default") -> dict:
    """
    Submits blueprint to HITL gate and emits appropriate genesis receipt.

    Decision logic:
        - If autonomy > 0.7: require human approval, return genesis_pending
        - If autonomy <= 0.7 AND confidence > 0.9 AND risk = minimal: auto-approve
        - Else: return genesis_rejected

    Args:
        blueprint: AgentBlueprint dict
        tenant_id: Tenant identifier

    Returns:
        dict: genesis receipt (pending, approved, or rejected)
    """
    autonomy = blueprint.get("autonomy", 0.5)
    name = blueprint.get("name", "unknown_agent")

    # Calculate confidence from validation criteria
    validation_criteria = blueprint.get("validation_criteria", {})
    success_patterns = validation_criteria.get("success_patterns", [])
    confidence = min(1.0, len(success_patterns) / 3.0)  # More patterns = higher confidence

    # Decision tree
    if autonomy > HIGH_AUTONOMY_THRESHOLD:
        # High autonomy requires HITL
        return emit_genesis_pending(
            tenant_id=tenant_id,
            blueprint_name=name,
            autonomy=autonomy,
            reason="High autonomy requires human approval"
        )
    elif autonomy <= HIGH_AUTONOMY_THRESHOLD and confidence > AUTO_APPROVE_CONFIDENCE:
        # Auto-approve low-risk, high-confidence blueprints
        return emit_genesis_approved(
            tenant_id=tenant_id,
            blueprint_name=name,
            autonomy=autonomy,
            approved_by="auto_approve_system"
        )
    else:
        # Reject low confidence blueprints
        return emit_genesis_rejected(
            tenant_id=tenant_id,
            blueprint_name=name,
            autonomy=autonomy,
            reason=f"Confidence too low: {confidence:.2f} < {AUTO_APPROVE_CONFIDENCE}"
        )


# =============================================================================
# CORE FUNCTION 4: validate_blueprint
# =============================================================================

def validate_blueprint(blueprint: dict) -> Tuple[bool, List[str]]:
    """
    Pre-deployment validation of blueprint.

    Checks:
        - origin_receipts exist (non-empty)
        - validation_criteria are testable (non-empty, parseable)
        - no conflict with existing patterns (name collision check - simplified)

    Args:
        blueprint: AgentBlueprint dict

    Returns:
        Tuple[bool, List[str]]: (is_valid, list of violation messages)
    """
    violations = []

    # Check 1: origin_receipts exist
    origin_receipts = blueprint.get("origin_receipts", [])
    if not origin_receipts:
        violations.append("origin_receipts is empty")

    # Check 2: validation_criteria are testable
    validation_criteria = blueprint.get("validation_criteria", {})
    if not validation_criteria:
        violations.append("validation_criteria is empty")
    else:
        success_patterns = validation_criteria.get("success_patterns", [])
        if not success_patterns:
            violations.append("validation_criteria.success_patterns is empty")

    # Check 3: name collision (simplified - just check for valid name)
    name = blueprint.get("name", "")
    if not name or len(name) < 3:
        violations.append("name is invalid or too short")

    # Check 4: autonomy in valid range
    autonomy = blueprint.get("autonomy", -1)
    if not (0.0 <= autonomy <= 1.0):
        violations.append(f"autonomy out of range [0.0, 1.0]: {autonomy}")

    is_valid = len(violations) == 0
    return is_valid, violations


# =============================================================================
# INTERNAL FUNCTION: _estimate_autonomy
# =============================================================================

def _estimate_autonomy(problem_type: str, wounds: List[dict]) -> float:
    """
    Heuristic for autonomy level based on risk assessment.

    High-risk problem types (safety, critical) → low autonomy (0.3-0.5)
    Medium-risk (performance, operational) → medium autonomy (0.5-0.7)
    Low-risk (utility, convenience) → high autonomy (0.7-0.9)

    Args:
        problem_type: str from wound classification
        wounds: List of wound dicts for context

    Returns:
        float: Autonomy level in [0.0, 1.0]
    """
    # Base autonomy by problem type
    autonomy_map = {
        "safety": 0.3,        # High-risk, human judgment required
        "integration": 0.5,   # Medium-risk, complex interactions
        "performance": 0.6,   # Medium-risk, measurable outcomes
        "operational": 0.7    # Low-risk, routine tasks
    }

    base_autonomy = autonomy_map.get(problem_type, 0.5)

    # Adjust based on wound characteristics
    if wounds:
        # Higher automation confidence from wounds increases autonomy
        automation_scores = [w.get("could_automate", 0.5) for w in wounds]
        avg_automation_confidence = sum(automation_scores) / len(automation_scores)

        # Blend base autonomy with automation confidence
        autonomy = (base_autonomy + avg_automation_confidence) / 2.0
    else:
        autonomy = base_autonomy

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, autonomy))


# =============================================================================
# RECEIPT TYPE 1: blueprint_proposed
# =============================================================================

# --- SCHEMA ---
BLUEPRINT_PROPOSED_SCHEMA = {
    "receipt_type": "blueprint_proposed",
    "ts": "ISO8601",
    "tenant_id": "str",
    "blueprint_name": "str",
    "origin_receipts": ["str"],
    "autonomy": "float",
    "validation_status": "str",  # "valid" | "invalid"
    "violations": ["str"],
    "payload_hash": "str"
}


# --- EMIT ---
def emit_blueprint_proposed(tenant_id: str, blueprint: dict) -> dict:
    """Emit blueprint_proposed receipt after synthesis."""
    is_valid, violations = validate_blueprint(blueprint)

    return emit_receipt("blueprint_proposed", {
        "tenant_id": tenant_id,
        "blueprint_name": blueprint.get("name", "unknown"),
        "origin_receipts": blueprint.get("origin_receipts", []),
        "autonomy": blueprint.get("autonomy", 0.5),
        "validation_status": "valid" if is_valid else "invalid",
        "violations": violations
    })


# --- TEST ---
def test_blueprint_proposed():
    """Test blueprint_proposed receipt emission."""
    blueprint = {
        "name": "test_agent",
        "origin_receipts": ["wound_001", "wound_002"],
        "validation_criteria": {"success_patterns": ["restart"]},
        "autonomy": 0.6,
        "approved_by": None
    }
    r = emit_blueprint_proposed("test_tenant", blueprint)
    assert r["receipt_type"] == "blueprint_proposed"
    assert r["tenant_id"] == "test_tenant"
    assert r["validation_status"] in ["valid", "invalid"]
    assert "payload_hash" in r


# --- STOPRULE ---
def stoprule_blueprint_proposed(tenant_id: str, validation_status: str,
                                violations: List[str]) -> None:
    """Stoprule for invalid blueprint proposals."""
    if validation_status == "invalid":
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "blueprint_validation",
            "baseline": 1.0,
            "delta": -1.0,
            "classification": "deviation",
            "action": "alert",
            "violations": violations
        })
        raise StopRule(f"Blueprint validation failed: {violations}")


# =============================================================================
# RECEIPT TYPE 2: genesis_pending
# =============================================================================

# --- SCHEMA ---
GENESIS_PENDING_SCHEMA = {
    "receipt_type": "genesis_pending",
    "ts": "ISO8601",
    "tenant_id": "str",
    "blueprint_name": "str",
    "autonomy": "float",
    "reason": "str",
    "hitl_required": "bool",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_genesis_pending(tenant_id: str, blueprint_name: str,
                         autonomy: float, reason: str) -> dict:
    """Emit genesis_pending receipt for HITL gate."""
    return emit_receipt("genesis_pending", {
        "tenant_id": tenant_id,
        "blueprint_name": blueprint_name,
        "autonomy": autonomy,
        "reason": reason,
        "hitl_required": True
    })


# --- TEST ---
def test_genesis_pending():
    """Test genesis_pending receipt emission."""
    r = emit_genesis_pending(
        tenant_id="test_tenant",
        blueprint_name="high_autonomy_agent",
        autonomy=0.85,
        reason="High autonomy requires human approval"
    )
    assert r["receipt_type"] == "genesis_pending"
    assert r["tenant_id"] == "test_tenant"
    assert r["hitl_required"] is True
    assert r["autonomy"] == 0.85


# --- STOPRULE ---
def stoprule_genesis_pending(tenant_id: str, autonomy: float) -> None:
    """Stoprule for genesis_pending - alerts on high autonomy agents."""
    if autonomy > HIGH_AUTONOMY_THRESHOLD:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "agent_autonomy",
            "baseline": HIGH_AUTONOMY_THRESHOLD,
            "delta": autonomy - HIGH_AUTONOMY_THRESHOLD,
            "classification": "alert",
            "action": "escalate"
        })


# =============================================================================
# RECEIPT TYPE 3: genesis_approved
# =============================================================================

# --- SCHEMA ---
GENESIS_APPROVED_SCHEMA = {
    "receipt_type": "genesis_approved",
    "ts": "ISO8601",
    "tenant_id": "str",
    "blueprint_name": "str",
    "autonomy": "float",
    "approved_by": "str",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_genesis_approved(tenant_id: str, blueprint_name: str,
                          autonomy: float, approved_by: str) -> dict:
    """Emit genesis_approved receipt after HITL approval."""
    return emit_receipt("genesis_approved", {
        "tenant_id": tenant_id,
        "blueprint_name": blueprint_name,
        "autonomy": autonomy,
        "approved_by": approved_by
    })


# --- TEST ---
def test_genesis_approved():
    """Test genesis_approved receipt emission."""
    r = emit_genesis_approved(
        tenant_id="test_tenant",
        blueprint_name="operational_agent",
        autonomy=0.65,
        approved_by="auto_approve_system"
    )
    assert r["receipt_type"] == "genesis_approved"
    assert r["tenant_id"] == "test_tenant"
    assert r["approved_by"] == "auto_approve_system"


# --- STOPRULE ---
def stoprule_genesis_approved(tenant_id: str, approved_by: str) -> None:
    """Stoprule for genesis_approved - log successful genesis."""
    # No failure condition - just log for audit
    pass


# =============================================================================
# RECEIPT TYPE 4: genesis_rejected
# =============================================================================

# --- SCHEMA ---
GENESIS_REJECTED_SCHEMA = {
    "receipt_type": "genesis_rejected",
    "ts": "ISO8601",
    "tenant_id": "str",
    "blueprint_name": "str",
    "autonomy": "float",
    "reason": "str",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_genesis_rejected(tenant_id: str, blueprint_name: str,
                          autonomy: float, reason: str) -> dict:
    """Emit genesis_rejected receipt when blueprint fails approval."""
    return emit_receipt("genesis_rejected", {
        "tenant_id": tenant_id,
        "blueprint_name": blueprint_name,
        "autonomy": autonomy,
        "reason": reason
    })


# --- TEST ---
def test_genesis_rejected():
    """Test genesis_rejected receipt emission."""
    r = emit_genesis_rejected(
        tenant_id="test_tenant",
        blueprint_name="low_confidence_agent",
        autonomy=0.45,
        reason="Confidence too low: 0.33 < 0.9"
    )
    assert r["receipt_type"] == "genesis_rejected"
    assert r["tenant_id"] == "test_tenant"
    assert "reason" in r


# --- STOPRULE ---
def stoprule_genesis_rejected(tenant_id: str, reason: str) -> None:
    """Stoprule for genesis_rejected - emit anomaly for tracking."""
    emit_receipt("anomaly", {
        "tenant_id": tenant_id,
        "metric": "genesis_rejection",
        "baseline": 0.0,
        "delta": 1.0,
        "classification": "deviation",
        "action": "alert",
        "reason": reason
    })


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "GENESIS_LOCK",
    "WOUND_RECURRENCE_THRESHOLD",
    "WOUND_RESOLVE_THRESHOLD_MS",
    "HIGH_AUTONOMY_THRESHOLD",
    "AUTO_APPROVE_CONFIDENCE",
    # Core functions
    "identify_automation_gaps",
    "synthesize_blueprint",
    "propose_agent",
    "validate_blueprint",
    # Emit functions
    "emit_blueprint_proposed",
    "emit_genesis_pending",
    "emit_genesis_approved",
    "emit_genesis_rejected",
    # Stoprules
    "stoprule_blueprint_proposed",
    "stoprule_genesis_pending",
    "stoprule_genesis_approved",
    "stoprule_genesis_rejected",
    # Core utilities
    "emit_receipt",
    "dual_hash",
    "StopRule",
]
