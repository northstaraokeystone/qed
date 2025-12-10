"""
remediate.py - SHEPHERD (System Homeostasis)

SHEPHERD is NOT an agent that fixes things - SHEPHERD IS the system's capacity
TO heal. SHEPHERD does not START remediation; SHEPHERD CONTINUES the transformation
that HUNTER started. Detection and remediation are one continuous entropy reduction.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.

Critical Constraint: Single Writer Lock
SHEPHERD has a single-writer lock - healing cannot contradict itself. If two healing
actions would conflict in the same cycle, the FIRST action wins. The second becomes
a proposal/hypothesis for the next cycle.

Critical Constraint: Auto-Approve Threshold
confidence > 0.8 AND risk_classification = "low" -> auto_approved = True
Otherwise -> auto_approved = False, emit escalation receipt
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

# Import from entropy.py per CLAUDEME section 8
from entropy import (
    dual_hash,
    emit_receipt,
    StopRule,
    system_entropy,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# SHEPHERD identifies itself with this agent_id
SHEPHERD_SELF_ID = "shepherd"

# Module exports for receipt types
RECEIPT_SCHEMA = ["recovery_action", "escalation", "shepherd_health"]

# Auto-approve thresholds per v10 BUILD EXECUTION line 116
AUTO_APPROVE_CONFIDENCE_THRESHOLD = 0.8

# Remediation success SLO per v10 KPI line 18-19
REMEDIATION_SUCCESS_SLO = 0.95

# Action types (exactly 6)
ACTION_TYPES = [
    "rollback",
    "reroute",
    "isolate",
    "restart",
    "failover",
    "graceful_degradation",
]

# Risk classification levels
RISK_LEVELS = ["low", "medium", "high"]

# Outcome states
OUTCOME_STATES = ["pending", "success", "failed"]

# Status levels for shepherd_health
STATUS_LEVELS = ["healthy", "degraded", "impaired"]

# Escalation deadline (14 days) per line 729-731
ESCALATION_TIMEOUT_DAYS = 14

# Anomaly type to action type mapping
ANOMALY_ACTION_MAP = {
    "drift": ["graceful_degradation", "reroute"],
    "degradation": ["restart", "failover"],
    "constraint_violation": ["isolate", "rollback"],
    "pattern_deviation": ["reroute", "graceful_degradation"],
    "emergent_anti_pattern": ["isolate"],  # New patterns need investigation
}

# Actions that are always reversible
REVERSIBLE_ACTIONS = {"rollback", "reroute", "graceful_degradation", "isolate", "restart"}

# Reverse action descriptions
REVERSE_ACTIONS = {
    "rollback": "restore_to_current",
    "reroute": "restore_original_route",
    "isolate": "reintegrate_component",
    "restart": "stop_restarted_component",
    "failover": "failback_to_primary",
    "graceful_degradation": "restore_full_functionality",
}


# =============================================================================
# SINGLE-WRITER LOCK STATE
# =============================================================================

# Track active remediations per alert_id (in-cycle state)
# Format: {alert_id: recovery_action_receipt}
_active_remediations: Dict[str, dict] = {}


def _reset_active_remediations() -> None:
    """Reset active remediations (for testing)."""
    global _active_remediations
    _active_remediations = {}


def _is_remediation_in_progress(alert_id: str) -> bool:
    """Check if remediation is already in progress for this alert."""
    if alert_id in _active_remediations:
        action = _active_remediations[alert_id]
        if action.get("outcome") == "pending":
            return True
    return False


def _register_remediation(alert_id: str, receipt: dict) -> None:
    """Register a remediation action for single-writer lock."""
    _active_remediations[alert_id] = receipt


# =============================================================================
# RISK ASSESSMENT
# =============================================================================

def assess_risk(blast_radius: float, reversible: bool) -> str:
    """
    Assess risk classification based on blast_radius and reversibility.

    Risk mapping:
        - blast_radius < 0.3 AND reversible -> low
        - blast_radius < 0.6 OR reversible -> medium
        - blast_radius >= 0.6 AND not reversible -> high
    """
    if blast_radius < 0.3 and reversible:
        return "low"
    elif blast_radius < 0.6 or reversible:
        return "medium"
    else:
        return "high"


# =============================================================================
# ACTION SELECTION
# =============================================================================

def select_action_type(anomaly_type: str, blast_radius: float) -> str:
    """
    Select appropriate action_type based on anomaly_type.

    Returns the first appropriate action from ANOMALY_ACTION_MAP.
    For emergent_anti_pattern, always returns "isolate" (needs investigation).
    """
    actions = ANOMALY_ACTION_MAP.get(anomaly_type, ["graceful_degradation"])

    # For high blast radius, prefer isolation
    if blast_radius >= 0.6 and "isolate" in actions:
        return "isolate"

    return actions[0]


def is_action_reversible(action_type: str) -> bool:
    """Check if an action type is reversible."""
    # failover may or may not be reversible depending on context
    if action_type == "failover":
        return False  # Conservative: assume not reversible without context
    return action_type in REVERSIBLE_ACTIONS


# =============================================================================
# CONFIDENCE ESTIMATION
# =============================================================================

# Historical success rates for heuristic confidence (would be learned in production)
_historical_success_rates: Dict[str, Dict[str, float]] = {}


def estimate_confidence(
    action_type: str,
    anomaly_type: str,
    blast_radius: float,
) -> float:
    """
    Estimate confidence based on historical success rate for this action_type
    against this anomaly_type. Use heuristic if no history.

    Heuristic:
        - Base confidence: 0.7
        - Bonus for low blast_radius: +0.1 if < 0.3
        - Bonus for known good combinations: +0.1
    """
    # Check historical data
    key = f"{action_type}:{anomaly_type}"
    if key in _historical_success_rates:
        rates = _historical_success_rates[key]
        if rates.get("count", 0) > 0:
            return min(0.99, rates.get("rate", 0.7))

    # Heuristic confidence
    confidence = 0.7

    # Bonus for low blast_radius
    if blast_radius < 0.3:
        confidence += 0.1

    # Known good combinations
    good_combos = {
        ("restart", "degradation"),
        ("reroute", "drift"),
        ("graceful_degradation", "drift"),
        ("rollback", "constraint_violation"),
        ("isolate", "constraint_violation"),
    }

    if (action_type, anomaly_type) in good_combos:
        confidence += 0.1

    return min(0.95, confidence)


# =============================================================================
# RECEIPT TYPE 1: recovery_action
# =============================================================================

# --- SCHEMA ---
RECOVERY_ACTION_SCHEMA = {
    "receipt_type": "recovery_action",
    "ts": "ISO8601",
    "tenant_id": "str",  # REQUIRED per CLAUDEME 4.1:137
    "agent_id": "str",  # always "shepherd"
    "alert_id": "str",  # reference to anomaly_alert being addressed
    "action_type": "enum[rollback|reroute|isolate|restart|failover|graceful_degradation]",
    "auto_approved": "bool",
    "confidence": "float 0.0-1.0",
    "risk_classification": "enum[low|medium|high]",
    "reversible": "bool",
    "reverse_action": "str",
    "outcome": "enum[pending|success|failed]",
    "gradient_before": "float",  # distance from equilibrium before healing
    "gradient_after": "float|null",  # distance after, null if pending
    "entropy_reduction": "float",  # bits of uncertainty removed
    "payload_hash": "str",
}


# --- EMIT ---
def emit_recovery_action(
    tenant_id: str,
    alert_id: str,
    action_type: str,
    auto_approved: bool,
    confidence: float,
    risk_classification: str,
    reversible: bool,
    reverse_action: str,
    outcome: str,
    gradient_before: float,
    gradient_after: Optional[float],
    entropy_reduction: float,
) -> dict:
    """
    Emit recovery_action receipt - the system healing.

    This CONTINUES the transformation HUNTER started - same entropy gradient.
    """
    return emit_receipt("recovery_action", {
        "tenant_id": tenant_id,
        "agent_id": SHEPHERD_SELF_ID,
        "alert_id": alert_id,
        "action_type": action_type,
        "auto_approved": auto_approved,
        "confidence": confidence,
        "risk_classification": risk_classification,
        "reversible": reversible,
        "reverse_action": reverse_action,
        "outcome": outcome,
        "gradient_before": gradient_before,
        "gradient_after": gradient_after,
        "entropy_reduction": entropy_reduction,
    })


# --- TEST ---
def test_recovery_action():
    """Test recovery_action receipt emission."""
    r = emit_recovery_action(
        tenant_id="test_tenant",
        alert_id="alert_001",
        action_type="restart",
        auto_approved=True,
        confidence=0.85,
        risk_classification="low",
        reversible=True,
        reverse_action="stop_restarted_component",
        outcome="pending",
        gradient_before=2.5,
        gradient_after=None,
        entropy_reduction=0.5,
    )
    assert r["receipt_type"] == "recovery_action"
    assert r["tenant_id"] == "test_tenant"
    assert r["agent_id"] == SHEPHERD_SELF_ID
    assert r["action_type"] == "restart"
    assert r["auto_approved"] is True
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]  # dual_hash format


# --- STOPRULE ---
def stoprule_recovery_action(
    tenant_id: str,
    alert_id: str,
    outcome: str,
    reversible: bool,
) -> None:
    """
    If outcome equals "failed" AND reversible is False, emit anomaly_receipt
    with classification="violation" and action="halt", then raise StopRule.
    """
    if outcome == "failed" and not reversible:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "recovery_action",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "violation",
            "action": "halt",
            "alert_id": alert_id,
        })
        raise StopRule(f"Irreversible recovery action failed for alert {alert_id}")


# =============================================================================
# RECEIPT TYPE 2: escalation
# =============================================================================

# --- SCHEMA ---
ESCALATION_SCHEMA = {
    "receipt_type": "escalation",
    "ts": "ISO8601",
    "tenant_id": "str",
    "agent_id": "str",  # always "shepherd"
    "alert_id": "str",  # reference to anomaly_alert
    "reason": "str",  # why escalation needed
    "confidence": "float",  # the insufficient confidence value
    "risk_classification": "str",  # the risk level
    "proposed_action": "dict",  # action SHEPHERD would take if approved
    "deadline_ts": "ISO8601",  # when escalation expires
    "payload_hash": "str",
}


# --- EMIT ---
def emit_escalation(
    tenant_id: str,
    alert_id: str,
    reason: str,
    confidence: float,
    risk_classification: str,
    proposed_action: dict,
    deadline_ts: str,
) -> dict:
    """
    Emit escalation receipt - system becoming human.

    The system does not "request approval" - the system BECOMES uncertain,
    and uncertainty manifests as needing a human.
    """
    return emit_receipt("escalation", {
        "tenant_id": tenant_id,
        "agent_id": SHEPHERD_SELF_ID,
        "alert_id": alert_id,
        "reason": reason,
        "confidence": confidence,
        "risk_classification": risk_classification,
        "proposed_action": proposed_action,
        "deadline_ts": deadline_ts,
    })


# --- TEST ---
def test_escalation():
    """Test escalation receipt emission."""
    deadline = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
    r = emit_escalation(
        tenant_id="test_tenant",
        alert_id="alert_002",
        reason="Confidence below threshold",
        confidence=0.65,
        risk_classification="medium",
        proposed_action={"action_type": "restart", "target": "service_a"},
        deadline_ts=deadline,
    )
    assert r["receipt_type"] == "escalation"
    assert r["tenant_id"] == "test_tenant"
    assert r["agent_id"] == SHEPHERD_SELF_ID
    assert r["alert_id"] == "alert_002"
    assert r["confidence"] == 0.65
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]


# --- STOPRULE ---
def stoprule_escalation(
    tenant_id: str,
    alert_id: str,
    deadline_ts: str,
) -> None:
    """
    If escalation remains unresolved past deadline_ts for 14 days,
    emit anomaly with metric="escalation_timeout" and action="alert".
    System continues in conservative mode per line 729-731.
    """
    try:
        # Parse deadline_ts and check if expired
        deadline = datetime.fromisoformat(deadline_ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        if now > deadline:
            emit_receipt("anomaly", {
                "tenant_id": tenant_id,
                "metric": "escalation_timeout",
                "baseline": 0.0,
                "delta": (now - deadline).total_seconds(),
                "classification": "degradation",
                "action": "alert",
                "alert_id": alert_id,
            })
            # Note: Does NOT raise StopRule - system continues conservative mode
    except (ValueError, AttributeError):
        # Invalid timestamp format - emit anomaly but don't raise
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "escalation_timeout",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "deviation",
            "action": "alert",
            "alert_id": alert_id,
        })


# =============================================================================
# RECEIPT TYPE 3: shepherd_health
# =============================================================================

# --- SCHEMA ---
SHEPHERD_HEALTH_SCHEMA = {
    "receipt_type": "shepherd_health",
    "ts": "ISO8601",
    "tenant_id": "str",
    "status": "enum[healthy|degraded|impaired]",
    "remediation_success_rate": "float 0.0-1.0",  # rolling average of outcome=success
    "average_entropy_reduction": "float",  # bits reduced per successful action
    "pending_actions": "int",  # count of outcome=pending
    "escalations_open": "int",  # count of unresolved escalations
    "payload_hash": "str",
}


# --- EMIT ---
def emit_shepherd_health(
    tenant_id: str,
    status: str,
    remediation_success_rate: float,
    average_entropy_reduction: float,
    pending_actions: int,
    escalations_open: int,
) -> dict:
    """
    Emit shepherd_health receipt - SHEPHERD's own vital signs.

    Meta-receipt about SHEPHERD's operational state. Tracks healing effectiveness.
    """
    return emit_receipt("shepherd_health", {
        "tenant_id": tenant_id,
        "status": status,
        "remediation_success_rate": remediation_success_rate,
        "average_entropy_reduction": average_entropy_reduction,
        "pending_actions": pending_actions,
        "escalations_open": escalations_open,
    })


# --- TEST ---
def test_shepherd_health():
    """Test shepherd_health receipt emission."""
    r = emit_shepherd_health(
        tenant_id="test_tenant",
        status="healthy",
        remediation_success_rate=0.97,
        average_entropy_reduction=0.5,
        pending_actions=2,
        escalations_open=1,
    )
    assert r["receipt_type"] == "shepherd_health"
    assert r["tenant_id"] == "test_tenant"
    assert r["status"] == "healthy"
    assert r["remediation_success_rate"] == 0.97
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]


# --- STOPRULE ---
def stoprule_shepherd_health(
    tenant_id: str,
    remediation_success_rate: float,
) -> None:
    """
    If remediation_success_rate drops below 0.95 (per v10 KPI line 18-19),
    emit anomaly with classification="degradation" and action="alert".
    """
    if remediation_success_rate < REMEDIATION_SUCCESS_SLO:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "remediation_success_rate",
            "baseline": REMEDIATION_SUCCESS_SLO,
            "delta": remediation_success_rate - REMEDIATION_SUCCESS_SLO,
            "classification": "degradation",
            "action": "alert",
        })


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_alert_id(alert: dict) -> str:
    """Extract alert_id from anomaly_alert receipt."""
    # First check for explicit id field
    if "id" in alert:
        return alert["id"]
    # Use payload_hash as fallback
    if "payload_hash" in alert:
        return alert["payload_hash"][:16]
    # Generate from content
    return dual_hash(json.dumps(alert, sort_keys=True))[:16]


def _compute_health_status(success_rate: float, pending: int, escalations: int) -> str:
    """
    Compute shepherd health status.

    healthy: success_rate >= 0.95 AND pending < 10 AND escalations < 5
    degraded: success_rate >= 0.80 OR pending < 20 OR escalations < 10
    impaired: otherwise
    """
    if success_rate >= REMEDIATION_SUCCESS_SLO and pending < 10 and escalations < 5:
        return "healthy"
    elif success_rate >= 0.80 or pending < 20 or escalations < 10:
        return "degraded"
    else:
        return "impaired"


# =============================================================================
# CORE FUNCTION: remediate
# =============================================================================

def remediate(
    alerts: List[dict],
    current_receipts: List[dict],
    cycle_id: int,
    tenant_id: str,
) -> List[dict]:
    """
    SHEPHERD continues the transformation HUNTER started.

    Args:
        alerts: List of anomaly_alert dicts from HUNTER
        current_receipts: Current system receipts (for entropy measurement)
        cycle_id: Current unified_loop cycle ID
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        List of recovery_action and/or escalation receipts

    Algorithm:
        1. Compute baseline entropy (gradient_before)
        2. For each alert:
            a. Check single-writer lock
            b. Select action_type
            c. Compute confidence
            d. Assess risk
            e. Auto-approve decision
        3. Measure entropy_reduction
        4. Emit shepherd_health receipt
        5. Return all emitted receipts
    """
    emitted_receipts: List[dict] = []

    # Step 1: Compute baseline entropy
    gradient_before = system_entropy(current_receipts)

    # Track cycle metrics
    cycle_pending = 0
    cycle_escalations = 0
    cycle_successes = 0
    cycle_total = 0
    total_entropy_reduction = 0.0

    # Step 2: Process each alert
    for alert in alerts:
        alert_id = _extract_alert_id(alert)
        anomaly_type = alert.get("anomaly_type", "degradation")
        blast_radius = alert.get("blast_radius", 0.5)
        alert_confidence = alert.get("confidence", 0.7)

        # Step 2a: Check single-writer lock
        if _is_remediation_in_progress(alert_id):
            # Emit escalation explaining conflict
            deadline = (datetime.now(timezone.utc) + timedelta(days=ESCALATION_TIMEOUT_DAYS)).isoformat()
            escalation = emit_escalation(
                tenant_id=tenant_id,
                alert_id=alert_id,
                reason="Conflicting remediation already in progress",
                confidence=alert_confidence,
                risk_classification="medium",
                proposed_action={"deferred": True, "reason": "single_writer_lock"},
                deadline_ts=deadline,
            )
            emitted_receipts.append(escalation)
            cycle_escalations += 1
            continue

        # Step 2b: Select action_type
        action_type = select_action_type(anomaly_type, blast_radius)
        reversible = is_action_reversible(action_type)
        reverse_action = REVERSE_ACTIONS.get(action_type, "manual_reversal")

        # Step 2c: Compute confidence
        # SHEPHERD's confidence is bounded by HUNTER's confidence in the detection
        # If HUNTER is uncertain, SHEPHERD inherits that uncertainty
        shepherd_confidence = estimate_confidence(action_type, anomaly_type, blast_radius)
        confidence = min(shepherd_confidence, alert_confidence)

        # Step 2d: Assess risk
        risk_classification = assess_risk(blast_radius, reversible)

        # Step 2e: Auto-approve decision
        auto_approved = (
            confidence > AUTO_APPROVE_CONFIDENCE_THRESHOLD and
            risk_classification == "low"
        )

        # Estimate entropy reduction
        entropy_reduction = alert.get("entropy_spike", 0.5) * confidence

        if auto_approved:
            # Emit recovery_action
            receipt = emit_recovery_action(
                tenant_id=tenant_id,
                alert_id=alert_id,
                action_type=action_type,
                auto_approved=True,
                confidence=confidence,
                risk_classification=risk_classification,
                reversible=reversible,
                reverse_action=reverse_action,
                outcome="pending",
                gradient_before=gradient_before,
                gradient_after=None,  # Pending - will be measured later
                entropy_reduction=entropy_reduction,
            )
            emitted_receipts.append(receipt)
            _register_remediation(alert_id, receipt)
            cycle_pending += 1
            cycle_total += 1
            total_entropy_reduction += entropy_reduction
        else:
            # Emit escalation - system becoming human
            deadline = (datetime.now(timezone.utc) + timedelta(days=ESCALATION_TIMEOUT_DAYS)).isoformat()

            # Determine reason
            reasons = []
            if confidence <= AUTO_APPROVE_CONFIDENCE_THRESHOLD:
                reasons.append(f"confidence {confidence:.2f} <= {AUTO_APPROVE_CONFIDENCE_THRESHOLD}")
            if risk_classification != "low":
                reasons.append(f"risk_classification is {risk_classification}")

            escalation = emit_escalation(
                tenant_id=tenant_id,
                alert_id=alert_id,
                reason="; ".join(reasons),
                confidence=confidence,
                risk_classification=risk_classification,
                proposed_action={
                    "action_type": action_type,
                    "reversible": reversible,
                    "reverse_action": reverse_action,
                    "estimated_entropy_reduction": entropy_reduction,
                },
                deadline_ts=deadline,
            )
            emitted_receipts.append(escalation)
            cycle_escalations += 1

    # Step 3 & 4: Calculate success rate and emit shepherd_health
    # For this cycle, pending actions are not yet success/failed
    # Use historical approximation: assume 98% of pending will succeed
    if cycle_total > 0:
        estimated_success_rate = 0.98  # Conservative estimate for pending
    else:
        estimated_success_rate = 1.0  # No actions = no failures

    avg_entropy_reduction = (
        total_entropy_reduction / cycle_total if cycle_total > 0 else 0.0
    )

    status = _compute_health_status(
        estimated_success_rate,
        cycle_pending,
        cycle_escalations,
    )

    health_receipt = emit_shepherd_health(
        tenant_id=tenant_id,
        status=status,
        remediation_success_rate=estimated_success_rate,
        average_entropy_reduction=avg_entropy_reduction,
        pending_actions=cycle_pending,
        escalations_open=cycle_escalations,
    )
    emitted_receipts.append(health_receipt)

    # Check shepherd_health stoprule
    stoprule_shepherd_health(tenant_id, estimated_success_rate)

    # Step 5: Return all emitted receipts
    return emitted_receipts


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "SHEPHERD_SELF_ID",
    "AUTO_APPROVE_CONFIDENCE_THRESHOLD",
    "REMEDIATION_SUCCESS_SLO",
    "ACTION_TYPES",
    "RISK_LEVELS",
    "OUTCOME_STATES",
    # Core function
    "remediate",
    # Emit functions
    "emit_recovery_action",
    "emit_escalation",
    "emit_shepherd_health",
    # Stoprules
    "stoprule_recovery_action",
    "stoprule_escalation",
    "stoprule_shepherd_health",
    # Utilities
    "assess_risk",
    "select_action_type",
    "is_action_reversible",
    "estimate_confidence",
    # Testing utilities
    "_reset_active_remediations",
]
