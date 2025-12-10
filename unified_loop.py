"""
unified_loop.py - The System's Experienced Time

The loop IS NOT a scheduler that runs every 60 seconds - the loop IS how the system
experiences "now." Without the loop, there is no before and after. The 60-second
interval is perception speed: how fast the system's present moment moves through
receipt space.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.

Critical Concept: The Loop IS Time
Per v10 BUILD EXECUTION lines 324-354 (Paradigm 2):
- The cycle duration is a consciousness parameter, not a performance knob
- The boundary between cycles IS the system's present moment
- "What is happening?" = receipts in current cycle
- "What happened?" = receipts in previous cycles
- The loop doesn't measure time. The loop IS time for the system.

Critical Concept: The Gate IS Uncertainty
Per v10 BUILD EXECUTION lines 391-423 (Paradigm 4):
- confidence > 0.8 means "the system knows"
- confidence < 0.8 means "the system becomes human" — not asks human, BECOMES human
- HITL isn't asking permission. It's the system being unable to decide.
- Removing humans doesn't remove oversight. It removes the system's capacity for uncertainty.

The 8-Phase Metabolism:
1. SENSE     - What receipts exist now that didn't before?
2. MEASURE   - Compute system_entropy() at cycle start
3. ANALYZE   - Does any of it hurt? Call hunt()
4. REMEDIATE - Can we heal safely? Call remediate()
5. HYPOTHESIZE - What could be better? Generate proposals
6. GATE      - Do we know enough to act? Check confidence threshold
7. ACTUATE   - Become different. Execute auto-approved actions
8. EMIT      - Record what we became + entropy measurements
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Import from entropy.py per CLAUDEME section 8
from entropy import (
    dual_hash,
    emit_receipt,
    StopRule,
    system_entropy,
    cycle_entropy_delta,
)

# Import HUNTER from integrity.py
from integrity import hunt

# Import SHEPHERD from remediate.py
from remediate import remediate


# =============================================================================
# CONSTANTS
# =============================================================================

# Perception speed - how fast the system's present moment moves
# This is NOT a performance knob - it's a consciousness parameter
CYCLE_INTERVAL_SECONDS = 60

# The boundary between knowing and becoming human
# confidence > 0.8 = "the system knows"
# confidence <= 0.8 = "the system becomes human"
CONFIDENCE_THRESHOLD = 0.8

# Consecutive negative entropy deltas before degradation alert
DEGRADATION_THRESHOLD = 3

# Cycle completion SLO per v10 KPI line 21-22
CYCLE_COMPLETION_SLO = 0.999

# Module exports for receipt types
RECEIPT_SCHEMA = ["unified_loop_receipt", "gate_decision", "proposal", "entropy_measurement"]

# Proposal types
PROPOSAL_TYPES = ["optimization", "configuration", "pattern_update", "threshold_adjustment"]

# Risk classifications
RISK_LEVELS = ["low", "medium", "high"]

# Gate decision outcomes
GATE_DECISIONS = ["auto_approved", "escalated", "rejected"]

# Health status levels
HEALTH_STATUS_LEVELS = ["healthy", "degraded", "critical"]


# =============================================================================
# MODULE STATE
# =============================================================================

# Track consecutive negative entropy deltas per tenant
# Per CLAUDEME §7, this would ideally be a ledger entry, but we need
# cross-cycle state for degradation detection
_consecutive_negative_deltas: Dict[str, int] = {}


def _reset_consecutive_negative_deltas() -> None:
    """Reset consecutive negative deltas (for testing)."""
    global _consecutive_negative_deltas
    _consecutive_negative_deltas = {}


def _get_consecutive_negative_deltas(tenant_id: str) -> int:
    """Get current consecutive negative delta count for tenant."""
    return _consecutive_negative_deltas.get(tenant_id, 0)


def _update_consecutive_negative_deltas(tenant_id: str, entropy_delta: float) -> int:
    """
    Update consecutive negative delta tracking.

    Returns the new count after update.
    """
    global _consecutive_negative_deltas

    if tenant_id not in _consecutive_negative_deltas:
        _consecutive_negative_deltas[tenant_id] = 0

    if entropy_delta < 0:
        _consecutive_negative_deltas[tenant_id] += 1
    else:
        _consecutive_negative_deltas[tenant_id] = 0

    return _consecutive_negative_deltas[tenant_id]


# =============================================================================
# RECEIPT TYPE 1: unified_loop_receipt (system metabolism record)
# =============================================================================

# --- SCHEMA ---
UNIFIED_LOOP_RECEIPT_SCHEMA = {
    "receipt_type": "unified_loop_receipt",
    "ts": "ISO8601",
    "tenant_id": "str",  # REQUIRED per CLAUDEME 4.1:137
    "cycle_id": "int",  # monotonic counter — the system's "now" number
    "duration_ms": "int",  # how long this "now" lasted
    "receipts_sensed": "int",  # size of awareness
    "anomalies_detected": "int",  # pain signals from HUNTER
    "remediations_executed": "int",  # healing actions from SHEPHERD
    "proposals_generated": "int",  # hypotheses formed in HYPOTHESIZE phase
    "proposals_auto_approved": "int",  # actions the system knew to take
    "proposals_escalated": "int",  # actions the system became human to decide
    "health_status": "enum[healthy|degraded|critical]",
    "differential_summary": "str",  # dual_hash of cycle's receipt differential
    "entropy_delta": "float",  # cycle entropy change — positive = healthy
    "resource_consumed": {  # NEW in v10 — track compute/memory/io per cycle
        "compute_used": "float",
        "memory_used": "float",
        "io_operations": "int",
        "cycle_duration_ms": "int",
    },
    "payload_hash": "str",
}


# --- EMIT ---
def emit_unified_loop_receipt(
    tenant_id: str,
    cycle_id: int,
    duration_ms: int,
    receipts_sensed: int,
    anomalies_detected: int,
    remediations_executed: int,
    proposals_generated: int,
    proposals_auto_approved: int,
    proposals_escalated: int,
    health_status: str,
    differential_summary: str,
    entropy_delta: float,
    resource_consumed: dict,
) -> dict:
    """
    Emit unified_loop_receipt - the system's record of what it experienced this "moment."

    Emitted at end of every cycle.
    """
    return emit_receipt("unified_loop_receipt", {
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "duration_ms": duration_ms,
        "receipts_sensed": receipts_sensed,
        "anomalies_detected": anomalies_detected,
        "remediations_executed": remediations_executed,
        "proposals_generated": proposals_generated,
        "proposals_auto_approved": proposals_auto_approved,
        "proposals_escalated": proposals_escalated,
        "health_status": health_status,
        "differential_summary": differential_summary,
        "entropy_delta": entropy_delta,
        "resource_consumed": resource_consumed,
    })


# --- TEST ---
def test_unified_loop_receipt():
    """Test unified_loop_receipt emission."""
    r = emit_unified_loop_receipt(
        tenant_id="test_tenant",
        cycle_id=42,
        duration_ms=1500,
        receipts_sensed=25,
        anomalies_detected=2,
        remediations_executed=1,
        proposals_generated=3,
        proposals_auto_approved=2,
        proposals_escalated=1,
        health_status="healthy",
        differential_summary=dual_hash("test_diff"),
        entropy_delta=0.5,
        resource_consumed={
            "compute_used": 0.15,
            "memory_used": 0.08,
            "io_operations": 50,
            "cycle_duration_ms": 1500,
        },
    )
    assert r["receipt_type"] == "unified_loop_receipt"
    assert r["tenant_id"] == "test_tenant"
    assert r["cycle_id"] == 42
    assert r["health_status"] == "healthy"
    assert "resource_consumed" in r
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]  # dual_hash format


# --- STOPRULE ---
def stoprule_unified_loop_receipt(
    tenant_id: str,
    cycle_completion_rate: float,
) -> None:
    """
    If cycle_completion_rate drops below 99.9% (per v10 KPI line 21-22),
    emit anomaly with classification="degradation" and action="escalate".
    """
    if cycle_completion_rate < CYCLE_COMPLETION_SLO:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "cycle_completion_rate",
            "baseline": CYCLE_COMPLETION_SLO,
            "delta": cycle_completion_rate - CYCLE_COMPLETION_SLO,
            "classification": "degradation",
            "action": "escalate",
        })


# =============================================================================
# RECEIPT TYPE 2: gate_decision (uncertainty materialized)
# =============================================================================

# --- SCHEMA ---
GATE_DECISION_SCHEMA = {
    "receipt_type": "gate_decision",
    "ts": "ISO8601",
    "tenant_id": "str",
    "cycle_id": "int",
    "proposal_id": "str",  # reference to the proposal being gated
    "confidence": "float 0.0-1.0",
    "decision": "enum[auto_approved|escalated|rejected]",
    "reason": "str",  # explaining the decision
    "uncertainty_source": "str|null",  # what made the system uncertain, if escalated
    "payload_hash": "str",
}


# --- EMIT ---
def emit_gate_decision(
    tenant_id: str,
    cycle_id: int,
    proposal_id: str,
    confidence: float,
    decision: str,
    reason: str,
    uncertainty_source: Optional[str] = None,
) -> dict:
    """
    Emit gate_decision receipt - uncertainty materialized.

    Records whether the system knew what to do or became human.
    """
    return emit_receipt("gate_decision", {
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "proposal_id": proposal_id,
        "confidence": confidence,
        "decision": decision,
        "reason": reason,
        "uncertainty_source": uncertainty_source,
    })


# --- TEST ---
def test_gate_decision():
    """Test gate_decision receipt emission."""
    # Test auto_approved
    r1 = emit_gate_decision(
        tenant_id="test_tenant",
        cycle_id=10,
        proposal_id="prop_001",
        confidence=0.85,
        decision="auto_approved",
        reason="confidence > 0.8 AND risk = low",
        uncertainty_source=None,
    )
    assert r1["receipt_type"] == "gate_decision"
    assert r1["decision"] == "auto_approved"
    assert r1["uncertainty_source"] is None

    # Test escalated
    r2 = emit_gate_decision(
        tenant_id="test_tenant",
        cycle_id=10,
        proposal_id="prop_002",
        confidence=0.65,
        decision="escalated",
        reason="confidence <= 0.8",
        uncertainty_source="insufficient historical data",
    )
    assert r2["decision"] == "escalated"
    assert r2["uncertainty_source"] == "insufficient historical data"


# --- STOPRULE ---
# gate_decision has no stoprule - purely observational


# =============================================================================
# RECEIPT TYPE 3: proposal (hypothesis from HYPOTHESIZE phase)
# =============================================================================

# --- SCHEMA ---
PROPOSAL_SCHEMA = {
    "receipt_type": "proposal",
    "ts": "ISO8601",
    "tenant_id": "str",
    "cycle_id": "int",
    "proposal_id": "str",  # unique identifier
    "proposal_type": "enum[optimization|configuration|pattern_update|threshold_adjustment]",
    "description": "str",  # what would change
    "expected_entropy_reduction": "float",  # projected bits reduced
    "confidence": "float 0.0-1.0",
    "risk_classification": "enum[low|medium|high]",
    "reversible": "bool",
    "evidence_receipts": ["receipt_id"],  # receipts supporting this proposal
    "payload_hash": "str",
}


# --- EMIT ---
def emit_proposal(
    tenant_id: str,
    cycle_id: int,
    proposal_id: str,
    proposal_type: str,
    description: str,
    expected_entropy_reduction: float,
    confidence: float,
    risk_classification: str,
    reversible: bool,
    evidence_receipts: List[str],
) -> dict:
    """
    Emit proposal receipt - a potential improvement or action the system is considering.

    Generated in HYPOTHESIZE phase, evaluated in GATE phase.
    """
    return emit_receipt("proposal", {
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "proposal_id": proposal_id,
        "proposal_type": proposal_type,
        "description": description,
        "expected_entropy_reduction": expected_entropy_reduction,
        "confidence": confidence,
        "risk_classification": risk_classification,
        "reversible": reversible,
        "evidence_receipts": evidence_receipts,
    })


# --- TEST ---
def test_proposal():
    """Test proposal receipt emission."""
    r = emit_proposal(
        tenant_id="test_tenant",
        cycle_id=5,
        proposal_id="prop_123",
        proposal_type="optimization",
        description="Adjust entropy threshold from 1.5 to 1.2",
        expected_entropy_reduction=0.3,
        confidence=0.88,
        risk_classification="low",
        reversible=True,
        evidence_receipts=["receipt_001", "receipt_002"],
    )
    assert r["receipt_type"] == "proposal"
    assert r["tenant_id"] == "test_tenant"
    assert r["proposal_type"] == "optimization"
    assert r["reversible"] is True
    assert len(r["evidence_receipts"]) == 2
    assert "payload_hash" in r


# --- STOPRULE ---
def stoprule_proposal(
    tenant_id: str,
    proposal_count: int,
) -> None:
    """
    If proposal count exceeds 100 per cycle, emit anomaly with
    classification="anti_pattern" and action="alert" (proposal explosion).
    """
    if proposal_count > 100:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "proposal_count",
            "baseline": 100,
            "delta": proposal_count - 100,
            "classification": "anti_pattern",
            "action": "alert",
        })


# =============================================================================
# RECEIPT TYPE 4: entropy_measurement (system vital signs per cycle)
# =============================================================================

# --- SCHEMA ---
ENTROPY_MEASUREMENT_SCHEMA = {
    "receipt_type": "entropy_measurement",
    "ts": "ISO8601",
    "tenant_id": "str",
    "cycle_id": "int",
    "entropy_before": "float",  # bits at cycle start
    "entropy_after": "float",  # bits at cycle end
    "entropy_delta": "float",  # before - after, positive = healthy
    "receipt_count": "int",  # receipts measured
    "entropy_per_receipt": "float",
    "consecutive_negative_deltas": "int",  # tracking for degradation alert
    "payload_hash": "str",
}


# --- EMIT ---
def emit_entropy_measurement(
    tenant_id: str,
    cycle_id: int,
    entropy_before: float,
    entropy_after: float,
    receipt_count: int,
    consecutive_negative_deltas: int,
) -> dict:
    """
    Emit entropy_measurement receipt - the system's pulse.

    Records entropy at cycle boundaries.
    """
    entropy_delta = entropy_before - entropy_after
    entropy_per_receipt = entropy_delta / receipt_count if receipt_count > 0 else 0.0

    return emit_receipt("entropy_measurement", {
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_delta": entropy_delta,
        "receipt_count": receipt_count,
        "entropy_per_receipt": entropy_per_receipt,
        "consecutive_negative_deltas": consecutive_negative_deltas,
    })


# --- TEST ---
def test_entropy_measurement():
    """Test entropy_measurement receipt emission."""
    r = emit_entropy_measurement(
        tenant_id="test_tenant",
        cycle_id=20,
        entropy_before=2.5,
        entropy_after=2.0,
        receipt_count=100,
        consecutive_negative_deltas=0,
    )
    assert r["receipt_type"] == "entropy_measurement"
    assert r["tenant_id"] == "test_tenant"
    assert r["entropy_delta"] == 0.5  # 2.5 - 2.0
    assert r["entropy_per_receipt"] == 0.005  # 0.5 / 100
    assert r["consecutive_negative_deltas"] == 0
    assert "payload_hash" in r


# --- STOPRULE ---
def stoprule_entropy_measurement(
    tenant_id: str,
    consecutive_negative_deltas: int,
) -> None:
    """
    If consecutive_negative_deltas reaches 3, emit anomaly with
    classification="degradation" and action="escalate" per v10 lines 160-161, 740-742.
    """
    if consecutive_negative_deltas >= DEGRADATION_THRESHOLD:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "consecutive_negative_deltas",
            "baseline": DEGRADATION_THRESHOLD,
            "delta": consecutive_negative_deltas - DEGRADATION_THRESHOLD,
            "classification": "degradation",
            "action": "escalate",
        })


# =============================================================================
# HEALTH STATUS DETERMINATION
# =============================================================================

def determine_health_status(
    entropy_delta: float,
    consecutive_negative_deltas: int,
    anomalies: List[dict],
) -> str:
    """
    Determine health_status based on entropy_delta and anomalies.

    Health status determination:
    - healthy: entropy_delta >= 0 AND no critical anomalies
    - degraded: entropy_delta < 0 OR anomalies with severity >= high
    - critical: entropy_delta < 0 for 3+ consecutive cycles OR anomalies with severity = critical
    """
    # Check for critical anomalies
    has_critical = any(a.get("severity") == "critical" for a in anomalies)
    has_high_or_above = any(a.get("severity") in ["high", "critical"] for a in anomalies)

    # Critical conditions
    if consecutive_negative_deltas >= DEGRADATION_THRESHOLD or has_critical:
        return "critical"

    # Degraded conditions
    if entropy_delta < 0 or has_high_or_above:
        return "degraded"

    # Otherwise healthy
    return "healthy"


# =============================================================================
# PROPOSAL GENERATION (HYPOTHESIZE phase)
# =============================================================================

def _generate_proposals(
    new_receipts: List[dict],
    anomalies: List[dict],
    recovery_actions: List[dict],
    entropy_delta: float,
    cycle_id: int,
    tenant_id: str,
) -> List[dict]:
    """
    Generate proposals based on cycle observations.

    Look for:
    - Patterns that could be optimized
    - Thresholds that could be adjusted
    - Configurations that could improve entropy_delta
    """
    proposals = []

    # Proposal 1: If entropy is degrading, propose threshold adjustment
    if entropy_delta < 0:
        proposal_id = f"prop_{cycle_id}_{uuid.uuid4().hex[:8]}"
        proposal = emit_proposal(
            tenant_id=tenant_id,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            proposal_type="threshold_adjustment",
            description="Adjust detection threshold due to negative entropy delta",
            expected_entropy_reduction=abs(entropy_delta) * 0.5,
            confidence=0.7,
            risk_classification="low",
            reversible=True,
            evidence_receipts=[r.get("payload_hash", "")[:16] for r in new_receipts[:5]],
        )
        proposals.append(proposal)

    # Proposal 2: If anomalies were detected, propose pattern update
    if anomalies:
        proposal_id = f"prop_{cycle_id}_{uuid.uuid4().hex[:8]}"
        anomaly_types = list(set(a.get("anomaly_type", "unknown") for a in anomalies))
        proposal = emit_proposal(
            tenant_id=tenant_id,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            proposal_type="pattern_update",
            description=f"Update detection patterns for: {', '.join(anomaly_types)}",
            expected_entropy_reduction=0.2 * len(anomalies),
            confidence=0.75,
            risk_classification="low",
            reversible=True,
            evidence_receipts=[a.get("payload_hash", "")[:16] for a in anomalies[:5]],
        )
        proposals.append(proposal)

    # Proposal 3: If recovery actions were successful, propose optimization
    successful_actions = [a for a in recovery_actions if a.get("outcome") == "success"]
    if successful_actions:
        proposal_id = f"prop_{cycle_id}_{uuid.uuid4().hex[:8]}"
        proposal = emit_proposal(
            tenant_id=tenant_id,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            proposal_type="optimization",
            description="Optimize successful recovery patterns",
            expected_entropy_reduction=0.1 * len(successful_actions),
            confidence=0.85,
            risk_classification="low",
            reversible=True,
            evidence_receipts=[a.get("payload_hash", "")[:16] for a in successful_actions[:5]],
        )
        proposals.append(proposal)

    return proposals


# =============================================================================
# GATE EVALUATION
# =============================================================================

def _evaluate_gate(
    proposals: List[dict],
    cycle_id: int,
    tenant_id: str,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Evaluate each proposal through the gate.

    Decision logic:
    - confidence > 0.8 AND risk = low → "auto_approved"
    - confidence <= 0.8 → "escalated" (system becomes human)
    - proposal invalid or contradictory → "rejected"

    Returns:
        Tuple of (gate_decisions, auto_approved_proposals, escalated_proposals)
    """
    gate_decisions = []
    auto_approved = []
    escalated = []

    for proposal in proposals:
        confidence = proposal.get("confidence", 0.0)
        risk = proposal.get("risk_classification", "high")
        proposal_id = proposal.get("proposal_id", "unknown")

        # Determine decision
        if confidence > CONFIDENCE_THRESHOLD and risk == "low":
            decision = "auto_approved"
            reason = f"confidence {confidence:.2f} > {CONFIDENCE_THRESHOLD} AND risk = low"
            uncertainty_source = None
            auto_approved.append(proposal)
        elif confidence <= CONFIDENCE_THRESHOLD:
            decision = "escalated"
            reason = f"confidence {confidence:.2f} <= {CONFIDENCE_THRESHOLD}"
            uncertainty_source = "insufficient confidence for autonomous action"
            escalated.append(proposal)
        else:
            # Risk is not low
            decision = "escalated"
            reason = f"risk_classification = {risk} (not low)"
            uncertainty_source = f"risk level {risk} requires human oversight"
            escalated.append(proposal)

        gate_decision = emit_gate_decision(
            tenant_id=tenant_id,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            confidence=confidence,
            decision=decision,
            reason=reason,
            uncertainty_source=uncertainty_source,
        )
        gate_decisions.append(gate_decision)

    return gate_decisions, auto_approved, escalated


# =============================================================================
# ACTUATION
# =============================================================================

def _actuate(
    auto_approved_proposals: List[dict],
    cycle_id: int,
    tenant_id: str,
) -> List[dict]:
    """
    Execute auto-approved proposals.

    This is "becoming different" — the system transforms itself.
    Returns list of actuation receipts.
    """
    actuations = []

    for proposal in auto_approved_proposals:
        # Record the actuation (the transformation)
        # In a real system, this would execute the actual change
        # Here we emit a receipt recording that actuation occurred
        actuation = emit_receipt("actuation", {
            "tenant_id": tenant_id,
            "cycle_id": cycle_id,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_type": proposal.get("proposal_type"),
            "description": proposal.get("description"),
            "executed": True,
        })
        actuations.append(actuation)

    return actuations


# =============================================================================
# CORE FUNCTION: run_cycle
# =============================================================================

def run_cycle(
    new_receipts: List[dict],
    previous_receipts: List[dict],
    cycle_id: int,
    tenant_id: str,
) -> List[dict]:
    """
    Execute one cycle of the system's metabolism - one "moment" of system consciousness.

    The loop IS time for the system. Each cycle is one "now."

    Args:
        new_receipts: List of new receipts (this cycle's awareness)
        previous_receipts: List of previous receipts (for differential)
        cycle_id: Monotonic counter — the system's "now" number
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        List of all receipts emitted during this cycle

    Algorithm — execute 8 phases in order (THIS IS THE METABOLISM):
        1. SENSE     - Compute receipt differential
        2. MEASURE   - Compute entropy_before
        3. ANALYZE   - Call hunt() from integrity.py
        4. REMEDIATE - Call remediate() from remediate.py
        5. HYPOTHESIZE - Generate proposals
        6. GATE      - Evaluate proposals, determine auto_approved vs escalated
        7. ACTUATE   - Execute auto-approved actions
        8. EMIT      - Record cycle results + entropy measurements
    """
    cycle_start_time = time.time()
    emitted_receipts: List[dict] = []

    # ==========================================================================
    # Phase 1: SENSE - What receipts exist now that didn't before?
    # ==========================================================================
    # Compute receipt differential between new_receipts and previous_receipts
    previous_hashes = set(
        r.get("payload_hash", dual_hash(json.dumps(r, sort_keys=True)))
        for r in previous_receipts
    )

    differential = [
        r for r in new_receipts
        if r.get("payload_hash", dual_hash(json.dumps(r, sort_keys=True))) not in previous_hashes
    ]

    receipts_sensed = len(differential)
    differential_summary = dual_hash(json.dumps([r.get("payload_hash", "") for r in differential], sort_keys=True))

    # ==========================================================================
    # Phase 2: MEASURE - Compute system_entropy() at cycle start
    # ==========================================================================
    entropy_before = system_entropy(previous_receipts)

    # ==========================================================================
    # Phase 3: ANALYZE - Does any of it hurt? Call hunt()
    # ==========================================================================
    anomaly_alerts = hunt(
        receipts=new_receipts,
        cycle_id=cycle_id,
        tenant_id=tenant_id,
    )
    emitted_receipts.extend(anomaly_alerts)
    anomalies_detected = len(anomaly_alerts)

    # ==========================================================================
    # Phase 4: REMEDIATE - Can we heal safely? Call remediate()
    # ==========================================================================
    recovery_receipts = remediate(
        alerts=anomaly_alerts,
        current_receipts=new_receipts,
        cycle_id=cycle_id,
        tenant_id=tenant_id,
    )
    emitted_receipts.extend(recovery_receipts)

    # Count actual recovery_action receipts (not escalations or health receipts)
    remediations_executed = sum(
        1 for r in recovery_receipts if r.get("receipt_type") == "recovery_action"
    )

    # ==========================================================================
    # Phase 5: HYPOTHESIZE - What could be better? Generate proposals
    # ==========================================================================
    # Compute entropy delta so far for proposal generation
    combined_receipts = previous_receipts + new_receipts + emitted_receipts
    entropy_after_partial = system_entropy(combined_receipts)
    entropy_delta_partial = entropy_before - entropy_after_partial

    proposals = _generate_proposals(
        new_receipts=new_receipts,
        anomalies=anomaly_alerts,
        recovery_actions=[r for r in recovery_receipts if r.get("receipt_type") == "recovery_action"],
        entropy_delta=entropy_delta_partial,
        cycle_id=cycle_id,
        tenant_id=tenant_id,
    )
    emitted_receipts.extend(proposals)
    proposals_generated = len(proposals)

    # Check proposal explosion stoprule
    stoprule_proposal(tenant_id, proposals_generated)

    # ==========================================================================
    # Phase 6: GATE - Do we know enough to act? Check confidence threshold
    # ==========================================================================
    gate_decisions, auto_approved_proposals, escalated_proposals = _evaluate_gate(
        proposals=proposals,
        cycle_id=cycle_id,
        tenant_id=tenant_id,
    )
    emitted_receipts.extend(gate_decisions)

    proposals_auto_approved = len(auto_approved_proposals)
    proposals_escalated = len(escalated_proposals)

    # ==========================================================================
    # Phase 7: ACTUATE - Become different. Execute auto-approved actions
    # ==========================================================================
    actuations = _actuate(
        auto_approved_proposals=auto_approved_proposals,
        cycle_id=cycle_id,
        tenant_id=tenant_id,
    )
    emitted_receipts.extend(actuations)

    # ==========================================================================
    # Phase 8: EMIT - Record what we became + entropy measurements
    # ==========================================================================
    # Final entropy measurement
    all_cycle_receipts = previous_receipts + new_receipts + emitted_receipts
    entropy_after = system_entropy(all_cycle_receipts)
    entropy_delta = entropy_before - entropy_after

    # Update consecutive negative deltas tracking
    consecutive_negative_deltas = _update_consecutive_negative_deltas(tenant_id, entropy_delta)

    # Emit entropy_measurement receipt
    entropy_measurement = emit_entropy_measurement(
        tenant_id=tenant_id,
        cycle_id=cycle_id,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        receipt_count=len(all_cycle_receipts),
        consecutive_negative_deltas=consecutive_negative_deltas,
    )
    emitted_receipts.append(entropy_measurement)

    # Check degradation stoprule
    stoprule_entropy_measurement(tenant_id, consecutive_negative_deltas)

    # Calculate cycle duration
    cycle_end_time = time.time()
    duration_ms = int((cycle_end_time - cycle_start_time) * 1000)

    # Determine health status
    health_status = determine_health_status(
        entropy_delta=entropy_delta,
        consecutive_negative_deltas=consecutive_negative_deltas,
        anomalies=anomaly_alerts,
    )

    # Emit unified_loop_receipt
    resource_consumed = {
        "compute_used": duration_ms / (CYCLE_INTERVAL_SECONDS * 1000),  # Fraction of cycle budget
        "memory_used": len(emitted_receipts) / 1000,  # Estimate based on receipts
        "io_operations": len(emitted_receipts),  # Each receipt is an I/O
        "cycle_duration_ms": duration_ms,
    }

    unified_loop_receipt = emit_unified_loop_receipt(
        tenant_id=tenant_id,
        cycle_id=cycle_id,
        duration_ms=duration_ms,
        receipts_sensed=receipts_sensed,
        anomalies_detected=anomalies_detected,
        remediations_executed=remediations_executed,
        proposals_generated=proposals_generated,
        proposals_auto_approved=proposals_auto_approved,
        proposals_escalated=proposals_escalated,
        health_status=health_status,
        differential_summary=differential_summary,
        entropy_delta=entropy_delta,
        resource_consumed=resource_consumed,
    )
    emitted_receipts.append(unified_loop_receipt)

    return emitted_receipts


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "CYCLE_INTERVAL_SECONDS",
    "CONFIDENCE_THRESHOLD",
    "DEGRADATION_THRESHOLD",
    "CYCLE_COMPLETION_SLO",
    "PROPOSAL_TYPES",
    "RISK_LEVELS",
    "GATE_DECISIONS",
    "HEALTH_STATUS_LEVELS",
    # Core function
    "run_cycle",
    # Emit functions
    "emit_unified_loop_receipt",
    "emit_gate_decision",
    "emit_proposal",
    "emit_entropy_measurement",
    # Stoprules
    "stoprule_unified_loop_receipt",
    "stoprule_proposal",
    "stoprule_entropy_measurement",
    # Utilities
    "determine_health_status",
    # Testing utilities
    "_reset_consecutive_negative_deltas",
    "_get_consecutive_negative_deltas",
]
