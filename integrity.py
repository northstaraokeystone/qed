"""
integrity.py - HUNTER (System Proprioception)

HUNTER is NOT an agent that detects anomalies - HUNTER IS the system's capacity
TO feel. When you call hunt(), you ask "what does the system feel right now?"
Anomaly alerts are sensations, not reports.

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.

Critical Constraint: Self-Reference Exclusion
HUNTER cannot feel itself feeling. If HUNTER processes its own anomaly_alert
receipts, infinite regress occurs. The hunt() function MUST filter out receipts
where receipt_type equals "anomaly_alert" or "detection_cycle" or "hunter_health"
before processing.
"""

import json
import time
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional, Tuple

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

# HUNTER identifies itself with this agent_id
HUNTER_SELF_ID = "hunter"

# Receipt types to exclude from processing (self-reference prevention)
SELF_RECEIPT_TYPES = ["anomaly_alert", "detection_cycle", "hunter_health"]

# Module exports for receipt types
RECEIPT_SCHEMA = ["anomaly_alert", "detection_cycle", "hunter_health"]

# Anomaly type taxonomy (exactly 5 types)
ANOMALY_TYPES = [
    "drift",
    "degradation",
    "constraint_violation",
    "pattern_deviation",
    "emergent_anti_pattern",
]

# Severity levels
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# SLO thresholds per CLAUDEME section 6
SLO_THRESHOLDS = {
    "latency_ms": 50,
    "entanglement": 0.92,
    "forgetting_rate": 0.01,
    "bias_disparity": 0.005,
    "acceptance_rate": 0.95,
    "fusion_match": 0.999,
}

# Detection parameters
DRIFT_WINDOW_COUNT = 3  # Number of windows for drift detection
DEGRADATION_THRESHOLD = 0.5  # Bits above baseline for degradation
PATTERN_DEVIATION_THRESHOLD = 0.3  # Probability threshold for deviation
MAX_SCAN_DURATION_MS = 60000  # 60 seconds max scan time


# =============================================================================
# SEVERITY MAPPING
# =============================================================================

def severity_from_entropy_spike(entropy_spike: float) -> str:
    """
    Map entropy spike magnitude to severity level.

    Severity mapping:
        - critical: entropy_spike > 2.0 bits
        - high: 1.0 < entropy_spike <= 2.0 bits
        - medium: 0.5 < entropy_spike <= 1.0 bits
        - low: entropy_spike <= 0.5 bits
    """
    if entropy_spike > 2.0:
        return "critical"
    elif entropy_spike > 1.0:
        return "high"
    elif entropy_spike > 0.5:
        return "medium"
    else:
        return "low"


# =============================================================================
# RECEIPT TYPE 1: anomaly_alert
# =============================================================================

# --- SCHEMA ---
ANOMALY_ALERT_SCHEMA = {
    "receipt_type": "anomaly_alert",
    "ts": "ISO8601",
    "tenant_id": "str",  # REQUIRED per CLAUDEME 4.1:137
    "agent_id": "str",  # always "hunter"
    "anomaly_type": "enum[drift|degradation|constraint_violation|pattern_deviation|emergent_anti_pattern]",
    "severity": "enum[low|medium|high|critical]",
    "blast_radius": "float 0.0-1.0",
    "confidence": "float 0.0-1.0",
    "evidence": ["receipt_id"],
    "differential_hash": "str",  # dual_hash of triggering differential
    "entropy_spike": "float",
    "payload_hash": "str",
}


# --- EMIT ---
def emit_anomaly_alert(
    tenant_id: str,
    anomaly_type: str,
    severity: str,
    blast_radius: float,
    confidence: float,
    evidence: List[str],
    differential_hash: str,
    entropy_spike: float,
) -> dict:
    """
    Emit anomaly_alert receipt - the system feeling pain.

    This IS the first remediation step - detection reduces uncertainty.
    """
    return emit_receipt("anomaly_alert", {
        "tenant_id": tenant_id,
        "agent_id": HUNTER_SELF_ID,
        "anomaly_type": anomaly_type,
        "severity": severity,
        "blast_radius": blast_radius,
        "confidence": confidence,
        "evidence": evidence,
        "differential_hash": differential_hash,
        "entropy_spike": entropy_spike,
    })


# --- TEST ---
def test_anomaly_alert():
    """Test anomaly_alert receipt emission."""
    r = emit_anomaly_alert(
        tenant_id="test_tenant",
        anomaly_type="drift",
        severity="medium",
        blast_radius=0.3,
        confidence=0.85,
        evidence=["receipt_001", "receipt_002"],
        differential_hash=dual_hash("test_diff"),
        entropy_spike=0.75,
    )
    assert r["receipt_type"] == "anomaly_alert"
    assert r["tenant_id"] == "test_tenant"
    assert r["agent_id"] == HUNTER_SELF_ID
    assert r["anomaly_type"] == "drift"
    assert r["severity"] == "medium"
    assert "payload_hash" in r
    assert ":" in r["payload_hash"]  # dual_hash format


# --- STOPRULE ---
def stoprule_anomaly_alert(tenant_id: str, severity: str, confidence: float) -> None:
    """
    When severity equals "critical" AND confidence > 0.9, emit anomaly_receipt
    per CLAUDEME section 4.7 with classification="violation" and action="halt",
    then raise StopRule.
    """
    if severity == "critical" and confidence > 0.9:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "anomaly_alert",
            "baseline": 0.0,
            "delta": confidence,
            "classification": "violation",
            "action": "halt",
        })
        raise StopRule(f"Critical anomaly detected with confidence {confidence} for {tenant_id}")


# =============================================================================
# RECEIPT TYPE 2: detection_cycle
# =============================================================================

# --- SCHEMA ---
DETECTION_CYCLE_SCHEMA = {
    "receipt_type": "detection_cycle",
    "ts": "ISO8601",
    "tenant_id": "str",
    "cycle_id": "int",
    "receipts_scanned": "int",
    "receipts_excluded": "int",
    "anomalies_found": "int",
    "baseline_entropy": "float",
    "scan_duration_ms": "int",
    "payload_hash": "str",
}


# --- EMIT ---
def emit_detection_cycle(
    tenant_id: str,
    cycle_id: int,
    receipts_scanned: int,
    receipts_excluded: int,
    anomalies_found: int,
    baseline_entropy: float,
    scan_duration_ms: int,
) -> dict:
    """
    Emit detection_cycle receipt - hunt heartbeat.

    Emitted EVERY hunt() call, even when no anomalies found.
    Proves HUNTER is alive and sensing.
    """
    return emit_receipt("detection_cycle", {
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "receipts_scanned": receipts_scanned,
        "receipts_excluded": receipts_excluded,
        "anomalies_found": anomalies_found,
        "baseline_entropy": baseline_entropy,
        "scan_duration_ms": scan_duration_ms,
    })


# --- TEST ---
def test_detection_cycle():
    """Test detection_cycle receipt emission."""
    r = emit_detection_cycle(
        tenant_id="test_tenant",
        cycle_id=42,
        receipts_scanned=100,
        receipts_excluded=5,
        anomalies_found=2,
        baseline_entropy=2.5,
        scan_duration_ms=150,
    )
    assert r["receipt_type"] == "detection_cycle"
    assert r["tenant_id"] == "test_tenant"
    assert r["cycle_id"] == 42
    assert r["receipts_scanned"] == 100
    assert r["receipts_excluded"] == 5
    assert "payload_hash" in r


# --- STOPRULE ---
def stoprule_detection_cycle(tenant_id: str, scan_duration_ms: int) -> None:
    """
    If scan_duration_ms exceeds 60000 (60 seconds), emit anomaly
    with metric="hunter_latency" and action="alert".
    """
    if scan_duration_ms > MAX_SCAN_DURATION_MS:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "hunter_latency",
            "baseline": MAX_SCAN_DURATION_MS,
            "delta": scan_duration_ms - MAX_SCAN_DURATION_MS,
            "classification": "violation",
            "action": "alert",
        })


# =============================================================================
# RECEIPT TYPE 3: hunter_health
# =============================================================================

# --- SCHEMA ---
HUNTER_HEALTH_SCHEMA = {
    "receipt_type": "hunter_health",
    "ts": "ISO8601",
    "tenant_id": "str",
    "status": "enum[healthy|degraded|impaired]",
    "detection_rate": "float",  # anomalies per cycle, rolling average
    "false_positive_estimate": "float 0.0-1.0",
    "coverage": "float 0.0-1.0",  # portion of receipt types HUNTER can evaluate
    "last_detection_ts": "ISO8601|null",
    "payload_hash": "str",
}


# --- EMIT ---
def emit_hunter_health(
    tenant_id: str,
    status: str,
    detection_rate: float,
    false_positive_estimate: float,
    coverage: float,
    last_detection_ts: Optional[str],
) -> dict:
    """
    Emit hunter_health receipt - HUNTER's own vital signs.

    Meta-receipt about HUNTER's operational state. Emitted periodically
    or when HUNTER detects degradation in its own capacity.
    """
    return emit_receipt("hunter_health", {
        "tenant_id": tenant_id,
        "status": status,
        "detection_rate": detection_rate,
        "false_positive_estimate": false_positive_estimate,
        "coverage": coverage,
        "last_detection_ts": last_detection_ts,
    })


# --- TEST ---
def test_hunter_health():
    """Test hunter_health receipt emission."""
    r = emit_hunter_health(
        tenant_id="test_tenant",
        status="healthy",
        detection_rate=0.05,
        false_positive_estimate=0.02,
        coverage=0.95,
        last_detection_ts="2024-01-01T12:00:00Z",
    )
    assert r["receipt_type"] == "hunter_health"
    assert r["tenant_id"] == "test_tenant"
    assert r["status"] == "healthy"
    assert "payload_hash" in r


# --- STOPRULE ---
def stoprule_hunter_health(tenant_id: str, status: str) -> None:
    """
    If status equals "impaired", emit anomaly with classification="degradation"
    and action="escalate".
    """
    if status == "impaired":
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "hunter_health",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "degradation",
            "action": "escalate",
        })


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def _extract_receipt_id(receipt: dict) -> str:
    """Extract or compute a receipt ID for evidence tracking."""
    if "id" in receipt:
        return receipt["id"]
    if "payload_hash" in receipt:
        return receipt["payload_hash"][:16]
    return dual_hash(json.dumps(receipt, sort_keys=True))[:16]


def _compute_sliding_window_entropy(receipts: List[dict], window_size: int) -> List[float]:
    """Compute entropy over sliding windows of receipts."""
    if len(receipts) < window_size:
        return [system_entropy(receipts)] if receipts else [0.0]

    entropies = []
    for i in range(len(receipts) - window_size + 1):
        window = receipts[i:i + window_size]
        entropies.append(system_entropy(window))
    return entropies


def detect_drift(receipts: List[dict], baseline_entropy: float) -> Optional[dict]:
    """
    Detect drift: sustained directional change over time.

    Gradient of entropy sustained positive over multiple windows.
    """
    if len(receipts) < DRIFT_WINDOW_COUNT * 2:
        return None

    window_size = max(len(receipts) // DRIFT_WINDOW_COUNT, 2)
    entropies = _compute_sliding_window_entropy(receipts, window_size)

    if len(entropies) < DRIFT_WINDOW_COUNT:
        return None

    # Check for sustained positive gradient (entropy increasing)
    gradients = [entropies[i + 1] - entropies[i] for i in range(len(entropies) - 1)]
    recent_gradients = gradients[-(DRIFT_WINDOW_COUNT - 1):]

    if all(g > 0 for g in recent_gradients):
        entropy_spike = sum(recent_gradients)
        evidence = [_extract_receipt_id(r) for r in receipts[-window_size:]]
        return {
            "anomaly_type": "drift",
            "entropy_spike": entropy_spike,
            "evidence": evidence,
            "confidence": min(0.95, 0.7 + entropy_spike * 0.1),
            "blast_radius": min(1.0, len(receipts) / 100),
        }
    return None


def detect_degradation(
    receipts: List[dict],
    current_entropy: float,
    baseline_entropy: float,
) -> Optional[dict]:
    """
    Detect degradation: system getting worse.

    Current entropy significantly above rolling baseline.
    """
    if current_entropy <= baseline_entropy:
        return None

    entropy_spike = current_entropy - baseline_entropy
    if entropy_spike < DEGRADATION_THRESHOLD:
        return None

    evidence = [_extract_receipt_id(r) for r in receipts[-10:]]
    return {
        "anomaly_type": "degradation",
        "entropy_spike": entropy_spike,
        "evidence": evidence,
        "confidence": min(0.95, 0.6 + entropy_spike * 0.15),
        "blast_radius": min(1.0, entropy_spike / 3.0),
    }


def detect_constraint_violation(receipts: List[dict]) -> List[dict]:
    """
    Detect constraint violation: SLO or bound exceeded.

    Receipt field exceeds threshold per CLAUDEME section 6.
    """
    violations = []

    for receipt in receipts:
        violation_found = False
        evidence = [_extract_receipt_id(receipt)]

        # Check latency SLO
        if "latency_ms" in receipt or "scan_duration_ms" in receipt:
            latency = receipt.get("latency_ms", receipt.get("scan_duration_ms", 0))
            if latency > SLO_THRESHOLDS["latency_ms"] * 10:  # 10x SLO = violation
                violation_found = True
                entropy_spike = latency / SLO_THRESHOLDS["latency_ms"] / 10

        # Check entanglement SLO
        if "entanglement_score" in receipt or "entanglement" in receipt:
            score = receipt.get("entanglement_score", receipt.get("entanglement", 1.0))
            if score < SLO_THRESHOLDS["entanglement"]:
                violation_found = True
                entropy_spike = (SLO_THRESHOLDS["entanglement"] - score) * 10

        # Check bias disparity SLO
        if "disparity" in receipt:
            disparity = receipt.get("disparity", 0)
            if disparity >= SLO_THRESHOLDS["bias_disparity"]:
                violation_found = True
                entropy_spike = disparity / SLO_THRESHOLDS["bias_disparity"]

        # Check forgetting rate SLO
        if "forgetting_rate" in receipt:
            rate = receipt.get("forgetting_rate", 0)
            if rate >= SLO_THRESHOLDS["forgetting_rate"]:
                violation_found = True
                entropy_spike = rate / SLO_THRESHOLDS["forgetting_rate"]

        if violation_found:
            violations.append({
                "anomaly_type": "constraint_violation",
                "entropy_spike": entropy_spike,
                "evidence": evidence,
                "confidence": 0.99,  # SLO violations are high confidence
                "blast_radius": 0.1,  # Single receipt impact
            })

    return violations


def detect_pattern_deviation(receipts: List[dict], baseline_distribution: dict) -> Optional[dict]:
    """
    Detect pattern deviation: unexpected distribution.

    Receipt type frequencies deviate from expected.
    """
    if not receipts or not baseline_distribution:
        return None

    # Compute current distribution
    type_counts = Counter(r.get("receipt_type", "unknown") for r in receipts)
    total = sum(type_counts.values())
    if total == 0:
        return None

    current_dist = {k: v / total for k, v in type_counts.items()}

    # Simple deviation check: look for types with significant difference
    max_deviation = 0.0
    deviating_types = []

    for receipt_type, expected_prob in baseline_distribution.items():
        actual_prob = current_dist.get(receipt_type, 0.0)
        deviation = abs(actual_prob - expected_prob)
        if deviation > PATTERN_DEVIATION_THRESHOLD:
            max_deviation = max(max_deviation, deviation)
            deviating_types.append(receipt_type)

    # Check for new types not in baseline
    for receipt_type, actual_prob in current_dist.items():
        if receipt_type not in baseline_distribution and actual_prob > PATTERN_DEVIATION_THRESHOLD:
            max_deviation = max(max_deviation, actual_prob)
            deviating_types.append(receipt_type)

    if max_deviation > PATTERN_DEVIATION_THRESHOLD:
        evidence = [
            _extract_receipt_id(r) for r in receipts
            if r.get("receipt_type") in deviating_types
        ][:10]  # Limit evidence size
        return {
            "anomaly_type": "pattern_deviation",
            "entropy_spike": max_deviation * 3,  # Scale deviation to entropy-like value
            "evidence": evidence,
            "confidence": min(0.9, 0.5 + max_deviation),
            "blast_radius": min(1.0, len(deviating_types) / 5),
        }
    return None


def detect_emergent_anti_pattern(
    receipts: List[dict],
    known_receipt_types: set,
) -> List[dict]:
    """
    Detect emergent anti-pattern: new unknown behavior.

    Previously unseen receipt_type appears.
    """
    anti_patterns = []

    # Find new receipt types
    current_types = set(r.get("receipt_type", "unknown") for r in receipts)
    new_types = current_types - known_receipt_types

    for new_type in new_types:
        if new_type == "unknown":
            continue

        # Find receipts of this new type
        new_type_receipts = [r for r in receipts if r.get("receipt_type") == new_type]
        evidence = [_extract_receipt_id(r) for r in new_type_receipts[:5]]

        anti_patterns.append({
            "anomaly_type": "emergent_anti_pattern",
            "entropy_spike": 1.0,  # New types add 1 bit of uncertainty
            "evidence": evidence,
            "confidence": 0.7,  # New patterns need investigation
            "blast_radius": min(1.0, len(new_type_receipts) / len(receipts)),
        })

    return anti_patterns


# =============================================================================
# CORE FUNCTION: hunt
# =============================================================================

def hunt(
    receipts: List[dict],
    cycle_id: int,
    tenant_id: str,
    baseline_distribution: Optional[dict] = None,
    known_receipt_types: Optional[set] = None,
    historical_baseline: Optional[float] = None,
) -> List[dict]:
    """
    Ask "what does the system feel right now?"

    HUNTER is the system's capacity to feel. Anomaly alerts are sensations.

    Args:
        receipts: List of receipt dicts to analyze
        cycle_id: Current unified_loop cycle ID
        tenant_id: Tenant identifier (required per CLAUDEME)
        baseline_distribution: Expected receipt type distribution (optional)
        known_receipt_types: Set of known receipt types (optional)
        historical_baseline: Historical entropy baseline (optional)

    Returns:
        List of anomaly_alert receipts (may be empty)

    Algorithm:
        1. Filter self-reference receipts
        2. Compute baseline entropy
        3. Detect by taxonomy (5 types)
        4. Calculate severity from entropy spike
        5. Emit detection_cycle receipt (ALWAYS)
        6. Return anomaly_alert receipts
    """
    start_time = time.time()

    # Step 1: Filter self-reference (CRITICAL - prevents infinite regress)
    original_count = len(receipts)
    filtered_receipts = [
        r for r in receipts
        if r.get("receipt_type") not in SELF_RECEIPT_TYPES
    ]
    excluded_count = original_count - len(filtered_receipts)

    # Step 2: Compute baseline entropy
    baseline_entropy = system_entropy(filtered_receipts)

    # Use historical baseline if provided, otherwise use current
    reference_entropy = historical_baseline if historical_baseline is not None else baseline_entropy

    # Initialize detection context
    if known_receipt_types is None:
        # Default known types from the codebase
        known_receipt_types = {
            "ingest", "anchor", "routing", "bias", "decision_health",
            "impact", "anomaly", "compaction", "entropy_measurement",
            "fitness_score", "selection_event",
        }

    if baseline_distribution is None:
        # Build baseline from filtered receipts or use uniform
        if filtered_receipts:
            type_counts = Counter(r.get("receipt_type", "unknown") for r in filtered_receipts)
            total = sum(type_counts.values())
            baseline_distribution = {k: v / total for k, v in type_counts.items()}
        else:
            baseline_distribution = {}

    # Step 3: Detect by taxonomy
    anomalies = []

    # 3a. Drift detection
    drift_result = detect_drift(filtered_receipts, baseline_entropy)
    if drift_result:
        anomalies.append(drift_result)

    # 3b. Degradation detection
    current_entropy = system_entropy(filtered_receipts)
    degradation_result = detect_degradation(
        filtered_receipts, current_entropy, reference_entropy
    )
    if degradation_result:
        anomalies.append(degradation_result)

    # 3c. Constraint violation detection
    violations = detect_constraint_violation(filtered_receipts)
    anomalies.extend(violations)

    # 3d. Pattern deviation detection
    deviation_result = detect_pattern_deviation(filtered_receipts, baseline_distribution)
    if deviation_result:
        anomalies.append(deviation_result)

    # 3e. Emergent anti-pattern detection
    anti_patterns = detect_emergent_anti_pattern(filtered_receipts, known_receipt_types)
    anomalies.extend(anti_patterns)

    # Step 4 & 6: Calculate severity and emit anomaly_alert receipts
    anomaly_receipts = []
    for anomaly in anomalies:
        severity = severity_from_entropy_spike(anomaly["entropy_spike"])
        differential_data = json.dumps(anomaly, sort_keys=True)

        receipt = emit_anomaly_alert(
            tenant_id=tenant_id,
            anomaly_type=anomaly["anomaly_type"],
            severity=severity,
            blast_radius=anomaly["blast_radius"],
            confidence=anomaly["confidence"],
            evidence=anomaly["evidence"],
            differential_hash=dual_hash(differential_data),
            entropy_spike=anomaly["entropy_spike"],
        )
        anomaly_receipts.append(receipt)

        # Check stoprule for critical anomalies
        stoprule_anomaly_alert(tenant_id, severity, anomaly["confidence"])

    # Step 5: Emit detection_cycle receipt (ALWAYS)
    scan_duration_ms = int((time.time() - start_time) * 1000)
    emit_detection_cycle(
        tenant_id=tenant_id,
        cycle_id=cycle_id,
        receipts_scanned=len(filtered_receipts),
        receipts_excluded=excluded_count,
        anomalies_found=len(anomaly_receipts),
        baseline_entropy=baseline_entropy,
        scan_duration_ms=scan_duration_ms,
    )

    # Check detection_cycle stoprule
    stoprule_detection_cycle(tenant_id, scan_duration_ms)

    return anomaly_receipts


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "HUNTER_SELF_ID",
    "SELF_RECEIPT_TYPES",
    "ANOMALY_TYPES",
    "SEVERITY_LEVELS",
    "SLO_THRESHOLDS",
    # Core function
    "hunt",
    # Emit functions
    "emit_anomaly_alert",
    "emit_detection_cycle",
    "emit_hunter_health",
    # Stoprules
    "stoprule_anomaly_alert",
    "stoprule_detection_cycle",
    "stoprule_hunter_health",
    # Detection functions
    "detect_drift",
    "detect_degradation",
    "detect_constraint_violation",
    "detect_pattern_deviation",
    "detect_emergent_anti_pattern",
    # Utilities
    "severity_from_entropy_spike",
]
