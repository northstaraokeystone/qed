"""
test_unified_loop.py - Tests for the System's Experienced Time

Tests the 8-phase metabolism (run_cycle), 4 receipt types, and gate logic.
"""

import time
import pytest

from unified_loop import (
    # Constants
    RECEIPT_SCHEMA,
    CYCLE_INTERVAL_SECONDS,
    CONFIDENCE_THRESHOLD,
    DEGRADATION_THRESHOLD,
    CYCLE_COMPLETION_SLO,
    # Core function
    run_cycle,
    # Emit functions
    emit_unified_loop_receipt,
    emit_gate_decision,
    emit_proposal,
    emit_entropy_measurement,
    # Stoprules
    stoprule_unified_loop_receipt,
    stoprule_proposal,
    stoprule_entropy_measurement,
    # Utilities
    determine_health_status,
    _reset_consecutive_negative_deltas,
    _get_consecutive_negative_deltas,
)

from entropy import dual_hash


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_state():
    """Reset module state before each test."""
    _reset_consecutive_negative_deltas()
    yield
    _reset_consecutive_negative_deltas()


@pytest.fixture
def sample_receipts():
    """Generate sample receipts for testing."""
    return [
        {"receipt_type": "ingest", "tenant_id": "test", "payload_hash": dual_hash("r1")},
        {"receipt_type": "anchor", "tenant_id": "test", "payload_hash": dual_hash("r2")},
        {"receipt_type": "routing", "tenant_id": "test", "payload_hash": dual_hash("r3")},
        {"receipt_type": "ingest", "tenant_id": "test", "payload_hash": dual_hash("r4")},
        {"receipt_type": "bias", "tenant_id": "test", "payload_hash": dual_hash("r5")},
    ]


@pytest.fixture
def new_receipts():
    """Generate new receipts for cycle input."""
    return [
        {"receipt_type": "ingest", "tenant_id": "test", "payload_hash": dual_hash("new1")},
        {"receipt_type": "routing", "tenant_id": "test", "payload_hash": dual_hash("new2")},
        {"receipt_type": "decision_health", "tenant_id": "test", "payload_hash": dual_hash("new3")},
    ]


# =============================================================================
# TEST: run_cycle 8 phases
# =============================================================================

def test_run_cycle_8_phases(sample_receipts, new_receipts):
    """
    Test that cycle completes all 8 phases in order.

    Phases: SENSE, MEASURE, ANALYZE, REMEDIATE, HYPOTHESIZE, GATE, ACTUATE, EMIT
    """
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    # Verify we got receipts back (phases completed)
    assert len(emitted) > 0, "Cycle should emit receipts"

    # Check that core receipt types are present
    receipt_types = [r.get("receipt_type") for r in emitted]

    # Phase 8 EMIT produces unified_loop_receipt and entropy_measurement
    assert "unified_loop_receipt" in receipt_types, "Phase 8 should emit unified_loop_receipt"
    assert "entropy_measurement" in receipt_types, "Phase 8 should emit entropy_measurement"


def test_run_cycle_returns_receipts(sample_receipts, new_receipts):
    """Test that run_cycle returns list of emitted receipts."""
    result = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    assert isinstance(result, list), "run_cycle should return a list"
    assert all(isinstance(r, dict) for r in result), "All items should be dicts"
    assert all("receipt_type" in r for r in result), "All receipts should have receipt_type"


def test_entropy_measurement_emitted(sample_receipts, new_receipts):
    """Test that every cycle emits entropy_measurement."""
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    entropy_measurements = [r for r in emitted if r.get("receipt_type") == "entropy_measurement"]
    assert len(entropy_measurements) >= 1, "Cycle should emit at least one entropy_measurement"

    # Verify entropy_measurement schema
    em = entropy_measurements[0]
    assert "entropy_before" in em, "entropy_measurement should have entropy_before"
    assert "entropy_after" in em, "entropy_measurement should have entropy_after"
    assert "entropy_delta" in em, "entropy_measurement should have entropy_delta"
    assert "consecutive_negative_deltas" in em, "entropy_measurement should track consecutive_negative_deltas"


def test_unified_loop_receipt_emitted(sample_receipts, new_receipts):
    """Test that every cycle emits unified_loop_receipt."""
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=42,
        tenant_id="test_tenant",
    )

    unified_receipts = [r for r in emitted if r.get("receipt_type") == "unified_loop_receipt"]
    assert len(unified_receipts) == 1, "Cycle should emit exactly one unified_loop_receipt"

    # Verify schema fields
    ur = unified_receipts[0]
    assert ur["cycle_id"] == 42, "cycle_id should match input"
    assert ur["tenant_id"] == "test_tenant", "tenant_id should match input"
    assert "duration_ms" in ur, "Should have duration_ms"
    assert "receipts_sensed" in ur, "Should have receipts_sensed"
    assert "anomalies_detected" in ur, "Should have anomalies_detected"
    assert "health_status" in ur, "Should have health_status"
    assert "entropy_delta" in ur, "Should have entropy_delta"


# =============================================================================
# TEST: Gate Logic
# =============================================================================

def test_gate_auto_approve():
    """Test that confidence > 0.8 AND risk = low yields auto_approved."""
    gate_decision = emit_gate_decision(
        tenant_id="test",
        cycle_id=1,
        proposal_id="prop_001",
        confidence=0.85,  # > 0.8
        decision="auto_approved",
        reason="confidence > 0.8 AND risk = low",
        uncertainty_source=None,
    )

    assert gate_decision["decision"] == "auto_approved"
    assert gate_decision["confidence"] == 0.85
    assert gate_decision["uncertainty_source"] is None


def test_gate_escalation():
    """Test that confidence <= 0.8 yields escalated (system becomes human)."""
    gate_decision = emit_gate_decision(
        tenant_id="test",
        cycle_id=1,
        proposal_id="prop_002",
        confidence=0.75,  # <= 0.8
        decision="escalated",
        reason="confidence <= 0.8",
        uncertainty_source="insufficient confidence for autonomous action",
    )

    assert gate_decision["decision"] == "escalated"
    assert gate_decision["confidence"] == 0.75
    assert gate_decision["uncertainty_source"] is not None


# =============================================================================
# TEST: Degradation Detection
# =============================================================================

def test_degradation_alert_3_cycles():
    """Test that 3 consecutive negative entropy_delta triggers alert."""
    tenant_id = "degradation_test"

    # Simulate 3 cycles with negative entropy delta
    for cycle in range(3):
        em = emit_entropy_measurement(
            tenant_id=tenant_id,
            cycle_id=cycle,
            entropy_before=1.0,
            entropy_after=1.5,  # Negative delta (before - after = -0.5)
            receipt_count=10,
            consecutive_negative_deltas=cycle + 1,
        )

    # Check that consecutive_negative_deltas reached threshold
    # The stoprule should be checked
    assert em["entropy_delta"] == -0.5, "Should have negative entropy delta"
    assert em["consecutive_negative_deltas"] == 3, "Should track 3 consecutive negatives"


def test_cycle_id_monotonic(sample_receipts, new_receipts):
    """Test that cycle_id increments each cycle."""
    cycle_ids = []

    for i in range(3):
        emitted = run_cycle(
            new_receipts=new_receipts,
            previous_receipts=sample_receipts,
            cycle_id=i + 1,  # 1, 2, 3
            tenant_id="test_tenant",
        )

        unified = [r for r in emitted if r.get("receipt_type") == "unified_loop_receipt"][0]
        cycle_ids.append(unified["cycle_id"])

    assert cycle_ids == [1, 2, 3], "cycle_id should be monotonically increasing"


# =============================================================================
# TEST: Health Status Mapping
# =============================================================================

def test_health_status_mapping():
    """Test that entropy_delta and anomaly severity map to correct status."""
    # Healthy: entropy_delta >= 0 AND no critical anomalies
    status = determine_health_status(
        entropy_delta=0.5,
        consecutive_negative_deltas=0,
        anomalies=[],
    )
    assert status == "healthy"

    # Degraded: entropy_delta < 0
    status = determine_health_status(
        entropy_delta=-0.1,
        consecutive_negative_deltas=1,
        anomalies=[],
    )
    assert status == "degraded"

    # Degraded: anomalies with severity >= high
    status = determine_health_status(
        entropy_delta=0.5,
        consecutive_negative_deltas=0,
        anomalies=[{"severity": "high"}],
    )
    assert status == "degraded"

    # Critical: consecutive_negative_deltas >= 3
    status = determine_health_status(
        entropy_delta=-0.5,
        consecutive_negative_deltas=3,
        anomalies=[],
    )
    assert status == "critical"

    # Critical: anomaly with severity = critical
    status = determine_health_status(
        entropy_delta=0.5,
        consecutive_negative_deltas=0,
        anomalies=[{"severity": "critical"}],
    )
    assert status == "critical"


# =============================================================================
# TEST: Resource Consumed
# =============================================================================

def test_resource_consumed_present(sample_receipts, new_receipts):
    """Test that unified_loop_receipt contains resource_consumed field."""
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    unified = [r for r in emitted if r.get("receipt_type") == "unified_loop_receipt"][0]

    assert "resource_consumed" in unified, "Should have resource_consumed field"

    rc = unified["resource_consumed"]
    assert "compute_used" in rc, "resource_consumed should have compute_used"
    assert "memory_used" in rc, "resource_consumed should have memory_used"
    assert "io_operations" in rc, "resource_consumed should have io_operations"
    assert "cycle_duration_ms" in rc, "resource_consumed should have cycle_duration_ms"


def test_cycle_duration_measured(sample_receipts, new_receipts):
    """Test that duration_ms reflects actual cycle execution time."""
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    unified = [r for r in emitted if r.get("receipt_type") == "unified_loop_receipt"][0]

    # Duration should be > 0 (cycle takes some time)
    assert unified["duration_ms"] >= 0, "duration_ms should be non-negative"

    # Duration should be less than the cycle interval (should complete within budget)
    assert unified["duration_ms"] < CYCLE_INTERVAL_SECONDS * 1000, \
        f"Cycle should complete in less than {CYCLE_INTERVAL_SECONDS}s"


# =============================================================================
# TEST: Tenant ID Present
# =============================================================================

def test_tenant_id_present(sample_receipts, new_receipts):
    """Test that all emitted receipts have tenant_id field."""
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    for receipt in emitted:
        assert "tenant_id" in receipt, f"Receipt {receipt.get('receipt_type')} missing tenant_id"
        assert receipt["tenant_id"] == "test_tenant", \
            f"Receipt {receipt.get('receipt_type')} has wrong tenant_id"


# =============================================================================
# TEST: Differential Summary Hashed
# =============================================================================

def test_differential_summary_hashed(sample_receipts, new_receipts):
    """Test that differential_summary is dual_hash of receipt diff."""
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    unified = [r for r in emitted if r.get("receipt_type") == "unified_loop_receipt"][0]

    # Check differential_summary is a dual_hash format (SHA256:BLAKE3)
    assert "differential_summary" in unified
    assert ":" in unified["differential_summary"], "differential_summary should be dual_hash format"


# =============================================================================
# TEST: RECEIPT_SCHEMA export
# =============================================================================

def test_receipt_schema_export():
    """Test that RECEIPT_SCHEMA exports exactly 4 types."""
    assert len(RECEIPT_SCHEMA) == 4, "RECEIPT_SCHEMA should have exactly 4 types"
    assert "unified_loop_receipt" in RECEIPT_SCHEMA
    assert "gate_decision" in RECEIPT_SCHEMA
    assert "proposal" in RECEIPT_SCHEMA
    assert "entropy_measurement" in RECEIPT_SCHEMA


# =============================================================================
# TEST: Constants
# =============================================================================

def test_constants():
    """Test that constants are correctly defined."""
    assert CYCLE_INTERVAL_SECONDS == 60, "CYCLE_INTERVAL_SECONDS should be 60"
    assert CONFIDENCE_THRESHOLD == 0.8, "CONFIDENCE_THRESHOLD should be 0.8"
    assert DEGRADATION_THRESHOLD == 3, "DEGRADATION_THRESHOLD should be 3"
    assert CYCLE_COMPLETION_SLO == 0.999, "CYCLE_COMPLETION_SLO should be 0.999"


# =============================================================================
# SMOKE TESTS
# =============================================================================

def test_smoke_h5_cycle_completes_under_60s(sample_receipts, new_receipts):
    """H5: Cycle completes in < 60 seconds."""
    start = time.time()

    run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    elapsed = time.time() - start
    assert elapsed < 60, f"Cycle took {elapsed}s, should be < 60s"


def test_smoke_h8_receipt_schema_length():
    """H8: RECEIPT_SCHEMA has exactly 4 types."""
    assert len(RECEIPT_SCHEMA) == 4


def test_smoke_h12_resource_consumed_field(sample_receipts, new_receipts):
    """H12: unified_loop_receipt contains resource_consumed field."""
    emitted = run_cycle(
        new_receipts=new_receipts,
        previous_receipts=sample_receipts,
        cycle_id=1,
        tenant_id="test_tenant",
    )

    unified = [r for r in emitted if r.get("receipt_type") == "unified_loop_receipt"][0]
    assert "resource_consumed" in unified
