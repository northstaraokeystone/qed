"""
tests/test_wound.py - Tests for wound.py

CLAUDEME v3.1 Compliant: Every test has assert statements.
"""

import pytest


class TestEmitWound:
    """Tests for emit_wound function."""

    def test_emit_wound_returns_valid_receipt(self):
        """emit_wound returns valid wound_receipt with all required fields."""
        from wound import emit_wound

        intervention = {
            "duration_ms": 600000,
            "action": "restarted the service",
            "context": "service was unresponsive",
            "operator_id": "operator_001",
            "tenant_id": "test_tenant"
        }
        r = emit_wound(intervention)

        assert r["receipt_type"] == "wound_receipt"
        assert "ts" in r
        assert r["tenant_id"] == "test_tenant"
        assert "intervention_id" in r
        assert "problem_type" in r
        assert "time_to_resolve_ms" in r
        assert "resolution_action" in r
        assert "could_automate" in r
        assert "operator_id" in r
        assert "context_hash" in r
        assert "payload_hash" in r

    def test_emit_wound_has_problem_type(self):
        """H1: wound.py exports emit_wound, result has problem_type field."""
        from wound import emit_wound

        intervention = {
            "duration_ms": 100000,
            "action": "cleared cache",
            "context": "cache was full",
            "operator_id": "op1"
        }
        r = emit_wound(intervention)

        assert "problem_type" in r, "emit_wound result must have problem_type field"

    def test_emit_wound_uses_provided_tenant_id(self):
        """emit_wound uses tenant_id from intervention."""
        from wound import emit_wound

        intervention = {
            "duration_ms": 100000,
            "action": "test action",
            "context": "test context",
            "operator_id": "op1",
            "tenant_id": "custom_tenant"
        }
        r = emit_wound(intervention)

        assert r["tenant_id"] == "custom_tenant"

    def test_emit_wound_defaults_tenant_id(self):
        """emit_wound defaults tenant_id to 'default' when not provided."""
        from wound import emit_wound

        intervention = {
            "duration_ms": 100000,
            "action": "test action",
            "context": "test context",
            "operator_id": "op1"
        }
        r = emit_wound(intervention)

        assert r["tenant_id"] == "default"


class TestClassifyWound:
    """Tests for classify_wound function."""

    def test_classify_wound_returns_valid_type(self):
        """classify_wound returns one of the four valid types."""
        from wound import classify_wound, WOUND_TYPES

        intervention = {
            "action": "did something",
            "context": "something happened"
        }
        result = classify_wound(intervention)

        assert result in WOUND_TYPES, f"Expected one of {WOUND_TYPES}, got {result}"

    def test_classify_wound_returns_valid_type_h2(self):
        """H2: classify_wound returns one of: operational, safety, performance, integration."""
        from wound import classify_wound

        intervention = {"action": "test", "context": "test"}
        result = classify_wound(intervention)

        assert result in ["operational", "safety", "performance", "integration"]

    def test_classify_wound_safety(self):
        """Safety keywords classify as 'safety'."""
        from wound import classify_wound

        intervention = {
            "action": "blocked malicious request",
            "context": "detected unauthorized access attempt"
        }
        result = classify_wound(intervention)

        assert result == "safety"

    def test_classify_wound_integration(self):
        """Integration keywords classify as 'integration'."""
        from wound import classify_wound

        intervention = {
            "action": "fixed API endpoint schema",
            "context": "data format mismatch"
        }
        result = classify_wound(intervention)

        assert result == "integration"

    def test_classify_wound_performance(self):
        """Performance keywords classify as 'performance'."""
        from wound import classify_wound

        intervention = {
            "action": "increased memory allocation",
            "context": "slow response times"
        }
        result = classify_wound(intervention)

        assert result == "performance"

    def test_classify_wound_operational_default(self):
        """Defaults to 'operational' when no specific keywords match."""
        from wound import classify_wound

        intervention = {
            "action": "restarted the service",
            "context": "service was down"
        }
        result = classify_wound(intervention)

        assert result == "operational"

    def test_classify_wound_matches_wound_types_constant(self):
        """Wound types match WOUND_TYPES constant."""
        from wound import classify_wound, WOUND_TYPES

        test_cases = [
            {"action": "blocked attack", "context": "security threat"},
            {"action": "fixed api schema", "context": "integration issue"},
            {"action": "optimized cpu usage", "context": "performance"},
            {"action": "restarted service", "context": "routine"},
        ]

        for intervention in test_cases:
            result = classify_wound(intervention)
            assert result in WOUND_TYPES, f"classify_wound returned {result}, not in WOUND_TYPES"


class TestCouldAutomate:
    """Tests for could_automate function."""

    def test_could_automate_returns_float(self):
        """could_automate returns float in [0,1]."""
        from wound import could_automate

        intervention = {
            "duration_ms": 600000,
            "action": "restarted service",
            "context": "service down"
        }
        result = could_automate(intervention)

        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert 0.0 <= result <= 1.0, f"Expected [0,1], got {result}"

    def test_could_automate_returns_float_in_range_h3(self):
        """H3: could_automate returns float, 0 <= result <= 1."""
        from wound import could_automate

        intervention = {"duration_ms": 100000, "action": "test", "context": "test"}
        result = could_automate(intervention)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_safety_wounds_lower_score_than_operational(self):
        """Safety wounds have lower could_automate score than operational."""
        from wound import could_automate

        operational = {
            "duration_ms": 600000,
            "action": "restarted service",
            "context": "routine maintenance"
        }
        safety = {
            "duration_ms": 600000,
            "action": "blocked dangerous request",
            "context": "security threat detected"
        }

        operational_score = could_automate(operational)
        safety_score = could_automate(safety)

        assert safety_score < operational_score, \
            f"Safety ({safety_score}) should be < operational ({operational_score})"

    def test_long_duration_increases_score(self):
        """Wound with >30 min resolve time increases could_automate."""
        from wound import could_automate, AUTOMATION_GAP_RESOLVE_THRESHOLD_MS

        short_duration = {
            "duration_ms": 60000,  # 1 minute
            "action": "restarted service",
            "context": "routine"
        }
        long_duration = {
            "duration_ms": AUTOMATION_GAP_RESOLVE_THRESHOLD_MS + 100000,  # >30 min
            "action": "restarted service",
            "context": "routine"
        }

        short_score = could_automate(short_duration)
        long_score = could_automate(long_duration)

        assert long_score > short_score, \
            f"Long duration ({long_score}) should be > short ({short_score})"

    def test_simple_actions_increase_score(self):
        """Simple deterministic actions increase could_automate score."""
        from wound import could_automate

        simple = {
            "duration_ms": 600000,
            "action": "restart the process",
            "context": "process crashed"
        }
        complex_action = {
            "duration_ms": 600000,
            "action": "investigate and analyze the root cause",
            "context": "process crashed"
        }

        simple_score = could_automate(simple)
        complex_score = could_automate(complex_action)

        assert simple_score > complex_score, \
            f"Simple ({simple_score}) should be > complex ({complex_score})"

    def test_novel_problems_decrease_score(self):
        """Novel/unknown problems decrease could_automate score."""
        from wound import could_automate

        routine = {
            "duration_ms": 600000,
            "action": "restarted service",
            "context": "same as usual recurring issue"
        }
        novel = {
            "duration_ms": 600000,
            "action": "restarted service",
            "context": "first time seeing this unknown error"
        }

        routine_score = could_automate(routine)
        novel_score = could_automate(novel)

        assert novel_score < routine_score, \
            f"Novel ({novel_score}) should be < routine ({routine_score})"


class TestIsAutomationGap:
    """Tests for is_automation_gap function."""

    def test_is_automation_gap_returns_true_when_thresholds_met(self):
        """is_automation_gap returns is_gap=True when thresholds met."""
        from wound import is_automation_gap, AUTOMATION_GAP_RESOLVE_THRESHOLD_MS

        wounds = [
            {
                "time_to_resolve_ms": AUTOMATION_GAP_RESOLVE_THRESHOLD_MS + 100000,
                "could_automate": 0.7,
                "problem_type": "operational"
            }
            for _ in range(6)  # > 5 occurrences
        ]
        r = is_automation_gap(wounds, "test_tenant")

        assert r["is_gap"] is True, "Should be a gap when both thresholds met"
        assert r["recommendation"] == "propose_blueprint"

    def test_is_automation_gap_returns_true_h4(self):
        """H4: is_automation_gap with 6 wounds over 30min returns is_gap=True."""
        from wound import is_automation_gap

        wounds = [
            {"time_to_resolve_ms": 2000000, "could_automate": 0.7, "problem_type": "operational"}
            for _ in range(6)
        ]
        r = is_automation_gap(wounds, "test_tenant")

        assert r["is_gap"] is True

    def test_is_automation_gap_returns_false_when_under_threshold(self):
        """is_automation_gap returns is_gap=False when under threshold."""
        from wound import is_automation_gap

        # Only 3 occurrences (below threshold of 5)
        wounds = [
            {"time_to_resolve_ms": 2000000, "could_automate": 0.7, "problem_type": "operational"}
            for _ in range(3)
        ]
        r = is_automation_gap(wounds, "test_tenant")

        assert r["is_gap"] is False, "Should not be a gap when occurrence threshold not met"

    def test_is_automation_gap_false_when_time_under_threshold(self):
        """is_automation_gap returns false when time under threshold."""
        from wound import is_automation_gap, AUTOMATION_GAP_RESOLVE_THRESHOLD_MS

        # Many occurrences but short resolve times
        wounds = [
            {
                "time_to_resolve_ms": AUTOMATION_GAP_RESOLVE_THRESHOLD_MS - 100000,
                "could_automate": 0.7,
                "problem_type": "operational"
            }
            for _ in range(10)
        ]
        r = is_automation_gap(wounds, "test_tenant")

        assert r["is_gap"] is False, "Should not be a gap when time threshold not met"

    def test_is_automation_gap_empty_list(self):
        """is_automation_gap handles empty wound list."""
        from wound import is_automation_gap

        r = is_automation_gap([], "test_tenant")

        assert r["is_gap"] is False
        assert r["occurrence_count"] == 0
        assert r["recommendation"] == "ignore"

    def test_is_automation_gap_has_required_fields(self):
        """automation_gap receipt has all required fields."""
        from wound import is_automation_gap

        wounds = [
            {"time_to_resolve_ms": 2000000, "could_automate": 0.7, "problem_type": "operational"}
        ]
        r = is_automation_gap(wounds, "test_tenant")

        assert r["receipt_type"] == "automation_gap"
        assert "ts" in r
        assert "tenant_id" in r
        assert "wound_type" in r
        assert "occurrence_count" in r
        assert "median_resolve_ms" in r
        assert "total_human_time_ms" in r
        assert "could_automate_avg" in r
        assert "is_gap" in r
        assert "recommendation" in r
        assert "payload_hash" in r


class TestReceiptSchema:
    """Tests for RECEIPT_SCHEMA export."""

    def test_receipt_schema_exported(self):
        """RECEIPT_SCHEMA exported with all three receipt types."""
        from wound import RECEIPT_SCHEMA

        assert len(RECEIPT_SCHEMA) == 3, f"Expected 3 receipt types, got {len(RECEIPT_SCHEMA)}"
        assert "wound_receipt" in RECEIPT_SCHEMA
        assert "automation_gap" in RECEIPT_SCHEMA
        assert "healing_record" in RECEIPT_SCHEMA

    def test_receipt_schema_contains_wound_receipt_h5(self):
        """H5: RECEIPT_SCHEMA contains 'wound_receipt'."""
        from wound import RECEIPT_SCHEMA

        assert "wound_receipt" in RECEIPT_SCHEMA


class TestStopruleWoundCaptureFailure:
    """Tests for stoprule_wound_capture_failure."""

    def test_stoprule_raises_stoprule(self):
        """stoprule_wound_capture_failure raises StopRule."""
        from wound import stoprule_wound_capture_failure, StopRule

        intervention = {
            "tenant_id": "test_tenant",
            "intervention_id": "wound_001"
        }

        with pytest.raises(StopRule):
            stoprule_wound_capture_failure(intervention)


class TestWoundTypes:
    """Tests for WOUND_TYPES constant."""

    def test_wound_types_constant(self):
        """WOUND_TYPES constant has correct values."""
        from wound import WOUND_TYPES

        expected = ['operational', 'safety', 'performance', 'integration']
        assert WOUND_TYPES == expected, f"Expected {expected}, got {WOUND_TYPES}"


class TestConstants:
    """Tests for module constants."""

    def test_automation_gap_occurrence_threshold(self):
        """AUTOMATION_GAP_OCCURRENCE_THRESHOLD is 5."""
        from wound import AUTOMATION_GAP_OCCURRENCE_THRESHOLD

        assert AUTOMATION_GAP_OCCURRENCE_THRESHOLD == 5

    def test_automation_gap_resolve_threshold_ms(self):
        """AUTOMATION_GAP_RESOLVE_THRESHOLD_MS is 1800000 (30 minutes)."""
        from wound import AUTOMATION_GAP_RESOLVE_THRESHOLD_MS

        assert AUTOMATION_GAP_RESOLVE_THRESHOLD_MS == 1800000


class TestDualHash:
    """Tests for dual_hash compliance."""

    def test_dual_hash_format(self):
        """dual_hash returns SHA256:BLAKE3 format."""
        from wound import dual_hash

        result = dual_hash(b"test data")
        assert ":" in result, "dual_hash must contain ':' separator"
        parts = result.split(":")
        assert len(parts) == 2, "dual_hash must have exactly two parts"
        assert len(parts[0]) == 64, "SHA256 hex should be 64 chars"

    def test_payload_hash_in_receipts(self):
        """All receipts have dual_hash format payload_hash."""
        from wound import emit_wound, is_automation_gap, emit_healing_record

        receipts = [
            emit_wound({
                "duration_ms": 100000,
                "action": "test",
                "context": "test",
                "operator_id": "op1"
            }),
            is_automation_gap([], "test"),
            emit_healing_record("test", "wound_001", True, "manual", 1000),
        ]

        for r in receipts:
            assert "payload_hash" in r, f"Missing payload_hash in {r['receipt_type']}"
            assert ":" in r["payload_hash"], f"payload_hash not dual_hash format in {r['receipt_type']}"


class TestHealingRecord:
    """Tests for healing_record receipt."""

    def test_emit_healing_record(self):
        """emit_healing_record creates valid receipt."""
        from wound import emit_healing_record

        r = emit_healing_record(
            tenant_id="test_tenant",
            wound_id="wound_001",
            healed=True,
            healing_method="automated",
            time_to_heal_ms=5000
        )

        assert r["receipt_type"] == "healing_record"
        assert r["tenant_id"] == "test_tenant"
        assert r["wound_id"] == "wound_001"
        assert r["healed"] is True
        assert r["healing_method"] == "automated"
        assert r["time_to_heal_ms"] == 5000

    def test_healing_methods(self):
        """healing_record supports manual, automated, hybrid methods."""
        from wound import emit_healing_record

        for method in ["manual", "automated", "hybrid"]:
            r = emit_healing_record("test", "wound_001", True, method, 1000)
            assert r["healing_method"] == method


class TestTenantIdPresent:
    """Tests that tenant_id is present in all emitted receipts."""

    def test_tenant_id_in_wound_receipt(self):
        """wound_receipt has tenant_id."""
        from wound import emit_wound

        r = emit_wound({
            "duration_ms": 100000,
            "action": "test",
            "context": "test",
            "operator_id": "op1",
            "tenant_id": "custom_tenant"
        })

        assert "tenant_id" in r, "Missing tenant_id in wound_receipt"
        assert r["tenant_id"] == "custom_tenant"

    def test_tenant_id_in_automation_gap(self):
        """automation_gap receipt has tenant_id."""
        from wound import is_automation_gap

        r = is_automation_gap([], "test_tenant")

        assert "tenant_id" in r, "Missing tenant_id in automation_gap"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_healing_record(self):
        """healing_record receipt has tenant_id."""
        from wound import emit_healing_record

        r = emit_healing_record("test_tenant", "wound_001", True, "manual", 1000)

        assert "tenant_id" in r, "Missing tenant_id in healing_record"
        assert r["tenant_id"] == "test_tenant"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """All required exports are available."""
        from wound import (
            RECEIPT_SCHEMA,
            WOUND_TYPES,
            AUTOMATION_GAP_OCCURRENCE_THRESHOLD,
            AUTOMATION_GAP_RESOLVE_THRESHOLD_MS,
            classify_wound,
            could_automate,
            is_automation_gap,
            emit_wound,
            emit_healing_record,
            stoprule_wound_capture_failure,
            stoprule_automation_gap,
            emit_receipt,
            dual_hash,
            StopRule,
        )

        # Verify callables
        assert callable(classify_wound)
        assert callable(could_automate)
        assert callable(is_automation_gap)
        assert callable(emit_wound)
        assert callable(emit_healing_record)
        assert callable(stoprule_wound_capture_failure)
        assert callable(stoprule_automation_gap)
        assert callable(emit_receipt)
        assert callable(dual_hash)

        # Verify constants
        assert AUTOMATION_GAP_OCCURRENCE_THRESHOLD == 5
        assert AUTOMATION_GAP_RESOLVE_THRESHOLD_MS == 1800000


class TestInternalTests:
    """Run the internal test functions defined in wound.py."""

    def test_internal_wound_receipt(self):
        """Run wound.py's internal test_wound_receipt."""
        from wound import test_wound_receipt
        test_wound_receipt()

    def test_internal_automation_gap(self):
        """Run wound.py's internal test_automation_gap."""
        from wound import test_automation_gap
        test_automation_gap()

    def test_internal_healing_record(self):
        """Run wound.py's internal test_healing_record."""
        from wound import test_healing_record
        test_healing_record()
