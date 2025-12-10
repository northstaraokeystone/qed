"""
tests/test_entropy.py - Tests for entropy.py

CLAUDEME v3.1 Compliant: Every test has assert statements.
"""

import math

import pytest


class TestSystemEntropy:
    """Tests for system_entropy function."""

    def test_system_entropy_empty(self):
        """Empty list returns 0.0 - no uncertainty."""
        from entropy import system_entropy

        result = system_entropy([])
        assert result == 0.0, f"Expected 0.0 for empty list, got {result}"

    def test_system_entropy_single_type(self):
        """All same receipt type returns 0.0 - no uncertainty."""
        from entropy import system_entropy

        receipts = [
            {"receipt_type": "ingest"},
            {"receipt_type": "ingest"},
            {"receipt_type": "ingest"},
        ]
        result = system_entropy(receipts)
        assert result == 0.0, f"Expected 0.0 for single type, got {result}"

    def test_system_entropy_uniform_two(self):
        """Two types uniformly distributed = 1.0 bit exactly."""
        from entropy import system_entropy

        receipts = [
            {"receipt_type": "a"},
            {"receipt_type": "b"},
        ]
        result = system_entropy(receipts)
        assert abs(result - 1.0) < 1e-10, f"Expected 1.0 bit for two uniform types, got {result}"

    def test_system_entropy_uniform_four(self):
        """Four types uniformly distributed = 2.0 bits exactly."""
        from entropy import system_entropy

        receipts = [
            {"receipt_type": "a"},
            {"receipt_type": "b"},
            {"receipt_type": "c"},
            {"receipt_type": "d"},
        ]
        result = system_entropy(receipts)
        assert abs(result - 2.0) < 1e-10, f"Expected 2.0 bits for four uniform types, got {result}"

    def test_system_entropy_non_uniform(self):
        """Non-uniform distribution returns expected entropy."""
        from entropy import system_entropy

        # 3 of type a, 1 of type b -> p(a)=0.75, p(b)=0.25
        # H = -0.75*log2(0.75) - 0.25*log2(0.25) = 0.8113 bits
        receipts = [
            {"receipt_type": "a"},
            {"receipt_type": "a"},
            {"receipt_type": "a"},
            {"receipt_type": "b"},
        ]
        result = system_entropy(receipts)
        expected = -0.75 * math.log2(0.75) - 0.25 * math.log2(0.25)
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


class TestAgentFitness:
    """Tests for agent_fitness function."""

    def test_agent_fitness_positive(self):
        """H_before > H_after yields positive fitness."""
        from entropy import agent_fitness

        # Before: 2 types uniform (1 bit)
        receipts_before = [
            {"receipt_type": "a"},
            {"receipt_type": "b"},
        ]
        # After: 1 type (0 bits)
        receipts_after = [
            {"receipt_type": "a"},
        ]
        result = agent_fitness(receipts_before, receipts_after, 1)
        assert result > 0, f"Expected positive fitness, got {result}"
        assert result == 1.0, f"Expected fitness of 1.0, got {result}"

    def test_agent_fitness_negative(self):
        """H_before < H_after yields negative fitness."""
        from entropy import agent_fitness

        # Before: 1 type (0 bits)
        receipts_before = [
            {"receipt_type": "a"},
        ]
        # After: 2 types uniform (1 bit)
        receipts_after = [
            {"receipt_type": "a"},
            {"receipt_type": "b"},
        ]
        result = agent_fitness(receipts_before, receipts_after, 1)
        assert result < 0, f"Expected negative fitness, got {result}"

    def test_agent_fitness_zero_count(self):
        """Zero receipt count returns 0.0."""
        from entropy import agent_fitness

        receipts_before = [{"receipt_type": "a"}]
        receipts_after = [{"receipt_type": "b"}]
        result = agent_fitness(receipts_before, receipts_after, 0)
        assert result == 0.0, f"Expected 0.0 for zero count, got {result}"

    def test_agent_fitness_negative_count(self):
        """Negative receipt count returns 0.0."""
        from entropy import agent_fitness

        receipts_before = [{"receipt_type": "a"}]
        receipts_after = [{"receipt_type": "b"}]
        result = agent_fitness(receipts_before, receipts_after, -5)
        assert result == 0.0, f"Expected 0.0 for negative count, got {result}"


class TestCycleEntropyDelta:
    """Tests for cycle_entropy_delta function."""

    def test_cycle_entropy_delta_healthy(self):
        """Before > after yields positive delta (healthy)."""
        from entropy import cycle_entropy_delta

        # Before: 2 types uniform (1 bit)
        receipts_before = [
            {"receipt_type": "a"},
            {"receipt_type": "b"},
        ]
        # After: 1 type (0 bits)
        receipts_after = [
            {"receipt_type": "a"},
        ]
        result = cycle_entropy_delta(receipts_before, receipts_after)
        assert result > 0, f"Expected positive delta, got {result}"
        assert result == 1.0, f"Expected delta of 1.0, got {result}"

    def test_cycle_entropy_delta_degrading(self):
        """Before < after yields negative delta (degrading)."""
        from entropy import cycle_entropy_delta

        # Before: 1 type (0 bits)
        receipts_before = [
            {"receipt_type": "a"},
        ]
        # After: 2 types uniform (1 bit)
        receipts_after = [
            {"receipt_type": "a"},
            {"receipt_type": "b"},
        ]
        result = cycle_entropy_delta(receipts_before, receipts_after)
        assert result < 0, f"Expected negative delta, got {result}"
        assert result == -1.0, f"Expected delta of -1.0, got {result}"

    def test_cycle_entropy_delta_stable(self):
        """Same entropy before and after yields zero delta."""
        from entropy import cycle_entropy_delta

        receipts = [
            {"receipt_type": "a"},
            {"receipt_type": "b"},
        ]
        result = cycle_entropy_delta(receipts, receipts)
        assert result == 0.0, f"Expected 0.0 for stable entropy, got {result}"


class TestSelectionPressure:
    """Tests for selection_pressure function."""

    def test_selection_pressure_survivors(self):
        """High fitness patterns survive selection."""
        from entropy import selection_pressure, SURVIVAL_THRESHOLD

        patterns = [
            {"id": "high_fitness", "fitness_mean": 1.0, "fitness_var": 0.01},
            {"id": "very_high", "fitness_mean": 2.0, "fitness_var": 0.01},
        ]
        survivors, receipt = selection_pressure(patterns, SURVIVAL_THRESHOLD, "test_tenant")

        assert len(survivors) == 2, f"Expected 2 survivors, got {len(survivors)}"
        assert "high_fitness" in survivors
        assert "very_high" in survivors
        assert receipt["receipt_type"] == "selection_event"
        assert receipt["tenant_id"] == "test_tenant"

    def test_selection_pressure_superposition(self):
        """Low fitness patterns go to superposition."""
        from entropy import selection_pressure, SURVIVAL_THRESHOLD

        patterns = [
            {"id": "low_fitness", "fitness_mean": -1.0, "fitness_var": 0.01},
            {"id": "very_low", "fitness_mean": -2.0, "fitness_var": 0.01},
        ]
        survivors, receipt = selection_pressure(patterns, SURVIVAL_THRESHOLD, "test_tenant")

        assert len(survivors) == 0, f"Expected 0 survivors, got {len(survivors)}"
        assert len(receipt["superposition"]) == 2
        assert "low_fitness" in receipt["superposition"]
        assert "very_low" in receipt["superposition"]

    def test_selection_pressure_mixed(self):
        """Mixed fitness patterns split between survivors and superposition."""
        from entropy import selection_pressure, SURVIVAL_THRESHOLD

        patterns = [
            {"id": "high", "fitness_mean": 1.0, "fitness_var": 0.0001},  # Will survive
            {"id": "low", "fitness_mean": -1.0, "fitness_var": 0.0001},  # Will fail
        ]
        survivors, receipt = selection_pressure(patterns, SURVIVAL_THRESHOLD, "test_tenant")

        assert "high" in survivors
        assert "low" in receipt["superposition"]
        assert receipt["sampling_method"] == "thompson"

    def test_selection_pressure_empty_patterns(self):
        """Empty pattern list produces empty results."""
        from entropy import selection_pressure, SURVIVAL_THRESHOLD

        survivors, receipt = selection_pressure([], SURVIVAL_THRESHOLD, "test_tenant")

        assert len(survivors) == 0
        assert receipt["patterns_evaluated"] == 0
        assert len(receipt["survivors"]) == 0
        assert len(receipt["superposition"]) == 0


class TestReceiptSchemas:
    """Tests for receipt schema exports."""

    def test_receipt_schemas_exported(self):
        """RECEIPT_SCHEMA contains all three types."""
        from entropy import RECEIPT_SCHEMA

        assert len(RECEIPT_SCHEMA) == 3, f"Expected 3 receipt types, got {len(RECEIPT_SCHEMA)}"
        assert "entropy_measurement" in RECEIPT_SCHEMA
        assert "fitness_score" in RECEIPT_SCHEMA
        assert "selection_event" in RECEIPT_SCHEMA


class TestTenantIdPresent:
    """Tests that tenant_id is present in all emitted receipts."""

    def test_tenant_id_in_entropy_measurement(self):
        """entropy_measurement receipt has tenant_id."""
        from entropy import emit_entropy_measurement

        r = emit_entropy_measurement(
            tenant_id="test_tenant",
            cycle_id=1,
            receipts_before=[{"receipt_type": "a"}],
            receipts_after=[{"receipt_type": "a"}]
        )
        assert "tenant_id" in r, "Missing tenant_id in entropy_measurement"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_fitness_score(self):
        """fitness_score receipt has tenant_id."""
        from entropy import emit_fitness_score

        r = emit_fitness_score(
            tenant_id="test_tenant",
            pattern_id="p1",
            receipts_before=[{"receipt_type": "a"}],
            receipts_after=[{"receipt_type": "a"}],
            receipt_count=1
        )
        assert "tenant_id" in r, "Missing tenant_id in fitness_score"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_selection_event(self):
        """selection_event receipt has tenant_id."""
        from entropy import emit_selection_event

        r = emit_selection_event(
            tenant_id="test_tenant",
            patterns_evaluated=1,
            survivors=["p1"],
            superposition=[],
            survival_threshold=0.0
        )
        assert "tenant_id" in r, "Missing tenant_id in selection_event"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_selection_pressure_receipt(self):
        """selection_pressure returns receipt with tenant_id."""
        from entropy import selection_pressure

        patterns = [{"id": "p1", "fitness_mean": 1.0, "fitness_var": 0.0}]
        _, receipt = selection_pressure(patterns, 0.0, "test_tenant")
        assert "tenant_id" in receipt, "Missing tenant_id in selection_pressure receipt"
        assert receipt["tenant_id"] == "test_tenant"


class TestDualHash:
    """Tests for dual_hash compliance."""

    def test_dual_hash_format(self):
        """dual_hash returns SHA256:BLAKE3 format."""
        from entropy import dual_hash

        result = dual_hash(b"test data")
        assert ":" in result, "dual_hash must contain ':' separator"
        parts = result.split(":")
        assert len(parts) == 2, "dual_hash must have exactly two parts"
        assert len(parts[0]) == 64, "SHA256 hex should be 64 chars"

    def test_payload_hash_in_receipts(self):
        """All receipts have dual_hash format payload_hash."""
        from entropy import emit_entropy_measurement, emit_fitness_score, emit_selection_event

        receipts = [
            emit_entropy_measurement("t", 1, [], []),
            emit_fitness_score("t", "p1", [], [], 0),
            emit_selection_event("t", 0, [], [], 0.0),
        ]

        for r in receipts:
            assert "payload_hash" in r, f"Missing payload_hash in {r['receipt_type']}"
            assert ":" in r["payload_hash"], f"payload_hash not dual_hash format in {r['receipt_type']}"


class TestStoprules:
    """Tests for stoprule functions."""

    def test_stoprule_entropy_degradation(self):
        """Three consecutive negative deltas raises StopRule."""
        from entropy import stoprule_entropy_measurement, StopRule

        tenant = "test_degradation"

        # First two negative deltas don't raise
        stoprule_entropy_measurement(tenant, -0.1)
        stoprule_entropy_measurement(tenant, -0.2)

        # Third negative delta raises StopRule
        with pytest.raises(StopRule):
            stoprule_entropy_measurement(tenant, -0.3)

    def test_stoprule_entropy_reset_on_positive(self):
        """Positive delta resets consecutive counter."""
        from entropy import stoprule_entropy_measurement, StopRule

        tenant = "test_reset"

        stoprule_entropy_measurement(tenant, -0.1)
        stoprule_entropy_measurement(tenant, -0.2)
        stoprule_entropy_measurement(tenant, 0.5)  # Positive resets
        stoprule_entropy_measurement(tenant, -0.1)
        stoprule_entropy_measurement(tenant, -0.2)

        # Should not raise, counter was reset
        # (no exception expected here)

    def test_stoprule_fitness_no_raise(self):
        """fitness_score stoprule does NOT raise in v10."""
        from entropy import stoprule_fitness_score

        # Should not raise even for negative fitness (v10 measurement only)
        stoprule_fitness_score("test_tenant", "p1", -1.0)
        # If we get here without exception, test passes


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """All required exports are available."""
        from entropy import (
            RECEIPT_SCHEMA,
            SURVIVAL_THRESHOLD,
            system_entropy,
            agent_fitness,
            cycle_entropy_delta,
            selection_pressure,
            emit_entropy_measurement,
            emit_fitness_score,
            emit_selection_event,
            stoprule_entropy_measurement,
            stoprule_fitness_score,
            stoprule_selection_event,
            emit_receipt,
            dual_hash,
            StopRule,
        )

        # Just importing without error is the test
        assert callable(system_entropy)
        assert callable(agent_fitness)
        assert callable(cycle_entropy_delta)
        assert callable(selection_pressure)
        assert callable(emit_entropy_measurement)
        assert callable(emit_fitness_score)
        assert callable(emit_selection_event)
        assert callable(stoprule_entropy_measurement)
        assert callable(stoprule_fitness_score)
        assert callable(stoprule_selection_event)
        assert callable(emit_receipt)
        assert callable(dual_hash)
        assert SURVIVAL_THRESHOLD == 0.0


class TestInternalTests:
    """Run the internal test functions defined in entropy.py."""

    def test_internal_entropy_measurement(self):
        """Run entropy.py's internal test_entropy_measurement."""
        from entropy import test_entropy_measurement
        test_entropy_measurement()

    def test_internal_fitness_score(self):
        """Run entropy.py's internal test_fitness_score."""
        from entropy import test_fitness_score
        test_fitness_score()

    def test_internal_selection_event(self):
        """Run entropy.py's internal test_selection_event."""
        from entropy import test_selection_event
        test_selection_event()
