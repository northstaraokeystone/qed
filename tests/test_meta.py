"""
tests/test_meta.py - Tests for meta.py (v11 Adaptive Immunity Layer)

CLAUDEME v3.1 Compliant: Every test has assert statements.
"""

import pytest


class TestEmitMetaFitness:
    """Tests for emit_meta_fitness function."""

    def test_emit_meta_fitness_returns_valid_receipt(self):
        """emit_meta_fitness returns valid meta_fitness_receipt with all required fields."""
        from meta import emit_meta_fitness

        paradigm = {
            "name": "test_paradigm",
            "introduced_at": "2024-01-01T00:00:00Z",
            "pre_metrics": {"roi": 0.5, "fitness_avg": 0.6, "entropy_delta": 0.1},
            "post_metrics": {"roi": 0.8, "fitness_avg": 0.7, "entropy_delta": 0.05},
            "survival_30d": True
        }
        r = emit_meta_fitness(paradigm, "test_tenant")

        assert r["receipt_type"] == "meta_fitness_receipt"
        assert r["tenant_id"] == "test_tenant"
        assert r["paradigm_name"] == "test_paradigm"
        assert r["introduced_at"] == "2024-01-01T00:00:00Z"
        assert "pre_metrics" in r
        assert "post_metrics" in r
        assert "delta_roi" in r
        assert "survival_30d" in r
        assert r["observation_window_days"] == 30
        assert "payload_hash" in r
        assert ":" in r["payload_hash"]  # dual_hash format

    def test_delta_roi_computed_correctly(self):
        """delta_roi is computed as post_roi - pre_roi."""
        from meta import emit_meta_fitness

        paradigm = {
            "name": "test_paradigm",
            "introduced_at": "2024-01-01T00:00:00Z",
            "pre_metrics": {"roi": 0.3, "fitness_avg": 0.5, "entropy_delta": 0.1},
            "post_metrics": {"roi": 0.7, "fitness_avg": 0.6, "entropy_delta": 0.05},
            "survival_30d": None
        }
        r = emit_meta_fitness(paradigm)

        expected_delta = 0.7 - 0.3
        assert abs(r["delta_roi"] - expected_delta) < 1e-10, f"Expected {expected_delta}, got {r['delta_roi']}"

    def test_delta_roi_negative(self):
        """delta_roi can be negative when ROI decreases."""
        from meta import emit_meta_fitness

        paradigm = {
            "name": "failing_paradigm",
            "introduced_at": "2024-01-01T00:00:00Z",
            "pre_metrics": {"roi": 0.8, "fitness_avg": 0.6, "entropy_delta": 0.1},
            "post_metrics": {"roi": 0.4, "fitness_avg": 0.5, "entropy_delta": 0.2},
            "survival_30d": False
        }
        r = emit_meta_fitness(paradigm)

        assert r["delta_roi"] < 0, f"Expected negative delta_roi, got {r['delta_roi']}"
        assert abs(r["delta_roi"] - (-0.4)) < 1e-10

    def test_survival_30d_preserved(self):
        """survival_30d field is preserved as-is."""
        from meta import emit_meta_fitness

        # Test True
        paradigm = {"name": "p", "pre_metrics": {"roi": 0}, "post_metrics": {"roi": 0}, "survival_30d": True}
        r = emit_meta_fitness(paradigm)
        assert r["survival_30d"] is True

        # Test False
        paradigm["survival_30d"] = False
        r = emit_meta_fitness(paradigm)
        assert r["survival_30d"] is False

        # Test None
        paradigm["survival_30d"] = None
        r = emit_meta_fitness(paradigm)
        assert r["survival_30d"] is None


class TestEmitEnergyAllocation:
    """Tests for emit_energy_allocation function."""

    def test_emit_energy_allocation_normalizes_energy_share(self):
        """emit_energy_allocation normalizes energy_share to sum to 1.0."""
        from meta import emit_energy_allocation

        cycle = {
            "cycle_id": 1,
            "patterns": [
                {"pattern_id": "p1", "fitness": 0.8, "resource_share": 60.0},
                {"pattern_id": "p2", "fitness": 0.6, "resource_share": 40.0}
            ]
        }
        r = emit_energy_allocation(cycle, "test_tenant")

        total_share = sum(a["energy_share"] for a in r["allocations"])
        assert abs(total_share - 1.0) < 1e-10, f"Energy shares should sum to 1.0, got {total_share}"

    def test_energy_allocation_correct_shares(self):
        """Energy shares are correctly normalized."""
        from meta import emit_energy_allocation

        cycle = {
            "cycle_id": 42,
            "patterns": [
                {"pattern_id": "p1", "fitness": 0.8, "resource_share": 75.0},
                {"pattern_id": "p2", "fitness": 0.6, "resource_share": 25.0}
            ]
        }
        r = emit_energy_allocation(cycle)

        assert r["allocations"][0]["energy_share"] == 0.75, f"Expected 0.75, got {r['allocations'][0]['energy_share']}"
        assert r["allocations"][1]["energy_share"] == 0.25, f"Expected 0.25, got {r['allocations'][1]['energy_share']}"

    def test_energy_allocation_two_patterns_returns_two_allocations(self):
        """H2: emit_energy_allocation with 2 patterns returns allocations list of length 2."""
        from meta import emit_energy_allocation

        cycle = {
            "cycle_id": 1,
            "patterns": [
                {"pattern_id": "p1", "fitness": 0.5, "resource_share": 50.0},
                {"pattern_id": "p2", "fitness": 0.5, "resource_share": 50.0}
            ]
        }
        r = emit_energy_allocation(cycle)

        assert len(r["allocations"]) == 2, f"Expected 2 allocations, got {len(r['allocations'])}"

    def test_energy_allocation_zero_total_energy(self):
        """Handle edge case where all resource_share values are 0."""
        from meta import emit_energy_allocation

        cycle = {
            "cycle_id": 1,
            "patterns": [
                {"pattern_id": "p1", "fitness": 0.5, "resource_share": 0.0},
                {"pattern_id": "p2", "fitness": 0.5, "resource_share": 0.0}
            ]
        }
        r = emit_energy_allocation(cycle)

        assert r["total_energy"] == 0.0
        for alloc in r["allocations"]:
            assert alloc["energy_share"] == 0.0

    def test_energy_allocation_receipt_fields(self):
        """Receipt has all required fields."""
        from meta import emit_energy_allocation

        cycle = {
            "cycle_id": 123,
            "patterns": [{"pattern_id": "p1", "fitness": 0.5, "resource_share": 100.0}]
        }
        r = emit_energy_allocation(cycle, "test_tenant")

        assert r["receipt_type"] == "energy_allocation_receipt"
        assert r["tenant_id"] == "test_tenant"
        assert r["cycle_id"] == 123
        assert "total_energy" in r
        assert "allocations" in r
        assert "pattern_count" in r
        assert r["pattern_count"] == 1
        assert "payload_hash" in r


class TestWeightFutureProposals:
    """Tests for weight_future_proposals function."""

    def test_weight_future_proposals_returns_weights_in_range(self):
        """weight_future_proposals returns weights in [0,1] range."""
        from meta import weight_future_proposals

        meta_receipts = [
            {"paradigm_name": "p1", "delta_roi": 0.5, "survival_30d": True},
            {"paradigm_name": "p2", "delta_roi": -0.5, "survival_30d": False},
            {"paradigm_name": "p3", "delta_roi": 0.0, "survival_30d": None}
        ]
        r = weight_future_proposals(meta_receipts)

        for paradigm, weight in r["proposed_weights"].items():
            assert 0.0 <= weight <= 1.0, f"Weight {weight} for {paradigm} out of range"

    def test_positive_roi_and_survival_gets_higher_weight(self):
        """Paradigm with positive roi and survival gets higher weight."""
        from meta import weight_future_proposals

        meta_receipts = [
            {"paradigm_name": "successful", "delta_roi": 0.5, "survival_30d": True},
            {"paradigm_name": "failed", "delta_roi": -0.5, "survival_30d": False},
            {"paradigm_name": "neutral", "delta_roi": 0.0, "survival_30d": None}
        ]
        r = weight_future_proposals(meta_receipts)

        assert r["proposed_weights"]["successful"] > r["proposed_weights"]["failed"]
        assert r["proposed_weights"]["successful"] > r["proposed_weights"]["neutral"]

    def test_negative_roi_gets_lower_weight(self):
        """Paradigm with negative roi gets lower weight."""
        from meta import weight_future_proposals

        meta_receipts = [
            {"paradigm_name": "good", "delta_roi": 0.3, "survival_30d": None},
            {"paradigm_name": "bad", "delta_roi": -0.3, "survival_30d": None},
            {"paradigm_name": "neutral", "delta_roi": 0.0, "survival_30d": None}
        ]
        r = weight_future_proposals(meta_receipts)

        # good: 0.5 + 0.3 = 0.8
        # bad: 0.5 (no positive roi bonus)
        # neutral: 0.5 (no positive roi bonus since delta_roi is exactly 0)
        assert r["proposed_weights"]["good"] > r["proposed_weights"]["bad"]

    def test_empty_meta_receipts_returns_neutral_weights(self):
        """Empty meta_receipts list returns empty proposed_weights."""
        from meta import weight_future_proposals

        r = weight_future_proposals([])

        assert r["receipt_type"] == "paradigm_outcome"
        assert r["paradigms_evaluated"] == 0
        assert r["proposed_weights"] == {}
        assert r["learning_confidence"] == 0.0

    def test_weight_future_proposals_returns_dict_with_float_values(self):
        """H3: weight_future_proposals returns dict with float values."""
        from meta import weight_future_proposals

        meta_receipts = [
            {"paradigm_name": "p1", "delta_roi": 0.1, "survival_30d": True}
        ]
        r = weight_future_proposals(meta_receipts)

        assert isinstance(r["proposed_weights"], dict)
        for value in r["proposed_weights"].values():
            assert isinstance(value, float), f"Expected float, got {type(value)}"

    def test_learning_confidence_increases_with_more_paradigms(self):
        """Learning confidence increases with more paradigms evaluated."""
        from meta import weight_future_proposals, MIN_PARADIGMS_FOR_WEIGHTING

        # Less than MIN_PARADIGMS_FOR_WEIGHTING -> confidence = 0
        meta_receipts = [{"paradigm_name": f"p{i}", "delta_roi": 0.1, "survival_30d": True}
                        for i in range(MIN_PARADIGMS_FOR_WEIGHTING - 1)]
        r1 = weight_future_proposals(meta_receipts)
        assert r1["learning_confidence"] == 0.0

        # At least MIN_PARADIGMS_FOR_WEIGHTING -> confidence > 0
        meta_receipts = [{"paradigm_name": f"p{i}", "delta_roi": 0.1, "survival_30d": True}
                        for i in range(MIN_PARADIGMS_FOR_WEIGHTING)]
        r2 = weight_future_proposals(meta_receipts)
        assert r2["learning_confidence"] > 0.0


class TestComputeParadigmDelta:
    """Tests for compute_paradigm_delta helper function."""

    def test_compute_paradigm_delta_returns_correct_deltas(self):
        """compute_paradigm_delta returns correct deltas."""
        from meta import compute_paradigm_delta

        pre = {"roi": 0.5, "fitness_avg": 0.3, "entropy_delta": 0.1}
        post = {"roi": 0.8, "fitness_avg": 0.4, "entropy_delta": 0.05}

        deltas = compute_paradigm_delta(pre, post)

        assert abs(deltas["roi"] - 0.3) < 1e-10
        assert abs(deltas["fitness_avg"] - 0.1) < 1e-10
        assert abs(deltas["entropy_delta"] - (-0.05)) < 1e-10

    def test_compute_paradigm_delta_handles_missing_keys(self):
        """compute_paradigm_delta handles missing keys gracefully."""
        from meta import compute_paradigm_delta

        pre = {"roi": 0.5}
        post = {"roi": 0.8, "new_metric": 0.3}

        deltas = compute_paradigm_delta(pre, post)

        assert abs(deltas["roi"] - 0.3) < 1e-10
        assert abs(deltas["new_metric"] - 0.3) < 1e-10  # pre has 0.0 default


class TestReceiptSchema:
    """Tests for RECEIPT_SCHEMA export."""

    def test_receipt_schema_exported_with_all_types(self):
        """RECEIPT_SCHEMA exported with all three receipt types."""
        from meta import RECEIPT_SCHEMA

        assert len(RECEIPT_SCHEMA) == 3, f"Expected 3 receipt types, got {len(RECEIPT_SCHEMA)}"
        assert "meta_fitness_receipt" in RECEIPT_SCHEMA
        assert "energy_allocation_receipt" in RECEIPT_SCHEMA
        assert "paradigm_outcome" in RECEIPT_SCHEMA


class TestStopruleMetaTrackingFailure:
    """Tests for stoprule_meta_tracking_failure."""

    def test_stoprule_meta_tracking_failure_raises_stoprule(self):
        """stoprule_meta_tracking_failure raises StopRule."""
        from meta import stoprule_meta_tracking_failure, StopRule

        event = {
            "tenant_id": "test_tenant",
            "error": "Failed to track paradigm",
            "context": {"cycle_id": 42}
        }

        with pytest.raises(StopRule) as exc_info:
            stoprule_meta_tracking_failure(event)

        assert "Meta tracking failure" in str(exc_info.value)


class TestTenantIdPresent:
    """Tests that tenant_id is present in all emitted receipts."""

    def test_tenant_id_in_meta_fitness(self):
        """meta_fitness_receipt has tenant_id."""
        from meta import emit_meta_fitness

        paradigm = {"name": "test", "pre_metrics": {"roi": 0}, "post_metrics": {"roi": 0}}
        r = emit_meta_fitness(paradigm, "test_tenant")

        assert "tenant_id" in r
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_energy_allocation(self):
        """energy_allocation_receipt has tenant_id."""
        from meta import emit_energy_allocation

        cycle = {"cycle_id": 1, "patterns": []}
        r = emit_energy_allocation(cycle, "test_tenant")

        assert "tenant_id" in r
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_paradigm_outcome(self):
        """paradigm_outcome receipt has tenant_id."""
        from meta import weight_future_proposals

        r = weight_future_proposals([], "test_tenant")

        assert "tenant_id" in r
        assert r["tenant_id"] == "test_tenant"


class TestDualHash:
    """Tests for dual_hash compliance."""

    def test_dual_hash_format(self):
        """dual_hash returns SHA256:BLAKE3 format."""
        from meta import dual_hash

        result = dual_hash(b"test data")
        assert ":" in result, "dual_hash must contain ':' separator"
        parts = result.split(":")
        assert len(parts) == 2, "dual_hash must have exactly two parts"
        assert len(parts[0]) == 64, "SHA256 hex should be 64 chars"

    def test_payload_hash_in_all_receipts(self):
        """All receipts have dual_hash format payload_hash."""
        from meta import emit_meta_fitness, emit_energy_allocation, weight_future_proposals

        receipts = [
            emit_meta_fitness({"name": "p", "pre_metrics": {"roi": 0}, "post_metrics": {"roi": 0}}),
            emit_energy_allocation({"cycle_id": 1, "patterns": []}),
            weight_future_proposals([]),
        ]

        for r in receipts:
            assert "payload_hash" in r, f"Missing payload_hash in {r['receipt_type']}"
            assert ":" in r["payload_hash"], f"payload_hash not dual_hash format in {r['receipt_type']}"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """All required exports are available."""
        from meta import (
            RECEIPT_SCHEMA,
            DEFAULT_OBSERVATION_WINDOW_DAYS,
            MIN_PARADIGMS_FOR_WEIGHTING,
            SURVIVAL_WEIGHT_BONUS,
            POSITIVE_ROI_WEIGHT_BONUS,
            FAILURE_WEIGHT_PENALTY,
            compute_paradigm_delta,
            emit_meta_fitness,
            emit_energy_allocation,
            weight_future_proposals,
            stoprule_meta_fitness,
            stoprule_energy_allocation,
            stoprule_paradigm_outcome,
            stoprule_meta_tracking_failure,
            emit_receipt,
            dual_hash,
            StopRule,
        )

        # Just importing without error is the test
        assert callable(compute_paradigm_delta)
        assert callable(emit_meta_fitness)
        assert callable(emit_energy_allocation)
        assert callable(weight_future_proposals)
        assert callable(stoprule_meta_fitness)
        assert callable(stoprule_energy_allocation)
        assert callable(stoprule_paradigm_outcome)
        assert callable(stoprule_meta_tracking_failure)
        assert callable(emit_receipt)
        assert callable(dual_hash)

        # Check constants
        assert DEFAULT_OBSERVATION_WINDOW_DAYS == 30
        assert MIN_PARADIGMS_FOR_WEIGHTING == 3
        assert SURVIVAL_WEIGHT_BONUS == 0.4
        assert POSITIVE_ROI_WEIGHT_BONUS == 0.3
        assert FAILURE_WEIGHT_PENALTY == 0.2


class TestInternalTests:
    """Run the internal test functions defined in meta.py."""

    def test_internal_meta_fitness(self):
        """Run meta.py's internal test_meta_fitness."""
        from meta import test_meta_fitness
        test_meta_fitness()

    def test_internal_energy_allocation(self):
        """Run meta.py's internal test_energy_allocation."""
        from meta import test_energy_allocation
        test_energy_allocation()

    def test_internal_weight_future_proposals(self):
        """Run meta.py's internal test_weight_future_proposals."""
        from meta import test_weight_future_proposals
        test_weight_future_proposals()


class TestSmokeTests:
    """Smoke tests from the spec."""

    def test_h1_meta_exports_emit_meta_fitness_with_delta_roi(self):
        """H1: meta.py exports emit_meta_fitness, result has delta_roi field."""
        from meta import emit_meta_fitness

        paradigm = {"name": "test", "pre_metrics": {"roi": 0.5}, "post_metrics": {"roi": 0.8}}
        r = emit_meta_fitness(paradigm)

        assert "delta_roi" in r, "emit_meta_fitness result missing delta_roi field"

    def test_h2_energy_allocation_returns_correct_allocations_length(self):
        """H2: emit_energy_allocation with 2 patterns returns allocations list of length 2."""
        from meta import emit_energy_allocation

        cycle = {
            "cycle_id": 1,
            "patterns": [
                {"pattern_id": "p1", "fitness": 0.5, "resource_share": 50.0},
                {"pattern_id": "p2", "fitness": 0.5, "resource_share": 50.0}
            ]
        }
        r = emit_energy_allocation(cycle)

        assert len(r["allocations"]) == 2

    def test_h3_weight_future_proposals_returns_dict_with_floats(self):
        """H3: weight_future_proposals returns dict with float values."""
        from meta import weight_future_proposals

        r = weight_future_proposals([{"paradigm_name": "p", "delta_roi": 0.1, "survival_30d": True}])

        assert isinstance(r["proposed_weights"], dict)
        for v in r["proposed_weights"].values():
            assert isinstance(v, float)

    def test_h4_energy_share_values_sum_to_one(self):
        """H4: energy_share values sum to approximately 1.0."""
        from meta import emit_energy_allocation

        cycle = {
            "cycle_id": 1,
            "patterns": [
                {"pattern_id": "p1", "fitness": 0.8, "resource_share": 30.0},
                {"pattern_id": "p2", "fitness": 0.6, "resource_share": 40.0},
                {"pattern_id": "p3", "fitness": 0.4, "resource_share": 30.0}
            ]
        }
        r = emit_energy_allocation(cycle)

        total = sum(a["energy_share"] for a in r["allocations"])
        assert abs(total - 1.0) < 1e-10, f"Energy shares sum to {total}, expected 1.0"

    def test_h5_receipt_schema_contains_meta_fitness_receipt(self):
        """H5: RECEIPT_SCHEMA contains 'meta_fitness_receipt'."""
        from meta import RECEIPT_SCHEMA

        assert "meta_fitness_receipt" in RECEIPT_SCHEMA
