"""
tests/test_fitness.py - Tests for fitness.py (v11)

CLAUDEME v3.1 Compliant: Every test has assert statements.

Tests:
1. multi_fitness returns receipt with score in [0,1]
2. All four components present and each in [0,1]
3. Weights sum to 1.0 (0.4 + 0.3 + 0.2 + 0.1)
4. cohort_balanced_review blocks kill when count == 1
5. cohort_balanced_review allows kill when count > 1
6. thompson_sample returns survivors and superposition lists
7. High variance pattern can survive even with low mean (exploration)
8. RECEIPT_SCHEMA exported with all three receipt types
9. Pattern with zero ROI but high diversity survives (multi-dimensional prevents death spiral)
10. SUPERPOSITION patterns are not in survivors list but are tracked
"""

import math
import random

import pytest


class TestMultiFitness:
    """Tests for multi_fitness function."""

    def test_multi_fitness_returns_score_in_range(self):
        """Test 1: multi_fitness returns receipt with score in [0,1]."""
        from fitness import multi_fitness

        pattern = {
            "id": "test_pattern",
            "archetype": "explorer",
            "fitness_mean": 0.5,
            "fitness_var": 0.1
        }
        result = multi_fitness(pattern, tenant_id="test_tenant")

        assert result["receipt_type"] == "multi_fitness_score"
        assert 0.0 <= result["score"] <= 1.0, f"Score {result['score']} not in [0,1]"

    def test_multi_fitness_all_components_present(self):
        """Test 2: All four components present and each in [0,1]."""
        from fitness import multi_fitness

        pattern = {
            "id": "test_pattern",
            "archetype": "builder",
            "fitness_mean": 1.0,
            "fitness_var": 0.5,
            "last_receipt_ts": "2025-12-09T00:00:00Z"
        }
        result = multi_fitness(pattern, tenant_id="test_tenant")

        components = result["components"]
        required_keys = ["roi", "diversity", "stability", "recency"]

        for key in required_keys:
            assert key in components, f"Missing component: {key}"
            assert 0.0 <= components[key] <= 1.0, f"Component {key}={components[key]} not in [0,1]"

    def test_weights_sum_to_one(self):
        """Test 3: Weights sum to 1.0 (0.4 + 0.3 + 0.2 + 0.1)."""
        from fitness import (
            WEIGHT_ROI, WEIGHT_DIVERSITY, WEIGHT_STABILITY, WEIGHT_RECENCY,
            multi_fitness
        )

        # Test constants
        total = WEIGHT_ROI + WEIGHT_DIVERSITY + WEIGHT_STABILITY + WEIGHT_RECENCY
        assert abs(total - 1.0) < 1e-10, f"Weights sum to {total}, not 1.0"

        # Test receipt contains correct weights
        pattern = {"id": "p1", "archetype": "test"}
        result = multi_fitness(pattern)
        weights = result["weights"]

        assert weights["roi"] == 0.4, f"ROI weight is {weights['roi']}, expected 0.4"
        assert weights["diversity"] == 0.3, f"Diversity weight is {weights['diversity']}, expected 0.3"
        assert weights["stability"] == 0.2, f"Stability weight is {weights['stability']}, expected 0.2"
        assert weights["recency"] == 0.1, f"Recency weight is {weights['recency']}, expected 0.1"


class TestCohortBalancedReview:
    """Tests for cohort_balanced_review function."""

    def test_blocks_kill_when_count_equals_one(self):
        """Test 4: cohort_balanced_review blocks kill when count == 1."""
        from fitness import cohort_balanced_review

        pattern = {"id": "last_explorer", "archetype": "explorer"}
        population = [
            {"id": "last_explorer", "archetype": "explorer"},  # Only one explorer
            {"id": "builder1", "archetype": "builder"},
            {"id": "builder2", "archetype": "builder"}
        ]

        result = cohort_balanced_review(pattern, population, tenant_id="test_tenant")

        assert result["receipt_type"] == "archetype_protection"
        assert result["action"] == "block", f"Expected 'block' for last of archetype, got '{result['action']}'"
        assert result["would_be_last"] is True
        assert result["count_before"] == 1

    def test_allows_kill_when_count_greater_than_one(self):
        """Test 5: cohort_balanced_review allows kill when count > 1."""
        from fitness import cohort_balanced_review

        pattern = {"id": "explorer1", "archetype": "explorer"}
        population = [
            {"id": "explorer1", "archetype": "explorer"},
            {"id": "explorer2", "archetype": "explorer"},  # Another explorer exists
            {"id": "builder1", "archetype": "builder"}
        ]

        result = cohort_balanced_review(pattern, population, tenant_id="test_tenant")

        assert result["action"] == "allow", f"Expected 'allow' when more archetypes exist, got '{result['action']}'"
        assert result["would_be_last"] is False
        assert result["count_before"] == 2


class TestThompsonSample:
    """Tests for thompson_sample function."""

    def test_returns_survivors_and_superposition_lists(self):
        """Test 6: thompson_sample returns survivors and superposition lists."""
        from fitness import thompson_sample

        patterns = [
            {"id": "p1", "fitness_mean": 1.0, "fitness_var": 0.01},
            {"id": "p2", "fitness_mean": -1.0, "fitness_var": 0.01}
        ]

        result = thompson_sample(patterns, tenant_id="test_tenant")

        assert result["receipt_type"] == "selection_event"
        assert isinstance(result["survivors"], list), "survivors should be a list"
        assert isinstance(result["superposition"], list), "superposition should be a list"
        assert result["patterns_evaluated"] == 2

    def test_high_variance_exploration(self):
        """Test 7: High variance pattern can survive even with low mean (exploration)."""
        from fitness import thompson_sample

        # Run many trials - high variance should sometimes sample above threshold
        # Pattern with mean=-0.5 but very high variance could sample > 0
        patterns = [
            {"id": "high_var", "fitness_mean": -0.5, "fitness_var": 4.0}  # std=2, can easily sample >0
        ]

        # Run 100 trials - with mean=-0.5 and std=2, P(sample > 0) is about 40%
        # So we expect at least some survivals
        survivals = 0
        for _ in range(100):
            result = thompson_sample(patterns, survival_threshold=0.0)
            if "high_var" in result["survivors"]:
                survivals += 1

        # Should have some survivals due to exploration (high variance)
        assert survivals > 0, "High variance pattern should sometimes survive due to exploration"
        # But not all, since mean is negative
        assert survivals < 100, "Pattern should not always survive with negative mean"

    def test_superposition_patterns_tracked(self):
        """Test 10: SUPERPOSITION patterns are not in survivors list but are tracked."""
        from fitness import thompson_sample

        # Use patterns with very low variance so results are deterministic
        patterns = [
            {"id": "survivor", "fitness_mean": 1.0, "fitness_var": 0.0001},
            {"id": "superposition_candidate", "fitness_mean": -1.0, "fitness_var": 0.0001}
        ]

        result = thompson_sample(patterns, survival_threshold=0.0)

        # Verify separation
        assert "survivor" in result["survivors"]
        assert "superposition_candidate" in result["superposition"]

        # Verify no overlap
        for pattern_id in result["superposition"]:
            assert pattern_id not in result["survivors"], \
                f"Pattern {pattern_id} in both survivors and superposition"


class TestReceiptSchema:
    """Tests for RECEIPT_SCHEMA export."""

    def test_receipt_schema_exported(self):
        """Test 8: RECEIPT_SCHEMA exported with all three receipt types."""
        from fitness import RECEIPT_SCHEMA

        assert len(RECEIPT_SCHEMA) == 3, f"Expected 3 receipt types, got {len(RECEIPT_SCHEMA)}"
        assert "multi_fitness_score" in RECEIPT_SCHEMA
        assert "selection_event" in RECEIPT_SCHEMA
        assert "archetype_protection" in RECEIPT_SCHEMA


class TestMultiDimensionalPreventsDeathSpiral:
    """Tests for multi-dimensional fitness preventing death spirals."""

    def test_zero_roi_high_diversity_survives(self):
        """Test 9: Pattern with zero ROI but high diversity survives (multi-dimensional prevents death spiral)."""
        from fitness import multi_fitness

        # Pattern with zero ROI but is sole representative of its archetype
        pattern = {
            "id": "unique_pattern",
            "archetype": "rare_archetype",
            "roi": 0.0,  # Zero ROI contribution
            "fitness_mean": 0.0,
            "fitness_var": 0.0,  # High stability (low variance)
        }
        # No other patterns of same archetype -> diversity = 1.0
        population = [pattern]

        result = multi_fitness(pattern, population=population)

        # With zero ROI (maps to ~0.5 via sigmoid), diversity=1.0, stability=1.0, recency=0.5
        # Score = 0.4*0.5 + 0.3*1.0 + 0.2*1.0 + 0.1*0.5 = 0.2 + 0.3 + 0.2 + 0.05 = 0.75
        # Pattern should have a reasonable score despite zero ROI
        assert result["score"] > 0.5, \
            f"Pattern with zero ROI but high diversity should score > 0.5, got {result['score']}"

        # Verify components
        assert result["components"]["diversity"] == 1.0, "Unique archetype should have diversity=1.0"


class TestTenantIdPresent:
    """Tests that tenant_id is present in all emitted receipts."""

    def test_tenant_id_in_multi_fitness(self):
        """multi_fitness receipt has tenant_id."""
        from fitness import multi_fitness

        r = multi_fitness({"id": "p1", "archetype": "test"}, tenant_id="test_tenant")
        assert "tenant_id" in r, "Missing tenant_id in multi_fitness"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_cohort_balanced_review(self):
        """cohort_balanced_review receipt has tenant_id."""
        from fitness import cohort_balanced_review

        pattern = {"id": "p1", "archetype": "test"}
        population = [pattern]
        r = cohort_balanced_review(pattern, population, tenant_id="test_tenant")
        assert "tenant_id" in r, "Missing tenant_id in cohort_balanced_review"
        assert r["tenant_id"] == "test_tenant"

    def test_tenant_id_in_thompson_sample(self):
        """thompson_sample receipt has tenant_id."""
        from fitness import thompson_sample

        patterns = [{"id": "p1", "fitness_mean": 1.0, "fitness_var": 0.0}]
        r = thompson_sample(patterns, tenant_id="test_tenant")
        assert "tenant_id" in r, "Missing tenant_id in thompson_sample"
        assert r["tenant_id"] == "test_tenant"


class TestDualHash:
    """Tests for dual_hash compliance."""

    def test_dual_hash_format(self):
        """dual_hash returns SHA256:BLAKE3 format."""
        from fitness import dual_hash

        result = dual_hash(b"test data")
        assert ":" in result, "dual_hash must contain ':' separator"
        parts = result.split(":")
        assert len(parts) == 2, "dual_hash must have exactly two parts"
        assert len(parts[0]) == 64, "SHA256 hex should be 64 chars"

    def test_payload_hash_in_receipts(self):
        """All receipts have dual_hash format payload_hash."""
        from fitness import multi_fitness, cohort_balanced_review, thompson_sample

        receipts = [
            multi_fitness({"id": "p1", "archetype": "test"}),
            cohort_balanced_review({"id": "p1", "archetype": "test"}, [{"id": "p1", "archetype": "test"}]),
            thompson_sample([{"id": "p1", "fitness_mean": 0.0, "fitness_var": 0.0}]),
        ]

        for r in receipts:
            assert "payload_hash" in r, f"Missing payload_hash in {r['receipt_type']}"
            assert ":" in r["payload_hash"], f"payload_hash not dual_hash format in {r['receipt_type']}"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """All required exports are available."""
        from fitness import (
            RECEIPT_SCHEMA,
            WEIGHT_ROI,
            WEIGHT_DIVERSITY,
            WEIGHT_STABILITY,
            WEIGHT_RECENCY,
            MIN_ARCHETYPE_DIVERSITY,
            DEFAULT_SURVIVAL_THRESHOLD,
            RECENCY_DECAY_DAYS,
            multi_fitness,
            cohort_balanced_review,
            thompson_sample,
            stoprule_multi_fitness,
            stoprule_archetype_protection,
            stoprule_selection_event,
            emit_receipt,
            dual_hash,
            StopRule,
        )

        # Constants
        assert WEIGHT_ROI == 0.4
        assert WEIGHT_DIVERSITY == 0.3
        assert WEIGHT_STABILITY == 0.2
        assert WEIGHT_RECENCY == 0.1
        assert MIN_ARCHETYPE_DIVERSITY == 1
        assert DEFAULT_SURVIVAL_THRESHOLD == 0.0
        assert RECENCY_DECAY_DAYS == 30

        # Functions
        assert callable(multi_fitness)
        assert callable(cohort_balanced_review)
        assert callable(thompson_sample)
        assert callable(stoprule_multi_fitness)
        assert callable(stoprule_archetype_protection)
        assert callable(stoprule_selection_event)
        assert callable(emit_receipt)
        assert callable(dual_hash)


class TestInternalTests:
    """Run the internal test functions defined in fitness.py."""

    def test_internal_multi_fitness_score(self):
        """Run fitness.py's internal test_multi_fitness_score."""
        from fitness import test_multi_fitness_score
        test_multi_fitness_score()

    def test_internal_archetype_protection(self):
        """Run fitness.py's internal test_archetype_protection."""
        from fitness import test_archetype_protection
        test_archetype_protection()

    def test_internal_selection_event(self):
        """Run fitness.py's internal test_selection_event."""
        from fitness import test_selection_event
        test_selection_event()


class TestSmokeTests:
    """Smoke tests for fitness.py (H1-H5)."""

    def test_h1_multi_fitness_score_in_range(self):
        """H1: fitness.py exports multi_fitness, result has score field in [0,1]."""
        from fitness import multi_fitness

        result = multi_fitness({"id": "smoke_test", "archetype": "test"})
        assert "score" in result, "multi_fitness result missing 'score' field"
        assert 0.0 <= result["score"] <= 1.0, f"Score {result['score']} not in [0,1]"

    def test_h2_multi_fitness_components(self):
        """H2: multi_fitness components dict has all four keys: roi, diversity, stability, recency."""
        from fitness import multi_fitness

        result = multi_fitness({"id": "smoke_test", "archetype": "test"})
        assert "components" in result, "multi_fitness result missing 'components' field"

        components = result["components"]
        for key in ["roi", "diversity", "stability", "recency"]:
            assert key in components, f"components missing '{key}' key"

    def test_h3_cohort_balanced_review_blocks_last(self):
        """H3: cohort_balanced_review returns action='block' for last-of-archetype."""
        from fitness import cohort_balanced_review

        pattern = {"id": "last_one", "archetype": "unique"}
        population = [pattern]  # Only one of this archetype

        result = cohort_balanced_review(pattern, population)
        assert result["action"] == "block", f"Expected 'block' for last-of-archetype, got '{result['action']}'"

    def test_h4_thompson_sample_returns_lists(self):
        """H4: thompson_sample returns dict with survivors and superposition lists."""
        from fitness import thompson_sample

        patterns = [{"id": "p1", "fitness_mean": 0.5, "fitness_var": 0.1}]
        result = thompson_sample(patterns)

        assert "survivors" in result, "thompson_sample result missing 'survivors'"
        assert "superposition" in result, "thompson_sample result missing 'superposition'"
        assert isinstance(result["survivors"], list), "'survivors' should be a list"
        assert isinstance(result["superposition"], list), "'superposition' should be a list"

    def test_h5_receipt_schema_contains_all_types(self):
        """H5: RECEIPT_SCHEMA contains all three receipt types."""
        from fitness import RECEIPT_SCHEMA

        expected_types = ["multi_fitness_score", "selection_event", "archetype_protection"]
        for receipt_type in expected_types:
            assert receipt_type in RECEIPT_SCHEMA, f"RECEIPT_SCHEMA missing '{receipt_type}'"
