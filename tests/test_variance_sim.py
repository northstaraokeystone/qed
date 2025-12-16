"""
tests/test_variance_sim.py - Variance Simulation Tests

Validates dynamic thresholding and variance sweep functionality.
CLAUDEME v3.1 Compliant: Tests with assertions.
"""

import pytest
import math

from sim.constants import (
    MIN_AFFINITY_THRESHOLD,
    STOCHASTIC_AFFINITY_THRESHOLD,
    VARIANCE_SWEEP_STEPS,
)
from sim.dynamics_genesis import compute_dynamic_threshold, get_affinity_with_variance
from sim.cycle import run_variance_sweep, scenario_stochastic_affinity
from sim.types_config import SimConfig, SCENARIO_STOCHASTIC_AFFINITY


class TestDynamicThreshold:
    """Test compute_dynamic_threshold function."""

    def test_returns_deterministic_when_variance_zero(self):
        """Dynamic threshold returns deterministic value when variance=0."""
        threshold = compute_dynamic_threshold(0.0)
        assert threshold == MIN_AFFINITY_THRESHOLD, (
            f"Expected {MIN_AFFINITY_THRESHOLD}, got {threshold}"
        )

    def test_increases_when_variance_positive(self):
        """Dynamic threshold increases when variance>0."""
        threshold_zero = compute_dynamic_threshold(0.0)
        threshold_positive = compute_dynamic_threshold(0.05)
        assert threshold_positive > threshold_zero, (
            f"Stochastic threshold {threshold_positive} should be > deterministic {threshold_zero}"
        )

    def test_scales_with_variance(self):
        """Higher variance produces higher threshold."""
        threshold_low = compute_dynamic_threshold(0.05)
        threshold_high = compute_dynamic_threshold(0.10)
        assert threshold_high >= threshold_low, (
            f"Threshold at 0.10 ({threshold_high}) should be >= threshold at 0.05 ({threshold_low})"
        )

    def test_clamped_at_floor(self):
        """Threshold is clamped at MIN_AFFINITY_THRESHOLD floor."""
        # Negative variance should still return floor
        threshold = compute_dynamic_threshold(-1.0)
        assert threshold >= MIN_AFFINITY_THRESHOLD, (
            f"Threshold {threshold} should be >= floor {MIN_AFFINITY_THRESHOLD}"
        )

    def test_clamped_at_ceiling(self):
        """Threshold is clamped at 0.95 ceiling."""
        # Very high variance should hit ceiling
        threshold = compute_dynamic_threshold(100.0)
        assert threshold <= 0.95, (
            f"Threshold {threshold} should be <= ceiling 0.95"
        )

    def test_no_nan_values(self):
        """Threshold computation never produces NaN."""
        test_values = [0.0, 0.01, 0.05, 0.10, 0.50, 1.0, -0.5]
        for v in test_values:
            threshold = compute_dynamic_threshold(v)
            assert not math.isnan(threshold), f"NaN produced for variance={v}"

    def test_stochastic_base_correct(self):
        """Stochastic mode starts from STOCHASTIC_AFFINITY_THRESHOLD."""
        # Small positive variance should be at or above stochastic base
        threshold = compute_dynamic_threshold(0.001)
        assert threshold >= STOCHASTIC_AFFINITY_THRESHOLD, (
            f"Threshold {threshold} should be >= stochastic base {STOCHASTIC_AFFINITY_THRESHOLD}"
        )


class TestGetAffinityWithVariance:
    """Test get_affinity_with_variance function."""

    def test_same_domain_returns_one(self):
        """Same domain always returns 1.0."""
        affinity = get_affinity_with_variance("tesla", "tesla", 0.05)
        assert affinity == 1.0

    def test_deterministic_when_zero_variance(self):
        """Returns base affinity when variance is 0."""
        affinity = get_affinity_with_variance("tesla", "spacex", 0.0)
        # Tesla-SpaceX base affinity is 0.7
        assert affinity == 0.7

    def test_clamped_to_valid_range(self):
        """Result is always clamped to [0.0, 1.0]."""
        # Run multiple times with high variance
        for _ in range(100):
            affinity = get_affinity_with_variance("tesla", "spacex", 0.5)
            assert 0.0 <= affinity <= 1.0, f"Affinity {affinity} out of range"


class TestVarianceSweep:
    """Test run_variance_sweep function."""

    def test_returns_expected_number_of_results(self):
        """Variance sweep returns VARIANCE_SWEEP_STEPS + 1 results."""
        config = SimConfig(n_cycles=10, random_seed=42)  # Minimal cycles for speed
        results = run_variance_sweep(config)
        expected = VARIANCE_SWEEP_STEPS + 1
        assert len(results) == expected, (
            f"Expected {expected} sweep points, got {len(results)}"
        )

    def test_each_result_has_required_keys(self):
        """Each sweep result has required keys."""
        config = SimConfig(n_cycles=10, random_seed=42)
        results = run_variance_sweep(config)
        required_keys = [
            "variance_level",
            "threshold",
            "final_population",
            "violations",
            "completeness_achieved",
            "births",
            "deaths",
        ]
        for result in results:
            for key in required_keys:
                assert key in result, f"Missing key '{key}' in sweep result"

    def test_variance_levels_increase(self):
        """Variance levels increase monotonically across sweep."""
        config = SimConfig(n_cycles=10, random_seed=42)
        results = run_variance_sweep(config)
        variance_levels = [r["variance_level"] for r in results]
        for i in range(1, len(variance_levels)):
            assert variance_levels[i] >= variance_levels[i - 1], (
                f"Variance levels not monotonic at index {i}"
            )


class TestScenarioStochasticAffinity:
    """Test scenario_stochastic_affinity function."""

    def test_scenario_passes(self):
        """Stochastic affinity scenario passes with default config."""
        result = scenario_stochastic_affinity()
        assert result.passed, f"Scenario failed: {result.message}"

    def test_scenario_returns_details(self):
        """Scenario result contains expected detail keys."""
        result = scenario_stochastic_affinity()
        assert "threshold_deterministic" in result.details
        assert "threshold_stochastic_low" in result.details
        assert "final_population" in result.details

    def test_custom_config_works(self):
        """Scenario accepts custom config."""
        config = SimConfig(n_cycles=50, random_seed=99)
        result = scenario_stochastic_affinity(config)
        # Should still pass with minimal cycles
        assert result.passed or "degraded" in result.message.lower()

    def test_thresholds_scale_correctly(self):
        """Validates threshold scaling is correct in details."""
        result = scenario_stochastic_affinity()
        if result.passed:
            det = result.details["threshold_deterministic"]
            stoch = result.details["threshold_stochastic_low"]
            assert stoch > det, "Stochastic threshold should exceed deterministic"
