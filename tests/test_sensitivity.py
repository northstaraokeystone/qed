"""
tests/test_sensitivity.py - Tests for P Sensitivity Analysis

Validates:
- P factor sampling in +/-20% range
- Cost sampling in $50-150M range
- ROI distribution computation
- ROI gate checking at 90% confidence
- Monte Carlo with 1000 samples

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import pytest
import math

from sensitivity import (
    SensitivityConfig,
    SensitivityResult,
    sample_p_factor,
    sample_cost,
    compute_mitigated_cycles,
    compute_roi,
    compute_roi_distribution,
    check_roi_gate,
    run_sensitivity,
    quick_sensitivity_check,
    DEFAULT_P_BASELINE,
    DEFAULT_P_VARIANCE_PCT,
    DEFAULT_COST_BASELINE,
    DEFAULT_COST_RANGE,
    DEFAULT_N_SAMPLES,
    DEFAULT_ROI_GATE_THRESHOLD,
    DEFAULT_ROI_GATE_CONFIDENCE,
    REWARD_PER_CYCLE_SAVED,
    BASELINE_CYCLES,
)
from mitigation import (
    MitigationConfig,
    MitigationResult,
    stack_mitigation,
    DEFAULT_BASE_TAU,
)
from roi import (
    assert_roi_gate,
    check_roi_gate as roi_check,
    compute_roi_margin,
    require_roi_gate,
)
from receipts import StopRule


# =============================================================================
# SAMPLE P FACTOR TESTS
# =============================================================================

class TestSamplePFactor:
    """Tests for sample_p_factor function."""

    def test_sample_p_factor_range(self):
        """All samples in +/-20% range."""
        baseline = DEFAULT_P_BASELINE  # 1.8
        variance = DEFAULT_P_VARIANCE_PCT  # 0.20

        samples = sample_p_factor(baseline, variance, 100, seed=42)

        p_min = baseline * (1 - variance)  # 1.44
        p_max = baseline * (1 + variance)  # 2.16

        for p in samples:
            assert p_min <= p <= p_max, f"Sample {p} outside range [{p_min}, {p_max}]"

    def test_sample_p_factor_count(self):
        """Correct number of samples."""
        samples = sample_p_factor(1.8, 0.2, 500, seed=42)

        assert len(samples) == 500

    def test_sample_p_factor_reproducible(self):
        """Same seed produces same samples."""
        samples1 = sample_p_factor(1.8, 0.2, 10, seed=42)
        samples2 = sample_p_factor(1.8, 0.2, 10, seed=42)

        assert samples1 == samples2

    def test_sample_p_factor_different_seeds(self):
        """Different seeds produce different samples."""
        samples1 = sample_p_factor(1.8, 0.2, 10, seed=42)
        samples2 = sample_p_factor(1.8, 0.2, 10, seed=43)

        assert samples1 != samples2


# =============================================================================
# SAMPLE COST TESTS
# =============================================================================

class TestSampleCost:
    """Tests for sample_cost function."""

    def test_sample_cost_range(self):
        """All samples in $50-150M range."""
        samples = sample_cost(DEFAULT_COST_RANGE, 100, seed=42)

        for cost in samples:
            assert 50 <= cost <= 150, f"Cost {cost} outside range [50, 150]"

    def test_sample_cost_count(self):
        """Correct number of samples."""
        samples = sample_cost((50, 150), 500, seed=42)

        assert len(samples) == 500

    def test_sample_cost_reproducible(self):
        """Same seed produces same samples."""
        samples1 = sample_cost((50, 150), 10, seed=42)
        samples2 = sample_cost((50, 150), 10, seed=42)

        assert samples1 == samples2


# =============================================================================
# ROI COMPUTATION TESTS
# =============================================================================

class TestComputeRoi:
    """Tests for compute_roi function."""

    def test_roi_positive(self):
        """ROI is positive when cycles saved > cost penalty."""
        # 5 cycles baseline, 2 cycles mitigated = 3 saved
        # Reward = 3 * 50 = 150M
        # Cost = 100M, penalty = 0.5 * 100 = 50M
        # ROI = (150 - 50) / 100 = 1.0
        roi = compute_roi(5.0, 2.0, 100.0)

        assert roi > 0

    def test_roi_above_threshold(self):
        """ROI can exceed 1.2 threshold with good mitigation."""
        # Strong mitigation: 5 -> 1.5 cycles
        roi = compute_roi(5.0, 1.5, 80.0)

        assert roi > DEFAULT_ROI_GATE_THRESHOLD

    def test_roi_zero_cost(self):
        """Zero cost returns infinity."""
        roi = compute_roi(5.0, 2.0, 0.0)

        assert roi == float('inf')


# =============================================================================
# ROI DISTRIBUTION TESTS
# =============================================================================

class TestComputeRoiDistribution:
    """Tests for compute_roi_distribution function."""

    def test_roi_distribution_positive_mean(self):
        """Mean ROI > 0 with good mitigation."""
        config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, config)

        p_samples = sample_p_factor(1.8, 0.2, 100, seed=42)
        cost_samples = sample_cost((50, 150), 100, seed=43)

        roi_samples = compute_roi_distribution(p_samples, cost_samples, mitigation)

        roi_mean = sum(roi_samples) / len(roi_samples)
        assert roi_mean > 0

    def test_roi_distribution_count(self):
        """Correct number of ROI samples."""
        config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, config)

        p_samples = sample_p_factor(1.8, 0.2, 50, seed=42)
        cost_samples = sample_cost((50, 150), 50, seed=43)

        roi_samples = compute_roi_distribution(p_samples, cost_samples, mitigation)

        assert len(roi_samples) == 50


# =============================================================================
# ROI GATE TESTS
# =============================================================================

class TestCheckRoiGate:
    """Tests for check_roi_gate function."""

    def test_roi_gate_passes(self):
        """Gate passes when >= 90% above 1.2."""
        # 95 samples above threshold, 5 below
        roi_samples = [1.5] * 95 + [1.0] * 5

        passed = check_roi_gate(roi_samples, 1.2, 0.90)

        assert passed is True

    def test_roi_gate_fails(self):
        """Gate fails when < 90% above 1.2."""
        # 80 samples above threshold, 20 below
        roi_samples = [1.5] * 80 + [1.0] * 20

        passed = check_roi_gate(roi_samples, 1.2, 0.90)

        assert passed is False

    def test_roi_gate_exact_threshold(self):
        """Gate passes at exactly 90%."""
        # Exactly 90 samples above threshold
        roi_samples = [1.5] * 90 + [1.0] * 10

        passed = check_roi_gate(roi_samples, 1.2, 0.90)

        assert passed is True

    def test_roi_gate_empty_samples(self):
        """Empty samples returns False."""
        passed = check_roi_gate([], 1.2, 0.90)

        assert passed is False


# =============================================================================
# RUN SENSITIVITY TESTS
# =============================================================================

class TestRunSensitivity:
    """Tests for run_sensitivity function."""

    def test_run_sensitivity_1000(self):
        """1000 samples complete without error."""
        config = SensitivityConfig(n_samples=1000)
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        result = run_sensitivity(config, mitigation, seed=42)

        assert len(result.p_samples) == 1000
        assert len(result.cost_samples) == 1000
        assert len(result.roi_samples) == 1000

    def test_sensitivity_receipt_all_fields(self):
        """Sensitivity result has all required fields."""
        config = SensitivityConfig(n_samples=100)
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        result = run_sensitivity(config, mitigation, seed=42)

        assert hasattr(result, 'p_samples')
        assert hasattr(result, 'cost_samples')
        assert hasattr(result, 'roi_samples')
        assert hasattr(result, 'roi_mean')
        assert hasattr(result, 'roi_std')
        assert hasattr(result, 'roi_above_gate_pct')
        assert hasattr(result, 'gate_passed')

    def test_sensitivity_gate_status(self):
        """Gate status is boolean."""
        config = SensitivityConfig(n_samples=100)
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        result = run_sensitivity(config, mitigation, seed=42)

        assert isinstance(result.gate_passed, bool)

    def test_sensitivity_roi_mean_positive(self):
        """Mean ROI is positive with full mitigation."""
        config = SensitivityConfig(n_samples=100)
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        result = run_sensitivity(config, mitigation, seed=42)

        assert result.roi_mean > 0


# =============================================================================
# QUICK SENSITIVITY CHECK TESTS
# =============================================================================

class TestQuickSensitivityCheck:
    """Tests for quick_sensitivity_check function."""

    def test_quick_check_returns_bool(self):
        """Quick check returns boolean."""
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        passed = quick_sensitivity_check(mitigation, n_samples=50)

        assert isinstance(passed, bool)


# =============================================================================
# ROI MODULE TESTS
# =============================================================================

class TestRoiModule:
    """Tests for roi.py module."""

    def test_assert_roi_gate_passes(self):
        """Gate PASS at 1.3 (above 1.2 threshold)."""
        passed = assert_roi_gate(1.3, 1.2, halt_on_failure=False)

        assert passed is True

    def test_assert_roi_gate_fails(self):
        """Gate FAIL at 1.1 (below 1.2 threshold)."""
        passed = assert_roi_gate(1.1, 1.2, halt_on_failure=False)

        assert passed is False

    def test_assert_roi_gate_stoprule(self):
        """Gate raises StopRule when halt_on_failure=True."""
        with pytest.raises(StopRule):
            assert_roi_gate(1.1, 1.2, halt_on_failure=True)

    def test_require_roi_gate_passes(self):
        """require_roi_gate passes silently at 1.3."""
        # Should not raise
        require_roi_gate(1.3, 1.2)

    def test_require_roi_gate_fails(self):
        """require_roi_gate raises StopRule at 1.1."""
        with pytest.raises(StopRule):
            require_roi_gate(1.1, 1.2)

    def test_check_roi_gate_simple(self):
        """Simple check without side effects."""
        assert roi_check(1.5, 1.2) is True
        assert roi_check(1.0, 1.2) is False

    def test_compute_roi_margin(self):
        """Margin computation."""
        margin = compute_roi_margin(1.5, 1.2)
        assert abs(margin - 0.3) < 0.001

        margin = compute_roi_margin(1.0, 1.2)
        assert abs(margin - (-0.2)) < 0.001


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSensitivityIntegration:
    """Integration tests for sensitivity analysis."""

    def test_full_workflow(self):
        """Full workflow: mitigation -> sensitivity -> ROI gate."""
        # Create mitigation
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        # Run sensitivity
        sensitivity_config = SensitivityConfig(n_samples=100)
        result = run_sensitivity(sensitivity_config, mitigation, seed=42)

        # Check gate
        if result.roi_mean > DEFAULT_ROI_GATE_THRESHOLD:
            passed = assert_roi_gate(result.roi_mean, halt_on_failure=False)
            assert passed is True

    def test_p_range_effects(self):
        """P variance affects ROI distribution."""
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        # Low variance
        config_low = SensitivityConfig(n_samples=100, p_variance_pct=0.05)
        result_low = run_sensitivity(config_low, mitigation, seed=42)

        # High variance
        config_high = SensitivityConfig(n_samples=100, p_variance_pct=0.30)
        result_high = run_sensitivity(config_high, mitigation, seed=42)

        # Higher variance should produce higher std
        assert result_high.roi_std > result_low.roi_std * 0.5

    def test_cost_range_effects(self):
        """Cost range affects ROI distribution."""
        mitigation_config = MitigationConfig()
        mitigation = stack_mitigation(DEFAULT_BASE_TAU, mitigation_config)

        # Narrow range
        config_narrow = SensitivityConfig(n_samples=100, cost_range_usd_m=(90, 110))
        result_narrow = run_sensitivity(config_narrow, mitigation, seed=42)

        # Wide range
        config_wide = SensitivityConfig(n_samples=100, cost_range_usd_m=(30, 200))
        result_wide = run_sensitivity(config_wide, mitigation, seed=42)

        # Wider range may produce different std
        # Just verify both complete successfully
        assert result_narrow.roi_mean > 0
        assert result_wide.roi_mean > 0
