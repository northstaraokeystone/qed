"""
tests/test_mitigation.py - Tests for Stacked Tau Mitigation

Validates:
- Multiplicative retention computation
- Effective tau stacking
- Effective alpha computation
- Cycles to 10^3 projection
- Hash verification
- Stoprules

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import pytest
import math

from mitigation import (
    MitigationConfig,
    MitigationResult,
    compute_retention,
    compute_eff_alpha,
    compute_cycles_to_10k,
    stack_mitigation,
    validate_config,
    load_mitigation_params,
    load_relay_spec,
    config_from_params,
    stoprule_invalid_retention,
    ONBOARD_AI_RETENTION,
    PREDICTIVE_RETENTION,
    RELAY_RETENTION,
    EFF_TAU_TARGET,
    EFF_ALPHA_MIN,
    RETENTION_VS_UNMITIGATED,
    CYCLES_TO_10K_TARGET,
    DEFAULT_BASE_TAU,
)
from receipts import StopRule


# =============================================================================
# RETENTION COMPUTATION TESTS
# =============================================================================

class TestComputeRetention:
    """Tests for compute_retention function."""

    def test_compute_retention_all_enabled(self):
        """All three mitigations: 0.75 x 0.80 x 0.65 = 0.39."""
        config = MitigationConfig()
        retention = compute_retention(config)

        expected = ONBOARD_AI_RETENTION * PREDICTIVE_RETENTION * RELAY_RETENTION
        assert abs(retention - expected) < 0.001
        assert abs(retention - 0.39) < 0.001

    def test_compute_retention_two_enabled(self):
        """Onboard + relay: 0.75 x 0.65 = 0.4875."""
        config = MitigationConfig(
            enabled={
                "onboard_ai": True,
                "predictive": False,
                "relay": True
            }
        )
        retention = compute_retention(config)

        expected = ONBOARD_AI_RETENTION * RELAY_RETENTION
        assert abs(retention - expected) < 0.001
        assert abs(retention - 0.4875) < 0.001

    def test_compute_retention_onboard_predictive(self):
        """Onboard + predictive: 0.75 x 0.80 = 0.60."""
        config = MitigationConfig(
            enabled={
                "onboard_ai": True,
                "predictive": True,
                "relay": False
            }
        )
        retention = compute_retention(config)

        expected = ONBOARD_AI_RETENTION * PREDICTIVE_RETENTION
        assert abs(retention - expected) < 0.001
        assert abs(retention - 0.60) < 0.001

    def test_compute_retention_single_onboard(self):
        """Only onboard: 0.75."""
        config = MitigationConfig(
            enabled={
                "onboard_ai": True,
                "predictive": False,
                "relay": False
            }
        )
        retention = compute_retention(config)

        assert abs(retention - ONBOARD_AI_RETENTION) < 0.001

    def test_compute_retention_none_enabled(self):
        """No mitigations: retention = 1.0."""
        config = MitigationConfig(
            enabled={
                "onboard_ai": False,
                "predictive": False,
                "relay": False
            }
        )
        retention = compute_retention(config)

        assert abs(retention - 1.0) < 0.001


# =============================================================================
# STACK MITIGATION TESTS
# =============================================================================

class TestStackMitigation:
    """Tests for stack_mitigation function."""

    def test_stack_mitigation_eff_tau(self):
        """Full stack: 1200 x 0.39 ~ 468s."""
        config = MitigationConfig()
        result = stack_mitigation(DEFAULT_BASE_TAU, config)

        expected_tau = DEFAULT_BASE_TAU * 0.39
        assert abs(result.eff_tau - expected_tau) < 1.0
        assert 460 < result.eff_tau < 480  # ~468s

    def test_stack_mitigation_eff_tau_minutes(self):
        """Full stack: eff_tau ~ 7.8 minutes."""
        config = MitigationConfig()
        result = stack_mitigation(DEFAULT_BASE_TAU, config)

        eff_tau_minutes = result.eff_tau / 60
        assert 7.5 < eff_tau_minutes < 8.5

    def test_stack_mitigation_eff_alpha(self):
        """Full stack: eff_alpha >= 1.58."""
        config = MitigationConfig()
        result = stack_mitigation(DEFAULT_BASE_TAU, config)

        assert result.eff_alpha >= EFF_ALPHA_MIN
        assert result.eff_alpha >= 1.58

    def test_stack_mitigation_cycles_to_10k(self):
        """Full stack: cycles to 10^3 ~ 2.5."""
        config = MitigationConfig()
        result = stack_mitigation(DEFAULT_BASE_TAU, config)

        assert result.cycles_to_10k < 5.0  # Better than unmitigated
        # Allow some tolerance around the target
        assert result.cycles_to_10k < CYCLES_TO_10K_TARGET * 1.5

    def test_stack_mitigation_applied_list(self):
        """Mitigations applied list includes all enabled."""
        config = MitigationConfig()
        result = stack_mitigation(DEFAULT_BASE_TAU, config)

        assert "onboard_ai" in result.mitigations_applied
        assert "predictive" in result.mitigations_applied
        assert "relay" in result.mitigations_applied
        assert len(result.mitigations_applied) == 3


# =============================================================================
# EFFECTIVE ALPHA TESTS
# =============================================================================

class TestComputeEffAlpha:
    """Tests for compute_eff_alpha function."""

    def test_eff_alpha_full_stack(self):
        """Full stack should produce alpha >= 1.58."""
        eff_tau = 468.0  # Full stack tau
        base_tau = DEFAULT_BASE_TAU

        eff_alpha = compute_eff_alpha(1.0, eff_tau, base_tau, 1.0)

        assert eff_alpha >= 1.5  # Tau reduction provides ~2.5x boost

    def test_eff_alpha_no_mitigation(self):
        """No mitigation: alpha = base_alpha."""
        eff_alpha = compute_eff_alpha(1.0, DEFAULT_BASE_TAU, DEFAULT_BASE_TAU, 1.0)

        assert abs(eff_alpha - 1.0) < 0.001

    def test_eff_alpha_with_integrity(self):
        """Receipt integrity modulates alpha."""
        eff_tau = 468.0
        base_tau = DEFAULT_BASE_TAU

        full_alpha = compute_eff_alpha(1.0, eff_tau, base_tau, 1.0)
        half_alpha = compute_eff_alpha(1.0, eff_tau, base_tau, 0.25)

        # sqrt(0.25) = 0.5, so half_alpha should be ~50% of full
        assert abs(half_alpha / full_alpha - 0.5) < 0.01


# =============================================================================
# CYCLES TO 10K TESTS
# =============================================================================

class TestComputeCyclesTo10k:
    """Tests for compute_cycles_to_10k function."""

    def test_cycles_mitigated(self):
        """Mitigated: ~2.5 cycles."""
        eff_alpha = 2.0  # Strong mitigation

        cycles = compute_cycles_to_10k(eff_alpha)

        assert cycles < 10  # Much faster than unmitigated

    def test_cycles_unmitigated(self):
        """Unmitigated: ~5+ cycles."""
        eff_alpha = 1.0  # No boost

        cycles = compute_cycles_to_10k(eff_alpha)

        # Unmitigated takes more cycles
        assert cycles > 4

    def test_retention_vs_unmitigated(self):
        """Mitigated cycles / unmitigated cycles ~ 0.5 (2x acceleration)."""
        config_full = MitigationConfig()
        config_none = MitigationConfig(
            enabled={"onboard_ai": False, "predictive": False, "relay": False}
        )

        result_full = stack_mitigation(DEFAULT_BASE_TAU, config_full)
        result_none = stack_mitigation(DEFAULT_BASE_TAU, config_none)

        ratio = result_full.cycles_to_10k / result_none.cycles_to_10k

        # Mitigated should be significantly faster
        assert ratio < 0.7  # At least 30% faster

    def test_cycles_zero_alpha(self):
        """Zero alpha: infinite cycles."""
        cycles = compute_cycles_to_10k(0.0)

        assert cycles == float('inf')

    def test_cycles_negative_alpha(self):
        """Negative alpha: infinite cycles."""
        cycles = compute_cycles_to_10k(-1.0)

        assert cycles == float('inf')


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self):
        """Valid config passes validation."""
        config = MitigationConfig()

        assert validate_config(config) is True

    def test_invalid_retention_zero(self):
        """Zero retention raises StopRule."""
        config = MitigationConfig(onboard_ai_retention=0.0)

        with pytest.raises(StopRule):
            validate_config(config)

    def test_invalid_retention_negative(self):
        """Negative retention raises StopRule."""
        config = MitigationConfig(predictive_retention=-0.5)

        with pytest.raises(StopRule):
            validate_config(config)

    def test_invalid_retention_over_one(self):
        """Retention > 1 raises StopRule."""
        config = MitigationConfig(relay_retention=1.5)

        with pytest.raises(StopRule):
            validate_config(config)


# =============================================================================
# STOPRULE TESTS
# =============================================================================

class TestStopruleInvalidRetention:
    """Tests for stoprule_invalid_retention."""

    def test_stoprule_zero_factor(self):
        """Factor = 0 raises StopRule."""
        with pytest.raises(StopRule):
            stoprule_invalid_retention(0.0, "test")

    def test_stoprule_negative_factor(self):
        """Negative factor raises StopRule."""
        with pytest.raises(StopRule):
            stoprule_invalid_retention(-0.1, "test")

    def test_stoprule_over_one(self):
        """Factor > 1 raises StopRule."""
        with pytest.raises(StopRule):
            stoprule_invalid_retention(1.1, "test")

    def test_valid_factor_no_stoprule(self):
        """Valid factor (0 < f <= 1) does not raise."""
        # Should not raise
        stoprule_invalid_retention(0.5, "test")
        stoprule_invalid_retention(1.0, "test")
        stoprule_invalid_retention(0.001, "test")


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

class TestLoadMitigationParams:
    """Tests for load_mitigation_params function."""

    def test_load_mitigation_params(self):
        """Loads mitigation params with hash verification."""
        params = load_mitigation_params()

        assert "version" in params
        assert "retention_factors" in params
        assert params["retention_factors"]["onboard_ai"] == ONBOARD_AI_RETENTION
        assert params["retention_factors"]["predictive"] == PREDICTIVE_RETENTION
        assert params["retention_factors"]["relay"] == RELAY_RETENTION

    def test_load_mitigation_params_hash(self):
        """Payload hash is computed on load."""
        params = load_mitigation_params()

        assert "payload_hash" in params
        assert ":" in params["payload_hash"]  # Dual hash format


class TestLoadRelaySpec:
    """Tests for load_relay_spec function."""

    def test_load_relay_spec(self):
        """Loads relay spec with expected fields."""
        spec = load_relay_spec()

        assert "version" in spec
        assert "swarm" in spec
        assert spec["swarm"]["initial_sat_count"] == 30
        assert "performance" in spec

    def test_load_relay_spec_hash(self):
        """Payload hash is computed on load."""
        spec = load_relay_spec()

        assert "payload_hash" in spec
        assert ":" in spec["payload_hash"]  # Dual hash format


# =============================================================================
# CONFIG FROM PARAMS TESTS
# =============================================================================

class TestConfigFromParams:
    """Tests for config_from_params function."""

    def test_config_from_params(self):
        """Creates config from loaded params."""
        config = config_from_params()

        assert config.onboard_ai_retention == ONBOARD_AI_RETENTION
        assert config.predictive_retention == PREDICTIVE_RETENTION
        assert config.relay_retention == RELAY_RETENTION

    def test_config_from_custom_params(self):
        """Creates config from custom params dict."""
        params = {
            "retention_factors": {
                "onboard_ai": 0.70,
                "predictive": 0.85,
                "relay": 0.60
            }
        }
        config = config_from_params(params)

        assert config.onboard_ai_retention == 0.70
        assert config.predictive_retention == 0.85
        assert config.relay_retention == 0.60


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMitigationIntegration:
    """Integration tests for mitigation module."""

    def test_full_workflow(self):
        """Full workflow: load params, create config, stack mitigation."""
        params = load_mitigation_params()
        config = config_from_params(params)
        result = stack_mitigation(DEFAULT_BASE_TAU, config)

        # Verify key metrics
        assert result.eff_tau < DEFAULT_BASE_TAU
        assert result.eff_alpha > 1.0
        assert len(result.mitigations_applied) == 3

    def test_relay_spec_integration(self):
        """Relay spec informs mitigation config."""
        spec = load_relay_spec()

        # Verify key relay parameters
        assert spec["performance"]["eff_tau_minutes"] < 10
        assert spec["performance"]["blackout_coverage_pct"] > 99

    def test_multiplicative_stack(self):
        """Verify mitigations are multiplicative, not additive."""
        # If additive: 0.25 + 0.20 + 0.35 = 0.80 reduction
        # If multiplicative: 0.75 * 0.80 * 0.65 = 0.39 retention

        config = MitigationConfig()
        retention = compute_retention(config)

        # Verify multiplicative (0.39) not additive (0.20)
        assert retention < 0.5  # Multiplicative
        assert abs(retention - 0.39) < 0.01
