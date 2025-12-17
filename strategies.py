"""
strategies.py - Mitigation Strategy Application

Defines strategies for tau mitigation and applies them via the mitigation stack.

CLAUDEME v3.1 Compliant: Uses mitigation.stack_mitigation() for COMBINED strategy.

Source: Grok insight 2025-01
- "Deploy combined tau strategies first"
- "integrate onboard AI, predictive sims, and relays"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from receipts import emit_receipt
from mitigation import (
    MitigationConfig,
    MitigationResult,
    stack_mitigation,
    DEFAULT_BASE_TAU,
    DEFAULT_BASE_ALPHA,
    ONBOARD_AI_RETENTION,
    PREDICTIVE_RETENTION,
    RELAY_RETENTION,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Module exports for receipt types
RECEIPT_SCHEMA = ["strategy_applied"]


# =============================================================================
# ENUMS
# =============================================================================

class Strategy(Enum):
    """Available mitigation strategies."""
    NONE = "none"                    # No mitigation (baseline)
    ONBOARD_ONLY = "onboard_only"    # Only onboard AI
    PREDICTIVE_ONLY = "predictive"   # Only predictive sims
    RELAY_ONLY = "relay"             # Only relay network
    ONBOARD_PREDICTIVE = "onboard_predictive"  # Onboard + predictive
    ONBOARD_RELAY = "onboard_relay"  # Onboard + relay
    PREDICTIVE_RELAY = "predictive_relay"  # Predictive + relay
    COMBINED = "combined"            # Full stack (all three)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StrategyResult:
    """Result from strategy application."""
    strategy: Strategy
    mitigation: MitigationResult
    description: str


# =============================================================================
# RECEIPT TYPE 1: strategy_applied
# =============================================================================

# --- SCHEMA ---
STRATEGY_APPLIED_SCHEMA = {
    "receipt_type": "strategy_applied",
    "ts": "ISO8601",
    "tenant_id": "str",
    "strategy": "str",
    "eff_tau": "float",
    "eff_alpha": "float",
    "mitigations_enabled": "list[str]",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_strategy_applied_receipt(
    tenant_id: str,
    strategy: str,
    eff_tau: float,
    eff_alpha: float,
    mitigations_enabled: list
) -> dict:
    """Emit strategy_applied receipt."""
    return emit_receipt("strategy_applied", {
        "tenant_id": tenant_id,
        "strategy": strategy,
        "eff_tau": eff_tau,
        "eff_alpha": eff_alpha,
        "mitigations_enabled": mitigations_enabled
    })


# =============================================================================
# STRATEGY CONFIGURATIONS
# =============================================================================

STRATEGY_CONFIGS = {
    Strategy.NONE: {
        "onboard_ai": False,
        "predictive": False,
        "relay": False
    },
    Strategy.ONBOARD_ONLY: {
        "onboard_ai": True,
        "predictive": False,
        "relay": False
    },
    Strategy.PREDICTIVE_ONLY: {
        "onboard_ai": False,
        "predictive": True,
        "relay": False
    },
    Strategy.RELAY_ONLY: {
        "onboard_ai": False,
        "predictive": False,
        "relay": True
    },
    Strategy.ONBOARD_PREDICTIVE: {
        "onboard_ai": True,
        "predictive": True,
        "relay": False
    },
    Strategy.ONBOARD_RELAY: {
        "onboard_ai": True,
        "predictive": False,
        "relay": True
    },
    Strategy.PREDICTIVE_RELAY: {
        "onboard_ai": False,
        "predictive": True,
        "relay": True
    },
    Strategy.COMBINED: {
        "onboard_ai": True,
        "predictive": True,
        "relay": True
    },
}

STRATEGY_DESCRIPTIONS = {
    Strategy.NONE: "No mitigation (baseline)",
    Strategy.ONBOARD_ONLY: "Onboard AI only (75% retention)",
    Strategy.PREDICTIVE_ONLY: "Predictive sims only (80% retention)",
    Strategy.RELAY_ONLY: "Relay network only (65% retention)",
    Strategy.ONBOARD_PREDICTIVE: "Onboard AI + Predictive sims",
    Strategy.ONBOARD_RELAY: "Onboard AI + Relay network",
    Strategy.PREDICTIVE_RELAY: "Predictive sims + Relay network",
    Strategy.COMBINED: "Full stack: Onboard + Predictive + Relay (39% retention)",
}


# =============================================================================
# CORE FUNCTION 1: get_strategy_config
# =============================================================================

def get_strategy_config(strategy: Strategy) -> MitigationConfig:
    """
    Get MitigationConfig for a strategy.

    Args:
        strategy: Strategy enum value

    Returns:
        MitigationConfig with appropriate enabled flags
    """
    enabled = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS[Strategy.NONE])

    return MitigationConfig(
        onboard_ai_retention=ONBOARD_AI_RETENTION,
        predictive_retention=PREDICTIVE_RETENTION,
        relay_retention=RELAY_RETENTION,
        enabled=enabled
    )


# =============================================================================
# CORE FUNCTION 2: apply_strategy
# =============================================================================

def apply_strategy(
    strategy: Strategy,
    base_tau: float = DEFAULT_BASE_TAU,
    base_alpha: float = DEFAULT_BASE_ALPHA,
    receipt_integrity: float = 1.0,
    tenant_id: str = "axiom-autonomy"
) -> StrategyResult:
    """
    Apply a mitigation strategy.

    For COMBINED strategy, uses full mitigation stack.

    Args:
        strategy: Strategy to apply
        base_tau: Base tau in seconds
        base_alpha: Baseline sovereignty alpha
        receipt_integrity: Receipt-based integrity factor
        tenant_id: Tenant identifier

    Returns:
        StrategyResult with mitigation details
    """
    config = get_strategy_config(strategy)

    # Delegate to stack_mitigation for multiplicative computation
    mitigation = stack_mitigation(
        base_tau=base_tau,
        config=config,
        base_alpha=base_alpha,
        receipt_integrity=receipt_integrity,
        tenant_id=tenant_id
    )

    # Emit strategy receipt
    emit_strategy_applied_receipt(
        tenant_id=tenant_id,
        strategy=strategy.value,
        eff_tau=mitigation.eff_tau,
        eff_alpha=mitigation.eff_alpha,
        mitigations_enabled=mitigation.mitigations_applied
    )

    return StrategyResult(
        strategy=strategy,
        mitigation=mitigation,
        description=STRATEGY_DESCRIPTIONS.get(strategy, "Unknown strategy")
    )


# =============================================================================
# CORE FUNCTION 3: compare_strategies
# =============================================================================

def compare_strategies(
    base_tau: float = DEFAULT_BASE_TAU,
    tenant_id: str = "axiom-autonomy"
) -> dict:
    """
    Compare all strategies and return results.

    Args:
        base_tau: Base tau in seconds
        tenant_id: Tenant identifier

    Returns:
        dict mapping strategy name to StrategyResult
    """
    results = {}

    for strategy in Strategy:
        result = apply_strategy(
            strategy=strategy,
            base_tau=base_tau,
            tenant_id=tenant_id
        )
        results[strategy.value] = result

    return results


# =============================================================================
# CORE FUNCTION 4: best_strategy
# =============================================================================

def best_strategy(
    base_tau: float = DEFAULT_BASE_TAU,
    tenant_id: str = "axiom-autonomy"
) -> StrategyResult:
    """
    Get the best strategy (lowest effective tau).

    Currently always returns COMBINED as it has the best retention.

    Args:
        base_tau: Base tau in seconds
        tenant_id: Tenant identifier

    Returns:
        StrategyResult for best strategy
    """
    return apply_strategy(
        strategy=Strategy.COMBINED,
        base_tau=base_tau,
        tenant_id=tenant_id
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "STRATEGY_CONFIGS",
    "STRATEGY_DESCRIPTIONS",
    # Enums
    "Strategy",
    # Data classes
    "StrategyResult",
    # Core functions
    "get_strategy_config",
    "apply_strategy",
    "compare_strategies",
    "best_strategy",
    # Receipt functions
    "emit_strategy_applied_receipt",
]
