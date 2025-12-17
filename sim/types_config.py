"""
sim/types_config.py - SimConfig Dataclass and Scenario Presets

Immutable configuration for simulation runs.
CLAUDEME v3.1 Compliant: Frozen dataclass, no behavior.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Simulation configuration (immutable)."""
    n_cycles: int = 1000
    n_initial_patterns: int = 5
    wound_rate: float = 0.1
    mutation_rate: float = 0.01
    resource_budget: float = 1.0
    random_seed: int = 42
    hitl_auto_approve_rate: float = 0.8
    scenario_name: str = "BASELINE"
    # Fitness variance inheritance strategy (per Grok recommendation)
    # "INHERIT" = preserve adaptive history for autocatalysis depth
    # "RESET" = promote exploration in high-entropy scenarios
    variance_inheritance: str = "INHERIT"
    inherit_variance_decay: float = 0.95  # Prevents runaway variance amplification
    reset_variance_prior: float = 0.1  # Uninformed prior variance for RESET mode
    transition_period: int = 100  # Cycles between mode switches in ADAPTIVE (MULTIVERSE)


# =============================================================================
# SCENARIO PRESETS (6 mandatory)
# =============================================================================

SCENARIO_BASELINE = SimConfig(
    n_cycles=1000,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=42,
    scenario_name="BASELINE"
)

SCENARIO_STRESS = SimConfig(
    n_cycles=1000,
    n_initial_patterns=5,
    wound_rate=0.5,
    resource_budget=0.3,
    random_seed=43,
    scenario_name="STRESS"
)

SCENARIO_GENESIS = SimConfig(
    n_cycles=500,
    n_initial_patterns=5,
    wound_rate=0.3,
    resource_budget=1.0,
    random_seed=44,
    scenario_name="GENESIS"
)

SCENARIO_SINGULARITY = SimConfig(
    n_cycles=10000,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=45,
    scenario_name="SINGULARITY"
)

SCENARIO_THERMODYNAMIC = SimConfig(
    n_cycles=1000,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=46,
    scenario_name="THERMODYNAMIC"
)

SCENARIO_GODEL = SimConfig(
    n_cycles=500,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=47,
    scenario_name="GODEL"
)

# =============================================================================
# VARIANCE INHERITANCE SCENARIOS (Per Grok recommendation: test both)
# =============================================================================

SCENARIO_VARIANCE_INHERIT = SimConfig(
    n_cycles=1000,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=48,
    scenario_name="VARIANCE_INHERIT",
    variance_inheritance="INHERIT",
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

SCENARIO_VARIANCE_RESET = SimConfig(
    n_cycles=1000,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=49,
    scenario_name="VARIANCE_RESET",
    variance_inheritance="RESET",
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

SCENARIO_VARIANCE_MULTIVERSE = SimConfig(
    n_cycles=1000,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=50,
    scenario_name="VARIANCE_MULTIVERSE",
    variance_inheritance="INHERIT",  # Will switch every 100 cycles in simulation
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

# =============================================================================
# CROSS-DOMAIN RECOMBINATION SCENARIO (Grok 500-cycle validation)
# =============================================================================

SCENARIO_CROSS_DOMAIN = SimConfig(
    n_cycles=500,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=51,
    scenario_name="CROSS_DOMAIN",
    variance_inheritance="INHERIT",
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

# =============================================================================
# STOCHASTIC AFFINITY SCENARIO (8th mandatory - Grok validated)
# Validates dynamic thresholding under variance mode
# =============================================================================

SCENARIO_STOCHASTIC_AFFINITY = SimConfig(
    n_cycles=500,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=52,
    scenario_name="STOCHASTIC_AFFINITY",
    variance_inheritance="INHERIT",
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

# =============================================================================
# STACKED MITIGATION SCENARIOS (Per Grok: deploy combined tau strategies)
# =============================================================================

SCENARIO_STACKED_MITIGATION = SimConfig(
    n_cycles=100,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=53,
    scenario_name="STACKED_MITIGATION",
    variance_inheritance="INHERIT",
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

SCENARIO_P_SENSITIVITY = SimConfig(
    n_cycles=100,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=54,
    scenario_name="P_SENSITIVITY",
    variance_inheritance="INHERIT",
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

SCENARIO_ROI_GATE = SimConfig(
    n_cycles=100,
    n_initial_patterns=5,
    wound_rate=0.1,
    resource_budget=1.0,
    random_seed=55,
    scenario_name="ROI_GATE",
    variance_inheritance="INHERIT",
    inherit_variance_decay=0.95,
    reset_variance_prior=0.1
)

# =============================================================================
# MANDATORY SCENARIOS LIST (11 total - including mitigation scenarios)
# =============================================================================

MANDATORY_SCENARIOS = [
    "BASELINE",
    "STRESS",
    "GENESIS",
    "SINGULARITY",
    "THERMODYNAMIC",
    "GODEL",
    "CROSS_DOMAIN",
    "STOCHASTIC_AFFINITY",
    "STACKED_MITIGATION",
    "P_SENSITIVITY",
    "ROI_GATE",
]
