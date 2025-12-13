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
