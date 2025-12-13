"""
sim - QED Simulation Package

Public API for v12 simulation dynamics.
CLAUDEME v3.1 Compliant: Flat, focused files. One file = one responsibility.
"""

# Dataclasses
from .config import (
    SimConfig,
    SCENARIO_BASELINE,
    SCENARIO_STRESS,
    SCENARIO_GENESIS,
    SCENARIO_SINGULARITY,
    SCENARIO_THERMODYNAMIC,
    SCENARIO_GODEL,
)
from .state import SimState, Seed, Beacon, Counselor, Crystal
from .result import SimResult

# Constants
from .constants import (
    PatternState,
    RECEIPT_SCHEMA,
    TOLERANCE_FLOOR,
    TOLERANCE_CEILING,
    PLANCK_ENTROPY,
    PLANCK_ENTROPY_BASE,
)

# Core functions
from .cycle import (
    run_simulation,
    run_multiverse,
    initialize_state,
    simulate_cycle,
)

# Dynamics
from .dynamics import (
    simulate_wound,
    simulate_autocatalysis,
    simulate_selection,
    simulate_recombination,
    simulate_genesis,
    simulate_completeness,
    simulate_superposition,
    simulate_measurement,
    wavefunction_collapse,
    simulate_godel_stress,
    hilbert_space_size,
    bound_violation_check,
)

# Validation
from .validation import (
    validate_conservation,
    detect_hidden_risk,
    compute_tolerance,
)

# Measurement
from .measurement import (
    measure_state,
    measure_observation_cost,
    measure_boundary_crossing,
    measure_genesis,
)

# Vacuum and emergence
from .vacuum import (
    vacuum_fluctuation,
    attempt_spontaneous_emergence,
    process_virtual_patterns,
    compute_hawking_flux,
    compute_collapse_rate,
    compute_emergence_rate,
    compute_system_criticality,
    emit_hawking_entropy,
)

# Perturbation
from .perturbation import (
    check_perturbation,
    check_basin_escape,
    check_resonance_peak,
    check_structure_formation,
)

# Nucleation
from .nucleation import (
    initialize_nucleation,
    counselor_compete,
    counselor_capture,
    check_crystallization,
    check_replication,
    check_hybrid_differentiation,
    evolve_seeds,
)

# Export
from .export import (
    export_to_grok,
    generate_report,
    plot_population_dynamics,
    plot_entropy_trace,
    plot_completeness_progression,
    plot_genealogy,
)

__all__ = [
    # Dataclasses
    "SimConfig",
    "SimState",
    "SimResult",
    "Seed",
    "Beacon",
    "Counselor",
    "Crystal",
    # Scenario presets
    "SCENARIO_BASELINE",
    "SCENARIO_STRESS",
    "SCENARIO_GENESIS",
    "SCENARIO_SINGULARITY",
    "SCENARIO_THERMODYNAMIC",
    "SCENARIO_GODEL",
    # Constants
    "PatternState",
    "RECEIPT_SCHEMA",
    "TOLERANCE_FLOOR",
    "TOLERANCE_CEILING",
    "PLANCK_ENTROPY",
    "PLANCK_ENTROPY_BASE",
    # Core functions
    "run_simulation",
    "run_multiverse",
    "initialize_state",
    "simulate_cycle",
    # Dynamics
    "simulate_wound",
    "simulate_autocatalysis",
    "simulate_selection",
    "simulate_recombination",
    "simulate_genesis",
    "simulate_completeness",
    "simulate_superposition",
    "simulate_measurement",
    "wavefunction_collapse",
    "simulate_godel_stress",
    "hilbert_space_size",
    "bound_violation_check",
    # Validation
    "validate_conservation",
    "detect_hidden_risk",
    "compute_tolerance",
    # Measurement
    "measure_state",
    "measure_observation_cost",
    "measure_boundary_crossing",
    "measure_genesis",
    # Vacuum
    "vacuum_fluctuation",
    "attempt_spontaneous_emergence",
    "process_virtual_patterns",
    "compute_hawking_flux",
    "compute_collapse_rate",
    "compute_emergence_rate",
    "compute_system_criticality",
    "emit_hawking_entropy",
    # Perturbation
    "check_perturbation",
    "check_basin_escape",
    "check_resonance_peak",
    "check_structure_formation",
    # Nucleation
    "initialize_nucleation",
    "counselor_compete",
    "counselor_capture",
    "check_crystallization",
    "check_replication",
    "check_hybrid_differentiation",
    "evolve_seeds",
    # Export
    "export_to_grok",
    "generate_report",
    "plot_population_dynamics",
    "plot_entropy_trace",
    "plot_completeness_progression",
    "plot_genealogy",
]
