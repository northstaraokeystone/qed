"""
sim - QED Simulation Package

Public API for v12 simulation dynamics.
CLAUDEME v3.1 Compliant: Flat, focused files. One file = one responsibility.
"""

# =============================================================================
# TYPES (Dataclasses)
# =============================================================================
from .types_config import (
    SimConfig,
    SCENARIO_BASELINE,
    SCENARIO_STRESS,
    SCENARIO_GENESIS,
    SCENARIO_SINGULARITY,
    SCENARIO_THERMODYNAMIC,
    SCENARIO_GODEL,
    # Variance inheritance scenarios (per Grok recommendation)
    SCENARIO_VARIANCE_INHERIT,
    SCENARIO_VARIANCE_RESET,
    SCENARIO_VARIANCE_MULTIVERSE,
    # Cross-domain recombination scenario (Grok 500-cycle validation)
    SCENARIO_CROSS_DOMAIN,
    # Stochastic affinity scenario (8th mandatory - Grok validated)
    SCENARIO_STOCHASTIC_AFFINITY,
    MANDATORY_SCENARIOS,
)
from .types_state import SimState, Seed, Beacon, Counselor, Crystal, FitnessDistribution
from .types_result import SimResult

# =============================================================================
# CONSTANTS
# =============================================================================
from .constants import (
    PatternState,
    RECEIPT_SCHEMA,
    TOLERANCE_FLOOR,
    TOLERANCE_CEILING,
    PLANCK_ENTROPY,
    PLANCK_ENTROPY_BASE,
    # Cross-domain constants (Grok validated)
    ADAPTIVE_PEAK_GENERATION,
    ADAPTIVE_PEAK_GENERATION_CROSS,
    AFFINITY_OPTIMIZATION_THRESHOLD,
    DOMAIN_AFFINITY_MATRIX,
)

# =============================================================================
# CORE SIMULATION
# =============================================================================
from .cycle import (
    run_simulation,
    run_multiverse,
    initialize_state,
    simulate_cycle,
)

# =============================================================================
# DYNAMICS
# =============================================================================
from .dynamics_lifecycle import (
    simulate_autocatalysis,
    simulate_selection,
)
from .dynamics_genesis import (
    simulate_wound,
    simulate_recombination,
    simulate_genesis,
    simulate_completeness,
    get_domain_affinity,
)
from .dynamics_quantum import (
    simulate_superposition,
    simulate_measurement,
    wavefunction_collapse,
    simulate_godel_stress,
    hilbert_space_size,
    bound_violation_check,
)

# =============================================================================
# VALIDATION
# =============================================================================
from .validation import (
    validate_conservation,
    detect_hidden_risk,
    compute_tolerance,
)

# =============================================================================
# MEASUREMENT
# =============================================================================
from .measurement import (
    measure_state,
    measure_observation_cost,
    measure_boundary_crossing,
    measure_genesis,
)

# =============================================================================
# VACUUM
# =============================================================================
from .vacuum_fluctuation import (
    vacuum_fluctuation,
    attempt_spontaneous_emergence,
    process_virtual_patterns,
)
from .vacuum_flux import (
    compute_hawking_flux,
    compute_collapse_rate,
    compute_emergence_rate,
    compute_system_criticality,
    emit_hawking_entropy,
)

# =============================================================================
# PERTURBATION
# =============================================================================
from .perturbation_core import (
    check_perturbation,
    check_basin_escape,
    check_resonance_peak,
)
from .perturbation_tracking import (
    check_structure_formation,
    track_baseline_shift,
    check_evolution_window,
    check_cluster_persistence,
    check_proto_form,
    check_symmetry_break,
)

# =============================================================================
# NUCLEATION
# =============================================================================
from .nucleation_seeds import (
    initialize_nucleation,
    counselor_compete,
    counselor_capture,
)
from .nucleation_crystals import (
    check_crystallization,
)
from .nucleation_evolution import (
    check_replication,
    check_hybrid_differentiation,
    evolve_seeds,
)

# =============================================================================
# VARIANCE INHERITANCE
# =============================================================================
from .variance import (
    variance_entropy,
    pooled_variance,
    inherit_variance,
    distribution_from_pattern,
    apply_distribution_to_pattern,
)

# =============================================================================
# EXPORT
# =============================================================================
from .export import (
    export_to_grok,
    generate_report,
    plot_population_dynamics,
    plot_entropy_trace,
    plot_completeness_progression,
    plot_genealogy,
)

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Types
    "SimConfig",
    "SimState",
    "SimResult",
    "Seed",
    "Beacon",
    "Counselor",
    "Crystal",
    "FitnessDistribution",
    # Scenario presets
    "SCENARIO_BASELINE",
    "SCENARIO_STRESS",
    "SCENARIO_GENESIS",
    "SCENARIO_SINGULARITY",
    "SCENARIO_THERMODYNAMIC",
    "SCENARIO_GODEL",
    # Variance inheritance scenarios
    "SCENARIO_VARIANCE_INHERIT",
    "SCENARIO_VARIANCE_RESET",
    "SCENARIO_VARIANCE_MULTIVERSE",
    # Cross-domain scenario
    "SCENARIO_CROSS_DOMAIN",
    # Stochastic affinity scenario
    "SCENARIO_STOCHASTIC_AFFINITY",
    "MANDATORY_SCENARIOS",
    # Constants
    "PatternState",
    "RECEIPT_SCHEMA",
    "TOLERANCE_FLOOR",
    "TOLERANCE_CEILING",
    "PLANCK_ENTROPY",
    "PLANCK_ENTROPY_BASE",
    # Cross-domain constants
    "ADAPTIVE_PEAK_GENERATION",
    "ADAPTIVE_PEAK_GENERATION_CROSS",
    "AFFINITY_OPTIMIZATION_THRESHOLD",
    "DOMAIN_AFFINITY_MATRIX",
    # Core simulation
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
    "get_domain_affinity",
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
    "track_baseline_shift",
    "check_evolution_window",
    "check_cluster_persistence",
    "check_proto_form",
    "check_symmetry_break",
    # Nucleation
    "initialize_nucleation",
    "counselor_compete",
    "counselor_capture",
    "check_crystallization",
    "check_replication",
    "check_hybrid_differentiation",
    "evolve_seeds",
    # Variance inheritance
    "variance_entropy",
    "pooled_variance",
    "inherit_variance",
    "distribution_from_pattern",
    "apply_distribution_to_pattern",
    # Export
    "export_to_grok",
    "generate_report",
    "plot_population_dynamics",
    "plot_entropy_trace",
    "plot_completeness_progression",
    "plot_genealogy",
]
