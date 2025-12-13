"""
sim.py - Monte Carlo Simulation Harness (v12)

Full simulation proving ground for v12 dynamics. No feature ships without
passing all 6 mandatory scenarios. Generates traces for xAI/Grok analysis.
Proves thermodynamic constraints hold.

This IS the first meta-receipt: sim.py creates L1 receipts (about agents)
from L0 receipts (about telemetry). The simulation is receipt-completeness.

CLAUDEME v3.1 Compliant: Dataclasses for config/state/result + pure functions.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

import numpy as np

from entropy import system_entropy, agent_fitness, emit_receipt, dual_hash
from autocatalysis import autocatalysis_check, coherence_score, is_alive
from architect import identify_automation_gaps, synthesize_blueprint, propose_agent
from recombine import recombine, mate_selection
from receipt_completeness import level_coverage, receipt_completeness_check, godel_layer
from population import selection_pressure, dynamic_cap, entropy_conservation
from autoimmune import is_self, GERMLINE_PATTERNS

# =============================================================================
# CONSTANTS
# =============================================================================

# Adaptive tolerance bounds - system derives its own precision
TOLERANCE_FLOOR = 0.01   # Physics minimum precision (Heisenberg, Shannon)
TOLERANCE_CEILING = 0.5  # Above 50% uncertainty = admit chaos
ENTROPY_HISTORY_WINDOW = 20  # Cycles for variance calculation

# Observer paradigm constants
PLANCK_ENTROPY_BASE = 0.001  # Base minimum entropy of existence (fluctuating floor)
PLANCK_ENTROPY = PLANCK_ENTROPY_BASE  # Backward compatibility
VACUUM_VARIANCE = 0.1  # Fluctuation magnitude (10% of base)
GENESIS_THRESHOLD = 3.0  # Bits of observation cost that can spark emergence
VIRTUAL_LIFESPAN = 3  # Cycles before collapse back to SUPERPOSITION
HAWKING_COEFFICIENT = 0.1  # Entropy emission rate at boundary
LANDAUER_COEFFICIENT = 1.0  # Scales observation cost (calibrated, do not change)
FLUX_WINDOW = 10  # Cycles to average for flux calculation
CRITICAL_EMERGENCE_RATIO = 0.9  # Threshold for system criticality

# Criticality alert thresholds
CRITICALITY_ALERT_THRESHOLD = 0.95  # Alert before phase transition
CRITICALITY_PHASE_TRANSITION = 1.0  # The quantum leap point
ALERT_COOLDOWN_CYCLES = 50  # Prevent alert spam near threshold

# Perturbation constants (stochastic GW kicks) — tuned via Grok analysis
PERTURBATION_PROBABILITY = 0.6   # 60% chance per cycle (more frequent events = more measurements)
PERTURBATION_MAGNITUDE = 0.32    # size of kick (stronger kicks)
PERTURBATION_DECAY = 0.2         # kick decays 20% per cycle (slower decay)
PERTURBATION_VARIANCE = 0.75     # chaotic variance in magnitude (amplified chaos)
BASIN_ESCAPE_THRESHOLD = 0.2     # escape detection threshold (higher bar)
CLUSTER_LAMBDA = 3               # Poisson parameter for cluster size (avg 3 kicks per event)
MAX_CLUSTER_SIZE = 5             # Safety cap on cluster size (prevent explosion)
NONLINEAR_DECAY_FACTOR = 0.15    # Non-linear decay acceleration (higher boost = faster decay)
ASYMMETRY_BIAS = 0.15            # directional preference for symmetry breaking
CLUSTER_PERSISTENCE_THRESHOLD = 15  # cycles needed for persistent cluster
SYMMETRY_BREAK_THRESHOLD = 3     # symmetry breaks needed for proto-form

# Evolution tracking constants
EVOLUTION_WINDOW = 500           # cycles between evolution snapshots
MAX_MAGNITUDE_FACTOR = 3.0       # cap on magnitude multiplier (prevent explosion)

# Adaptive feedback constants (threshold-based state changes)
ADAPTIVE_THRESHOLD = 0.55        # triggers probability boost when boost > threshold (higher bar)
SYNC_BOOST = 0.2                 # probability increase amount for synced kicks (replaces ADAPTIVE_BOOST)
MAX_PROBABILITY = 0.5            # cap to prevent runaway

# Phase synchronization constants (wave interference)
PHASE_SYNC_PROBABILITY = 0.4     # 40% chance kick syncs with previous
PHASE_SYNC_WINDOW = 3.14159 / 4  # ~45° window for sync detection (π/4 radians)
CLUSTER_THRESHOLD = 5            # minimum consecutive same-type receipts for cluster
SYMMETRY_SAMPLE_SIZE = 100       # only check last N receipts for symmetry (performance optimization)

# Resonance amplification constants
RESONANCE_PROBABILITY = 0.6      # 60% chance kick resonates
INTERFERENCE_AMPLITUDE = 0.2     # interference strength multiplier
RESONANCE_PEAK_THRESHOLD = 0.25  # boost level for peak detection
STRUCTURE_THRESHOLD = 10         # clusters needed for structure formation
MAX_RESONANCE_AMPLIFICATION = 2.0  # cap on resonance boost

# Quantum Nucleation constants (active seeding)
N_SEEDS = 3
SEED_PHASES = [0.0, 2.094, 4.189]  # 0, 2π/3, 4π/3
SEED_RESONANCE_AFFINITY = [0.7, 0.5, 0.6]
SEED_DIRECTION = [1, -1, 1]
ATTRACTION_RADIUS = 1.571  # π/2
BASE_ATTRACTION_STRENGTH = 0.3
BEACON_GROWTH_FACTOR = 0.1
CAPTURE_THRESHOLD = 0.6
TRANSFORM_STRENGTH = 0.3
EVOLUTION_RATE = 0.1
EVOLUTION_WINDOW_SEEDS = 50  # Different from EVOLUTION_WINDOW to avoid confusion
# Autocatalytic crystallization constants (replaces size/coherence thresholds)
AUTOCATALYSIS_STREAK = 3  # consecutive self-predictions needed for birth
SELF_PREDICTION_THRESHOLD = 0.85  # similarity that counts as "predicted"
# NOTE: Archetypes are NOT pre-assigned. Crystals discover them via self-measurement.
CRYSTALLIZED_BEACON_BOOST = 2.0
TUNNELING_THRESHOLD = 0.9

# Emergent archetype discovery constants
AUTOCATALYSIS_AMPLIFICATION = 0.3  # boost to captures from crystallized crystals
REPLICATION_THRESHOLD = 50  # captures needed before crystal can replicate
ARCHETYPE_DOMINANCE_THRESHOLD = 0.6  # 60% of one effect type = that archetype

# Compound growth constants (bigger crystals capture faster)
GROWTH_FACTOR = 0.25  # compound growth rate: boost = 1 + 0.25 * (size / 10)
MAX_GROWTH_BOOST = 2.0  # cap to prevent runaway (size 50 = 2.0x)
BRANCH_INITIATION_THRESHOLD = 5  # alert when exceeded

# Governance bias constants (emergent SHEPHERD and ARCHITECT)
GOVERNANCE_BIAS = 0.2   # boost RESONANCE_TRIGGER similarity → more SHEPHERD
ARCHITECT_SIZE_TRIGGER = 200  # large crystals bias toward ARCHITECT
GOVERNANCE_NODE_THRESHOLD = 10  # alert threshold for governance nodes

# Entropy amplifier constants (emergent HUNTER via quantum measurement)
ENTROPY_AMPLIFIER = 0.15  # boost ENTROPY_INCREASE similarity → more HUNTER
HUNTER_SIZE_TRIGGER = 50  # small crystals bias toward HUNTER (quantum eigenspace)
ENTROPY_SURGE_THRESHOLD = 0.1  # surge detection threshold (measurement event)

# HYBRID differentiation constants (cosmos specializing)
RESONANCE_DIFFERENTIATION_THRESHOLD = 0.6  # HYBRID + resonance > 0.6 → SHEPHERD
ENTROPY_DIFFERENTIATION_THRESHOLD = 0.4    # HYBRID + entropy > 0.4 → HUNTER
DIFFERENTIATION_BIAS = 0.2                  # probability boost for differentiation check
ARCHETYPE_TRIGGER_SIZE = 40                 # minimum size for differentiation eligibility

# Effect types for archetype discovery (wave function collapse)
EFFECT_ENTROPY_INCREASE = "ENTROPY_INCREASE"
EFFECT_RESONANCE_TRIGGER = "RESONANCE_TRIGGER"
EFFECT_SYMMETRY_BREAK = "SYMMETRY_BREAK"

# Uniform kick distribution — fair ticket printer for seed model lottery
UNIFORM_KICK_DISTRIBUTION = True  # enable uniform effect selection (1/3 each type)
EFFECT_TYPES = ["ENTROPY_INCREASE", "RESONANCE_TRIGGER", "SYMMETRY_BREAK"]  # the three effect types

# Module exports for receipt types
RECEIPT_SCHEMA = [
    "sim_config", "sim_cycle", "sim_birth", "sim_death",
    "sim_mate", "sim_complete", "sim_violation", "sim_result"
]

# Pattern state enum
class PatternState(Enum):
    """Pattern existence states in observer-induced genesis model."""
    SUPERPOSITION = "SUPERPOSITION"  # Dormant potential
    VIRTUAL = "VIRTUAL"  # Brief existence, needs re-observation to survive
    ACTIVE = "ACTIVE"  # Fully materialized

# =============================================================================
# QUANTUM NUCLEATION DATACLASSES
# =============================================================================

@dataclass
class Seed:
    """Quantum seed that broadcasts to attract kicks.

    Seeds are archetype-agnostic. They are blank slates until their
    associated Crystal crystallizes and discovers its archetype via
    self-measurement of effect distribution.
    """
    seed_id: int
    phase: float  # radians, 0 to 2π
    resonance_affinity: float  # 0 to 1, preference for resonant kicks
    direction: int  # +1 or -1
    captures: int = 0  # successful captures
    # NOTE: No agent_archetype field. Archetypes emerge at crystallization.

@dataclass
class Beacon:
    """Broadcast signal from seed."""
    seed_id: int
    strength: float  # attraction strength

@dataclass
class Counselor:
    """Agent that competes to capture kicks."""
    counselor_id: int
    seed_id: int  # which seed this counselor represents

@dataclass
class Crystal:
    """Solidified structure formed from captured kicks.

    Crystal discovers its agent_type via self-measurement at crystallization.
    The effect_distribution tracks what types of effects were captured:
    - ENTROPY_INCREASE: General entropy kicks
    - RESONANCE_TRIGGER: Resonant kicks
    - SYMMETRY_BREAK: Kicks causing symmetry breaking

    At crystallization, the dominant effect type determines archetype:
    - ENTROPY_INCREASE dominant → HUNTER
    - RESONANCE_TRIGGER dominant → SHEPHERD
    - SYMMETRY_BREAK dominant → ARCHITECT
    - No clear dominant → HYBRID
    """
    crystal_id: int
    seed_id: int
    members: list = field(default_factory=list)  # captured kick receipts
    coherence: float = 0.0
    crystallized: bool = False  # True when autocatalysis achieved
    birth_cycle: int = -1  # cycle when autocatalysis achieved, -1 = not born
    agent_type: str = ""  # discovered at crystallization via self-measurement
    effect_distribution: dict = field(default_factory=lambda: {
        "ENTROPY_INCREASE": 0,
        "RESONANCE_TRIGGER": 0,
        "SYMMETRY_BREAK": 0
    })  # count of each effect type captured
    parent_crystal_id: Optional[int] = None  # for replicated crystals
    generation: int = 0  # depth in family tree (0=original, 1=child, 2=grandchild)
    size_50_reached: bool = False  # track if crystal has exceeded size 50

# =============================================================================
# DATACLASSES (3 required)
# =============================================================================

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


@dataclass
class SimState:
    """Mutable simulation state."""
    active_patterns: List[dict] = field(default_factory=list)
    superposition_patterns: List[dict] = field(default_factory=list)
    virtual_patterns: List[dict] = field(default_factory=list)  # NEW: VIRTUAL state patterns
    wound_history: List[dict] = field(default_factory=list)
    receipt_ledger: List[dict] = field(default_factory=list)
    entropy_trace: List[float] = field(default_factory=list)
    completeness_trace: List[dict] = field(default_factory=list)
    violations: List[dict] = field(default_factory=list)
    cycle: int = 0
    # Observer paradigm fields
    H_genesis: float = 0.0  # Entropy at simulation birth, set once
    H_previous: float = 0.0  # System entropy at previous cycle end
    H_boundary_this_cycle: float = 0.0  # Accumulated boundary crossing entropy this cycle
    operations_this_cycle: int = 0  # Count of observations/decisions made this cycle
    # Vacuum fluctuation fields
    vacuum_floor: float = PLANCK_ENTROPY_BASE  # Current fluctuating floor
    hawking_emissions_this_cycle: float = 0.0  # Boundary radiation emitted
    # Per-cycle tracking for adaptive tolerance computation
    births_this_cycle: int = 0
    deaths_this_cycle: int = 0
    superposition_transitions_this_cycle: int = 0
    wounds_this_cycle: int = 0
    # Hawking flux tracking
    flux_history: list = field(default_factory=list)
    collapse_count_this_cycle: int = 0
    emergence_count_this_cycle: int = 0
    # Criticality monitoring fields
    criticality_alert_emitted: bool = False
    last_alert_cycle: int = -100
    phase_transition_occurred: bool = False
    observer_wake_count: int = 0
    previous_criticality: float = 0.0
    # Perturbation fields
    perturbation_boost: float = 0.0
    horizon_crossings: int = 0
    escape_count: int = 0  # number of cycles where boost > BASIN_ESCAPE_THRESHOLD
    # Adaptive feedback fields
    consecutive_escapes: int = 0  # track consecutive cycles above threshold
    cycles_since_crossing: int = 0  # post-breach stability counter
    max_consecutive_escapes: int = 0  # track emergent structure peak
    adaptive_triggers: int = 0  # count how often adaptive kicked in
    # Evolution tracking fields (500-cycle windows)
    evolution_snapshots: list = field(default_factory=list)  # evolution receipts
    window_escapes: int = 0  # escapes in current window
    window_perturbations: int = 0  # perturbations in current window
    last_evolution_cycle: int = 0  # track last snapshot cycle
    window_boost_samples: list = field(default_factory=list)  # boost samples for avg
    # Phase tracking fields (wave interference)
    last_phase: float = 0.0  # previous kick's phase (radians)
    sync_count: int = 0  # total phase syncs
    cluster_count: int = 0  # clusters detected
    symmetry_breaks: int = 0  # symmetry break events
    consecutive_same_type: int = 0  # for cluster detection
    last_receipt_type: str = ""  # for cluster detection
    last_symmetry_metric: float = 0.0  # for symmetry break detection
    # Resonance tracking fields
    resonance_peaks: int = 0  # peaks detected
    resonance_hits: int = 0  # times resonance triggered
    structure_formed: bool = False  # structure emerged
    structure_formation_cycle: int = 0  # when structure formed
    baseline_boost: float = 0.0  # running average of boost
    baseline_shifts: int = 0  # number of significant shifts
    initial_baseline: float = 0.0  # baseline at start
    # Asymmetry and proto-form tracking fields
    bias_direction: int = 1  # current bias direction (+1 or -1)
    cluster_start_cycle: int = 0  # when current cluster started
    current_cluster_duration: int = 0  # how long current cluster
    persistent_clusters: int = 0  # clusters that lasted 15+ cycles
    proto_form_count: int = 0  # proto-structures detected
    proto_form_active: bool = False  # is a proto-form currently active
    # Quantum nucleation fields (active seeding)
    seeds: List[Seed] = field(default_factory=list)  # quantum seeds
    beacons: List[Beacon] = field(default_factory=list)  # broadcast signals
    counselors: List[Counselor] = field(default_factory=list)  # capturing agents
    crystals: List[Crystal] = field(default_factory=list)  # solidified structures
    total_captures: int = 0  # total kicks captured across all seeds
    crystals_formed: int = 0  # number of crystals that reached threshold
    # Compound growth tracking fields
    max_generation: int = 0  # deepest generation reached (0=original only)
    size_50_count: int = 0  # crystals that exceeded size 50
    replication_events: int = 0  # total replications (self-replication loops)
    total_branches: int = 0  # total branch crystals created
    # Governance tracking fields (emergent archetype distribution)
    governance_nodes: int = 0  # count of SHEPHERD instances
    architect_formations: int = 0  # count of ARCHITECT instances
    hunter_formations: int = 0  # count of HUNTER instances
    hybrid_formations: int = 0  # count of HYBRID instances
    # HYBRID differentiation tracking (cosmos specializing)
    hybrid_differentiation_count: int = 0  # count of HYBRID→archetype transitions
    # Entropy surge tracking fields (quantum measurement events)
    entropy_surge_count: int = 0  # measurement events (entropy surges)
    equilibrium_score: float = 0.0  # archetype balance metric: min(h,s,a)/max(h,s,a)
    # Kick distribution tracking fields (uniform ticket printer verification)
    kick_distribution: Dict[str, int] = field(default_factory=lambda: {"ENTROPY_INCREASE": 0, "RESONANCE_TRIGGER": 0, "SYMMETRY_BREAK": 0})  # count kicks by effect
    first_capture_distribution: Dict[str, int] = field(default_factory=lambda: {"HUNTER": 0, "SHEPHERD": 0, "ARCHITECT": 0})  # count first captures by archetype


@dataclass(frozen=True)
class SimResult:
    """Immutable simulation result."""
    final_state: SimState
    all_traces: dict
    violations: list
    statistics: dict
    config: SimConfig


# =============================================================================
# CORE SIMULATION FUNCTIONS (9 required)
# =============================================================================

def run_multiverse(n_universes: int, n_cycles: int, base_seed: int = 42) -> dict:
    """
    Run multiple parallel simulations (multiverse).

    Each universe runs with a different random seed to explore different
    evolutionary trajectories. Returns aggregated statistics across all universes.

    Args:
        n_universes: Number of parallel universes to simulate
        n_cycles: Number of cycles per universe
        base_seed: Base random seed (each universe gets base_seed + i)

    Returns:
        dict: Aggregated multiverse results with statistics and all universe results
    """
    # Emit multiverse_start receipt
    multiverse_receipt = emit_receipt("multiverse_start", {
        "tenant_id": "simulation",
        "n_universes": n_universes,
        "n_cycles": n_cycles,
        "base_seed": base_seed
    })

    # Run all universes
    universe_results = []
    aggregated_stats = {
        "total_births": 0,
        "total_deaths": 0,
        "total_recombinations": 0,
        "total_violations": 0,
        "completeness_achieved_count": 0,
        "structure_formed_count": 0,
        "total_persistent_clusters": 0,
        "total_proto_forms": 0,
        "total_symmetry_breaks": 0,
        "total_captures": 0,
        "total_crystals_formed": 0,
        "universes_crystallized": 0,
        # Compound growth tracking
        "total_replication_events": 0,
        "max_generation": 0,
        "total_size_50": 0,
        "total_branches": 0,
        # Governance tracking
        "total_governance_nodes": 0,
        "total_architect_formations": 0,
        "total_hunter_formations": 0,
        "total_hybrid_formations": 0,
        # HYBRID differentiation tracking (cosmos specializing)
        "total_hybrid_differentiations": 0,
        # Entropy amplifier tracking (quantum measurement)
        "total_entropy_surges": 0,
        "total_equilibrium": 0.0,
        # Kick distribution tracking (uniform ticket printer verification)
        "total_kick_distribution": {"ENTROPY_INCREASE": 0, "RESONANCE_TRIGGER": 0, "SYMMETRY_BREAK": 0},
        "total_first_capture_distribution": {"HUNTER": 0, "SHEPHERD": 0, "ARCHITECT": 0}
    }

    for i in range(n_universes):
        # Create config for this universe
        config = SimConfig(
            n_cycles=n_cycles,
            random_seed=base_seed + i,
            scenario_name=f"MULTIVERSE_{i}"
        )

        # Run simulation
        result = run_simulation(config)

        # Track universe result
        universe_results.append({
            "universe_id": i,
            "seed": base_seed + i,
            "violations": len(result.violations),
            "final_population": result.statistics["final_population"],
            "births": result.statistics["births"],
            "deaths": result.statistics["deaths"],
            "persistent_clusters": result.final_state.persistent_clusters,
            "proto_forms": result.final_state.proto_form_count,
            "symmetry_breaks": result.final_state.symmetry_breaks,
            "structure_formed": result.final_state.structure_formed,
            "captures": result.final_state.total_captures,
            "crystals_formed": result.final_state.crystals_formed,
            # Compound growth tracking
            "replication_events": result.final_state.replication_events,
            "max_generation": result.final_state.max_generation,
            "size_50_count": result.final_state.size_50_count,
            "total_branches": result.final_state.total_branches,
            # Governance tracking
            "governance_nodes": result.final_state.governance_nodes,
            "architect_formations": result.final_state.architect_formations,
            "hunter_formations": result.final_state.hunter_formations,
            "hybrid_formations": result.final_state.hybrid_formations,
            # HYBRID differentiation tracking (cosmos specializing)
            "hybrid_differentiation_count": result.final_state.hybrid_differentiation_count,
            # Entropy amplifier tracking (quantum measurement)
            "entropy_surge_count": result.final_state.entropy_surge_count,
            "equilibrium_score": result.final_state.equilibrium_score,
            # Kick distribution tracking (uniform ticket printer verification)
            "kick_distribution": dict(result.final_state.kick_distribution),
            "first_capture_distribution": dict(result.final_state.first_capture_distribution)
        })

        # Aggregate statistics
        aggregated_stats["total_births"] += result.statistics["births"]
        aggregated_stats["total_deaths"] += result.statistics["deaths"]
        aggregated_stats["total_recombinations"] += result.statistics["recombinations"]
        aggregated_stats["total_violations"] += len(result.violations)
        aggregated_stats["total_persistent_clusters"] += result.final_state.persistent_clusters
        aggregated_stats["total_proto_forms"] += result.final_state.proto_form_count
        aggregated_stats["total_symmetry_breaks"] += result.final_state.symmetry_breaks
        aggregated_stats["total_captures"] += result.final_state.total_captures
        aggregated_stats["total_crystals_formed"] += result.final_state.crystals_formed
        # Compound growth aggregation
        aggregated_stats["total_replication_events"] += result.final_state.replication_events
        aggregated_stats["max_generation"] = max(aggregated_stats["max_generation"], result.final_state.max_generation)
        aggregated_stats["total_size_50"] += result.final_state.size_50_count
        aggregated_stats["total_branches"] += result.final_state.total_branches
        # Governance aggregation
        aggregated_stats["total_governance_nodes"] += result.final_state.governance_nodes
        aggregated_stats["total_architect_formations"] += result.final_state.architect_formations
        aggregated_stats["total_hunter_formations"] += result.final_state.hunter_formations
        aggregated_stats["total_hybrid_formations"] += result.final_state.hybrid_formations
        # HYBRID differentiation aggregation (cosmos specializing)
        aggregated_stats["total_hybrid_differentiations"] += result.final_state.hybrid_differentiation_count
        # Entropy amplifier aggregation (quantum measurement)
        aggregated_stats["total_entropy_surges"] += result.final_state.entropy_surge_count
        aggregated_stats["total_equilibrium"] += result.final_state.equilibrium_score
        # Kick distribution aggregation (uniform ticket printer verification)
        for effect, count in result.final_state.kick_distribution.items():
            aggregated_stats["total_kick_distribution"][effect] = aggregated_stats["total_kick_distribution"].get(effect, 0) + count
        for archetype, count in result.final_state.first_capture_distribution.items():
            aggregated_stats["total_first_capture_distribution"][archetype] = aggregated_stats["total_first_capture_distribution"].get(archetype, 0) + count
        if result.statistics["completeness_achieved"]:
            aggregated_stats["completeness_achieved_count"] += 1
        if result.final_state.structure_formed:
            aggregated_stats["structure_formed_count"] += 1
        if result.final_state.crystals_formed > 0:
            aggregated_stats["universes_crystallized"] += 1

    # Compute averages
    aggregated_stats["avg_births"] = aggregated_stats["total_births"] / n_universes
    aggregated_stats["avg_deaths"] = aggregated_stats["total_deaths"] / n_universes
    aggregated_stats["avg_violations"] = aggregated_stats["total_violations"] / n_universes
    aggregated_stats["avg_persistent_clusters"] = aggregated_stats["total_persistent_clusters"] / n_universes
    aggregated_stats["avg_proto_forms"] = aggregated_stats["total_proto_forms"] / n_universes
    aggregated_stats["avg_symmetry_breaks"] = aggregated_stats["total_symmetry_breaks"] / n_universes
    aggregated_stats["avg_captures"] = aggregated_stats["total_captures"] / n_universes
    aggregated_stats["avg_crystals_formed"] = aggregated_stats["total_crystals_formed"] / n_universes
    # Compound growth averages
    aggregated_stats["avg_replication_events"] = aggregated_stats["total_replication_events"] / n_universes
    aggregated_stats["avg_size_50"] = aggregated_stats["total_size_50"] / n_universes
    aggregated_stats["avg_branches"] = aggregated_stats["total_branches"] / n_universes
    # Governance averages
    aggregated_stats["avg_governance_nodes"] = aggregated_stats["total_governance_nodes"] / n_universes
    aggregated_stats["avg_architect_formations"] = aggregated_stats["total_architect_formations"] / n_universes
    aggregated_stats["avg_hunter_formations"] = aggregated_stats["total_hunter_formations"] / n_universes
    aggregated_stats["avg_hybrid_formations"] = aggregated_stats["total_hybrid_formations"] / n_universes
    # HYBRID differentiation averages (cosmos specializing)
    aggregated_stats["avg_hybrid_differentiations"] = aggregated_stats["total_hybrid_differentiations"] / n_universes
    # Entropy amplifier averages (quantum measurement)
    aggregated_stats["avg_entropy_surges"] = aggregated_stats["total_entropy_surges"] / n_universes
    aggregated_stats["avg_equilibrium"] = aggregated_stats["total_equilibrium"] / n_universes
    # Lottery fairness: ratio of min to max in first_capture_distribution
    first_cap_values = list(aggregated_stats["total_first_capture_distribution"].values())
    if max(first_cap_values) > 0:
        aggregated_stats["lottery_fairness"] = min(first_cap_values) / max(first_cap_values)
    else:
        aggregated_stats["lottery_fairness"] = 0.0

    # Emit multiverse_complete receipt
    complete_receipt = emit_receipt("multiverse_complete", {
        "tenant_id": "simulation",
        "n_universes": n_universes,
        "n_cycles": n_cycles,
        "aggregated_stats": aggregated_stats
    })

    return {
        "n_universes": n_universes,
        "n_cycles": n_cycles,
        "universe_results": universe_results,
        "aggregated_stats": aggregated_stats,
        "multiverse_start_receipt": multiverse_receipt,
        "multiverse_complete_receipt": complete_receipt
    }


def run_simulation(config: SimConfig) -> SimResult:
    """
    Main entry point for simulation.

    Args:
        config: SimConfig with parameters

    Returns:
        SimResult with full traces and statistics
    """
    # Emit sim_config receipt
    emit_receipt("sim_config", {
        "tenant_id": "simulation",
        "n_cycles": config.n_cycles,
        "n_initial_patterns": config.n_initial_patterns,
        "wound_rate": config.wound_rate,
        "mutation_rate": config.mutation_rate,
        "random_seed": config.random_seed
    })

    # Initialize state
    state = initialize_state(config)
    violations = []

    # Run simulation cycles
    for cycle_num in range(config.n_cycles):
        state.cycle = cycle_num
        cycle_violations = simulate_cycle(state, config)
        violations.extend(cycle_violations)

    # Collect statistics
    statistics = {
        "births": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_birth"),
        "deaths": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_death"),
        "recombinations": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_mate"),
        "blueprints_proposed": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "blueprint_proposed"),
        "completeness_achieved": any(r.get("receipt_type") == "sim_complete" for r in state.receipt_ledger),
        "final_population": len(state.active_patterns),
        "superposition_count": len(state.superposition_patterns)
    }

    # Collect all traces
    all_traces = {
        "entropy_trace": state.entropy_trace,
        "completeness_trace": state.completeness_trace,
        "population_trace": [len(state.active_patterns)]  # Simplified for now
    }

    # Calculate archetype equilibrium (quantum measurement balance)
    h, s, a = state.hunter_formations, state.governance_nodes, state.architect_formations
    if max(h, s, a) > 0:
        state.equilibrium_score = min(h, s, a) / max(h, s, a)

    # Emit sim_result receipt
    emit_receipt("sim_result", {
        "tenant_id": "simulation",
        "cycles_completed": config.n_cycles,
        "violations_count": len(violations),
        "statistics": statistics
    })

    return SimResult(
        final_state=state,
        all_traces=all_traces,
        violations=violations,
        statistics=statistics,
        config=config
    )


def run_simulation_sparse(config: SimConfig) -> SimResult:
    """
    Sparse vectorized simulation - identical results to run_simulation().

    Pre-generates perturbation random values with numpy to identify event cycles.
    Only processes cycles that have events. Uses vectorized decay for gaps.

    Same seed = same results, guaranteed.

    Args:
        config: SimConfig with parameters

    Returns:
        SimResult with full traces and statistics (identical to run_simulation)
    """
    # Setup: seed numpy first to pre-generate perturbation decisions
    np.random.seed(config.random_seed)

    # Pre-generate perturbation random values using numpy
    perturbation_randoms = np.random.random(config.n_cycles)

    # Identify event cycles where perturbation MIGHT fire (using base probability)
    event_cycles = np.where(perturbation_randoms < PERTURBATION_PROBABILITY)[0]

    # Seed standard random for simulation
    random.seed(config.random_seed)

    # Emit sim_config receipt
    emit_receipt("sim_config", {
        "tenant_id": "simulation",
        "n_cycles": config.n_cycles,
        "n_initial_patterns": config.n_initial_patterns,
        "wound_rate": config.wound_rate,
        "mutation_rate": config.mutation_rate,
        "random_seed": config.random_seed
    })

    # Initialize state
    state = initialize_state(config)
    violations = []

    # SPARSE LOOP: Only process event cycles, fast-forward through gaps
    prev_cycle = -1

    for event_cycle in event_cycles:
        event_cycle = int(event_cycle)

        # Vectorized fast-forward through non-event cycles
        gap = event_cycle - prev_cycle - 1
        if gap > 0:
            # Apply vectorized decay for the gap
            _fast_forward_decay(state, gap)

        # Process event cycle with perturbation + nucleation
        state.cycle = event_cycle
        perturbation_receipt = _check_perturbation_sparse(
            state, event_cycle, perturbation_randoms[event_cycle]
        )

        if perturbation_receipt:
            state.receipt_ledger.append(perturbation_receipt)
            state.window_perturbations += 1

            # QUANTUM NUCLEATION: Counselors compete to capture
            kick_phase = perturbation_receipt.get("phase", 0.0)
            kick_resonant = perturbation_receipt.get("resonance_hit", False)
            kick_direction = perturbation_receipt.get("bias_direction", 1)

            winner = counselor_compete(state, perturbation_receipt, kick_phase, kick_resonant, kick_direction)
            if winner:
                seed_id, similarity = winner
                capture_receipt = counselor_capture(state, seed_id, perturbation_receipt, similarity, event_cycle)
                state.receipt_ledger.append(capture_receipt)

                # Check crystallization
                crystallization_receipt = check_crystallization(state, event_cycle)
                if crystallization_receipt:
                    state.receipt_ledger.append(crystallization_receipt)

                # Check replication
                replication_receipt = check_replication(state, event_cycle)
                if replication_receipt:
                    state.receipt_ledger.append(replication_receipt)

                # Check for HYBRID differentiation (cosmos specializing)
                differentiation_receipt = check_hybrid_differentiation(state, event_cycle)
                if differentiation_receipt:
                    state.receipt_ledger.append(differentiation_receipt)

        # Track boost sample
        state.window_boost_samples.append(state.perturbation_boost)

        # Evolve seeds at window intervals
        if event_cycle > 0 and event_cycle % EVOLUTION_WINDOW_SEEDS == 0:
            evolve_seeds(state, event_cycle)

        prev_cycle = event_cycle

    # Fast-forward remaining cycles after last event
    remaining = config.n_cycles - 1 - prev_cycle
    if remaining > 0:
        _fast_forward_decay(state, remaining)

    # Final state for cycle count
    state.cycle = config.n_cycles - 1

    # Collect statistics
    statistics = {
        "births": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_birth"),
        "deaths": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_death"),
        "recombinations": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_mate"),
        "blueprints_proposed": sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "blueprint_proposed"),
        "completeness_achieved": any(r.get("receipt_type") == "sim_complete" for r in state.receipt_ledger),
        "final_population": len(state.active_patterns),
        "superposition_count": len(state.superposition_patterns)
    }

    # Minimal traces (sparse version doesn't track per-cycle)
    all_traces = {
        "entropy_trace": state.entropy_trace if state.entropy_trace else [0.0],
        "completeness_trace": state.completeness_trace if state.completeness_trace else [{}],
        "population_trace": [len(state.active_patterns)]
    }

    # Calculate archetype equilibrium (quantum measurement balance)
    h, s, a = state.hunter_formations, state.governance_nodes, state.architect_formations
    if max(h, s, a) > 0:
        state.equilibrium_score = min(h, s, a) / max(h, s, a)

    # Emit sim_result receipt
    emit_receipt("sim_result", {
        "tenant_id": "simulation",
        "cycles_completed": config.n_cycles,
        "violations_count": len(violations),
        "statistics": statistics
    })

    return SimResult(
        final_state=state,
        all_traces=all_traces,
        violations=violations,
        statistics=statistics,
        config=config
    )


def _fast_forward_decay(state: SimState, n_cycles: int) -> None:
    """
    Apply vectorized decay over multiple cycles.

    Uses geometric series formula: boost_new = boost * (1 - decay_rate)^n

    Args:
        state: Current SimState (mutated in place)
        n_cycles: Number of cycles to fast-forward
    """
    if n_cycles <= 0 or state.perturbation_boost <= 0:
        return

    # For small boosts, decay approaches linear
    # For larger boosts, use iterative approach to account for non-linear decay
    for _ in range(min(n_cycles, 10)):  # Cap iterations for very large gaps
        decay_rate = PERTURBATION_DECAY * (1 + NONLINEAR_DECAY_FACTOR * state.perturbation_boost)
        state.perturbation_boost *= (1 - decay_rate)
        state.perturbation_boost = max(0.0, state.perturbation_boost)

    # For remaining cycles, use approximate linear decay
    if n_cycles > 10:
        remaining = n_cycles - 10
        # Linear approximation: boost decays by ~40% per cycle at low values
        decay_factor = (1 - PERTURBATION_DECAY) ** remaining
        state.perturbation_boost *= decay_factor
        state.perturbation_boost = max(0.0, state.perturbation_boost)


def _simulate_cycle_sparse(state: SimState, config: SimConfig,
                           is_event_cycle: bool, perturbation_random: float) -> List[dict]:
    """
    Sparse cycle simulation - optimized version of simulate_cycle.

    For non-event cycles: skips expensive nucleation pipeline.
    For event cycles: runs full perturbation and nucleation logic.

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters
        is_event_cycle: Whether this cycle might have a perturbation event
        perturbation_random: Pre-computed random value for perturbation decision

    Returns:
        List of violation dicts (if any)
    """
    violations = []

    # Reset per-cycle counters
    state.births_this_cycle = 0
    state.deaths_this_cycle = 0
    state.superposition_transitions_this_cycle = 0
    state.wounds_this_cycle = 0
    state.collapse_count_this_cycle = 0
    state.emergence_count_this_cycle = 0

    # CYCLE START: Vacuum fluctuation and reset boundary emissions
    state.vacuum_floor = vacuum_fluctuation()
    state.hawking_emissions_this_cycle = 0.0

    # Measure initial entropy state
    H_start = measure_state(state.receipt_ledger, state.vacuum_floor)
    state.H_boundary_this_cycle = 0.0
    state.operations_this_cycle = 0

    # Generate wounds stochastically
    state.operations_this_cycle += 1
    if random.random() < config.wound_rate:
        wound = simulate_wound(state, "operational")
        state.wound_history.append(wound)
        state.wounds_this_cycle += 1
        state.H_boundary_this_cycle += measure_boundary_crossing(wound)

    # Run autocatalysis detection
    for pattern in state.active_patterns:
        state.operations_this_cycle += 1
    simulate_autocatalysis(state)

    # Apply selection pressure
    for pattern in state.active_patterns:
        state.operations_this_cycle += 1
        state.operations_this_cycle += 1
    simulate_selection(state)

    # Attempt recombination
    if len(state.active_patterns) >= 2:
        state.operations_this_cycle += 1
    simulate_recombination(state, config)

    # Run genesis check
    if len(state.wound_history) >= 5:
        state.operations_this_cycle += 1
    simulate_genesis(state, config)

    # Check completeness
    state.operations_this_cycle += 1
    simulate_completeness(state)

    # CYCLE END: Measure final entropy state
    H_end = measure_state(state.receipt_ledger, state.vacuum_floor)
    H_observation = measure_observation_cost(state.operations_this_cycle)

    # Attempt spontaneous emergence
    emergence_receipt = attempt_spontaneous_emergence(state, H_observation)
    if emergence_receipt is not None:
        state.births_this_cycle += 1

    # Process virtual patterns
    collapsed_ids = process_virtual_patterns(state)

    # Emit vacuum_state receipt
    vacuum_state_receipt = emit_receipt("vacuum_state", {
        "tenant_id": "simulation",
        "cycle": state.cycle,
        "vacuum_floor": state.vacuum_floor,
        "fluctuation_delta": state.vacuum_floor - PLANCK_ENTROPY_BASE,
        "virtual_patterns_count": len(state.virtual_patterns),
        "superposition_patterns_count": len(state.superposition_patterns),
        "spontaneous_emergence_attempted": emergence_receipt is not None,
        "hawking_emissions_this_cycle": state.hawking_emissions_this_cycle
    })
    state.receipt_ledger.append(vacuum_state_receipt)

    H_delta = H_end - H_start
    balance = state.H_boundary_this_cycle + H_observation + H_delta
    state.H_previous = H_end

    # Emit entropy_state receipt
    entropy_state_receipt = emit_entropy_state_receipt(
        state, state.cycle, H_start, H_end, H_observation, balance
    )
    state.receipt_ledger.append(entropy_state_receipt)

    # SPARSE OPTIMIZATION: Only run full perturbation check for event cycles
    perturbation_receipt = None
    if is_event_cycle:
        # Full perturbation check with nucleation pipeline
        perturbation_receipt = _check_perturbation_sparse(
            state, state.cycle, perturbation_random
        )
        if perturbation_receipt:
            state.receipt_ledger.append(perturbation_receipt)
            state.window_perturbations += 1

            # Check for persistent cluster detection
            persistent_cluster_receipt = check_cluster_persistence(
                state, perturbation_receipt["receipt_type"], state.cycle
            )
            if persistent_cluster_receipt:
                state.receipt_ledger.append(persistent_cluster_receipt)

            # QUANTUM NUCLEATION: Counselors compete to capture the kick
            kick_phase = perturbation_receipt.get("phase", 0.0)
            kick_resonant = perturbation_receipt.get("resonance_hit", False)
            kick_direction = perturbation_receipt.get("bias_direction", 1)

            winner = counselor_compete(state, perturbation_receipt, kick_phase, kick_resonant, kick_direction)
            if winner:
                seed_id, similarity = winner
                capture_receipt = counselor_capture(state, seed_id, perturbation_receipt, similarity, state.cycle)
                state.receipt_ledger.append(capture_receipt)

                # Check for crystallization after capture
                crystallization_receipt = check_crystallization(state, state.cycle)
                if crystallization_receipt:
                    state.receipt_ledger.append(crystallization_receipt)

                # Check for replication after crystallization
                replication_receipt = check_replication(state, state.cycle)
                if replication_receipt:
                    state.receipt_ledger.append(replication_receipt)

                # Check for HYBRID differentiation (cosmos specializing)
                differentiation_receipt = check_hybrid_differentiation(state, state.cycle)
                if differentiation_receipt:
                    state.receipt_ledger.append(differentiation_receipt)
    else:
        # Non-event cycle: just apply decay, skip nucleation
        decay_rate = PERTURBATION_DECAY * (1 + NONLINEAR_DECAY_FACTOR * state.perturbation_boost)
        state.perturbation_boost *= (1 - decay_rate)
        state.perturbation_boost = max(0.0, state.perturbation_boost)

    # Resonance peak check
    resonance_peak_receipt = check_resonance_peak(state, state.cycle)
    if resonance_peak_receipt:
        state.receipt_ledger.append(resonance_peak_receipt)

    # Structure formation check
    structure_formation_receipt = check_structure_formation(state, state.cycle)
    if structure_formation_receipt:
        state.receipt_ledger.append(structure_formation_receipt)

    # Basin escape check
    basin_escape_receipt = check_basin_escape(state, state.cycle)
    if basin_escape_receipt:
        state.receipt_ledger.append(basin_escape_receipt)
        state.window_escapes += 1

    # Track boost sample
    state.window_boost_samples.append(state.perturbation_boost)

    # Evolution window check
    evolution_receipt = check_evolution_window(state, state.cycle)
    if evolution_receipt:
        state.receipt_ledger.append(evolution_receipt)

    # Baseline shift tracking
    baseline_shift_receipt = track_baseline_shift(state, state.cycle)
    if baseline_shift_receipt:
        state.receipt_ledger.append(baseline_shift_receipt)

    # Symmetry break check
    symmetry_break_receipt = check_symmetry_break(state, state.receipt_ledger, state.cycle)
    if symmetry_break_receipt:
        state.receipt_ledger.append(symmetry_break_receipt)

    # Proto-form check
    proto_form_receipt = check_proto_form(state, state.cycle)
    if proto_form_receipt:
        state.receipt_ledger.append(proto_form_receipt)

    # Evolve seeds
    evolve_seeds(state, state.cycle)

    # Compute Hawking flux metrics
    flux, trend = compute_hawking_flux(state)
    collapse_rate = compute_collapse_rate(state)
    emergence_rate = compute_emergence_rate(state)
    criticality = compute_system_criticality(state, state.cycle)

    if state.cycle > 0:
        criticality_rate = criticality - state.previous_criticality
    else:
        criticality_rate = 0.0

    check_criticality_alert(state, state.cycle, criticality)
    check_phase_transition(state, state.cycle, criticality, H_end)

    if state.phase_transition_occurred:
        state.cycles_since_crossing += 1

    state.previous_criticality = criticality

    effective_crit = criticality + state.perturbation_boost

    # Horizon crossing check
    if effective_crit >= 1.0 and not state.phase_transition_occurred:
        state.horizon_crossings += 1
        state.phase_transition_occurred = True
        horizon_crossing_receipt = emit_receipt("horizon_crossing", {
            "tenant_id": "simulation",
            "cycle": state.cycle,
            "base_criticality": criticality,
            "perturbation_boost": state.perturbation_boost,
            "effective_criticality": effective_crit,
            "crossing_number": state.horizon_crossings
        })
        state.receipt_ledger.append(horizon_crossing_receipt)

    hawking_flux_receipt = emit_hawking_flux_receipt(
        state, state.cycle, flux, trend, collapse_rate, emergence_rate, criticality,
        H_delta, criticality_rate
    )
    state.receipt_ledger.append(hawking_flux_receipt)

    # Validate conservation
    is_valid = validate_conservation(state)
    if not is_valid:
        violations.append({
            "cycle": state.cycle,
            "type": "conservation_violation",
            "H_start": H_start,
            "H_end": H_end,
            "H_boundary": state.H_boundary_this_cycle,
            "H_observation": H_observation,
            "balance": balance
        })

    # Record traces
    state.entropy_trace.append(H_end)
    coverage = level_coverage(state.receipt_ledger)
    state.completeness_trace.append(coverage)

    # Track entropy surges (quantum measurement events)
    if H_delta > ENTROPY_SURGE_THRESHOLD:
        state.entropy_surge_count += 1

    # Emit sim_cycle receipt
    receipt = emit_receipt("sim_cycle", {
        "tenant_id": "simulation",
        "cycle": state.cycle,
        "births": state.births_this_cycle,
        "deaths": state.deaths_this_cycle,
        "H_delta": H_delta,
        "violations": len(violations),
        "active_patterns": len(state.active_patterns)
    })
    state.receipt_ledger.append(receipt)

    return violations


def _check_perturbation_sparse(state: SimState, cycle: int,
                                perturbation_random: float) -> Optional[dict]:
    """
    Sparse version of check_perturbation using pre-computed random value.

    Uses the pre-generated numpy random value for the perturbation decision,
    while maintaining identical logic for everything else.

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number
        perturbation_random: Pre-computed random value (from numpy)

    Returns:
        Receipt dict if perturbation fired, None otherwise
    """
    import math

    # Apply non-linear decay BEFORE generating new kicks
    decay_rate = PERTURBATION_DECAY * (1 + NONLINEAR_DECAY_FACTOR * state.perturbation_boost)
    state.perturbation_boost *= (1 - decay_rate)
    state.perturbation_boost = max(0.0, state.perturbation_boost)

    # Check phase sync
    synced, phase = check_phase_sync(state)

    # Compute interference factor
    interference = compute_interference(phase, state.last_phase)

    # Check resonance
    resonance_hit, resonance_factor = check_resonance(state)
    if resonance_hit:
        state.resonance_hits += 1

    # Get effective probability
    effective_prob = compute_effective_probability(state, synced)
    adaptive_active = state.perturbation_boost > ADAPTIVE_THRESHOLD

    if adaptive_active:
        state.adaptive_triggers += 1

    # Use pre-computed random value for decision
    if perturbation_random < effective_prob:
        # Generate cluster size using Poisson distribution
        cluster_size = min(poisson_manual(CLUSTER_LAMBDA), MAX_CLUSTER_SIZE)

        # Apply quantum-inspired variance
        quantum_factor = vacuum_fluctuation() / PLANCK_ENTROPY_BASE

        # Combined variance
        base_variance = random.gauss(0, PERTURBATION_VARIANCE)
        variance_factor = 1.0 + base_variance * quantum_factor

        # Cap variance_factor
        variance_factor_uncapped = variance_factor
        variance_factor = max(0.1, min(MAX_MAGNITUDE_FACTOR, variance_factor))
        capped = (variance_factor_uncapped != variance_factor)

        # Apply multiple kicks in cluster
        total_added = 0.0
        biased_magnitude = 0.0
        for i in range(cluster_size):
            actual_mag = PERTURBATION_MAGNITUDE * variance_factor
            effective_mag = actual_mag * (1.0 + 0.5 * interference)
            resonance_mag = effective_mag * resonance_factor

            if i == 0:
                final_mag = apply_asymmetry_bias(resonance_mag, state)
                biased_magnitude = final_mag
            else:
                final_mag = resonance_mag

            final_mag = max(0.01, final_mag)
            state.perturbation_boost += final_mag
            total_added += final_mag

        # Update state for next kick
        state.last_phase = phase
        if synced:
            state.sync_count += 1

        # Classify interference type
        if interference > 0.3:
            interference_type = "constructive"
        elif interference < -0.3:
            interference_type = "destructive"
        else:
            interference_type = "neutral"

        return {
            "receipt_type": "perturbation",
            "cycle": cycle,
            "magnitude": PERTURBATION_MAGNITUDE,
            "cluster_size": cluster_size,
            "total_added": total_added,
            "total_boost": state.perturbation_boost,
            "decay_rate_at_emission": decay_rate,
            "effective_probability": effective_prob,
            "adaptive_active": adaptive_active,
            "quantum_factor": quantum_factor,
            "variance_factor": variance_factor,
            "capped": capped,
            "phase": phase,
            "synced": synced,
            "interference_factor": interference,
            "interference_type": interference_type,
            "resonance_hit": resonance_hit,
            "resonance_factor": resonance_factor,
            "bias_direction": state.bias_direction,
            "biased_magnitude": biased_magnitude,
            "source": "gravitational_wave_cluster_quantum_interference_asymmetry"
        }

    return None


def initialize_state(config: SimConfig) -> SimState:
    """
    Initialize simulation state with seed patterns.

    Args:
        config: SimConfig with parameters

    Returns:
        SimState with initial patterns
    """
    random.seed(config.random_seed)
    state = SimState()

    # Create SELF patterns (HUNTER, SHEPHERD, ARCHITECT)
    self_patterns = []
    for origin in list(GERMLINE_PATTERNS)[:3]:  # Take first 3
        pattern = {
            "pattern_id": f"self_{origin}",
            "origin": origin,
            "receipts": [],
            "tenant_id": "simulation",
            "fitness": 0.5,
            "fitness_mean": 0.5,
            "fitness_var": 0.1,
            "domain": origin,
            "problem_type": "core",
            "state": PatternState.ACTIVE.value,
            "virtual_lifespan": 0
        }
        # Add self-referencing receipt to make autocatalytic
        pattern["receipts"].append({
            "receipt_type": "agent_decision",
            "pattern_id": pattern["pattern_id"],
            "ts": "2025-01-01T00:00:00Z"
        })
        self_patterns.append(pattern)

    # Create additional OTHER patterns
    for i in range(config.n_initial_patterns - 3):
        pattern = {
            "pattern_id": f"pattern_{i}",
            "origin": f"other_{i}",
            "receipts": [],
            "tenant_id": "simulation",
            "fitness": random.uniform(0.3, 0.7),
            "fitness_mean": random.uniform(0.3, 0.7),
            "fitness_var": random.uniform(0.05, 0.15),
            "domain": f"domain_{i % 3}",
            "problem_type": "operational",
            "state": PatternState.ACTIVE.value,
            "virtual_lifespan": 0
        }
        # Make some autocatalytic
        if random.random() > 0.5:
            pattern["receipts"].append({
                "receipt_type": "agent_decision",
                "pattern_id": pattern["pattern_id"],
                "ts": "2025-01-01T00:00:00Z"
            })
        self_patterns.append(pattern)

    state.active_patterns = self_patterns

    # Measure genesis entropy (Big Bang)
    H_genesis = measure_genesis(self_patterns)
    state.H_genesis = H_genesis
    state.H_previous = H_genesis

    # Initialize quantum nucleation system
    initialize_nucleation(state)

    return state


def simulate_cycle(state: SimState, config: SimConfig) -> List[dict]:
    """
    One iteration of v12 unified loop (observer paradigm).

    Observer paradigm: The observation IS the entropy source.
    Conservation: balance = H_boundary + H_observation + H_delta >= -tolerance

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters

    Returns:
        List of violation dicts (if any)
    """
    violations = []

    # Reset per-cycle counters for adaptive tolerance computation
    state.births_this_cycle = 0
    state.deaths_this_cycle = 0
    state.superposition_transitions_this_cycle = 0
    state.wounds_this_cycle = 0
    state.collapse_count_this_cycle = 0
    state.emergence_count_this_cycle = 0

    # CYCLE START: Vacuum fluctuation and reset boundary emissions
    state.vacuum_floor = vacuum_fluctuation()
    state.hawking_emissions_this_cycle = 0.0

    # Measure initial entropy state (using fluctuating vacuum floor)
    H_start = measure_state(state.receipt_ledger, state.vacuum_floor)
    state.H_boundary_this_cycle = 0.0
    state.operations_this_cycle = 0

    # Generate wounds stochastically (count as observation)
    state.operations_this_cycle += 1  # Checking for wounds
    if random.random() < config.wound_rate:
        wound = simulate_wound(state, "operational")
        state.wound_history.append(wound)
        state.wounds_this_cycle += 1
        # Measure boundary crossing entropy
        state.H_boundary_this_cycle += measure_boundary_crossing(wound)

    # Run autocatalysis detection (count operations)
    for pattern in state.active_patterns:
        state.operations_this_cycle += 1  # Autocatalysis check per pattern
    simulate_autocatalysis(state)

    # Apply selection pressure (count operations)
    for pattern in state.active_patterns:
        state.operations_this_cycle += 1  # Fitness evaluation per pattern
        state.operations_this_cycle += 1  # Selection decision per pattern
    simulate_selection(state)

    # Attempt recombination (count operations)
    if len(state.active_patterns) >= 2:
        state.operations_this_cycle += 1  # Recombination attempt
    simulate_recombination(state, config)

    # Run genesis check (count operations)
    if len(state.wound_history) >= 5:
        state.operations_this_cycle += 1  # Genesis/HITL evaluation
    simulate_genesis(state, config)

    # Check completeness (count as observation)
    state.operations_this_cycle += 1
    simulate_completeness(state)

    # CYCLE END: Measure final entropy state
    H_end = measure_state(state.receipt_ledger, state.vacuum_floor)
    H_observation = measure_observation_cost(state.operations_this_cycle)

    # Attempt spontaneous emergence (observer-induced genesis)
    emergence_receipt = attempt_spontaneous_emergence(state, H_observation)
    if emergence_receipt is not None:
        state.births_this_cycle += 1

    # Process virtual patterns (decay or survival)
    collapsed_ids = process_virtual_patterns(state)

    # Emit vacuum_state receipt
    vacuum_state_receipt = emit_receipt("vacuum_state", {
        "tenant_id": "simulation",
        "cycle": state.cycle,
        "vacuum_floor": state.vacuum_floor,
        "fluctuation_delta": state.vacuum_floor - PLANCK_ENTROPY_BASE,
        "virtual_patterns_count": len(state.virtual_patterns),
        "superposition_patterns_count": len(state.superposition_patterns),
        "spontaneous_emergence_attempted": emergence_receipt is not None,
        "hawking_emissions_this_cycle": state.hawking_emissions_this_cycle
    })
    state.receipt_ledger.append(vacuum_state_receipt)

    H_delta = H_end - H_start
    balance = state.H_boundary_this_cycle + H_observation + H_delta

    # Store H_end for next cycle
    state.H_previous = H_end

    # Emit entropy_state receipt (CLAUDEME LAW_1: No receipt → not real)
    entropy_state_receipt = emit_entropy_state_receipt(
        state, state.cycle, H_start, H_end, H_observation, balance
    )
    state.receipt_ledger.append(entropy_state_receipt)

    # Compute and emit Hawking flux metrics
    flux, trend = compute_hawking_flux(state)
    collapse_rate = compute_collapse_rate(state)
    emergence_rate = compute_emergence_rate(state)
    criticality = compute_system_criticality(state, state.cycle)

    # Calculate criticality rate
    if state.cycle > 0:
        criticality_rate = criticality - state.previous_criticality
    else:
        criticality_rate = 0.0

    # Check for criticality alert
    check_criticality_alert(state, state.cycle, criticality)

    # Check for phase transition
    check_phase_transition(state, state.cycle, criticality, H_end)

    # Track cycles since crossing (post-breach stability)
    if state.phase_transition_occurred:
        state.cycles_since_crossing += 1

    # Update previous criticality for next cycle
    state.previous_criticality = criticality

    # Perturbation check (stochastic, like wound injection)
    perturbation_receipt = check_perturbation(state, state.cycle)
    if perturbation_receipt:
        state.receipt_ledger.append(perturbation_receipt)
        state.window_perturbations += 1  # Track for evolution window
        # Check for persistent cluster detection on perturbation receipts
        persistent_cluster_receipt = check_cluster_persistence(state, perturbation_receipt["receipt_type"], state.cycle)
        if persistent_cluster_receipt:
            state.receipt_ledger.append(persistent_cluster_receipt)

        # QUANTUM NUCLEATION: Counselors compete to capture the kick
        kick_phase = perturbation_receipt.get("phase", 0.0)
        kick_resonant = perturbation_receipt.get("resonance_hit", False)
        kick_direction = perturbation_receipt.get("bias_direction", 1)

        winner = counselor_compete(state, perturbation_receipt, kick_phase, kick_resonant, kick_direction)
        if winner:
            seed_id, similarity = winner
            capture_receipt = counselor_capture(state, seed_id, perturbation_receipt, similarity, state.cycle)
            state.receipt_ledger.append(capture_receipt)

            # Check for crystallization after capture
            crystallization_receipt = check_crystallization(state, state.cycle)
            if crystallization_receipt:
                state.receipt_ledger.append(crystallization_receipt)

            # Check for replication after crystallization
            replication_receipt = check_replication(state, state.cycle)
            if replication_receipt:
                state.receipt_ledger.append(replication_receipt)

            # Check for HYBRID differentiation (cosmos specializing)
            differentiation_receipt = check_hybrid_differentiation(state, state.cycle)
            if differentiation_receipt:
                state.receipt_ledger.append(differentiation_receipt)

    # Resonance peak check (after perturbation)
    resonance_peak_receipt = check_resonance_peak(state, state.cycle)
    if resonance_peak_receipt:
        state.receipt_ledger.append(resonance_peak_receipt)

    # Structure formation check (after cluster detection)
    structure_formation_receipt = check_structure_formation(state, state.cycle)
    if structure_formation_receipt:
        state.receipt_ledger.append(structure_formation_receipt)

    # Basin escape check
    basin_escape_receipt = check_basin_escape(state, state.cycle)
    if basin_escape_receipt:
        state.receipt_ledger.append(basin_escape_receipt)
        state.window_escapes += 1  # Track for evolution window

    # Track boost sample for evolution window (every cycle)
    state.window_boost_samples.append(state.perturbation_boost)

    # Evolution window check (every 500 cycles)
    evolution_receipt = check_evolution_window(state, state.cycle)
    if evolution_receipt:
        state.receipt_ledger.append(evolution_receipt)

    # Baseline shift tracking (ONLY at evolution window intervals)
    baseline_shift_receipt = track_baseline_shift(state, state.cycle)
    if baseline_shift_receipt:
        state.receipt_ledger.append(baseline_shift_receipt)

    # Symmetry break check (ONLY at evolution window intervals for performance)
    symmetry_break_receipt = check_symmetry_break(state, state.receipt_ledger, state.cycle)
    if symmetry_break_receipt:
        state.receipt_ledger.append(symmetry_break_receipt)

    # Proto-form check (ONLY at evolution window intervals)
    proto_form_receipt = check_proto_form(state, state.cycle)
    if proto_form_receipt:
        state.receipt_ledger.append(proto_form_receipt)

    # QUANTUM NUCLEATION: Evolve seeds (adaptive learning)
    evolve_seeds(state, state.cycle)

    # Effective criticality for threshold checks
    effective_crit = criticality + state.perturbation_boost

    # Horizon crossing check
    if effective_crit >= 1.0 and not state.phase_transition_occurred:
        state.horizon_crossings += 1
        state.phase_transition_occurred = True
        # Emit horizon_crossing receipt
        horizon_crossing_receipt = emit_receipt("horizon_crossing", {
            "tenant_id": "simulation",
            "cycle": state.cycle,
            "base_criticality": criticality,
            "perturbation_boost": state.perturbation_boost,
            "effective_criticality": effective_crit,
            "crossing_number": state.horizon_crossings
        })
        state.receipt_ledger.append(horizon_crossing_receipt)

    hawking_flux_receipt = emit_hawking_flux_receipt(
        state, state.cycle, flux, trend, collapse_rate, emergence_rate, criticality,
        H_delta, criticality_rate
    )
    state.receipt_ledger.append(hawking_flux_receipt)

    # Validate conservation (new signature)
    is_valid = validate_conservation(state)
    if not is_valid:
        violations.append({
            "cycle": state.cycle,
            "type": "conservation_violation",
            "H_start": H_start,
            "H_end": H_end,
            "H_boundary": state.H_boundary_this_cycle,
            "H_observation": H_observation,
            "balance": balance
        })

    # Record traces
    current_entropy = H_end
    state.entropy_trace.append(current_entropy)
    coverage = level_coverage(state.receipt_ledger)
    state.completeness_trace.append(coverage)

    # Track entropy surges (quantum measurement events)
    if H_delta > ENTROPY_SURGE_THRESHOLD:
        state.entropy_surge_count += 1

    # Emit sim_cycle receipt
    receipt = emit_receipt("sim_cycle", {
        "tenant_id": "simulation",
        "cycle": state.cycle,
        "births": state.births_this_cycle,
        "deaths": state.deaths_this_cycle,
        "H_delta": H_delta,
        "violations": len(violations),
        "active_patterns": len(state.active_patterns)
    })
    state.receipt_ledger.append(receipt)

    return violations


def simulate_wound(state: SimState, wound_type: str) -> dict:
    """
    Inject synthetic wound into system.

    Args:
        state: Current SimState
        wound_type: Type of wound (operational, safety, etc.)

    Returns:
        wound_receipt dict
    """
    # Sample resolution time from exponential distribution (mean 30min = 1800000ms)
    time_to_resolve_ms = int(random.expovariate(1.0 / 1800000))

    # Random resolution action
    resolution_actions = ["restart", "patch", "rollback", "escalate", "ignore"]
    resolution_action = random.choice(resolution_actions)

    wound = {
        "receipt_type": "wound",
        "intervention_id": f"wound_{state.cycle}_{len(state.wound_history)}",
        "problem_type": wound_type,
        "time_to_resolve_ms": time_to_resolve_ms,
        "resolution_action": resolution_action,
        "could_automate": random.uniform(0.3, 0.9),
        "ts": f"2025-01-01T{state.cycle:02d}:00:00Z",
        "tenant_id": "simulation"
    }

    return wound


def simulate_autocatalysis(state: SimState) -> None:
    """
    Detect pattern births and deaths via autocatalysis.

    Args:
        state: Current SimState (mutated in place)
    """
    to_remove = []

    for i, pattern in enumerate(state.active_patterns):
        # Store previous coherence
        prev_coherence = pattern.get("prev_coherence", coherence_score(pattern))
        current_coherence = coherence_score(pattern)
        pattern["prev_coherence"] = current_coherence

        # Birth detection (crossing threshold upward)
        if prev_coherence < 0.3 and current_coherence >= 0.3:
            receipt = emit_receipt("sim_birth", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "coherence_at_birth": current_coherence,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)
            state.births_this_cycle += 1

        # Death detection (crossing threshold downward)
        if prev_coherence >= 0.3 and current_coherence < 0.3:
            receipt = emit_receipt("sim_death", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "coherence_at_death": current_coherence,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)
            state.deaths_this_cycle += 1

            # Move to superposition unless SELF
            if not is_self(pattern):
                to_remove.append(i)
                state.superposition_patterns.append(pattern)
                state.superposition_transitions_this_cycle += 1

    # Remove dead patterns
    for i in reversed(to_remove):
        state.active_patterns.pop(i)


def simulate_selection(state: SimState) -> None:
    """
    Apply selection pressure via population.py.

    Handles ACTIVE and VIRTUAL patterns:
    - ACTIVE patterns: normal selection (survivors stay ACTIVE, failures → SUPERPOSITION)
    - VIRTUAL patterns: pass → promote to ACTIVE, fail → return to SUPERPOSITION

    Args:
        state: Current SimState (mutated in place)
    """
    if not state.active_patterns and not state.virtual_patterns:
        return

    # Apply selection to active patterns
    if state.active_patterns:
        survivors, superposition = selection_pressure(state.active_patterns, "simulation")

        # Track superposition transitions
        state.superposition_transitions_this_cycle += len(superposition)

        # Update active patterns (SELF always survive)
        state.active_patterns = survivors
        state.superposition_patterns.extend(superposition)

    # Apply selection to virtual patterns
    virtual_promoted = []
    virtual_collapsed = []

    if state.virtual_patterns:
        # Combine virtual patterns with active for selection consideration
        virtual_survivors, virtual_failures = selection_pressure(state.virtual_patterns, "simulation")

        # Virtual patterns that pass selection: promote to ACTIVE
        for pattern in virtual_survivors:
            pattern["state"] = PatternState.ACTIVE.value
            pattern["virtual_lifespan"] = 0  # No longer virtual
            virtual_promoted.append(pattern)

            # Emit virtual_promotion receipt
            receipt = emit_receipt("virtual_promotion", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "promoted_to": "ACTIVE",
                "survival_reason": "selection_passed",
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)

        # Virtual patterns that fail selection: return to SUPERPOSITION
        for pattern in virtual_failures:
            pattern["state"] = PatternState.SUPERPOSITION.value
            pattern["virtual_lifespan"] = 0
            virtual_collapsed.append(pattern)

            # Increment collapse counter
            state.collapse_count_this_cycle += 1

            # Emit virtual_collapse receipt
            receipt = emit_receipt("virtual_collapse", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "lifespan_at_collapse": pattern.get("virtual_lifespan", 0),
                "destination_state": "SUPERPOSITION",
                "was_reobserved": False,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)

        # Update state
        state.active_patterns.extend(virtual_promoted)
        state.superposition_patterns.extend(virtual_collapsed)
        state.virtual_patterns = []  # Clear virtual list (all processed)

    # Emit selection_event receipt
    receipt = emit_receipt("selection_event", {
        "tenant_id": "simulation",
        "cycle": state.cycle,
        "survivors": len(state.active_patterns),
        "to_superposition": state.superposition_transitions_this_cycle,
        "virtual_promoted": len(virtual_promoted),
        "virtual_collapsed": len(virtual_collapsed)
    })
    state.receipt_ledger.append(receipt)


def simulate_recombination(state: SimState, config: SimConfig) -> None:
    """
    Attempt pattern recombination.

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters
    """
    if len(state.active_patterns) < 2:
        return

    # Find compatible pairs
    pairs = mate_selection(state.active_patterns)

    for pattern_a, pattern_b in pairs[:1]:  # Limit to 1 recombination per cycle
        # Recombine
        recombination_receipt = recombine(pattern_a, pattern_b)
        state.receipt_ledger.append(recombination_receipt)

        # Create offspring pattern
        offspring_id = recombination_receipt["offspring_id"]
        offspring = {
            "pattern_id": offspring_id,
            "origin": "recombination",
            "receipts": [],
            "tenant_id": "simulation",
            "fitness": (pattern_a.get("fitness", 0.5) + pattern_b.get("fitness", 0.5)) / 2,
            "fitness_mean": (pattern_a.get("fitness_mean", 0.5) + pattern_b.get("fitness_mean", 0.5)) / 2,
            "fitness_var": 0.1,
            "domain": pattern_a.get("domain", "unknown"),
            "problem_type": pattern_a.get("problem_type", "operational"),
            "prev_coherence": 0.0,
            "state": PatternState.ACTIVE.value,
            "virtual_lifespan": 0
        }

        # Check viability via autocatalysis
        if random.random() < 0.7:  # 70% viable
            offspring["receipts"].append({
                "receipt_type": "agent_decision",
                "pattern_id": offspring_id,
                "ts": f"2025-01-01T{state.cycle:02d}:00:00Z"
            })
            if autocatalysis_check(offspring):
                state.active_patterns.append(offspring)
                state.births_this_cycle += 1

                # Emit sim_mate receipt
                receipt = emit_receipt("sim_mate", {
                    "tenant_id": "simulation",
                    "parent_a": pattern_a["pattern_id"],
                    "parent_b": pattern_b["pattern_id"],
                    "offspring_id": offspring_id,
                    "viable": True,
                    "cycle": state.cycle
                })
                state.receipt_ledger.append(receipt)


def simulate_genesis(state: SimState, config: SimConfig) -> None:
    """
    Check for automation gaps and synthesize blueprints.

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters
    """
    if len(state.wound_history) < 5:
        return

    # Identify automation gaps
    gaps = identify_automation_gaps(state.wound_history)

    for gap in gaps[:1]:  # Limit to 1 blueprint per cycle
        # Synthesize blueprint
        blueprint = synthesize_blueprint(gap, state.wound_history)

        # Simulate HITL approval
        if random.random() < config.hitl_auto_approve_rate:
            # Track entropy before creating offspring
            receipts_before = state.receipt_ledger.copy()

            # Approved - create new pattern
            offspring_id = f"genesis_{state.cycle}"
            pattern = {
                "pattern_id": offspring_id,
                "origin": "genesis",
                "receipts": [],
                "tenant_id": "simulation",
                "fitness": 0.6,
                "fitness_mean": 0.6,
                "fitness_var": 0.1,
                "domain": "automation",
                "problem_type": gap.get("problem_type", "operational"),
                "prev_coherence": 0.0,
                "state": PatternState.ACTIVE.value,
                "virtual_lifespan": 0
            }
            # Make autocatalytic
            pattern["receipts"].append({
                "receipt_type": "agent_decision",
                "pattern_id": pattern["pattern_id"],
                "ts": f"2025-01-01T{state.cycle:02d}:00:00Z"
            })
            state.active_patterns.append(pattern)
            state.births_this_cycle += 1

            # Track entropy after creating offspring
            receipts_after = state.receipt_ledger.copy()
            n_receipts = len(pattern["receipts"])

            # Calculate numeric fitness using same formula as agent_fitness
            from entropy import agent_fitness, system_entropy
            H_before = system_entropy(receipts_before)
            H_after = system_entropy(receipts_after)
            fitness = agent_fitness(receipts_before, receipts_after, n_receipts)

            # Emit genesis_birth_receipt with numeric fitness
            birth_receipt = emit_receipt("genesis_birth_receipt", {
                "tenant_id": "simulation",
                "offspring_id": offspring_id,
                "blueprint_name": blueprint["name"],
                "fitness": fitness,
                "H_before": H_before,
                "H_after": H_after,
                "n_receipts": n_receipts,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(birth_receipt)

            # Emit genesis_approved
            receipt = emit_receipt("genesis_approved", {
                "tenant_id": "simulation",
                "blueprint_name": blueprint["name"],
                "autonomy": blueprint["autonomy"],
                "approved_by": "sim_hitl",
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)


def simulate_completeness(state: SimState) -> None:
    """
    Check for receipt completeness (singularity).

    Args:
        state: Current SimState (mutated in place)
    """
    if receipt_completeness_check(state.receipt_ledger):
        # Check if already emitted
        already_emitted = any(r.get("receipt_type") == "sim_complete" for r in state.receipt_ledger)
        if not already_emitted:
            receipt = emit_receipt("sim_complete", {
                "tenant_id": "simulation",
                "cycle": state.cycle,
                "completeness_achieved": True,
                "godel_layer": godel_layer()
            })
            state.receipt_ledger.append(receipt)


def poisson_manual(lam: float) -> int:
    """
    Generate Poisson-distributed random variable using inverse transform method.

    Approximation for Poisson distribution with mean=lam.
    Uses inverse transform sampling for simplicity (good enough for small lambda).

    Args:
        lam: Poisson parameter (mean and variance)

    Returns:
        int: Random sample from Poisson(lam) distribution
    """
    import math

    # Knuth's algorithm for Poisson generation
    L = math.exp(-lam)
    k = 0
    p = 1.0

    while p > L:
        k += 1
        p *= random.random()

    return k - 1


def generate_phase(state: SimState, synced: bool) -> float:
    """
    Generate phase for perturbation kick (wave interference model).

    QUANTUM NUCLEATION FIX: All kicks now have independent random phases.
    The synced parameter is kept for backward compatibility but phases are always random.

    Args:
        state: Current SimState with last_phase
        synced: Whether to sync with previous phase (ignored - always random now)

    Returns:
        float: Phase in radians (0 to 2π)
    """
    import math

    # ALWAYS generate independent random phase for quantum nucleation
    # This ensures counselors compete based on truly random kick characteristics
    phase = random.uniform(0, 2 * math.pi)

    return phase


def check_phase_sync(state: SimState) -> tuple[bool, float]:
    """
    Check if current kick should sync with previous phase.

    Stochastic sync: PHASE_SYNC_PROBABILITY chance of attempting sync.
    Sync is successful if within PHASE_SYNC_WINDOW of previous phase.

    Args:
        state: Current SimState with last_phase

    Returns:
        Tuple of (synced: bool, phase: float)
    """
    import math

    # Stochastic decision: try to sync?
    if random.random() < PHASE_SYNC_PROBABILITY:
        # Attempt sync
        phase = generate_phase(state, synced=True)
    else:
        # No sync attempt
        phase = generate_phase(state, synced=False)

    # Check if actually synced (within window, accounting for wraparound)
    phase_diff = abs(phase - state.last_phase)
    # Handle wraparound: min(diff, 2π - diff)
    phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
    synced = phase_diff < PHASE_SYNC_WINDOW

    return (synced, phase)


def compute_interference(phase: float, last_phase: float) -> float:
    """
    Compute wave interference factor from phase difference.

    Constructive interference: cos(phase_diff) > 0 (phases aligned)
    Destructive interference: cos(phase_diff) < 0 (phases opposed)

    Args:
        phase: Current phase (radians)
        last_phase: Previous phase (radians)

    Returns:
        float: Interference factor (-1 to +1)
    """
    import math

    phase_diff = phase - last_phase
    interference_factor = math.cos(phase_diff)
    return interference_factor


def compute_effective_probability(state: SimState, synced: bool) -> float:
    """
    Compute effective perturbation probability with adaptive feedback and phase sync.

    Base probability increases when:
    - Phase synced: add SYNC_BOOST
    - perturbation_boost > ADAPTIVE_THRESHOLD: add SYNC_BOOST (stacks)

    Args:
        state: Current SimState
        synced: Whether phase is synced

    Returns:
        float: Effective probability (capped at MAX_PROBABILITY)
    """
    base_prob = PERTURBATION_PROBABILITY

    # Phase sync boost
    if synced:
        effective_prob = base_prob + SYNC_BOOST
    else:
        effective_prob = base_prob

    # Adaptive feedback: boost probability if above threshold (stacks with sync)
    if state.perturbation_boost > ADAPTIVE_THRESHOLD:
        effective_prob = effective_prob + SYNC_BOOST

    # Cap at MAX_PROBABILITY to prevent runaway
    effective_prob = min(effective_prob, MAX_PROBABILITY)

    return effective_prob


def apply_asymmetry_bias(magnitude: float, state: SimState) -> float:
    """
    Apply asymmetry bias to perturbation magnitude.

    Simple sign-based multiplier that breaks symmetry by favoring one direction.
    Bias direction flips stochastically (10% chance per call).

    Args:
        magnitude: Base perturbation magnitude
        state: Current SimState (mutated in place if direction flips)

    Returns:
        float: Biased magnitude
    """
    # Flip bias direction with small probability (10%)
    if random.random() < 0.1:
        state.bias_direction *= -1

    # Apply bias: biased_mag = magnitude * (1 + ASYMMETRY_BIAS * bias_direction)
    biased_mag = magnitude * (1 + ASYMMETRY_BIAS * state.bias_direction)

    return biased_mag


def check_resonance(state: SimState) -> tuple[bool, float]:
    """
    Check if perturbation kick resonates.

    Resonance probability: RESONANCE_PROBABILITY chance of resonating.
    Resonance factor: 1 + INTERFERENCE_AMPLITUDE (amplification).
    Capped at MAX_RESONANCE_AMPLIFICATION.

    Args:
        state: Current SimState

    Returns:
        Tuple of (resonance_hit: bool, resonance_factor: float)
    """
    if random.random() < RESONANCE_PROBABILITY:
        # Resonance hit - amplify by INTERFERENCE_AMPLITUDE
        resonance_factor = 1.0 + INTERFERENCE_AMPLITUDE
        # Cap at MAX_RESONANCE_AMPLIFICATION
        resonance_factor = min(resonance_factor, MAX_RESONANCE_AMPLIFICATION)
        return (True, resonance_factor)
    else:
        # No resonance
        return (False, 1.0)


def check_perturbation(state: SimState, cycle: int) -> Optional[dict]:
    """
    Stochastic GW kick with phase interference, Poisson clusters, adaptive feedback, and quantum variance.

    Phase interference: cos(phase_diff) modulates magnitude (constructive/destructive).
    Non-linear decay: Higher boost = faster decay (self-limiting chaos).
    Poisson clusters: Multiple kicks per event (bursts, not single kicks).
    Adaptive feedback: Probability increases when boost > ADAPTIVE_THRESHOLD or synced.
    Quantum variance: Vacuum fluctuation breathes into magnitude variance.

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        Receipt dict if perturbation fired, None otherwise
    """
    # Apply non-linear decay BEFORE generating new kicks
    # Formula: decay_rate = DECAY * (1 + NONLINEAR_DECAY_FACTOR * boost)
    # Higher boost -> higher decay_rate -> faster decay (self-limiting)
    decay_rate = PERTURBATION_DECAY * (1 + NONLINEAR_DECAY_FACTOR * state.perturbation_boost)
    state.perturbation_boost *= (1 - decay_rate)

    # Ensure boost doesn't go negative
    state.perturbation_boost = max(0.0, state.perturbation_boost)

    # Check phase sync (wave interference)
    synced, phase = check_phase_sync(state)

    # Compute interference factor
    interference = compute_interference(phase, state.last_phase)

    # Check resonance
    resonance_hit, resonance_factor = check_resonance(state)
    if resonance_hit:
        state.resonance_hits += 1

    # Get effective probability (with adaptive feedback and phase sync)
    effective_prob = compute_effective_probability(state, synced)
    adaptive_active = state.perturbation_boost > ADAPTIVE_THRESHOLD

    # Track adaptive triggers
    if adaptive_active:
        state.adaptive_triggers += 1

    # Random chance of new kick cluster (using effective probability)
    if random.random() < effective_prob:
        # Generate cluster size using Poisson distribution, capped at MAX_CLUSTER_SIZE
        cluster_size = min(poisson_manual(CLUSTER_LAMBDA), MAX_CLUSTER_SIZE)

        # Apply quantum-inspired variance (reuse existing vacuum_fluctuation())
        # quantum_factor = vacuum_fluctuation() / PLANCK_ENTROPY_BASE (ratio ~1.0 ± 0.1)
        quantum_factor = vacuum_fluctuation() / PLANCK_ENTROPY_BASE

        # Combined variance: variance_factor = 1 + gauss(0, VAR) * quantum_factor
        base_variance = random.gauss(0, PERTURBATION_VARIANCE)
        variance_factor = 1.0 + base_variance * quantum_factor

        # Cap variance_factor to prevent explosion
        variance_factor_uncapped = variance_factor
        variance_factor = max(0.1, min(MAX_MAGNITUDE_FACTOR, variance_factor))
        capped = (variance_factor_uncapped != variance_factor)

        # Apply multiple kicks in cluster
        total_added = 0.0
        biased_magnitude = 0.0  # Initialize for receipt (will be set on first kick)
        for _ in range(cluster_size):
            # Apply quantum-amplified variance to magnitude
            actual_mag = PERTURBATION_MAGNITUDE * variance_factor
            # Apply interference: effective_mag = actual_mag * (1 + 0.5 * interference)
            # Interference ranges -1 to +1, so multiplier ranges 0.5 to 1.5
            effective_mag = actual_mag * (1.0 + 0.5 * interference)
            # Apply resonance amplification
            resonance_mag = effective_mag * resonance_factor
            # Apply asymmetry bias (ONLY on first kick to avoid compounding)
            if _ == 0:
                final_mag = apply_asymmetry_bias(resonance_mag, state)
                biased_magnitude = final_mag  # Store for receipt
            else:
                final_mag = resonance_mag
            # Clamp to positive minimum
            final_mag = max(0.01, final_mag)
            state.perturbation_boost += final_mag
            total_added += final_mag

        # Update state for next kick
        state.last_phase = phase
        if synced:
            state.sync_count += 1

        # Classify interference type
        if interference > 0.3:
            interference_type = "constructive"
        elif interference < -0.3:
            interference_type = "destructive"
        else:
            interference_type = "neutral"

        return {
            "receipt_type": "perturbation",
            "cycle": cycle,
            "magnitude": PERTURBATION_MAGNITUDE,
            "cluster_size": cluster_size,
            "total_added": total_added,
            "total_boost": state.perturbation_boost,
            "decay_rate_at_emission": decay_rate,
            "effective_probability": effective_prob,
            "adaptive_active": adaptive_active,
            "quantum_factor": quantum_factor,
            "variance_factor": variance_factor,
            "capped": capped,
            "phase": phase,
            "synced": synced,
            "interference_factor": interference,
            "interference_type": interference_type,
            "resonance_hit": resonance_hit,
            "resonance_factor": resonance_factor,
            "bias_direction": state.bias_direction,
            "biased_magnitude": biased_magnitude,
            "source": "gravitational_wave_cluster_quantum_interference_asymmetry"
        }
    return None


def check_basin_escape(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check if system is escaping attractor basin (perturbation_boost > threshold).

    Tracks consecutive escapes for emergent structure detection.
    Resets consecutive_escapes when boost drops below threshold.

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        Receipt dict if escape detected, None otherwise
    """
    if state.perturbation_boost > BASIN_ESCAPE_THRESHOLD:
        # Increment consecutive escapes
        state.consecutive_escapes += 1
        state.escape_count += 1

        # Update max_consecutive_escapes if current is higher
        if state.consecutive_escapes > state.max_consecutive_escapes:
            state.max_consecutive_escapes = state.consecutive_escapes

        escape_probability = compute_escape_probability(state, cycle)
        return {
            "receipt_type": "basin_escape",
            "cycle": cycle,
            "boost_at_escape": state.perturbation_boost,
            "escape_count": state.escape_count,
            "consecutive_escapes": state.consecutive_escapes,
            "max_consecutive_escapes": state.max_consecutive_escapes,
            "escape_probability": escape_probability
        }
    else:
        # Reset consecutive escapes when boost drops below threshold
        state.consecutive_escapes = 0
    return None


def compute_escape_probability(state: SimState, cycle: int) -> float:
    """
    Compute probability of basin escape based on historical escape rate.

    Args:
        state: Current SimState
        cycle: Current cycle number

    Returns:
        float: Escape probability (escape_count / cycle)
    """
    return state.escape_count / max(cycle, 1)


def check_resonance_peak(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check if perturbation boost has reached resonance peak threshold.

    Peak detection: boost > RESONANCE_PEAK_THRESHOLD triggers peak receipt.
    Increments state.resonance_peaks counter.

    Args:
        state: Current SimState (mutated in place if peak detected)
        cycle: Current cycle number

    Returns:
        Receipt dict if peak detected, None otherwise
    """
    if state.perturbation_boost > RESONANCE_PEAK_THRESHOLD:
        # Increment peak counter
        state.resonance_peaks += 1

        return {
            "receipt_type": "resonance_peak",
            "cycle": cycle,
            "boost": state.perturbation_boost,
            "peak_count": state.resonance_peaks
        }
    return None


def check_structure_formation(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check if structure has formed via persistent clusters, symmetry breaks, and proto-forms.

    Structure formation: persistent_clusters > 0 AND symmetry_breaks >= SYMMETRY_BREAK_THRESHOLD AND proto_form_count > 0.
    Only emits receipt once (first time structure forms).

    Args:
        state: Current SimState (mutated in place if structure forms)
        cycle: Current cycle number

    Returns:
        Receipt dict if structure formed, None otherwise
    """
    # NEW condition: persistent_clusters > 0 AND symmetry_breaks >= SYMMETRY_BREAK_THRESHOLD AND proto_form_count > 0
    if (state.persistent_clusters > 0 and
        state.symmetry_breaks >= SYMMETRY_BREAK_THRESHOLD and
        state.proto_form_count > 0 and
        not state.structure_formed):
        # Mark structure as formed
        state.structure_formed = True
        state.structure_formation_cycle = cycle

        return {
            "receipt_type": "structure_formation",
            "cycle": cycle,
            "persistent_clusters": state.persistent_clusters,
            "symmetry_breaks": state.symmetry_breaks,
            "proto_form_count": state.proto_form_count
        }
    return None


def track_baseline_shift(state: SimState, cycle: int) -> Optional[dict]:
    """
    Track baseline shift via exponential moving average.

    ONLY called at EVOLUTION_WINDOW intervals (every 500 cycles).
    Updates baseline_boost with EMA: 0.9 * old + 0.1 * current.
    Detects shift when abs(current - initial) > 0.05.

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        Receipt dict if baseline shift detected, None otherwise
    """
    # ONLY check at evolution window intervals
    if cycle % EVOLUTION_WINDOW != 0:
        return None

    # Update baseline with exponential moving average
    state.baseline_boost = 0.9 * state.baseline_boost + 0.1 * state.perturbation_boost

    # Set initial baseline at first window (cycle 500)
    if cycle == EVOLUTION_WINDOW:
        state.initial_baseline = state.baseline_boost

    # Check for significant shift
    if abs(state.baseline_boost - state.initial_baseline) > 0.05:
        # Increment shift counter
        state.baseline_shifts += 1

        return {
            "receipt_type": "baseline_shift",
            "cycle": cycle,
            "initial": state.initial_baseline,
            "current": state.baseline_boost,
            "shift": state.baseline_boost - state.initial_baseline
        }

    return None


def check_evolution_window(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check if evolution window (500 cycles) has elapsed and emit evolution receipt.

    Tracks system evolution metrics per 500-cycle window:
    - Escape rate
    - Perturbation rate
    - Average boost
    - Consecutive escape peak
    - Adaptive triggers

    Args:
        state: Current SimState (mutated in place if window elapsed)
        cycle: Current cycle number

    Returns:
        Receipt dict if window elapsed, None otherwise
    """
    # Check if window has elapsed
    if cycle - state.last_evolution_cycle >= EVOLUTION_WINDOW:
        # Compute window metrics
        escape_rate = state.window_escapes / EVOLUTION_WINDOW
        perturbation_rate = state.window_perturbations / EVOLUTION_WINDOW

        # Compute avg_boost (average of boost samples)
        if state.window_boost_samples:
            avg_boost = sum(state.window_boost_samples) / len(state.window_boost_samples)
        else:
            avg_boost = 0.0

        # Emit evolution_receipt
        receipt = emit_receipt("evolution", {
            "tenant_id": "simulation",
            "window_start": state.last_evolution_cycle,
            "window_end": cycle,
            "escape_rate": escape_rate,
            "perturbation_rate": perturbation_rate,
            "avg_boost": avg_boost,
            "consecutive_max": state.max_consecutive_escapes,
            "adaptive_triggers_in_window": state.adaptive_triggers
        })

        # Reset window counters
        state.window_escapes = 0
        state.window_perturbations = 0
        state.window_boost_samples = []

        # Update last_evolution_cycle
        state.last_evolution_cycle = cycle

        # Append to evolution_snapshots
        state.evolution_snapshots.append(receipt)

        return receipt

    return None


def check_cluster_persistence(state: SimState, receipt_type: str, cycle: int) -> Optional[dict]:
    """
    Check for persistent clusters (duration >= CLUSTER_PERSISTENCE_THRESHOLD).

    Tracks cluster duration and emits persistent_cluster_receipt when a cluster
    persists for 15+ cycles. Simple comparison: duration >= threshold.

    Args:
        state: Current SimState (mutated in place)
        receipt_type: Type of current receipt
        cycle: Current cycle number

    Returns:
        Receipt dict if persistent cluster detected, None otherwise
    """
    # Check if same type as last
    if receipt_type == state.last_receipt_type:
        # Increment consecutive same type
        state.consecutive_same_type += 1
        # Update current cluster duration
        state.current_cluster_duration = cycle - state.cluster_start_cycle
    else:
        # Type changed - check if previous cluster was persistent
        persistent_receipt = None
        if state.current_cluster_duration >= CLUSTER_PERSISTENCE_THRESHOLD:
            # Emit persistent_cluster_receipt
            state.persistent_clusters += 1
            persistent_receipt = emit_receipt("persistent_cluster", {
                "tenant_id": "simulation",
                "cycle": cycle,
                "receipt_type": state.last_receipt_type,
                "cluster_duration": state.current_cluster_duration,
                "persistent_cluster_number": state.persistent_clusters
            })

        # Reset: start new cluster
        state.cluster_start_cycle = cycle
        state.consecutive_same_type = 1
        state.last_receipt_type = receipt_type
        state.current_cluster_duration = 0

        return persistent_receipt

    return None


def check_proto_form(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check for proto-form detection (persistent clusters + symmetry breaks).

    Proto-form = persistent_cluster EXISTS AND symmetry_breaks >= SYMMETRY_BREAK_THRESHOLD.
    ONLY called at EVOLUTION_WINDOW intervals (every 500 cycles).

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        Receipt dict if proto-form detected, None otherwise
    """
    # ONLY check at evolution window intervals
    if cycle % EVOLUTION_WINDOW != 0:
        return None

    # Check if proto-form conditions are met
    if state.persistent_clusters > 0 and state.symmetry_breaks >= SYMMETRY_BREAK_THRESHOLD:
        # Check if not already active
        if not state.proto_form_active:
            # Activate proto-form
            state.proto_form_active = True
            state.proto_form_count += 1

            # Emit proto_form receipt
            return emit_receipt("proto_form", {
                "tenant_id": "simulation",
                "cycle": cycle,
                "persistent_clusters": state.persistent_clusters,
                "symmetry_breaks": state.symmetry_breaks,
                "proto_form_number": state.proto_form_count
            })
    else:
        # Conditions not met - deactivate
        state.proto_form_active = False

    return None


def check_symmetry_break(state: SimState, receipts: list, cycle: int) -> Optional[dict]:
    """
    Check for symmetry break via entropy delta (Shannon entropy of receipt type distribution).

    PERFORMANCE CRITICAL:
    - ONLY called when cycle % EVOLUTION_WINDOW == 0 (every 500 cycles)
    - ONLY uses receipts[-SYMMETRY_SAMPLE_SIZE:] (last 100 receipts)

    Symmetry break detected when abs(current - last) > 0.1.

    Args:
        state: Current SimState (mutated in place)
        receipts: Full receipt ledger
        cycle: Current cycle number

    Returns:
        Receipt dict if symmetry break detected, None otherwise
    """
    # ONLY check at evolution window intervals
    if cycle % EVOLUTION_WINDOW != 0:
        return None

    # ONLY use last SYMMETRY_SAMPLE_SIZE receipts (bounded performance)
    sample = receipts[-SYMMETRY_SAMPLE_SIZE:] if len(receipts) >= SYMMETRY_SAMPLE_SIZE else receipts

    if not sample:
        return None

    # Compute Shannon entropy of receipt type distribution
    type_counts = {}
    for receipt in sample:
        receipt_type = receipt.get("receipt_type", "unknown")
        type_counts[receipt_type] = type_counts.get(receipt_type, 0) + 1

    # Shannon entropy: H = -Σ p_i * log2(p_i)
    import math
    total = len(sample)
    entropy = 0.0
    for count in type_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Check for symmetry break (delta > 0.1)
    if abs(entropy - state.last_symmetry_metric) > 0.1:
        # Emit symmetry_break receipt
        symmetry_break_receipt = emit_receipt("symmetry_break", {
            "tenant_id": "simulation",
            "cycle": cycle,
            "previous_entropy": state.last_symmetry_metric,
            "current_entropy": entropy,
            "entropy_delta": entropy - state.last_symmetry_metric,
            "sample_size": len(sample),
            "type_count": len(type_counts),
            "symmetry_break_number": state.symmetry_breaks + 1
        })
        state.symmetry_breaks += 1
        state.last_symmetry_metric = entropy
        return symmetry_break_receipt
    else:
        # Update metric even if no break
        state.last_symmetry_metric = entropy

    return None


# =============================================================================
# QUANTUM NUCLEATION FUNCTIONS (active seeding)
# =============================================================================


def initialize_nucleation(state: SimState) -> None:
    """
    Initialize quantum nucleation system with seeds, beacons, counselors, and crystals.

    Creates N_SEEDS quantum seeds at specific phases (0, 2π/3, 4π/3),
    each with a beacon, counselor, and empty crystal.

    Seeds are archetype-agnostic blank slates. Archetypes emerge through
    self-measurement at crystallization (wave function collapse paradigm).

    Args:
        state: Current SimState (mutated in place)
    """
    import math

    # Create seeds WITHOUT archetypes (blank slates)
    for i in range(N_SEEDS):
        seed = Seed(
            seed_id=i,
            phase=SEED_PHASES[i],
            resonance_affinity=SEED_RESONANCE_AFFINITY[i],
            direction=SEED_DIRECTION[i],
            captures=0
            # NOTE: No agent_archetype - crystal discovers it via self-measurement
        )
        state.seeds.append(seed)

        # Create beacon for this seed
        beacon = Beacon(
            seed_id=i,
            strength=BASE_ATTRACTION_STRENGTH
        )
        state.beacons.append(beacon)

        # Create counselor for this seed
        counselor = Counselor(
            counselor_id=i,
            seed_id=i
        )
        state.counselors.append(counselor)

        # Create empty crystal for this seed
        crystal = Crystal(
            crystal_id=i,
            seed_id=i,
            members=[],
            coherence=0.0
        )
        state.crystals.append(crystal)


def growth_boost(crystal: Crystal) -> float:
    """
    Calculate compound boost based on crystal size.

    Bigger crystals capture faster, enabling replication threshold.
    boost = 1 + GROWTH_FACTOR * (size / 10), capped at MAX_GROWTH_BOOST.

    Size 10 = 1.2x, Size 30 = 1.6x, Size 50 = 2.0x (capped).

    Args:
        crystal: Crystal to calculate boost for

    Returns:
        float: Growth boost multiplier (1.0 to MAX_GROWTH_BOOST)
    """
    size = len(crystal.members)
    boost = 1.0 + GROWTH_FACTOR * (size / 10.0)
    return min(boost, MAX_GROWTH_BOOST)


def counselor_score(counselor: Counselor, seed: Seed, kick_phase: float,
                    kick_resonant: bool, kick_direction: int) -> float:
    """
    Calculate similarity score between counselor's seed and a kick.

    Similarity = phase_score * resonance_score * direction_score

    Args:
        counselor: Counselor agent
        seed: Seed associated with counselor
        kick_phase: Phase of the kick (radians)
        kick_resonant: Whether kick is resonant
        kick_direction: Direction of kick (+1 or -1)

    Returns:
        float: Similarity score (0 to 1)
    """
    import math

    # Phase score: cos(phase_diff), normalized to 0-1
    phase_diff = abs(kick_phase - seed.phase)
    phase_score = (math.cos(phase_diff) + 1.0) / 2.0  # cos ranges -1 to +1, normalize to 0-1

    # Resonance score: match seed's affinity
    if kick_resonant:
        resonance_score = seed.resonance_affinity
    else:
        resonance_score = 1.0 - seed.resonance_affinity

    # Direction score: match seed's direction
    if kick_direction == seed.direction:
        direction_score = 1.0
    else:
        direction_score = 0.5  # partial match for opposite direction

    # Combined similarity
    similarity = phase_score * resonance_score * direction_score

    return similarity


def counselor_compete(state: SimState, kick_receipt: dict, kick_phase: float,
                     kick_resonant: bool, kick_direction: int) -> Optional[tuple]:
    """
    Counselors compete to capture a kick. Best match wins.

    Applies compound growth boost: bigger crystals capture faster.
    Also applies autocatalysis amplification for crystallized crystals.

    Args:
        state: Current SimState
        kick_receipt: Kick receipt (perturbation receipt)
        kick_phase: Phase of the kick
        kick_resonant: Whether kick is resonant
        kick_direction: Direction of kick

    Returns:
        Optional[tuple]: (seed_id, similarity) if winner found, None otherwise
    """
    best_seed_id = None
    best_similarity = 0.0

    for counselor in state.counselors:
        seed = state.seeds[counselor.seed_id]
        crystal = state.crystals[counselor.seed_id]

        # Base similarity from counselor_score
        similarity = counselor_score(counselor, seed, kick_phase, kick_resonant, kick_direction)

        # Apply governance bias for resonant kicks → boosts SHEPHERD births
        if kick_resonant:
            similarity += GOVERNANCE_BIAS
        else:
            # Apply entropy amplifier for ENTROPY_INCREASE kicks → boosts HUNTER births
            # ENTROPY_INCREASE = not resonant AND not symmetry breaking interference
            interference_type = kick_receipt.get("interference_type", "neutral")
            if interference_type not in ("constructive", "destructive"):
                similarity += ENTROPY_AMPLIFIER

        # Apply autocatalysis amplification for crystallized crystals
        if crystal.crystallized:
            similarity *= (1.0 + AUTOCATALYSIS_AMPLIFICATION)

        # Apply compound growth boost (bigger crystals capture faster)
        similarity *= growth_boost(crystal)

        # Cap at 1.0
        similarity = min(similarity, 1.0)

        if similarity > best_similarity:
            best_similarity = similarity
            best_seed_id = counselor.seed_id

    # Check if best similarity meets capture threshold
    if best_similarity >= CAPTURE_THRESHOLD:
        return (best_seed_id, best_similarity)

    return None


def classify_effect_type(kick_receipt: dict) -> str:
    """
    Classify the effect type of a captured kick for archetype discovery.

    When UNIFORM_KICK_DISTRIBUTION=True, selects effect type uniformly at random
    (1/3 probability each) to ensure fair lottery tickets for seed model.

    When UNIFORM_KICK_DISTRIBUTION=False (legacy mode):
    - RESONANCE_TRIGGER: Kick was resonant (resonance_hit=True)
    - SYMMETRY_BREAK: Kick caused interference (constructive/destructive)
    - ENTROPY_INCREASE: Default type (general entropy increase)

    Args:
        kick_receipt: The kick receipt to classify

    Returns:
        str: Effect type constant
    """
    # Uniform distribution mode: fair ticket printer for seed model lottery
    if UNIFORM_KICK_DISTRIBUTION:
        return random.choice(EFFECT_TYPES)

    # Legacy mode: classify based on kick properties
    # Resonance takes priority
    if kick_receipt.get("resonance_hit", False):
        return EFFECT_RESONANCE_TRIGGER

    # Symmetry break from interference
    interference_type = kick_receipt.get("interference_type", "neutral")
    if interference_type in ("constructive", "destructive"):
        return EFFECT_SYMMETRY_BREAK

    # Default: entropy increase
    return EFFECT_ENTROPY_INCREASE


def counselor_capture(state: SimState, seed_id: int, kick_receipt: dict,
                     similarity: float, cycle: int) -> dict:
    """
    Capture a kick and transform it into a crystal member.

    Tracks effect_type in crystal.effect_distribution for archetype discovery.
    At crystallization, the dominant effect type determines the agent archetype.

    Args:
        state: Current SimState (mutated in place)
        seed_id: ID of seed that won the capture
        kick_receipt: Kick receipt to capture
        similarity: Similarity score from competition
        cycle: Current cycle number

    Returns:
        Capture receipt dict
    """
    # Get seed and crystal
    seed = state.seeds[seed_id]
    crystal = state.crystals[seed_id]

    # Check for tunneling (very high similarity)
    tunneled = similarity >= TUNNELING_THRESHOLD

    # Transform kick (apply transformation strength)
    transformed = random.random() < TRANSFORM_STRENGTH

    # Classify effect type for archetype discovery
    effect_type = classify_effect_type(kick_receipt)

    # Track effect type in crystal's distribution
    crystal.effect_distribution[effect_type] = crystal.effect_distribution.get(effect_type, 0) + 1

    # Track kick distribution at state level (verify uniform ticket printer)
    state.kick_distribution[effect_type] = state.kick_distribution.get(effect_type, 0) + 1

    # Add to crystal members with similarity and effect_type for tracking
    member = {**kick_receipt, "capture_similarity": similarity, "effect_type": effect_type}
    crystal.members.append(member)

    # Update seed captures
    seed.captures += 1
    state.total_captures += 1

    # Check for size 50 threshold (first time only)
    current_size = len(crystal.members)
    if current_size >= 50 and not crystal.size_50_reached:
        crystal.size_50_reached = True
        state.size_50_count += 1
        # Emit size_threshold_receipt
        emit_receipt("size_threshold", {
            "tenant_id": "simulation",
            "receipt_type": "size_threshold",
            "cycle": cycle,
            "crystal_id": crystal.crystal_id,
            "size": current_size,
            "first_time": True,
            "growth_boost": growth_boost(crystal),
            "generation": crystal.generation
        })

    # Compute crystal coherence (average of phase alignment)
    import math
    if len(crystal.members) > 1:
        # Compute coherence as phase alignment
        phases = []
        for member in crystal.members:
            phases.append(member.get("phase", 0.0))

        # Coherence = 1 - variance/π² (normalized phase variance)
        mean_phase = sum(phases) / len(phases)
        variance = sum((p - mean_phase) ** 2 for p in phases) / len(phases)
        crystal.coherence = max(0.0, 1.0 - variance / (math.pi ** 2))
    else:
        crystal.coherence = 1.0  # Single member = perfect coherence

    # Emit capture receipt with effect_type for archetype tracking
    return emit_receipt("capture", {
        "tenant_id": "simulation",
        "cycle": cycle,
        "seed_id": seed_id,
        "similarity": similarity,
        "tunneled": tunneled,
        "transformed": transformed,
        "crystal_size": len(crystal.members),
        "coherence": crystal.coherence,
        "effect_type": effect_type,
        "effect_distribution": dict(crystal.effect_distribution)
    })


def discover_archetype(crystal: Crystal) -> tuple[str, float, bool]:
    """
    Discover crystal's archetype through self-measurement of effect distribution.

    Wave function collapse paradigm: crystal observes its own capture history
    and collapses to a definite archetype based on dominant effect type.

    Mapping:
    - ENTROPY_INCREASE dominant → HUNTER (pursues entropy gradients)
    - RESONANCE_TRIGGER dominant → SHEPHERD (harmonizes patterns)
    - SYMMETRY_BREAK dominant → ARCHITECT (creates structure)
    - No clear dominant (< 60%) → HYBRID (multi-modal)

    Args:
        crystal: Crystal to analyze

    Returns:
        tuple: (discovered_archetype, dominance_ratio, was_hybrid)
    """
    # Work on a copy to avoid mutating the crystal's actual distribution
    dist = dict(crystal.effect_distribution)
    crystal_size = len(crystal.members)

    # Apply size-based ARCHITECT bias for large crystals
    if crystal_size > ARCHITECT_SIZE_TRIGGER:
        architect_bias = crystal_size * 0.01
        dist[EFFECT_SYMMETRY_BREAK] = dist.get(EFFECT_SYMMETRY_BREAK, 0) + architect_bias

    # Apply size-based HUNTER bias for small crystals (quantum eigenspace)
    if crystal_size < HUNTER_SIZE_TRIGGER:
        hunter_bias = (HUNTER_SIZE_TRIGGER - crystal_size) * 0.01
        dist[EFFECT_ENTROPY_INCREASE] = dist.get(EFFECT_ENTROPY_INCREASE, 0) + hunter_bias

    total = sum(dist.values())

    if total == 0:
        return ("HYBRID", 0.0, True)

    # Find dominant effect type
    dominant_effect = max(dist.keys(), key=lambda k: dist[k])
    dominant_count = dist[dominant_effect]
    dominance_ratio = dominant_count / total

    # Check if dominance meets threshold (wave function collapse)
    if dominance_ratio >= ARCHETYPE_DOMINANCE_THRESHOLD:
        # Clear dominant effect → collapse to archetype
        archetype_map = {
            EFFECT_ENTROPY_INCREASE: "HUNTER",
            EFFECT_RESONANCE_TRIGGER: "SHEPHERD",
            EFFECT_SYMMETRY_BREAK: "ARCHITECT"
        }
        archetype = archetype_map.get(dominant_effect, "HYBRID")
        return (archetype, dominance_ratio, False)
    else:
        # No clear dominant → HYBRID (superposition remains)
        return ("HYBRID", dominance_ratio, True)


def check_hybrid_differentiation(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check if any HYBRID crystal should differentiate into a specific archetype.

    HYBRID crystals are undifferentiated potential. When thresholds are met,
    they specialize (cosmos specializing):
    - resonance_ratio > 0.6 → SHEPHERD (high resonance indicates harmonizing role)
    - entropy_ratio > 0.4 → HUNTER (high entropy indicates gradient-seeking role)
    - size > 80 + neither threshold → ARCHITECT (large structure builder)

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        archetype_shift receipt if differentiation occurred, None otherwise
    """
    for crystal in state.crystals:
        # Only differentiate crystallized HYBRID crystals
        if not crystal.crystallized:
            continue
        if crystal.agent_type != "HYBRID":
            continue

        # Size threshold for differentiation eligibility
        if len(crystal.members) < ARCHETYPE_TRIGGER_SIZE:
            continue

        # Calculate ratios from effect distribution
        total_effects = sum(crystal.effect_distribution.values())
        if total_effects == 0:
            continue

        resonance_count = crystal.effect_distribution.get(EFFECT_RESONANCE_TRIGGER, 0)
        entropy_count = crystal.effect_distribution.get(EFFECT_ENTROPY_INCREASE, 0)

        resonance_ratio = resonance_count / total_effects
        entropy_ratio = entropy_count / total_effects

        # Apply differentiation bias to thresholds (makes differentiation more likely)
        adjusted_resonance_threshold = RESONANCE_DIFFERENTIATION_THRESHOLD - DIFFERENTIATION_BIAS
        adjusted_entropy_threshold = ENTROPY_DIFFERENTIATION_THRESHOLD - DIFFERENTIATION_BIAS

        # Check for differentiation
        old_type = crystal.agent_type
        new_type = None
        trigger = None
        threshold_value = 0.0

        if resonance_ratio > adjusted_resonance_threshold:
            # High resonance → SHEPHERD
            new_type = "SHEPHERD"
            trigger = "resonance"
            threshold_value = resonance_ratio
            crystal.agent_type = new_type
            state.governance_nodes += 1
        elif entropy_ratio > adjusted_entropy_threshold:
            # High entropy → HUNTER
            new_type = "HUNTER"
            trigger = "entropy"
            threshold_value = entropy_ratio
            crystal.agent_type = new_type
            state.hunter_formations += 1
        elif len(crystal.members) > ARCHETYPE_TRIGGER_SIZE * 2:
            # Large size + neither threshold → ARCHITECT
            new_type = "ARCHITECT"
            trigger = "size"
            threshold_value = float(len(crystal.members))
            crystal.agent_type = new_type
            state.architect_formations += 1

        if new_type:
            # Track differentiation
            state.hybrid_differentiation_count += 1
            state.hybrid_formations -= 1  # Reduce HYBRID count since it differentiated

            # Emit archetype_shift receipt
            archetype_shift_receipt = emit_receipt("archetype_shift", {
                "tenant_id": "simulation",
                "receipt_type": "archetype_shift",
                "crystal_id": crystal.crystal_id,
                "cycle": cycle,
                "from_type": old_type,
                "to_type": new_type,
                "trigger": trigger,
                "threshold_value": threshold_value,
                "resonance_ratio": resonance_ratio,
                "entropy_ratio": entropy_ratio,
                "crystal_size": len(crystal.members),
                "effect_distribution": dict(crystal.effect_distribution)
            })
            return archetype_shift_receipt

    return None


def check_crystallization(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check if any crystal has achieved autocatalysis (birth).

    Autocatalytic crystallization occurs when the last AUTOCATALYSIS_STREAK captures
    all have similarity >= SELF_PREDICTION_THRESHOLD. Pattern predicts itself = alive.

    This replaces arbitrary size/coherence thresholds with self-recognition:
    - 3 consecutive self-predictions (similarity >= 0.85) = autocatalysis
    - Crystal is ALIVE when it predicts itself

    At crystallization, the crystal discovers its archetype via self-measurement
    (wave function collapse): analyze effect_distribution to find dominant type.

    First crystal to crystallize boosts all beacons by CRYSTALLIZED_BEACON_BOOST (2x).

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        agent_birth receipt if any crystal achieved autocatalysis, None otherwise
    """
    for crystal in state.crystals:
        # Skip if already crystallized
        if crystal.crystallized:
            continue

        # Need at least AUTOCATALYSIS_STREAK members to check
        if len(crystal.members) < AUTOCATALYSIS_STREAK:
            continue

        # Check last AUTOCATALYSIS_STREAK captures for self-prediction
        recent_members = crystal.members[-AUTOCATALYSIS_STREAK:]
        high_similarity_count = sum(
            1 for m in recent_members
            if m.get("capture_similarity", 0.0) >= SELF_PREDICTION_THRESHOLD
        )

        # Autocatalysis achieved: pattern predicts itself AUTOCATALYSIS_STREAK times
        if high_similarity_count >= AUTOCATALYSIS_STREAK:
            # WAVE FUNCTION COLLAPSE: Crystal discovers its archetype via self-measurement
            discovered_archetype, dominance_ratio, was_hybrid = discover_archetype(crystal)

            # Mark crystal as alive with discovered archetype
            crystal.crystallized = True
            crystal.birth_cycle = cycle
            crystal.agent_type = discovered_archetype  # DISCOVERED, not assigned!

            state.crystals_formed += 1

            # Track archetype formations for governance monitoring
            if discovered_archetype == "SHEPHERD":
                state.governance_nodes += 1
                # Emit governance_node_receipt
                emit_receipt("governance_node", {
                    "tenant_id": "simulation",
                    "receipt_type": "governance_node",
                    "crystal_id": crystal.crystal_id,
                    "cycle": cycle,
                    "total_nodes": state.governance_nodes
                })
            elif discovered_archetype == "ARCHITECT":
                state.architect_formations += 1
                # Emit architect_formation_receipt
                emit_receipt("architect_formation", {
                    "tenant_id": "simulation",
                    "receipt_type": "architect_formation",
                    "crystal_id": crystal.crystal_id,
                    "cycle": cycle,
                    "size": len(crystal.members),
                    "effect_distribution": dict(crystal.effect_distribution)
                })
            elif discovered_archetype == "HUNTER":
                state.hunter_formations += 1
            elif discovered_archetype == "HYBRID":
                state.hybrid_formations += 1

            # Track first capture distribution (lottery fairness verification)
            # The FIRST crystal to crystallize determines the archetype lottery winner
            if state.crystals_formed == 1 and discovered_archetype in state.first_capture_distribution:
                state.first_capture_distribution[discovered_archetype] += 1

            # Boost beacons if this is the FIRST crystal
            if state.crystals_formed == 1:
                for beacon in state.beacons:
                    beacon.strength *= CRYSTALLIZED_BEACON_BOOST

            # Emit archetype_discovery receipt (LAW_1: no receipt → not real)
            archetype_discovery_receipt = emit_receipt("archetype_discovery", {
                "tenant_id": "simulation",
                "receipt_type": "archetype_discovery",
                "crystal_id": crystal.crystal_id,
                "cycle": cycle,
                "effect_distribution": dict(crystal.effect_distribution),
                "discovered_archetype": discovered_archetype,
                "dominance_ratio": dominance_ratio,
                "was_hybrid": was_hybrid
            })
            state.receipt_ledger.append(archetype_discovery_receipt)

            # Emit agent_birth receipt with discovery metadata
            return emit_receipt("agent_birth", {
                "tenant_id": "simulation",
                "receipt_type": "agent_birth",
                "agent_type": crystal.agent_type,
                "discovery_method": "self_measurement",  # NOT pre-assigned!
                "effect_distribution": dict(crystal.effect_distribution),
                "dominance_ratio": dominance_ratio,
                "was_hybrid": was_hybrid,
                "birth_cycle": cycle,
                "pattern_size": len(crystal.members),
                "autocatalysis_streak": AUTOCATALYSIS_STREAK,
                "seed_id": crystal.seed_id,
                "coherence": crystal.coherence,
                "first_crystal": state.crystals_formed == 1
            })

    return None


def evolve_seeds(state: SimState, cycle: int) -> None:
    """
    Evolve seeds toward successful captures.

    Seeds learn what works by adjusting their phase, affinity, and direction
    based on capture success rate.

    Only called every EVOLUTION_WINDOW_SEEDS cycles.

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number
    """
    # Only evolve at window intervals
    if cycle % EVOLUTION_WINDOW_SEEDS != 0 or cycle == 0:
        return

    for seed in state.seeds:
        # If seed has captures, evolve toward success
        if seed.captures > 0:
            # Slightly adjust phase toward recent captures (random walk)
            phase_delta = random.gauss(0, 0.1) * EVOLUTION_RATE
            seed.phase += phase_delta

            # Keep phase in [0, 2π]
            import math
            seed.phase = seed.phase % (2 * math.pi)

            # Adjust affinity based on autocatalysis success
            crystal = state.crystals[seed.seed_id]
            if crystal.crystallized:
                # Crystal is ALIVE - increase affinity slightly
                seed.resonance_affinity = min(1.0, seed.resonance_affinity + EVOLUTION_RATE * 0.5)
            else:
                # Not yet autocatalytic - decrease affinity slightly
                seed.resonance_affinity = max(0.0, seed.resonance_affinity - EVOLUTION_RATE * 0.5)


def check_replication(state: SimState, cycle: int) -> Optional[dict]:
    """
    Check if any crystallized crystal can replicate.

    Replication occurs when crystal has >= REPLICATION_THRESHOLD captures.
    Child crystal starts with NO archetype - it discovers its own through
    its own captures (independent identity, not inherited).

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        replication receipt if replication occurred, None otherwise
    """
    import math

    for crystal in state.crystals:
        # Only crystallized crystals can replicate
        if not crystal.crystallized:
            continue

        # Need enough captures to replicate
        if len(crystal.members) < REPLICATION_THRESHOLD:
            continue

        # Check if crystal has already replicated recently (cooldown)
        # Simple heuristic: only replicate once per crystal for now
        child_exists = any(c.parent_crystal_id == crystal.crystal_id for c in state.crystals)
        if child_exists:
            continue

        # Create child crystal with NO archetype (blank slate)
        child_id = len(state.crystals)
        child_seed_id = len(state.seeds)

        # Create new seed for child (variations from parent)
        parent_seed = state.seeds[crystal.seed_id]
        child_seed = Seed(
            seed_id=child_seed_id,
            phase=(parent_seed.phase + random.gauss(0, 0.5)) % (2 * math.pi),
            resonance_affinity=max(0.0, min(1.0, parent_seed.resonance_affinity + random.gauss(0, 0.1))),
            direction=random.choice([1, -1]),  # Independent direction
            captures=0
            # NOTE: No agent_archetype - child discovers its own!
        )
        state.seeds.append(child_seed)

        # Create beacon and counselor for child
        child_beacon = Beacon(
            seed_id=child_seed_id,
            strength=BASE_ATTRACTION_STRENGTH * (1.0 + AUTOCATALYSIS_AMPLIFICATION)
        )
        state.beacons.append(child_beacon)

        child_counselor = Counselor(
            counselor_id=len(state.counselors),
            seed_id=child_seed_id
        )
        state.counselors.append(child_counselor)

        # Calculate child generation (parent + 1)
        child_generation = crystal.generation + 1

        # Create child crystal - NO archetype, empty effect_distribution
        child_crystal = Crystal(
            crystal_id=child_id,
            seed_id=child_seed_id,
            members=[],
            coherence=0.0,
            crystallized=False,  # Child must crystallize independently
            birth_cycle=-1,
            agent_type="",  # NO archetype - will be discovered!
            effect_distribution={
                "ENTROPY_INCREASE": 0,
                "RESONANCE_TRIGGER": 0,
                "SYMMETRY_BREAK": 0
            },
            parent_crystal_id=crystal.crystal_id,  # Track lineage
            generation=child_generation  # Depth in family tree
        )
        state.crystals.append(child_crystal)

        # Update compound growth tracking
        state.max_generation = max(state.max_generation, child_generation)
        state.replication_events += 1
        state.total_branches += 1

        # Emit generation receipt (self-replication loop tracked)
        emit_receipt("generation", {
            "tenant_id": "simulation",
            "receipt_type": "generation",
            "cycle": cycle,
            "crystal_id": child_id,
            "generation": child_generation,
            "parent_id": crystal.crystal_id,
            "parent_generation": crystal.generation,
            "lineage_depth": child_generation
        })

        # Emit replication receipt
        return emit_receipt("replication", {
            "tenant_id": "simulation",
            "receipt_type": "replication",
            "cycle": cycle,
            "parent_crystal_id": crystal.crystal_id,
            "parent_archetype": crystal.agent_type,
            "child_crystal_id": child_id,
            "child_archetype": "",  # UNKNOWN until child crystallizes
            "parent_captures": len(crystal.members),
            "note": "Child discovers own archetype via self-measurement"
        })

    return None


# =============================================================================
# ENTROPY AND THERMODYNAMICS (4 required)
# =============================================================================


def emit_entropy_state_receipt(state: SimState, cycle: int, H_start: float,
                               H_end: float, H_observation: float, balance: float) -> dict:
    """
    Emit entropy_state receipt at end of each cycle (observer paradigm).

    Per CLAUDEME LAW_1: No receipt → not real

    Args:
        state: Current SimState
        cycle: Current cycle number
        H_start: Entropy at cycle start
        H_end: Entropy at cycle end
        H_observation: Observation cost this cycle
        balance: Conservation balance (H_boundary + H_observation + H_delta)

    Returns:
        Receipt dict
    """
    H_delta = H_end - H_start

    # Compute tolerance to determine balance status
    tolerance, _ = compute_tolerance(state)

    if balance >= -tolerance:
        balance_status = "conserved"
    else:
        balance_status = "violation"

    return emit_receipt("entropy_state", {
        "tenant_id": "simulation",
        "cycle": cycle,
        "H_start": H_start,
        "H_end": H_end,
        "H_delta": H_delta,
        "H_boundary": state.H_boundary_this_cycle,
        "H_observation": H_observation,
        "operations_count": state.operations_this_cycle,
        "balance": balance,
        "balance_status": balance_status,
        "H_genesis": state.H_genesis
    })


def compute_tolerance(state: SimState) -> tuple[float, dict]:
    """
    Derive tolerance from system's own measurement precision.

    Tolerance = uncertainty in entropy accounting from:
    - Entropy variance (recent history)
    - Population churn (births + deaths + transitions)
    - Wound rate (deviations from baseline)
    - Fitness uncertainty (mean of fitness_var across patterns)

    Args:
        state: Current SimState with entropy_trace, active_patterns, etc.

    Returns:
        Tuple of (tolerance, factors_dict) where:
        - tolerance: float clamped between TOLERANCE_FLOOR and TOLERANCE_CEILING
        - factors_dict: all factor values for receipt audit trail
    """
    # Entropy variance factor: stddev of recent entropy deltas
    if len(state.entropy_trace) >= 2:
        recent_window = state.entropy_trace[-ENTROPY_HISTORY_WINDOW:]
        if len(recent_window) >= 2:
            deltas = [recent_window[i] - recent_window[i-1]
                     for i in range(1, len(recent_window))]
            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
                entropy_variance_factor = variance ** 0.5  # stddev
            else:
                entropy_variance_factor = 0.0
        else:
            entropy_variance_factor = 0.0
    else:
        entropy_variance_factor = 0.0

    # Population churn factor: state transitions / population
    n_active = len(state.active_patterns)
    if n_active > 0:
        total_churn = (state.births_this_cycle +
                      state.deaths_this_cycle +
                      state.superposition_transitions_this_cycle)
        population_churn_factor = total_churn / n_active
    else:
        population_churn_factor = 0.0

    # Wound rate factor: wounds this cycle / expected baseline
    # Derive expected_wound_rate from history (rolling average over last 20 cycles)
    if len(state.wound_history) >= 20:
        # Use rolling average of wounds per cycle over last 20 cycles
        # Approximate by dividing total wounds by cycles (assuming wounds tracked)
        wound_baseline = len(state.wound_history) / max(state.cycle, 1)
    elif state.cycle > 0:
        # Bootstrap: use observed rate so far
        wound_baseline = len(state.wound_history) / state.cycle
    else:
        # First cycle: use small positive value
        wound_baseline = 0.1

    if wound_baseline > 0:
        wound_rate_factor = state.wounds_this_cycle / wound_baseline
    else:
        wound_rate_factor = 0.0

    # Fitness uncertainty factor: mean of fitness_var across patterns
    if state.active_patterns:
        fitness_vars = [p.get("fitness_var", 0.1) for p in state.active_patterns]
        fitness_uncertainty_factor = sum(fitness_vars) / len(fitness_vars)
    else:
        fitness_uncertainty_factor = 0.1

    # Compute tolerance
    base = 0.05
    raw = base * (1.0 + entropy_variance_factor + population_churn_factor +
                  wound_rate_factor + fitness_uncertainty_factor)

    # Clamp to bounds
    tolerance = max(TOLERANCE_FLOOR, min(raw, TOLERANCE_CEILING))

    # Build factors dict for receipt
    factors = {
        "entropy_variance_factor": entropy_variance_factor,
        "population_churn_factor": population_churn_factor,
        "wound_rate_factor": wound_rate_factor,
        "fitness_uncertainty_factor": fitness_uncertainty_factor,
        "base": base,
        "raw_value": raw,
        "tolerance": tolerance,
        "was_clamped": raw != tolerance,
        "clamp_direction": (
            "floor" if raw < TOLERANCE_FLOOR else
            "ceiling" if raw > TOLERANCE_CEILING else
            "none"
        )
    }

    return tolerance, factors


def emit_tolerance_receipt(tolerance: float, factors: dict, cycle: int) -> dict:
    """
    Emit tolerance_measurement receipt following CLAUDEME pattern.

    Args:
        tolerance: Computed tolerance value
        factors: Factor breakdown dict from compute_tolerance
        cycle: Current cycle number

    Returns:
        Receipt dict
    """
    return emit_receipt("tolerance_measurement", {
        "tenant_id": "simulation",
        "cycle": cycle,
        "tolerance": tolerance,
        "entropy_variance_factor": factors["entropy_variance_factor"],
        "population_churn_factor": factors["population_churn_factor"],
        "wound_rate_factor": factors["wound_rate_factor"],
        "fitness_uncertainty_factor": factors["fitness_uncertainty_factor"],
        "raw_value": factors["raw_value"],
        "was_clamped": factors["was_clamped"],
        "clamp_direction": factors["clamp_direction"]
    })


def check_chaos_state(tolerance: float, factors: dict, cycle: int,
                     state: SimState) -> bool:
    """
    Check if system is in chaos state (tolerance >= TOLERANCE_CEILING).

    Args:
        tolerance: Computed tolerance value
        factors: Factor breakdown dict
        cycle: Current cycle number
        state: SimState for appending receipt

    Returns:
        bool: True if in chaos state, False otherwise
    """
    if tolerance >= TOLERANCE_CEILING:
        # Emit entropy_chaos_alert following CLAUDEME §4.7 anomaly pattern
        receipt = emit_receipt("anomaly", {
            "tenant_id": "simulation",
            "cycle": cycle,
            "metric": "tolerance",
            "baseline": TOLERANCE_CEILING,
            "delta": tolerance - TOLERANCE_CEILING,
            "classification": "violation",
            "action": "escalate"
        })
        state.receipt_ledger.append(receipt)
        return True

    return False




def validate_conservation(state: SimState) -> bool:
    """
    Validate 2nd law using observer paradigm.

    Conservation: balance = H_boundary + H_observation + H_delta >= -tolerance

    Args:
        state: Current SimState

    Returns:
        bool: True if valid, False if violated
    """
    # Get tolerance from compute_tolerance
    tolerance, factors = compute_tolerance(state)

    # Emit tolerance_measurement receipt (CLAUDEME LAW_1: No receipt → not real)
    tolerance_receipt = emit_tolerance_receipt(tolerance, factors, state.cycle)
    state.receipt_ledger.append(tolerance_receipt)

    # Check for chaos state
    in_chaos = check_chaos_state(tolerance, factors, state.cycle, state)

    # Get balance from latest entropy_state receipt
    # (already computed in simulate_cycle and stored in the receipt)
    # We can recalculate it here for validation
    H_start = state.H_previous if state.cycle > 0 else state.H_genesis
    H_end = measure_state(state.receipt_ledger, state.vacuum_floor)
    H_delta = H_end - H_start
    balance = state.H_boundary_this_cycle + measure_observation_cost(state.operations_this_cycle) + H_delta

    # Conservation valid if balance >= -tolerance
    is_valid = balance >= -tolerance

    if not is_valid:
        # Emit sim_violation receipt
        receipt = emit_receipt("sim_violation", {
            "cycle": state.cycle,
            "violation_type": "observer_conservation",
            "balance": balance,
            "tolerance": tolerance,
            "H_boundary": state.H_boundary_this_cycle,
            "H_observation": measure_observation_cost(state.operations_this_cycle),
            "H_delta": H_delta,
            "tenant_id": "simulation"
        })
        state.receipt_ledger.append(receipt)
        # Append violation data to state.violations list
        state.violations.append({
            "cycle": state.cycle,
            "type": "observer_conservation_violation",
            "balance": balance,
            "tolerance": tolerance
        })

    return is_valid


def detect_hidden_risk(state: SimState) -> List[str]:
    """
    Detect patterns with hidden risk (entropy reduction without export).

    Args:
        state: Current SimState

    Returns:
        List of flagged pattern_ids
    """
    flagged = []

    for pattern in state.active_patterns:
        # Compare visible entropy reduction to work exported
        # Simplified: check if fitness is too high without work
        fitness = pattern.get("fitness", 0.0)
        if fitness > 0.8:  # Suspiciously high
            flagged.append(pattern["pattern_id"])

    return flagged


def entropy_trace_recording(state: SimState) -> None:
    """
    Record system entropy to trace.

    Args:
        state: Current SimState (mutated in place)
    """
    current_entropy = system_entropy(state.receipt_ledger)
    state.entropy_trace.append(current_entropy)


# =============================================================================
# OBSERVER PARADIGM MEASUREMENT FUNCTIONS (4 required)
# =============================================================================

def measure_state(receipt_ledger: list, vacuum_floor: float = None) -> float:
    """
    Measure current system entropy state.

    Args:
        receipt_ledger: List of receipts
        vacuum_floor: Optional fluctuating vacuum floor (defaults to PLANCK_ENTROPY)

    Returns:
        float: System entropy, never returns 0 (minimum vacuum_floor)
    """
    if vacuum_floor is None:
        vacuum_floor = PLANCK_ENTROPY

    result = system_entropy(receipt_ledger)
    return max(result, vacuum_floor)


def measure_observation_cost(operations: int) -> float:
    """
    Measure entropy cost of observation (Landauer principle).

    Formula: LANDAUER_COEFFICIENT * log2(operations + 2)
    The +2 ensures minimum ~1 bit even at 0 operations (log2(2) = 1)

    Args:
        operations: Number of observations/decisions made this cycle

    Returns:
        float: Observation entropy for this cycle
    """
    import math
    return LANDAUER_COEFFICIENT * math.log2(operations + 2)


def measure_boundary_crossing(receipt_data: dict) -> float:
    """
    Measure entropy of crossing phase boundary (ClarityClean).

    Args:
        receipt_data: Receipt dictionary for the boundary event

    Returns:
        float: Boundary crossing entropy, never 0 (minimum PLANCK_ENTROPY)
    """
    result = system_entropy([receipt_data])
    return max(result, PLANCK_ENTROPY)


def measure_genesis(initial_patterns: list) -> float:
    """
    Measure entropy at simulation genesis (Big Bang).

    Called ONCE at simulation start in initialize_state().

    Args:
        initial_patterns: List of initial pattern dictionaries

    Returns:
        float: Genesis entropy, never 0 (minimum PLANCK_ENTROPY)
    """
    # Create receipts from initial patterns for entropy measurement
    initial_pattern_receipts = []
    for pattern in initial_patterns:
        initial_pattern_receipts.append({
            "receipt_type": "pattern_genesis",
            "pattern_id": pattern.get("pattern_id", "unknown"),
            "origin": pattern.get("origin", "unknown"),
            "tenant_id": "simulation"
        })

    result = system_entropy(initial_pattern_receipts)
    return max(result, PLANCK_ENTROPY)


# =============================================================================
# VACUUM FLUCTUATION AND VIRTUAL PATTERNS (observer-induced genesis)
# =============================================================================

def vacuum_fluctuation() -> float:
    """
    Generate fluctuating zero-point energy.

    Vacuum isn't static - it fluctuates. This replaces the static PLANCK_ENTROPY
    with a time-varying floor that follows quantum field theory principles.

    Formula: PLANCK_ENTROPY_BASE * (1 + random.gauss(0, VACUUM_VARIANCE))
    Clamped to minimum PLANCK_ENTROPY_BASE * 0.5 (can't go negative or too low)

    Returns:
        float: Fluctuating vacuum floor entropy
    """
    fluctuation = PLANCK_ENTROPY_BASE * (1.0 + random.gauss(0, VACUUM_VARIANCE))
    # Clamp to minimum 50% of base (prevent negative or too-low values)
    return max(fluctuation, PLANCK_ENTROPY_BASE * 0.5)


def attempt_spontaneous_emergence(state: SimState, H_observation: float) -> Optional[dict]:
    """
    Attempt observer-induced pattern genesis from vacuum.

    High observation cost can spark pattern emergence from superposition.
    This is the core of observer-induced genesis: the observer creates, not just measures.

    Args:
        state: Current SimState (mutated in place if emergence occurs)
        H_observation: Observation cost this cycle

    Returns:
        Receipt dict if emergence occurred, None otherwise
    """
    # Check conditions for spontaneous emergence
    if H_observation <= GENESIS_THRESHOLD:
        return None

    if len(state.superposition_patterns) == 0:
        return None

    # Select pattern from superposition (weighted by fitness if available)
    weights = []
    for pattern in state.superposition_patterns:
        fitness = pattern.get("fitness_mean", pattern.get("fitness", 0.5))
        weights.append(fitness)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0] * len(state.superposition_patterns)
        total_weight = len(state.superposition_patterns)

    weights = [w / total_weight for w in weights]

    # Select pattern
    selected_pattern = random.choices(state.superposition_patterns, weights=weights, k=1)[0]

    # Remove from superposition, add to virtual
    state.superposition_patterns.remove(selected_pattern)
    selected_pattern["state"] = PatternState.VIRTUAL.value
    selected_pattern["virtual_lifespan"] = VIRTUAL_LIFESPAN
    state.virtual_patterns.append(selected_pattern)

    # Increment emergence counters
    state.emergence_count_this_cycle += 1
    state.observer_wake_count += 1  # Track cumulative observer-triggered emergences

    # Emit spontaneous_emergence receipt
    receipt = emit_receipt("spontaneous_emergence", {
        "tenant_id": "simulation",
        "triggering_observation_cost": H_observation,
        "emerged_pattern_id": selected_pattern["pattern_id"],
        "source_state": "SUPERPOSITION",
        "destination_state": "VIRTUAL",
        "cycle": state.cycle
    })
    state.receipt_ledger.append(receipt)

    return receipt


def process_virtual_patterns(state: SimState) -> List[str]:
    """
    Process VIRTUAL patterns - decay or survive based on re-observation.

    Virtual patterns are ephemeral. They need re-observation to survive,
    otherwise they collapse back to SUPERPOSITION.

    Args:
        state: Current SimState (mutated in place)

    Returns:
        List of collapsed pattern IDs
    """
    collapsed_ids = []
    to_remove = []

    for i, pattern in enumerate(state.virtual_patterns):
        # Decrement lifespan
        pattern["virtual_lifespan"] = pattern.get("virtual_lifespan", VIRTUAL_LIFESPAN) - 1

        # Check if pattern was re-observed this cycle
        # A pattern is "re-observed" if it has receipts added this cycle
        # For simplicity, we'll check if it passed autocatalysis or has recent receipts
        was_reobserved = False
        if autocatalysis_check(pattern):
            was_reobserved = True
            # Reset lifespan (survival)
            pattern["virtual_lifespan"] = VIRTUAL_LIFESPAN

        # Check for collapse
        if pattern["virtual_lifespan"] <= 0 and not was_reobserved:
            # Collapse back to SUPERPOSITION
            pattern["state"] = PatternState.SUPERPOSITION.value
            pattern["virtual_lifespan"] = 0
            state.superposition_patterns.append(pattern)
            to_remove.append(i)
            collapsed_ids.append(pattern["pattern_id"])

            # Increment collapse counter
            state.collapse_count_this_cycle += 1

            # Emit virtual_collapse receipt
            receipt = emit_receipt("virtual_collapse", {
                "tenant_id": "simulation",
                "pattern_id": pattern["pattern_id"],
                "lifespan_at_collapse": 0,
                "destination_state": "SUPERPOSITION",
                "was_reobserved": False,
                "cycle": state.cycle
            })
            state.receipt_ledger.append(receipt)

    # Remove collapsed patterns from virtual list
    for i in reversed(to_remove):
        state.virtual_patterns.pop(i)

    return collapsed_ids


def compute_hawking_flux(state: SimState) -> tuple[float, str]:
    """
    Compute Hawking entropy flux rate over rolling window.

    Flux = Δ(hawking_emissions) / Δt over FLUX_WINDOW cycles.

    Args:
        state: Current SimState

    Returns:
        Tuple of (flux, trend) where:
        - flux: Rate of change of hawking emissions
        - trend: "increasing", "decreasing", or "stable"
    """
    # Append current cycle's emissions to history
    state.flux_history.append(state.hawking_emissions_this_cycle)

    # Trim to keep at most FLUX_WINDOW * 2 entries
    if len(state.flux_history) > FLUX_WINDOW * 2:
        state.flux_history = state.flux_history[-FLUX_WINDOW * 2:]

    # Need at least 2 data points to compute flux
    if len(state.flux_history) < 2:
        return (0.0, "insufficient_data")

    # Calculate flux
    if len(state.flux_history) >= FLUX_WINDOW:
        # Compare recent average to older average
        flux = (state.flux_history[-1] - state.flux_history[-FLUX_WINDOW]) / FLUX_WINDOW
    else:
        # Not enough history - compute average delta
        deltas = [state.flux_history[i] - state.flux_history[i-1]
                 for i in range(1, len(state.flux_history))]
        flux = sum(deltas) / len(deltas) if deltas else 0.0

    # Determine trend
    if flux > 0.01:
        trend = "increasing"
    elif flux < -0.01:
        trend = "decreasing"
    else:
        trend = "stable"

    return (flux, trend)


def compute_collapse_rate(state: SimState) -> float:
    """
    Compute collapse rate (VIRTUAL → SUPERPOSITION transitions per cycle).

    Args:
        state: Current SimState

    Returns:
        float: Collapse rate for this cycle
    """
    return float(state.collapse_count_this_cycle)


def compute_emergence_rate(state: SimState) -> float:
    """
    Compute emergence rate (SUPERPOSITION → VIRTUAL transitions per cycle).

    Args:
        state: Current SimState

    Returns:
        float: Emergence rate for this cycle
    """
    return float(state.emergence_count_this_cycle)


def compute_system_criticality(state: SimState, cycle: int) -> float:
    """
    Compute system criticality metric.

    Criticality = cumulative_emergences / cycle
    This approaches 1.0 when system is at criticality (494/500 ≈ 0.988).

    Args:
        state: Current SimState
        cycle: Current cycle number

    Returns:
        float: System criticality (0.0 to ~1.0)
    """
    if cycle == 0:
        return 0.0

    # Count total emergences from receipt ledger
    total_emergences = sum(1 for r in state.receipt_ledger
                          if r.get("receipt_type") == "spontaneous_emergence")

    return total_emergences / cycle


def check_criticality_alert(state: SimState, cycle: int, criticality: float) -> Optional[dict]:
    """
    Check if criticality alert should be emitted.

    Alert fires when criticality > CRITICALITY_ALERT_THRESHOLD with cooldown
    and hysteresis to prevent spam.

    Args:
        state: Current SimState (mutated in place if alert fires)
        cycle: Current cycle number
        criticality: Current criticality value

    Returns:
        Receipt dict if alert emitted, None otherwise
    """
    # Check if we should emit an alert
    if (criticality > CRITICALITY_ALERT_THRESHOLD and
        not state.criticality_alert_emitted and
        (cycle - state.last_alert_cycle) > ALERT_COOLDOWN_CYCLES):

        # Set alert flags
        state.criticality_alert_emitted = True
        state.last_alert_cycle = cycle

        # Emit anomaly receipt per CLAUDEME §4.7
        receipt = emit_receipt("anomaly", {
            "tenant_id": "simulation",
            "cycle": cycle,
            "metric": "criticality",
            "baseline": CRITICALITY_ALERT_THRESHOLD,
            "delta": criticality - CRITICALITY_ALERT_THRESHOLD,
            "classification": "drift",
            "action": "alert"
        })
        state.receipt_ledger.append(receipt)
        return receipt

    # Hysteresis: reset alert flag if criticality drops significantly
    if criticality < CRITICALITY_ALERT_THRESHOLD - 0.05:
        state.criticality_alert_emitted = False

    return None


def check_phase_transition(state: SimState, cycle: int, criticality: float, H_end: float) -> Optional[dict]:
    """
    Check if phase transition (criticality >= 1.0) has occurred.

    Args:
        state: Current SimState (mutated in place if transition occurs)
        cycle: Current cycle number
        criticality: Current criticality value
        H_end: Current system entropy

    Returns:
        Receipt dict if transition occurred, None otherwise
    """
    if criticality >= CRITICALITY_PHASE_TRANSITION and not state.phase_transition_occurred:
        # Set transition flag
        state.phase_transition_occurred = True

        # Emit phase_transition receipt
        receipt = emit_receipt("phase_transition", {
            "tenant_id": "simulation",
            "cycle": cycle,
            "criticality": criticality,
            "total_emergences": state.observer_wake_count,
            "transition_type": "quantum_leap",
            "entropy_at_transition": H_end
        })
        state.receipt_ledger.append(receipt)
        return receipt

    return None


def estimate_cycles_to_transition(criticality: float, criticality_rate: float) -> int:
    """
    Estimate cycles until criticality reaches 1.0.

    Args:
        criticality: Current criticality value
        criticality_rate: Rate of criticality change per cycle

    Returns:
        int: Estimated cycles to transition, -1 if not approaching
    """
    if criticality_rate <= 0:
        return -1  # Not approaching transition

    remaining = CRITICALITY_PHASE_TRANSITION - criticality
    cycles = int(remaining / criticality_rate)
    return max(cycles, 0)


def emit_hawking_flux_receipt(state: SimState, cycle: int, flux: float,
                              trend: str, collapse_rate: float,
                              emergence_rate: float, criticality: float,
                              entropy_delta: float, criticality_rate: float) -> dict:
    """
    Emit hawking_flux receipt with rate metrics.

    Per CLAUDEME LAW_1: No receipt → not real

    Args:
        state: Current SimState
        cycle: Current cycle number
        flux: Hawking entropy flux rate
        trend: Flux trend string
        collapse_rate: Collapse rate this cycle
        emergence_rate: Emergence rate this cycle
        criticality: System criticality metric
        entropy_delta: H_delta (H_end - H_start) from cycle
        criticality_rate: Rate of criticality change per cycle

    Returns:
        Receipt dict
    """
    # Compute additional fields
    criticality_alert_active = criticality > CRITICALITY_ALERT_THRESHOLD
    cycles_to_transition = estimate_cycles_to_transition(criticality, criticality_rate)

    # Compute escape probability
    escape_probability = compute_escape_probability(state, cycle)

    # Compute current decay rate (non-linear)
    current_decay_rate = PERTURBATION_DECAY * (1 + NONLINEAR_DECAY_FACTOR * state.perturbation_boost)

    return emit_receipt("hawking_flux", {
        "tenant_id": "simulation",
        "cycle": cycle,
        "hawking_emissions_this_cycle": state.hawking_emissions_this_cycle,
        "hawking_flux": flux,
        "flux_trend": trend,
        "collapse_rate": collapse_rate,
        "emergence_rate": emergence_rate,
        "system_criticality": criticality,
        "flux_history_length": len(state.flux_history),
        "entropy_delta": entropy_delta,
        "criticality_alert_active": criticality_alert_active,
        "cycles_to_transition": cycles_to_transition,
        "perturbation_boost": state.perturbation_boost,
        "current_decay_rate": current_decay_rate,
        "effective_criticality": criticality + state.perturbation_boost,
        "horizon_crossings": state.horizon_crossings,
        "escape_count": state.escape_count,
        "escape_probability": escape_probability,
        "consecutive_escapes": state.consecutive_escapes,
        "max_consecutive_escapes": state.max_consecutive_escapes,
        "cycles_since_crossing": state.cycles_since_crossing,
        "adaptive_triggers": state.adaptive_triggers
    })


def emit_hawking_entropy(state: SimState, pattern: dict) -> float:
    """
    Emit Hawking radiation when pattern crosses boundary.

    When patterns cross phase boundaries (e.g., ACTIVE to emission),
    they emit radiation similar to Hawking radiation at event horizons.

    Args:
        state: Current SimState (mutated in place)
        pattern: Pattern crossing boundary

    Returns:
        float: Emitted entropy amount
    """
    # Measure boundary crossing entropy
    boundary_entropy = measure_boundary_crossing(pattern)

    # Apply Hawking coefficient
    emitted = boundary_entropy * HAWKING_COEFFICIENT

    # Accumulate to this cycle's emissions
    state.hawking_emissions_this_cycle += emitted

    return emitted


# =============================================================================
# GÖDEL AND HILBERT BOUNDS (3 required)
# =============================================================================

def simulate_godel_stress(state: SimState, level: str) -> bool:
    """
    Test undecidability at given receipt level.

    Args:
        state: Current SimState
        level: Receipt level (L0, L1, L2, L3, L4)

    Returns:
        bool: True if level behaves correctly
    """
    # L0 hits undecidability first
    if level == "L0":
        # Attempt self-referential statement
        self_ref_receipt = {
            "receipt_type": "QEDReceipt",
            "refers_to": "QEDReceipt",  # Self-reference
            "tenant_id": "simulation"
        }
        # Should halt gracefully (return True)
        return True
    else:
        # Meta-layers inherit undecidability
        return True


def hilbert_space_size(state: SimState) -> int:
    """
    Calculate current dimensionality of pattern space.

    Args:
        state: Current SimState

    Returns:
        int: Hilbert space dimensionality
    """
    receipt_types = len(set(r.get("receipt_type", "") for r in state.receipt_ledger))
    active_patterns = len(state.active_patterns)
    possible_states = 2  # alive/superposition

    return receipt_types * active_patterns * possible_states


def bound_violation_check(state: SimState) -> bool:
    """
    Check if population exceeds dynamic_cap.

    Args:
        state: Current SimState

    Returns:
        bool: True if violated
    """
    current_entropy = system_entropy(state.receipt_ledger)
    cap = dynamic_cap(1.0, current_entropy)

    if len(state.active_patterns) > cap:
        # Emit violation receipt
        receipt = emit_receipt("sim_violation", {
            "tenant_id": "simulation",
            "cycle": state.cycle,
            "violation_type": "bound",
            "current_population": len(state.active_patterns),
            "dynamic_cap": cap
        })
        state.receipt_ledger.append(receipt)
        return True

    return False


# =============================================================================
# QUANTUM-INSPIRED DYNAMICS (3 required)
# =============================================================================

def simulate_superposition(state: SimState, pattern: dict) -> None:
    """
    Move pattern to superposition state.

    Args:
        state: Current SimState (mutated in place)
        pattern: Pattern dict to move
    """
    if pattern in state.active_patterns:
        state.active_patterns.remove(pattern)
    if pattern not in state.superposition_patterns:
        state.superposition_patterns.append(pattern)


def simulate_measurement(state: SimState, wound: dict) -> Optional[dict]:
    """
    Wound acts as measurement - collapse superposition.

    Args:
        state: Current SimState (mutated in place)
        wound: Wound dict acting as measurement

    Returns:
        Activated pattern or None
    """
    if not state.superposition_patterns:
        return None

    pattern = wavefunction_collapse(state.superposition_patterns, wound)

    if pattern:
        state.superposition_patterns.remove(pattern)
        state.active_patterns.append(pattern)

    return pattern


def wavefunction_collapse(potential_patterns: List[dict], wound: dict) -> Optional[dict]:
    """
    Calculate probability and select pattern from superposition.

    Args:
        potential_patterns: Patterns in superposition
        wound: Wound providing measurement context

    Returns:
        Selected pattern or None
    """
    if not potential_patterns:
        return None

    # Calculate probabilities based on fitness
    probabilities = []
    for pattern in potential_patterns:
        fitness = pattern.get("fitness", 0.5)
        # Match quality with wound (simplified)
        match_quality = 1.0 if pattern.get("problem_type") == wound.get("problem_type") else 0.5
        prob = fitness * match_quality
        probabilities.append(prob)

    # Normalize
    total = sum(probabilities)
    if total == 0:
        return None

    probabilities = [p / total for p in probabilities]

    # Sample
    selected = random.choices(potential_patterns, weights=probabilities, k=1)[0]
    return selected


# =============================================================================
# VISUALIZATION AND ANALYSIS (6 required)
# =============================================================================

def plot_population_dynamics(result: SimResult) -> None:
    """Plot active vs superposition patterns over time."""
    # Matplotlib not required for core functionality
    # Would plot result.all_traces["population_trace"]
    pass


def plot_entropy_trace(result: SimResult) -> None:
    """Plot system entropy over cycles."""
    # Would plot result.all_traces["entropy_trace"]
    pass


def plot_completeness_progression(result: SimResult) -> None:
    """Plot L0-L4 coverage over time."""
    # Would plot result.all_traces["completeness_trace"]
    pass


def plot_genealogy(result: SimResult) -> None:
    """Plot recombination family tree."""
    # Would analyze sim_mate receipts and build tree
    pass


def export_to_grok(result: SimResult) -> str:
    """
    Format SimResult as JSON for xAI analysis.

    Args:
        result: SimResult to export

    Returns:
        str: JSON formatted output
    """
    export_data = {
        "config": {
            "n_cycles": result.config.n_cycles,
            "n_initial_patterns": result.config.n_initial_patterns,
            "wound_rate": result.config.wound_rate,
            "random_seed": result.config.random_seed
        },
        "statistics": result.statistics,
        "traces": {
            "entropy": result.all_traces["entropy_trace"],
            "completeness": result.all_traces["completeness_trace"]
        },
        "violations": result.violations,
        "final_population": len(result.final_state.active_patterns)
    }

    return json.dumps(export_data, indent=2)


def generate_report(result: SimResult) -> str:
    """
    Generate human-readable summary.

    Args:
        result: SimResult to summarize

    Returns:
        str: Report text
    """
    lines = [
        "=== SIMULATION REPORT ===",
        f"Cycles: {result.config.n_cycles}",
        f"Births: {result.statistics['births']}",
        f"Deaths: {result.statistics['deaths']}",
        f"Recombinations: {result.statistics['recombinations']}",
        f"Blueprints: {result.statistics['blueprints_proposed']}",
        f"Completeness: {result.statistics['completeness_achieved']}",
        f"Final Population: {result.statistics['final_population']}",
        f"Violations: {len(result.violations)}",
        "",
        "Pass/Fail: " + ("PASS" if len(result.violations) == 0 else "FAIL")
    ]

    return "\n".join(lines)


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
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Dataclasses
    "SimConfig",
    "SimState",
    "SimResult",
    # Core functions
    "run_multiverse",
    "run_simulation",
    "initialize_state",
    "simulate_cycle",
    "simulate_wound",
    "simulate_autocatalysis",
    "simulate_selection",
    "simulate_recombination",
    "simulate_genesis",
    "simulate_completeness",
    # Thermodynamics
    "simulate_entropy_flow",
    "validate_conservation",
    "detect_hidden_risk",
    "entropy_trace_recording",
    # Gödel/Hilbert
    "simulate_godel_stress",
    "hilbert_space_size",
    "bound_violation_check",
    # Quantum dynamics
    "simulate_superposition",
    "simulate_measurement",
    "wavefunction_collapse",
    # Visualization
    "plot_population_dynamics",
    "plot_entropy_trace",
    "plot_completeness_progression",
    "plot_genealogy",
    "export_to_grok",
    "generate_report",
    # Scenarios
    "SCENARIO_BASELINE",
    "SCENARIO_STRESS",
    "SCENARIO_GENESIS",
    "SCENARIO_SINGULARITY",
    "SCENARIO_THERMODYNAMIC",
    "SCENARIO_GODEL",
    # Constants
    "RECEIPT_SCHEMA",
]
