"""
sim/types_state.py - SimState and Nucleation Dataclasses

Mutable simulation state and quantum nucleation structures.
CLAUDEME v3.1 Compliant: Dataclasses for state, pure functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import PLANCK_ENTROPY_BASE


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
    phase: float  # radians, 0 to 2pi
    resonance_affinity: float  # 0 to 1, preference for resonant kicks
    direction: int  # +1 or -1
    captures: int = 0  # successful captures


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
    - ENTROPY_INCREASE dominant -> HUNTER
    - RESONANCE_TRIGGER dominant -> SHEPHERD
    - SYMMETRY_BREAK dominant -> ARCHITECT
    - No clear dominant -> HYBRID
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
    })
    parent_crystal_id: Optional[int] = None  # for replicated crystals
    generation: int = 0  # depth in family tree
    size_50_reached: bool = False  # track if crystal has exceeded size 50


# =============================================================================
# FITNESS DISTRIBUTION DATACLASS
# =============================================================================

@dataclass
class FitnessDistribution:
    """Encapsulates pattern fitness as a distribution for Thompson sampling.

    Per Grok recommendation: fitness variance IS entropy - it represents
    information about adaptive history. Inheriting variance preserves this
    information; resetting it exports entropy (increases exploration).

    Attributes:
        mean: Expected fitness value (point estimate)
        variance: Uncertainty in fitness estimate (minimum 0.001 to prevent degenerate distributions)
        n_samples: Number of fitness measurements that informed this distribution
        lineage_depth: Inheritance generations (resets to 0 on RESET mode)
    """
    mean: float = 0.5
    variance: float = 0.1
    n_samples: int = 1
    lineage_depth: int = 0

    # Variance floor to prevent degenerate distributions (Thompson sampling needs uncertainty)
    VARIANCE_FLOOR: float = 0.001

    def __post_init__(self):
        """Ensure variance floor is respected."""
        if self.variance < self.VARIANCE_FLOOR:
            object.__setattr__(self, 'variance', self.VARIANCE_FLOOR)

    def entropy(self) -> float:
        """Shannon entropy of this distribution (differential entropy for continuous).

        For a Gaussian approximation: H = 0.5 * log2(2 * pi * e * variance)

        Returns:
            Entropy in bits
        """
        import math
        # Differential entropy of Gaussian: 0.5 * ln(2*pi*e*var)
        # Convert to bits: divide by ln(2)
        if self.variance <= 0:
            return 0.0
        return 0.5 * math.log2(2 * math.pi * math.e * self.variance)


# =============================================================================
# SIMSTATE DATACLASS
# =============================================================================

@dataclass
class SimState:
    """Mutable simulation state."""
    active_patterns: List[dict] = field(default_factory=list)
    superposition_patterns: List[dict] = field(default_factory=list)
    virtual_patterns: List[dict] = field(default_factory=list)
    wound_history: List[dict] = field(default_factory=list)
    receipt_ledger: List[dict] = field(default_factory=list)
    entropy_trace: List[float] = field(default_factory=list)
    completeness_trace: List[dict] = field(default_factory=list)
    violations: List[dict] = field(default_factory=list)
    cycle: int = 0

    # Observer paradigm fields
    H_genesis: float = 0.0
    H_previous: float = 0.0
    H_boundary_this_cycle: float = 0.0
    operations_this_cycle: int = 0

    # Vacuum fluctuation fields
    vacuum_floor: float = PLANCK_ENTROPY_BASE
    hawking_emissions_this_cycle: float = 0.0

    # Per-cycle tracking for adaptive tolerance
    births_this_cycle: int = 0
    deaths_this_cycle: int = 0
    superposition_transitions_this_cycle: int = 0
    wounds_this_cycle: int = 0

    # Hawking flux tracking
    flux_history: list = field(default_factory=list)
    collapse_count_this_cycle: int = 0
    emergence_count_this_cycle: int = 0

    # Criticality monitoring
    criticality_alert_emitted: bool = False
    last_alert_cycle: int = -100
    phase_transition_occurred: bool = False
    observer_wake_count: int = 0
    previous_criticality: float = 0.0

    # Perturbation fields
    perturbation_boost: float = 0.0
    horizon_crossings: int = 0
    escape_count: int = 0
    consecutive_escapes: int = 0
    cycles_since_crossing: int = 0
    max_consecutive_escapes: int = 0
    adaptive_triggers: int = 0

    # Evolution tracking fields
    evolution_snapshots: list = field(default_factory=list)
    window_escapes: int = 0
    window_perturbations: int = 0
    last_evolution_cycle: int = 0
    window_boost_samples: list = field(default_factory=list)

    # Phase tracking fields
    last_phase: float = 0.0
    sync_count: int = 0
    cluster_count: int = 0
    symmetry_breaks: int = 0
    consecutive_same_type: int = 0
    last_receipt_type: str = ""
    last_symmetry_metric: float = 0.0

    # Resonance tracking fields
    resonance_peaks: int = 0
    resonance_hits: int = 0
    structure_formed: bool = False
    structure_formation_cycle: int = 0
    baseline_boost: float = 0.0
    baseline_shifts: int = 0
    initial_baseline: float = 0.0

    # Asymmetry and proto-form tracking
    bias_direction: int = 1
    cluster_start_cycle: int = 0
    current_cluster_duration: int = 0
    persistent_clusters: int = 0
    proto_form_count: int = 0
    proto_form_active: bool = False

    # Quantum nucleation fields
    seeds: List[Seed] = field(default_factory=list)
    beacons: List[Beacon] = field(default_factory=list)
    counselors: List[Counselor] = field(default_factory=list)
    crystals: List[Crystal] = field(default_factory=list)
    total_captures: int = 0
    crystals_formed: int = 0

    # Compound growth tracking
    max_generation: int = 0
    size_50_count: int = 0
    replication_events: int = 0
    total_branches: int = 0

    # Governance tracking
    governance_nodes: int = 0
    architect_formations: int = 0
    hunter_formations: int = 0
    hybrid_formations: int = 0

    # HYBRID differentiation tracking
    hybrid_differentiation_count: int = 0

    # Entropy surge tracking
    entropy_surge_count: int = 0
    equilibrium_score: float = 0.0

    # Kick distribution tracking
    kick_distribution: Dict[str, int] = field(default_factory=lambda: {
        "ENTROPY_INCREASE": 0, "RESONANCE_TRIGGER": 0, "SYMMETRY_BREAK": 0
    })
    first_capture_distribution: Dict[str, int] = field(default_factory=lambda: {
        "HUNTER": 0, "SHEPHERD": 0, "ARCHITECT": 0
    })

    # Quantum ARCHITECT tracking
    tunneling_events: int = 0
    entanglement_boosts: int = 0
    hunter_delays: int = 0
