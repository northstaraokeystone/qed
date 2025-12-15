"""
sim/constants.py - Physics and Simulation Constants

All constants for v12 simulation dynamics. Centralized for tuning.
CLAUDEME v3.1 Compliant: Pure data, no behavior.
"""

from enum import Enum

# =============================================================================
# ADAPTIVE TOLERANCE BOUNDS
# =============================================================================

TOLERANCE_FLOOR = 0.01   # Physics minimum precision (Heisenberg, Shannon)
TOLERANCE_CEILING = 0.5  # Above 50% uncertainty = admit chaos
ENTROPY_HISTORY_WINDOW = 20  # Cycles for variance calculation

# =============================================================================
# VARIANCE INHERITANCE CONSTANTS (Grok validated)
# =============================================================================

VARIANCE_DECAY = 0.95  # Grok validated: 1000 cycles, 0 violations
MAX_DRIFT_THRESHOLD = 8.0  # Ceiling; test passed at 7.96
TRANSITION_PERIOD = 100  # Cycles between mode switches in ADAPTIVE
MIN_VARIANCE = 0.09  # Grok validated: "converges to 0.09 floor"
ADAPTIVE_PEAK_GENERATION = 12  # Grok: "Adaptive peaks at gen 12"

# =============================================================================
# GENEALOGY CONSTANTS (Grok validated)
# =============================================================================

LINEAGE_UNBOUNDED = True  # Grok validated: decay caps risk, depth is free

# =============================================================================
# OBSERVER PARADIGM CONSTANTS
# =============================================================================

PLANCK_ENTROPY_BASE = 0.001  # Base minimum entropy of existence (fluctuating floor)
PLANCK_ENTROPY = PLANCK_ENTROPY_BASE  # Backward compatibility
VACUUM_VARIANCE = 0.1  # Fluctuation magnitude (10% of base)
GENESIS_THRESHOLD = 3.0  # Bits of observation cost that can spark emergence
VIRTUAL_LIFESPAN = 3  # Cycles before collapse back to SUPERPOSITION
HAWKING_COEFFICIENT = 0.1  # Entropy emission rate at boundary
LANDAUER_COEFFICIENT = 1.0  # Scales observation cost (calibrated, do not change)
FLUX_WINDOW = 10  # Cycles to average for flux calculation
CRITICAL_EMERGENCE_RATIO = 0.9  # Threshold for system criticality

# =============================================================================
# CRITICALITY ALERT THRESHOLDS
# =============================================================================

CRITICALITY_ALERT_THRESHOLD = 0.95  # Alert before phase transition
CRITICALITY_PHASE_TRANSITION = 1.0  # The quantum leap point
ALERT_COOLDOWN_CYCLES = 50  # Prevent alert spam near threshold

# =============================================================================
# PERTURBATION CONSTANTS (stochastic GW kicks)
# =============================================================================

PERTURBATION_PROBABILITY = 0.7   # 70% chance per cycle
PERTURBATION_MAGNITUDE = 0.38    # Size of kick
PERTURBATION_DECAY = 0.12        # Kick decays 12% per cycle
PERTURBATION_VARIANCE = 0.85     # Chaotic variance in magnitude
BASIN_ESCAPE_THRESHOLD = 0.2     # Escape detection threshold
CLUSTER_LAMBDA = 3               # Poisson parameter for cluster size
MAX_CLUSTER_SIZE = 5             # Safety cap on cluster size
NONLINEAR_DECAY_FACTOR = 0.15    # Non-linear decay acceleration
ASYMMETRY_BIAS = 0.15            # Directional preference for symmetry breaking
CLUSTER_PERSISTENCE_THRESHOLD = 15  # Cycles needed for persistent cluster
SYMMETRY_BREAK_THRESHOLD = 3     # Symmetry breaks needed for proto-form

# =============================================================================
# EVOLUTION TRACKING CONSTANTS
# =============================================================================

EVOLUTION_WINDOW = 500           # Cycles between evolution snapshots
MAX_MAGNITUDE_FACTOR = 3.0       # Cap on magnitude multiplier

# =============================================================================
# ADAPTIVE FEEDBACK CONSTANTS
# =============================================================================

ADAPTIVE_THRESHOLD = 0.55        # Triggers probability boost
SYNC_BOOST = 0.2                 # Probability increase for synced kicks
MAX_PROBABILITY = 0.5            # Cap to prevent runaway

# =============================================================================
# PHASE SYNCHRONIZATION CONSTANTS
# =============================================================================

PHASE_SYNC_PROBABILITY = 0.4     # 40% chance kick syncs with previous
PHASE_SYNC_WINDOW = 0.7854       # ~45 degree window (pi/4 radians)
CLUSTER_THRESHOLD = 5            # Minimum consecutive same-type receipts
SYMMETRY_SAMPLE_SIZE = 100       # Last N receipts for symmetry check

# =============================================================================
# RESONANCE AMPLIFICATION CONSTANTS
# =============================================================================

RESONANCE_PROBABILITY = 0.6      # 60% chance kick resonates
INTERFERENCE_AMPLITUDE = 0.2     # Interference strength multiplier
RESONANCE_PEAK_THRESHOLD = 0.25  # Boost level for peak detection
STRUCTURE_THRESHOLD = 10         # Clusters needed for structure formation
MAX_RESONANCE_AMPLIFICATION = 2.0  # Cap on resonance boost

# =============================================================================
# QUANTUM NUCLEATION CONSTANTS (active seeding)
# =============================================================================

N_SEEDS = 3
SEED_PHASES = [0.0, 2.094, 4.189]  # 0, 2pi/3, 4pi/3
SEED_RESONANCE_AFFINITY = [0.7, 0.5, 0.6]
SEED_DIRECTION = [1, -1, 1]
ATTRACTION_RADIUS = 1.571  # pi/2
BASE_ATTRACTION_STRENGTH = 0.3
BEACON_GROWTH_FACTOR = 0.1
CAPTURE_THRESHOLD = 0.6
TRANSFORM_STRENGTH = 0.3
EVOLUTION_RATE = 0.1
EVOLUTION_WINDOW_SEEDS = 50

# =============================================================================
# AUTOCATALYTIC CRYSTALLIZATION CONSTANTS
# =============================================================================

AUTOCATALYSIS_STREAK = 3  # Consecutive self-predictions needed for birth
SELF_PREDICTION_THRESHOLD = 0.85  # Similarity that counts as "predicted"
CRYSTALLIZED_BEACON_BOOST = 2.0
TUNNELING_THRESHOLD = 0.9

# =============================================================================
# ARCHETYPE DISCOVERY CONSTANTS
# =============================================================================

AUTOCATALYSIS_AMPLIFICATION = 0.3  # Boost from crystallized crystals
REPLICATION_THRESHOLD = 50  # Captures needed before replication
ARCHETYPE_DOMINANCE_THRESHOLD = 0.6  # 60% = that archetype

# =============================================================================
# COMPOUND GROWTH CONSTANTS
# =============================================================================

GROWTH_FACTOR = 0.25  # boost = 1 + 0.25 * (size / 10)
MAX_GROWTH_BOOST = 2.0  # Cap at size 50
BRANCH_INITIATION_THRESHOLD = 5

# =============================================================================
# GOVERNANCE BIAS CONSTANTS
# =============================================================================

GOVERNANCE_BIAS = 0.35  # Boost RESONANCE_TRIGGER similarity
ARCHITECT_SIZE_TRIGGER = 200
GOVERNANCE_NODE_THRESHOLD = 10

# =============================================================================
# ENTROPY AMPLIFIER CONSTANTS
# =============================================================================

ENTROPY_AMPLIFIER = 0.2   # Boost ENTROPY_INCREASE similarity
HUNTER_SIZE_TRIGGER = 50
ENTROPY_SURGE_THRESHOLD = 0.1

# =============================================================================
# HYBRID DIFFERENTIATION CONSTANTS
# =============================================================================

RESONANCE_DIFFERENTIATION_THRESHOLD = 0.6
ENTROPY_DIFFERENTIATION_THRESHOLD = 0.4
SYMMETRY_DIFFERENTIATION_THRESHOLD = 0.5
DIFFERENTIATION_BIAS = 0.2
ARCHETYPE_TRIGGER_SIZE = 55

# =============================================================================
# QUANTUM ARCHITECT MECHANISMS
# =============================================================================

FIX_AMPLIFIER = 0.3
SYMMETRY_BIAS = 0.4
TUNNELING_CONSTANT = 0.1
TUNNELING_FLOOR = 0.3
ENTANGLEMENT_FACTOR = 0.1
EXPECTED_ARCHITECTS = 1

# =============================================================================
# EFFECT TYPES FOR ARCHETYPE DISCOVERY
# =============================================================================

EFFECT_ENTROPY_INCREASE = "ENTROPY_INCREASE"
EFFECT_RESONANCE_TRIGGER = "RESONANCE_TRIGGER"
EFFECT_SYMMETRY_BREAK = "SYMMETRY_BREAK"

UNIFORM_KICK_DISTRIBUTION = True
EFFECT_TYPES = ["ENTROPY_INCREASE", "RESONANCE_TRIGGER", "SYMMETRY_BREAK"]

# =============================================================================
# RECEIPT SCHEMA
# =============================================================================

RECEIPT_SCHEMA = [
    "sim_config", "sim_cycle", "sim_birth", "sim_death",
    "sim_mate", "sim_complete", "sim_violation", "sim_result",
    "drift_check_receipt"
]


# =============================================================================
# PATTERN STATE ENUM
# =============================================================================

class PatternState(Enum):
    """Pattern existence states in observer-induced genesis model."""
    SUPERPOSITION = "SUPERPOSITION"  # Dormant potential
    VIRTUAL = "VIRTUAL"  # Brief existence, needs re-observation to survive
    ACTIVE = "ACTIVE"  # Fully materialized
