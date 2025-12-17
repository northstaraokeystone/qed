"""
timeline.py - Sovereignty Timeline with Mitigation Support

Computes sovereignty timeline with optional mitigation stack integration.

CLAUDEME v3.1 Compliant: Accepts MitigationResult for tau/alpha computation.

Source: Grok insight 2025-01
- "Mars at 2.5 cycles to 10^3, beating unmitigated by 2x"
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from receipts import emit_receipt
from mitigation import (
    MitigationResult,
    MitigationConfig,
    stack_mitigation,
    DEFAULT_BASE_TAU,
    DEFAULT_BASE_ALPHA,
    CYCLES_TO_10K_TARGET,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Timeline parameters
SOVEREIGNTY_TARGET = 1000          # Target sovereignty units (10^3)
DEFAULT_INITIAL_SOVEREIGNTY = 1.0  # Starting sovereignty
GROWTH_RATE_FACTOR = 0.5           # Growth rate scaling

# Tau penalty constants
TAU_PENALTY_SCALE = 0.001          # Penalty per second of tau

# Module exports for receipt types
RECEIPT_SCHEMA = ["sovereignty_timeline", "tau_penalty"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimelinePoint:
    """A point on the sovereignty timeline."""
    cycle: int
    sovereignty: float
    eff_tau: float
    eff_alpha: float
    tau_penalty: float
    growth_rate: float


@dataclass
class TimelineResult:
    """Result from sovereignty timeline computation."""
    points: List[TimelinePoint]
    cycles_to_target: float
    final_sovereignty: float
    mitigation_applied: bool
    retention_factor: float


# =============================================================================
# RECEIPT TYPE 1: sovereignty_timeline
# =============================================================================

# --- SCHEMA ---
SOVEREIGNTY_TIMELINE_SCHEMA = {
    "receipt_type": "sovereignty_timeline",
    "ts": "ISO8601",
    "tenant_id": "str",
    "cycles_computed": "int",
    "cycles_to_target": "float",
    "final_sovereignty": "float",
    "mitigation_applied": "bool",
    "retention_factor": "float",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_sovereignty_timeline_receipt(
    tenant_id: str,
    cycles_computed: int,
    cycles_to_target: float,
    final_sovereignty: float,
    mitigation_applied: bool,
    retention_factor: float
) -> dict:
    """Emit sovereignty_timeline receipt."""
    return emit_receipt("sovereignty_timeline", {
        "tenant_id": tenant_id,
        "cycles_computed": cycles_computed,
        "cycles_to_target": cycles_to_target,
        "final_sovereignty": final_sovereignty,
        "mitigation_applied": mitigation_applied,
        "retention_factor": retention_factor
    })


# =============================================================================
# RECEIPT TYPE 2: tau_penalty
# =============================================================================

# --- SCHEMA ---
TAU_PENALTY_SCHEMA = {
    "receipt_type": "tau_penalty",
    "ts": "ISO8601",
    "tenant_id": "str",
    "eff_tau": "float",
    "penalty": "float",
    "payload_hash": "str (SHA256:BLAKE3)"
}


# --- EMIT ---
def emit_tau_penalty_receipt(
    tenant_id: str,
    eff_tau: float,
    penalty: float
) -> dict:
    """Emit tau_penalty receipt."""
    return emit_receipt("tau_penalty", {
        "tenant_id": tenant_id,
        "eff_tau": eff_tau,
        "penalty": penalty
    })


# =============================================================================
# CORE FUNCTION 1: compute_tau_penalty
# =============================================================================

def compute_tau_penalty(
    eff_tau: float,
    base_tau: float = DEFAULT_BASE_TAU
) -> float:
    """
    Compute tau penalty for sovereignty growth.

    Higher tau = higher penalty = slower growth.

    Args:
        eff_tau: Effective tau after mitigation
        base_tau: Base tau for normalization

    Returns:
        float: Penalty factor (0 = no penalty, 1 = full penalty)
    """
    # Penalty scales with effective tau
    # At base_tau, penalty is ~50%
    # At eff_tau=468s (full stack), penalty is ~20%
    penalty = (eff_tau / base_tau) * 0.5

    return min(1.0, max(0.0, penalty))


# =============================================================================
# CORE FUNCTION 2: compute_growth_rate
# =============================================================================

def compute_growth_rate(
    eff_alpha: float,
    tau_penalty: float
) -> float:
    """
    Compute sovereignty growth rate per cycle.

    Args:
        eff_alpha: Effective alpha after mitigation
        tau_penalty: Tau penalty factor

    Returns:
        float: Growth rate multiplier per cycle
    """
    # Growth = alpha * rate_factor * (1 - penalty)
    return eff_alpha * GROWTH_RATE_FACTOR * (1 - tau_penalty)


# =============================================================================
# CORE FUNCTION 3: sovereignty_timeline
# =============================================================================

def sovereignty_timeline(
    n_cycles: int,
    mitigation: Optional[MitigationResult] = None,
    initial_sovereignty: float = DEFAULT_INITIAL_SOVEREIGNTY,
    base_tau: float = DEFAULT_BASE_TAU,
    base_alpha: float = DEFAULT_BASE_ALPHA,
    tenant_id: str = "axiom-autonomy"
) -> TimelineResult:
    """
    Compute sovereignty timeline with optional mitigation.

    When mitigation provided:
    - Uses mitigation.eff_tau in tau_penalty()
    - Uses mitigation.eff_alpha directly
    - Tracks mitigation.retention_factor in output

    Args:
        n_cycles: Number of cycles to compute
        mitigation: Pre-computed mitigation stack (optional)
        initial_sovereignty: Starting sovereignty value
        base_tau: Base tau in seconds (used if no mitigation)
        base_alpha: Base alpha (used if no mitigation)
        tenant_id: Tenant identifier

    Returns:
        TimelineResult with points and summary
    """
    # Determine effective values
    if mitigation is not None:
        eff_tau = mitigation.eff_tau
        eff_alpha = mitigation.eff_alpha
        retention_factor = mitigation.retention_factor
        mitigation_applied = True
    else:
        eff_tau = base_tau
        eff_alpha = base_alpha
        retention_factor = 1.0
        mitigation_applied = False

    # Compute tau penalty
    tau_penalty = compute_tau_penalty(eff_tau, base_tau)

    # Emit tau penalty receipt
    emit_tau_penalty_receipt(
        tenant_id=tenant_id,
        eff_tau=eff_tau,
        penalty=tau_penalty
    )

    # Compute growth rate
    growth_rate = compute_growth_rate(eff_alpha, tau_penalty)

    # Generate timeline points
    points = []
    sovereignty = initial_sovereignty
    cycles_to_target = float('inf')

    for cycle in range(n_cycles):
        point = TimelinePoint(
            cycle=cycle,
            sovereignty=sovereignty,
            eff_tau=eff_tau,
            eff_alpha=eff_alpha,
            tau_penalty=tau_penalty,
            growth_rate=growth_rate
        )
        points.append(point)

        # Check if target reached
        if sovereignty >= SOVEREIGNTY_TARGET and cycles_to_target == float('inf'):
            cycles_to_target = cycle

        # Apply growth for next cycle
        sovereignty = sovereignty * (1 + growth_rate)

    # Compute final cycles to target if not reached
    if cycles_to_target == float('inf') and growth_rate > 0:
        # Extrapolate: S0 * (1 + r)^n = target
        # n = log(target/S0) / log(1+r)
        cycles_to_target = math.log(SOVEREIGNTY_TARGET / initial_sovereignty) / math.log(1 + growth_rate)

    final_sovereignty = sovereignty

    # Emit timeline receipt
    emit_sovereignty_timeline_receipt(
        tenant_id=tenant_id,
        cycles_computed=n_cycles,
        cycles_to_target=cycles_to_target,
        final_sovereignty=final_sovereignty,
        mitigation_applied=mitigation_applied,
        retention_factor=retention_factor
    )

    return TimelineResult(
        points=points,
        cycles_to_target=cycles_to_target,
        final_sovereignty=final_sovereignty,
        mitigation_applied=mitigation_applied,
        retention_factor=retention_factor
    )


# =============================================================================
# CORE FUNCTION 4: compare_timelines
# =============================================================================

def compare_timelines(
    n_cycles: int,
    base_tau: float = DEFAULT_BASE_TAU,
    tenant_id: str = "axiom-autonomy"
) -> dict:
    """
    Compare mitigated vs unmitigated timelines.

    Args:
        n_cycles: Number of cycles to compute
        base_tau: Base tau in seconds
        tenant_id: Tenant identifier

    Returns:
        dict with 'unmitigated' and 'mitigated' TimelineResults
    """
    # Unmitigated timeline
    unmitigated = sovereignty_timeline(
        n_cycles=n_cycles,
        mitigation=None,
        base_tau=base_tau,
        tenant_id=tenant_id
    )

    # Mitigated timeline (full stack)
    config = MitigationConfig()
    mitigation = stack_mitigation(base_tau, config, tenant_id=tenant_id)

    mitigated = sovereignty_timeline(
        n_cycles=n_cycles,
        mitigation=mitigation,
        base_tau=base_tau,
        tenant_id=tenant_id
    )

    # Compute acceleration ratio
    if unmitigated.cycles_to_target > 0:
        acceleration = unmitigated.cycles_to_target / mitigated.cycles_to_target
    else:
        acceleration = 1.0

    return {
        "unmitigated": unmitigated,
        "mitigated": mitigated,
        "acceleration": acceleration
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "SOVEREIGNTY_TARGET",
    "DEFAULT_INITIAL_SOVEREIGNTY",
    "GROWTH_RATE_FACTOR",
    "TAU_PENALTY_SCALE",
    # Data classes
    "TimelinePoint",
    "TimelineResult",
    # Core functions
    "compute_tau_penalty",
    "compute_growth_rate",
    "sovereignty_timeline",
    "compare_timelines",
    # Receipt functions
    "emit_sovereignty_timeline_receipt",
    "emit_tau_penalty_receipt",
]
