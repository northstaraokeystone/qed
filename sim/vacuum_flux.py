"""
sim/vacuum_flux.py - Hawking Flux and Criticality Functions

Hawking flux computation, criticality monitoring, and phase transitions.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

from typing import Optional

from entropy import emit_receipt

from .constants import (
    HAWKING_COEFFICIENT, FLUX_WINDOW,
    CRITICALITY_ALERT_THRESHOLD, CRITICALITY_PHASE_TRANSITION,
    ALERT_COOLDOWN_CYCLES, PERTURBATION_DECAY, NONLINEAR_DECAY_FACTOR
)
from .types_state import SimState
from .measurement import measure_boundary_crossing


def compute_hawking_flux(state: SimState) -> tuple:
    """
    Compute Hawking entropy flux rate over rolling window.

    Args:
        state: Current SimState

    Returns:
        Tuple of (flux, trend)
    """
    state.flux_history.append(state.hawking_emissions_this_cycle)

    if len(state.flux_history) > FLUX_WINDOW * 2:
        state.flux_history = state.flux_history[-FLUX_WINDOW * 2:]

    if len(state.flux_history) < 2:
        return (0.0, "insufficient_data")

    if len(state.flux_history) >= FLUX_WINDOW:
        flux = (state.flux_history[-1] - state.flux_history[-FLUX_WINDOW]) / FLUX_WINDOW
    else:
        deltas = [state.flux_history[i] - state.flux_history[i-1]
                 for i in range(1, len(state.flux_history))]
        flux = sum(deltas) / len(deltas) if deltas else 0.0

    if flux > 0.01:
        trend = "increasing"
    elif flux < -0.01:
        trend = "decreasing"
    else:
        trend = "stable"

    return (flux, trend)


def compute_collapse_rate(state: SimState) -> float:
    """Compute collapse rate for this cycle."""
    return float(state.collapse_count_this_cycle)


def compute_emergence_rate(state: SimState) -> float:
    """Compute emergence rate for this cycle."""
    return float(state.emergence_count_this_cycle)


def compute_system_criticality(state: SimState, cycle: int) -> float:
    """
    Compute system criticality metric.

    Criticality = cumulative_emergences / cycle

    Args:
        state: Current SimState
        cycle: Current cycle number

    Returns:
        float: System criticality (0.0 to ~1.0)
    """
    if cycle == 0:
        return 0.0

    total_emergences = sum(1 for r in state.receipt_ledger
                          if r.get("receipt_type") == "spontaneous_emergence")
    return total_emergences / cycle


def check_criticality_alert(state: SimState, cycle: int, criticality: float) -> Optional[dict]:
    """Check if criticality alert should be emitted."""
    if (criticality > CRITICALITY_ALERT_THRESHOLD and
        not state.criticality_alert_emitted and
        (cycle - state.last_alert_cycle) > ALERT_COOLDOWN_CYCLES):

        state.criticality_alert_emitted = True
        state.last_alert_cycle = cycle

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

    if criticality < CRITICALITY_ALERT_THRESHOLD - 0.05:
        state.criticality_alert_emitted = False

    return None


def check_phase_transition(state: SimState, cycle: int, criticality: float, H_end: float) -> Optional[dict]:
    """Check if phase transition has occurred."""
    if criticality >= CRITICALITY_PHASE_TRANSITION and not state.phase_transition_occurred:
        state.phase_transition_occurred = True

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
    """Estimate cycles until criticality reaches 1.0."""
    if criticality_rate <= 0:
        return -1

    remaining = CRITICALITY_PHASE_TRANSITION - criticality
    cycles = int(remaining / criticality_rate)
    return max(cycles, 0)


def emit_hawking_entropy(state: SimState, pattern: dict) -> float:
    """
    Emit Hawking radiation when pattern crosses boundary.

    Args:
        state: Current SimState (mutated in place)
        pattern: Pattern crossing boundary

    Returns:
        float: Emitted entropy amount
    """
    boundary_entropy = measure_boundary_crossing(pattern)
    emitted = boundary_entropy * HAWKING_COEFFICIENT
    state.hawking_emissions_this_cycle += emitted
    return emitted


def emit_hawking_flux_receipt(state: SimState, cycle: int, flux: float,
                              trend: str, collapse_rate: float,
                              emergence_rate: float, criticality: float,
                              entropy_delta: float, criticality_rate: float) -> dict:
    """Emit hawking_flux receipt with rate metrics."""
    criticality_alert_active = criticality > 0.95
    cycles_to_transition = estimate_cycles_to_transition(criticality, criticality_rate)
    escape_probability = state.escape_count / max(cycle, 1)
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
