"""
sim/perturbation.py - Perturbation and Phase Functions

Stochastic perturbation dynamics, phase synchronization, and resonance.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import math
import random
from typing import Optional

from entropy import emit_receipt

from .constants import (
    PERTURBATION_PROBABILITY, PERTURBATION_MAGNITUDE, PERTURBATION_DECAY,
    PERTURBATION_VARIANCE, BASIN_ESCAPE_THRESHOLD, CLUSTER_LAMBDA,
    MAX_CLUSTER_SIZE, NONLINEAR_DECAY_FACTOR, ASYMMETRY_BIAS,
    CLUSTER_PERSISTENCE_THRESHOLD, SYMMETRY_BREAK_THRESHOLD,
    EVOLUTION_WINDOW, MAX_MAGNITUDE_FACTOR, ADAPTIVE_THRESHOLD,
    SYNC_BOOST, MAX_PROBABILITY, PHASE_SYNC_PROBABILITY, PHASE_SYNC_WINDOW,
    SYMMETRY_SAMPLE_SIZE, RESONANCE_PROBABILITY, INTERFERENCE_AMPLITUDE,
    RESONANCE_PEAK_THRESHOLD, MAX_RESONANCE_AMPLIFICATION
)
from .state import SimState
from .vacuum import vacuum_fluctuation


def poisson_manual(lam: float) -> int:
    """Generate Poisson-distributed random variable using Knuth's algorithm."""
    L = math.exp(-lam)
    k = 0
    p = 1.0

    while p > L:
        k += 1
        p *= random.random()

    return k - 1


def generate_phase(state: SimState, synced: bool) -> float:
    """Generate independent random phase for perturbation kick."""
    return random.uniform(0, 2 * math.pi)


def check_phase_sync(state: SimState) -> tuple:
    """Check if current kick should sync with previous phase."""
    if random.random() < PHASE_SYNC_PROBABILITY:
        phase = generate_phase(state, synced=True)
    else:
        phase = generate_phase(state, synced=False)

    phase_diff = abs(phase - state.last_phase)
    phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
    synced = phase_diff < PHASE_SYNC_WINDOW

    return (synced, phase)


def compute_interference(phase: float, last_phase: float) -> float:
    """Compute wave interference factor from phase difference."""
    phase_diff = phase - last_phase
    return math.cos(phase_diff)


def compute_effective_probability(state: SimState, synced: bool) -> float:
    """Compute effective perturbation probability with adaptive feedback."""
    base_prob = PERTURBATION_PROBABILITY

    if synced:
        effective_prob = base_prob + SYNC_BOOST
    else:
        effective_prob = base_prob

    if state.perturbation_boost > ADAPTIVE_THRESHOLD:
        effective_prob = effective_prob + SYNC_BOOST

    return min(effective_prob, MAX_PROBABILITY)


def apply_asymmetry_bias(magnitude: float, state: SimState) -> float:
    """Apply asymmetry bias to perturbation magnitude."""
    if random.random() < 0.1:
        state.bias_direction *= -1

    return magnitude * (1 + ASYMMETRY_BIAS * state.bias_direction)


def check_resonance(state: SimState) -> tuple:
    """Check if perturbation kick resonates."""
    if random.random() < RESONANCE_PROBABILITY:
        resonance_factor = min(1.0 + INTERFERENCE_AMPLITUDE, MAX_RESONANCE_AMPLIFICATION)
        return (True, resonance_factor)
    return (False, 1.0)


def check_perturbation(state: SimState, cycle: int) -> Optional[dict]:
    """
    Stochastic GW kick with phase interference, clusters, and adaptive feedback.

    Args:
        state: Current SimState (mutated in place)
        cycle: Current cycle number

    Returns:
        Receipt dict if perturbation fired, None otherwise
    """
    decay_rate = PERTURBATION_DECAY * (1 + NONLINEAR_DECAY_FACTOR * state.perturbation_boost)
    state.perturbation_boost *= (1 - decay_rate)
    state.perturbation_boost = max(0.0, state.perturbation_boost)

    synced, phase = check_phase_sync(state)
    interference = compute_interference(phase, state.last_phase)
    resonance_hit, resonance_factor = check_resonance(state)

    if resonance_hit:
        state.resonance_hits += 1

    effective_prob = compute_effective_probability(state, synced)
    adaptive_active = state.perturbation_boost > ADAPTIVE_THRESHOLD

    if adaptive_active:
        state.adaptive_triggers += 1

    if random.random() < effective_prob:
        cluster_size = min(poisson_manual(CLUSTER_LAMBDA), MAX_CLUSTER_SIZE)
        quantum_factor = vacuum_fluctuation() / 0.001
        base_variance = random.gauss(0, PERTURBATION_VARIANCE)
        variance_factor = 1.0 + base_variance * quantum_factor
        variance_factor_uncapped = variance_factor
        variance_factor = max(0.1, min(MAX_MAGNITUDE_FACTOR, variance_factor))
        capped = (variance_factor_uncapped != variance_factor)

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

        state.last_phase = phase
        if synced:
            state.sync_count += 1

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
    """Check if system is escaping attractor basin."""
    if state.perturbation_boost > BASIN_ESCAPE_THRESHOLD:
        state.consecutive_escapes += 1
        state.escape_count += 1

        if state.consecutive_escapes > state.max_consecutive_escapes:
            state.max_consecutive_escapes = state.consecutive_escapes

        escape_probability = state.escape_count / max(cycle, 1)
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
        state.consecutive_escapes = 0
    return None


def check_resonance_peak(state: SimState, cycle: int) -> Optional[dict]:
    """Check if perturbation boost has reached resonance peak threshold."""
    if state.perturbation_boost > RESONANCE_PEAK_THRESHOLD:
        state.resonance_peaks += 1
        return {
            "receipt_type": "resonance_peak",
            "cycle": cycle,
            "boost": state.perturbation_boost,
            "peak_count": state.resonance_peaks
        }
    return None


def check_structure_formation(state: SimState, cycle: int) -> Optional[dict]:
    """Check if structure has formed via persistent clusters and symmetry breaks."""
    if (state.persistent_clusters > 0 and
        state.symmetry_breaks >= SYMMETRY_BREAK_THRESHOLD and
        state.proto_form_count > 0 and
        not state.structure_formed):

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
    """Track baseline shift via exponential moving average."""
    if cycle % EVOLUTION_WINDOW != 0:
        return None

    state.baseline_boost = 0.9 * state.baseline_boost + 0.1 * state.perturbation_boost

    if cycle == EVOLUTION_WINDOW:
        state.initial_baseline = state.baseline_boost

    if abs(state.baseline_boost - state.initial_baseline) > 0.05:
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
    """Check if evolution window has elapsed and emit evolution receipt."""
    if cycle - state.last_evolution_cycle >= EVOLUTION_WINDOW:
        escape_rate = state.window_escapes / EVOLUTION_WINDOW
        perturbation_rate = state.window_perturbations / EVOLUTION_WINDOW

        if state.window_boost_samples:
            avg_boost = sum(state.window_boost_samples) / len(state.window_boost_samples)
        else:
            avg_boost = 0.0

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

        state.window_escapes = 0
        state.window_perturbations = 0
        state.window_boost_samples = []
        state.last_evolution_cycle = cycle
        state.evolution_snapshots.append(receipt)

        return receipt
    return None


def check_cluster_persistence(state: SimState, receipt_type: str, cycle: int) -> Optional[dict]:
    """Check for persistent clusters."""
    if receipt_type == state.last_receipt_type:
        state.consecutive_same_type += 1
        state.current_cluster_duration = cycle - state.cluster_start_cycle
    else:
        persistent_receipt = None
        if state.current_cluster_duration >= CLUSTER_PERSISTENCE_THRESHOLD:
            state.persistent_clusters += 1
            persistent_receipt = emit_receipt("persistent_cluster", {
                "tenant_id": "simulation",
                "cycle": cycle,
                "receipt_type": state.last_receipt_type,
                "cluster_duration": state.current_cluster_duration,
                "persistent_cluster_number": state.persistent_clusters
            })

        state.cluster_start_cycle = cycle
        state.consecutive_same_type = 1
        state.last_receipt_type = receipt_type
        state.current_cluster_duration = 0

        return persistent_receipt
    return None


def check_proto_form(state: SimState, cycle: int) -> Optional[dict]:
    """Check for proto-form detection."""
    if cycle % EVOLUTION_WINDOW != 0:
        return None

    if state.persistent_clusters > 0 and state.symmetry_breaks >= SYMMETRY_BREAK_THRESHOLD:
        if not state.proto_form_active:
            state.proto_form_active = True
            state.proto_form_count += 1

            return emit_receipt("proto_form", {
                "tenant_id": "simulation",
                "cycle": cycle,
                "persistent_clusters": state.persistent_clusters,
                "symmetry_breaks": state.symmetry_breaks,
                "proto_form_number": state.proto_form_count
            })
    else:
        state.proto_form_active = False

    return None


def check_symmetry_break(state: SimState, receipts: list, cycle: int) -> Optional[dict]:
    """Check for symmetry break via entropy delta."""
    if cycle % EVOLUTION_WINDOW != 0:
        return None

    sample = receipts[-SYMMETRY_SAMPLE_SIZE:] if len(receipts) >= SYMMETRY_SAMPLE_SIZE else receipts

    if not sample:
        return None

    type_counts = {}
    for receipt in sample:
        receipt_type = receipt.get("receipt_type", "unknown")
        type_counts[receipt_type] = type_counts.get(receipt_type, 0) + 1

    total = len(sample)
    entropy = 0.0
    for count in type_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    if abs(entropy - state.last_symmetry_metric) > 0.1:
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
        state.last_symmetry_metric = entropy

    return None
