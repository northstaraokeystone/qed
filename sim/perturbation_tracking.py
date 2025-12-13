"""
sim/perturbation_tracking.py - Evolution and Cluster Tracking

Evolution windows, cluster persistence, symmetry breaks, and structure formation.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import math
from typing import Optional

from entropy import emit_receipt

from .constants import (
    CLUSTER_PERSISTENCE_THRESHOLD, SYMMETRY_BREAK_THRESHOLD,
    EVOLUTION_WINDOW, SYMMETRY_SAMPLE_SIZE
)
from .types_state import SimState


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
