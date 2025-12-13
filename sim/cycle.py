"""
sim/cycle.py - Core Simulation Loop

Main simulation entry points: simulate_cycle, run_simulation, run_multiverse.
CLAUDEME v3.1 Compliant: Pure functions with receipts.
"""

import random
from typing import List

from entropy import emit_receipt
from receipt_completeness import level_coverage
from autoimmune import GERMLINE_PATTERNS

from .constants import PatternState, ENTROPY_SURGE_THRESHOLD
from .types_config import SimConfig
from .types_state import SimState
from .types_result import SimResult
from .measurement import measure_state, measure_observation_cost, measure_boundary_crossing, measure_genesis
from .vacuum_fluctuation import (
    vacuum_fluctuation, attempt_spontaneous_emergence, process_virtual_patterns
)
from .vacuum_flux import (
    compute_hawking_flux, compute_collapse_rate, compute_emergence_rate,
    compute_system_criticality, check_criticality_alert, check_phase_transition,
    emit_hawking_flux_receipt
)
from .perturbation_core import check_perturbation, check_basin_escape, check_resonance_peak
from .perturbation_tracking import (
    track_baseline_shift, check_evolution_window, check_cluster_persistence,
    check_proto_form, check_symmetry_break, check_structure_formation
)
from .nucleation_seeds import initialize_nucleation, counselor_compete, counselor_capture
from .nucleation_crystals import check_crystallization
from .nucleation_evolution import check_replication, check_hybrid_differentiation, evolve_seeds
from .dynamics_lifecycle import simulate_autocatalysis, simulate_selection
from .dynamics_genesis import (
    simulate_wound, simulate_recombination, simulate_genesis, simulate_completeness
)
from .validation import validate_conservation, emit_entropy_state_receipt


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

    self_patterns = []
    for origin in list(GERMLINE_PATTERNS)[:3]:
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
        pattern["receipts"].append({
            "receipt_type": "agent_decision",
            "pattern_id": pattern["pattern_id"],
            "ts": "2025-01-01T00:00:00Z"
        })
        self_patterns.append(pattern)

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
        if random.random() > 0.5:
            pattern["receipts"].append({
                "receipt_type": "agent_decision",
                "pattern_id": pattern["pattern_id"],
                "ts": "2025-01-01T00:00:00Z"
            })
        self_patterns.append(pattern)

    state.active_patterns = self_patterns

    H_genesis = measure_genesis(self_patterns)
    state.H_genesis = H_genesis
    state.H_previous = H_genesis

    initialize_nucleation(state)

    return state


def simulate_cycle(state: SimState, config: SimConfig) -> List[dict]:
    """
    One iteration of v12 unified loop (observer paradigm).

    Args:
        state: Current SimState (mutated in place)
        config: SimConfig with parameters

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

    # CYCLE START: Vacuum fluctuation
    state.vacuum_floor = vacuum_fluctuation()
    state.hawking_emissions_this_cycle = 0.0

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
        "fluctuation_delta": state.vacuum_floor - 0.001,
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

    # Perturbation check
    perturbation_receipt = check_perturbation(state, state.cycle)
    if perturbation_receipt:
        state.receipt_ledger.append(perturbation_receipt)
        state.window_perturbations += 1

        persistent_cluster_receipt = check_cluster_persistence(
            state, perturbation_receipt["receipt_type"], state.cycle
        )
        if persistent_cluster_receipt:
            state.receipt_ledger.append(persistent_cluster_receipt)

        # Quantum nucleation: counselors compete
        kick_phase = perturbation_receipt.get("phase", 0.0)
        kick_resonant = perturbation_receipt.get("resonance_hit", False)
        kick_direction = perturbation_receipt.get("bias_direction", 1)

        winner = counselor_compete(state, perturbation_receipt, kick_phase, kick_resonant, kick_direction)
        if winner:
            seed_id, similarity = winner
            capture_receipt = counselor_capture(state, seed_id, perturbation_receipt, similarity, state.cycle)
            state.receipt_ledger.append(capture_receipt)

            crystallization_receipt = check_crystallization(state, state.cycle)
            if crystallization_receipt:
                state.receipt_ledger.append(crystallization_receipt)

            replication_receipt = check_replication(state, state.cycle)
            if replication_receipt:
                state.receipt_ledger.append(replication_receipt)

            differentiation_receipt = check_hybrid_differentiation(state, state.cycle)
            if differentiation_receipt:
                state.receipt_ledger.append(differentiation_receipt)

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

    # Effective criticality
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
    current_entropy = H_end
    state.entropy_trace.append(current_entropy)
    coverage = level_coverage(state.receipt_ledger)
    state.completeness_trace.append(coverage)

    # Track entropy surges
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


def run_simulation(config: SimConfig) -> SimResult:
    """
    Run complete simulation.

    Args:
        config: SimConfig with parameters

    Returns:
        SimResult with final state, traces, violations, and statistics
    """
    state = initialize_state(config)
    all_violations = []

    for cycle in range(config.n_cycles):
        state.cycle = cycle
        violations = simulate_cycle(state, config)
        all_violations.extend(violations)

    # Compute statistics
    births = sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_birth")
    deaths = sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "sim_death")
    recombinations = sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "recombination")
    blueprints = sum(1 for r in state.receipt_ledger if r.get("receipt_type") == "genesis_approved")
    completeness = any(r.get("receipt_type") == "sim_complete" for r in state.receipt_ledger)

    statistics = {
        "births": births,
        "deaths": deaths,
        "recombinations": recombinations,
        "blueprints_proposed": blueprints,
        "completeness_achieved": completeness,
        "final_population": len(state.active_patterns)
    }

    all_traces = {
        "entropy_trace": state.entropy_trace,
        "completeness_trace": state.completeness_trace
    }

    return SimResult(
        final_state=state,
        all_traces=all_traces,
        violations=all_violations,
        statistics=statistics,
        config=config
    )


def run_multiverse(configs: List[SimConfig]) -> List[SimResult]:
    """
    Run multiple simulations in sequence.

    Args:
        configs: List of SimConfig objects

    Returns:
        List of SimResult objects
    """
    results = []
    for config in configs:
        result = run_simulation(config)
        results.append(result)
    return results
