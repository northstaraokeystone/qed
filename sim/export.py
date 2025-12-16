"""
sim/export.py - Visualization and Export Functions

Visualization and Grok export utilities.
CLAUDEME v3.1 Compliant: Pure functions.
"""

import json
from typing import Optional

from entropy import emit_receipt, dual_hash
from .types_result import SimResult
from .types_config import MANDATORY_SCENARIOS
from .constants import (
    DOMAIN_AFFINITY_MATRIX,
    MIN_AFFINITY_THRESHOLD,
    STOCHASTIC_AFFINITY_THRESHOLD,
    DYNAMIC_THRESHOLD_SCALE,
    VARIANCE_SWEEP_RANGE,
    VARIANCE_SWEEP_STEPS,
    VARIANCE_DECAY,
    RECEIPT_SCHEMA,
)


def plot_population_dynamics(result: SimResult) -> None:
    """Plot active vs superposition patterns over time."""
    pass


def plot_entropy_trace(result: SimResult) -> None:
    """Plot system entropy over cycles."""
    pass


def plot_completeness_progression(result: SimResult) -> None:
    """Plot L0-L4 coverage over time."""
    pass


def plot_genealogy(result: SimResult) -> None:
    """Plot recombination family tree."""
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


def export_model_details(output_path: Optional[str] = None) -> dict:
    """
    Export model architecture for Grok collaboration.

    Grok: "Share your model details!"

    Args:
        output_path: Optional file path to write JSON export

    Returns:
        dict: Model architecture containing constants, affinity matrix,
              scenario list, receipt schemas, and physics notes
    """
    # Build model details dict
    model = {
        "version": "v12",
        "name": "QED Variance Simulation",
        "constants": {
            "MIN_AFFINITY_THRESHOLD": MIN_AFFINITY_THRESHOLD,
            "STOCHASTIC_AFFINITY_THRESHOLD": STOCHASTIC_AFFINITY_THRESHOLD,
            "DYNAMIC_THRESHOLD_SCALE": DYNAMIC_THRESHOLD_SCALE,
            "VARIANCE_SWEEP_RANGE": list(VARIANCE_SWEEP_RANGE),
            "VARIANCE_SWEEP_STEPS": VARIANCE_SWEEP_STEPS,
            "VARIANCE_DECAY": VARIANCE_DECAY,
        },
        "affinity_matrix": {
            f"{k[0]}_{k[1]}": v for k, v in DOMAIN_AFFINITY_MATRIX.items()
        },
        "scenarios": MANDATORY_SCENARIOS,
        "receipt_schemas": RECEIPT_SCHEMA,
        "physics_notes": {
            "deterministic_threshold": "0.48 - Grok 500-run sweep",
            "stochastic_threshold": "0.52+ - Grok variance analysis",
            "dynamic_formula": "threshold = base + scale * variance",
            "decay_constant": "0.95 - prevents variance amplification",
            "grok_citation": "noise can amplify penalties below ~0.5"
        }
    }

    # Compute dual_hash of content per CLAUDEME
    content_str = json.dumps(model, sort_keys=True)
    model["dual_hash"] = dual_hash(content_str)

    # Write to file if path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(model, f, indent=2)

    # Emit export receipt per CLAUDEME LAW_1
    emit_receipt("model_export_receipt", {
        "tenant_id": "simulation",
        "model_version": model["version"],
        "constants_count": len(model["constants"]),
        "scenarios_count": len(model["scenarios"]),
        "dual_hash": model["dual_hash"],
        "output_path": output_path
    })

    return model


def model_to_grok_format(model: dict) -> str:
    """
    Convert model dict to tweet-length string for X/Grok sharing.

    Purpose: Quick summary for X/Grok sharing (<=280 chars)

    Args:
        model: Model dict from export_model_details()

    Returns:
        str: Tweet-length summary (<=280 chars)
    """
    # Build compact summary
    det_thresh = model.get("constants", {}).get("MIN_AFFINITY_THRESHOLD", 0.48)
    stoch_thresh = model.get("constants", {}).get("STOCHASTIC_AFFINITY_THRESHOLD", 0.52)
    scenarios = len(model.get("scenarios", []))
    version = model.get("version", "v12")

    summary = (
        f"QED {version}: Dynamic thresholding for stochastic affinity. "
        f"Det={det_thresh}, Stoch={stoch_thresh}+. "
        f"{scenarios} scenarios. "
        f"Grok: noise amplifies <0.5. "
        f"Hash:{model.get('dual_hash', 'N/A')[:16]}..."
    )

    # Ensure within 280 chars
    if len(summary) > 280:
        summary = summary[:277] + "..."

    return summary
