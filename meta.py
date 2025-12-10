"""
meta.py - Adaptive Immunity Layer (v11)

Tracks paradigm shift outcomes, allocates energy per cycle, weights future proposals.
Provides L1 and L2 receipts in the receipt hierarchy:
- L1: meta_fitness_receipt (receipts about agents)
- L2: paradigm_outcome (receipts about paradigm shifts)

CLAUDEME v3.1 Compliant: SCHEMA + EMIT + TEST + STOPRULE for each receipt type.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# =============================================================================
# CLAUDEME §8 CORE FUNCTIONS
# =============================================================================

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 - ALWAYS use this, never single hash."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Every function calls this. No exceptions."""
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.now(timezone.utc).isoformat(),
        "tenant_id": data.get("tenant_id", "default"),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data
    }
    return receipt


class StopRule(Exception):
    """Raised when stoprule triggers. Never catch silently."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_OBSERVATION_WINDOW_DAYS = 30
MIN_PARADIGMS_FOR_WEIGHTING = 3  # Need at least 3 paradigm outcomes to learn from
SURVIVAL_WEIGHT_BONUS = 0.4
POSITIVE_ROI_WEIGHT_BONUS = 0.3
FAILURE_WEIGHT_PENALTY = 0.2

# Module exports for receipt types
RECEIPT_SCHEMA = ["meta_fitness_receipt", "energy_allocation_receipt", "paradigm_outcome"]


# =============================================================================
# HELPER FUNCTION: compute_paradigm_delta
# =============================================================================

def compute_paradigm_delta(pre: Dict[str, float], post: Dict[str, float]) -> Dict[str, float]:
    """
    Helper function to compute changes between pre and post metrics.

    Args:
        pre: Dict of metrics before paradigm shift
        post: Dict of metrics after paradigm shift

    Returns:
        Dict of deltas (post[metric] - pre[metric]) for each metric
    """
    deltas = {}
    all_keys = set(pre.keys()) | set(post.keys())

    for key in all_keys:
        pre_val = pre.get(key, 0.0)
        post_val = post.get(key, 0.0)
        deltas[key] = post_val - pre_val

    return deltas


# =============================================================================
# RECEIPT TYPE 1: meta_fitness_receipt (L1 - receipts about agents)
# =============================================================================

# --- SCHEMA ---
META_FITNESS_SCHEMA = {
    "receipt_type": "meta_fitness_receipt",
    "ts": "ISO8601",
    "tenant_id": "str",
    "paradigm_name": "str",
    "introduced_at": "ISO8601",
    "pre_metrics": {"roi": "float", "fitness_avg": "float", "entropy_delta": "float"},
    "post_metrics": {"roi": "float", "fitness_avg": "float", "entropy_delta": "float"},
    "delta_roi": "float",
    "survival_30d": "bool|null",
    "observation_window_days": "int",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_meta_fitness(paradigm_shift: dict, tenant_id: str = "default") -> dict:
    """
    Records paradigm shift outcomes for learning. L1 receipt about agents.

    Args:
        paradigm_shift: Dict with required fields:
            - name: identifier for the paradigm shift
            - introduced_at: ISO8601 timestamp when shift was introduced
            - pre_metrics: dict with roi, fitness_avg, entropy_delta
            - post_metrics: dict with roi, fitness_avg, entropy_delta
            - survival_30d: bool or None if too early to tell
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        meta_fitness_receipt dict
    """
    name = paradigm_shift.get("name", "unknown")
    introduced_at = paradigm_shift.get("introduced_at", datetime.now(timezone.utc).isoformat())
    pre_metrics = paradigm_shift.get("pre_metrics", {"roi": 0.0, "fitness_avg": 0.0, "entropy_delta": 0.0})
    post_metrics = paradigm_shift.get("post_metrics", {"roi": 0.0, "fitness_avg": 0.0, "entropy_delta": 0.0})
    survival_30d = paradigm_shift.get("survival_30d", None)

    # Compute delta_roi
    delta_roi = post_metrics.get("roi", 0.0) - pre_metrics.get("roi", 0.0)

    return emit_receipt("meta_fitness_receipt", {
        "tenant_id": tenant_id,
        "paradigm_name": name,
        "introduced_at": introduced_at,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "delta_roi": delta_roi,
        "survival_30d": survival_30d,
        "observation_window_days": DEFAULT_OBSERVATION_WINDOW_DAYS
    })


# --- TEST ---
def test_meta_fitness():
    """Test meta_fitness_receipt emission."""
    paradigm = {
        "name": "test_paradigm",
        "introduced_at": "2024-01-01T00:00:00Z",
        "pre_metrics": {"roi": 0.5, "fitness_avg": 0.6, "entropy_delta": 0.1},
        "post_metrics": {"roi": 0.8, "fitness_avg": 0.7, "entropy_delta": 0.05},
        "survival_30d": True
    }
    r = emit_meta_fitness(paradigm, "test_tenant")
    assert r["receipt_type"] == "meta_fitness_receipt"
    assert r["tenant_id"] == "test_tenant"
    assert r["paradigm_name"] == "test_paradigm"
    assert abs(r["delta_roi"] - 0.3) < 1e-10  # 0.8 - 0.5
    assert r["survival_30d"] is True
    assert "payload_hash" in r


# --- STOPRULE ---
def stoprule_meta_fitness(tenant_id: str, delta_roi: float, survival_30d: Optional[bool]) -> None:
    """
    Meta fitness stoprule - triggers when paradigm shows consistent failure.
    """
    if survival_30d is False and delta_roi < 0:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "paradigm_fitness",
            "baseline": 0.0,
            "delta": delta_roi,
            "classification": "degradation",
            "action": "alert"
        })


# =============================================================================
# RECEIPT TYPE 2: energy_allocation_receipt
# =============================================================================

# --- SCHEMA ---
ENERGY_ALLOCATION_SCHEMA = {
    "receipt_type": "energy_allocation_receipt",
    "ts": "ISO8601",
    "tenant_id": "str",
    "cycle_id": "int",
    "total_energy": "float",
    "allocations": [{"agent_id": "str", "fitness": "float", "energy_share": "float"}],
    "pattern_count": "int",
    "payload_hash": "str"
}


# --- EMIT ---
def emit_energy_allocation(cycle: dict, tenant_id: str = "default") -> dict:
    """
    Logs per-cycle energy distribution. Used for hostile audit trail.

    Args:
        cycle: Dict with required fields:
            - cycle_id: int, monotonic cycle counter
            - patterns: list of pattern dicts, each with {pattern_id, fitness, resource_share}
        tenant_id: Tenant identifier (required per CLAUDEME)

    Returns:
        energy_allocation_receipt dict

    The function:
    - Computes total_energy as sum of resource_share across all patterns
    - Normalizes allocations so energy_share = resource_share / total_energy
    """
    cycle_id = cycle.get("cycle_id", 0)
    patterns = cycle.get("patterns", [])

    # Compute total energy
    total_energy = sum(p.get("resource_share", 0.0) for p in patterns)

    # Normalize allocations
    allocations = []
    for pattern in patterns:
        pattern_id = pattern.get("pattern_id", "unknown")
        fitness = pattern.get("fitness", 0.0)
        resource_share = pattern.get("resource_share", 0.0)

        # Avoid division by zero
        energy_share = resource_share / total_energy if total_energy > 0 else 0.0

        allocations.append({
            "agent_id": pattern_id,
            "fitness": fitness,
            "energy_share": energy_share
        })

    return emit_receipt("energy_allocation_receipt", {
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "total_energy": total_energy,
        "allocations": allocations,
        "pattern_count": len(patterns)
    })


# --- TEST ---
def test_energy_allocation():
    """Test energy_allocation_receipt emission."""
    cycle = {
        "cycle_id": 42,
        "patterns": [
            {"pattern_id": "p1", "fitness": 0.8, "resource_share": 60.0},
            {"pattern_id": "p2", "fitness": 0.6, "resource_share": 40.0}
        ]
    }
    r = emit_energy_allocation(cycle, "test_tenant")
    assert r["receipt_type"] == "energy_allocation_receipt"
    assert r["tenant_id"] == "test_tenant"
    assert r["cycle_id"] == 42
    assert r["total_energy"] == 100.0
    assert len(r["allocations"]) == 2
    assert r["allocations"][0]["energy_share"] == 0.6  # 60/100
    assert r["allocations"][1]["energy_share"] == 0.4  # 40/100
    # Verify energy shares sum to 1.0
    total_share = sum(a["energy_share"] for a in r["allocations"])
    assert abs(total_share - 1.0) < 1e-9


# --- STOPRULE ---
def stoprule_energy_allocation(tenant_id: str, allocations: List[dict]) -> None:
    """
    Energy allocation stoprule - triggers when allocation is severely imbalanced.
    """
    if not allocations:
        return

    max_share = max(a.get("energy_share", 0.0) for a in allocations)
    if max_share > 0.95 and len(allocations) > 1:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "energy_concentration",
            "baseline": 1.0 / len(allocations),
            "delta": max_share - (1.0 / len(allocations)),
            "classification": "anti_pattern",
            "action": "alert"
        })


# =============================================================================
# RECEIPT TYPE 3: paradigm_outcome (L2 - receipts about paradigm shifts)
# =============================================================================

# --- SCHEMA ---
PARADIGM_OUTCOME_SCHEMA = {
    "receipt_type": "paradigm_outcome",
    "ts": "ISO8601",
    "tenant_id": "str",
    "paradigms_evaluated": "int",
    "proposed_weights": {"paradigm_type": "float"},  # mapping paradigm_type to weight 0.0-1.0
    "learning_confidence": "float",  # 0.0-1.0
    "payload_hash": "str"
}


# --- EMIT ---
def weight_future_proposals(meta_receipts: List[dict], tenant_id: str = "default") -> dict:
    """
    Uses past paradigm outcomes to weight future proposals. Adaptive immunity.

    Logic:
    - For each paradigm in meta_receipts, extract delta_roi and survival_30d
    - Compute success_weight:
        - positive delta_roi: +0.3 base weight
        - survival_30d=True: +0.4 weight
        - survival_30d=False: -0.2 weight (penalize failed paradigms)
    - Normalize all weights to [0,1]

    Args:
        meta_receipts: List of meta_fitness_receipt dicts
        tenant_id: Tenant identifier

    Returns:
        paradigm_outcome receipt with proposed_weights
    """
    paradigm_weights: Dict[str, float] = {}

    for receipt in meta_receipts:
        # Extract from meta_fitness_receipt
        paradigm_name = receipt.get("paradigm_name", "unknown")
        delta_roi = receipt.get("delta_roi", 0.0)
        survival_30d = receipt.get("survival_30d", None)

        # Start with neutral weight
        weight = 0.5

        # Apply weighting factors
        if delta_roi > 0:
            weight += POSITIVE_ROI_WEIGHT_BONUS  # +0.3

        if survival_30d is True:
            weight += SURVIVAL_WEIGHT_BONUS  # +0.4
        elif survival_30d is False:
            weight -= FAILURE_WEIGHT_PENALTY  # -0.2

        # Aggregate weights per paradigm type (use paradigm_name as type)
        if paradigm_name in paradigm_weights:
            # Average with existing weight
            paradigm_weights[paradigm_name] = (paradigm_weights[paradigm_name] + weight) / 2
        else:
            paradigm_weights[paradigm_name] = weight

    # Normalize weights to [0, 1]
    for name in paradigm_weights:
        paradigm_weights[name] = max(0.0, min(1.0, paradigm_weights[name]))

    # Compute learning confidence based on number of paradigms evaluated
    paradigms_evaluated = len(meta_receipts)
    if paradigms_evaluated >= MIN_PARADIGMS_FOR_WEIGHTING:
        learning_confidence = min(1.0, paradigms_evaluated / 10.0)
    else:
        learning_confidence = 0.0  # Not enough data to be confident

    return emit_receipt("paradigm_outcome", {
        "tenant_id": tenant_id,
        "paradigms_evaluated": paradigms_evaluated,
        "proposed_weights": paradigm_weights,
        "learning_confidence": learning_confidence
    })


# --- TEST ---
def test_weight_future_proposals():
    """Test paradigm_outcome receipt emission."""
    meta_receipts = [
        {"paradigm_name": "good_paradigm", "delta_roi": 0.5, "survival_30d": True},
        {"paradigm_name": "bad_paradigm", "delta_roi": -0.2, "survival_30d": False},
        {"paradigm_name": "neutral_paradigm", "delta_roi": 0.1, "survival_30d": None}
    ]
    r = weight_future_proposals(meta_receipts, "test_tenant")
    assert r["receipt_type"] == "paradigm_outcome"
    assert r["tenant_id"] == "test_tenant"
    assert r["paradigms_evaluated"] == 3
    assert "proposed_weights" in r
    # good_paradigm: 0.5 + 0.3 + 0.4 = 1.2 -> clamped to 1.0
    assert r["proposed_weights"]["good_paradigm"] == 1.0
    # bad_paradigm: 0.5 - 0.2 = 0.3
    assert r["proposed_weights"]["bad_paradigm"] == 0.3
    # neutral_paradigm: 0.5 + 0.3 = 0.8
    assert r["proposed_weights"]["neutral_paradigm"] == 0.8


# --- STOPRULE ---
def stoprule_paradigm_outcome(tenant_id: str, learning_confidence: float) -> None:
    """
    Paradigm outcome stoprule - triggers when confidence is critically low.
    """
    if learning_confidence < 0.1:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "learning_confidence",
            "baseline": 0.3,
            "delta": learning_confidence - 0.3,
            "classification": "deviation",
            "action": "alert"
        })


# =============================================================================
# STOPRULE: meta_tracking_failure
# =============================================================================

def stoprule_meta_tracking_failure(event: dict) -> None:
    """
    Stoprule for when meta tracking fails. Critical for audit trail.

    Args:
        event: Dict with failure details (error, context, etc.)

    Raises:
        StopRule: Always raises to halt on tracking failure
    """
    tenant_id = event.get("tenant_id", "default")
    error = event.get("error", "unknown")
    context = event.get("context", {})

    emit_receipt("anomaly", {
        "tenant_id": tenant_id,
        "metric": "meta_tracking",
        "baseline": 1.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "error": str(error),
        "context": context
    })

    raise StopRule(f"Meta tracking failure: {error}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "RECEIPT_SCHEMA",
    "DEFAULT_OBSERVATION_WINDOW_DAYS",
    "MIN_PARADIGMS_FOR_WEIGHTING",
    "SURVIVAL_WEIGHT_BONUS",
    "POSITIVE_ROI_WEIGHT_BONUS",
    "FAILURE_WEIGHT_PENALTY",
    # Helper functions
    "compute_paradigm_delta",
    # Emit functions
    "emit_meta_fitness",
    "emit_energy_allocation",
    "weight_future_proposals",
    # Stoprules
    "stoprule_meta_fitness",
    "stoprule_energy_allocation",
    "stoprule_paradigm_outcome",
    "stoprule_meta_tracking_failure",
    # Core utilities
    "emit_receipt",
    "dual_hash",
    "StopRule",
]
