"""
merge_rules.py - v9 Config Merge Rules with Centrality-Based Thresholds

Replaces mode-based validation with centrality-computed policies.
Safety constraints can only tighten across config layers.
Compaction resistance computed from centrality and safety tags.

v9 Paradigm Shifts:
===================
1. DELETE mode transitions: No shadow_mode_allowed, no PatternMode enum
2. Value is Topology: CENTRALITY_FLOOR = 0.2 replaces $1M threshold
3. Safety patterns are immortal: safety_tag=True → infinite compaction resistance

Core Invariants:
================
- Safety only tightens: recall_floor can only increase, max_fp can only decrease
- Safety tags cannot downgrade: True→False is VIOLATION
- Centrality floor warnings: patterns < 0.2 centrality emit warnings (not hard block)
- Policy diffs required: every config change emits policy_diff_receipt

What does NOT exist in this file:
==================================
- shadow_mode_allowed field or validation
- PatternMode enum or mode references
- Mode transition validation (live→shadow→deprecated)
- Dollar amount thresholds ($1M, $10M) - use centrality instead

References:
===========
- CLAUDEME Section 2.4: DIFFONLY, StepLock, no hotpath learning
- CLAUDEME Section 5.2: receipt schemas and self-describing modules
- CLAUDEME Section 3.6: Simplicity Rule (explainable in 5 minutes)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "RECEIPT_SCHEMA",
    "CENTRALITY_FLOOR",
    "SAFETY_FIELDS",
    "MergeResult",
    "validate_merge",
    "safety_only_tightens",
    "compute_compaction_resistance",
    "check_centrality_floor",
    "emit_policy_diff",
    "merge_configs",
]


# =============================================================================
# Constants
# =============================================================================

# Centrality floor - patterns below this emit warnings
# Replaces $1M threshold from v4-v8
CENTRALITY_FLOOR = 0.2

# Safety-critical fields that can only tighten
SAFETY_FIELDS = frozenset({
    "recall_floor",
    "max_false_positive_rate",
    "safety_tag",
    "regulatory_tag",
})


# =============================================================================
# Receipt Schema (self-describing module contract per CLAUDEME 3.6)
# =============================================================================

RECEIPT_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "merge_receipt",
        "version": "1.0.0",
        "description": "Receipt emitted by validate_merge() for config merge validation",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of base_hash + overlay_hash + timestamp",
            "timestamp": "ISO UTC timestamp",
            "base_hash": "SHA3-256 hash of base config",
            "overlay_hash": "SHA3-256 hash of overlay config",
            "merged_hash": "SHA3-256 hash of merged config (empty if invalid)",
            "valid": "bool - whether merge satisfies safety constraints",
            "violations": "List[str] - violation messages if any",
            "warnings": "List[str] - warnings about low centrality patterns",
            "tightened_fields": "List[str] - safety fields that were tightened",
        },
    },
    {
        "type": "policy_diff_receipt",
        "version": "1.0.0",
        "description": "Receipt emitted when config changes (per Charter line 88-89)",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of diff content",
            "timestamp": "ISO UTC timestamp",
            "base_hash": "SHA3-256 hash of base config",
            "overlay_hash": "SHA3-256 hash of overlay config",
            "diffs": "Dict[field_name, Dict] - changes detected",
            "owner": "str - who made the change",
            "reason": "str - why change was made",
            "auto_expiry": "ISO timestamp - when this policy expires (default 7 days)",
        },
    },
]


# =============================================================================
# MergeResult Dataclass
# =============================================================================

@dataclass(frozen=True)
class MergeResult:
    """
    Result of config merge validation.

    Attributes:
        valid: True if merge satisfies all safety constraints
        merged_config: Merged config dict if valid, None otherwise
        violations: List of violation messages
        policy_diffs: List of policy_diff_receipt dicts
        receipt: merge_receipt dict for audit trail
    """
    valid: bool
    merged_config: Optional[Dict[str, Any]]
    violations: List[str]
    policy_diffs: List[Dict[str, Any]]
    receipt: Dict[str, Any]


# =============================================================================
# Safety Validation Functions
# =============================================================================

def safety_only_tightens(
    base_value: Any,
    overlay_value: Any,
    field_name: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check if overlay value tightens (or maintains) safety constraint.

    Safety tightening rules:
    - recall_floor: overlay >= base (can only increase)
    - max_false_positive_rate: overlay <= base (can only decrease)
    - safety_tag: False→True OK, True→False VIOLATION
    - regulatory_tag: False→True OK, True→False VIOLATION

    Args:
        base_value: Value from base config
        overlay_value: Value from overlay config
        field_name: Name of the field being merged

    Returns:
        Tuple of (is_valid, violation_reason)
        If is_valid=True, violation_reason is None
        If is_valid=False, violation_reason explains the violation
    """
    if field_name == "recall_floor":
        # recall_floor can only increase (tighten)
        if overlay_value < base_value:
            return False, f"recall_floor cannot decrease: {base_value} -> {overlay_value}"
        return True, None

    if field_name == "max_false_positive_rate":
        # max_fp_rate can only decrease (tighten)
        if overlay_value > base_value:
            return False, f"max_false_positive_rate cannot increase: {base_value} -> {overlay_value}"
        return True, None

    if field_name == "safety_tag":
        # safety_tag: True→False is violation
        if base_value is True and overlay_value is False:
            return False, "safety_tag cannot downgrade from True to False"
        return True, None

    if field_name == "regulatory_tag":
        # regulatory_tag: True→False is violation
        if base_value is True and overlay_value is False:
            return False, "regulatory_tag cannot downgrade from True to False"
        return True, None

    # Unknown field - allow
    return True, None


def check_centrality_floor(
    patterns: Dict[str, float],
    floor: float = CENTRALITY_FLOOR,
) -> List[str]:
    """
    Check which patterns fall below centrality floor.

    Patterns below floor should not be in live operations.
    This is a warning, not a hard block (pattern may be new/growing).

    Args:
        patterns: Dict mapping pattern_id to centrality value
        floor: Minimum centrality threshold (default: 0.2)

    Returns:
        List of pattern_ids with centrality < floor
    """
    below_floor = []
    for pattern_id, centrality in patterns.items():
        if centrality < floor:
            below_floor.append(pattern_id)
    return below_floor


# =============================================================================
# Compaction Resistance
# =============================================================================

def compute_compaction_resistance(
    pattern_id: str,
    centrality: float,
    safety_tag: bool,
    age_days: int,
) -> float:
    """
    Compute compaction resistance score for pattern retention policy.

    Resistance formula (replacing mode-based retention):
    - If safety_tag=True: return infinity (immortal, never compact)
    - Else: resistance = centrality * (1 + 1/max(age_days, 1))

    High centrality + young age = high resistance (keep longer)
    Low centrality + old age = low resistance (compact candidate)

    This function enables causal_graph.compact() to compute retention
    tiers without needing stored mode state.

    Args:
        pattern_id: Pattern identifier (for logging/debug)
        centrality: Pattern's graph centrality [0, 1]
        safety_tag: If True, pattern is safety-critical
        age_days: Age of pattern in days

    Returns:
        Resistance score in [0, infinity)
        - infinity if safety_tag=True (immortal)
        - centrality * (1 + 1/age_days) otherwise
    """
    if safety_tag:
        # Safety patterns never killed (per QED_Build_Strat_v5 line 439-441)
        return float('inf')

    # Non-safety patterns: decay with age, weighted by centrality
    # Young patterns resist more: age_days=1 → multiplier=2.0
    # Old patterns resist less: age_days=365 → multiplier=1.003
    age_divisor = max(age_days, 1)  # Prevent division by zero
    resistance = centrality * (1 + 1 / age_divisor)

    return resistance


# =============================================================================
# Policy Diff Emission
# =============================================================================

def emit_policy_diff(
    base: Dict[str, Any],
    overlay: Dict[str, Any],
    owner: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Emit policy_diff_receipt tracking config changes.

    Per Charter line 88-89: every config change emits policy_diff.

    Args:
        base: Base config dict
        overlay: Overlay config dict
        owner: Who made the change
        reason: Why change was made

    Returns:
        policy_diff_receipt dict
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Compute hashes
    base_canonical = json.dumps(base, sort_keys=True, separators=(",", ":"))
    base_hash = hashlib.sha3_256(base_canonical.encode()).hexdigest()[:16]

    overlay_canonical = json.dumps(overlay, sort_keys=True, separators=(",", ":"))
    overlay_hash = hashlib.sha3_256(overlay_canonical.encode()).hexdigest()[:16]

    # Detect diffs
    diffs: Dict[str, Dict[str, Any]] = {}
    all_keys = set(base.keys()) | set(overlay.keys())

    for key in all_keys:
        base_val = base.get(key)
        overlay_val = overlay.get(key)

        if base_val != overlay_val:
            diffs[key] = {
                "base": base_val,
                "overlay": overlay_val,
            }

    # Auto-expiry: 7 days from now (default policy TTL)
    auto_expiry = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

    # Generate receipt ID
    diff_content = json.dumps({
        "base_hash": base_hash,
        "overlay_hash": overlay_hash,
        "diffs": diffs,
        "timestamp": timestamp,
    }, separators=(",", ":"))
    receipt_id = hashlib.sha3_256(diff_content.encode()).hexdigest()[:16]

    return {
        "type": "policy_diff_receipt",
        "receipt_id": receipt_id,
        "timestamp": timestamp,
        "base_hash": base_hash,
        "overlay_hash": overlay_hash,
        "diffs": diffs,
        "owner": owner,
        "reason": reason,
        "auto_expiry": auto_expiry,
    }


# =============================================================================
# Core Merge Validation
# =============================================================================

def validate_merge(
    base: Dict[str, Any],
    overlay: Dict[str, Any],
    centrality_lookup: Optional[Dict[str, float]] = None,
) -> MergeResult:
    """
    Validate and merge overlay config onto base config.

    Merge rules:
    1. Safety fields only tighten (recall_floor up, max_fp down)
    2. Safety tags cannot downgrade (True→False is violation)
    3. Centrality floor warnings for patterns < 0.2
    4. Emits merge_receipt for audit trail

    Args:
        base: Base config dict
        overlay: Overlay config dict to merge
        centrality_lookup: Optional dict mapping pattern_id to centrality

    Returns:
        MergeResult with merged config or violations
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    violations: List[str] = []
    warnings: List[str] = []
    tightened_fields: List[str] = []

    # Start with base config
    merged = dict(base)

    # Merge each field from overlay
    for key, overlay_val in overlay.items():
        base_val = base.get(key)

        # Check safety fields
        if key in SAFETY_FIELDS:
            valid, violation_msg = safety_only_tightens(base_val, overlay_val, key)
            if not valid:
                violations.append(f"{key}: {violation_msg}")
                # Don't apply overlay value on violation
                continue

            # Check if field was tightened
            if overlay_val != base_val:
                tightened_fields.append(key)

        # Apply overlay value (either safe or non-safety field)
        merged[key] = overlay_val

    # Check centrality floor if provided
    if centrality_lookup:
        below_floor = check_centrality_floor(centrality_lookup, CENTRALITY_FLOOR)
        for pattern_id in below_floor:
            warnings.append(
                f"Pattern '{pattern_id}' has centrality {centrality_lookup[pattern_id]:.3f} "
                f"below floor {CENTRALITY_FLOOR} (may not be ready for live operations)"
            )

    # Compute hashes for receipt
    base_canonical = json.dumps(base, sort_keys=True, separators=(",", ":"))
    base_hash = hashlib.sha3_256(base_canonical.encode()).hexdigest()[:16]

    overlay_canonical = json.dumps(overlay, sort_keys=True, separators=(",", ":"))
    overlay_hash = hashlib.sha3_256(overlay_canonical.encode()).hexdigest()[:16]

    merged_canonical = json.dumps(merged, sort_keys=True, separators=(",", ":"))
    merged_hash = hashlib.sha3_256(merged_canonical.encode()).hexdigest()[:16]

    # Determine validity
    valid = len(violations) == 0

    # Generate receipt ID
    receipt_content = f"{base_hash}:{overlay_hash}:{timestamp}"
    receipt_id = hashlib.sha3_256(receipt_content.encode()).hexdigest()[:16]

    # Create merge receipt
    merge_receipt = {
        "type": "merge_receipt",
        "receipt_id": receipt_id,
        "timestamp": timestamp,
        "base_hash": base_hash,
        "overlay_hash": overlay_hash,
        "merged_hash": merged_hash if valid else "",
        "valid": valid,
        "violations": violations,
        "warnings": warnings,
        "tightened_fields": tightened_fields,
    }

    return MergeResult(
        valid=valid,
        merged_config=merged if valid else None,
        violations=violations,
        policy_diffs=[],  # Populated by caller if needed
        receipt=merge_receipt,
    )


# =============================================================================
# Layered Config Merge
# =============================================================================

def merge_configs(
    configs: List[Dict[str, Any]],
    centrality_lookup: Optional[Dict[str, float]] = None,
) -> MergeResult:
    """
    Merge list of configs in order (global → regional → deployment).

    Each layer validated via validate_merge().
    Stops on first violation.
    Returns final MergeResult with full audit chain.

    Args:
        configs: List of config dicts to merge in order
        centrality_lookup: Optional dict mapping pattern_id to centrality

    Returns:
        MergeResult with final merged config or first violation
    """
    if not configs:
        raise ValueError("merge_configs requires at least 1 config")

    if len(configs) == 1:
        # Single config - trivial merge
        return validate_merge(configs[0], {}, centrality_lookup)

    # Accumulate policy diffs
    all_policy_diffs: List[Dict[str, Any]] = []

    # Start with first config as base
    current = configs[0]

    # Merge each subsequent config
    for i, overlay in enumerate(configs[1:], 1):
        result = validate_merge(current, overlay, centrality_lookup)

        # Emit policy diff for this layer
        policy_diff = emit_policy_diff(
            base=current,
            overlay=overlay,
            owner=overlay.get("owner", "system"),
            reason=overlay.get("reason", f"layer {i} merge"),
        )
        all_policy_diffs.append(policy_diff)

        if not result.valid:
            # Merge failed - return with violations
            return MergeResult(
                valid=False,
                merged_config=None,
                violations=result.violations,
                policy_diffs=all_policy_diffs,
                receipt=result.receipt,
            )

        # Continue with merged config as new base
        current = result.merged_config

    # Final merge successful
    final_canonical = json.dumps(current, sort_keys=True, separators=(",", ":"))
    final_hash = hashlib.sha3_256(final_canonical.encode()).hexdigest()[:16]

    timestamp = datetime.now(timezone.utc).isoformat()
    receipt_id = hashlib.sha3_256(f"chain:{final_hash}:{timestamp}".encode()).hexdigest()[:16]

    # Create final receipt
    final_receipt = {
        "type": "merge_receipt",
        "receipt_id": receipt_id,
        "timestamp": timestamp,
        "base_hash": hashlib.sha3_256(
            json.dumps(configs[0], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16],
        "overlay_hash": hashlib.sha3_256(
            json.dumps(configs[-1], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16],
        "merged_hash": final_hash,
        "valid": True,
        "violations": [],
        "warnings": [],
        "tightened_fields": [],
    }

    return MergeResult(
        valid=True,
        merged_config=current,
        violations=[],
        policy_diffs=all_policy_diffs,
        receipt=final_receipt,
    )
