"""
QED v8 Merge Rules Engine - Governance for "Safety Only Tightens"

This module is the governance engine ensuring safety constraints only tighten
across configuration layers. It's not just a validator - it's a collaborator
that simulates, auto-repairs, and explains merges with full auditability.

Supports multi-level chains (global -> company -> region -> deployment) with:
- Auto-repair: Suggests and applies fixes instead of just rejecting
- Simulation: Preview merges before committing
- Conflict detection: Pre-flight checks for N configs
- Audit trail: Every merge emits cryptographic receipt

Design Principles:
- Collaborative: Suggests fixes, doesn't just reject
- Predictive: Simulate before commit
- Multi-level: Chains of N configs, not just pairs
- Auditable: Every merge emits receipt automatically
- Defensive: Guards against edge cases (empty intersection, etc.)

Consumed by:
- qed.py (runtime config loading)
- proof.py (CLI config operations)
- TruthLink (packet config validation)
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from config_schema import QEDConfig, ConfigProvenance


__all__ = [
    'merge',
    'merge_chain',
    'simulate_merge',
    'suggest_repairs',
    'apply_repairs',
    'detect_conflicts',
    'emit_receipt',
    'MergeResult',
    'MergeExplanation',
    'MergeReceipt',
    'MergeTrace',
    'MergeSimulation',
    'Violation',
    'Repair',
    'Conflict',
    'FieldDecision',
]


# =============================================================================
# Safety Field Detection
# =============================================================================

_SAFETY_FIELDS = frozenset({
    'recall_floor',
    'max_fp_rate',
    'slo_latency_ms',
    'slo_breach_budget',
})

_REGULATORY_PREFIX = 'regulatory_'


def is_safety_field(field_name: str) -> bool:
    """
    Check if a field is a safety-critical field.

    Safety fields can only be tightened (never loosened) in child configs.

    Args:
        field_name: Name of the config field

    Returns:
        True if field is safety-critical
    """
    return (
        field_name in _SAFETY_FIELDS or
        field_name.startswith('slo_') or
        field_name.startswith(_REGULATORY_PREFIX)
    )


def cannot_loosen(field_name: str, parent_val: Any, child_val: Any) -> bool:
    """
    Check if child value attempts to loosen a safety constraint.

    Handles numeric comparison, set comparison, and flag comparison.

    Args:
        field_name: Name of the config field
        parent_val: Parent config value
        child_val: Child config value

    Returns:
        True if child attempts to loosen (violation)
    """
    if field_name == 'recall_floor':
        # Higher = stricter, so child < parent is loosening
        return child_val < parent_val

    if field_name == 'max_fp_rate':
        # Lower = stricter, so child > parent is loosening
        return child_val > parent_val

    if field_name == 'slo_latency_ms':
        # Lower = stricter, so child > parent is loosening
        return child_val > parent_val

    if field_name == 'slo_breach_budget':
        # Lower = stricter, so child > parent is loosening
        return child_val > parent_val

    if field_name == 'enabled_patterns':
        # Intersection rule - child adding patterns parent doesn't have
        parent_set = set(parent_val) if parent_val else set()
        child_set = set(child_val) if child_val else set()
        # If child has patterns not in parent, that's loosening
        return len(child_set - parent_set) > 0 if parent_set else False

    if field_name == 'regulatory_flags':
        # OR rule - removing required flags is loosening
        if isinstance(parent_val, dict) and isinstance(child_val, dict):
            for flag, required in parent_val.items():
                if required and not child_val.get(flag, False):
                    return True  # Removing required flag
        return False

    # Unknown field - allow
    return False


def empty_intersection_guard(
    parent_patterns: Union[List[str], Tuple[str, ...]],
    child_patterns: Union[List[str], Tuple[str, ...]]
) -> Tuple[List[str], Optional[str]]:
    """
    Guard against empty pattern intersection.

    If intersection would be empty, emit warning to prevent
    accidental "block everything" configs.

    Args:
        parent_patterns: Patterns from parent config
        child_patterns: Patterns from child config

    Returns:
        Tuple of (intersection_list, warning_message or None)
    """
    parent_set = set(parent_patterns) if parent_patterns else set()
    child_set = set(child_patterns) if child_patterns else set()

    # Special case: empty means "all patterns"
    if not parent_set and not child_set:
        return [], None  # Both empty = all patterns allowed

    if not parent_set:
        # Parent allows all, use child's
        return sorted(child_set), None

    if not child_set:
        # Child allows all, use parent's
        return sorted(parent_set), None

    intersection = parent_set & child_set

    if not intersection:
        warning = (
            f"Empty pattern intersection! Parent has {len(parent_set)} patterns, "
            f"child has {len(child_set)} patterns, but none overlap. "
            f"This would block ALL patterns."
        )
        return [], warning

    return sorted(intersection), None


# =============================================================================
# Violation Dataclass
# =============================================================================

@dataclass(frozen=True)
class Violation:
    """
    Represents a merge rule violation.

    A violation occurs when a child config attempts to loosen
    safety constraints relative to its parent.
    """
    field: str
    rule: str
    parent_value: Any
    child_value: Any
    attempted_direction: Literal["loosen", "disable", "conflict"]
    severity: Literal["error", "warning"]

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.field}: "
            f"Cannot {self.attempted_direction} from {self.parent_value} to {self.child_value} "
            f"(rule: {self.rule})"
        )


# =============================================================================
# Repair Dataclass
# =============================================================================

@dataclass(frozen=True)
class Repair:
    """
    Represents an auto-repair action.

    When auto_repair=True, repairs are computed and applied to
    bring child config into compliance with merge rules.
    """
    field: str
    original_value: Any
    repaired_value: Any
    repair_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.field}: {self.original_value} -> {self.repaired_value} "
            f"({self.repair_action})"
        )


# =============================================================================
# Conflict Dataclass
# =============================================================================

@dataclass(frozen=True)
class Conflict:
    """
    Represents a conflict between multiple configs.

    Used by detect_conflicts() for pre-flight validation before
    attempting a merge chain.
    """
    field: str
    configs_involved: Tuple[str, ...]
    values: Dict[str, Any]
    resolution_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'field': self.field,
            'configs_involved': list(self.configs_involved),
            'values': self.values,
            'resolution_strategy': self.resolution_strategy
        }

    def __str__(self) -> str:
        configs = ", ".join(self.configs_involved)
        return f"{self.field}: conflict between [{configs}] - {self.resolution_strategy}"


# =============================================================================
# FieldDecision Dataclass
# =============================================================================

@dataclass(frozen=True)
class FieldDecision:
    """
    Documents how a single field was merged.

    Part of MergeExplanation - provides transparency into
    merge logic for each field.
    """
    field: str
    parent_value: Any
    child_value: Any
    merged_value: Any
    rule_applied: str
    direction: Literal["from_parent", "from_child", "combined", "intersection"]

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return asdict(self)


# =============================================================================
# MergeExplanation Dataclass
# =============================================================================

@dataclass(frozen=True)
class MergeExplanation:
    """
    Human-readable explanation of a merge operation.

    Provides complete transparency into what happened during merge,
    why decisions were made, and the overall safety assessment.
    """
    summary: str
    field_decisions: Dict[str, FieldDecision]
    patterns_kept: Tuple[str, ...]
    patterns_removed: Tuple[str, ...]
    regulatory_combined: Dict[str, bool]
    safety_direction: Literal["tightened", "unchanged", "VIOLATION"]
    narration: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'summary': self.summary,
            'field_decisions': {k: v.to_dict() for k, v in self.field_decisions.items()},
            'patterns_kept': list(self.patterns_kept),
            'patterns_removed': list(self.patterns_removed),
            'regulatory_combined': self.regulatory_combined,
            'safety_direction': self.safety_direction,
            'narration': self.narration
        }


# =============================================================================
# MergeReceipt Dataclass
# =============================================================================

@dataclass(frozen=True)
class MergeReceipt:
    """
    Cryptographic receipt for audit trail.

    Every merge operation generates a receipt that can be used to:
    - Prove what configs were merged
    - Verify integrity of the result
    - Track who/when/why for compliance
    """
    timestamp: str
    receipt_id: str
    parent_hash: str
    child_hash: str
    merged_hash: str
    is_valid: bool
    violations_count: int
    repairs_count: int
    fields_tightened: Tuple[str, ...]
    patterns_removed: Tuple[str, ...]
    author: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'timestamp': self.timestamp,
            'receipt_id': self.receipt_id,
            'parent_hash': self.parent_hash,
            'child_hash': self.child_hash,
            'merged_hash': self.merged_hash,
            'is_valid': self.is_valid,
            'violations_count': self.violations_count,
            'repairs_count': self.repairs_count,
            'fields_tightened': list(self.fields_tightened),
            'patterns_removed': list(self.patterns_removed),
            'author': self.author,
            'reason': self.reason
        }

    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))


def emit_receipt(receipt: MergeReceipt, output_path: str = "merge_receipts.jsonl") -> None:
    """
    Append receipt to JSONL audit log.

    Creates the file if it doesn't exist, appends if it does.

    Args:
        receipt: MergeReceipt to emit
        output_path: Path to JSONL file (default: merge_receipts.jsonl)
    """
    path = Path(output_path)
    with path.open('a', encoding='utf-8') as f:
        f.write(receipt.to_json() + '\n')


# =============================================================================
# MergeTrace Dataclass (for chain merges)
# =============================================================================

@dataclass(frozen=True)
class MergeTrace:
    """
    Trace of a multi-level merge chain.

    Records what happened at each level for full auditability.
    """
    levels: Tuple[str, ...]
    per_level_violations: Dict[str, Tuple[Violation, ...]]
    per_level_repairs: Dict[str, Tuple[Repair, ...]]
    cumulative_tightening: Dict[str, Tuple[Any, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'levels': list(self.levels),
            'per_level_violations': {
                k: [v.to_dict() for v in vlist]
                for k, vlist in self.per_level_violations.items()
            },
            'per_level_repairs': {
                k: [r.to_dict() for r in rlist]
                for k, rlist in self.per_level_repairs.items()
            },
            'cumulative_tightening': {
                k: [v[0], v[1]] for k, v in self.cumulative_tightening.items()
            }
        }


# =============================================================================
# MergeSimulation Dataclass
# =============================================================================

@dataclass(frozen=True)
class MergeSimulation:
    """
    Result of simulating a merge (dry run).

    Preview what would happen without actually committing.
    """
    would_be_valid: bool
    merged_preview: Optional[QEDConfig]
    violations_preview: Tuple[Violation, ...]
    repairs_available: Tuple[Repair, ...]
    impact_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'would_be_valid': self.would_be_valid,
            'merged_preview': self.merged_preview.to_dict() if self.merged_preview else None,
            'violations_preview': [v.to_dict() for v in self.violations_preview],
            'repairs_available': [r.to_dict() for r in self.repairs_available],
            'impact_summary': self.impact_summary
        }


# =============================================================================
# MergeResult Dataclass
# =============================================================================

@dataclass(frozen=True)
class MergeResult:
    """
    Complete result of a merge operation.

    Single return type containing everything: merged config,
    validation status, violations, repairs, explanation, and receipt.
    """
    merged: Optional[QEDConfig]
    is_valid: bool
    violations: Tuple[Violation, ...]
    repairs_applied: Tuple[Repair, ...]
    explanation: MergeExplanation
    receipt: MergeReceipt
    trace: Optional[MergeTrace] = None

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'merged': self.merged.to_dict() if self.merged else None,
            'is_valid': self.is_valid,
            'violations': [v.to_dict() for v in self.violations],
            'repairs_applied': [r.to_dict() for r in self.repairs_applied],
            'explanation': self.explanation.to_dict(),
            'receipt': self.receipt.to_dict(),
            'trace': self.trace.to_dict() if self.trace else None
        }


# =============================================================================
# Core Merge Implementation
# =============================================================================

def _compute_receipt_id(
    parent_hash: str,
    child_hash: str,
    timestamp: str
) -> str:
    """Compute SHA3 receipt ID from merge inputs."""
    data = f"{parent_hash}:{child_hash}:{timestamp}"
    return hashlib.sha3_256(data.encode()).hexdigest()[:16]


def _merge_field(
    field_name: str,
    parent_val: Any,
    child_val: Any
) -> Tuple[Any, str, str]:
    """
    Merge a single field according to safety rules.

    Returns: (merged_value, rule_applied, direction)
    """
    if field_name == 'recall_floor':
        # Take stricter (higher)
        merged = max(parent_val, child_val)
        direction = "from_child" if merged == child_val else "from_parent"
        return merged, "max(parent, child) - take stricter", direction

    if field_name == 'max_fp_rate':
        # Take stricter (lower)
        merged = min(parent_val, child_val)
        direction = "from_child" if merged == child_val else "from_parent"
        return merged, "min(parent, child) - take stricter", direction

    if field_name == 'slo_latency_ms':
        # Take tighter (lower)
        merged = min(parent_val, child_val)
        direction = "from_child" if merged == child_val else "from_parent"
        return merged, "min(parent, child) - take tighter", direction

    if field_name == 'slo_breach_budget':
        # Take tighter (lower)
        merged = min(parent_val, child_val)
        direction = "from_child" if merged == child_val else "from_parent"
        return merged, "min(parent, child) - take tighter", direction

    if field_name == 'enabled_patterns':
        # Intersection - only patterns in BOTH
        parent_set = set(parent_val) if parent_val else set()
        child_set = set(child_val) if child_val else set()

        if not parent_set and not child_set:
            return tuple(), "intersection - both empty = all allowed", "combined"
        if not parent_set:
            return tuple(sorted(child_set)), "child patterns (parent allows all)", "from_child"
        if not child_set:
            return tuple(sorted(parent_set)), "parent patterns (child allows all)", "from_parent"

        merged = tuple(sorted(parent_set & child_set))
        return merged, "intersection - only patterns in BOTH", "intersection"

    if field_name == 'regulatory_flags':
        # OR - if either requires, merged requires
        parent_flags = parent_val if isinstance(parent_val, dict) else {}
        child_flags = child_val if isinstance(child_val, dict) else {}

        merged = {}
        all_flags = set(parent_flags.keys()) | set(child_flags.keys())
        for flag in all_flags:
            # OR: True if either is True
            merged[flag] = parent_flags.get(flag, False) or child_flags.get(flag, False)

        return merged, "OR - if either requires, merged requires", "combined"

    if field_name == 'safety_overrides':
        # Union with parent wins on conflict
        parent_overrides = parent_val if isinstance(parent_val, dict) else {}
        child_overrides = child_val if isinstance(child_val, dict) else {}

        merged = dict(child_overrides)  # Start with child
        merged.update(parent_overrides)  # Parent wins on conflict

        return merged, "union - parent wins on conflict", "combined"

    # Non-safety fields: take child value
    return child_val, "child value (non-safety field)", "from_child"


def _detect_violations(
    parent: QEDConfig,
    child: QEDConfig
) -> List[Violation]:
    """Detect all violations in child relative to parent."""
    violations = []

    # recall_floor: child cannot be lower
    if child.recall_floor < parent.recall_floor:
        violations.append(Violation(
            field='recall_floor',
            rule='safety_only_tightens',
            parent_value=parent.recall_floor,
            child_value=child.recall_floor,
            attempted_direction='loosen',
            severity='error'
        ))

    # max_fp_rate: child cannot be higher
    if child.max_fp_rate > parent.max_fp_rate:
        violations.append(Violation(
            field='max_fp_rate',
            rule='safety_only_tightens',
            parent_value=parent.max_fp_rate,
            child_value=child.max_fp_rate,
            attempted_direction='loosen',
            severity='error'
        ))

    # slo_latency_ms: child cannot be higher
    if child.slo_latency_ms > parent.slo_latency_ms:
        violations.append(Violation(
            field='slo_latency_ms',
            rule='safety_only_tightens',
            parent_value=parent.slo_latency_ms,
            child_value=child.slo_latency_ms,
            attempted_direction='loosen',
            severity='error'
        ))

    # slo_breach_budget: child cannot be higher
    if child.slo_breach_budget > parent.slo_breach_budget:
        violations.append(Violation(
            field='slo_breach_budget',
            rule='safety_only_tightens',
            parent_value=parent.slo_breach_budget,
            child_value=child.slo_breach_budget,
            attempted_direction='loosen',
            severity='error'
        ))

    # enabled_patterns: child cannot add patterns not in parent
    parent_patterns = set(parent.enabled_patterns) if parent.enabled_patterns else set()
    child_patterns = set(child.enabled_patterns) if child.enabled_patterns else set()

    if parent_patterns:  # Only check if parent has restrictions
        added_patterns = child_patterns - parent_patterns
        if added_patterns:
            violations.append(Violation(
                field='enabled_patterns',
                rule='intersection_only',
                parent_value=sorted(parent_patterns),
                child_value=sorted(child_patterns),
                attempted_direction='loosen',
                severity='error'
            ))

    # regulatory_flags: child cannot disable required flags
    for flag, required in parent.regulatory_flags.items():
        if required and not child.regulatory_flags.get(flag, False):
            violations.append(Violation(
                field=f'regulatory_flags.{flag}',
                rule='cannot_disable_required',
                parent_value=True,
                child_value=child.regulatory_flags.get(flag, False),
                attempted_direction='disable',
                severity='error'
            ))

    return violations


def _compute_repairs(violations: List[Violation], parent: QEDConfig) -> List[Repair]:
    """Compute repairs for all violations."""
    repairs = []

    for v in violations:
        if v.field == 'recall_floor':
            repairs.append(Repair(
                field='recall_floor',
                original_value=v.child_value,
                repaired_value=v.parent_value,
                repair_action='tightened to parent value'
            ))

        elif v.field == 'max_fp_rate':
            repairs.append(Repair(
                field='max_fp_rate',
                original_value=v.child_value,
                repaired_value=v.parent_value,
                repair_action='tightened to parent value'
            ))

        elif v.field == 'slo_latency_ms':
            repairs.append(Repair(
                field='slo_latency_ms',
                original_value=v.child_value,
                repaired_value=v.parent_value,
                repair_action='tightened to parent value'
            ))

        elif v.field == 'slo_breach_budget':
            repairs.append(Repair(
                field='slo_breach_budget',
                original_value=v.child_value,
                repaired_value=v.parent_value,
                repair_action='tightened to parent value'
            ))

        elif v.field == 'enabled_patterns':
            # Repair by taking intersection
            parent_set = set(parent.enabled_patterns) if parent.enabled_patterns else set()
            child_set = set(v.child_value) if v.child_value else set()
            repaired = sorted(parent_set & child_set) if parent_set else sorted(child_set)
            repairs.append(Repair(
                field='enabled_patterns',
                original_value=v.child_value,
                repaired_value=repaired,
                repair_action='intersection with parent patterns'
            ))

        elif v.field.startswith('regulatory_flags.'):
            flag_name = v.field.split('.')[1]
            repairs.append(Repair(
                field=v.field,
                original_value=v.child_value,
                repaired_value=True,
                repair_action=f'enabled required flag {flag_name}'
            ))

    return repairs


def _build_explanation(
    parent: QEDConfig,
    child: QEDConfig,
    merged: QEDConfig,
    violations: List[Violation]
) -> MergeExplanation:
    """Build human-readable merge explanation."""
    field_decisions: Dict[str, FieldDecision] = {}
    fields_tightened: List[str] = []

    # Track decisions for safety fields
    safety_fields = ['recall_floor', 'max_fp_rate', 'slo_latency_ms', 'slo_breach_budget']

    for fname in safety_fields:
        parent_val = getattr(parent, fname)
        child_val = getattr(child, fname)
        merged_val = getattr(merged, fname)
        merged_result, rule, direction = _merge_field(fname, parent_val, child_val)

        field_decisions[fname] = FieldDecision(
            field=fname,
            parent_value=parent_val,
            child_value=child_val,
            merged_value=merged_val,
            rule_applied=rule,
            direction=direction
        )

        if merged_val != child_val:
            fields_tightened.append(fname)

    # Patterns
    parent_patterns = set(parent.enabled_patterns) if parent.enabled_patterns else set()
    child_patterns = set(child.enabled_patterns) if child.enabled_patterns else set()
    merged_patterns = set(merged.enabled_patterns) if merged.enabled_patterns else set()

    if parent_patterns or child_patterns:
        all_patterns = parent_patterns | child_patterns
        patterns_kept = sorted(merged_patterns)
        patterns_removed = sorted(all_patterns - merged_patterns)
    else:
        patterns_kept = []
        patterns_removed = []

    _, pattern_rule, pattern_dir = _merge_field(
        'enabled_patterns',
        parent.enabled_patterns,
        child.enabled_patterns
    )
    field_decisions['enabled_patterns'] = FieldDecision(
        field='enabled_patterns',
        parent_value=list(parent.enabled_patterns),
        child_value=list(child.enabled_patterns),
        merged_value=list(merged.enabled_patterns),
        rule_applied=pattern_rule,
        direction=pattern_dir
    )

    # Regulatory flags
    merged_reg, reg_rule, reg_dir = _merge_field(
        'regulatory_flags',
        parent.regulatory_flags,
        child.regulatory_flags
    )
    field_decisions['regulatory_flags'] = FieldDecision(
        field='regulatory_flags',
        parent_value=parent.regulatory_flags,
        child_value=child.regulatory_flags,
        merged_value=merged.regulatory_flags,
        rule_applied=reg_rule,
        direction=reg_dir
    )

    # Safety overrides
    merged_so, so_rule, so_dir = _merge_field(
        'safety_overrides',
        parent.safety_overrides,
        child.safety_overrides
    )
    field_decisions['safety_overrides'] = FieldDecision(
        field='safety_overrides',
        parent_value=parent.safety_overrides,
        child_value=child.safety_overrides,
        merged_value=merged.safety_overrides,
        rule_applied=so_rule,
        direction=so_dir
    )

    # Determine safety direction
    if violations:
        safety_direction: Literal["tightened", "unchanged", "VIOLATION"] = "VIOLATION"
    elif fields_tightened:
        safety_direction = "tightened"
    else:
        safety_direction = "unchanged"

    # Build summary
    summary_parts = [f"Merged {parent.deployment_id} -> {child.deployment_id}"]
    if fields_tightened:
        summary_parts.append(f"{len(fields_tightened)} fields tightened")
    if patterns_removed:
        summary_parts.append(f"{len(patterns_removed)} patterns removed")
    summary = ": ".join(summary_parts)

    # Build narration
    narration_lines = [
        f"Merging parent config '{parent.deployment_id}' with child config '{child.deployment_id}'.",
    ]

    if fields_tightened:
        narration_lines.append(
            f"The following safety fields were tightened to ensure compliance: {', '.join(fields_tightened)}."
        )

    if patterns_removed:
        narration_lines.append(
            f"Pattern intersection removed {len(patterns_removed)} patterns that were not common to both configs."
        )

    if merged.regulatory_flags:
        enabled_flags = [f for f, v in merged.regulatory_flags.items() if v]
        if enabled_flags:
            narration_lines.append(
                f"Regulatory flags enabled in merged config: {', '.join(enabled_flags)}."
            )

    if violations:
        narration_lines.append(
            f"WARNING: {len(violations)} violations detected. "
            "Child config attempted to loosen safety constraints."
        )
    else:
        narration_lines.append(
            "Merge completed successfully with no violations. Safety constraints maintained."
        )

    narration = " ".join(narration_lines)

    return MergeExplanation(
        summary=summary,
        field_decisions=field_decisions,
        patterns_kept=tuple(patterns_kept),
        patterns_removed=tuple(patterns_removed),
        regulatory_combined=dict(merged.regulatory_flags),
        safety_direction=safety_direction,
        narration=narration
    )


def _create_merged_config(
    parent: QEDConfig,
    child: QEDConfig,
    repairs: Optional[List[Repair]] = None,
    author: str = "merge_rules",
    reason: str = "config merge"
) -> QEDConfig:
    """Create merged config from parent and child."""
    # Merge each field
    recall_floor, _, _ = _merge_field('recall_floor', parent.recall_floor, child.recall_floor)
    max_fp_rate, _, _ = _merge_field('max_fp_rate', parent.max_fp_rate, child.max_fp_rate)
    slo_latency_ms, _, _ = _merge_field('slo_latency_ms', parent.slo_latency_ms, child.slo_latency_ms)
    slo_breach_budget, _, _ = _merge_field('slo_breach_budget', parent.slo_breach_budget, child.slo_breach_budget)
    enabled_patterns, _, _ = _merge_field('enabled_patterns', parent.enabled_patterns, child.enabled_patterns)
    regulatory_flags, _, _ = _merge_field('regulatory_flags', parent.regulatory_flags, child.regulatory_flags)
    safety_overrides, _, _ = _merge_field('safety_overrides', parent.safety_overrides, child.safety_overrides)

    # Apply repairs if provided
    if repairs:
        for repair in repairs:
            if repair.field == 'recall_floor':
                recall_floor = repair.repaired_value
            elif repair.field == 'max_fp_rate':
                max_fp_rate = repair.repaired_value
            elif repair.field == 'slo_latency_ms':
                slo_latency_ms = repair.repaired_value
            elif repair.field == 'slo_breach_budget':
                slo_breach_budget = repair.repaired_value
            elif repair.field == 'enabled_patterns':
                enabled_patterns = tuple(repair.repaired_value)
            elif repair.field.startswith('regulatory_flags.'):
                flag_name = repair.field.split('.')[1]
                regulatory_flags = dict(regulatory_flags)
                regulatory_flags[flag_name] = repair.repaired_value

    # Check for empty intersection warning
    _, empty_warning = empty_intersection_guard(parent.enabled_patterns, child.enabled_patterns)
    if empty_warning:
        warnings.warn(f"MergeRules: {empty_warning}", UserWarning, stacklevel=4)

    # Create provenance
    provenance = ConfigProvenance.create(
        author=author,
        reason=reason,
        parent_hash=parent.provenance.config_hash
    )

    # Build merged config dict
    merged_data = {
        'version': child.version,
        'deployment_id': child.deployment_id,
        'hook': child.hook,
        'compression_target': child.compression_target,
        'recall_floor': recall_floor,
        'max_fp_rate': max_fp_rate,
        'slo_latency_ms': slo_latency_ms,
        'slo_breach_budget': slo_breach_budget,
        'enabled_patterns': list(enabled_patterns),
        'safety_overrides': dict(safety_overrides),
        'regulatory_flags': dict(regulatory_flags),
        'provenance': provenance.to_dict()
    }

    return QEDConfig.from_dict(merged_data, validate=True, strict=False)


# =============================================================================
# Public API Functions
# =============================================================================

def merge(
    parent: QEDConfig,
    child: QEDConfig,
    auto_repair: bool = False,
    author: str = "merge_rules",
    reason: str = "config merge",
    emit_receipt_flag: bool = True,
    receipt_path: str = "merge_receipts.jsonl"
) -> MergeResult:
    """
    Merge child config into parent with safety-only-tightens enforcement.

    This is the single entry point for config merging, replacing separate
    validate/merge/explain calls. Returns complete MergeResult with merged
    config, validation status, violations, repairs, explanation, and receipt.

    Merge rules enforced:
    - recall_floor: max(parent, child) - take stricter (higher)
    - max_fp_rate: min(parent, child) - take stricter (lower)
    - slo_latency_ms: min(parent, child) - take tighter
    - slo_breach_budget: min(parent, child) - take tighter
    - enabled_patterns: intersection - only patterns in BOTH
    - regulatory_flags: OR - if either requires, merged requires
    - safety_overrides: union with parent wins on conflict

    Args:
        parent: Parent config (stricter baseline)
        child: Child config (can only tighten)
        auto_repair: If True, automatically fix violations
        author: Who triggered the merge (for audit)
        reason: Why merge was attempted (for audit)
        emit_receipt_flag: If True, emit receipt to JSONL
        receipt_path: Path for receipt JSONL file

    Returns:
        MergeResult with all merge details
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Detect violations
    violations = _detect_violations(parent, child)
    repairs: List[Repair] = []

    # Determine if we can produce valid merged config
    is_valid = len(violations) == 0
    merged: Optional[QEDConfig] = None

    if is_valid:
        # No violations - merge directly
        merged = _create_merged_config(parent, child, author=author, reason=reason)

    elif auto_repair:
        # Violations exist but auto_repair enabled
        repairs = _compute_repairs(violations, parent)
        merged = _create_merged_config(parent, child, repairs=repairs, author=author, reason=reason)
        is_valid = True  # After repair, it's valid

    # Build explanation
    if merged:
        explanation = _build_explanation(parent, child, merged, violations)
    else:
        # Can't merge - provide explanation of why
        explanation = _build_explanation(
            parent, child,
            child,  # Use child as placeholder
            violations
        )

    # Compute tightened fields
    fields_tightened: List[str] = []
    if merged:
        for fname in ['recall_floor', 'max_fp_rate', 'slo_latency_ms', 'slo_breach_budget']:
            if getattr(merged, fname) != getattr(child, fname):
                fields_tightened.append(fname)

    # Build receipt
    receipt = MergeReceipt(
        timestamp=timestamp,
        receipt_id=_compute_receipt_id(
            parent.provenance.config_hash,
            child.provenance.config_hash,
            timestamp
        ),
        parent_hash=parent.provenance.config_hash,
        child_hash=child.provenance.config_hash,
        merged_hash=merged.provenance.config_hash if merged else "",
        is_valid=is_valid,
        violations_count=len(violations),
        repairs_count=len(repairs),
        fields_tightened=tuple(fields_tightened),
        patterns_removed=explanation.patterns_removed,
        author=author,
        reason=reason
    )

    # Emit receipt if enabled
    if emit_receipt_flag:
        try:
            emit_receipt(receipt, receipt_path)
        except Exception as e:
            warnings.warn(f"Failed to emit merge receipt: {e}", UserWarning, stacklevel=2)

    return MergeResult(
        merged=merged,
        is_valid=is_valid,
        violations=tuple(violations),
        repairs_applied=tuple(repairs),
        explanation=explanation,
        receipt=receipt
    )


def merge_chain(
    configs: List[QEDConfig],
    auto_repair: bool = False,
    author: str = "merge_rules",
    reason: str = "chain merge",
    emit_receipt_flag: bool = True,
    receipt_path: str = "merge_receipts.jsonl"
) -> MergeResult:
    """
    Merge N configs in order: configs[0] -> configs[1] -> ... -> configs[N-1].

    Each merge validates against the result of the previous merge.
    Accumulates all violations and repairs across levels.
    Returns final merged config with full trace.

    Use case: global -> company -> region -> deployment in one call.

    Args:
        configs: List of configs to merge in order
        auto_repair: If True, auto-repair violations at each level
        author: Who triggered the merge (for audit)
        reason: Why merge was attempted (for audit)
        emit_receipt_flag: If True, emit receipt to JSONL
        receipt_path: Path for receipt JSONL file

    Returns:
        MergeResult with final config and trace of all levels

    Raises:
        ValueError: If fewer than 2 configs provided
    """
    if len(configs) < 2:
        raise ValueError("merge_chain requires at least 2 configs")

    # Track per-level results
    levels: List[str] = []
    per_level_violations: Dict[str, List[Violation]] = {}
    per_level_repairs: Dict[str, List[Repair]] = {}
    all_violations: List[Violation] = []
    all_repairs: List[Repair] = []

    # Track cumulative tightening from original values
    original_values: Dict[str, Any] = {}
    safety_fields = ['recall_floor', 'max_fp_rate', 'slo_latency_ms', 'slo_breach_budget']
    for fname in safety_fields:
        original_values[fname] = getattr(configs[0], fname)

    # Merge chain
    current = configs[0]
    levels.append(current.deployment_id)

    for i, child in enumerate(configs[1:], 1):
        level_id = child.deployment_id
        levels.append(level_id)

        result = merge(
            current, child,
            auto_repair=auto_repair,
            author=author,
            reason=f"{reason} (level {i})",
            emit_receipt_flag=False  # Emit one receipt at end
        )

        # Track per-level
        per_level_violations[level_id] = list(result.violations)
        per_level_repairs[level_id] = list(result.repairs_applied)
        all_violations.extend(result.violations)
        all_repairs.extend(result.repairs_applied)

        if not result.is_valid and not auto_repair:
            # Chain fails - return with what we have
            trace = MergeTrace(
                levels=tuple(levels),
                per_level_violations={k: tuple(v) for k, v in per_level_violations.items()},
                per_level_repairs={k: tuple(v) for k, v in per_level_repairs.items()},
                cumulative_tightening={}
            )

            # Build receipt for failed chain
            timestamp = datetime.now(timezone.utc).isoformat()
            receipt = MergeReceipt(
                timestamp=timestamp,
                receipt_id=_compute_receipt_id(
                    configs[0].provenance.config_hash,
                    configs[-1].provenance.config_hash,
                    timestamp
                ),
                parent_hash=configs[0].provenance.config_hash,
                child_hash=configs[-1].provenance.config_hash,
                merged_hash="",
                is_valid=False,
                violations_count=len(all_violations),
                repairs_count=len(all_repairs),
                fields_tightened=tuple(),
                patterns_removed=tuple(),
                author=author,
                reason=reason
            )

            if emit_receipt_flag:
                try:
                    emit_receipt(receipt, receipt_path)
                except Exception:
                    pass

            return MergeResult(
                merged=None,
                is_valid=False,
                violations=tuple(all_violations),
                repairs_applied=tuple(all_repairs),
                explanation=result.explanation,
                receipt=receipt,
                trace=trace
            )

        if result.merged:
            current = result.merged

    # Compute cumulative tightening
    cumulative_tightening: Dict[str, Tuple[Any, Any]] = {}
    for fname in safety_fields:
        original = original_values[fname]
        final = getattr(current, fname)
        if original != final:
            cumulative_tightening[fname] = (original, final)

    # Build final trace
    trace = MergeTrace(
        levels=tuple(levels),
        per_level_violations={k: tuple(v) for k, v in per_level_violations.items()},
        per_level_repairs={k: tuple(v) for k, v in per_level_repairs.items()},
        cumulative_tightening=cumulative_tightening
    )

    # Build explanation for full chain
    explanation = _build_explanation(configs[0], configs[-1], current, all_violations)

    # Fields tightened across chain
    fields_tightened = list(cumulative_tightening.keys())

    # Build final receipt
    timestamp = datetime.now(timezone.utc).isoformat()
    receipt = MergeReceipt(
        timestamp=timestamp,
        receipt_id=_compute_receipt_id(
            configs[0].provenance.config_hash,
            configs[-1].provenance.config_hash,
            timestamp
        ),
        parent_hash=configs[0].provenance.config_hash,
        child_hash=configs[-1].provenance.config_hash,
        merged_hash=current.provenance.config_hash,
        is_valid=True,
        violations_count=len(all_violations),
        repairs_count=len(all_repairs),
        fields_tightened=tuple(fields_tightened),
        patterns_removed=explanation.patterns_removed,
        author=author,
        reason=reason
    )

    if emit_receipt_flag:
        try:
            emit_receipt(receipt, receipt_path)
        except Exception as e:
            warnings.warn(f"Failed to emit merge receipt: {e}", UserWarning, stacklevel=2)

    return MergeResult(
        merged=current,
        is_valid=True,
        violations=tuple(all_violations),
        repairs_applied=tuple(all_repairs),
        explanation=explanation,
        receipt=receipt,
        trace=trace
    )


def simulate_merge(
    parent: QEDConfig,
    child: QEDConfig
) -> MergeSimulation:
    """
    Preview what merge WOULD produce without committing.

    Enables "dry run" before actual merge - see the result without
    creating receipts or modifying state.

    Args:
        parent: Parent config
        child: Child config

    Returns:
        MergeSimulation with preview of merge result
    """
    # Detect violations
    violations = _detect_violations(parent, child)

    # Compute available repairs
    repairs = _compute_repairs(violations, parent) if violations else []

    # Create preview of merged config
    merged_preview: Optional[QEDConfig] = None
    try:
        merged_preview = _create_merged_config(
            parent, child,
            repairs=repairs if violations else None,
            author="simulation",
            reason="merge simulation"
        )
    except Exception:
        pass

    # Build impact summary
    would_be_valid = len(violations) == 0 or len(repairs) > 0

    impact_parts = []
    if not violations:
        impact_parts.append("Merge would succeed with no violations")
    else:
        impact_parts.append(f"{len(violations)} violations detected")
        if repairs:
            impact_parts.append(f"{len(repairs)} repairs available")

    # Track what would change
    if merged_preview:
        changes = []
        for fname in ['recall_floor', 'max_fp_rate', 'slo_latency_ms', 'slo_breach_budget']:
            child_val = getattr(child, fname)
            merged_val = getattr(merged_preview, fname)
            if child_val != merged_val:
                changes.append(f"{fname}: {child_val} -> {merged_val}")

        parent_patterns = set(parent.enabled_patterns) if parent.enabled_patterns else set()
        child_patterns = set(child.enabled_patterns) if child.enabled_patterns else set()
        merged_patterns = set(merged_preview.enabled_patterns) if merged_preview.enabled_patterns else set()

        if parent_patterns or child_patterns:
            removed_count = len((parent_patterns | child_patterns) - merged_patterns)
            if removed_count:
                changes.append(f"{removed_count} patterns would be removed")

        if changes:
            impact_parts.append("Changes: " + "; ".join(changes))

    impact_summary = ". ".join(impact_parts)

    return MergeSimulation(
        would_be_valid=would_be_valid,
        merged_preview=merged_preview,
        violations_preview=tuple(violations),
        repairs_available=tuple(repairs),
        impact_summary=impact_summary
    )


def suggest_repairs(
    parent: QEDConfig,
    child: QEDConfig
) -> List[Repair]:
    """
    Suggest repairs for violations without applying them.

    Enables "preview mode" before committing - see what WOULD be fixed.

    Args:
        parent: Parent config
        child: Child config

    Returns:
        List of Repair objects that would fix violations
    """
    violations = _detect_violations(parent, child)
    return _compute_repairs(violations, parent)


def apply_repairs(
    child: QEDConfig,
    repairs: List[Repair],
    author: str = "merge_rules",
    reason: str = "applied repairs"
) -> QEDConfig:
    """
    Apply specific repairs to a child config.

    Takes repair suggestions and applies them, returning a new valid config.

    Args:
        child: Child config to repair
        repairs: List of repairs to apply
        author: Who applied repairs (for audit)
        reason: Why repairs were applied (for audit)

    Returns:
        New QEDConfig with repairs applied
    """
    # Build new config data from child
    config_data = child.to_dict()

    # Apply each repair
    for repair in repairs:
        if repair.field == 'recall_floor':
            config_data['recall_floor'] = repair.repaired_value
        elif repair.field == 'max_fp_rate':
            config_data['max_fp_rate'] = repair.repaired_value
        elif repair.field == 'slo_latency_ms':
            config_data['slo_latency_ms'] = repair.repaired_value
        elif repair.field == 'slo_breach_budget':
            config_data['slo_breach_budget'] = repair.repaired_value
        elif repair.field == 'enabled_patterns':
            config_data['enabled_patterns'] = list(repair.repaired_value)
        elif repair.field.startswith('regulatory_flags.'):
            flag_name = repair.field.split('.')[1]
            if 'regulatory_flags' not in config_data:
                config_data['regulatory_flags'] = {}
            config_data['regulatory_flags'][flag_name] = repair.repaired_value

    # Update provenance
    config_data['provenance'] = ConfigProvenance.create(
        author=author,
        reason=reason,
        parent_hash=child.provenance.config_hash
    ).to_dict()

    return QEDConfig.from_dict(config_data, validate=True, strict=False)


def detect_conflicts(configs: List[QEDConfig]) -> List[Conflict]:
    """
    Detect conflicts between N configs that SHOULD be mergeable.

    Pre-flight check before attempting merge chain. Identifies:
    - Patterns enabled in one config but disabled in another
    - Regulatory flags required in one but missing in another
    - Incompatible SLO budgets across configs

    Args:
        configs: List of configs to check for conflicts

    Returns:
        List of Conflict objects describing issues found
    """
    if len(configs) < 2:
        return []

    conflicts: List[Conflict] = []

    # Collect all config IDs
    config_ids = [c.deployment_id for c in configs]

    # Check pattern conflicts
    all_patterns: Dict[str, List[str]] = {}  # pattern -> list of configs that have it
    for config in configs:
        if config.enabled_patterns:
            for pattern in config.enabled_patterns:
                if pattern not in all_patterns:
                    all_patterns[pattern] = []
                all_patterns[pattern].append(config.deployment_id)

    # Find patterns not in all configs (when patterns are specified)
    configs_with_patterns = [c for c in configs if c.enabled_patterns]
    if len(configs_with_patterns) >= 2:
        for pattern, config_list in all_patterns.items():
            if len(config_list) < len(configs_with_patterns):
                missing = [c.deployment_id for c in configs_with_patterns
                          if c.deployment_id not in config_list]
                if missing:
                    conflicts.append(Conflict(
                        field='enabled_patterns',
                        configs_involved=tuple(sorted(config_list + missing)),
                        values={cid: pattern in [p for c in configs if c.deployment_id == cid
                                                for p in c.enabled_patterns]
                               for cid in config_list + missing},
                        resolution_strategy=f"Pattern '{pattern}' will be removed in merge (intersection rule)"
                    ))

    # Check regulatory flag conflicts
    all_flags: Dict[str, Dict[str, bool]] = {}  # flag -> {config_id: value}
    for config in configs:
        for flag, value in config.regulatory_flags.items():
            if flag not in all_flags:
                all_flags[flag] = {}
            all_flags[flag][config.deployment_id] = value

    for flag, values in all_flags.items():
        # Check if any config has it True and another has it False/missing
        has_true = [cid for cid, val in values.items() if val]
        has_false = [cid for cid in config_ids if cid not in values or not values.get(cid, False)]
        has_false = [cid for cid in has_false if cid in [c.deployment_id for c in configs]]

        if has_true and has_false:
            conflicts.append(Conflict(
                field=f'regulatory_flags.{flag}',
                configs_involved=tuple(sorted(has_true + has_false)),
                values={cid: values.get(cid, False) for cid in has_true + has_false},
                resolution_strategy=f"Flag '{flag}' will be True in merge (OR rule)"
            ))

    # Check SLO budget incompatibilities
    # Find configs where SLO budgets vary significantly
    latencies = [(c.deployment_id, c.slo_latency_ms) for c in configs]
    breach_budgets = [(c.deployment_id, c.slo_breach_budget) for c in configs]

    # Latency variance check
    latency_values = [l[1] for l in latencies]
    if max(latency_values) > min(latency_values) * 3:  # More than 3x variance
        conflicts.append(Conflict(
            field='slo_latency_ms',
            configs_involved=tuple(config_ids),
            values={cid: lat for cid, lat in latencies},
            resolution_strategy=f"Will use minimum ({min(latency_values)}ms)"
        ))

    # Breach budget variance check
    budget_values = [b[1] for b in breach_budgets]
    if max(budget_values) > min(budget_values) * 10:  # More than 10x variance
        conflicts.append(Conflict(
            field='slo_breach_budget',
            configs_involved=tuple(config_ids),
            values={cid: bud for cid, bud in breach_budgets},
            resolution_strategy=f"Will use minimum ({min(budget_values)})"
        ))

    return conflicts
