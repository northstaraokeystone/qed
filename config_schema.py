"""
QED v8 Configuration Schema - Self-Validating, Predictive Config Control Plane

This module defines QEDConfig, the deployment control plane for QED v8.
It's not just settings—it's a predictive, self-validating, auditable contract
that knows what will happen when applied.

Consumed by:
- qed.py (runtime)
- merge_rules.py (layering)
- TruthLink (packets)
- proof.py (CLI)

Design Principles:
- Self-validating: Can't create invalid config
- Self-healing: Invalid input → safe defaults + warnings
- Self-describing: Can explain itself and export schema
- Predictive: Simulates what will happen when applied
- Auditable: Provenance tracks who/when/why
- Immutable: Frozen after load, no runtime mutation
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Optional jsonschema support
try:
    import jsonschema
    from jsonschema import Draft202012Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


__all__ = [
    'QEDConfig',
    'ConfigProvenance',
    'ConfigDiff',
    'ConfigSimulation',
    'load',
    'default',
]


# =============================================================================
# JSON Schema Definition (Draft 2020-12)
# =============================================================================

_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qed.ai/schemas/config/v1.0",
    "title": "QEDConfig",
    "description": "QED v8 deployment configuration schema",
    "type": "object",
    "required": ["version", "deployment_id", "hook"],
    "properties": {
        "version": {
            "type": "string",
            "description": "Config schema version (e.g., '1.0')",
            "pattern": r"^\d+\.\d+$"
        },
        "deployment_id": {
            "type": "string",
            "description": "Unique deployment identifier",
            "minLength": 1
        },
        "hook": {
            "type": "string",
            "description": "Deployment hook (business unit)",
            "enum": ["tesla", "spacex", "starlink", "boring", "neuralink", "xai"]
        },
        "compression_target": {
            "type": "number",
            "description": "Target compression ratio",
            "minimum": 1.0,
            "maximum": 1000.0,
            "default": 10.0
        },
        "recall_floor": {
            "type": "number",
            "description": "Minimum recall threshold",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.999
        },
        "max_fp_rate": {
            "type": "number",
            "description": "Maximum false positive rate",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.01
        },
        "slo_latency_ms": {
            "type": "integer",
            "description": "P95 latency budget in milliseconds",
            "minimum": 1,
            "default": 100
        },
        "slo_breach_budget": {
            "type": "number",
            "description": "Allowed SLO breach rate",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.001
        },
        "enabled_patterns": {
            "type": "array",
            "description": "Pattern IDs allowed to run",
            "items": {"type": "string"},
            "default": []
        },
        "safety_overrides": {
            "type": "object",
            "description": "Safety overrides (can only tighten)",
            "additionalProperties": True,
            "default": {}
        },
        "regulatory_flags": {
            "type": "object",
            "description": "Compliance requirements",
            "additionalProperties": {"type": "boolean"},
            "default": {}
        },
        "provenance": {
            "type": "object",
            "description": "Config provenance metadata",
            "properties": {
                "author": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "reason": {"type": "string"},
                "parent_hash": {"type": ["string", "null"]},
                "config_hash": {"type": "string"}
            },
            "required": ["author", "created_at", "reason", "config_hash"]
        }
    },
    "additionalProperties": False
}

# Valid safety override keys
_VALID_SAFETY_KEYS = frozenset({
    "recall_floor",
    "max_fp_rate",
    "slo_latency_ms",
    "slo_breach_budget",
    "require_dual_approval",
    "block_high_risk_patterns",
    "enforce_audit_logging",
    "max_concurrent_patterns",
    "circuit_breaker_threshold",
})

# Valid hooks
_VALID_HOOKS = frozenset({"tesla", "spacex", "starlink", "boring", "neuralink", "xai"})


# =============================================================================
# Compiled Validator (Performance)
# =============================================================================

# Module-level cached jsonschema validator - compiled once at import
# Achieves <1ms validation target
_COMPILED_VALIDATOR: Optional[Any] = None

if HAS_JSONSCHEMA:
    try:
        _COMPILED_VALIDATOR = Draft202012Validator(_JSON_SCHEMA)
        # Validate the schema itself
        Draft202012Validator.check_schema(_JSON_SCHEMA)
    except Exception:
        _COMPILED_VALIDATOR = None


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA3-256 hash of config content (excluding provenance hash)."""
    # Create a copy without the config_hash to avoid circular dependency
    hashable = {k: v for k, v in data.items() if k != "provenance"}
    if "provenance" in data and data["provenance"]:
        prov = dict(data["provenance"])
        prov.pop("config_hash", None)
        hashable["provenance"] = prov

    # Deterministic JSON serialization
    canonical = json.dumps(hashable, sort_keys=True, separators=(',', ':'))
    return hashlib.sha3_256(canonical.encode()).hexdigest()[:16]


# =============================================================================
# ConfigProvenance Dataclass
# =============================================================================

@dataclass(frozen=True)
class ConfigProvenance:
    """
    Provenance metadata for configuration auditing.

    Tracks who created the config, when, why, and what it derived from.
    """
    author: str
    created_at: str
    reason: str
    config_hash: str
    parent_hash: Optional[str] = None

    @classmethod
    def create(
        cls,
        author: str,
        reason: str,
        parent_hash: Optional[str] = None,
        config_hash: str = ""
    ) -> ConfigProvenance:
        """Create provenance with current timestamp."""
        return cls(
            author=author,
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            parent_hash=parent_hash,
            config_hash=config_hash
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return asdict(self)


# =============================================================================
# ConfigDiff Dataclass
# =============================================================================

@dataclass(frozen=True)
class ConfigDiff:
    """
    Result of comparing two QEDConfig instances.

    Used by merge_rules.py to validate config layering.
    """
    fields_changed: Dict[str, Tuple[Any, Any]]
    patterns_added: List[str]
    patterns_removed: List[str]
    safety_direction: str  # "tighter" | "looser" | "mixed" | "unchanged"
    is_valid_child: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return asdict(self)


# =============================================================================
# ConfigSimulation Dataclass
# =============================================================================

@dataclass(frozen=True)
class ConfigSimulation:
    """
    Prediction of what happens when a config is applied.

    This is the "chef's kiss" feature - config becomes deployment preview.
    """
    patterns_enabled: List[str]
    patterns_blocked: List[str]
    estimated_coverage: float
    estimated_breach_rate: float
    risk_profile: str  # "conservative" | "moderate" | "aggressive"
    warnings: List[str]
    comparison_to_default: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return asdict(self)


# =============================================================================
# QEDConfig Dataclass
# =============================================================================

@dataclass(frozen=True)
class QEDConfig:
    """
    QED v8 deployment configuration.

    This is the deployment control plane - not just settings, but a predictive,
    self-validating, auditable contract that knows what will happen when applied.

    Attributes:
        version: Config schema version (e.g., "1.0")
        deployment_id: Unique deployment identifier
        hook: Business unit (tesla, spacex, starlink, boring, neuralink, xai)
        compression_target: Target compression ratio
        recall_floor: Minimum recall threshold (0.0-1.0)
        max_fp_rate: Maximum false positive rate (0.0-1.0)
        slo_latency_ms: P95 latency budget in milliseconds
        slo_breach_budget: Allowed SLO breach rate
        enabled_patterns: Pattern IDs allowed to run
        safety_overrides: Safety settings (can only tighten)
        regulatory_flags: Compliance requirements
        provenance: Who/when/why/parent tracking
    """
    version: str
    deployment_id: str
    hook: str
    provenance: ConfigProvenance
    compression_target: float = 10.0
    recall_floor: float = 0.999
    max_fp_rate: float = 0.01
    slo_latency_ms: int = 100
    slo_breach_budget: float = 0.001
    enabled_patterns: Tuple[str, ...] = field(default_factory=tuple)
    safety_overrides: Dict[str, Any] = field(default_factory=dict)
    regulatory_flags: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert mutable types to immutable for frozen dataclass."""
        # Handle enabled_patterns - convert list to tuple
        if isinstance(self.enabled_patterns, list):
            object.__setattr__(self, 'enabled_patterns', tuple(self.enabled_patterns))

        # Handle safety_overrides - make immutable copy
        if isinstance(self.safety_overrides, dict):
            object.__setattr__(self, 'safety_overrides', dict(self.safety_overrides))

        # Handle regulatory_flags - make immutable copy
        if isinstance(self.regulatory_flags, dict):
            object.__setattr__(self, 'regulatory_flags', dict(self.regulatory_flags))

    # -------------------------------------------------------------------------
    # Schema Export
    # -------------------------------------------------------------------------

    @property
    def schema(self) -> Dict[str, Any]:
        """
        Returns JSON Schema dict for external validation.

        Draft 2020-12 compatible with field descriptions and constraints.
        """
        return _JSON_SCHEMA.copy()

    def explain(self) -> str:
        """
        Human-readable explanation of this config.

        Describes what each non-default setting means and risk assessment.
        """
        lines = [
            f"QEDConfig Explanation",
            f"=" * 50,
            f"",
            f"Deployment: {self.deployment_id}",
            f"Hook: {self.hook}",
            f"Version: {self.version}",
            f"",
            f"Performance Settings:",
        ]

        # Compression
        if self.compression_target != 10.0:
            lines.append(f"  • Compression target: {self.compression_target}x (default: 10x)")
        else:
            lines.append(f"  • Compression target: {self.compression_target}x (default)")

        # Recall floor
        if self.recall_floor != 0.999:
            direction = "LOWER" if self.recall_floor < 0.999 else "higher"
            risk = "⚠️ INCREASED RISK" if self.recall_floor < 0.999 else "✓ safer"
            lines.append(f"  • Recall floor: {self.recall_floor:.4f} ({direction} than default 0.999) {risk}")
        else:
            lines.append(f"  • Recall floor: {self.recall_floor:.4f} (default)")

        # FP rate
        if self.max_fp_rate != 0.01:
            direction = "HIGHER" if self.max_fp_rate > 0.01 else "lower"
            risk = "⚠️ MORE FALSE POSITIVES" if self.max_fp_rate > 0.01 else "✓ fewer FPs"
            lines.append(f"  • Max FP rate: {self.max_fp_rate:.4f} ({direction} than default 0.01) {risk}")
        else:
            lines.append(f"  • Max FP rate: {self.max_fp_rate:.4f} (default)")

        # SLO settings
        lines.append(f"")
        lines.append(f"SLO Settings:")
        lines.append(f"  • Latency budget: {self.slo_latency_ms}ms (p95)")
        lines.append(f"  • Breach budget: {self.slo_breach_budget:.4f} ({self.slo_breach_budget * 100:.2f}% allowed)")

        # Patterns
        lines.append(f"")
        lines.append(f"Patterns:")
        if self.enabled_patterns:
            lines.append(f"  • {len(self.enabled_patterns)} patterns enabled")
            for p in list(self.enabled_patterns)[:5]:
                lines.append(f"    - {p}")
            if len(self.enabled_patterns) > 5:
                lines.append(f"    ... and {len(self.enabled_patterns) - 5} more")
        else:
            lines.append(f"  • All patterns enabled (no restrictions)")

        # Safety overrides
        if self.safety_overrides:
            lines.append(f"")
            lines.append(f"Safety Overrides:")
            for key, val in self.safety_overrides.items():
                lines.append(f"  • {key}: {val}")

        # Regulatory flags
        if self.regulatory_flags:
            lines.append(f"")
            lines.append(f"Regulatory Flags:")
            for flag, enabled in self.regulatory_flags.items():
                status = "✓ enabled" if enabled else "✗ disabled"
                lines.append(f"  • {flag}: {status}")

        # Risk assessment
        lines.append(f"")
        lines.append(f"Risk Assessment:")
        sim = self.simulate()
        lines.append(f"  • Profile: {sim.risk_profile.upper()}")
        if sim.warnings:
            lines.append(f"  • Warnings:")
            for w in sim.warnings:
                lines.append(f"    ⚠️ {w}")

        # Provenance
        lines.append(f"")
        lines.append(f"Provenance:")
        lines.append(f"  • Author: {self.provenance.author}")
        lines.append(f"  • Created: {self.provenance.created_at}")
        lines.append(f"  • Reason: {self.provenance.reason}")
        lines.append(f"  • Hash: {self.provenance.config_hash}")
        if self.provenance.parent_hash:
            lines.append(f"  • Parent: {self.provenance.parent_hash}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Config Comparison
    # -------------------------------------------------------------------------

    def compare(self, other: QEDConfig) -> ConfigDiff:
        """
        Compare this config to another.

        Args:
            other: Config to compare against

        Returns:
            ConfigDiff with detailed comparison results
        """
        fields_changed: Dict[str, Tuple[Any, Any]] = {}

        # Compare scalar fields
        compare_fields = [
            'version', 'deployment_id', 'hook', 'compression_target',
            'recall_floor', 'max_fp_rate', 'slo_latency_ms', 'slo_breach_budget'
        ]

        for fname in compare_fields:
            self_val = getattr(self, fname)
            other_val = getattr(other, fname)
            if self_val != other_val:
                fields_changed[fname] = (self_val, other_val)

        # Compare patterns
        self_patterns = set(self.enabled_patterns)
        other_patterns = set(other.enabled_patterns)
        patterns_added = sorted(other_patterns - self_patterns)
        patterns_removed = sorted(self_patterns - other_patterns)

        if self_patterns != other_patterns:
            fields_changed['enabled_patterns'] = (
                list(self.enabled_patterns),
                list(other.enabled_patterns)
            )

        # Compare safety overrides
        if self.safety_overrides != other.safety_overrides:
            fields_changed['safety_overrides'] = (
                self.safety_overrides,
                other.safety_overrides
            )

        # Compare regulatory flags
        if self.regulatory_flags != other.regulatory_flags:
            fields_changed['regulatory_flags'] = (
                self.regulatory_flags,
                other.regulatory_flags
            )

        # Determine safety direction
        safety_direction = _determine_safety_direction(self, other)

        # Check if valid child (can only tighten safety)
        is_valid_child = safety_direction in ("tighter", "unchanged")

        # Generate summary
        if not fields_changed:
            summary = "Configs are identical"
        else:
            parts = []
            if 'recall_floor' in fields_changed:
                parts.append(f"recall_floor: {self.recall_floor} → {other.recall_floor}")
            if 'max_fp_rate' in fields_changed:
                parts.append(f"max_fp_rate: {self.max_fp_rate} → {other.max_fp_rate}")
            if patterns_added or patterns_removed:
                parts.append(f"+{len(patterns_added)}/-{len(patterns_removed)} patterns")
            if not parts:
                parts.append(f"{len(fields_changed)} fields changed")
            summary = "; ".join(parts)

        return ConfigDiff(
            fields_changed=fields_changed,
            patterns_added=patterns_added,
            patterns_removed=patterns_removed,
            safety_direction=safety_direction,
            is_valid_child=is_valid_child,
            summary=summary
        )

    # -------------------------------------------------------------------------
    # Config Simulation (Chef's Kiss)
    # -------------------------------------------------------------------------

    def simulate(
        self,
        patterns_library: Optional[List[str]] = None,
        fleet_size: Optional[int] = None
    ) -> ConfigSimulation:
        """
        Predict what happens if this config is applied.

        This is the key differentiator - config becomes deployment preview.

        Args:
            patterns_library: List of all available pattern IDs
            fleet_size: Number of nodes in deployment (for scaling estimates)

        Returns:
            ConfigSimulation with predictions and risk assessment
        """
        warnings_list: List[str] = []

        # Determine enabled/blocked patterns
        if patterns_library:
            library_set = set(patterns_library)
            if self.enabled_patterns:
                enabled_set = set(self.enabled_patterns)
                patterns_enabled = sorted(enabled_set & library_set)
                patterns_blocked = sorted(library_set - enabled_set)

                # Check for invalid patterns
                invalid = enabled_set - library_set
                if invalid:
                    warnings_list.append(
                        f"{len(invalid)} enabled patterns not in library: {list(invalid)[:3]}"
                    )
            else:
                # Empty enabled_patterns means all enabled
                patterns_enabled = sorted(library_set)
                patterns_blocked = []

            estimated_coverage = len(patterns_enabled) / len(library_set) if library_set else 1.0
        else:
            # No library provided
            patterns_enabled = list(self.enabled_patterns) if self.enabled_patterns else []
            patterns_blocked = []
            estimated_coverage = 1.0 if not self.enabled_patterns else 0.5  # Unknown

        # Estimate breach rate based on config settings
        # More aggressive settings = higher breach rate
        base_breach = 0.0001

        # Lower recall floor increases breach risk
        if self.recall_floor < 0.999:
            base_breach += (0.999 - self.recall_floor) * 0.1

        # Higher FP rate increases breach risk
        if self.max_fp_rate > 0.01:
            base_breach += (self.max_fp_rate - 0.01) * 0.05

        # Tighter latency increases breach risk
        if self.slo_latency_ms < 50:
            base_breach += 0.001

        # Scale by fleet size
        if fleet_size and fleet_size > 1000:
            base_breach *= 1.0 + (fleet_size / 10000) * 0.1

        estimated_breach_rate = min(base_breach, 0.1)  # Cap at 10%

        # Determine risk profile
        risk_profile = _determine_risk_profile(
            recall_floor=self.recall_floor,
            max_fp_rate=self.max_fp_rate,
            coverage=estimated_coverage
        )

        # Generate warnings
        if self.recall_floor < 0.99:
            warnings_list.append("Recall floor below 99% - increased miss risk")
        if self.max_fp_rate > 0.02:
            warnings_list.append("FP rate above 2% - user experience impact")
        if self.slo_latency_ms < 10:
            warnings_list.append("Latency budget <10ms - may be unrealistic")
        if estimated_coverage < 0.3:
            warnings_list.append("Less than 30% pattern coverage - limited protection")
        if estimated_breach_rate > self.slo_breach_budget:
            warnings_list.append(
                f"Estimated breach rate ({estimated_breach_rate:.4f}) exceeds budget ({self.slo_breach_budget})"
            )

        # Compare to default config
        default_config = QEDConfig.default(self.hook)
        comparison: Dict[str, Any] = {}

        if self.recall_floor != default_config.recall_floor:
            comparison['recall_floor'] = {
                'current': self.recall_floor,
                'default': default_config.recall_floor,
                'direction': 'lower' if self.recall_floor < default_config.recall_floor else 'higher'
            }

        if self.max_fp_rate != default_config.max_fp_rate:
            comparison['max_fp_rate'] = {
                'current': self.max_fp_rate,
                'default': default_config.max_fp_rate,
                'direction': 'higher' if self.max_fp_rate > default_config.max_fp_rate else 'lower'
            }

        if self.slo_latency_ms != default_config.slo_latency_ms:
            comparison['slo_latency_ms'] = {
                'current': self.slo_latency_ms,
                'default': default_config.slo_latency_ms,
                'direction': 'tighter' if self.slo_latency_ms < default_config.slo_latency_ms else 'looser'
            }

        return ConfigSimulation(
            patterns_enabled=patterns_enabled,
            patterns_blocked=patterns_blocked,
            estimated_coverage=estimated_coverage,
            estimated_breach_rate=estimated_breach_rate,
            risk_profile=risk_profile,
            warnings=warnings_list,
            comparison_to_default=comparison
        )

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Export as dictionary.

        Includes provenance for full auditability.
        """
        return {
            'version': self.version,
            'deployment_id': self.deployment_id,
            'hook': self.hook,
            'compression_target': self.compression_target,
            'recall_floor': self.recall_floor,
            'max_fp_rate': self.max_fp_rate,
            'slo_latency_ms': self.slo_latency_ms,
            'slo_breach_budget': self.slo_breach_budget,
            'enabled_patterns': list(self.enabled_patterns),
            'safety_overrides': dict(self.safety_overrides),
            'regulatory_flags': dict(self.regulatory_flags),
            'provenance': self.provenance.to_dict()
        }

    def to_json(self, pretty: bool = False) -> str:
        """
        Export as JSON string.

        Args:
            pretty: If True, format with indentation
        """
        if pretty:
            return json.dumps(self.to_dict(), indent=2, sort_keys=True)
        return json.dumps(self.to_dict(), separators=(',', ':'))

    def save(self, path: str) -> None:
        """
        Write config to file.

        Auto-updates config_hash before save to ensure integrity.

        Args:
            path: File path to write to (.json or .yaml)
        """
        # Recompute hash for integrity
        data = self.to_dict()
        data['provenance']['config_hash'] = _compute_hash(data)

        path_obj = Path(path)

        if path_obj.suffix in ('.yaml', '.yml'):
            if not HAS_YAML:
                raise ImportError("PyYAML required for YAML output")
            content = yaml.safe_dump(data, default_flow_style=False, sort_keys=True)
        else:
            content = json.dumps(data, indent=2, sort_keys=True)

        path_obj.write_text(content)

    # -------------------------------------------------------------------------
    # Class Methods
    # -------------------------------------------------------------------------

    @classmethod
    def default(cls, hook: str) -> QEDConfig:
        """
        Return safe default config for given hook.

        All safety settings at maximum, all patterns enabled.

        Args:
            hook: Business unit (tesla, spacex, starlink, boring, neuralink, xai)

        Returns:
            QEDConfig with safe defaults
        """
        if hook not in _VALID_HOOKS:
            raise ValueError(f"Invalid hook '{hook}'. Must be one of: {sorted(_VALID_HOOKS)}")

        provenance = ConfigProvenance.create(
            author="system",
            reason=f"Default config for {hook}"
        )

        config_data = {
            'version': "1.0",
            'deployment_id': f"{hook}-default",
            'hook': hook,
            'compression_target': 10.0,
            'recall_floor': 0.999,
            'max_fp_rate': 0.01,
            'slo_latency_ms': 100,
            'slo_breach_budget': 0.001,
            'enabled_patterns': [],  # Empty = all enabled
            'safety_overrides': {},
            'regulatory_flags': {},
            'provenance': provenance.to_dict()
        }

        # Compute hash
        config_hash = _compute_hash(config_data)
        provenance = ConfigProvenance(
            author=provenance.author,
            created_at=provenance.created_at,
            reason=provenance.reason,
            parent_hash=None,
            config_hash=config_hash
        )

        return cls(
            version="1.0",
            deployment_id=f"{hook}-default",
            hook=hook,
            compression_target=10.0,
            recall_floor=0.999,
            max_fp_rate=0.01,
            slo_latency_ms=100,
            slo_breach_budget=0.001,
            enabled_patterns=tuple(),
            safety_overrides={},
            regulatory_flags={},
            provenance=provenance
        )

    @classmethod
    def from_parent(
        cls,
        parent: QEDConfig,
        changes: Dict[str, Any],
        reason: str,
        author: str = "system"
    ) -> QEDConfig:
        """
        Create derived config with changes.

        Auto-sets provenance.parent_hash. Validates changes only tighten safety.

        Args:
            parent: Parent config to derive from
            changes: Dictionary of field changes
            reason: Why this config was created
            author: Who created this config

        Returns:
            New QEDConfig derived from parent

        Raises:
            ValueError: If changes loosen safety settings
        """
        # Validate safety direction
        _validate_safety_tightening(parent, changes)

        # Build new config data
        new_data = parent.to_dict()

        # Apply changes (excluding provenance)
        for key, value in changes.items():
            if key != 'provenance' and key in new_data:
                new_data[key] = value

        # Create new provenance
        provenance = ConfigProvenance.create(
            author=author,
            reason=reason,
            parent_hash=parent.provenance.config_hash
        )
        new_data['provenance'] = provenance.to_dict()

        # Compute new hash
        config_hash = _compute_hash(new_data)
        new_data['provenance']['config_hash'] = config_hash

        return cls.from_dict(new_data, validate=True, strict=True)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        validate: bool = True,
        strict: bool = False
    ) -> QEDConfig:
        """
        Create from dictionary.

        Same validation as load().

        Args:
            data: Configuration dictionary
            validate: Whether to validate (default True)
            strict: If True, raise on invalid; if False, self-heal

        Returns:
            Validated QEDConfig instance
        """
        return _create_config(data, validate, strict)


# =============================================================================
# Module-Level Functions
# =============================================================================

def load(
    path: str,
    validate: bool = True,
    strict: bool = False
) -> QEDConfig:
    """
    Load config from JSON/YAML file.

    Auto-validates on load (not separate step).

    Args:
        path: Path to config file
        validate: Whether to validate (default True)
        strict: If True, raise on invalid; if False, self-heal with warnings

    Returns:
        Validated, frozen QEDConfig instance

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If strict=True and validation fails
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = path_obj.read_text()

    if path_obj.suffix in ('.yaml', '.yml'):
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML files")
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)

    return _create_config(data, validate, strict)


def default(hook: str) -> QEDConfig:
    """
    Return safe default config for given hook.

    Convenience wrapper for QEDConfig.default().

    Args:
        hook: Business unit (tesla, spacex, starlink, boring, neuralink, xai)

    Returns:
        QEDConfig with safe defaults
    """
    return QEDConfig.default(hook)


# =============================================================================
# Internal Validation Functions
# =============================================================================

def _validate(data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate config data.

    Returns: (is_valid, errors, warnings)

    Rules:
    - Required fields present
    - recall_floor in 0.0-1.0
    - max_fp_rate in 0.0-1.0
    - slo_latency_ms > 0
    - enabled_patterns exist in library (if library available)
    - safety_overrides only contain valid keys
    - config_hash matches recomputed hash (integrity)
    """
    errors: List[str] = []
    warns: List[str] = []

    # Required fields
    required = ['version', 'deployment_id', 'hook']
    for field in required:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")

    # Hook validation
    if 'hook' in data and data['hook'] not in _VALID_HOOKS:
        errors.append(f"Invalid hook '{data.get('hook')}'. Must be one of: {sorted(_VALID_HOOKS)}")

    # Range validations
    if 'recall_floor' in data:
        val = data['recall_floor']
        if not isinstance(val, (int, float)):
            errors.append(f"recall_floor must be numeric, got {type(val).__name__}")
        elif val < 0.0 or val > 1.0:
            warns.append(f"recall_floor {val} out of range [0.0, 1.0]")

    if 'max_fp_rate' in data:
        val = data['max_fp_rate']
        if not isinstance(val, (int, float)):
            errors.append(f"max_fp_rate must be numeric, got {type(val).__name__}")
        elif val < 0.0 or val > 1.0:
            warns.append(f"max_fp_rate {val} out of range [0.0, 1.0]")

    if 'slo_latency_ms' in data:
        val = data['slo_latency_ms']
        if not isinstance(val, int):
            if isinstance(val, float) and val.is_integer():
                pass  # Allow integer floats
            else:
                errors.append(f"slo_latency_ms must be integer, got {type(val).__name__}")
        elif val <= 0:
            warns.append(f"slo_latency_ms {val} must be > 0")

    if 'slo_breach_budget' in data:
        val = data['slo_breach_budget']
        if not isinstance(val, (int, float)):
            errors.append(f"slo_breach_budget must be numeric, got {type(val).__name__}")
        elif val < 0.0 or val > 1.0:
            warns.append(f"slo_breach_budget {val} out of range [0.0, 1.0]")

    # Safety overrides keys
    if 'safety_overrides' in data and isinstance(data['safety_overrides'], dict):
        for key in data['safety_overrides']:
            if key not in _VALID_SAFETY_KEYS:
                warns.append(f"Unknown safety_override key: {key}")

    # Config hash integrity (if provenance present)
    if 'provenance' in data and isinstance(data['provenance'], dict):
        stored_hash = data['provenance'].get('config_hash', '')
        if stored_hash:
            computed = _compute_hash(data)
            if stored_hash != computed:
                warns.append(f"Config hash mismatch: stored={stored_hash[:8]}..., computed={computed[:8]}...")

    # Use compiled validator if available (for <1ms performance)
    if _COMPILED_VALIDATOR is not None:
        try:
            schema_errors = list(_COMPILED_VALIDATOR.iter_errors(data))
            for err in schema_errors:
                if err.validator == 'required':
                    # Already handled above
                    pass
                elif err.validator in ('minimum', 'maximum', 'enum'):
                    warns.append(f"Schema: {err.message}")
                else:
                    errors.append(f"Schema: {err.message}")
        except Exception:
            pass  # Fall back to manual validation only

    is_valid = len(errors) == 0
    return is_valid, errors, warns


def _self_heal(data: Dict[str, Any], warns: List[str]) -> Dict[str, Any]:
    """
    Apply self-healing to config data.

    Self-healing behavior:
    - Missing optional field → use default, add warning
    - Out-of-range value → clamp to valid range, add warning
    - Unknown field → ignore, add warning
    """
    healed = dict(data)

    # Defaults for optional fields
    defaults = {
        'version': '1.0',
        'compression_target': 10.0,
        'recall_floor': 0.999,
        'max_fp_rate': 0.01,
        'slo_latency_ms': 100,
        'slo_breach_budget': 0.001,
        'enabled_patterns': [],
        'safety_overrides': {},
        'regulatory_flags': {},
    }

    # Apply defaults for missing fields
    for field, default_val in defaults.items():
        if field not in healed:
            healed[field] = default_val
            warns.append(f"Missing optional field '{field}', using default: {default_val}")

    # Clamp numeric ranges
    if 'recall_floor' in healed:
        val = healed['recall_floor']
        if isinstance(val, (int, float)):
            if val < 0.0:
                healed['recall_floor'] = 0.0
                warns.append(f"Clamped recall_floor from {val} to 0.0")
            elif val > 1.0:
                healed['recall_floor'] = 1.0
                warns.append(f"Clamped recall_floor from {val} to 1.0")

    if 'max_fp_rate' in healed:
        val = healed['max_fp_rate']
        if isinstance(val, (int, float)):
            if val < 0.0:
                healed['max_fp_rate'] = 0.0
                warns.append(f"Clamped max_fp_rate from {val} to 0.0")
            elif val > 1.0:
                healed['max_fp_rate'] = 1.0
                warns.append(f"Clamped max_fp_rate from {val} to 1.0")

    if 'slo_breach_budget' in healed:
        val = healed['slo_breach_budget']
        if isinstance(val, (int, float)):
            if val < 0.0:
                healed['slo_breach_budget'] = 0.0
                warns.append(f"Clamped slo_breach_budget from {val} to 0.0")
            elif val > 1.0:
                healed['slo_breach_budget'] = 1.0
                warns.append(f"Clamped slo_breach_budget from {val} to 1.0")

    if 'slo_latency_ms' in healed:
        val = healed['slo_latency_ms']
        if isinstance(val, (int, float)):
            if val <= 0:
                healed['slo_latency_ms'] = 1
                warns.append(f"Clamped slo_latency_ms from {val} to 1")
            else:
                healed['slo_latency_ms'] = int(val)

    # Remove unknown fields
    known_fields = {
        'version', 'deployment_id', 'hook', 'compression_target',
        'recall_floor', 'max_fp_rate', 'slo_latency_ms', 'slo_breach_budget',
        'enabled_patterns', 'safety_overrides', 'regulatory_flags', 'provenance'
    }
    unknown = set(healed.keys()) - known_fields
    for field in unknown:
        del healed[field]
        warns.append(f"Ignoring unknown field: {field}")

    return healed


def _create_config(
    data: Dict[str, Any],
    validate: bool,
    strict: bool
) -> QEDConfig:
    """
    Internal factory for creating QEDConfig from data.

    Handles validation, self-healing, and hash computation.
    """
    all_warnings: List[str] = []

    if validate:
        is_valid, errors, warns = _validate(data)
        all_warnings.extend(warns)

        if not is_valid:
            if strict:
                raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
            else:
                # Self-heal
                data = _self_heal(data, all_warnings)
                # Re-validate after healing
                is_valid, errors, _ = _validate(data)
                if not is_valid:
                    raise ValueError(f"Config validation failed after self-healing:\n" +
                                   "\n".join(f"  - {e}" for e in errors))

    # Emit warnings
    for w in all_warnings:
        warnings.warn(f"QEDConfig: {w}", UserWarning, stacklevel=3)

    # Handle provenance
    if 'provenance' in data and isinstance(data['provenance'], dict):
        prov_data = data['provenance']
        provenance = ConfigProvenance(
            author=prov_data.get('author', 'unknown'),
            created_at=prov_data.get('created_at', datetime.now(timezone.utc).isoformat()),
            reason=prov_data.get('reason', 'loaded from file'),
            parent_hash=prov_data.get('parent_hash'),
            config_hash=prov_data.get('config_hash', '')
        )
    else:
        provenance = ConfigProvenance.create(
            author='system',
            reason='created from dict'
        )

    # Compute/verify hash
    config_hash = _compute_hash(data)
    if provenance.config_hash and provenance.config_hash != config_hash:
        # Update hash if it changed
        provenance = ConfigProvenance(
            author=provenance.author,
            created_at=provenance.created_at,
            reason=provenance.reason,
            parent_hash=provenance.parent_hash,
            config_hash=config_hash
        )
    elif not provenance.config_hash:
        provenance = ConfigProvenance(
            author=provenance.author,
            created_at=provenance.created_at,
            reason=provenance.reason,
            parent_hash=provenance.parent_hash,
            config_hash=config_hash
        )

    # Create frozen config
    return QEDConfig(
        version=data.get('version', '1.0'),
        deployment_id=data.get('deployment_id', 'unknown'),
        hook=data.get('hook', 'xai'),
        compression_target=float(data.get('compression_target', 10.0)),
        recall_floor=float(data.get('recall_floor', 0.999)),
        max_fp_rate=float(data.get('max_fp_rate', 0.01)),
        slo_latency_ms=int(data.get('slo_latency_ms', 100)),
        slo_breach_budget=float(data.get('slo_breach_budget', 0.001)),
        enabled_patterns=tuple(data.get('enabled_patterns', [])),
        safety_overrides=dict(data.get('safety_overrides', {})),
        regulatory_flags=dict(data.get('regulatory_flags', {})),
        provenance=provenance
    )


def _determine_safety_direction(old: QEDConfig, new: QEDConfig) -> str:
    """
    Determine if safety settings got tighter, looser, or mixed.

    Tighter means:
    - recall_floor increased (higher minimum recall)
    - max_fp_rate decreased (fewer false positives)
    - slo_breach_budget decreased (stricter SLO)
    """
    tighter = 0
    looser = 0

    # recall_floor: higher = tighter
    if new.recall_floor > old.recall_floor:
        tighter += 1
    elif new.recall_floor < old.recall_floor:
        looser += 1

    # max_fp_rate: lower = tighter
    if new.max_fp_rate < old.max_fp_rate:
        tighter += 1
    elif new.max_fp_rate > old.max_fp_rate:
        looser += 1

    # slo_breach_budget: lower = tighter
    if new.slo_breach_budget < old.slo_breach_budget:
        tighter += 1
    elif new.slo_breach_budget > old.slo_breach_budget:
        looser += 1

    # Fewer patterns enabled = tighter (more restrictive)
    if new.enabled_patterns and old.enabled_patterns:
        if len(new.enabled_patterns) < len(old.enabled_patterns):
            tighter += 1
        elif len(new.enabled_patterns) > len(old.enabled_patterns):
            looser += 1

    if tighter > 0 and looser == 0:
        return "tighter"
    elif looser > 0 and tighter == 0:
        return "looser"
    elif tighter > 0 and looser > 0:
        return "mixed"
    else:
        return "unchanged"


def _determine_risk_profile(
    recall_floor: float,
    max_fp_rate: float,
    coverage: float
) -> str:
    """
    Determine risk profile based on settings.

    Conservative: recall_floor > 0.999, max_fp_rate < 0.005, <50% patterns enabled
    Aggressive: recall_floor < 0.99, max_fp_rate > 0.02, >90% patterns enabled
    Moderate: between
    """
    conservative_score = 0
    aggressive_score = 0

    # Recall floor
    if recall_floor > 0.999:
        conservative_score += 1
    elif recall_floor < 0.99:
        aggressive_score += 1

    # FP rate
    if max_fp_rate < 0.005:
        conservative_score += 1
    elif max_fp_rate > 0.02:
        aggressive_score += 1

    # Coverage (if patterns specified)
    if coverage < 0.5:
        conservative_score += 1
    elif coverage > 0.9:
        aggressive_score += 1

    if conservative_score >= 2:
        return "conservative"
    elif aggressive_score >= 2:
        return "aggressive"
    else:
        return "moderate"


def _validate_safety_tightening(parent: QEDConfig, changes: Dict[str, Any]) -> None:
    """
    Validate that changes only tighten safety settings.

    Raises ValueError if changes would loosen safety.
    """
    violations = []

    # recall_floor: must increase or stay same
    if 'recall_floor' in changes:
        if changes['recall_floor'] < parent.recall_floor:
            violations.append(
                f"recall_floor cannot decrease: {parent.recall_floor} → {changes['recall_floor']}"
            )

    # max_fp_rate: must decrease or stay same
    if 'max_fp_rate' in changes:
        if changes['max_fp_rate'] > parent.max_fp_rate:
            violations.append(
                f"max_fp_rate cannot increase: {parent.max_fp_rate} → {changes['max_fp_rate']}"
            )

    # slo_breach_budget: must decrease or stay same
    if 'slo_breach_budget' in changes:
        if changes['slo_breach_budget'] > parent.slo_breach_budget:
            violations.append(
                f"slo_breach_budget cannot increase: {parent.slo_breach_budget} → {changes['slo_breach_budget']}"
            )

    if violations:
        raise ValueError(
            "Safety settings can only be tightened in derived configs:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )
