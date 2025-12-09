"""
decision_packet.py - Core audit artifact for QED v8

DecisionPacket is the foundational data structure for TruthLink, mesh_view_v3,
portfolio binder (v9), and meta layer (v10). It is:
- Self-validating: packet_id = hash of contents, integrity verifiable
- Self-describing: carries its own schema and generates narratives
- Diffable: structured comparison between packets
- Chainable: parent_packet_id enables audit lineage

Consumers:
- TruthLink: Builds packets from raw components
- mesh_view_v3: Uses packets as graph nodes
- proof.py CLI: Pretty-print, diff, validation
- Portfolio binder (v9): Aggregates across deployments
- Meta layer (v10): Pattern recommendations based on packet history
- Compliance/audit: Human-readable proof trails
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# PatternSummary: Lightweight pattern reference embedded in packets
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PatternSummary:
    """
    Lightweight reference to a pattern used in a deployment.

    Captures essential metrics for audit without full pattern definition.
    """
    pattern_id: str
    validation_recall: float
    false_positive_rate: float
    dollar_value_annual: float
    exploit_grade: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with normalized types for consistent hashing."""
        return {
            "pattern_id": self.pattern_id,
            "validation_recall": float(self.validation_recall),
            "false_positive_rate": float(self.false_positive_rate),
            "dollar_value_annual": float(self.dollar_value_annual),
            "exploit_grade": bool(self.exploit_grade),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PatternSummary:
        """Deserialize from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            validation_recall=float(data["validation_recall"]),
            false_positive_rate=float(data["false_positive_rate"]),
            dollar_value_annual=float(data["dollar_value_annual"]),
            exploit_grade=bool(data["exploit_grade"]),
        )


# -----------------------------------------------------------------------------
# PacketMetrics: Deployment-level aggregate metrics
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PacketMetrics:
    """
    Deployment-level metrics aggregated from receipts.

    Captures volume, compression, savings, and SLO performance.
    """
    window_volume: int
    avg_compression: float
    annual_savings: float
    slo_breach_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with normalized types for consistent hashing."""
        return {
            "window_volume": int(self.window_volume),
            "avg_compression": float(self.avg_compression),
            "annual_savings": float(self.annual_savings),
            "slo_breach_rate": float(self.slo_breach_rate),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PacketMetrics:
        """Deserialize from dictionary."""
        return cls(
            window_volume=int(data["window_volume"]),
            avg_compression=float(data["avg_compression"]),
            annual_savings=float(data["annual_savings"]),
            slo_breach_rate=float(data["slo_breach_rate"]),
        )


# -----------------------------------------------------------------------------
# DecisionPacket: Core audit artifact
# -----------------------------------------------------------------------------
@dataclass
class DecisionPacket:
    """
    Core audit artifact for QED v8 deployments.

    DecisionPacket bundles all deployment artifacts into a single, verifiable,
    chainable unit. It is self-validating (packet_id = hash of contents),
    self-describing (generates human-readable narratives), and diffable
    (structured comparison between packets).

    Attributes:
        packet_id: SHA3-256 hash (16 chars) of packet contents. Auto-generated
            in __post_init__ if not provided.
        deployment_id: Unique identifier for this deployment.
        timestamp: ISO UTC timestamp. Auto-set to now() if not provided.
        manifest_ref: Reference to deployment manifest (path or hash).
        sampled_receipts: List of receipt IDs included in this packet.
        clarity_audit_ref: Reference to ClarityClean audit output.
        edge_lab_summary: Summary dict from edge_lab_v2 simulation.
        pattern_usage: List of PatternSummary for patterns used.
        metrics: PacketMetrics with aggregate deployment metrics.
        exploit_coverage: Fraction of patterns that are exploit-grade.
            Auto-computed from pattern_usage if not provided.
        signature: Optional cryptographic signature for packet verification.
        parent_packet_id: Optional link to previous packet for lineage chain.
    """
    deployment_id: str
    manifest_ref: str
    sampled_receipts: List[str]
    clarity_audit_ref: str
    edge_lab_summary: Dict[str, Any]
    pattern_usage: List[PatternSummary]
    metrics: PacketMetrics

    # Auto-generated fields
    packet_id: str = field(default="")
    timestamp: str = field(default="")
    exploit_coverage: float = field(default=-1.0)

    # Optional fields
    signature: Optional[str] = field(default=None)
    parent_packet_id: Optional[str] = field(default=None)

    # Internal state for integrity tracking
    _content_hash: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        """
        Auto-generate packet_id, timestamp, and exploit_coverage if not provided.

        Validates packet state and computes derived fields.

        Raises:
            ValueError: If required fields are invalid.
        """
        # Validate required fields
        if not self.deployment_id:
            raise ValueError("deployment_id is required")
        if not self.manifest_ref:
            raise ValueError("manifest_ref is required")
        if self.metrics is None:
            raise ValueError("metrics is required")

        # Auto-set timestamp if not provided
        if not self.timestamp:
            object.__setattr__(
                self,
                "timestamp",
                datetime.now(timezone.utc).isoformat()
            )

        # Auto-compute exploit_coverage if not provided
        if self.exploit_coverage < 0:
            if self.pattern_usage:
                exploit_count = sum(1 for p in self.pattern_usage if p.exploit_grade)
                coverage = exploit_count / len(self.pattern_usage)
            else:
                coverage = 0.0
            object.__setattr__(self, "exploit_coverage", coverage)

        # Generate content hash (used for packet_id and integrity checking)
        content_hash = self._compute_content_hash()
        object.__setattr__(self, "_content_hash", content_hash)

        # Auto-generate packet_id if not provided
        if not self.packet_id:
            object.__setattr__(self, "packet_id", content_hash)

    def _compute_content_hash(self) -> str:
        """
        Compute SHA3-256 hash of packet contents.

        Uses deterministic JSON serialization for reproducibility.

        Returns:
            16-character hex hash string.
        """
        # Build content dict excluding auto-generated and optional fields
        # Note: All numeric values must be normalized to ensure consistent hashing
        content = {
            "deployment_id": self.deployment_id,
            "timestamp": self.timestamp,
            "manifest_ref": self.manifest_ref,
            "sampled_receipts": sorted(self.sampled_receipts),
            "clarity_audit_ref": self.clarity_audit_ref,
            "edge_lab_summary": self.edge_lab_summary,
            "pattern_usage": [p.to_dict() for p in self.pattern_usage],
            "metrics": self.metrics.to_dict(),
            "exploit_coverage": float(self.exploit_coverage),
            "parent_packet_id": self.parent_packet_id,
        }

        # Deterministic serialization
        content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha3_256(content_str.encode()).hexdigest()[:16]

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------

    @property
    def health_score(self) -> int:
        """
        Single 0-100 number summarizing packet quality.

        Formula: int(100 * exploit_coverage * (1 - slo_breach_rate) * avg_pattern_recall)

        Factors:
        - exploit_coverage: Fraction of patterns that are exploit-grade
        - slo_breach_rate: Rate of SLO violations (lower is better)
        - avg_pattern_recall: Average validation recall across patterns

        Returns:
            Integer score from 0 to 100.
        """
        if not self.pattern_usage:
            avg_recall = 1.0
        else:
            avg_recall = sum(p.validation_recall for p in self.pattern_usage) / len(self.pattern_usage)

        slo_factor = 1.0 - min(1.0, max(0.0, self.metrics.slo_breach_rate))
        coverage_factor = min(1.0, max(0.0, self.exploit_coverage))
        recall_factor = min(1.0, max(0.0, avg_recall))

        score = 100.0 * coverage_factor * slo_factor * recall_factor
        return int(max(0, min(100, score)))

    @property
    def glyph(self) -> str:
        """
        8-character visual fingerprint for quick packet identification.

        Format: XX-XX-XX-XX (e.g., A7-3F-B2-C9)
        Derived deterministically from packet_id.

        Returns:
            Formatted glyph string.
        """
        if len(self.packet_id) < 8:
            # Pad if somehow too short
            padded = self.packet_id.ljust(8, "0")
        else:
            padded = self.packet_id[:8]

        return f"{padded[0:2]}-{padded[2:4]}-{padded[4:6]}-{padded[6:8]}".upper()

    @property
    def integrity_status(self) -> str:
        """
        Check if packet_id matches regenerated hash.

        Returns:
            "VERIFIED" if packet_id matches computed hash
            "TAMPERED" if mismatch detected
            "UNCHECKED" if validation could not be performed
        """
        try:
            current_hash = self._compute_content_hash()
            if self.packet_id == current_hash:
                return "VERIFIED"
            else:
                return "TAMPERED"
        except Exception:
            return "UNCHECKED"

    # -------------------------------------------------------------------------
    # Serialization Methods
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize packet to dictionary.

        Includes all fields in a format suitable for JSON serialization.

        Returns:
            Dictionary representation of the packet.
        """
        return {
            "packet_id": self.packet_id,
            "deployment_id": self.deployment_id,
            "timestamp": self.timestamp,
            "manifest_ref": self.manifest_ref,
            "sampled_receipts": list(self.sampled_receipts),
            "clarity_audit_ref": self.clarity_audit_ref,
            "edge_lab_summary": self.edge_lab_summary,
            "pattern_usage": [p.to_dict() for p in self.pattern_usage],
            "metrics": self.metrics.to_dict(),
            "exploit_coverage": float(self.exploit_coverage),
            "signature": self.signature,
            "parent_packet_id": self.parent_packet_id,
            # Include computed properties for convenience
            "health_score": self.health_score,
            "glyph": self.glyph,
            "integrity_status": self.integrity_status,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize packet to JSON string.

        Args:
            indent: Optional indentation for pretty-printing.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecisionPacket:
        """
        Deserialize packet from dictionary.

        Args:
            data: Dictionary with packet fields.

        Returns:
            New DecisionPacket instance.
        """
        # Parse nested structures
        pattern_usage = [
            PatternSummary.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("pattern_usage", [])
        ]

        metrics_data = data.get("metrics", {})
        if isinstance(metrics_data, dict):
            metrics = PacketMetrics.from_dict(metrics_data)
        else:
            metrics = metrics_data

        return cls(
            packet_id=data.get("packet_id", ""),
            deployment_id=data["deployment_id"],
            timestamp=data.get("timestamp", ""),
            manifest_ref=data["manifest_ref"],
            sampled_receipts=data.get("sampled_receipts", []),
            clarity_audit_ref=data.get("clarity_audit_ref", ""),
            edge_lab_summary=data.get("edge_lab_summary", {}),
            pattern_usage=pattern_usage,
            metrics=metrics,
            exploit_coverage=float(data.get("exploit_coverage", -1.0)),
            signature=data.get("signature"),
            parent_packet_id=data.get("parent_packet_id"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> DecisionPacket:
        """
        Deserialize packet from JSON string.

        Args:
            json_str: JSON string with packet data.

        Returns:
            New DecisionPacket instance.
        """
        return cls.from_dict(json.loads(json_str))

    # -------------------------------------------------------------------------
    # Comparison and Diff Methods
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Content-based equality: same packet_id = same packet."""
        if not isinstance(other, DecisionPacket):
            return NotImplemented
        return self.packet_id == other.packet_id

    def __hash__(self) -> int:
        """Hash based on packet_id for use in sets and dicts."""
        return hash(self.packet_id)

    def __lt__(self, other: DecisionPacket) -> bool:
        """Sort by annual_savings (ascending) for sortable collections."""
        return self.metrics.annual_savings < other.metrics.annual_savings

    def diff(self, other: DecisionPacket) -> Dict[str, Any]:
        """
        Compute structured delta between this packet and another.

        Compares patterns, metrics, coverage, and health scores.

        Args:
            other: Another DecisionPacket to compare against.

        Returns:
            Dictionary with structured differences:
            - patterns_added: Patterns in other but not in self
            - patterns_removed: Patterns in self but not in other
            - metrics_delta: Changes in metrics values
            - exploit_coverage_delta: Change in exploit coverage
            - health_score_delta: Change in health score
            - savings_delta: Change in annual savings
        """
        # Pattern comparison
        self_patterns = {p.pattern_id for p in self.pattern_usage}
        other_patterns = {p.pattern_id for p in other.pattern_usage}

        patterns_added = other_patterns - self_patterns
        patterns_removed = self_patterns - other_patterns

        # Metrics comparison
        metrics_delta = {
            "window_volume": other.metrics.window_volume - self.metrics.window_volume,
            "avg_compression": other.metrics.avg_compression - self.metrics.avg_compression,
            "annual_savings": other.metrics.annual_savings - self.metrics.annual_savings,
            "slo_breach_rate": other.metrics.slo_breach_rate - self.metrics.slo_breach_rate,
        }

        # Pattern-level detail for changed patterns
        self_pattern_map = {p.pattern_id: p for p in self.pattern_usage}
        other_pattern_map = {p.pattern_id: p for p in other.pattern_usage}

        pattern_changes = {}
        common_patterns = self_patterns & other_patterns
        for pid in common_patterns:
            sp = self_pattern_map[pid]
            op = other_pattern_map[pid]
            if (sp.validation_recall != op.validation_recall or
                sp.false_positive_rate != op.false_positive_rate or
                sp.dollar_value_annual != op.dollar_value_annual or
                sp.exploit_grade != op.exploit_grade):
                pattern_changes[pid] = {
                    "recall_delta": op.validation_recall - sp.validation_recall,
                    "fp_rate_delta": op.false_positive_rate - sp.false_positive_rate,
                    "value_delta": op.dollar_value_annual - sp.dollar_value_annual,
                    "exploit_grade_changed": sp.exploit_grade != op.exploit_grade,
                }

        return {
            "self_packet_id": self.packet_id,
            "other_packet_id": other.packet_id,
            "self_deployment_id": self.deployment_id,
            "other_deployment_id": other.deployment_id,
            "patterns_added": list(patterns_added),
            "patterns_removed": list(patterns_removed),
            "patterns_modified": pattern_changes,
            "metrics_delta": metrics_delta,
            "exploit_coverage_delta": other.exploit_coverage - self.exploit_coverage,
            "health_score_delta": other.health_score - self.health_score,
            "savings_delta": other.metrics.annual_savings - self.metrics.annual_savings,
            "is_lineage_linked": other.parent_packet_id == self.packet_id,
        }

    # -------------------------------------------------------------------------
    # Self-Narrating Methods
    # -------------------------------------------------------------------------

    def narrative(self, include_parent_diff: bool = True) -> str:
        """
        Generate human-readable audit narrative.

        Produces a plain English summary suitable for compliance officers,
        executives, and auditors without requiring JSON parsing.

        Args:
            include_parent_diff: If True and parent_packet_id exists, include
                comparison with parent packet (requires parent to be loaded).

        Returns:
            Multi-sentence narrative string.
        """
        # Count exploit-grade patterns
        exploit_count = sum(1 for p in self.pattern_usage if p.exploit_grade)
        total_patterns = len(self.pattern_usage)

        # Format savings
        savings = self.metrics.annual_savings
        if savings >= 1_000_000:
            savings_str = f"${savings / 1_000_000:.1f}M"
        elif savings >= 1_000:
            savings_str = f"${savings / 1_000:.0f}K"
        else:
            savings_str = f"${savings:.0f}"

        # Build narrative
        lines = []

        # Opening sentence with key stats
        lines.append(
            f"Deployment {self.deployment_id} processed {self.metrics.window_volume:,} "
            f"windows at {self.metrics.avg_compression:.1f}x compression, "
            f"saving an estimated {savings_str} annually."
        )

        # Pattern summary
        if total_patterns > 0:
            lines.append(
                f"Running {total_patterns} pattern{'s' if total_patterns != 1 else ''} "
                f"({exploit_count} exploit-grade), with {self.exploit_coverage * 100:.0f}% "
                f"exploit coverage and {self.metrics.slo_breach_rate * 100:.2f}% SLO breach rate."
            )
        else:
            lines.append("No patterns configured for this deployment.")

        # Health score
        lines.append(f"Health score: {self.health_score}/100.")

        # Integrity status
        lines.append(f"Integrity: {self.integrity_status} [{self.glyph}]")

        # Parent lineage if present
        if self.parent_packet_id:
            lines.append(f"Parent: {self.parent_packet_id[:8]}...")

        return " ".join(lines)

    def narrative_diff(self, parent: DecisionPacket) -> str:
        """
        Generate narrative describing changes from parent packet.

        Args:
            parent: The parent DecisionPacket to compare against.

        Returns:
            Narrative string describing the delta.
        """
        diff_data = parent.diff(self)

        changes = []

        # Savings change
        savings_delta = diff_data["savings_delta"]
        if abs(savings_delta) > 0:
            if savings_delta > 0:
                if parent.metrics.annual_savings > 0:
                    pct = (savings_delta / parent.metrics.annual_savings) * 100
                    changes.append(f"+{pct:.0f}% savings")
                else:
                    changes.append(f"+${savings_delta:,.0f} savings")
            else:
                if parent.metrics.annual_savings > 0:
                    pct = (savings_delta / parent.metrics.annual_savings) * 100
                    changes.append(f"{pct:.0f}% savings")
                else:
                    changes.append(f"-${abs(savings_delta):,.0f} savings")

        # Pattern changes
        added = len(diff_data["patterns_added"])
        removed = len(diff_data["patterns_removed"])
        if added > 0:
            changes.append(f"+{added} pattern{'s' if added != 1 else ''} added")
        if removed > 0:
            changes.append(f"-{removed} pattern{'s' if removed != 1 else ''} removed")

        # Health score change
        health_delta = diff_data["health_score_delta"]
        if health_delta != 0:
            sign = "+" if health_delta > 0 else ""
            changes.append(f"{sign}{health_delta} health")

        if changes:
            return f"vs parent: {', '.join(changes)}"
        else:
            return "vs parent: no significant changes"

    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------

    def verify_integrity(self) -> Tuple[bool, str]:
        """
        Verify packet integrity by recomputing hash.

        Returns:
            Tuple of (is_valid, status_message).
        """
        try:
            current_hash = self._compute_content_hash()
            if self.packet_id == current_hash:
                return (True, f"VERIFIED: packet_id {self.packet_id} matches content hash")
            else:
                return (False, f"TAMPERED: packet_id {self.packet_id} != computed {current_hash}")
        except Exception as e:
            return (False, f"UNCHECKED: verification failed with {type(e).__name__}: {e}")

    def validate_schema(self) -> Tuple[bool, List[str]]:
        """
        Validate packet against expected schema.

        Checks all required fields, types, and value ranges.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []

        # Required string fields
        if not isinstance(self.packet_id, str) or not self.packet_id:
            errors.append("packet_id must be a non-empty string")
        if not isinstance(self.deployment_id, str) or not self.deployment_id:
            errors.append("deployment_id must be a non-empty string")
        if not isinstance(self.timestamp, str) or not self.timestamp:
            errors.append("timestamp must be a non-empty string")
        if not isinstance(self.manifest_ref, str) or not self.manifest_ref:
            errors.append("manifest_ref must be a non-empty string")

        # List fields
        if not isinstance(self.sampled_receipts, list):
            errors.append("sampled_receipts must be a list")
        if not isinstance(self.pattern_usage, list):
            errors.append("pattern_usage must be a list")
        else:
            for i, p in enumerate(self.pattern_usage):
                if not isinstance(p, PatternSummary):
                    errors.append(f"pattern_usage[{i}] must be a PatternSummary")

        # Metrics validation
        if not isinstance(self.metrics, PacketMetrics):
            errors.append("metrics must be a PacketMetrics instance")
        else:
            if self.metrics.window_volume < 0:
                errors.append("metrics.window_volume must be non-negative")
            if self.metrics.avg_compression < 0:
                errors.append("metrics.avg_compression must be non-negative")
            if not 0 <= self.metrics.slo_breach_rate <= 1:
                errors.append("metrics.slo_breach_rate must be between 0 and 1")

        # Coverage validation
        if not 0 <= self.exploit_coverage <= 1:
            errors.append("exploit_coverage must be between 0 and 1")

        # Timestamp format check
        if self.timestamp:
            try:
                datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except ValueError:
                errors.append("timestamp must be valid ISO format")

        return (len(errors) == 0, errors)

    def get_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing the packet structure.

        Useful for external validation and documentation.

        Returns:
            JSON Schema dictionary.
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "DecisionPacket",
            "description": "QED v8 deployment audit artifact",
            "type": "object",
            "required": [
                "packet_id",
                "deployment_id",
                "timestamp",
                "manifest_ref",
                "sampled_receipts",
                "clarity_audit_ref",
                "edge_lab_summary",
                "pattern_usage",
                "metrics",
                "exploit_coverage",
            ],
            "properties": {
                "packet_id": {
                    "type": "string",
                    "description": "SHA3-256 hash (16 chars) of packet contents",
                    "minLength": 16,
                    "maxLength": 16,
                },
                "deployment_id": {
                    "type": "string",
                    "description": "Unique deployment identifier",
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "ISO UTC timestamp",
                },
                "manifest_ref": {
                    "type": "string",
                    "description": "Reference to deployment manifest",
                },
                "sampled_receipts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of receipt IDs included",
                },
                "clarity_audit_ref": {
                    "type": "string",
                    "description": "Reference to ClarityClean audit",
                },
                "edge_lab_summary": {
                    "type": "object",
                    "description": "Summary from edge_lab_v2 simulation",
                },
                "pattern_usage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "pattern_id",
                            "validation_recall",
                            "false_positive_rate",
                            "dollar_value_annual",
                            "exploit_grade",
                        ],
                        "properties": {
                            "pattern_id": {"type": "string"},
                            "validation_recall": {"type": "number", "minimum": 0, "maximum": 1},
                            "false_positive_rate": {"type": "number", "minimum": 0, "maximum": 1},
                            "dollar_value_annual": {"type": "number"},
                            "exploit_grade": {"type": "boolean"},
                        },
                    },
                    "description": "Patterns used in this deployment",
                },
                "metrics": {
                    "type": "object",
                    "required": [
                        "window_volume",
                        "avg_compression",
                        "annual_savings",
                        "slo_breach_rate",
                    ],
                    "properties": {
                        "window_volume": {"type": "integer", "minimum": 0},
                        "avg_compression": {"type": "number", "minimum": 0},
                        "annual_savings": {"type": "number"},
                        "slo_breach_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "description": "Deployment-level aggregate metrics",
                },
                "exploit_coverage": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Fraction of patterns that are exploit-grade",
                },
                "signature": {
                    "type": ["string", "null"],
                    "description": "Optional cryptographic signature",
                },
                "parent_packet_id": {
                    "type": ["string", "null"],
                    "description": "Link to previous packet for lineage",
                },
            },
        }


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def load_packet(path: str) -> DecisionPacket:
    """
    Load a DecisionPacket from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        DecisionPacket instance.
    """
    from pathlib import Path
    content = Path(path).read_text()
    return DecisionPacket.from_json(content)


def save_packet(packet: DecisionPacket, path: str, indent: int = 2) -> None:
    """
    Save a DecisionPacket to a JSON file.

    Args:
        packet: DecisionPacket to save.
        path: Output file path.
        indent: JSON indentation level.
    """
    from pathlib import Path
    Path(path).write_text(packet.to_json(indent=indent))


def compare_packets(a: DecisionPacket, b: DecisionPacket) -> str:
    """
    Generate human-readable comparison of two packets.

    Args:
        a: First packet (typically older).
        b: Second packet (typically newer).

    Returns:
        Formatted comparison string.
    """
    diff = a.diff(b)

    lines = [
        f"Comparison: {a.deployment_id} [{a.glyph}] → {b.deployment_id} [{b.glyph}]",
        "",
    ]

    # Pattern changes
    if diff["patterns_added"]:
        lines.append(f"Patterns added: {', '.join(diff['patterns_added'])}")
    if diff["patterns_removed"]:
        lines.append(f"Patterns removed: {', '.join(diff['patterns_removed'])}")
    if diff["patterns_modified"]:
        lines.append(f"Patterns modified: {len(diff['patterns_modified'])}")

    # Metrics changes
    md = diff["metrics_delta"]
    lines.append("")
    lines.append("Metrics delta:")
    lines.append(f"  Window volume: {md['window_volume']:+,}")
    lines.append(f"  Avg compression: {md['avg_compression']:+.2f}x")
    lines.append(f"  Annual savings: ${md['annual_savings']:+,.0f}")
    lines.append(f"  SLO breach rate: {md['slo_breach_rate'] * 100:+.2f}%")

    lines.append("")
    lines.append(f"Exploit coverage: {diff['exploit_coverage_delta'] * 100:+.1f}%")
    lines.append(f"Health score: {diff['health_score_delta']:+d}")

    if diff["is_lineage_linked"]:
        lines.append("")
        lines.append("✓ Lineage linked (b.parent_packet_id == a.packet_id)")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI Interface (when run directly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Demo usage
    print("DecisionPacket Demo")
    print("=" * 60)

    # Create sample patterns
    patterns = [
        PatternSummary(
            pattern_id="bat_thermal_001",
            validation_recall=0.995,
            false_positive_rate=0.008,
            dollar_value_annual=1_500_000,
            exploit_grade=True,
        ),
        PatternSummary(
            pattern_id="comms_dropout_002",
            validation_recall=0.98,
            false_positive_rate=0.02,
            dollar_value_annual=800_000,
            exploit_grade=False,
        ),
        PatternSummary(
            pattern_id="motion_spike_003",
            validation_recall=0.999,
            false_positive_rate=0.005,
            dollar_value_annual=2_000_000,
            exploit_grade=True,
        ),
    ]

    # Create sample metrics
    metrics = PacketMetrics(
        window_volume=1_247_832,
        avg_compression=11.3,
        annual_savings=2_400_000,
        slo_breach_rate=0.0002,
    )

    # Create packet
    packet = DecisionPacket(
        deployment_id="tesla-prod-2024-12-08",
        manifest_ref="manifests/tesla-prod-v8.yaml",
        sampled_receipts=["rcpt_001", "rcpt_002", "rcpt_003"],
        clarity_audit_ref="audits/clarity_2024-12-08.json",
        edge_lab_summary={
            "n_tests": 10000,
            "n_hits": 9850,
            "aggregate_recall": 0.985,
        },
        pattern_usage=patterns,
        metrics=metrics,
    )

    print(f"\nPacket ID: {packet.packet_id}")
    print(f"Glyph: {packet.glyph}")
    print(f"Health Score: {packet.health_score}/100")
    print(f"Integrity: {packet.integrity_status}")

    print(f"\n{'-' * 60}")
    print("Narrative:")
    print(packet.narrative())

    print(f"\n{'-' * 60}")
    print("Schema validation:")
    is_valid, errors = packet.validate_schema()
    print(f"Valid: {is_valid}")
    if errors:
        for e in errors:
            print(f"  - {e}")

    print(f"\n{'-' * 60}")
    print("JSON output (truncated):")
    json_str = packet.to_json(indent=2)
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
