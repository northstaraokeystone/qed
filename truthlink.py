"""
truthlink.py - TruthLink Fusion Layer for QED v8

TruthLink is the fusion layer that builds DecisionPackets AND projects future packets.
It's historian + oracle. Streaming-capable, attack-resistant, and self-aware.

Core Capabilities:
- Build DecisionPackets from manifests, receipts, and optional sources
- Project future packets based on hypothetical changes
- Compare packets with structured diffs and narratives
- Persist and load packets with flexible filtering
- Self-monitor health and detect drift

Design Principles:
- Streaming: Never load what you can sample
- Embedded: Build metadata in packet, not beside it
- Projective: Build past AND simulate future
- Self-aware: Track own health, detect drift
- Attack-resistant: Auditable seeds, nonces, source hashes
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

from decision_packet import DecisionPacket, PatternSummary, PacketMetrics
from mesh_view_v2 import load_manifest, extract_hook_from_receipt, parse_company_from_hook
from shared_anomalies import AnomalyPattern, load_library, get_patterns_for_hook


# -----------------------------------------------------------------------------
# Version and Constants
# -----------------------------------------------------------------------------

TRUTHLINK_VERSION = "1.0.0"
DEFAULT_SAMPLE_SIZE = 100
TOP_EXTREMES_COUNT = 5


# -----------------------------------------------------------------------------
# Change Types for Packet Projection
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AddPattern:
    """Change type: Add a new pattern to deployment."""
    pattern_id: str

    def describe(self) -> str:
        return f"Add pattern {self.pattern_id}"


@dataclass(frozen=True)
class RemovePattern:
    """Change type: Remove an existing pattern from deployment."""
    pattern_id: str

    def describe(self) -> str:
        return f"Remove pattern {self.pattern_id}"


@dataclass(frozen=True)
class ScaleFleet:
    """Change type: Scale fleet by multiplier."""
    multiplier: float

    def describe(self) -> str:
        return f"Scale fleet by {self.multiplier}x"


@dataclass(frozen=True)
class AdjustConfig:
    """Change type: Adjust a configuration field."""
    field_name: str
    value: Any

    def describe(self) -> str:
        return f"Adjust {self.field_name} to {self.value}"


# Type alias for all change types
Change = Union[AddPattern, RemovePattern, ScaleFleet, AdjustConfig]


# -----------------------------------------------------------------------------
# Projected Packet
# -----------------------------------------------------------------------------

@dataclass
class ProjectedPacket:
    """
    Projected future state of a DecisionPacket.

    Contains all DecisionPacket fields as projections plus projection metadata.
    """
    # Projected DecisionPacket fields
    deployment_id: str
    manifest_ref: str
    sampled_receipts: List[str]
    clarity_audit_ref: str
    edge_lab_summary: Dict[str, Any]
    pattern_usage: List[PatternSummary]
    metrics: PacketMetrics
    exploit_coverage: float

    # Projection metadata
    base_packet_id: str
    projected_savings_delta: float
    projected_health_delta: int
    confidence: float  # 0.0-1.0
    assumptions: List[str]
    similar_deployments: List[str]  # packet_ids used for estimation
    changes_applied: List[str]
    projected_at: str = field(default="")

    def __post_init__(self) -> None:
        if not self.projected_at:
            object.__setattr__(
                self,
                "projected_at",
                datetime.now(timezone.utc).isoformat()
            )

    @property
    def projected_health_score(self) -> int:
        """Compute projected health score."""
        if not self.pattern_usage:
            avg_recall = 1.0
        else:
            avg_recall = sum(p.validation_recall for p in self.pattern_usage) / len(self.pattern_usage)

        slo_factor = 1.0 - min(1.0, max(0.0, self.metrics.slo_breach_rate))
        coverage_factor = min(1.0, max(0.0, self.exploit_coverage))
        recall_factor = min(1.0, max(0.0, avg_recall))

        score = 100.0 * coverage_factor * slo_factor * recall_factor
        return int(max(0, min(100, score)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "manifest_ref": self.manifest_ref,
            "sampled_receipts": self.sampled_receipts,
            "clarity_audit_ref": self.clarity_audit_ref,
            "edge_lab_summary": self.edge_lab_summary,
            "pattern_usage": [p.to_dict() for p in self.pattern_usage],
            "metrics": self.metrics.to_dict(),
            "exploit_coverage": self.exploit_coverage,
            "base_packet_id": self.base_packet_id,
            "projected_savings_delta": self.projected_savings_delta,
            "projected_health_delta": self.projected_health_delta,
            "projected_health_score": self.projected_health_score,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
            "similar_deployments": self.similar_deployments,
            "changes_applied": self.changes_applied,
            "projected_at": self.projected_at,
        }


# -----------------------------------------------------------------------------
# Comparison Result
# -----------------------------------------------------------------------------

@dataclass
class Comparison:
    """
    Structured comparison between two DecisionPackets.
    """
    packet_a_id: str
    packet_b_id: str
    delta: Dict[str, Tuple[Any, Any, float]]  # field: (old, new, pct_change)
    patterns_added: List[str]
    patterns_removed: List[str]
    classification: Literal["improvement", "regression", "mixed", "neutral"]
    recommendation: str
    narration: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "packet_a_id": self.packet_a_id,
            "packet_b_id": self.packet_b_id,
            "delta": {k: {"old": v[0], "new": v[1], "pct_change": v[2]} for k, v in self.delta.items()},
            "patterns_added": self.patterns_added,
            "patterns_removed": self.patterns_removed,
            "classification": self.classification,
            "recommendation": self.recommendation,
            "narration": self.narration,
        }


# -----------------------------------------------------------------------------
# Self-Awareness Types
# -----------------------------------------------------------------------------

@dataclass
class TruthLinkHealth:
    """Health metrics for TruthLink system."""
    avg_build_time_ms: float
    packets_built_24h: int
    sources_missing_rate: float
    packet_health_trend: Literal["improving", "stable", "degrading"]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_build_time_ms": self.avg_build_time_ms,
            "packets_built_24h": self.packets_built_24h,
            "sources_missing_rate": self.sources_missing_rate,
            "packet_health_trend": self.packet_health_trend,
            "warnings": self.warnings,
        }


@dataclass
class DriftReport:
    """Report on metric drift over time."""
    window_days: int
    health_score_trend: Literal["improving", "stable", "degrading"]
    breach_rate_trend: Literal["improving", "stable", "degrading"]
    savings_trend: Literal["improving", "stable", "degrading"]
    avg_health_score_recent: float
    avg_health_score_older: float
    avg_breach_rate_recent: float
    avg_breach_rate_older: float
    flags: List[str]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_days": self.window_days,
            "health_score_trend": self.health_score_trend,
            "breach_rate_trend": self.breach_rate_trend,
            "savings_trend": self.savings_trend,
            "avg_health_score_recent": self.avg_health_score_recent,
            "avg_health_score_older": self.avg_health_score_older,
            "avg_breach_rate_recent": self.avg_breach_rate_recent,
            "avg_breach_rate_older": self.avg_breach_rate_older,
            "flags": self.flags,
            "recommendation": self.recommendation,
        }


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------

def _compute_source_hash(path: str) -> str:
    """Compute SHA3-256 hash of source file content, or hash of NOT_FOUND:{path}."""
    path_obj = Path(path)
    if not path_obj.exists():
        return hashlib.sha3_256(f"NOT_FOUND:{path}".encode()).hexdigest()[:16]

    try:
        content = path_obj.read_bytes()
        return hashlib.sha3_256(content).hexdigest()[:16]
    except IOError:
        return hashlib.sha3_256(f"READ_ERROR:{path}".encode()).hexdigest()[:16]


def _generate_sampling_seed(manifest_hash: str, deployment_id: str) -> int:
    """Generate deterministic sampling seed from manifest hash and deployment ID."""
    seed_input = f"{manifest_hash}:{deployment_id}"
    seed_hash = hashlib.sha3_256(seed_input.encode()).hexdigest()
    return int(seed_hash[:8], 16)


def _stream_sample_receipts(
    receipts_path: str,
    sample_size: int,
    seed: int,
    top_slo_breaches: int = TOP_EXTREMES_COUNT,
    top_savings: int = TOP_EXTREMES_COUNT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stream-sample receipts using reservoir sampling with stratified extremes.

    O(1) memory regardless of receipt count.

    Returns:
        Tuple of (sampled_receipts, sampling_stats)
    """
    path_obj = Path(receipts_path)
    if not path_obj.exists():
        return [], {"total_seen": 0, "sampled": 0, "extremes_included": 0}

    # Initialize RNG with seed
    rng = random.Random(seed)

    # Reservoir for main sample
    reservoir: List[Dict[str, Any]] = []

    # Priority queues for extremes (track worst SLO breaches and top savings)
    slo_breach_heap: List[Tuple[float, Dict[str, Any]]] = []  # (breach_severity, receipt)
    savings_heap: List[Tuple[float, Dict[str, Any]]] = []  # (savings, receipt)

    total_seen = 0

    # Handle directory or file
    if path_obj.is_dir():
        files = list(path_obj.glob("*.jsonl")) + list(path_obj.glob("*.json"))
    else:
        files = [path_obj]

    for file_path in files:
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    try:
                        receipt = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    total_seen += 1

                    # Track SLO breaches (higher breach rate = worse)
                    breach_severity = 0.0
                    if not receipt.get("verified", True):
                        breach_severity = 1.0
                    if receipt.get("slo_breach"):
                        breach_severity = float(receipt.get("slo_breach_severity", 1.0))

                    if breach_severity > 0:
                        if len(slo_breach_heap) < top_slo_breaches:
                            slo_breach_heap.append((breach_severity, receipt))
                            slo_breach_heap.sort(key=lambda x: x[0], reverse=True)
                        elif breach_severity > slo_breach_heap[-1][0]:
                            slo_breach_heap[-1] = (breach_severity, receipt)
                            slo_breach_heap.sort(key=lambda x: x[0], reverse=True)

                    # Track top savings
                    savings = float(receipt.get("savings_M", 0) or 0)
                    if len(savings_heap) < top_savings:
                        savings_heap.append((savings, receipt))
                        savings_heap.sort(key=lambda x: x[0], reverse=True)
                    elif savings > savings_heap[-1][0]:
                        savings_heap[-1] = (savings, receipt)
                        savings_heap.sort(key=lambda x: x[0], reverse=True)

                    # Reservoir sampling for main sample
                    if len(reservoir) < sample_size:
                        reservoir.append(receipt)
                    else:
                        # Replace with probability sample_size/total_seen
                        j = rng.randint(0, total_seen - 1)
                        if j < sample_size:
                            reservoir[j] = receipt
        except IOError:
            continue

    # Combine: extremes first, then reservoir (deduped by window_id)
    sampled: List[Dict[str, Any]] = []
    seen_ids: set = set()

    # Add SLO breach extremes
    for _, receipt in slo_breach_heap:
        wid = receipt.get("window_id", id(receipt))
        if wid not in seen_ids:
            sampled.append(receipt)
            seen_ids.add(wid)

    # Add savings extremes
    for _, receipt in savings_heap:
        wid = receipt.get("window_id", id(receipt))
        if wid not in seen_ids:
            sampled.append(receipt)
            seen_ids.add(wid)

    extremes_count = len(sampled)

    # Fill remaining from reservoir
    for receipt in reservoir:
        if len(sampled) >= sample_size:
            break
        wid = receipt.get("window_id", id(receipt))
        if wid not in seen_ids:
            sampled.append(receipt)
            seen_ids.add(wid)

    stats = {
        "total_seen": total_seen,
        "sampled": len(sampled),
        "extremes_included": extremes_count,
        "slo_breaches_captured": len(slo_breach_heap),
        "top_savings_captured": len(savings_heap),
    }

    return sampled, stats


def _extract_metrics_from_receipts(receipts: List[Dict[str, Any]]) -> Tuple[int, float, float, float]:
    """
    Extract aggregate metrics from receipts.

    Returns:
        Tuple of (window_volume, avg_compression, annual_savings, slo_breach_rate)
    """
    if not receipts:
        return 0, 0.0, 0.0, 0.0

    ratios = []
    savings = []
    breach_count = 0

    for r in receipts:
        if "ratio" in r and r["ratio"] is not None:
            ratios.append(float(r["ratio"]))

        if "savings_M" in r and r["savings_M"] is not None:
            savings.append(float(r["savings_M"]))

        # Count breaches
        verified = r.get("verified")
        if verified is False or verified is None:
            breach_count += 1

    window_volume = len(receipts)
    avg_compression = sum(ratios) / len(ratios) if ratios else 0.0
    annual_savings = sum(savings) * 1_000_000  # Convert M to actual dollars
    slo_breach_rate = breach_count / len(receipts) if receipts else 0.0

    return window_volume, avg_compression, annual_savings, slo_breach_rate


def _extract_metrics_from_manifest(manifest: Dict[str, Any]) -> Tuple[int, float, float, float]:
    """
    Extract metrics from manifest if available.

    Returns:
        Tuple of (window_volume, avg_compression, annual_savings, slo_breach_rate)
    """
    window_volume = manifest.get("total_windows", manifest.get("windows", 0))
    avg_compression = manifest.get("avg_compression", manifest.get("compression_ratio", 0.0))
    annual_savings = manifest.get("annual_savings", manifest.get("estimated_savings", 0.0))
    slo_breach_rate = manifest.get("slo_breach_rate", 0.0)

    return window_volume, avg_compression, annual_savings, slo_breach_rate


def _build_pattern_usage(
    manifest: Dict[str, Any],
    patterns_path: Optional[str] = None,
) -> List[PatternSummary]:
    """Build PatternSummary list from manifest and patterns library."""
    pattern_usage: List[PatternSummary] = []

    # Try to get pattern IDs from manifest
    pattern_ids = manifest.get("patterns", manifest.get("pattern_ids", []))
    hook_name = manifest.get("hook", manifest.get("scenario", "generic"))

    # Load patterns library if path provided
    patterns_lib: Dict[str, AnomalyPattern] = {}
    if patterns_path:
        path_obj = Path(patterns_path)
        if path_obj.exists():
            all_patterns = load_library(patterns_path)
            patterns_lib = {p.pattern_id: p for p in all_patterns}

    # Also try default path
    if not patterns_lib:
        try:
            all_patterns = get_patterns_for_hook(hook_name)
            patterns_lib = {p.pattern_id: p for p in all_patterns}
        except Exception:
            pass

    # Build PatternSummary for each pattern
    for pid in pattern_ids:
        if pid in patterns_lib:
            p = patterns_lib[pid]
            pattern_usage.append(PatternSummary(
                pattern_id=p.pattern_id,
                validation_recall=p.validation_recall,
                false_positive_rate=p.false_positive_rate,
                dollar_value_annual=p.dollar_value_annual,
                exploit_grade=p.exploit_grade,
            ))
        else:
            # Create stub for unknown pattern
            pattern_usage.append(PatternSummary(
                pattern_id=pid,
                validation_recall=0.0,
                false_positive_rate=1.0,
                dollar_value_annual=0.0,
                exploit_grade=False,
            ))

    # If no patterns in manifest, try to get from patterns lib for this hook
    if not pattern_usage and patterns_lib:
        for p in patterns_lib.values():
            pattern_usage.append(PatternSummary(
                pattern_id=p.pattern_id,
                validation_recall=p.validation_recall,
                false_positive_rate=p.false_positive_rate,
                dollar_value_annual=p.dollar_value_annual,
                exploit_grade=p.exploit_grade,
            ))

    return pattern_usage


def _generate_attack_resistant_packet_id(
    content: Dict[str, Any],
    timestamp: str,
    build_metadata_hash: str,
) -> str:
    """
    Generate attack-resistant packet_id.

    Input: canonical content + timestamp + nonce + build_metadata_hash
    Output: SHA3-256, first 16 chars
    Nonce prevents replay attacks.
    """
    # Generate cryptographic nonce
    nonce = os.urandom(16).hex()

    # Build canonical content string
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))

    # Combine all inputs
    combined = f"{content_str}:{timestamp}:{nonce}:{build_metadata_hash}"

    return hashlib.sha3_256(combined.encode()).hexdigest()[:16]


def _compute_pct_change(old: float, new: float) -> float:
    """Compute percentage change, guarding against division by zero."""
    if old == 0:
        return 100.0 if new != 0 else 0.0
    return ((new - old) / abs(old)) * 100.0


# -----------------------------------------------------------------------------
# Core Builder
# -----------------------------------------------------------------------------

def build(
    deployment_id: str,
    manifest_path: str,
    receipts_dir: str,
    clarity_path: Optional[str] = None,
    edge_lab_path: Optional[str] = None,
    patterns_path: Optional[str] = None,
    mode: Literal["single", "batch", "watch"] = "single",
    parent_packet_id: Optional[str] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> Union[DecisionPacket, List[DecisionPacket], Generator[DecisionPacket, None, None]]:
    """
    Unified builder for DecisionPackets.

    Args:
        deployment_id: Unique identifier for this deployment.
        manifest_path: Path to deployment manifest (required).
        receipts_dir: Path to receipts directory or JSONL file.
        clarity_path: Optional path to ClarityClean audit output.
        edge_lab_path: Optional path to edge_lab simulation results.
        patterns_path: Optional path to patterns library.
        mode: Build mode - "single", "batch", or "watch".
        parent_packet_id: Optional link to previous packet for lineage.
        sample_size: Number of receipts to sample (default: 100).

    Returns:
        single mode: Single DecisionPacket
        batch mode: List of DecisionPackets
        watch mode: Generator yielding packets as manifests appear

    Raises:
        FileNotFoundError: If manifest not found.
        ValueError: If manifest missing required metrics.
    """
    if mode == "single":
        return _build_single(
            deployment_id=deployment_id,
            manifest_path=manifest_path,
            receipts_dir=receipts_dir,
            clarity_path=clarity_path,
            edge_lab_path=edge_lab_path,
            patterns_path=patterns_path,
            parent_packet_id=parent_packet_id,
            sample_size=sample_size,
        )
    elif mode == "batch":
        return _build_batch(
            deployment_id=deployment_id,
            manifest_path=manifest_path,
            receipts_dir=receipts_dir,
            clarity_path=clarity_path,
            edge_lab_path=edge_lab_path,
            patterns_path=patterns_path,
            sample_size=sample_size,
        )
    elif mode == "watch":
        return _build_watch(
            deployment_id=deployment_id,
            manifest_path=manifest_path,
            receipts_dir=receipts_dir,
            clarity_path=clarity_path,
            edge_lab_path=edge_lab_path,
            patterns_path=patterns_path,
            sample_size=sample_size,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'single', 'batch', or 'watch'.")


def _build_single(
    deployment_id: str,
    manifest_path: str,
    receipts_dir: str,
    clarity_path: Optional[str] = None,
    edge_lab_path: Optional[str] = None,
    patterns_path: Optional[str] = None,
    parent_packet_id: Optional[str] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> DecisionPacket:
    """Build a single DecisionPacket."""
    start_time = time.time()

    # 1. Load manifest (required)
    manifest_path_obj = Path(manifest_path)
    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    manifest_hash = _compute_source_hash(manifest_path)

    # Extract deployment_id from manifest if not provided
    if not deployment_id:
        deployment_id = manifest.get("deployment_id", manifest.get("run_id", "unknown"))

    # 2. Generate sampling seed for auditability
    sampling_seed = _generate_sampling_seed(manifest_hash, deployment_id)

    # 3. Stream-sample receipts
    sampled_receipts, sampling_stats = _stream_sample_receipts(
        receipts_dir,
        sample_size,
        sampling_seed,
    )

    # 4. Load optional sources with graceful degradation
    sources_loaded: List[str] = ["manifest"]
    sources_missing: List[str] = []
    source_hashes: Dict[str, str] = {"manifest": manifest_hash}

    # ClarityClean
    clarity_audit_ref = ""
    if clarity_path:
        clarity_hash = _compute_source_hash(clarity_path)
        source_hashes["clarity"] = clarity_hash
        if Path(clarity_path).exists():
            sources_loaded.append("clarity")
            clarity_audit_ref = clarity_path
        else:
            sources_missing.append("clarity")
            clarity_audit_ref = f"NOT_FOUND:{clarity_path}"

    # Edge Lab
    edge_lab_summary: Dict[str, Any] = {}
    if edge_lab_path:
        edge_lab_hash = _compute_source_hash(edge_lab_path)
        source_hashes["edge_lab"] = edge_lab_hash
        if Path(edge_lab_path).exists():
            sources_loaded.append("edge_lab")
            try:
                with open(edge_lab_path, "r") as f:
                    edge_lab_summary = json.load(f)
            except (json.JSONDecodeError, IOError):
                edge_lab_summary = {"error": "failed_to_load"}
        else:
            sources_missing.append("edge_lab")
            edge_lab_summary = {"status": "NOT_FOUND"}

    # Patterns
    if patterns_path:
        patterns_hash = _compute_source_hash(patterns_path)
        source_hashes["patterns"] = patterns_hash
        if Path(patterns_path).exists():
            sources_loaded.append("patterns")
        else:
            sources_missing.append("patterns")

    # 5. Compute metrics with guards
    # Try manifest first, then receipts
    manifest_metrics = _extract_metrics_from_manifest(manifest)
    receipt_metrics = _extract_metrics_from_receipts(sampled_receipts)

    # Use manifest metrics if available, otherwise use receipt metrics
    window_volume = manifest_metrics[0] if manifest_metrics[0] > 0 else receipt_metrics[0]
    avg_compression = manifest_metrics[1] if manifest_metrics[1] > 0 else receipt_metrics[1]
    annual_savings = manifest_metrics[2] if manifest_metrics[2] > 0 else receipt_metrics[2]
    slo_breach_rate = receipt_metrics[3] if sampled_receipts else manifest_metrics[3]

    # Guard against division by zero
    slo_breach_rate = max(0.0, min(1.0, slo_breach_rate))

    metrics = PacketMetrics(
        window_volume=int(window_volume),
        avg_compression=float(avg_compression),
        annual_savings=float(annual_savings),
        slo_breach_rate=float(slo_breach_rate),
    )

    # 6. Build pattern usage
    pattern_usage = _build_pattern_usage(manifest, patterns_path)

    # 7. Compute exploit coverage with guard
    if pattern_usage:
        exploit_count = sum(1 for p in pattern_usage if p.exploit_grade)
        exploit_coverage = exploit_count / len(pattern_usage)
    else:
        exploit_coverage = 0.0

    # If no active patterns, exploit_coverage = 0.0
    if not patterns_path or not Path(patterns_path).exists() if patterns_path else True:
        if not pattern_usage:
            exploit_coverage = 0.0

    # 8. Build timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    # 9. Compute build duration
    build_duration_ms = (time.time() - start_time) * 1000

    # 10. Create build metadata to embed in packet
    build_metadata = {
        "build_timestamp": timestamp,
        "build_duration_ms": round(build_duration_ms, 2),
        "sources_loaded": sources_loaded,
        "sources_missing": sources_missing,
        "source_hashes": source_hashes,
        "sampling_seed": sampling_seed,
        "sampling_stats": sampling_stats,
        "truthlink_version": TRUTHLINK_VERSION,
    }

    # Embed build metadata in edge_lab_summary for storage
    edge_lab_summary["_build_metadata"] = build_metadata

    # 11. Extract sampled receipt IDs
    sampled_receipt_ids = [
        r.get("window_id", r.get("receipt_id", f"receipt_{i}"))
        for i, r in enumerate(sampled_receipts)
    ]

    # 12. Create DecisionPacket
    # Note: packet_id will be auto-generated by DecisionPacket.__post_init__
    # but we can override with attack-resistant version
    packet = DecisionPacket(
        deployment_id=deployment_id,
        manifest_ref=manifest_path,
        sampled_receipts=sampled_receipt_ids,
        clarity_audit_ref=clarity_audit_ref,
        edge_lab_summary=edge_lab_summary,
        pattern_usage=pattern_usage,
        metrics=metrics,
        exploit_coverage=exploit_coverage,
        parent_packet_id=parent_packet_id,
        timestamp=timestamp,
    )

    return packet


def _build_batch(
    deployment_id: str,
    manifest_path: str,
    receipts_dir: str,
    clarity_path: Optional[str] = None,
    edge_lab_path: Optional[str] = None,
    patterns_path: Optional[str] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> List[DecisionPacket]:
    """Build all pending packets in directory."""
    packets: List[DecisionPacket] = []

    # If manifest_path is a directory, find all manifests
    manifest_path_obj = Path(manifest_path)

    if manifest_path_obj.is_dir():
        manifest_files = list(manifest_path_obj.glob("*manifest*.json"))
    else:
        manifest_files = [manifest_path_obj]

    for mf in manifest_files:
        try:
            # Extract deployment_id from manifest filename or content
            manifest_data = load_manifest(str(mf))
            dep_id = manifest_data.get("deployment_id", manifest_data.get("run_id", mf.stem))

            packet = _build_single(
                deployment_id=dep_id,
                manifest_path=str(mf),
                receipts_dir=receipts_dir,
                clarity_path=clarity_path,
                edge_lab_path=edge_lab_path,
                patterns_path=patterns_path,
                sample_size=sample_size,
            )
            packets.append(packet)
        except (FileNotFoundError, ValueError) as e:
            # Log error but continue with other manifests
            continue

    return packets


def _build_watch(
    deployment_id: str,
    manifest_path: str,
    receipts_dir: str,
    clarity_path: Optional[str] = None,
    edge_lab_path: Optional[str] = None,
    patterns_path: Optional[str] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    poll_interval: float = 1.0,
) -> Generator[DecisionPacket, None, None]:
    """Generator that yields packets as manifests appear."""
    manifest_dir = Path(manifest_path)
    if not manifest_dir.is_dir():
        manifest_dir = manifest_dir.parent

    processed: set = set()

    while True:
        # Find new manifest files
        manifest_files = list(manifest_dir.glob("*manifest*.json"))

        for mf in manifest_files:
            if str(mf) in processed:
                continue

            try:
                manifest_data = load_manifest(str(mf))
                dep_id = manifest_data.get("deployment_id", manifest_data.get("run_id", mf.stem))

                packet = _build_single(
                    deployment_id=dep_id,
                    manifest_path=str(mf),
                    receipts_dir=receipts_dir,
                    clarity_path=clarity_path,
                    edge_lab_path=edge_lab_path,
                    patterns_path=patterns_path,
                    sample_size=sample_size,
                )
                processed.add(str(mf))
                yield packet
            except (FileNotFoundError, ValueError):
                # Skip invalid manifests
                processed.add(str(mf))
                continue

        time.sleep(poll_interval)


# -----------------------------------------------------------------------------
# Packet Projection (Chef's Kiss)
# -----------------------------------------------------------------------------

def project(
    base_packet: DecisionPacket,
    changes: List[Change],
    historical_packets: Optional[List[DecisionPacket]] = None,
    patterns_library: Optional[List[AnomalyPattern]] = None,
) -> ProjectedPacket:
    """
    Project future state of a packet based on hypothetical changes.

    Args:
        base_packet: The current DecisionPacket to project from.
        changes: List of Change objects to apply.
        historical_packets: Optional historical packets for confidence estimation.
        patterns_library: Optional patterns library for pattern lookups.

    Returns:
        ProjectedPacket with projected metrics and confidence.
    """
    # Load patterns library if not provided
    if patterns_library is None:
        try:
            patterns_library = load_library()
        except Exception:
            patterns_library = []

    patterns_by_id = {p.pattern_id: p for p in patterns_library}

    # Start with base packet values
    projected_patterns = list(base_packet.pattern_usage)
    projected_savings = base_packet.metrics.annual_savings
    projected_window_volume = base_packet.metrics.window_volume
    projected_compression = base_packet.metrics.avg_compression
    projected_breach_rate = base_packet.metrics.slo_breach_rate

    assumptions: List[str] = []
    changes_applied: List[str] = []
    confidence_factors: List[float] = []

    for change in changes:
        changes_applied.append(change.describe())

        if isinstance(change, AddPattern):
            # Add pattern
            pattern = patterns_by_id.get(change.pattern_id)
            if pattern:
                new_summary = PatternSummary(
                    pattern_id=pattern.pattern_id,
                    validation_recall=pattern.validation_recall,
                    false_positive_rate=pattern.false_positive_rate,
                    dollar_value_annual=pattern.dollar_value_annual,
                    exploit_grade=pattern.exploit_grade,
                )
                projected_patterns.append(new_summary)
                projected_savings += pattern.dollar_value_annual
                assumptions.append(f"Pattern {change.pattern_id} adds ${pattern.dollar_value_annual:,.0f} annual savings")
                confidence_factors.append(0.8)  # Good confidence if pattern exists
            else:
                assumptions.append(f"Pattern {change.pattern_id} not found in library, impact unknown")
                confidence_factors.append(0.3)  # Low confidence for unknown pattern

        elif isinstance(change, RemovePattern):
            # Remove pattern
            for i, p in enumerate(projected_patterns):
                if p.pattern_id == change.pattern_id:
                    projected_savings -= p.dollar_value_annual
                    assumptions.append(f"Removing pattern {change.pattern_id} reduces savings by ${p.dollar_value_annual:,.0f}")
                    projected_patterns.pop(i)
                    confidence_factors.append(0.9)  # High confidence for removal
                    break
            else:
                assumptions.append(f"Pattern {change.pattern_id} not found in current usage")
                confidence_factors.append(0.5)

        elif isinstance(change, ScaleFleet):
            # Scale fleet
            projected_window_volume = int(projected_window_volume * change.multiplier)
            projected_savings *= change.multiplier

            # Confidence decay at >2x scaling
            if change.multiplier > 2.0:
                scale_confidence = 0.5 / (change.multiplier - 1)
                assumptions.append(f"Fleet scaled {change.multiplier}x, confidence reduced due to extrapolation")
            else:
                scale_confidence = 0.9 - (0.1 * abs(change.multiplier - 1))
                assumptions.append(f"Fleet scaled {change.multiplier}x with linear extrapolation")

            confidence_factors.append(max(0.2, min(0.95, scale_confidence)))

        elif isinstance(change, AdjustConfig):
            # Config adjustment - lookup impact from historical data if available
            if historical_packets:
                # Look for similar config changes in history
                similar_changes = _find_similar_config_changes(
                    historical_packets, change.field_name, change.value
                )
                if similar_changes:
                    avg_impact = sum(s["impact"] for s in similar_changes) / len(similar_changes)
                    projected_savings *= (1 + avg_impact)
                    assumptions.append(f"Config {change.field_name}={change.value} estimated from {len(similar_changes)} historical changes")
                    confidence_factors.append(0.7)
                else:
                    assumptions.append(f"No historical data for {change.field_name} adjustment, impact unknown")
                    confidence_factors.append(0.3)
            else:
                assumptions.append(f"Config {change.field_name}={change.value} applied, no historical data for impact estimation")
                confidence_factors.append(0.4)

    # Compute new exploit coverage
    if projected_patterns:
        exploit_count = sum(1 for p in projected_patterns if p.exploit_grade)
        projected_exploit_coverage = exploit_count / len(projected_patterns)
    else:
        projected_exploit_coverage = 0.0

    # Build projected metrics
    projected_metrics = PacketMetrics(
        window_volume=projected_window_volume,
        avg_compression=projected_compression,
        annual_savings=projected_savings,
        slo_breach_rate=projected_breach_rate,
    )

    # Compute confidence
    confidence = estimate_confidence(base_packet, changes, historical_packets or [])

    # Find similar deployments
    similar_deployments = _find_similar_deployments(base_packet, historical_packets or [])

    # Compute deltas
    base_health = base_packet.health_score

    # Create projected packet to compute projected health
    projected_packet = ProjectedPacket(
        deployment_id=base_packet.deployment_id,
        manifest_ref=base_packet.manifest_ref,
        sampled_receipts=base_packet.sampled_receipts,
        clarity_audit_ref=base_packet.clarity_audit_ref,
        edge_lab_summary=base_packet.edge_lab_summary,
        pattern_usage=projected_patterns,
        metrics=projected_metrics,
        exploit_coverage=projected_exploit_coverage,
        base_packet_id=base_packet.packet_id,
        projected_savings_delta=projected_savings - base_packet.metrics.annual_savings,
        projected_health_delta=0,  # Will compute below
        confidence=confidence,
        assumptions=assumptions,
        similar_deployments=similar_deployments,
        changes_applied=changes_applied,
    )

    # Update health delta
    projected_health = projected_packet.projected_health_score
    object.__setattr__(projected_packet, "projected_health_delta", projected_health - base_health)

    return projected_packet


def estimate_confidence(
    base_packet: DecisionPacket,
    changes: List[Change],
    historical_packets: List[DecisionPacket],
) -> float:
    """
    Estimate confidence in projection.

    High confidence: Many similar deployments, small changes
    Low confidence: Novel deployment, large changes, few comparables
    """
    confidence = 0.8  # Start with moderate confidence

    # Factor 1: Number of similar historical packets
    similar_count = len(_find_similar_deployments(base_packet, historical_packets))
    if similar_count >= 10:
        confidence += 0.1
    elif similar_count >= 5:
        confidence += 0.05
    elif similar_count == 0:
        confidence -= 0.2

    # Factor 2: Size of changes
    for change in changes:
        if isinstance(change, ScaleFleet):
            if change.multiplier > 3.0:
                confidence -= 0.3
            elif change.multiplier > 2.0:
                confidence -= 0.15
            elif change.multiplier < 0.5:
                confidence -= 0.1
        elif isinstance(change, AddPattern) or isinstance(change, RemovePattern):
            confidence -= 0.05  # Small penalty for pattern changes
        elif isinstance(change, AdjustConfig):
            confidence -= 0.1  # Config changes are less predictable

    # Factor 3: Number of changes
    if len(changes) > 5:
        confidence -= 0.15
    elif len(changes) > 3:
        confidence -= 0.05

    # Clamp to valid range
    return max(0.1, min(0.95, confidence))


def _find_similar_deployments(
    base_packet: DecisionPacket,
    historical_packets: List[DecisionPacket],
    max_results: int = 5,
) -> List[str]:
    """Find packet_ids of similar deployments."""
    if not historical_packets:
        return []

    # Score similarity based on pattern overlap and metrics similarity
    scores: List[Tuple[float, str]] = []
    base_patterns = {p.pattern_id for p in base_packet.pattern_usage}

    for packet in historical_packets:
        if packet.packet_id == base_packet.packet_id:
            continue

        # Pattern overlap score
        packet_patterns = {p.pattern_id for p in packet.pattern_usage}
        if base_patterns or packet_patterns:
            overlap = len(base_patterns & packet_patterns)
            union = len(base_patterns | packet_patterns)
            pattern_score = overlap / union if union > 0 else 0
        else:
            pattern_score = 1.0  # Both empty = similar

        # Metrics similarity score
        metric_diffs = [
            abs(base_packet.metrics.window_volume - packet.metrics.window_volume) / max(1, base_packet.metrics.window_volume),
            abs(base_packet.metrics.annual_savings - packet.metrics.annual_savings) / max(1, abs(base_packet.metrics.annual_savings)),
        ]
        metric_score = 1.0 - min(1.0, sum(metric_diffs) / len(metric_diffs))

        # Combined score
        total_score = (pattern_score * 0.6) + (metric_score * 0.4)
        scores.append((total_score, packet.packet_id))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)

    return [pid for _, pid in scores[:max_results]]


def _find_similar_config_changes(
    historical_packets: List[DecisionPacket],
    field_name: str,
    value: Any,
) -> List[Dict[str, Any]]:
    """Find historical config changes with estimated impact."""
    # This would require tracking config changes in packet metadata
    # For now, return empty list - can be enhanced with actual config tracking
    return []


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------

def compare(packet_a: DecisionPacket, packet_b: DecisionPacket) -> Comparison:
    """
    Compare two DecisionPackets.

    Args:
        packet_a: First packet (typically older).
        packet_b: Second packet (typically newer).

    Returns:
        Comparison with structured delta and classification.
    """
    # Compute deltas
    delta: Dict[str, Tuple[Any, Any, float]] = {}

    # Health score
    delta["health_score"] = (
        packet_a.health_score,
        packet_b.health_score,
        _compute_pct_change(packet_a.health_score, packet_b.health_score),
    )

    # Annual savings
    delta["annual_savings"] = (
        packet_a.metrics.annual_savings,
        packet_b.metrics.annual_savings,
        _compute_pct_change(packet_a.metrics.annual_savings, packet_b.metrics.annual_savings),
    )

    # SLO breach rate
    delta["slo_breach_rate"] = (
        packet_a.metrics.slo_breach_rate,
        packet_b.metrics.slo_breach_rate,
        _compute_pct_change(packet_a.metrics.slo_breach_rate, packet_b.metrics.slo_breach_rate),
    )

    # Window volume
    delta["window_volume"] = (
        packet_a.metrics.window_volume,
        packet_b.metrics.window_volume,
        _compute_pct_change(packet_a.metrics.window_volume, packet_b.metrics.window_volume),
    )

    # Exploit coverage
    delta["exploit_coverage"] = (
        packet_a.exploit_coverage,
        packet_b.exploit_coverage,
        _compute_pct_change(packet_a.exploit_coverage, packet_b.exploit_coverage),
    )

    # Pattern changes
    a_patterns = {p.pattern_id for p in packet_a.pattern_usage}
    b_patterns = {p.pattern_id for p in packet_b.pattern_usage}
    patterns_added = list(b_patterns - a_patterns)
    patterns_removed = list(a_patterns - b_patterns)

    # Classification
    health_up = packet_b.health_score > packet_a.health_score
    savings_up = packet_b.metrics.annual_savings > packet_a.metrics.annual_savings
    breach_down = packet_b.metrics.slo_breach_rate < packet_a.metrics.slo_breach_rate

    # Check for neutral (< 1% change across all metrics)
    all_changes_small = all(abs(v[2]) < 1.0 for v in delta.values())

    if all_changes_small:
        classification: Literal["improvement", "regression", "mixed", "neutral"] = "neutral"
    elif health_up and savings_up and breach_down:
        classification = "improvement"
    elif not health_up and not savings_up and not breach_down:
        classification = "regression"
    else:
        classification = "mixed"

    # Recommendation
    if classification == "improvement":
        recommendation = "Proceed with changes. All key metrics improved."
    elif classification == "regression":
        recommendation = "Review changes. Key metrics have regressed."
    elif classification == "mixed":
        improvements = []
        regressions = []
        if health_up:
            improvements.append("health")
        else:
            regressions.append("health")
        if savings_up:
            improvements.append("savings")
        else:
            regressions.append("savings")
        if breach_down:
            improvements.append("SLO compliance")
        else:
            regressions.append("SLO compliance")
        recommendation = f"Mixed results. Improved: {', '.join(improvements)}. Regressed: {', '.join(regressions)}. Evaluate tradeoffs."
    else:
        recommendation = "No significant changes detected. Monitor for trends."

    # Narration
    narration = _generate_comparison_narration(packet_a, packet_b, delta, classification, patterns_added, patterns_removed)

    return Comparison(
        packet_a_id=packet_a.packet_id,
        packet_b_id=packet_b.packet_id,
        delta=delta,
        patterns_added=patterns_added,
        patterns_removed=patterns_removed,
        classification=classification,
        recommendation=recommendation,
        narration=narration,
    )


def _generate_comparison_narration(
    packet_a: DecisionPacket,
    packet_b: DecisionPacket,
    delta: Dict[str, Tuple[Any, Any, float]],
    classification: str,
    patterns_added: List[str],
    patterns_removed: List[str],
) -> str:
    """Generate human-readable comparison narrative."""
    parts = []

    # Opening
    parts.append(f"Comparing deployment {packet_a.deployment_id} across two snapshots.")

    # Health score
    health_old, health_new, health_pct = delta["health_score"]
    if health_pct > 0:
        parts.append(f"Health score improved from {health_old} to {health_new} (+{health_pct:.1f}%).")
    elif health_pct < 0:
        parts.append(f"Health score declined from {health_old} to {health_new} ({health_pct:.1f}%).")
    else:
        parts.append(f"Health score remained stable at {health_new}.")

    # Savings
    savings_old, savings_new, savings_pct = delta["annual_savings"]
    if savings_pct > 5:
        savings_delta = savings_new - savings_old
        if savings_delta >= 1_000_000:
            parts.append(f"Annual savings increased by ${savings_delta/1_000_000:.1f}M ({savings_pct:.1f}%).")
        else:
            parts.append(f"Annual savings increased by ${savings_delta:,.0f} ({savings_pct:.1f}%).")
    elif savings_pct < -5:
        savings_delta = savings_old - savings_new
        if savings_delta >= 1_000_000:
            parts.append(f"Annual savings decreased by ${savings_delta/1_000_000:.1f}M ({savings_pct:.1f}%).")
        else:
            parts.append(f"Annual savings decreased by ${savings_delta:,.0f} ({savings_pct:.1f}%).")

    # Pattern changes
    if patterns_added:
        parts.append(f"Added {len(patterns_added)} pattern(s): {', '.join(patterns_added[:3])}{'...' if len(patterns_added) > 3 else ''}.")
    if patterns_removed:
        parts.append(f"Removed {len(patterns_removed)} pattern(s): {', '.join(patterns_removed[:3])}{'...' if len(patterns_removed) > 3 else ''}.")

    # Classification summary
    if classification == "improvement":
        parts.append("Overall, this represents a clear improvement across all key metrics.")
    elif classification == "regression":
        parts.append("Overall, this represents a regression that warrants investigation.")
    elif classification == "mixed":
        parts.append("The changes show mixed results with both improvements and regressions.")
    else:
        parts.append("The changes are minimal and within normal variation.")

    return " ".join(parts)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

def save(
    packet: DecisionPacket,
    output_dir: str = "data/packets/",
) -> str:
    """
    Save packet to persistent storage.

    Args:
        packet: DecisionPacket to save.
        output_dir: Directory for packet storage.

    Returns:
        packet_id of saved packet.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    packets_file = output_path / "packets.jsonl"

    # Append to JSONL file
    with open(packets_file, "a") as f:
        f.write(packet.to_json() + "\n")

    return packet.packet_id


def load(
    packet_id: Optional[str] = None,
    deployment_id: Optional[str] = None,
    packets_dir: str = "data/packets/",
) -> List[DecisionPacket]:
    """
    Load packets from persistent storage.

    Args:
        packet_id: Load specific packet by ID.
        deployment_id: Load all packets for deployment.
        packets_dir: Directory containing packets.

    Returns:
        List of DecisionPackets, sorted by timestamp desc.
    """
    packets_path = Path(packets_dir)
    packets_file = packets_path / "packets.jsonl"

    if not packets_file.exists():
        return []

    packets: List[DecisionPacket] = []

    with open(packets_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                packet = DecisionPacket.from_json(line)

                # Filter by packet_id
                if packet_id and packet.packet_id != packet_id:
                    continue

                # Filter by deployment_id
                if deployment_id and packet.deployment_id != deployment_id:
                    continue

                packets.append(packet)

                # If searching for specific packet_id, stop after finding it
                if packet_id and packet.packet_id == packet_id:
                    break

            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    # Sort by timestamp descending
    packets.sort(key=lambda p: p.timestamp, reverse=True)

    return packets


# -----------------------------------------------------------------------------
# Self-Awareness
# -----------------------------------------------------------------------------

def health_check(packets_dir: str = "data/packets/") -> TruthLinkHealth:
    """
    Check TruthLink system health.

    Returns:
        TruthLinkHealth with system metrics and warnings.
    """
    packets = load(packets_dir=packets_dir)
    warnings: List[str] = []

    # Compute average build time from recent packets
    build_times: List[float] = []
    sources_missing_count = 0
    total_packets_checked = 0

    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    packets_24h = 0

    recent_health_scores: List[int] = []
    older_health_scores: List[int] = []
    cutoff_week = now - timedelta(days=7)

    for packet in packets[:100]:  # Check last 100 packets
        total_packets_checked += 1

        # Extract build metadata
        build_metadata = packet.edge_lab_summary.get("_build_metadata", {})

        if "build_duration_ms" in build_metadata:
            build_times.append(build_metadata["build_duration_ms"])

        if "sources_missing" in build_metadata:
            if build_metadata["sources_missing"]:
                sources_missing_count += 1

        # Check if within 24h
        try:
            packet_time = datetime.fromisoformat(packet.timestamp.replace("Z", "+00:00"))
            if packet_time >= cutoff_24h:
                packets_24h += 1

            # Track health scores for trend
            if packet_time >= cutoff_week:
                recent_health_scores.append(packet.health_score)
            else:
                older_health_scores.append(packet.health_score)
        except ValueError:
            pass

    # Compute averages
    avg_build_time_ms = sum(build_times) / len(build_times) if build_times else 0.0
    sources_missing_rate = sources_missing_count / total_packets_checked if total_packets_checked > 0 else 0.0

    # Determine health trend
    if recent_health_scores and older_health_scores:
        recent_avg = sum(recent_health_scores) / len(recent_health_scores)
        older_avg = sum(older_health_scores) / len(older_health_scores)

        if recent_avg > older_avg + 5:
            health_trend: Literal["improving", "stable", "degrading"] = "improving"
        elif recent_avg < older_avg - 5:
            health_trend = "degrading"
        else:
            health_trend = "stable"
    else:
        health_trend = "stable"

    # Generate warnings
    if avg_build_time_ms > 5000:
        warnings.append(f"Build time averaging {avg_build_time_ms:.0f}ms, exceeds 5s target")

    if sources_missing_rate > 0.2:
        warnings.append(f"Missing sources rate at {sources_missing_rate*100:.1f}%, check data paths")

    if health_trend == "degrading":
        warnings.append("Health scores trending downward over past week")

    if packets_24h == 0 and total_packets_checked > 0:
        warnings.append("No packets built in last 24 hours")

    return TruthLinkHealth(
        avg_build_time_ms=round(avg_build_time_ms, 2),
        packets_built_24h=packets_24h,
        sources_missing_rate=round(sources_missing_rate, 4),
        packet_health_trend=health_trend,
        warnings=warnings,
    )


def detect_drift(
    packets: Optional[List[DecisionPacket]] = None,
    window_days: int = 7,
    packets_dir: str = "data/packets/",
) -> DriftReport:
    """
    Detect metric drift over time.

    Args:
        packets: Optional list of packets to analyze.
        window_days: Window for recent vs older comparison.
        packets_dir: Directory to load packets from if not provided.

    Returns:
        DriftReport with trend analysis and recommendations.
    """
    if packets is None:
        packets = load(packets_dir=packets_dir)

    if not packets:
        return DriftReport(
            window_days=window_days,
            health_score_trend="stable",
            breach_rate_trend="stable",
            savings_trend="stable",
            avg_health_score_recent=0.0,
            avg_health_score_older=0.0,
            avg_breach_rate_recent=0.0,
            avg_breach_rate_older=0.0,
            flags=["No packets available for drift analysis"],
            recommendation="Build more packets to enable drift detection.",
        )

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    recent_health: List[int] = []
    older_health: List[int] = []
    recent_breach: List[float] = []
    older_breach: List[float] = []
    recent_savings: List[float] = []
    older_savings: List[float] = []

    for packet in packets:
        try:
            packet_time = datetime.fromisoformat(packet.timestamp.replace("Z", "+00:00"))

            if packet_time >= cutoff:
                recent_health.append(packet.health_score)
                recent_breach.append(packet.metrics.slo_breach_rate)
                recent_savings.append(packet.metrics.annual_savings)
            else:
                older_health.append(packet.health_score)
                older_breach.append(packet.metrics.slo_breach_rate)
                older_savings.append(packet.metrics.annual_savings)
        except ValueError:
            continue

    # Compute averages
    avg_health_recent = sum(recent_health) / len(recent_health) if recent_health else 0.0
    avg_health_older = sum(older_health) / len(older_health) if older_health else 0.0
    avg_breach_recent = sum(recent_breach) / len(recent_breach) if recent_breach else 0.0
    avg_breach_older = sum(older_breach) / len(older_breach) if older_breach else 0.0
    avg_savings_recent = sum(recent_savings) / len(recent_savings) if recent_savings else 0.0
    avg_savings_older = sum(older_savings) / len(older_savings) if older_savings else 0.0

    # Determine trends
    def get_trend(recent: float, older: float, improvement_direction: str) -> Literal["improving", "stable", "degrading"]:
        if older == 0:
            return "stable"
        pct_change = ((recent - older) / abs(older)) * 100

        if abs(pct_change) < 5:
            return "stable"

        if improvement_direction == "up":
            return "improving" if pct_change > 0 else "degrading"
        else:
            return "improving" if pct_change < 0 else "degrading"

    health_trend = get_trend(avg_health_recent, avg_health_older, "up")
    breach_trend = get_trend(avg_breach_recent, avg_breach_older, "down")
    savings_trend = get_trend(avg_savings_recent, avg_savings_older, "up")

    # Generate flags
    flags: List[str] = []

    if health_trend == "degrading":
        flags.append(f"Health score trending down: {avg_health_older:.1f} -> {avg_health_recent:.1f}")

    if breach_trend == "degrading":
        flags.append(f"SLO breach rate trending up: {avg_breach_older*100:.2f}% -> {avg_breach_recent*100:.2f}%")

    if savings_trend == "degrading":
        flags.append(f"Savings trending down: ${avg_savings_older:,.0f} -> ${avg_savings_recent:,.0f}")

    if not recent_health:
        flags.append(f"No recent packets in last {window_days} days")

    if not older_health:
        flags.append("Insufficient historical data for comparison")

    # Generate recommendation
    if health_trend == "degrading" or breach_trend == "degrading":
        recommendation = "Investigate recent changes. Key metrics are degrading. Review pattern performance and deployment configurations."
    elif savings_trend == "degrading":
        recommendation = "Review savings calculations and pattern coverage. Consider adding high-value patterns."
    elif all(t == "improving" for t in [health_trend, breach_trend, savings_trend]):
        recommendation = "All metrics improving. Continue current trajectory and monitor for sustainability."
    else:
        recommendation = "Metrics are stable. Continue monitoring for any emerging trends."

    return DriftReport(
        window_days=window_days,
        health_score_trend=health_trend,
        breach_rate_trend=breach_trend,
        savings_trend=savings_trend,
        avg_health_score_recent=round(avg_health_recent, 2),
        avg_health_score_older=round(avg_health_older, 2),
        avg_breach_rate_recent=round(avg_breach_recent, 4),
        avg_breach_rate_older=round(avg_breach_older, 4),
        flags=flags,
        recommendation=recommendation,
    )


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Core builder
    "build",
    # Projection
    "project",
    # Comparison
    "compare",
    # Persistence
    "save",
    "load",
    # Self-awareness
    "health_check",
    "detect_drift",
    # Types
    "ProjectedPacket",
    "Comparison",
    "Change",
    "AddPattern",
    "RemovePattern",
    "ScaleFleet",
    "AdjustConfig",
    "TruthLinkHealth",
    "DriftReport",
    # Constants
    "TRUTHLINK_VERSION",
]


# -----------------------------------------------------------------------------
# CLI Interface (when run directly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("TruthLink Fusion Layer Demo")
    print("=" * 60)
    print(f"Version: {TRUTHLINK_VERSION}")
    print()

    # Demo build (requires actual data)
    print("TruthLink provides:")
    print("  - build(): Unified packet builder with streaming sampling")
    print("  - project(): Future state projection (historian + oracle)")
    print("  - compare(): Structured packet comparison with narratives")
    print("  - save()/load(): Packet persistence with flexible filtering")
    print("  - health_check(): System self-monitoring")
    print("  - detect_drift(): Metric drift detection")
    print()
    print("Example usage:")
    print("  packet = build('deploy-1', 'manifest.json', 'receipts/')")
    print("  projected = project(packet, [AddPattern('new-pattern')])")
    print("  diff = compare(old_packet, new_packet)")
