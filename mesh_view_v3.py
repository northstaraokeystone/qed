"""
mesh_view_v3.py - Living Fleet Topology Graph for QED v8

This module transforms packets into a living fleet topology. It's not just
visualization—it's a queryable graph that predicts pattern propagation,
detects fleet anomalies, and recommends cross-deployment actions.

Foundation for:
- Portfolio binder (v9)
- Meta layer (v10)

Design Principles:
- Queryable: Graph answers questions, not just displays data
- Predictive: Propagation prediction powers v10 proposals
- Self-diagnosing: Graph knows when it's unhealthy
- Temporal: Evolution tracking catches drift
- Backward compatible: v2 exports preserved

Changes from v2:
- Single build() entry point (vs load_packets + build_deployment_graph)
- similarity_score + weight on edges (vs unweighted)
- find_clusters + find_outliers (vs find_reuse_clusters only)
- track_evolution() temporal tracking (vs static graph)
- is_stale flag on nodes (vs nodes just exist)
- predict_propagation() recommendations (vs visualization only)
- diagnose() health checks (vs no health check)
- recommend_patterns() per deployment (vs no recommendations)
- fleet_cohesion score (vs basic fleet metrics)
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Import from sibling modules
from decision_packet import DecisionPacket, PatternSummary, PacketMetrics

# Backward compatibility: re-export all mesh_view_v2 public functions
from mesh_view_v2 import (
    load_manifest,
    sample_receipts,
    extract_hook_from_receipt,
    parse_company_from_hook,
    compute_metrics,
    emit_view,
    print_table,
    load_qed_receipts,
    load_clarity_receipts,
    load_sim_results,
    load_anomaly_library,
    load_cross_domain_validations,
    join_all_sources,
    compute_metrics_v2,
    emit_view_v2,
    generate_mesh_view,
)

# Note: These are added for v2 compatibility per spec
# compute_exploit_count and compute_cross_domain_links are internal in v2
# but we expose wrapper functions here
from mesh_view_v2 import _compute_exploit_count as compute_exploit_count
from mesh_view_v2 import _compute_cross_domain_links as compute_cross_domain_links

# Build fake build_company_table if not already defined
# (mesh_view_v2 has compute_metrics but we alias it for compatibility)
build_company_table = compute_metrics

# Import config schema
try:
    from config_schema import QEDConfig
except ImportError:
    QEDConfig = None  # type: ignore

# Default stale threshold in days
DEFAULT_STALE_THRESHOLD_DAYS = 7


# =============================================================================
# NodeMetrics Dataclass
# =============================================================================

@dataclass(frozen=True)
class NodeMetrics:
    """
    Deployment node health metrics.

    Captures the essential health indicators for a single deployment.
    """
    health_score: int  # 0-100
    savings: float  # annual savings in dollars
    breach_rate: float  # SLO breach rate (0.0-1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "health_score": self.health_score,
            "savings": float(self.savings),
            "breach_rate": float(self.breach_rate),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NodeMetrics:
        """Deserialize from dictionary."""
        return cls(
            health_score=int(data.get("health_score", 0)),
            savings=float(data.get("savings", 0.0)),
            breach_rate=float(data.get("breach_rate", 0.0)),
        )


# =============================================================================
# DeploymentNode Dataclass
# =============================================================================

@dataclass
class DeploymentNode:
    """
    Node in the deployment graph representing a single deployment.

    Contains all relevant deployment information extracted from packets
    and enriched with config data.
    """
    deployment_id: str
    packet_id: str  # most recent packet
    hook: str
    region: str  # from config, or "unknown"
    hardware_profile: str  # from config
    patterns: Set[str]  # enabled pattern_ids
    exploit_patterns: Set[str]  # subset with exploit_grade=True
    metrics: NodeMetrics
    last_updated: str  # timestamp
    is_stale: bool  # no packet in >7 days

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "packet_id": self.packet_id,
            "hook": self.hook,
            "region": self.region,
            "hardware_profile": self.hardware_profile,
            "patterns": sorted(self.patterns),
            "exploit_patterns": sorted(self.exploit_patterns),
            "metrics": self.metrics.to_dict(),
            "last_updated": self.last_updated,
            "is_stale": self.is_stale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeploymentNode:
        """Deserialize from dictionary."""
        return cls(
            deployment_id=data["deployment_id"],
            packet_id=data["packet_id"],
            hook=data["hook"],
            region=data.get("region", "unknown"),
            hardware_profile=data.get("hardware_profile", "unknown"),
            patterns=set(data.get("patterns", [])),
            exploit_patterns=set(data.get("exploit_patterns", [])),
            metrics=NodeMetrics.from_dict(data.get("metrics", {})),
            last_updated=data.get("last_updated", ""),
            is_stale=data.get("is_stale", False),
        )


# =============================================================================
# DeploymentEdge Dataclass
# =============================================================================

@dataclass
class DeploymentEdge:
    """
    Edge connecting two deployment nodes.

    Represents relationships between deployments based on shared attributes.
    """
    source: str  # deployment_id
    target: str  # deployment_id
    connection_types: Set[str]  # "hook", "hardware", "region", "pattern", "exploit_pattern"
    shared_patterns: Set[str]
    similarity_score: float  # 0.0-1.0
    weight: float  # combined connection strength

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "connection_types": sorted(self.connection_types),
            "shared_patterns": sorted(self.shared_patterns),
            "similarity_score": float(self.similarity_score),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeploymentEdge:
        """Deserialize from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            connection_types=set(data.get("connection_types", [])),
            shared_patterns=set(data.get("shared_patterns", [])),
            similarity_score=float(data.get("similarity_score", 0.0)),
            weight=float(data.get("weight", 0.0)),
        )


# =============================================================================
# Cluster Dataclass
# =============================================================================

@dataclass
class Cluster:
    """
    A cluster of similar deployments.

    Identified by union-find clustering based on edge weights.
    """
    cluster_id: int
    deployment_ids: List[str]
    centroid_deployment: str  # most connected node in cluster
    avg_similarity: float
    patterns_in_common: Set[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "deployment_ids": self.deployment_ids,
            "centroid_deployment": self.centroid_deployment,
            "avg_similarity": float(self.avg_similarity),
            "patterns_in_common": sorted(self.patterns_in_common),
        }


# =============================================================================
# FleetSummary Dataclass
# =============================================================================

@dataclass
class FleetSummary:
    """
    Summary metrics for the entire fleet.

    Aggregates deployment-level metrics into fleet-wide view.
    """
    total_deployments: int
    active_deployments: int  # non-stale
    stale_deployments: int
    total_patterns_in_use: int
    unique_patterns: Set[str]
    exploit_patterns_coverage: float  # % of deployments using exploit patterns
    total_annual_savings: float
    avg_health_score: float
    avg_breach_rate: float
    cluster_count: int
    largest_cluster_size: int
    orphan_count: int  # nodes with no edges
    fleet_cohesion: float  # 0.0-1.0, how connected is the fleet

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_deployments": self.total_deployments,
            "active_deployments": self.active_deployments,
            "stale_deployments": self.stale_deployments,
            "total_patterns_in_use": self.total_patterns_in_use,
            "unique_patterns": sorted(self.unique_patterns),
            "exploit_patterns_coverage": float(self.exploit_patterns_coverage),
            "total_annual_savings": float(self.total_annual_savings),
            "avg_health_score": float(self.avg_health_score),
            "avg_breach_rate": float(self.avg_breach_rate),
            "cluster_count": self.cluster_count,
            "largest_cluster_size": self.largest_cluster_size,
            "orphan_count": self.orphan_count,
            "fleet_cohesion": float(self.fleet_cohesion),
        }


# =============================================================================
# PropagationCandidate Dataclass
# =============================================================================

@dataclass
class PropagationCandidate:
    """
    Candidate deployment for pattern propagation.

    Ranks deployments by likelihood of successful pattern adoption.
    """
    deployment_id: str
    propagation_score: float  # 0.0-1.0
    reasons: List[str]  # why this is a good candidate
    risk_factors: List[str]  # why it might not work
    estimated_savings_delta: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "propagation_score": float(self.propagation_score),
            "reasons": self.reasons,
            "risk_factors": self.risk_factors,
            "estimated_savings_delta": float(self.estimated_savings_delta),
            "confidence": float(self.confidence),
        }


# =============================================================================
# PatternRecommendation Dataclass
# =============================================================================

@dataclass
class PatternRecommendation:
    """
    Pattern recommendation for a specific deployment.

    Suggests patterns that similar deployments are using successfully.
    """
    pattern_id: str
    score: float
    source_deployments: List[str]  # where it's working
    estimated_impact: float  # predicted savings
    adoption_rate: float  # % of similar deployments using it

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "score": float(self.score),
            "source_deployments": self.source_deployments,
            "estimated_impact": float(self.estimated_impact),
            "adoption_rate": float(self.adoption_rate),
        }


# =============================================================================
# GraphSnapshot Dataclass
# =============================================================================

@dataclass
class GraphSnapshot:
    """
    Point-in-time snapshot of graph state.

    Used for temporal evolution tracking.
    """
    timestamp: str
    node_count: int
    edge_count: int
    fleet_metrics: FleetSummary
    graph_hash: str  # SHA3 of structure

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "fleet_metrics": self.fleet_metrics.to_dict(),
            "graph_hash": self.graph_hash,
        }


# =============================================================================
# EvolutionReport Dataclass
# =============================================================================

@dataclass
class EvolutionReport:
    """
    Report on graph evolution over time.

    Tracks changes in nodes, edges, patterns, and cohesion.
    """
    snapshots: List[GraphSnapshot]
    nodes_added: List[str]
    nodes_removed: List[str]
    patterns_spreading: List[str]  # adoption increasing
    patterns_declining: List[str]  # adoption decreasing
    cohesion_trend: str  # "improving" | "stable" | "fragmenting"
    anomalies: List[str]  # unusual changes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "nodes_added": self.nodes_added,
            "nodes_removed": self.nodes_removed,
            "patterns_spreading": self.patterns_spreading,
            "patterns_declining": self.patterns_declining,
            "cohesion_trend": self.cohesion_trend,
            "anomalies": self.anomalies,
        }


# =============================================================================
# GraphDiagnosis Dataclass
# =============================================================================

@dataclass
class GraphDiagnosis:
    """
    Health diagnosis of the deployment graph.

    Self-diagnosing capability for the graph.
    """
    is_healthy: bool
    warnings: List[str]
    issues: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_healthy": self.is_healthy,
            "warnings": self.warnings,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


# =============================================================================
# DeploymentGraph Class
# =============================================================================

class DeploymentGraph:
    """
    The living fleet topology graph.

    Contains nodes (deployments) and edges (relationships) with
    rich querying, clustering, and diagnostic capabilities.
    """

    def __init__(
        self,
        nodes: Optional[Dict[str, DeploymentNode]] = None,
        edges: Optional[List[DeploymentEdge]] = None,
        created_at: Optional[str] = None,
        packet_count: int = 0,
    ):
        """
        Initialize deployment graph.

        Args:
            nodes: Dictionary mapping deployment_id to DeploymentNode
            edges: List of DeploymentEdge connecting nodes
            created_at: ISO timestamp of graph creation
            packet_count: Number of packets used to build the graph
        """
        self.nodes: Dict[str, DeploymentNode] = nodes or {}
        self.edges: List[DeploymentEdge] = edges or []
        self.created_at: str = created_at or datetime.now(timezone.utc).isoformat()
        self.packet_count: int = packet_count

        # Build adjacency index for fast neighbor lookups
        self._adjacency: Dict[str, List[Tuple[str, DeploymentEdge]]] = defaultdict(list)
        self._build_adjacency_index()

    def _build_adjacency_index(self) -> None:
        """Build adjacency list for fast neighbor queries."""
        self._adjacency.clear()
        for edge in self.edges:
            self._adjacency[edge.source].append((edge.target, edge))
            self._adjacency[edge.target].append((edge.source, edge))

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def neighbors(
        self,
        deployment_id: str,
        min_similarity: float = 0.0
    ) -> List[DeploymentNode]:
        """
        Get connected nodes filtered by similarity threshold.

        Args:
            deployment_id: The deployment to find neighbors for
            min_similarity: Minimum similarity score to include

        Returns:
            List of neighboring DeploymentNode objects
        """
        if deployment_id not in self.nodes:
            return []

        result = []
        for neighbor_id, edge in self._adjacency.get(deployment_id, []):
            if edge.similarity_score >= min_similarity:
                if neighbor_id in self.nodes:
                    result.append(self.nodes[neighbor_id])

        return result

    def find_clusters(self, min_similarity: float = 0.3) -> List[Cluster]:
        """
        Find clusters of similar deployments using union-find.

        Args:
            min_similarity: Minimum edge similarity to consider connected

        Returns:
            List of Cluster objects
        """
        # Union-find data structure
        parent: Dict[str, str] = {nid: nid for nid in self.nodes}
        rank: Dict[str, int] = {nid: 0 for nid in self.nodes}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Union nodes connected by edges above threshold
        for edge in self.edges:
            if edge.similarity_score >= min_similarity:
                if edge.source in self.nodes and edge.target in self.nodes:
                    union(edge.source, edge.target)

        # Group nodes by root
        clusters_map: Dict[str, List[str]] = defaultdict(list)
        for nid in self.nodes:
            root = find(nid)
            clusters_map[root].append(nid)

        # Build Cluster objects
        clusters = []
        for cluster_id, (root, members) in enumerate(clusters_map.items()):
            if len(members) < 1:
                continue

            # Find centroid (most connected node)
            connection_counts = {}
            for member in members:
                count = sum(
                    1 for _, e in self._adjacency.get(member, [])
                    if e.similarity_score >= min_similarity
                )
                connection_counts[member] = count

            centroid = max(members, key=lambda m: connection_counts.get(m, 0))

            # Calculate average similarity within cluster
            similarities = []
            for edge in self.edges:
                if edge.source in members and edge.target in members:
                    similarities.append(edge.similarity_score)
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

            # Find common patterns
            pattern_sets = [self.nodes[m].patterns for m in members]
            common_patterns = set.intersection(*pattern_sets) if pattern_sets else set()

            clusters.append(Cluster(
                cluster_id=cluster_id,
                deployment_ids=sorted(members),
                centroid_deployment=centroid,
                avg_similarity=avg_sim,
                patterns_in_common=common_patterns,
            ))

        return sorted(clusters, key=lambda c: len(c.deployment_ids), reverse=True)

    def find_outliers(self, threshold: float = 0.1) -> List[DeploymentNode]:
        """
        Find nodes with low average similarity to neighbors.

        These are isolated or unique deployments that may need attention.

        Args:
            threshold: Maximum avg similarity to be considered outlier

        Returns:
            List of outlier DeploymentNode objects
        """
        outliers = []

        for nid, node in self.nodes.items():
            neighbors = self._adjacency.get(nid, [])
            if not neighbors:
                # No edges at all = definitely an outlier
                outliers.append(node)
            else:
                avg_sim = sum(e.similarity_score for _, e in neighbors) / len(neighbors)
                if avg_sim < threshold:
                    outliers.append(node)

        return outliers

    def subgraph(self, filter_fn: Callable[[DeploymentNode], bool]) -> DeploymentGraph:
        """
        Extract a subgraph matching the filter function.

        Args:
            filter_fn: Function taking DeploymentNode, returning True to include

        Returns:
            New DeploymentGraph containing only matching nodes and their edges
        """
        # Filter nodes
        filtered_nodes = {
            nid: node for nid, node in self.nodes.items()
            if filter_fn(node)
        }

        # Filter edges (both endpoints must be in filtered nodes)
        filtered_edges = [
            edge for edge in self.edges
            if edge.source in filtered_nodes and edge.target in filtered_nodes
        ]

        return DeploymentGraph(
            nodes=filtered_nodes,
            edges=filtered_edges,
            created_at=self.created_at,
            packet_count=self.packet_count,
        )

    def path(self, source_id: str, target_id: str) -> List[str]:
        """
        Find shortest path between two deployments using BFS.

        Args:
            source_id: Starting deployment_id
            target_id: Ending deployment_id

        Returns:
            List of deployment_ids forming the path, empty if no path exists
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return []

        if source_id == target_id:
            return [source_id]

        # BFS
        visited = {source_id}
        queue = [(source_id, [source_id])]

        while queue:
            current, current_path = queue.pop(0)

            for neighbor_id, _ in self._adjacency.get(current, []):
                if neighbor_id == target_id:
                    return current_path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, current_path + [neighbor_id]))

        return []  # No path found

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at,
            "packet_count": self.packet_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeploymentGraph:
        """Deserialize graph from dictionary."""
        nodes = {
            nid: DeploymentNode.from_dict(ndata)
            for nid, ndata in data.get("nodes", {}).items()
        }
        edges = [
            DeploymentEdge.from_dict(edata)
            for edata in data.get("edges", [])
        ]
        return cls(
            nodes=nodes,
            edges=edges,
            created_at=data.get("created_at", ""),
            packet_count=data.get("packet_count", 0),
        )


# =============================================================================
# Graph Construction
# =============================================================================

def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compute_similarity(
    node_a: DeploymentNode,
    node_b: DeploymentNode
) -> float:
    """
    Compute similarity score between two nodes.

    Formula:
    jaccard(patterns_a, patterns_b) * 0.5 +
    jaccard(exploit_a, exploit_b) * 0.3 +
    (1 if same_hook else 0) * 0.2
    """
    pattern_sim = _jaccard(node_a.patterns, node_b.patterns)
    exploit_sim = _jaccard(node_a.exploit_patterns, node_b.exploit_patterns)
    hook_sim = 1.0 if node_a.hook == node_b.hook else 0.0

    return pattern_sim * 0.5 + exploit_sim * 0.3 + hook_sim * 0.2


def _is_stale(timestamp: str, threshold_days: int = DEFAULT_STALE_THRESHOLD_DAYS) -> bool:
    """Check if a timestamp is older than threshold."""
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - ts) > timedelta(days=threshold_days)
    except (ValueError, TypeError):
        return True  # Invalid timestamp = stale


def build(
    packets: List[DecisionPacket],
    configs: Optional[Dict[str, Any]] = None,
    stale_threshold_days: int = DEFAULT_STALE_THRESHOLD_DAYS,
) -> DeploymentGraph:
    """
    Build deployment graph from packets.

    Single entry point for graph construction:
    - Creates node for each unique deployment_id (uses most recent packet)
    - Enriches with config data if provided (region, hardware)
    - Computes edges based on shared attributes
    - Calculates similarity scores and weights

    Args:
        packets: List of DecisionPacket objects
        configs: Optional dict mapping deployment_id to QEDConfig or config dict
        stale_threshold_days: Days after which a deployment is considered stale

    Returns:
        Complete DeploymentGraph
    """
    if not packets:
        return DeploymentGraph(packet_count=0)

    configs = configs or {}

    # Group packets by deployment_id, keep most recent
    deployment_packets: Dict[str, DecisionPacket] = {}
    for packet in packets:
        existing = deployment_packets.get(packet.deployment_id)
        if existing is None or packet.timestamp > existing.timestamp:
            deployment_packets[packet.deployment_id] = packet

    # Build nodes
    nodes: Dict[str, DeploymentNode] = {}
    for deployment_id, packet in deployment_packets.items():
        # Extract patterns from packet
        patterns = {p.pattern_id for p in packet.pattern_usage}
        exploit_patterns = {p.pattern_id for p in packet.pattern_usage if p.exploit_grade}

        # Get config enrichment
        config = configs.get(deployment_id, {})
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        elif not isinstance(config, dict):
            config = {}

        region = config.get("region", "unknown")
        hardware_profile = config.get("hardware_profile", config.get("hook", "unknown"))

        # Extract hook from config or infer from deployment_id
        hook = config.get("hook", "")
        if not hook:
            # Try to infer from deployment_id
            if "_" in deployment_id:
                hook = deployment_id.split("_")[0]
            else:
                hook = deployment_id.split("-")[0] if "-" in deployment_id else "unknown"

        # Check staleness
        is_stale = _is_stale(packet.timestamp, stale_threshold_days)

        # Build node metrics
        metrics = NodeMetrics(
            health_score=packet.health_score,
            savings=packet.metrics.annual_savings,
            breach_rate=packet.metrics.slo_breach_rate,
        )

        nodes[deployment_id] = DeploymentNode(
            deployment_id=deployment_id,
            packet_id=packet.packet_id,
            hook=hook,
            region=region,
            hardware_profile=hardware_profile,
            patterns=patterns,
            exploit_patterns=exploit_patterns,
            metrics=metrics,
            last_updated=packet.timestamp,
            is_stale=is_stale,
        )

    # Build edges (O(n²) but using sparse representation)
    edges: List[DeploymentEdge] = []
    node_ids = list(nodes.keys())

    for i, nid_a in enumerate(node_ids):
        node_a = nodes[nid_a]
        for nid_b in node_ids[i + 1:]:
            node_b = nodes[nid_b]

            # Determine connection types
            connection_types: Set[str] = set()
            shared_patterns: Set[str] = set()

            # Hook match
            if node_a.hook == node_b.hook and node_a.hook != "unknown":
                connection_types.add("hook")

            # Hardware match
            if (node_a.hardware_profile == node_b.hardware_profile and
                    node_a.hardware_profile != "unknown"):
                connection_types.add("hardware")

            # Region match
            if node_a.region == node_b.region and node_a.region != "unknown":
                connection_types.add("region")

            # Pattern overlap (Jaccard > 0.1)
            pattern_jaccard = _jaccard(node_a.patterns, node_b.patterns)
            if pattern_jaccard > 0.1:
                connection_types.add("pattern")
                shared_patterns = node_a.patterns & node_b.patterns

            # Exploit pattern overlap
            exploit_shared = node_a.exploit_patterns & node_b.exploit_patterns
            if exploit_shared:
                connection_types.add("exploit_pattern")
                shared_patterns.update(exploit_shared)

            # Only create edge if there's some connection
            if connection_types:
                similarity = _compute_similarity(node_a, node_b)

                # Weight combines connection types and similarity
                weight = len(connection_types) * 0.2 + similarity * 0.6

                edges.append(DeploymentEdge(
                    source=nid_a,
                    target=nid_b,
                    connection_types=connection_types,
                    shared_patterns=shared_patterns,
                    similarity_score=similarity,
                    weight=weight,
                ))

    return DeploymentGraph(
        nodes=nodes,
        edges=edges,
        packet_count=len(packets),
    )


# =============================================================================
# Fleet Metrics
# =============================================================================

def compute_fleet_metrics(graph: DeploymentGraph) -> FleetSummary:
    """
    Compute aggregate fleet metrics from deployment graph.

    fleet_cohesion formula:
    (actual_edges / max_possible_edges) * (1 - orphan_ratio) * avg_similarity

    High cohesion = fleet shares patterns, low = fragmented silos.

    Args:
        graph: DeploymentGraph to analyze

    Returns:
        FleetSummary with aggregate metrics
    """
    n_nodes = len(graph.nodes)

    if n_nodes == 0:
        return FleetSummary(
            total_deployments=0,
            active_deployments=0,
            stale_deployments=0,
            total_patterns_in_use=0,
            unique_patterns=set(),
            exploit_patterns_coverage=0.0,
            total_annual_savings=0.0,
            avg_health_score=0.0,
            avg_breach_rate=0.0,
            cluster_count=0,
            largest_cluster_size=0,
            orphan_count=0,
            fleet_cohesion=0.0,
        )

    # Count stale/active
    active = sum(1 for n in graph.nodes.values() if not n.is_stale)
    stale = n_nodes - active

    # Collect patterns
    all_patterns: Set[str] = set()
    pattern_counts: Dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        all_patterns.update(node.patterns)
        for p in node.patterns:
            pattern_counts[p] += 1

    # Count deployments using exploit patterns
    exploit_users = sum(
        1 for n in graph.nodes.values()
        if n.exploit_patterns
    )
    exploit_coverage = exploit_users / n_nodes if n_nodes > 0 else 0.0

    # Aggregate metrics
    total_savings = sum(n.metrics.savings for n in graph.nodes.values())
    health_scores = [n.metrics.health_score for n in graph.nodes.values()]
    breach_rates = [n.metrics.breach_rate for n in graph.nodes.values()]

    avg_health = sum(health_scores) / n_nodes
    avg_breach = sum(breach_rates) / n_nodes

    # Clustering
    clusters = graph.find_clusters(min_similarity=0.3)
    cluster_count = len(clusters)
    largest_cluster = max((len(c.deployment_ids) for c in clusters), default=0)

    # Orphans (nodes with no edges)
    nodes_with_edges = set()
    for edge in graph.edges:
        nodes_with_edges.add(edge.source)
        nodes_with_edges.add(edge.target)
    orphan_count = n_nodes - len(nodes_with_edges)

    # Fleet cohesion
    max_possible_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    actual_edges = len(graph.edges)
    edge_ratio = actual_edges / max_possible_edges if max_possible_edges > 0 else 0

    orphan_ratio = orphan_count / n_nodes if n_nodes > 0 else 0

    avg_similarity = (
        sum(e.similarity_score for e in graph.edges) / len(graph.edges)
        if graph.edges else 0.0
    )

    fleet_cohesion = edge_ratio * (1 - orphan_ratio) * (avg_similarity + 0.5)
    fleet_cohesion = min(1.0, max(0.0, fleet_cohesion))

    return FleetSummary(
        total_deployments=n_nodes,
        active_deployments=active,
        stale_deployments=stale,
        total_patterns_in_use=sum(pattern_counts.values()),
        unique_patterns=all_patterns,
        exploit_patterns_coverage=exploit_coverage,
        total_annual_savings=total_savings,
        avg_health_score=avg_health,
        avg_breach_rate=avg_breach,
        cluster_count=cluster_count,
        largest_cluster_size=largest_cluster,
        orphan_count=orphan_count,
        fleet_cohesion=fleet_cohesion,
    )


# =============================================================================
# Pattern Propagation Prediction (Chef's Kiss)
# =============================================================================

def predict_propagation(
    graph: DeploymentGraph,
    pattern_id: str,
    source_deployment: str,
) -> List[PropagationCandidate]:
    """
    Predict which deployments should adopt a pattern next.

    Given a pattern working in source deployment, rank which deployments
    should adopt it based on:
    - Similarity to source
    - Same hook (strong compatibility indicator)
    - Historical success (other patterns from source worked in target)
    - Gap analysis (target doesn't have pattern but neighbors do)
    - Risk profile match

    Args:
        graph: DeploymentGraph to analyze
        pattern_id: Pattern to propagate
        source_deployment: Deployment where pattern is working

    Returns:
        List of PropagationCandidate ranked by propagation_score
    """
    if source_deployment not in graph.nodes:
        return []

    source_node = graph.nodes[source_deployment]

    # Only consider deployments that don't have this pattern
    candidates = []
    for nid, node in graph.nodes.items():
        if nid == source_deployment:
            continue
        if pattern_id in node.patterns:
            continue  # Already has the pattern

        # Calculate propagation score
        reasons: List[str] = []
        risk_factors: List[str] = []

        # Base similarity
        similarity = _compute_similarity(source_node, node)
        score = similarity * 0.4

        if similarity > 0.5:
            reasons.append(f"High similarity ({similarity:.2f}) to source")
        elif similarity > 0.3:
            reasons.append(f"Moderate similarity ({similarity:.2f}) to source")
        else:
            risk_factors.append(f"Low similarity ({similarity:.2f}) to source")

        # Same hook bonus
        if node.hook == source_node.hook:
            score += 0.25
            reasons.append(f"Same hook ({node.hook})")
        else:
            risk_factors.append(f"Different hook ({node.hook} vs {source_node.hook})")

        # Gap analysis: neighbors have the pattern
        neighbors = graph.neighbors(nid, min_similarity=0.2)
        neighbor_adoption = sum(
            1 for n in neighbors if pattern_id in n.patterns
        )
        if neighbors:
            adoption_ratio = neighbor_adoption / len(neighbors)
            score += adoption_ratio * 0.2
            if adoption_ratio > 0.5:
                reasons.append(f"{neighbor_adoption}/{len(neighbors)} neighbors have pattern")

        # Shared patterns (historical success proxy)
        shared = source_node.patterns & node.patterns
        if shared:
            score += min(len(shared) * 0.05, 0.15)
            reasons.append(f"Shares {len(shared)} patterns with source")

        # Risk factors
        if node.is_stale:
            score *= 0.5
            risk_factors.append("Deployment is stale")

        if node.metrics.breach_rate > 0.01:
            score *= 0.8
            risk_factors.append(f"High breach rate ({node.metrics.breach_rate:.2%})")

        # Estimate savings delta based on source pattern value
        source_pattern_value = 0.0
        for p in source_node.patterns:
            if p == pattern_id:
                # Try to find pattern value from source patterns
                source_pattern_value = source_node.metrics.savings / max(len(source_node.patterns), 1)
                break

        estimated_savings = source_pattern_value * similarity

        # Confidence based on data quality
        confidence = 0.5 + similarity * 0.3
        if neighbor_adoption > 0:
            confidence += 0.1
        confidence = min(1.0, confidence)

        candidates.append(PropagationCandidate(
            deployment_id=nid,
            propagation_score=min(1.0, max(0.0, score)),
            reasons=reasons,
            risk_factors=risk_factors,
            estimated_savings_delta=estimated_savings,
            confidence=confidence,
        ))

    # Sort by propagation score descending
    return sorted(candidates, key=lambda c: c.propagation_score, reverse=True)


def recommend_patterns(
    graph: DeploymentGraph,
    deployment_id: str,
    top_k: int = 5,
) -> List[PatternRecommendation]:
    """
    Recommend patterns for a deployment based on similar deployments.

    Analyzes what patterns similar deployments are using and identifies
    gaps where this deployment could benefit.

    Args:
        graph: DeploymentGraph to analyze
        deployment_id: Deployment to recommend patterns for
        top_k: Maximum number of recommendations

    Returns:
        List of PatternRecommendation ranked by score
    """
    if deployment_id not in graph.nodes:
        return []

    target_node = graph.nodes[deployment_id]
    current_patterns = target_node.patterns

    # Find similar deployments
    similar_nodes = graph.neighbors(deployment_id, min_similarity=0.2)

    if not similar_nodes:
        return []

    # Collect patterns from similar deployments
    pattern_sources: Dict[str, List[str]] = defaultdict(list)
    pattern_values: Dict[str, List[float]] = defaultdict(list)

    for node in similar_nodes:
        for pattern in node.patterns:
            if pattern not in current_patterns:
                pattern_sources[pattern].append(node.deployment_id)
                # Estimate per-pattern value
                per_pattern_value = node.metrics.savings / max(len(node.patterns), 1)
                pattern_values[pattern].append(per_pattern_value)

    # Score patterns
    recommendations = []
    total_similar = len(similar_nodes)

    for pattern_id, sources in pattern_sources.items():
        adoption_rate = len(sources) / total_similar
        avg_value = sum(pattern_values[pattern_id]) / len(pattern_values[pattern_id])

        # Score based on adoption rate and value
        score = adoption_rate * 0.6 + min(avg_value / 1_000_000, 0.4)

        # Boost exploit patterns
        is_exploit = any(
            pattern_id in graph.nodes[s].exploit_patterns
            for s in sources if s in graph.nodes
        )
        if is_exploit:
            score *= 1.2

        recommendations.append(PatternRecommendation(
            pattern_id=pattern_id,
            score=min(1.0, score),
            source_deployments=sources[:5],  # Limit to 5 examples
            estimated_impact=avg_value,
            adoption_rate=adoption_rate,
        ))

    # Sort by score and return top_k
    recommendations.sort(key=lambda r: r.score, reverse=True)
    return recommendations[:top_k]


# =============================================================================
# Temporal Evolution
# =============================================================================

def _compute_graph_hash(graph: DeploymentGraph) -> str:
    """Compute SHA3 hash of graph structure."""
    structure = {
        "nodes": sorted(graph.nodes.keys()),
        "edges": sorted(
            (e.source, e.target) for e in graph.edges
        ),
    }
    canonical = json.dumps(structure, sort_keys=True, separators=(",", ":"))
    return hashlib.sha3_256(canonical.encode()).hexdigest()[:16]


def track_evolution(graphs: List[DeploymentGraph]) -> EvolutionReport:
    """
    Track graph evolution over time.

    Compares graphs to identify:
    - Nodes added/removed
    - Edges added/removed
    - Patterns spreading (adoption increasing)
    - Patterns dying (adoption decreasing)
    - Fleet cohesion trend

    Args:
        graphs: List of DeploymentGraph objects (ordered by time)

    Returns:
        EvolutionReport with change analysis
    """
    if not graphs:
        return EvolutionReport(
            snapshots=[],
            nodes_added=[],
            nodes_removed=[],
            patterns_spreading=[],
            patterns_declining=[],
            cohesion_trend="stable",
            anomalies=[],
        )

    # Create snapshots
    snapshots = []
    for g in graphs:
        metrics = compute_fleet_metrics(g)
        snapshots.append(GraphSnapshot(
            timestamp=g.created_at,
            node_count=len(g.nodes),
            edge_count=len(g.edges),
            fleet_metrics=metrics,
            graph_hash=_compute_graph_hash(g),
        ))

    if len(graphs) < 2:
        return EvolutionReport(
            snapshots=snapshots,
            nodes_added=[],
            nodes_removed=[],
            patterns_spreading=[],
            patterns_declining=[],
            cohesion_trend="stable",
            anomalies=[],
        )

    # Compare first and last
    first, last = graphs[0], graphs[-1]

    first_nodes = set(first.nodes.keys())
    last_nodes = set(last.nodes.keys())

    nodes_added = sorted(last_nodes - first_nodes)
    nodes_removed = sorted(first_nodes - last_nodes)

    # Pattern adoption tracking
    def get_pattern_adoption(g: DeploymentGraph) -> Dict[str, float]:
        n_nodes = len(g.nodes)
        if n_nodes == 0:
            return {}
        pattern_counts: Dict[str, int] = defaultdict(int)
        for node in g.nodes.values():
            for p in node.patterns:
                pattern_counts[p] += 1
        return {p: count / n_nodes for p, count in pattern_counts.items()}

    first_adoption = get_pattern_adoption(first)
    last_adoption = get_pattern_adoption(last)

    all_patterns = set(first_adoption.keys()) | set(last_adoption.keys())

    patterns_spreading = []
    patterns_declining = []

    for p in all_patterns:
        first_rate = first_adoption.get(p, 0.0)
        last_rate = last_adoption.get(p, 0.0)
        if last_rate > first_rate + 0.1:
            patterns_spreading.append(p)
        elif last_rate < first_rate - 0.1:
            patterns_declining.append(p)

    # Cohesion trend
    first_cohesion = snapshots[0].fleet_metrics.fleet_cohesion
    last_cohesion = snapshots[-1].fleet_metrics.fleet_cohesion

    if last_cohesion > first_cohesion + 0.1:
        cohesion_trend = "improving"
    elif last_cohesion < first_cohesion - 0.1:
        cohesion_trend = "fragmenting"
    else:
        cohesion_trend = "stable"

    # Detect anomalies
    anomalies = []

    if len(nodes_added) > len(first_nodes) * 0.5:
        anomalies.append(f"Rapid fleet growth: {len(nodes_added)} new deployments")

    if len(nodes_removed) > len(first_nodes) * 0.2:
        anomalies.append(f"Significant churn: {len(nodes_removed)} deployments removed")

    if len(patterns_declining) > len(patterns_spreading) * 2:
        anomalies.append("More patterns declining than spreading")

    return EvolutionReport(
        snapshots=snapshots,
        nodes_added=nodes_added,
        nodes_removed=nodes_removed,
        patterns_spreading=patterns_spreading,
        patterns_declining=patterns_declining,
        cohesion_trend=cohesion_trend,
        anomalies=anomalies,
    )


# =============================================================================
# Export and Visualization
# =============================================================================

def to_json(graph: DeploymentGraph, include_edges: bool = True) -> str:
    """
    Export graph as JSON string.

    Args:
        graph: DeploymentGraph to export
        include_edges: Whether to include edges (default True)

    Returns:
        JSON string representation
    """
    data = {
        "nodes": [n.to_dict() for n in graph.nodes.values()],
        "created_at": graph.created_at,
        "packet_count": graph.packet_count,
    }

    if include_edges:
        data["edges"] = [e.to_dict() for e in graph.edges]
    else:
        data["edges"] = []

    return json.dumps(data, indent=2)


def to_dot(graph: DeploymentGraph) -> str:
    """
    Export graph as GraphViz DOT format.

    Args:
        graph: DeploymentGraph to export

    Returns:
        DOT format string for visualization
    """
    lines = [
        "digraph DeploymentGraph {",
        "  rankdir=LR;",
        "  node [shape=box, style=filled];",
        "",
    ]

    # Color nodes by hook
    hook_colors = {
        "tesla": "#e74c3c",
        "spacex": "#3498db",
        "starlink": "#9b59b6",
        "boring": "#f1c40f",
        "neuralink": "#1abc9c",
        "xai": "#e67e22",
        "unknown": "#95a5a6",
    }

    # Add nodes
    for nid, node in graph.nodes.items():
        color = hook_colors.get(node.hook, "#95a5a6")
        style = "filled,dashed" if node.is_stale else "filled"
        label = f"{nid}\\n{node.hook}\\npatterns:{len(node.patterns)}"
        lines.append(f'  "{nid}" [label="{label}", fillcolor="{color}", style="{style}"];')

    lines.append("")

    # Add edges (undirected graph, use -- but DOT digraph uses ->)
    seen_edges = set()
    for edge in graph.edges:
        edge_key = tuple(sorted([edge.source, edge.target]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        # Edge weight affects line width
        penwidth = 1 + edge.weight * 3
        label = f"{edge.similarity_score:.2f}"
        lines.append(
            f'  "{edge.source}" -> "{edge.target}" '
            f'[dir=none, penwidth={penwidth:.1f}, label="{label}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def save(
    graph: DeploymentGraph,
    output_path: str = "deployment_graph.json"
) -> None:
    """
    Save graph to JSON file.

    Args:
        graph: DeploymentGraph to save
        output_path: File path to write to
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(graph))


def export_for_ui(graph: DeploymentGraph) -> dict:
    """
    Export graph in simplified format for dashboards.

    Includes x/y positions computed via simple force-directed layout.

    Args:
        graph: DeploymentGraph to export

    Returns:
        Dictionary with nodes and edges for UI consumption
    """
    # Simple force-directed layout
    positions: Dict[str, Tuple[float, float]] = {}
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)

    if n == 0:
        return {"nodes": [], "edges": []}

    # Initial circular layout
    for i, nid in enumerate(node_ids):
        angle = 2 * math.pi * i / n
        positions[nid] = (
            500 + 300 * math.cos(angle),
            500 + 300 * math.sin(angle),
        )

    # Simple force simulation (10 iterations)
    for _ in range(10):
        forces: Dict[str, Tuple[float, float]] = {nid: (0.0, 0.0) for nid in node_ids}

        # Repulsion between all nodes
        for i, nid_a in enumerate(node_ids):
            xa, ya = positions[nid_a]
            for nid_b in node_ids[i + 1:]:
                xb, yb = positions[nid_b]
                dx, dy = xa - xb, ya - yb
                dist = max(1, math.sqrt(dx * dx + dy * dy))
                force = 5000 / (dist * dist)
                fx, fy = force * dx / dist, force * dy / dist
                fa = forces[nid_a]
                fb = forces[nid_b]
                forces[nid_a] = (fa[0] + fx, fa[1] + fy)
                forces[nid_b] = (fb[0] - fx, fb[1] - fy)

        # Attraction along edges
        for edge in graph.edges:
            if edge.source in positions and edge.target in positions:
                xa, ya = positions[edge.source]
                xb, yb = positions[edge.target]
                dx, dy = xb - xa, yb - ya
                dist = max(1, math.sqrt(dx * dx + dy * dy))
                force = edge.weight * dist * 0.01
                fx, fy = force * dx / dist, force * dy / dist
                fa = forces[edge.source]
                fb = forces[edge.target]
                forces[edge.source] = (fa[0] + fx, fa[1] + fy)
                forces[edge.target] = (fb[0] - fx, fb[1] - fy)

        # Apply forces
        for nid in node_ids:
            x, y = positions[nid]
            fx, fy = forces[nid]
            positions[nid] = (x + fx * 0.1, y + fy * 0.1)

    # Build output
    nodes_out = []
    for nid, node in graph.nodes.items():
        x, y = positions.get(nid, (500, 500))
        nodes_out.append({
            "id": nid,
            "x": round(x, 2),
            "y": round(y, 2),
            "hook": node.hook,
            "patterns_count": len(node.patterns),
            "health_score": node.metrics.health_score,
            "is_stale": node.is_stale,
        })

    edges_out = [
        {
            "source": e.source,
            "target": e.target,
            "similarity": round(e.similarity_score, 3),
        }
        for e in graph.edges
    ]

    return {
        "nodes": nodes_out,
        "edges": edges_out,
    }


# =============================================================================
# Graph Health Diagnosis
# =============================================================================

def diagnose(graph: DeploymentGraph) -> GraphDiagnosis:
    """
    Diagnose graph health and generate recommendations.

    Checks:
    - Stale node ratio > 20% -> warning
    - Orphan ratio > 10% -> issue
    - Cohesion < 0.3 -> "fleet is fragmented"
    - Any deployment with breach_rate > 0.01 -> "deployment X needs attention"
    - Pattern used by <2 deployments -> "pattern Y is underutilized"

    Args:
        graph: DeploymentGraph to diagnose

    Returns:
        GraphDiagnosis with health status and recommendations
    """
    warnings: List[str] = []
    issues: List[str] = []
    recommendations: List[str] = []

    metrics = compute_fleet_metrics(graph)
    n_nodes = metrics.total_deployments

    if n_nodes == 0:
        return GraphDiagnosis(
            is_healthy=True,
            warnings=["Graph is empty"],
            issues=[],
            recommendations=["Add deployments to the graph"],
        )

    # Check stale ratio
    stale_ratio = metrics.stale_deployments / n_nodes
    if stale_ratio > 0.2:
        warnings.append(
            f"Stale deployment ratio ({stale_ratio:.0%}) exceeds 20% threshold"
        )
        recommendations.append("Review and update stale deployments")

    # Check orphan ratio
    orphan_ratio = metrics.orphan_count / n_nodes
    if orphan_ratio > 0.1:
        issues.append(
            f"Orphan ratio ({orphan_ratio:.0%}) exceeds 10% - {metrics.orphan_count} isolated deployments"
        )
        recommendations.append("Investigate isolated deployments for integration opportunities")

    # Check cohesion
    if metrics.fleet_cohesion < 0.3:
        issues.append(
            f"Fleet cohesion ({metrics.fleet_cohesion:.2f}) below 0.3 - fleet is fragmented"
        )
        recommendations.append("Consider standardizing patterns across deployments")

    # Check individual deployments
    high_breach_deployments = []
    for nid, node in graph.nodes.items():
        if node.metrics.breach_rate > 0.01:
            high_breach_deployments.append((nid, node.metrics.breach_rate))

    if high_breach_deployments:
        for nid, rate in high_breach_deployments[:5]:  # Limit to 5
            warnings.append(f"Deployment {nid} has high breach rate ({rate:.2%})")
        if len(high_breach_deployments) > 5:
            warnings.append(f"... and {len(high_breach_deployments) - 5} more")
        recommendations.append("Review deployments with high breach rates")

    # Check underutilized patterns
    pattern_counts: Dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        for p in node.patterns:
            pattern_counts[p] += 1

    underutilized = [p for p, count in pattern_counts.items() if count < 2]
    if underutilized:
        for p in underutilized[:5]:  # Limit to 5
            warnings.append(f"Pattern {p} is underutilized (<2 deployments)")
        if len(underutilized) > 5:
            warnings.append(f"... and {len(underutilized) - 5} more underutilized patterns")
        recommendations.append("Consider propagating underutilized patterns or retiring them")

    # Determine overall health
    is_healthy = len(issues) == 0

    return GraphDiagnosis(
        is_healthy=is_healthy,
        warnings=warnings,
        issues=issues,
        recommendations=recommendations,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # v2 compatibility
    'load_manifest',
    'build_company_table',
    'compute_exploit_count',
    'compute_cross_domain_links',
    # v3 new
    'build',
    'DeploymentGraph',
    'DeploymentNode',
    'DeploymentEdge',
    'NodeMetrics',
    'FleetSummary',
    'Cluster',
    'PropagationCandidate',
    'PatternRecommendation',
    'GraphSnapshot',
    'EvolutionReport',
    'GraphDiagnosis',
    'compute_fleet_metrics',
    'predict_propagation',
    'recommend_patterns',
    'track_evolution',
    'diagnose',
    'to_json',
    'to_dot',
    'save',
    'export_for_ui',
]
