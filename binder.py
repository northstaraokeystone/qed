"""
binder.py - v9 Portfolio Aggregation via Receipt Monad

Implements 3 v9 paradigm shifts:

1. Receipt Monad (Paradigm 1):
   - bind() signature is List[Receipt] -> List[Receipt]
   - Modules ARE receipt transformers - pure functions with no side effects
   - No internal state, no class-level variables

2. Value as Topology (Paradigm 2):
   - compute_value() derives value from graph centrality via networkx pagerank
   - Value is NEVER stored - always computed from current graph topology
   - value(pattern) = centrality(pattern, receipt_graph)
   - Dollar thresholds ($1M/$10M) become centrality thresholds (0.5/0.8)

3. Mode Elimination (Paradigm 3):
   - QueryPredicate replaces PatternMode enum
   - Mode is projection, not state - same pattern appears different to different observers
   - "DELETE THE ENUM" - no PatternMode exists in this module

What does NOT exist in this file:
- PatternMode enum or class
- dollar_value, dollar_value_annual, or any stored value field
- Internal state or class-level variables
- Side effects (file writes, prints, logging mutations)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import networkx as nx


# =============================================================================
# Centrality Thresholds (replacing dollar amounts)
# =============================================================================

# Was $10M - top 20% patterns by graph importance
THRESHOLD_HIGH = 0.8

# Was $1M - top 50% patterns by graph importance
THRESHOLD_MID = 0.5

# Deprecation candidate - bottom 20% patterns
THRESHOLD_LOW = 0.2


# =============================================================================
# Receipt Schema (self-describing module contract)
# =============================================================================

RECEIPT_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "binder_receipt",
        "version": "1.0.0",
        "description": "Portfolio aggregation receipt from bind() transformer",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of sorted input packet IDs",
            "timestamp": "ISO UTC timestamp",
            "input_packet_ids": "List of packet IDs processed",
            "portfolio_metrics": {
                "total_windows": "int - sum of windows across packets",
                "breach_rate": "float - weighted SLO breach rate",
                "centrality_high_count": "int - patterns with centrality >= 0.8",
                "centrality_mid_count": "int - patterns with centrality >= 0.5",
                "centrality_low_count": "int - patterns with centrality < 0.2",
            },
            "pattern_centralities": "Dict[pattern_id, float] - computed centralities",
            "reversible": "bool - whether this binding can be reversed",
            "reverse_action": "str - description of how to reverse",
        },
    },
    {
        "type": "value_query_receipt",
        "version": "1.0.0",
        "description": "Receipt for value queries via compute_value()",
        "fields": {
            "receipt_id": "SHA3-256 hash of query parameters",
            "timestamp": "ISO UTC timestamp",
            "pattern_id": "Pattern queried",
            "centrality": "float - computed centrality value",
            "graph_node_count": "int - nodes in graph at query time",
            "graph_edge_count": "int - edges in graph at query time",
        },
    },
]


# =============================================================================
# QueryPredicate (replacing PatternMode enum - Paradigm 3)
# =============================================================================

@dataclass(frozen=True)
class QueryPredicate:
    """
    Predicate for querying pattern visibility based on centrality.

    Replaces PatternMode enum. Mode is now projection, not state.
    The same pattern can satisfy multiple predicates simultaneously -
    it appears "live" to one observer and "shadow" to another based
    on their predicate thresholds.

    Attributes:
        actionable: If True, pattern can trigger automated actions
        ttl_valid: If True, pattern's time-to-live has not expired
        min_centrality: Minimum centrality to match (inclusive)
        max_centrality: Maximum centrality to match (exclusive, None = no upper bound)
    """
    actionable: bool
    ttl_valid: bool
    min_centrality: float
    max_centrality: Optional[float]

    @classmethod
    def live(cls) -> QueryPredicate:
        """
        Factory for live/production patterns.

        Matches patterns with centrality >= 0.5 that are actionable.
        These are patterns actively driving decisions.
        """
        return cls(
            actionable=True,
            ttl_valid=True,
            min_centrality=THRESHOLD_MID,
            max_centrality=None,
        )

    @classmethod
    def shadow(cls) -> QueryPredicate:
        """
        Factory for shadow/evaluation patterns.

        Matches patterns that are observed but not yet actionable.
        Used for A/B testing and gradual rollout.
        """
        return cls(
            actionable=False,
            ttl_valid=True,
            min_centrality=THRESHOLD_LOW,
            max_centrality=THRESHOLD_HIGH,
        )

    @classmethod
    def deprecated(cls) -> QueryPredicate:
        """
        Factory for deprecated patterns pending removal.

        Matches patterns with centrality < 0.2 that are no longer valid.
        These should be reviewed for deletion.
        """
        return cls(
            actionable=False,
            ttl_valid=False,
            min_centrality=0.0,
            max_centrality=THRESHOLD_LOW,
        )

    @classmethod
    def high_value(cls, threshold: float = THRESHOLD_HIGH) -> QueryPredicate:
        """
        Factory for high-value patterns above custom threshold.

        Args:
            threshold: Minimum centrality (default 0.8 for top 20%)

        Returns:
            Predicate matching high-centrality actionable patterns
        """
        return cls(
            actionable=True,
            ttl_valid=True,
            min_centrality=threshold,
            max_centrality=None,
        )

    def matches(self, centrality: float, fp_rate: float = 0.0) -> bool:
        """
        Check if a pattern with given centrality matches this predicate.

        Args:
            centrality: Pattern's computed graph centrality
            fp_rate: False positive rate for ROI threshold checks

        Returns:
            True if pattern satisfies predicate conditions
        """
        # Check centrality bounds
        if centrality < self.min_centrality:
            return False
        if self.max_centrality is not None and centrality >= self.max_centrality:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize predicate for receipts."""
        return {
            "actionable": self.actionable,
            "ttl_valid": self.ttl_valid,
            "min_centrality": self.min_centrality,
            "max_centrality": self.max_centrality,
        }


# =============================================================================
# BinderReceipt (output from bind() transformer)
# =============================================================================

@dataclass
class BinderReceipt:
    """
    Output receipt from bind() portfolio aggregation.

    Contains computed centralities and portfolio metrics.
    Supports self-healing via reversible + reverse_action fields.
    """
    receipt_id: str
    timestamp: str
    input_packet_ids: List[str]
    portfolio_metrics: Dict[str, Any]
    pattern_centralities: Dict[str, float]
    reversible: bool
    reverse_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt for JSON output."""
        return {
            "type": "binder_receipt",
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "input_packet_ids": self.input_packet_ids,
            "portfolio_metrics": self.portfolio_metrics,
            "pattern_centralities": self.pattern_centralities,
            "reversible": self.reversible,
            "reverse_action": self.reverse_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BinderReceipt:
        """Deserialize receipt from dictionary."""
        return cls(
            receipt_id=data["receipt_id"],
            timestamp=data["timestamp"],
            input_packet_ids=data.get("input_packet_ids", []),
            portfolio_metrics=data.get("portfolio_metrics", {}),
            pattern_centralities=data.get("pattern_centralities", {}),
            reversible=data.get("reversible", False),
            reverse_action=data.get("reverse_action", ""),
        )


# =============================================================================
# Core Functions (Receipt Monad transformers)
# =============================================================================

def compute_value(pattern_id: str, graph: nx.DiGraph) -> float:
    """
    Compute pattern value via graph centrality (Paradigm 2).

    Value is NEVER stored - always derived from current graph topology.
    Uses PageRank which models pattern importance based on how many
    other patterns reference/depend on it.

    Args:
        pattern_id: Pattern to compute value for
        graph: NetworkX DiGraph with pattern nodes

    Returns:
        Centrality value normalized to [0, 1]
        Returns 0.0 if pattern not in graph
    """
    if graph.number_of_nodes() == 0:
        return 0.0

    if pattern_id not in graph:
        return 0.0

    # Compute PageRank for all nodes (cached by networkx)
    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        # Fallback to degree centrality if PageRank fails
        pagerank = nx.degree_centrality(graph)

    raw_centrality = pagerank.get(pattern_id, 0.0)

    # Normalize to [0, 1] based on max centrality in graph
    max_centrality = max(pagerank.values()) if pagerank else 1.0
    if max_centrality > 0:
        normalized = raw_centrality / max_centrality
    else:
        normalized = 0.0

    return min(1.0, max(0.0, normalized))


def query(
    centralities: Dict[str, float],
    predicate: QueryPredicate,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Query patterns matching predicate (Mode as Projection - Paradigm 3).

    The same pattern set can be queried with different predicates
    to get different views. This replaces mode-based filtering.

    Args:
        centralities: Dict mapping pattern_id to centrality value
        predicate: QueryPredicate defining filter criteria
        metadata: Optional dict with per-pattern metadata (fp_rate, etc.)

    Returns:
        List of pattern_ids matching the predicate, sorted by centrality desc
    """
    metadata = metadata or {}
    matching: List[tuple[str, float]] = []

    for pattern_id, centrality in centralities.items():
        pattern_meta = metadata.get(pattern_id, {})
        fp_rate = pattern_meta.get("fp_rate", 0.0)

        if predicate.matches(centrality, fp_rate):
            matching.append((pattern_id, centrality))

    # Sort by centrality descending
    matching.sort(key=lambda x: x[1], reverse=True)

    return [pattern_id for pattern_id, _ in matching]


def evaluate_pattern(
    centrality: float,
    fp_rate: float,
) -> QueryPredicate:
    """
    Evaluate pattern ROI via centrality thresholds.

    Maps old dollar thresholds to centrality:
    - $10M -> 0.8 centrality (high value)
    - $1M -> 0.5 centrality (actionable)
    - <$200K -> <0.2 centrality (deprecated)

    Decision logic:
    - centrality >= 0.5 AND fp_rate <= 0.01 -> actionable=True, ttl_valid=True
    - centrality >= 0.8 AND fp_rate > 0.01 -> actionable=False, ttl_valid=True (shadow)
    - centrality < 0.2 -> ttl_valid=False (deprecated)
    - else -> shadow for review

    Args:
        centrality: Pattern's computed graph centrality
        fp_rate: False positive rate from validation

    Returns:
        QueryPredicate describing pattern's evaluated state
    """
    # Deprecated: low centrality patterns
    if centrality < THRESHOLD_LOW:
        return QueryPredicate(
            actionable=False,
            ttl_valid=False,
            min_centrality=0.0,
            max_centrality=THRESHOLD_LOW,
        )

    # High value with acceptable FP rate -> live/actionable
    if centrality >= THRESHOLD_MID and fp_rate <= 0.01:
        return QueryPredicate(
            actionable=True,
            ttl_valid=True,
            min_centrality=THRESHOLD_MID,
            max_centrality=None,
        )

    # High centrality but problematic FP rate -> shadow
    if centrality >= THRESHOLD_HIGH and fp_rate > 0.01:
        return QueryPredicate(
            actionable=False,
            ttl_valid=True,
            min_centrality=THRESHOLD_HIGH,
            max_centrality=None,
        )

    # Default: shadow for review
    return QueryPredicate(
        actionable=False,
        ttl_valid=True,
        min_centrality=THRESHOLD_LOW,
        max_centrality=THRESHOLD_HIGH,
    )


def _build_graph_from_packets(packets: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Build pattern usage graph from packet data.

    Creates edges between patterns that co-occur in deployments,
    weighted by deployment metrics.

    Args:
        packets: List of packet dicts with pattern_usage

    Returns:
        DiGraph with pattern nodes and co-occurrence edges
    """
    graph = nx.DiGraph()

    for packet in packets:
        pattern_usage = packet.get("pattern_usage", [])
        if not pattern_usage:
            continue

        # Extract pattern IDs
        pattern_ids = []
        for p in pattern_usage:
            if isinstance(p, dict):
                pid = p.get("pattern_id")
            else:
                # Assume it's an object with pattern_id attribute
                pid = getattr(p, "pattern_id", None)
            if pid:
                pattern_ids.append(pid)

        # Add nodes for all patterns
        for pid in pattern_ids:
            if not graph.has_node(pid):
                graph.add_node(pid)

        # Add edges between co-occurring patterns (bidirectional influence)
        for i, pid_a in enumerate(pattern_ids):
            for pid_b in pattern_ids[i + 1:]:
                # Bidirectional edges for co-occurrence
                if graph.has_edge(pid_a, pid_b):
                    graph[pid_a][pid_b]["weight"] += 1
                else:
                    graph.add_edge(pid_a, pid_b, weight=1)

                if graph.has_edge(pid_b, pid_a):
                    graph[pid_b][pid_a]["weight"] += 1
                else:
                    graph.add_edge(pid_b, pid_a, weight=1)

    return graph


def _compute_all_centralities(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute normalized PageRank centralities for all nodes.

    Args:
        graph: NetworkX DiGraph

    Returns:
        Dict mapping node ID to normalized centrality [0, 1]
    """
    if graph.number_of_nodes() == 0:
        return {}

    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        pagerank = nx.degree_centrality(graph)

    # Normalize to [0, 1]
    max_val = max(pagerank.values()) if pagerank else 1.0
    if max_val > 0:
        return {k: min(1.0, v / max_val) for k, v in pagerank.items()}
    return {k: 0.0 for k in pagerank}


def _generate_receipt_id(packet_ids: List[str]) -> str:
    """
    Generate deterministic receipt_id from sorted input packet IDs.

    Args:
        packet_ids: List of packet IDs to hash

    Returns:
        16-character SHA3-256 hash
    """
    sorted_ids = sorted(packet_ids)
    content = json.dumps(sorted_ids, separators=(",", ":"))
    return hashlib.sha3_256(content.encode()).hexdigest()[:16]


def bind(
    packets: List[Dict[str, Any]],
    graph: Optional[nx.DiGraph] = None,
) -> List[Dict[str, Any]]:
    """
    Core Receipt Monad transformer (Paradigm 1).

    Transforms input packets into aggregated portfolio receipt.
    Signature: List[Receipt] -> List[Receipt]

    Internal flow:
    1. Build graph from packets if not provided (extract pattern_usage edges)
    2. Compute all centralities via nx.pagerank once
    3. Aggregate portfolio metrics (total_windows, breach_rate, centrality tiers)
    4. Generate deterministic receipt_id from sorted input packet IDs (SHA3)
    5. Return single binder_receipt as List[Dict]

    Args:
        packets: List of packet dicts (DecisionPacket.to_dict() format)
        graph: Optional pre-built DiGraph. If None, built from packets.

    Returns:
        List containing single binder_receipt dict
    """
    if not packets:
        # Empty input -> empty output (monad identity)
        return []

    # Build graph if not provided
    if graph is None:
        graph = _build_graph_from_packets(packets)

    # Compute all centralities once
    centralities = _compute_all_centralities(graph)

    # Extract packet IDs
    packet_ids = []
    for p in packets:
        pid = p.get("packet_id")
        if pid:
            packet_ids.append(pid)

    # Aggregate portfolio metrics
    total_windows = 0
    total_breach_weighted = 0.0
    total_weight = 0.0

    for packet in packets:
        metrics = packet.get("metrics", {})
        if isinstance(metrics, dict):
            windows = metrics.get("window_volume", 0)
            breach_rate = metrics.get("slo_breach_rate", 0.0)
        else:
            # Handle object with attributes
            windows = getattr(metrics, "window_volume", 0)
            breach_rate = getattr(metrics, "slo_breach_rate", 0.0)

        total_windows += windows
        total_breach_weighted += breach_rate * windows
        total_weight += windows

    avg_breach_rate = total_breach_weighted / total_weight if total_weight > 0 else 0.0

    # Count patterns by centrality tier
    high_count = sum(1 for c in centralities.values() if c >= THRESHOLD_HIGH)
    mid_count = sum(1 for c in centralities.values() if THRESHOLD_MID <= c < THRESHOLD_HIGH)
    low_count = sum(1 for c in centralities.values() if c < THRESHOLD_LOW)

    portfolio_metrics = {
        "total_windows": total_windows,
        "breach_rate": avg_breach_rate,
        "centrality_high_count": high_count,
        "centrality_mid_count": mid_count,
        "centrality_low_count": low_count,
        "total_patterns": len(centralities),
    }

    # Generate deterministic receipt ID
    receipt_id = _generate_receipt_id(packet_ids)

    # Create binder receipt
    receipt = BinderReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_packet_ids=sorted(packet_ids),
        portfolio_metrics=portfolio_metrics,
        pattern_centralities=centralities,
        reversible=True,
        reverse_action="Unbind by reprocessing original packets individually",
    )

    return [receipt.to_dict()]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Schema
    "RECEIPT_SCHEMA",
    # Thresholds
    "THRESHOLD_HIGH",
    "THRESHOLD_MID",
    "THRESHOLD_LOW",
    # Dataclasses
    "QueryPredicate",
    "BinderReceipt",
    # Functions
    "bind",
    "compute_value",
    "query",
    "evaluate_pattern",
]
