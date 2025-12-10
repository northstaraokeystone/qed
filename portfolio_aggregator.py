"""
portfolio_aggregator.py - v9 Cross-Deployment Portfolio Synthesis via Entanglement Physics

Paradigm Shift: Entanglement-Weighted Aggregation, Not Sums
============================================================

v9 replaces simple portfolio summation with entanglement-weighted combination.
Companies are observation lenses, not containers. Value is topology, not dollars.

Key Innovations:
1. Entanglement weighting - high entanglement REDUCES contribution weight
2. Systemic risk detection - high entanglement + low centrality = danger zone
3. Portfolio health as f(entanglement, centrality, self_compression)
4. Companies as lenses - patterns exist independently, companies observe them

What does NOT exist in this file:
- sum(values) or total = a + b + c for portfolio value
- Per-company value fields that get summed
- "Tesla_value + SpaceX_value" pattern
- Hard-coded dollar amounts (value is centrality)

SLO Enforcement:
- Entanglement SLO >= 0.92 for patterns in same physics domain
- Reference: SDD line 305-306
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from causal_graph import (
    centrality,
    self_compression_ratio,
    entanglement_coefficient,
    ENTANGLEMENT_SLO,
)
from binder import compute_value, QueryPredicate


# =============================================================================
# Constants
# =============================================================================

# Valid company identifiers (observation lenses)
COMPANIES: List[str] = ["tesla", "spacex", "starlink", "boring", "neuralink", "xai"]

# Danger zone thresholds
DANGER_ENTANGLEMENT_THRESHOLD = 0.8  # High entanglement
DANGER_CENTRALITY_THRESHOLD = 0.3    # Low centrality

# Health interpretation thresholds
HEALTH_HEALTHY = 0.5
HEALTH_CAUTION = 0.3


# =============================================================================
# Receipt Schema (self-describing module contract)
# =============================================================================

RECEIPT_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "portfolio_receipt",
        "version": "1.0.0",
        "description": "Receipt emitted by aggregate() - entanglement-weighted portfolio value",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of input parameters",
            "timestamp": "ISO UTC timestamp",
            "input_packet_ids": "List of packet IDs processed",
            "companies": "List[str] - companies included in aggregation",
            "total_value": "float - entanglement-weighted value (NOT sum)",
            "diversification_score": "float - inverse of avg entanglement",
            "pattern_contributions": "Dict[pattern_id, float] - weighted contributions",
            "entanglement_warnings": "List[str] - patterns below SLO",
            "reversible": "bool - always True",
            "reverse_action": "str - how to reverse this operation",
        },
    },
    {
        "type": "risk_receipt",
        "version": "1.0.0",
        "description": "Receipt emitted by systemic_risk() - danger pattern detection",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of risk query",
            "timestamp": "ISO UTC timestamp",
            "companies": "List[str] - companies analyzed",
            "risk_score": "float [0,1] - higher = more systemic risk",
            "danger_patterns": "List[Dict] - patterns with high entanglement + low centrality",
            "total_patterns": "int - total patterns analyzed",
            "threshold_entanglement": "float - entanglement threshold used",
            "threshold_centrality": "float - centrality threshold used",
        },
    },
    {
        "type": "health_receipt",
        "version": "1.0.0",
        "description": "Receipt emitted by portfolio_health() - composite health metric",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of health query",
            "timestamp": "ISO UTC timestamp",
            "companies": "List[str] - companies analyzed",
            "health_score": "float [0,1] - higher = healthier",
            "health_status": "str - 'healthy', 'caution', or 'unhealthy'",
            "components": {
                "avg_centrality": "float - average pattern centrality",
                "avg_entanglement": "float - average cross-company entanglement",
                "self_compression": "float - graph self-compression ratio",
            },
        },
    },
]


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class CompanyLens:
    """
    Observation lens for a company.

    Companies are not containers of patterns - they are lenses through which
    patterns are observed. The same pattern P exists independently and is
    observed through Tesla's lens, SpaceX's lens, etc.

    Attributes:
        company_id: Company identifier (tesla, spacex, starlink, boring, neuralink, xai)
        pattern_ids: Set of pattern IDs observed through this lens
        graph_view: Optional subgraph representing this company's view
    """
    company_id: str
    pattern_ids: Set[str] = field(default_factory=frozenset)
    graph_view: Optional[nx.DiGraph] = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        if self.company_id not in COMPANIES:
            raise ValueError(f"Invalid company_id: {self.company_id}. Must be one of {COMPANIES}")


@dataclass(frozen=True)
class PortfolioMetrics:
    """
    Aggregated portfolio metrics using entanglement-weighted computation.

    Attributes:
        total_value: Entanglement-weighted portfolio value (NOT simple sum)
        systemic_risk_score: Risk score [0,1], higher = more dangerous
        health_score: Health score [0,1], higher = healthier
        entanglement_matrix: pattern_id -> company_id -> entanglement coefficient
        diversification_score: Inverse of average entanglement
        timestamp: ISO UTC timestamp of computation
    """
    total_value: float
    systemic_risk_score: float
    health_score: float
    entanglement_matrix: Dict[str, Dict[str, float]]
    diversification_score: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_value": self.total_value,
            "systemic_risk_score": self.systemic_risk_score,
            "health_score": self.health_score,
            "entanglement_matrix": self.entanglement_matrix,
            "diversification_score": self.diversification_score,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Receipt Dataclasses
# =============================================================================

@dataclass
class PortfolioReceipt:
    """Receipt emitted by aggregate() function."""
    receipt_id: str
    timestamp: str
    input_packet_ids: List[str]
    companies: List[str]
    total_value: float
    diversification_score: float
    pattern_contributions: Dict[str, float]
    entanglement_warnings: List[str]
    reversible: bool = True
    reverse_action: str = "Disaggregate by reprocessing packets individually per company"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "receipt_type": "portfolio_receipt",
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "input_packet_ids": self.input_packet_ids,
            "companies": self.companies,
            "total_value": self.total_value,
            "diversification_score": self.diversification_score,
            "pattern_contributions": self.pattern_contributions,
            "entanglement_warnings": self.entanglement_warnings,
            "reversible": self.reversible,
            "reverse_action": self.reverse_action,
        }


@dataclass
class RiskReceipt:
    """Receipt emitted by systemic_risk() function."""
    receipt_id: str
    timestamp: str
    companies: List[str]
    risk_score: float
    danger_patterns: List[Dict[str, Any]]
    total_patterns: int
    threshold_entanglement: float
    threshold_centrality: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "receipt_type": "risk_receipt",
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "companies": self.companies,
            "risk_score": self.risk_score,
            "danger_patterns": self.danger_patterns,
            "total_patterns": self.total_patterns,
            "threshold_entanglement": self.threshold_entanglement,
            "threshold_centrality": self.threshold_centrality,
        }


@dataclass
class HealthReceipt:
    """Receipt emitted by portfolio_health() function."""
    receipt_id: str
    timestamp: str
    companies: List[str]
    health_score: float
    health_status: str
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "receipt_type": "health_receipt",
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "companies": self.companies,
            "health_score": self.health_score,
            "health_status": self.health_status,
            "components": self.components,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_receipt_id(content: Dict[str, Any]) -> str:
    """Generate deterministic receipt_id from content."""
    serialized = json.dumps(content, separators=(",", ":"), sort_keys=True)
    return hashlib.sha3_256(serialized.encode()).hexdigest()[:16]


def _get_all_patterns(graph: nx.DiGraph) -> Set[str]:
    """Extract all pattern nodes from graph."""
    patterns = set()
    for node in graph.nodes():
        if graph.nodes[node].get("is_pattern", False):
            patterns.add(node)
        elif isinstance(node, str) and node.startswith("pattern_"):
            patterns.add(node)
    # If no explicit pattern markers, treat all nodes as patterns
    if not patterns:
        patterns = set(graph.nodes())
    return patterns


def get_shared_patterns(
    graph: nx.DiGraph,
    companies: List[str],
) -> Dict[str, Set[str]]:
    """
    Get patterns shared across multiple companies.

    Args:
        graph: Flow network with company-tagged nodes
        companies: Companies to analyze

    Returns:
        Dict mapping pattern_id to set of companies observing it.
        Only includes patterns observed by > 1 company (truly shared).
    """
    pattern_companies: Dict[str, Set[str]] = {}

    for node in graph.nodes():
        node_data = graph.nodes[node]
        company = node_data.get("company")

        if company and company in companies:
            # Find patterns connected to this node
            for neighbor in graph.neighbors(node):
                neighbor_data = graph.nodes.get(neighbor, {})
                if neighbor_data.get("is_pattern", False):
                    if neighbor not in pattern_companies:
                        pattern_companies[neighbor] = set()
                    pattern_companies[neighbor].add(company)

    # Also check for patterns directly in nodes
    for node in graph.nodes():
        if graph.nodes[node].get("is_pattern", False):
            # Check which companies have edges to this pattern
            for pred in graph.predecessors(node):
                pred_company = graph.nodes[pred].get("company")
                if pred_company and pred_company in companies:
                    if node not in pattern_companies:
                        pattern_companies[node] = set()
                    pattern_companies[node].add(pred_company)

    # Filter to only shared patterns (observed by > 1 company)
    shared = {p: c for p, c in pattern_companies.items() if len(c) > 1}

    return shared


def build_entanglement_matrix(
    graph: nx.DiGraph,
    patterns: List[str],
    companies: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Build entanglement matrix for visualization.

    Args:
        graph: Flow network
        patterns: Pattern IDs to include
        companies: Companies to compute entanglement across

    Returns:
        Nested dict: pattern_id -> company_id -> entanglement_coefficient
        Enables heatmap visualization of entanglement structure.
    """
    matrix: Dict[str, Dict[str, float]] = {}

    for pattern_id in patterns:
        matrix[pattern_id] = {}

        # Compute entanglement for each company subset
        for company in companies:
            # Entanglement of this pattern across this company and all others
            other_companies = [c for c in companies if c != company]
            if other_companies:
                coeff = entanglement_coefficient(
                    pattern_id,
                    [company] + other_companies,
                    graph,
                )
            else:
                coeff = 1.0  # Single company = perfect entanglement with itself

            matrix[pattern_id][company] = coeff

    return matrix


# =============================================================================
# Core Functions (R -> R Transformers)
# =============================================================================

def aggregate(
    packets: List[Dict[str, Any]],
    graph: nx.DiGraph,
    companies: List[str],
) -> List[Dict[str, Any]]:
    """
    Core R -> R transformer: entanglement-weighted portfolio aggregation.

    REPLACES simple sum. High entanglement REDUCES contribution weight.

    Formula (critical):
        For pattern P across companies C:
            e = entanglement_coefficient(P, C, graph)  # [0,1]
            c = centrality(P, graph)                   # [0,1]
            weight = 1.0 / (1.0 + e)                   # high entanglement → low weight
            contribution = c * weight

        total_value = Σ contribution for all patterns

    Args:
        packets: Input packets (for receipt ID generation)
        graph: Flow network with pattern and company nodes
        companies: Company identifiers for cross-company analysis

    Returns:
        List containing portfolio_receipt with entanglement-weighted metrics
    """
    if not graph.nodes():
        return []

    # Extract packet IDs for receipt tracking
    packet_ids = []
    for p in packets:
        pid = p.get("packet_id") or p.get("receipt_id") or p.get("window_id")
        if pid:
            packet_ids.append(pid)

    # Get all patterns in the graph
    all_patterns = _get_all_patterns(graph)

    if not all_patterns:
        return []

    # Compute entanglement-weighted contributions
    pattern_contributions: Dict[str, float] = {}
    entanglement_warnings: List[str] = []
    total_entanglement = 0.0
    pattern_count = 0

    for pattern_id in all_patterns:
        # Get entanglement coefficient across all companies
        e = entanglement_coefficient(pattern_id, companies, graph)

        # Get centrality (value from topology)
        c = centrality(pattern_id, graph)

        # Entanglement weighting: high entanglement REDUCES weight
        # weight = 1.0 / (1.0 + e)
        # When e=0 (no entanglement/diversified): weight=1.0 (full contribution)
        # When e=1 (max entanglement/concentrated): weight=0.5 (half contribution)
        weight = 1.0 / (1.0 + e)

        # Weighted contribution
        contribution = c * weight
        pattern_contributions[pattern_id] = contribution

        # Track entanglement for diversification score
        total_entanglement += e
        pattern_count += 1

        # SLO enforcement: warn if entanglement below threshold for shared patterns
        # Shared patterns SHOULD have high entanglement (coherent across companies)
        if e < ENTANGLEMENT_SLO and e > 0:
            entanglement_warnings.append(
                f"Pattern {pattern_id}: entanglement {e:.3f} < SLO {ENTANGLEMENT_SLO}"
            )

    # Total value is sum of weighted contributions (NOT raw centrality sum)
    total_value = sum(pattern_contributions.values())

    # Diversification score = inverse of average entanglement
    avg_entanglement = total_entanglement / pattern_count if pattern_count > 0 else 0.0
    diversification_score = 1.0 / (1.0 + avg_entanglement)

    # Generate receipt ID
    receipt_id = _generate_receipt_id({
        "packet_ids": sorted(packet_ids),
        "companies": sorted(companies),
        "pattern_count": len(all_patterns),
    })

    receipt = PortfolioReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_packet_ids=sorted(packet_ids),
        companies=sorted(companies),
        total_value=total_value,
        diversification_score=diversification_score,
        pattern_contributions=pattern_contributions,
        entanglement_warnings=entanglement_warnings,
    )

    return [receipt.to_dict()]


def systemic_risk(
    graph: nx.DiGraph,
    companies: List[str],
    threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Detect systemic risk from entanglement structure.

    Danger zones: patterns with HIGH entanglement + LOW centrality.
    These are highly connected across companies but provide low individual value.
    If one fails, ripple effects propagate everywhere.

    Formula (critical):
        danger_patterns = [p for p in patterns
                          if entanglement_coefficient(p, companies) >= 0.8
                          AND centrality(p) <= 0.3]

        systemic_risk_score = len(danger_patterns) / len(all_patterns)

    Args:
        graph: Flow network
        companies: Companies to analyze
        threshold: Entanglement threshold for danger (default 0.8)

    Returns:
        List containing risk_receipt with danger patterns and risk score
    """
    all_patterns = _get_all_patterns(graph)

    if not all_patterns:
        receipt = RiskReceipt(
            receipt_id=_generate_receipt_id({"companies": sorted(companies), "empty": True}),
            timestamp=datetime.now(timezone.utc).isoformat(),
            companies=sorted(companies),
            risk_score=0.0,
            danger_patterns=[],
            total_patterns=0,
            threshold_entanglement=threshold,
            threshold_centrality=DANGER_CENTRALITY_THRESHOLD,
        )
        return [receipt.to_dict()]

    danger_patterns: List[Dict[str, Any]] = []

    for pattern_id in all_patterns:
        e = entanglement_coefficient(pattern_id, companies, graph)
        c = centrality(pattern_id, graph)

        # Danger zone: high entanglement (>= 0.8) AND low centrality (<= 0.3)
        if e >= threshold and c <= DANGER_CENTRALITY_THRESHOLD:
            danger_patterns.append({
                "pattern_id": pattern_id,
                "entanglement": e,
                "centrality": c,
                "risk_factor": e * (1.0 - c),  # Higher when high e and low c
            })

    # Systemic risk score = proportion of danger patterns
    risk_score = len(danger_patterns) / len(all_patterns)

    # Sort danger patterns by risk factor descending
    danger_patterns.sort(key=lambda x: x["risk_factor"], reverse=True)

    receipt = RiskReceipt(
        receipt_id=_generate_receipt_id({
            "companies": sorted(companies),
            "threshold": threshold,
            "pattern_count": len(all_patterns),
        }),
        timestamp=datetime.now(timezone.utc).isoformat(),
        companies=sorted(companies),
        risk_score=risk_score,
        danger_patterns=danger_patterns,
        total_patterns=len(all_patterns),
        threshold_entanglement=threshold,
        threshold_centrality=DANGER_CENTRALITY_THRESHOLD,
    )

    return [receipt.to_dict()]


def portfolio_health(
    graph: nx.DiGraph,
    companies: List[str],
) -> List[Dict[str, Any]]:
    """
    Compute composite portfolio health metric.

    Combines three metrics into single health score:
    - avg_centrality: mean centrality of all patterns (value component)
    - avg_entanglement: mean entanglement of shared patterns (risk component)
    - compression: self_compression_ratio of graph (understanding component)

    Formula (critical):
        health = (avg_centrality * self_compression_ratio) / (1.0 + avg_entanglement)

    Where:
        - avg_centrality in [0,1]: higher = more valuable patterns
        - self_compression_ratio in (0,1]: higher = better self-understanding
        - avg_entanglement in [0,1]: higher = more systemic risk

    Health interpretation:
        - health > 0.5: healthy portfolio
        - health 0.3-0.5: caution zone
        - health < 0.3: unhealthy, high systemic risk

    Args:
        graph: Flow network
        companies: Companies to analyze

    Returns:
        List containing health_receipt with composite score and components
    """
    all_patterns = _get_all_patterns(graph)

    if not all_patterns:
        receipt = HealthReceipt(
            receipt_id=_generate_receipt_id({"companies": sorted(companies), "empty": True}),
            timestamp=datetime.now(timezone.utc).isoformat(),
            companies=sorted(companies),
            health_score=1.0,  # Empty = no risk
            health_status="healthy",
            components={
                "avg_centrality": 0.0,
                "avg_entanglement": 0.0,
                "self_compression": 1.0,
            },
        )
        return [receipt.to_dict()]

    # Compute avg_centrality
    total_centrality = 0.0
    for pattern_id in all_patterns:
        total_centrality += centrality(pattern_id, graph)
    avg_centrality = total_centrality / len(all_patterns)

    # Compute avg_entanglement (for shared patterns)
    total_entanglement = 0.0
    entanglement_count = 0
    for pattern_id in all_patterns:
        e = entanglement_coefficient(pattern_id, companies, graph)
        if e > 0:  # Only count patterns with some cross-company presence
            total_entanglement += e
            entanglement_count += 1

    avg_entanglement = total_entanglement / entanglement_count if entanglement_count > 0 else 0.0

    # Compute self-compression ratio
    compression = self_compression_ratio(graph)

    # Health formula: high value + high compression + low entanglement = healthy
    # health = (avg_centrality * compression) / (1.0 + avg_entanglement)
    health_score = (avg_centrality * compression) / (1.0 + avg_entanglement)

    # Clamp to [0, 1]
    health_score = max(0.0, min(1.0, health_score))

    # Determine health status
    if health_score >= HEALTH_HEALTHY:
        health_status = "healthy"
    elif health_score >= HEALTH_CAUTION:
        health_status = "caution"
    else:
        health_status = "unhealthy"

    receipt = HealthReceipt(
        receipt_id=_generate_receipt_id({
            "companies": sorted(companies),
            "pattern_count": len(all_patterns),
        }),
        timestamp=datetime.now(timezone.utc).isoformat(),
        companies=sorted(companies),
        health_score=health_score,
        health_status=health_status,
        components={
            "avg_centrality": avg_centrality,
            "avg_entanglement": avg_entanglement,
            "self_compression": compression,
        },
    )

    return [receipt.to_dict()]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Schema
    "RECEIPT_SCHEMA",
    # Constants
    "COMPANIES",
    "DANGER_ENTANGLEMENT_THRESHOLD",
    "DANGER_CENTRALITY_THRESHOLD",
    "HEALTH_HEALTHY",
    "HEALTH_CAUTION",
    # Dataclasses
    "CompanyLens",
    "PortfolioMetrics",
    "PortfolioReceipt",
    "RiskReceipt",
    "HealthReceipt",
    # Core functions (R -> R transformers)
    "aggregate",
    "systemic_risk",
    "portfolio_health",
    # Helper functions
    "get_shared_patterns",
    "build_entanglement_matrix",
]
