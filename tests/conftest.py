"""
Shared pytest fixtures for QED v9 tests.

Provides test data fixtures matching v9 schemas:
- temp_data_dir: Temporary directory with graph/, events/, receipts.jsonl
- sample_receipts: Test receipt dicts with type, id, pattern_id, timestamp
- sample_graph_with_nodes: Graph with nodes A, B, C and edges for testing
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pytest


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Create temporary data directory with v9 schema structure.

    Structure:
        temp_data_dir/
        ├── graph/
        │   ├── flow_network.json
        │   └── centrality_cache.json
        ├── events/
        │   └── events.jsonl
        └── receipts.jsonl

    Returns:
        Path: Path to temporary data directory
    """
    # Create subdirectories
    graph_dir = tmp_path / "graph"
    events_dir = tmp_path / "events"
    graph_dir.mkdir()
    events_dir.mkdir()

    # Create flow_network.json (empty v9 schema)
    flow_network = {
        "schema_version": "v9.0",
        "graph_type": "bidirectional_flow",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_modified": datetime.now(timezone.utc).isoformat(),
        "content_hash": {
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "blake3": "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
        },
        "nodes": {},
        "edges": [],
        "statistics": {
            "node_count": 0,
            "edge_count": 0,
            "avg_degree": 0.0,
            "connected_components": 0,
        },
    }
    (graph_dir / "flow_network.json").write_text(json.dumps(flow_network, indent=2))

    # Create centrality_cache.json (empty v9 schema)
    centrality_cache = {
        "schema_version": "v9.0",
        "cache_type": "centrality",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "valid": False,
        "invalidation_trigger": {
            "source": "data/receipts.jsonl",
            "last_known_hash": {"sha256": "", "blake3": ""},
            "last_known_line_count": 0,
        },
        "algorithm": "pagerank",
        "algorithm_params": {
            "damping_factor": 0.85,
            "max_iterations": 100,
            "convergence_threshold": 0.0001,
        },
        "values": {},
        "statistics": {
            "total_nodes": 0,
            "mean_centrality": 0.0,
            "max_centrality": 0.0,
            "min_centrality": 0.0,
            "nodes_above_floor": 0,
            "floor_threshold": 0.2,
        },
    }
    (graph_dir / "centrality_cache.json").write_text(
        json.dumps(centrality_cache, indent=2)
    )

    # Create events.jsonl (single init event - v9 schema, no mode field)
    init_event = {
        "event_id": "evt_test_init",
        "event_type": "stream_initialized",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence_number": 1,
        "tenant_id": "test",
        "payload": {
            "message": "Test event stream initialized",
            "schema_version": "v9.0",
        },
        "hash_chain": {
            "payload_hash": {
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "blake3": "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            },
            "previous_hash": {"sha256": "", "blake3": ""},
        },
        "source": {
            "source_type": "system",
            "source_id": "test_initialization",
            "receipt_id": "",
        },
        "metadata": {"companies": [], "tags": ["test", "init"]},
    }
    (events_dir / "events.jsonl").write_text(json.dumps(init_event) + "\n")

    # Create empty receipts.jsonl
    (tmp_path / "receipts.jsonl").write_text("")

    return tmp_path


@pytest.fixture
def sample_receipts() -> List[Dict[str, Any]]:
    """
    Sample test receipts for v9 testing.

    Returns at least 3 receipts with:
    - type: receipt type string
    - id: unique identifier
    - pattern_id: pattern identifier
    - timestamp: ISO UTC timestamp

    Returns:
        List[Dict]: List of sample receipt dicts
    """
    base_ts = datetime.now(timezone.utc)
    return [
        {
            "type": "qed_receipt",
            "id": "rcpt_001",
            "pattern_id": "PAT_BATTERY_THERMAL_001",
            "timestamp": base_ts.isoformat(),
            "ratio": 8.5,
            "recall": 0.9998,
            "verified": True,
        },
        {
            "type": "qed_receipt",
            "id": "rcpt_002",
            "pattern_id": "PAT_STEERING_TORQUE_002",
            "timestamp": base_ts.isoformat(),
            "ratio": 12.3,
            "recall": 0.9997,
            "verified": True,
        },
        {
            "type": "qed_receipt",
            "id": "rcpt_003",
            "pattern_id": "PAT_BRAKE_PRESSURE_003",
            "timestamp": base_ts.isoformat(),
            "ratio": 6.7,
            "recall": 0.9996,
            "verified": True,
        },
        {
            "type": "binder_receipt",
            "id": "rcpt_004",
            "receipt_id": "binder_001",
            "timestamp": base_ts.isoformat(),
            "input_packet_ids": ["pkt_001", "pkt_002"],
            "portfolio_metrics": {
                "total_windows": 1000,
                "breach_rate": 0.001,
                "centrality_high_count": 2,
                "centrality_mid_count": 1,
                "centrality_low_count": 0,
            },
        },
    ]


@pytest.fixture
def sample_graph_with_nodes() -> nx.DiGraph:
    """
    Create sample graph with nodes and edges for testing.

    Graph structure:
    - Nodes: A, B, C (where C is shared across tesla+spacex)
    - Edges: A->B, B->C (bidirectional flow)
    - Node C has metadata indicating cross-company usage

    Returns:
        nx.DiGraph: Sample directed graph
    """
    graph = nx.DiGraph()

    # Add nodes with metadata
    graph.add_node("A", pattern_id="PAT_TESLA_A", company="tesla")
    graph.add_node("B", pattern_id="PAT_TESLA_B", company="tesla")
    graph.add_node(
        "C",
        pattern_id="PAT_SHARED_C",
        company="shared",
        companies=["tesla", "spacex"],
    )

    # Add bidirectional edges (v9 bidirectional flow)
    graph.add_edge("A", "B", weight=1.0, label="thermal_correlation")
    graph.add_edge("B", "A", weight=1.0, label="thermal_correlation")
    graph.add_edge("B", "C", weight=1.0, label="cross_domain")
    graph.add_edge("C", "B", weight=1.0, label="cross_domain")

    return graph
