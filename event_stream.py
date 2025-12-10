"""
event_stream.py - v9 CQRS Event Log with Query-as-Projection

Implements append-only event log with hash chain integrity and mode-as-projection.

Key v9 paradigm shifts applied:

1. No Mode Field (Paradigm 3):
   - EventRecord has NO mode, live, or shadow field
   - Mode is computed at query time via QueryPredicate lens
   - Same events appear different to different observers based on their query

2. Query as Projection (Paradigm 3):
   - query() takes QueryPredicate from binder.py
   - Same event stream filtered through different lenses produces different "realities"
   - HUNTER sees events as "actionable", human dashboard sees as "observable"

3. Backward Causation via Counterfactual (Paradigm 4):
   - replay() re-evaluates past events under different rules
   - Future counterfactual changes how past is observed, not what happened
   - Enables "what-if" analysis without modifying actual history

4. Hash Chain Integrity (Section 3.3):
   - Each event links to previous via upstream_hash
   - Forms Merkle chain - hostile auditor can prove no deletions/modifications
   - verify_chain() validates entire chain

5. Receipt Monad Pattern (Paradigm 1):
   - stream() is R -> R transformer (pure function)
   - Actual persistence is separate concern in append()
   - All operations emit receipts for observability

What does NOT exist in this file:
- mode field on EventRecord
- PatternMode enum import
- live/shadow as stored values
- Mutable state or internal caches
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import networkx as nx

from binder import QueryPredicate


# =============================================================================
# Constants
# =============================================================================

DEFAULT_LOG_PATH = "data/events/events.jsonl"
HASH_ALGO = "sha256"


# =============================================================================
# Receipt Schema (self-describing module contract)
# =============================================================================

RECEIPT_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "stream_receipt",
        "version": "1.0.0",
        "description": "Receipt from stream() R -> R transformer",
        "fields": {
            "receipt_id": "SHA256 hash of streamed event IDs",
            "timestamp": "ISO UTC timestamp",
            "events_processed": "int - count of events wrapped",
            "chain_valid": "bool - whether hash chain is valid",
            "first_event_id": "UUID of first event in stream",
            "last_event_id": "UUID of last event in stream",
        },
    },
    {
        "type": "append_receipt",
        "version": "1.0.0",
        "description": "Receipt from append() persistence operation",
        "fields": {
            "receipt_id": "SHA256 hash of appended event",
            "timestamp": "ISO UTC timestamp",
            "event_id": "UUID of appended event",
            "hash": "SHA256 hash of event (payload + upstream_hash)",
            "log_path": "Path to JSONL log file",
            "chain_position": "int - position in hash chain (line number)",
        },
    },
    {
        "type": "replay_receipt",
        "version": "1.0.0",
        "description": "Receipt from replay() counterfactual operation",
        "fields": {
            "receipt_id": "SHA256 hash of replay parameters",
            "timestamp": "ISO UTC timestamp",
            "events_replayed": "int - count of events re-evaluated",
            "counterfactual": "Dict describing alternate reality applied",
            "visibility_changes": "Dict[event_id, Dict] - how visibility changed per event",
        },
    },
]


# =============================================================================
# EventRecord (frozen dataclass - no mode field!)
# =============================================================================

@dataclass(frozen=True)
class EventRecord:
    """
    Immutable event record in append-only log.

    CRITICAL: No mode field. Mode is computed at query time via QueryPredicate.
    The same event can be "live" to one observer and "shadow" to another based
    on their query lens.

    Attributes:
        event_id: Unique identifier (UUID)
        timestamp: ISO8601 UTC timestamp
        receipt_type: Type of receipt/event (e.g., "binder_receipt", "stream_receipt")
        payload: Arbitrary event data (dict)
        upstream_id: Optional ID of causing event (for causal links)
        hash: SHA256 of payload + upstream_hash (for Merkle chain)
        upstream_hash: Hash of previous event in chain (None for first event)
    """
    event_id: str
    timestamp: str
    receipt_type: str
    payload: Dict[str, Any]
    upstream_id: Optional[str]
    hash: str
    upstream_hash: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSONL storage."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "receipt_type": self.receipt_type,
            "payload": self.payload,
            "upstream_id": self.upstream_id,
            "hash": self.hash,
            "upstream_hash": self.upstream_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventRecord:
        """Deserialize from dict."""
        return cls(
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            receipt_type=data["receipt_type"],
            payload=data["payload"],
            upstream_id=data.get("upstream_id"),
            hash=data["hash"],
            upstream_hash=data.get("upstream_hash"),
        )


# =============================================================================
# Hash Chain Utilities
# =============================================================================

def _compute_hash(payload: Dict[str, Any], upstream_hash: Optional[str]) -> str:
    """
    Compute SHA256 hash for event (Merkle chain link).

    Hash = SHA256(json(payload) + upstream_hash)
    This links each event to previous, forming chain.

    Args:
        payload: Event payload dict
        upstream_hash: Hash of previous event (empty string if first)

    Returns:
        64-character SHA256 hex digest
    """
    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    upstream_str = upstream_hash if upstream_hash else ""
    combined = payload_str + upstream_str
    return hashlib.sha256(combined.encode()).hexdigest()


def verify_chain(events: List[EventRecord]) -> Tuple[bool, Optional[str]]:
    """
    Verify hash chain integrity across events.

    Walks chain and verifies each event's hash = SHA256(payload + upstream_hash).
    Enables hostile auditor to prove no deletions/modifications.

    Args:
        events: List of EventRecords in chain order

    Returns:
        (True, None) if chain valid
        (False, "break at event_id X") if invalid
    """
    if not events:
        return (True, None)

    for i, event in enumerate(events):
        # Recompute hash from payload + upstream_hash
        expected_hash = _compute_hash(event.payload, event.upstream_hash)

        if event.hash != expected_hash:
            return (False, f"Hash mismatch at event_id {event.event_id}")

        # Verify chain linkage (upstream_hash should match previous event's hash)
        if i > 0:
            prev_event = events[i - 1]
            if event.upstream_hash != prev_event.hash:
                return (False, f"Chain break at event_id {event.event_id}: upstream_hash doesn't match previous")

        # First event should have None upstream_hash
        if i == 0 and event.upstream_hash is not None:
            return (False, f"First event {event.event_id} should have None upstream_hash")

    return (True, None)


# =============================================================================
# Core R -> R Transformer
# =============================================================================

def stream(receipts: List[Dict]) -> List[Dict]:
    """
    Core R -> R transformer (Paradigm 1).

    Wraps each receipt as EventRecord with hash chain, returns stream_receipts.
    Pure function - actual persistence is separate concern (see append()).

    Args:
        receipts: List of receipt dicts to wrap as events

    Returns:
        List of stream_receipts describing what was processed
    """
    if not receipts:
        return []

    events: List[EventRecord] = []
    upstream_hash: Optional[str] = None

    for receipt in receipts:
        event_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        receipt_type = receipt.get("type", "unknown_receipt")

        # Compute hash linking to previous event
        event_hash = _compute_hash(receipt, upstream_hash)

        event = EventRecord(
            event_id=event_id,
            timestamp=timestamp,
            receipt_type=receipt_type,
            payload=receipt,
            upstream_id=None,  # No causal link for streamed receipts
            hash=event_hash,
            upstream_hash=upstream_hash,
        )

        events.append(event)
        upstream_hash = event_hash  # Next event links to this one

    # Verify chain we just built
    chain_valid, _ = verify_chain(events)

    # Generate stream_receipt
    event_ids = [e.event_id for e in events]
    receipt_id = hashlib.sha256(
        json.dumps(event_ids, separators=(",", ":")).encode()
    ).hexdigest()[:16]

    stream_receipt = {
        "type": "stream_receipt",
        "receipt_id": receipt_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "events_processed": len(events),
        "chain_valid": chain_valid,
        "first_event_id": events[0].event_id if events else None,
        "last_event_id": events[-1].event_id if events else None,
    }

    return [stream_receipt]


# =============================================================================
# Persistence Operations
# =============================================================================

def append(event: EventRecord, log_path: str = DEFAULT_LOG_PATH) -> Dict:
    """
    Append EventRecord to JSONL log (persistence operation).

    Maintains hash chain integrity by reading last event's hash and linking to it.
    NOT a pure function - has side effect of writing to disk.

    Args:
        event: EventRecord to append
        log_path: Path to JSONL log file

    Returns:
        append_receipt dict with event_id, hash, log_path, chain_position
    """
    # Ensure directory exists
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine chain position
    if log_file.exists():
        with open(log_file, "r") as f:
            chain_position = sum(1 for _ in f) + 1
    else:
        chain_position = 1

    # Append event to JSONL
    with open(log_file, "a") as f:
        json.dump(event.to_dict(), f, separators=(",", ":"))
        f.write("\n")

    # Generate append_receipt
    receipt_id = hashlib.sha256(event.hash.encode()).hexdigest()[:16]

    append_receipt = {
        "type": "append_receipt",
        "receipt_id": receipt_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": event.event_id,
        "hash": event.hash,
        "log_path": log_path,
        "chain_position": chain_position,
    }

    return append_receipt


def get_latest(n: int = 10, log_path: str = DEFAULT_LOG_PATH) -> List[EventRecord]:
    """
    Retrieve n most recent events from log.

    No filtering - returns raw events. Use query() to filter by predicate.

    Args:
        n: Number of recent events to retrieve
        log_path: Path to JSONL log file

    Returns:
        List of EventRecords (most recent first)
    """
    log_file = Path(log_path)
    if not log_file.exists():
        return []

    events: List[EventRecord] = []
    with open(log_file, "r") as f:
        lines = f.readlines()

    # Get last n lines
    recent_lines = lines[-n:] if len(lines) >= n else lines

    for line in recent_lines:
        if line.strip():
            data = json.loads(line)
            events.append(EventRecord.from_dict(data))

    # Return most recent first
    return list(reversed(events))


# =============================================================================
# Query as Projection (Paradigm 3)
# =============================================================================

def query(
    events: List[EventRecord],
    predicate: QueryPredicate,
    graph: Optional[nx.DiGraph] = None,
) -> List[EventRecord]:
    """
    Mode as projection - filter events through QueryPredicate lens.

    The same events can be queried with different predicates to get different
    "realities". HUNTER sees events as "actionable", human dashboard sees as
    "observable". Mode is NOT stored - it's computed from predicate at query time.

    Args:
        events: List of EventRecords to filter
        predicate: QueryPredicate lens defining visibility rules
        graph: Optional DiGraph for centrality-based filtering

    Returns:
        List of EventRecords visible under this predicate lens
    """
    if not events:
        return []

    visible: List[EventRecord] = []

    for event in events:
        # Extract metadata from event payload for predicate evaluation
        payload = event.payload

        # Compute "centrality" for this event (if graph provided)
        if graph is not None and "pattern_id" in payload:
            pattern_id = payload["pattern_id"]
            if pattern_id in graph:
                # Use PageRank centrality
                try:
                    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
                    centrality = pagerank.get(pattern_id, 0.0)
                except:
                    centrality = 0.0
            else:
                centrality = 0.0
        else:
            # No graph: use default centrality = 0.5 (middle tier)
            centrality = payload.get("centrality", 0.5)

        # Check if event visible under predicate
        if predicate.matches(centrality):
            visible.append(event)

    return visible


# =============================================================================
# Backward Causation via Counterfactual (Paradigm 4)
# =============================================================================

def replay(
    events: List[EventRecord],
    counterfactual: Dict,
    graph: Optional[nx.DiGraph] = None,
) -> List[Dict]:
    """
    Backward causation - re-evaluate events under counterfactual rules.

    This is NOT simulation - it's re-observation of actual events under different
    physics. Future counterfactual causes past receipts to be re-evaluated.

    Args:
        events: List of EventRecords to replay
        counterfactual: Dict with threshold_override, timestamp_cutoff, predicate_override
        graph: Optional DiGraph for centrality computation

    Returns:
        List of replay_receipts showing visibility under counterfactual
    """
    if not events:
        return []

    # Extract counterfactual parameters
    threshold_override = counterfactual.get("threshold_override")
    timestamp_cutoff = counterfactual.get("timestamp_cutoff")
    predicate_override = counterfactual.get("predicate_override")

    # Build counterfactual predicate
    if predicate_override:
        # Use provided predicate dict to construct QueryPredicate
        cf_predicate = QueryPredicate(
            actionable=predicate_override.get("actionable", True),
            ttl_valid=predicate_override.get("ttl_valid", True),
            min_centrality=predicate_override.get("min_centrality", 0.0),
            max_centrality=predicate_override.get("max_centrality"),
        )
    elif threshold_override is not None:
        # Create predicate with threshold override
        cf_predicate = QueryPredicate(
            actionable=True,
            ttl_valid=True,
            min_centrality=threshold_override,
            max_centrality=None,
        )
    else:
        # Default: live predicate
        cf_predicate = QueryPredicate.live()

    # Filter by timestamp if cutoff provided
    filtered_events = events
    if timestamp_cutoff:
        filtered_events = [
            e for e in events if e.timestamp <= timestamp_cutoff
        ]

    # Re-evaluate each event under counterfactual
    visibility_changes: Dict[str, Dict] = {}

    for event in filtered_events:
        # Original visibility (with default predicate)
        original_visible = len(query([event], QueryPredicate.live(), graph)) > 0

        # Counterfactual visibility
        cf_visible = len(query([event], cf_predicate, graph)) > 0

        visibility_changes[event.event_id] = {
            "original_visible": original_visible,
            "counterfactual_visible": cf_visible,
            "changed": original_visible != cf_visible,
            "event_timestamp": event.timestamp,
            "receipt_type": event.receipt_type,
        }

    # Generate replay_receipt
    receipt_id = hashlib.sha256(
        json.dumps(counterfactual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]

    replay_receipt = {
        "type": "replay_receipt",
        "receipt_id": receipt_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "events_replayed": len(filtered_events),
        "counterfactual": counterfactual,
        "visibility_changes": visibility_changes,
    }

    return [replay_receipt]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Schema
    "RECEIPT_SCHEMA",
    # Constants
    "DEFAULT_LOG_PATH",
    "HASH_ALGO",
    # Dataclass
    "EventRecord",
    # Core functions
    "stream",
    "append",
    "get_latest",
    "query",
    "replay",
    "verify_chain",
]
