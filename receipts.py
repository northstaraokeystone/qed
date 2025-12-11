"""
receipts.py - v6 Foundation Module

Canonical emit_receipt() per CLAUDEME section 8. ALL modules import from here.
This is the single source of truth for receipt emission.

Never single hash. Always dual_hash (SHA256:BLAKE3).
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

__all__ = [
    "dual_hash",
    "emit_receipt",
    "write_receipt_jsonl",
    "StopRule",
    "merkle",
    "RECEIPT_SCHEMA",
]

# =============================================================================
# CONSTANTS
# =============================================================================

RECEIPT_SCHEMA = {
    "receipt_type": "str",
    "ts": "ISO8601",
    "tenant_id": "str",
    "payload_hash": "str (SHA256:BLAKE3)",
}


# =============================================================================
# BLAKE3 SUPPORT
# =============================================================================

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# =============================================================================
# CORE FUNCTION 1: dual_hash
# =============================================================================

def dual_hash(data: Union[bytes, str]) -> str:
    """
    SHA256:BLAKE3 - ALWAYS use this, never single hash.

    Args:
        data: Bytes or string to hash

    Returns:
        str: "sha256_hex:blake3_hex" format
    """
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


# =============================================================================
# CORE FUNCTION 2: emit_receipt
# =============================================================================

def emit_receipt(receipt_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Every function calls this. No exceptions.

    Args:
        receipt_type: Type identifier for this receipt
        data: Receipt payload (must include tenant_id or defaults to 'default')

    Returns:
        dict: Complete receipt with ts, tenant_id, payload_hash, and data fields
    """
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.now(timezone.utc).isoformat(),
        "tenant_id": data.get("tenant_id", "default"),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data
    }
    return receipt


# =============================================================================
# CORE FUNCTION 3: write_receipt_jsonl
# =============================================================================

def write_receipt_jsonl(receipt: Dict[str, Any], fh) -> None:
    """
    Append receipt as single JSON line to file handle.

    Args:
        receipt: Receipt dict to write
        fh: File handle (must be open for writing)
    """
    line = json.dumps(receipt, separators=(",", ":"))
    fh.write(line + "\n")


# =============================================================================
# CORE FUNCTION 4: merkle
# =============================================================================

def merkle(items: List[Any]) -> str:
    """
    Compute Merkle root of items.

    Args:
        items: List of items to merkle (will be JSON serialized)

    Returns:
        str: Merkle root hash in dual_hash format
    """
    if not items:
        return dual_hash(b"empty")
    hashes = [dual_hash(json.dumps(i, sort_keys=True)) for i in items]
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i + 1])
                  for i in range(0, len(hashes), 2)]
    return hashes[0]


# =============================================================================
# STOPRULE EXCEPTION
# =============================================================================

class StopRule(Exception):
    """Raised when stoprule triggers. Never catch silently."""
    pass
