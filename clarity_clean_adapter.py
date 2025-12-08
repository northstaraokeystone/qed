"""
ClarityClean Adapter for QED v7
Converts QEDReceipts to text corpus with quality audit.
Lightweight bridge - not the full ClarityClean engine.
"""

import json
import re
import zlib
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional


@dataclass
class ClarityCleanReceipt:
    """Quality audit receipt for a corpus cleaning run."""
    timestamp: str
    source_file: str
    token_count: int
    anomaly_density: float      # ratio of anomaly-flagged receipts (0.0-1.0)
    noise_ratio: float          # compression-based noise estimate (0.0-1.0)
    corpus_hash: str            # SHA256 of output corpus
    receipt_count: int          # total receipts processed


def _iso_now() -> str:
    """UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _estimate_noise_ratio(text: str) -> float:
    """
    Estimate noise via compression ratio.
    High compression = repetitive/low-info = high noise.
    Returns 0.0 (clean) to 1.0 (noisy).
    """
    if not text or len(text) < 10:
        return 0.0
    original = len(text.encode('utf-8'))
    compressed = len(zlib.compress(text.encode('utf-8')))
    ratio = compressed / original
    # Invert: low compression ratio = noisy (repetitive)
    # Typical text compresses to 0.3-0.5, noise compresses to 0.1-0.2
    noise = max(0.0, min(1.0, 1.0 - ratio))
    return round(noise, 4)


def _count_tokens(text: str) -> int:
    """Simple whitespace tokenizer count."""
    return len(text.split())


def _normalize_text(text: str) -> str:
    """Basic text normalization - collapse whitespace, strip."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def receipts_to_text(receipts_path: str) -> str:
    """
    Convert QEDReceipts JSONL to a single text corpus.
    Extracts key fields and concatenates into readable text.

    Args:
        receipts_path: Path to QEDReceipts JSONL file

    Returns:
        Text corpus string
    """
    corpus_parts: List[str] = []
    path = Path(receipts_path)

    if not path.exists():
        return ""

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                receipt = json.loads(line)
                # Extract relevant text fields from receipt
                parts = []

                # Core fields
                if 'hook' in receipt:
                    parts.append(f"hook:{receipt['hook']}")
                if 'classification' in receipt:
                    parts.append(f"class:{receipt['classification']}")
                if 'slo_status' in receipt:
                    parts.append(f"slo:{receipt['slo_status']}")
                if 'score' in receipt:
                    parts.append(f"score:{receipt['score']}")
                if 'compression_ratio' in receipt:
                    parts.append(f"ratio:{receipt['compression_ratio']}")

                # Violations/anomalies (important for analysis)
                if 'violations' in receipt and receipt['violations']:
                    parts.append(f"violations:{','.join(str(v) for v in receipt['violations'])}")

                # Pattern ID if present (v7)
                if 'pattern_id' in receipt:
                    parts.append(f"pattern:{receipt['pattern_id']}")

                if parts:
                    corpus_parts.append(' | '.join(parts))

            except json.JSONDecodeError:
                continue

    return '\n'.join(corpus_parts)


def clean_corpus(corpus: str) -> Tuple[str, ClarityCleanReceipt]:
    """
    Clean corpus and generate quality audit receipt.

    Args:
        corpus: Raw text corpus from receipts_to_text()

    Returns:
        Tuple of (cleaned_corpus, quality_audit_receipt)
    """
    # Normalize
    cleaned = _normalize_text(corpus)

    # Count anomalies (lines with violations or anomaly flags)
    lines = corpus.strip().split('\n') if corpus.strip() else []
    total_lines = len(lines)
    anomaly_lines = sum(1 for line in lines if 'violation' in line.lower() or 'anomaly' in line.lower())
    anomaly_density = (anomaly_lines / total_lines) if total_lines > 0 else 0.0

    # Build receipt
    receipt = ClarityCleanReceipt(
        timestamp=_iso_now(),
        source_file="",  # Set by caller
        token_count=_count_tokens(cleaned),
        anomaly_density=round(anomaly_density, 4),
        noise_ratio=_estimate_noise_ratio(cleaned),
        corpus_hash=hashlib.sha256(cleaned.encode('utf-8')).hexdigest()[:16],
        receipt_count=total_lines
    )

    return cleaned, receipt


def process_receipts(
    receipts_path: str,
    output_corpus_path: Optional[str] = None,
    output_receipt_path: Optional[str] = None
) -> Tuple[str, ClarityCleanReceipt]:
    """
    Full pipeline: receipts -> text -> clean -> audit receipt.

    Args:
        receipts_path: Path to QEDReceipts JSONL
        output_corpus_path: Optional path to write cleaned corpus
        output_receipt_path: Optional path to write audit receipt JSONL

    Returns:
        Tuple of (cleaned_corpus, quality_audit_receipt)
    """
    # Convert receipts to text
    corpus = receipts_to_text(receipts_path)

    # Clean and audit
    cleaned, receipt = clean_corpus(corpus)
    receipt.source_file = receipts_path

    # Write outputs if paths provided
    if output_corpus_path:
        Path(output_corpus_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_corpus_path).write_text(cleaned, encoding='utf-8')

    if output_receipt_path:
        Path(output_receipt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_receipt_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(receipt)) + '\n')

    return cleaned, receipt


# Convenience exports
__all__ = [
    'ClarityCleanReceipt',
    'receipts_to_text',
    'clean_corpus',
    'process_receipts'
]
