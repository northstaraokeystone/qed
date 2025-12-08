"""
mesh_view_v1.py - Mesh View Aggregator for QED v6 Telemetry

View aggregator that reads qed_run_manifest.json and QEDReceipt lines from
receipts.jsonl, then emits per-company/per-hook metrics tables.

Schema:
  Manifest (qed_run_manifest.json):
    {
      "run_id": str,
      "fleet_size": int,
      "total_windows": int,
      "receipts_file": str,
      "timestamps": {"start": str, "end": str}
    }

  Receipt (receipts.jsonl lines):
    {
      "ts": str,
      "window_id": str,
      "params": {"A": float, "f": float, ...},
      "ratio": float,
      "H_bits": float,
      "recall": float,
      "savings_M": float,
      "verified": bool | null,
      "violations": list,
      "trace": str
    }

Output View:
  {
    "<company>": {
      "<hook>": {
        "windows": int,
        "avg_ratio": float,
        "total_savings": float,
        "slo_breach_rate": float,
        "constraint_violations": int
      }
    }
  }

Usage:
  python mesh_view_v1.py <manifest.json> <receipts.jsonl> [sample_n]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_manifest(path: str) -> Dict[str, Any]:
    """
    Load and validate qed_run_manifest.json.

    Args:
        path: Path to manifest JSON file.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        json.JSONDecodeError: If manifest is not valid JSON.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(manifest_path, "r") as f:
        return json.load(f)


def sample_receipts(receipts_file: str, n: int = 100) -> List[Dict[str, Any]]:
    """
    Sample first N receipt lines from JSONL file.

    Args:
        receipts_file: Path to receipts JSONL file.
        n: Maximum number of receipts to sample (default: 100).

    Returns:
        List of parsed receipt dictionaries.

    Raises:
        FileNotFoundError: If receipts file does not exist.
    """
    receipts_path = Path(receipts_file)
    if not receipts_path.exists():
        raise FileNotFoundError(f"Receipts file not found: {receipts_file}")

    receipts: List[Dict[str, Any]] = []
    with open(receipts_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                receipt = json.loads(line)
                receipts.append(receipt)
                if len(receipts) >= n:
                    break
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

    return receipts


def extract_hook_from_receipt(receipt: Dict[str, Any]) -> str:
    """
    Extract hook name from receipt.

    Checks multiple possible locations:
    - Direct 'hook' field
    - Inside params dict under 'scenario' or 'hook_name'
    - From window_id prefix (e.g., 'tesla_fsd_1000_abc' -> 'tesla_fsd')
    - From trace field

    Args:
        receipt: Receipt dictionary.

    Returns:
        Hook name string.
    """
    # Direct hook field
    if "hook" in receipt and receipt["hook"]:
        return receipt["hook"]

    # From params
    params = receipt.get("params", {})
    if "scenario" in params:
        return params["scenario"]
    if "hook_name" in params:
        return params["hook_name"]

    # From window_id (e.g., "tesla_fsd_1000_abc123")
    window_id = receipt.get("window_id", "")
    if window_id:
        parts = window_id.rsplit("_", 2)
        if len(parts) >= 2:
            return "_".join(parts[:-2]) if len(parts) > 2 else parts[0]

    # From trace
    trace = receipt.get("trace", "")
    if "scenario=" in trace:
        # Parse "scenario=tesla_fsd" from trace
        for part in trace.split():
            if part.startswith("scenario="):
                return part.split("=", 1)[1]

    return "generic"


def parse_company_from_hook(hook: str) -> str:
    """
    Parse company name from hook identifier.

    Examples:
        'tesla_fsd' -> 'tesla'
        'spacex_flight' -> 'spacex'
        'generic' -> 'generic'
        'boring_tunnel' -> 'boring'

    Args:
        hook: Hook name string.

    Returns:
        Company name (first part before underscore).
    """
    if "_" in hook:
        return hook.split("_")[0]
    return hook


def compute_metrics(receipts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute aggregated metrics grouped by company and hook.

    For each company/hook group, computes:
      - windows: count of receipts
      - avg_ratio: mean compression ratio
      - total_savings: sum of savings_M
      - slo_breach_rate: percentage of unverified windows (SLO breach)
      - constraint_violations: sum of violation counts

    Args:
        receipts: List of receipt dictionaries.

    Returns:
        Nested dict: {company: {hook: {metrics}}}
    """
    # Group receipts by (company, hook)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for receipt in receipts:
        hook = extract_hook_from_receipt(receipt)
        company = parse_company_from_hook(hook)
        groups[(company, hook)].append(receipt)

    # Compute metrics per group
    aggregated: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for (company, hook), group_receipts in groups.items():
        # Extract values with defaults for missing fields
        ratios = []
        savings = []
        verified_flags = []
        violation_counts = []

        for r in group_receipts:
            # Ratio
            if "ratio" in r and r["ratio"] is not None:
                ratios.append(float(r["ratio"]))

            # Savings
            if "savings_M" in r and r["savings_M"] is not None:
                savings.append(float(r["savings_M"]))

            # Verified status (None/null counts as breach for safety)
            verified = r.get("verified")
            if verified is None:
                verified_flags.append(False)  # Unverified = breach
            else:
                verified_flags.append(bool(verified))

            # Violations count
            violations = r.get("violations", [])
            if isinstance(violations, list):
                violation_counts.append(len(violations))
            elif isinstance(violations, int):
                violation_counts.append(violations)
            else:
                violation_counts.append(0)

        # Compute aggregates
        n_windows = len(group_receipts)
        avg_ratio = statistics.mean(ratios) if ratios else 0.0
        total_savings = sum(savings)

        # SLO breach rate: percentage of unverified windows
        if verified_flags:
            breach_count = sum(1 for v in verified_flags if not v)
            slo_breach_rate = (breach_count / len(verified_flags)) * 100.0
        else:
            slo_breach_rate = 0.0

        total_violations = sum(violation_counts)

        # Store in nested dict
        if company not in aggregated:
            aggregated[company] = {}

        aggregated[company][hook] = {
            "windows": n_windows,
            "avg_ratio": round(avg_ratio, 2),
            "total_savings": round(total_savings, 2),
            "slo_breach_rate": round(slo_breach_rate, 2),
            "constraint_violations": total_violations,
        }

    return aggregated


def emit_view(
    aggregated: Dict[str, Dict[str, Dict[str, Any]]],
    manifest: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> str:
    """
    Emit aggregated metrics as formatted JSON string.

    Args:
        aggregated: Nested metrics dict from compute_metrics().
        manifest: Optional manifest to include run metadata.
        indent: JSON indentation level (default: 2).

    Returns:
        JSON string representation.
    """
    output: Dict[str, Any] = {}

    # Include manifest metadata if provided
    if manifest:
        output["_meta"] = {
            "run_id": manifest.get("run_id", "unknown"),
            "fleet_size": manifest.get("fleet_size", 0),
            "total_windows": manifest.get("total_windows", 0),
        }

    # Add company/hook metrics
    output["companies"] = aggregated

    return json.dumps(output, indent=indent)


def print_table(aggregated: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
    Print metrics as a formatted table to stdout.

    Args:
        aggregated: Nested metrics dict from compute_metrics().
    """
    # Header
    print(f"{'Company':<12} {'Hook':<20} {'Windows':>8} {'AvgRatio':>10} "
          f"{'Savings($M)':>12} {'SLO Breach%':>12} {'Violations':>11}")
    print("-" * 90)

    # Rows
    for company in sorted(aggregated.keys()):
        for hook in sorted(aggregated[company].keys()):
            m = aggregated[company][hook]
            print(f"{company:<12} {hook:<20} {m['windows']:>8} {m['avg_ratio']:>10.2f} "
                  f"{m['total_savings']:>12.2f} {m['slo_breach_rate']:>12.2f} "
                  f"{m['constraint_violations']:>11}")


def generate_mesh_view(
    manifest_path: str,
    receipts_path: str,
    sample_n: int = 100,
) -> Dict[str, Any]:
    """
    Generate mesh view from manifest and receipts.

    Args:
        manifest_path: Path to qed_run_manifest.json.
        receipts_path: Path to receipts.jsonl.
        sample_n: Number of receipts to sample (default: 100).

    Returns:
        Dict with keys: _meta (run metadata), companies (per-company/hook metrics).
    """
    manifest = load_manifest(manifest_path)
    receipts = sample_receipts(receipts_path, sample_n)
    metrics = compute_metrics(receipts)

    output: Dict[str, Any] = {
        "_meta": {
            "run_id": manifest.get("run_id", "unknown"),
            "fleet_size": manifest.get("fleet_size", 0),
            "total_windows": manifest.get("total_windows", manifest.get("windows", 0)),
        },
        "companies": metrics,
    }

    return output


def main(
    manifest_path: str,
    receipts_file: str,
    sample_n: int = 100,
    output_format: str = "json",
) -> int:
    """
    Main entry point for mesh view aggregation.

    Args:
        manifest_path: Path to qed_run_manifest.json.
        receipts_file: Path to receipts.jsonl.
        sample_n: Number of receipts to sample (default: 100).
        output_format: Output format - 'json' or 'table' (default: 'json').

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing manifest: {e}", file=sys.stderr)
        return 1

    try:
        receipts = sample_receipts(receipts_file, sample_n)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not receipts:
        print("Warning: No receipts found", file=sys.stderr)

    metrics = compute_metrics(receipts)

    if output_format == "table":
        print_table(metrics)
    else:
        print(emit_view(metrics, manifest))

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mesh View v1 - QED v6 Telemetry Aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mesh_view_v1.py manifest.json receipts.jsonl
  python mesh_view_v1.py manifest.json receipts.jsonl --sample 500
  python mesh_view_v1.py manifest.json receipts.jsonl --format table
        """,
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to qed_run_manifest.json",
    )
    parser.add_argument(
        "receipts_file",
        type=str,
        help="Path to receipts.jsonl",
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=100,
        help="Number of receipts to sample (default: 100)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "table"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    sys.exit(main(
        manifest_path=args.manifest_path,
        receipts_file=args.receipts_file,
        sample_n=args.sample,
        output_format=args.format,
    ))
