"""
mesh_view_v2.py - Mesh View Aggregator for QED v7 Telemetry

Extended mesh view that aggregates data from multiple sources into per-company/per-hook
tables. Extends v1 with exploit tracking, cross-domain links, and ClarityClean quality scores.

Data Sources:
  - QEDReceipts (receipts.jsonl): Core telemetry receipts from v1
  - ClarityClean receipts (clarity_receipts.jsonl): Quality/noise metrics
  - Simulation results (sim_results.json): Simulation run outcomes
  - Anomaly library (shared_anomalies.jsonl): Pattern anomalies with exploit grades
  - Cross-domain validations (cross_domain_validations.jsonl): Cross-company validation links

v7 Output View (extends v6):
  {
    "<company>": {
      "<hook>": {
        "windows": int,
        "avg_ratio": float,
        "total_savings": float,
        "slo_breach_rate": float,
        "constraint_violations": int,
        "exploit_count": int,           # v7 new
        "cross_domain_links": int,      # v7 new
        "clarity_quality_score": float  # v7 new
      }
    }
  }

Usage:
  python mesh_view_v2.py <manifest.json> <receipts_path> [--clarity path] [--sim path]
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging for warnings on missing optional sources
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# v1 BACKWARD-COMPATIBLE EXPORTS (preserved signatures)
# =============================================================================


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

    Mapping:
        tesla_* -> tesla
        spacex_* -> spacex
        starlink_* -> starlink
        boring_* -> boring
        xai_* -> xai

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
    Compute aggregated metrics grouped by company and hook (v1 compatible).

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
    Emit aggregated metrics as formatted JSON string (v1 compatible).

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
    # Check if v2 fields are present
    has_v2_fields = False
    for company in aggregated.values():
        for hook_data in company.values():
            if "exploit_count" in hook_data:
                has_v2_fields = True
                break
        if has_v2_fields:
            break

    if has_v2_fields:
        # v2 extended table format
        print(
            f"{'Company':<12} {'Hook':<20} {'Windows':>8} {'AvgRatio':>10} "
            f"{'Savings($M)':>12} {'SLO%':>8} {'Viols':>6} "
            f"{'Exploit':>8} {'XDomain':>8} {'Quality':>8}"
        )
        print("-" * 115)

        for company in sorted(aggregated.keys()):
            for hook in sorted(aggregated[company].keys()):
                m = aggregated[company][hook]
                print(
                    f"{company:<12} {hook:<20} {m['windows']:>8} {m['avg_ratio']:>10.2f} "
                    f"{m['total_savings']:>12.2f} {m['slo_breach_rate']:>8.2f} "
                    f"{m['constraint_violations']:>6} "
                    f"{m.get('exploit_count', 0):>8} "
                    f"{m.get('cross_domain_links', 0):>8} "
                    f"{m.get('clarity_quality_score', 0.0):>8.2f}"
                )
    else:
        # v1 table format (backward compatible)
        print(
            f"{'Company':<12} {'Hook':<20} {'Windows':>8} {'AvgRatio':>10} "
            f"{'Savings($M)':>12} {'SLO Breach%':>12} {'Violations':>11}"
        )
        print("-" * 90)

        for company in sorted(aggregated.keys()):
            for hook in sorted(aggregated[company].keys()):
                m = aggregated[company][hook]
                print(
                    f"{company:<12} {hook:<20} {m['windows']:>8} {m['avg_ratio']:>10.2f} "
                    f"{m['total_savings']:>12.2f} {m['slo_breach_rate']:>12.2f} "
                    f"{m['constraint_violations']:>11}"
                )


# =============================================================================
# v2 DATA SOURCE LOADERS
# =============================================================================


def load_qed_receipts(path: str) -> List[Dict[str, Any]]:
    """
    Load all QED receipts from path (file or directory).

    Args:
        path: Path to receipts JSONL file or directory containing receipts.

    Returns:
        List of parsed receipt dictionaries.
    """
    receipts: List[Dict[str, Any]] = []
    path_obj = Path(path)

    if not path_obj.exists():
        logger.warning(f"QED receipts path not found: {path}")
        return receipts

    # Handle directory of receipt files
    if path_obj.is_dir():
        for file_path in path_obj.glob("*.jsonl"):
            receipts.extend(_load_jsonl_file(str(file_path)))
        # Also check for .json files
        for file_path in path_obj.glob("*.json"):
            receipts.extend(_load_jsonl_file(str(file_path)))
    else:
        receipts = _load_jsonl_file(str(path_obj))

    return receipts


def load_clarity_receipts(path: str = "data/clarity_receipts.jsonl") -> List[Dict[str, Any]]:
    """
    Load ClarityClean receipts with noise/quality metrics.

    Expected schema per receipt:
    {
        "hook": str,
        "window_id": str,
        "noise_ratio": float (0.0-1.0),
        "clarity_score": float,
        "ts": str
    }

    Args:
        path: Path to clarity receipts JSONL file.

    Returns:
        List of clarity receipt dictionaries.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"ClarityClean receipts not found: {path} - clarity_quality_score will be 0")
        return []

    return _load_jsonl_file(str(path_obj))


def load_sim_results(path: str = "data/sim_results.json") -> Dict[str, Any]:
    """
    Load simulation results JSON.

    Expected schema:
    {
        "run_id": str,
        "scenarios": {
            "<hook>": {
                "windows_simulated": int,
                "pass_rate": float,
                ...
            }
        }
    }

    Args:
        path: Path to simulation results JSON file.

    Returns:
        Simulation results dictionary (empty dict if not found).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Simulation results not found: {path}")
        return {}

    try:
        with open(path_obj, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse sim_results: {e}")
        return {}


def load_anomaly_library(path: str = "data/shared_anomalies.jsonl") -> List[Dict[str, Any]]:
    """
    Load anomaly library with pattern definitions and exploit grades.

    Expected schema per anomaly:
    {
        "pattern_id": str,
        "hook": str,
        "exploit_grade": bool,
        "severity": str,
        "description": str
    }

    Args:
        path: Path to anomaly library JSONL file.

    Returns:
        List of anomaly dictionaries.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Anomaly library not found: {path} - exploit_count will be 0")
        return []

    return _load_jsonl_file(str(path_obj))


def load_cross_domain_validations(
    path: str = "data/cross_domain_validations.jsonl",
) -> List[Dict[str, Any]]:
    """
    Load cross-domain validation entries.

    Expected schema per validation:
    {
        "source_hook": str,
        "target_hook": str,
        "validation_type": str,
        "passed": bool,
        "ts": str
    }

    Args:
        path: Path to cross-domain validations JSONL file.

    Returns:
        List of validation dictionaries.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Cross-domain validations not found: {path} - cross_domain_links will be 0")
        return []

    return _load_jsonl_file(str(path_obj))


def _load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    """
    Internal helper to load a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed dictionaries.
    """
    items: List[Dict[str, Any]] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except IOError as e:
        logger.warning(f"Failed to read file {path}: {e}")

    return items


# =============================================================================
# v2 JOIN FUNCTION
# =============================================================================


def join_all_sources(
    receipts_path: str,
    manifest_path: str,
    clarity_path: str = "data/clarity_receipts.jsonl",
    sim_results_path: str = "data/sim_results.json",
    anomalies_path: str = "data/shared_anomalies.jsonl",
    cross_domain_path: str = "data/cross_domain_validations.jsonl",
) -> Dict[str, Any]:
    """
    Join all data sources by hook into unified structure.

    Args:
        receipts_path: Path to QED receipts file or directory.
        manifest_path: Path to manifest JSON file.
        clarity_path: Path to ClarityClean receipts.
        sim_results_path: Path to simulation results.
        anomalies_path: Path to anomaly library.
        cross_domain_path: Path to cross-domain validations.

    Returns:
        Unified data structure:
        {
            "manifest": dict,
            "receipts_by_hook": {hook: [receipts]},
            "clarity_by_hook": {hook: [clarity_receipts]},
            "sim_results": dict,
            "anomalies_by_hook": {hook: [anomalies]},
            "exploit_patterns_by_hook": {hook: set(pattern_ids)},
            "cross_domain_by_target": {target_hook: [validations]}
        }
    """
    # Load manifest (required)
    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError:
        manifest = {}

    # Load QED receipts (required)
    qed_receipts = load_qed_receipts(receipts_path)

    # Load optional v7 sources
    clarity_receipts = load_clarity_receipts(clarity_path)
    sim_results = load_sim_results(sim_results_path)
    anomalies = load_anomaly_library(anomalies_path)
    cross_domain = load_cross_domain_validations(cross_domain_path)

    # Group QED receipts by hook
    receipts_by_hook: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for receipt in qed_receipts:
        hook = extract_hook_from_receipt(receipt)
        receipts_by_hook[hook].append(receipt)

    # Group clarity receipts by hook
    clarity_by_hook: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cr in clarity_receipts:
        hook = cr.get("hook", "generic")
        clarity_by_hook[hook].append(cr)

    # Group anomalies by hook and track exploit patterns
    anomalies_by_hook: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    exploit_patterns_by_hook: Dict[str, set] = defaultdict(set)

    for anomaly in anomalies:
        hook = anomaly.get("hook", "generic")
        anomalies_by_hook[hook].append(anomaly)

        # Track exploit-grade patterns
        if anomaly.get("exploit_grade", False):
            pattern_id = anomaly.get("pattern_id", "")
            if pattern_id:
                exploit_patterns_by_hook[hook].add(pattern_id)

    # Group cross-domain validations by target hook
    cross_domain_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for validation in cross_domain:
        target_hook = validation.get("target_hook", "")
        if target_hook:
            cross_domain_by_target[target_hook].append(validation)

    return {
        "manifest": manifest,
        "receipts_by_hook": dict(receipts_by_hook),
        "clarity_by_hook": dict(clarity_by_hook),
        "sim_results": sim_results,
        "anomalies_by_hook": dict(anomalies_by_hook),
        "exploit_patterns_by_hook": {k: v for k, v in exploit_patterns_by_hook.items()},
        "cross_domain_by_target": dict(cross_domain_by_target),
    }


# =============================================================================
# v2 METRIC COMPUTATIONS
# =============================================================================


def _compute_exploit_count(
    receipts: List[Dict[str, Any]],
    exploit_patterns: set,
) -> int:
    """
    Count patterns with exploit_grade=true that appear in receipts.

    Args:
        receipts: List of receipts for a hook.
        exploit_patterns: Set of pattern_ids with exploit_grade=true.

    Returns:
        Count of exploit patterns found in receipts.
    """
    if not exploit_patterns:
        return 0

    found_patterns: set = set()

    for receipt in receipts:
        # Check pattern_id in receipt
        pattern_id = receipt.get("pattern_id", "")
        if pattern_id and pattern_id in exploit_patterns:
            found_patterns.add(pattern_id)

        # Also check anomalies/violations list for pattern references
        violations = receipt.get("violations", [])
        if isinstance(violations, list):
            for v in violations:
                if isinstance(v, dict):
                    pid = v.get("pattern_id", "")
                    if pid and pid in exploit_patterns:
                        found_patterns.add(pid)
                elif isinstance(v, str) and v in exploit_patterns:
                    found_patterns.add(v)

    return len(found_patterns)


def _compute_cross_domain_links(
    validations: List[Dict[str, Any]],
) -> int:
    """
    Count validated cross-domain entries where passed=true.

    Args:
        validations: List of cross-domain validations for target hook.

    Returns:
        Count of passed validations.
    """
    return sum(1 for v in validations if v.get("passed", False))


def _compute_clarity_quality_score(
    clarity_receipts: List[Dict[str, Any]],
) -> float:
    """
    Compute clarity quality score as (1 - avg_noise_ratio).

    Args:
        clarity_receipts: List of clarity receipts for a hook.

    Returns:
        Quality score in range 0.0-1.0 (0.0 if no data).
    """
    if not clarity_receipts:
        return 0.0

    noise_ratios = []
    for cr in clarity_receipts:
        noise_ratio = cr.get("noise_ratio")
        if noise_ratio is not None:
            # Clamp to valid range
            noise_ratios.append(max(0.0, min(1.0, float(noise_ratio))))

    if not noise_ratios:
        return 0.0

    avg_noise = statistics.mean(noise_ratios)
    return round(1.0 - avg_noise, 4)


def compute_metrics_v2(joined_data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute aggregated metrics with v7 extensions.

    Returns per-company/per-hook dict with all fields:
      - hook (str)
      - windows (int)
      - avg_ratio (float)
      - total_savings (float)
      - slo_breach_rate (float)
      - constraint_violations (int) - from v6
      - exploit_count (int) - v7 new
      - cross_domain_links (int) - v7 new
      - clarity_quality_score (float) - v7 new

    Args:
        joined_data: Unified data structure from join_all_sources().

    Returns:
        Nested dict: {company: {hook: {metrics}}}
    """
    receipts_by_hook = joined_data.get("receipts_by_hook", {})
    clarity_by_hook = joined_data.get("clarity_by_hook", {})
    exploit_patterns_by_hook = joined_data.get("exploit_patterns_by_hook", {})
    cross_domain_by_target = joined_data.get("cross_domain_by_target", {})

    aggregated: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for hook, receipts in receipts_by_hook.items():
        company = parse_company_from_hook(hook)

        # Compute v1/v6 metrics
        ratios = []
        savings = []
        verified_flags = []
        violation_counts = []

        for r in receipts:
            if "ratio" in r and r["ratio"] is not None:
                ratios.append(float(r["ratio"]))

            if "savings_M" in r and r["savings_M"] is not None:
                savings.append(float(r["savings_M"]))

            verified = r.get("verified")
            if verified is None:
                verified_flags.append(False)
            else:
                verified_flags.append(bool(verified))

            violations = r.get("violations", [])
            if isinstance(violations, list):
                violation_counts.append(len(violations))
            elif isinstance(violations, int):
                violation_counts.append(violations)
            else:
                violation_counts.append(0)

        n_windows = len(receipts)
        avg_ratio = statistics.mean(ratios) if ratios else 0.0
        total_savings = sum(savings)

        if verified_flags:
            breach_count = sum(1 for v in verified_flags if not v)
            slo_breach_rate = (breach_count / len(verified_flags)) * 100.0
        else:
            slo_breach_rate = 0.0

        total_violations = sum(violation_counts)

        # Compute v7 metrics
        exploit_patterns = exploit_patterns_by_hook.get(hook, set())
        exploit_count = _compute_exploit_count(receipts, exploit_patterns)

        cross_domain_validations = cross_domain_by_target.get(hook, [])
        cross_domain_links = _compute_cross_domain_links(cross_domain_validations)

        clarity_receipts = clarity_by_hook.get(hook, [])
        clarity_quality_score = _compute_clarity_quality_score(clarity_receipts)

        # Store in nested dict
        if company not in aggregated:
            aggregated[company] = {}

        aggregated[company][hook] = {
            "hook": hook,
            "windows": n_windows,
            "avg_ratio": round(avg_ratio, 2),
            "total_savings": round(total_savings, 2),
            "slo_breach_rate": round(slo_breach_rate, 2),
            "constraint_violations": total_violations,
            "exploit_count": exploit_count,
            "cross_domain_links": cross_domain_links,
            "clarity_quality_score": clarity_quality_score,
        }

    return aggregated


# =============================================================================
# v2 OUTPUT EMITTER
# =============================================================================


def emit_view_v2(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: str = "data/mesh_view.json",
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Emit aggregated metrics to JSON file and receipt.

    Args:
        metrics: Nested metrics dict from compute_metrics_v2().
        output_path: Path to write mesh view JSON.
        manifest: Optional manifest for metadata.

    Returns:
        Path to emitted mesh view JSON file.
    """
    output_path_obj = Path(output_path)

    # Ensure parent directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Build output structure
    output: Dict[str, Any] = {
        "_meta": {
            "version": "v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    if manifest:
        output["_meta"]["run_id"] = manifest.get("run_id", "unknown")
        output["_meta"]["fleet_size"] = manifest.get("fleet_size", 0)
        output["_meta"]["total_windows"] = manifest.get("total_windows", 0)

    output["companies"] = metrics

    # Compute summary stats
    total_hooks = 0
    total_windows = 0
    total_savings = 0.0
    total_exploit_count = 0
    total_cross_domain_links = 0

    for company_data in metrics.values():
        for hook_data in company_data.values():
            total_hooks += 1
            total_windows += hook_data.get("windows", 0)
            total_savings += hook_data.get("total_savings", 0.0)
            total_exploit_count += hook_data.get("exploit_count", 0)
            total_cross_domain_links += hook_data.get("cross_domain_links", 0)

    output["_meta"]["summary"] = {
        "total_companies": len(metrics),
        "total_hooks": total_hooks,
        "total_windows": total_windows,
        "total_savings_M": round(total_savings, 2),
        "total_exploit_count": total_exploit_count,
        "total_cross_domain_links": total_cross_domain_links,
    }

    # Write mesh view JSON
    with open(output_path_obj, "w") as f:
        json.dump(output, f, indent=2)

    # Write receipt entry
    receipt_path = output_path_obj.parent / "mesh_view_receipt.jsonl"
    receipt_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "mesh_view_generated",
        "version": "v2",
        "output_path": str(output_path_obj),
        "summary": output["_meta"]["summary"],
    }

    with open(receipt_path, "a") as f:
        f.write(json.dumps(receipt_entry) + "\n")

    return str(output_path_obj)


# =============================================================================
# v2 CONVENIENCE FUNCTION
# =============================================================================


def generate_mesh_view(
    receipts_path: str,
    manifest_path: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    One-call function that loads all sources, joins, computes, and returns.

    Args:
        receipts_path: Path to QED receipts file or directory.
        manifest_path: Path to manifest JSON file.
        **kwargs: Optional paths for alternate data sources:
            - clarity_path: Path to clarity receipts (default: data/clarity_receipts.jsonl)
            - sim_results_path: Path to sim results (default: data/sim_results.json)
            - anomalies_path: Path to anomaly library (default: data/shared_anomalies.jsonl)
            - cross_domain_path: Path to cross-domain validations (default: data/cross_domain_validations.jsonl)
            - output_path: Path to write output (default: data/mesh_view.json)
            - emit: Whether to emit output files (default: True)

    Returns:
        Dict with keys:
            - _meta: run metadata and summary
            - companies: per-company/hook metrics
    """
    # Extract optional paths with defaults
    clarity_path = kwargs.get("clarity_path", "data/clarity_receipts.jsonl")
    sim_results_path = kwargs.get("sim_results_path", "data/sim_results.json")
    anomalies_path = kwargs.get("anomalies_path", "data/shared_anomalies.jsonl")
    cross_domain_path = kwargs.get("cross_domain_path", "data/cross_domain_validations.jsonl")
    output_path = kwargs.get("output_path", "data/mesh_view.json")
    emit = kwargs.get("emit", True)

    # Join all sources
    joined_data = join_all_sources(
        receipts_path=receipts_path,
        manifest_path=manifest_path,
        clarity_path=clarity_path,
        sim_results_path=sim_results_path,
        anomalies_path=anomalies_path,
        cross_domain_path=cross_domain_path,
    )

    # Compute metrics
    metrics = compute_metrics_v2(joined_data)

    # Build output
    manifest = joined_data.get("manifest", {})

    output: Dict[str, Any] = {
        "_meta": {
            "version": "v2",
            "run_id": manifest.get("run_id", "unknown"),
            "fleet_size": manifest.get("fleet_size", 0),
            "total_windows": manifest.get("total_windows", manifest.get("windows", 0)),
        },
        "companies": metrics,
    }

    # Emit if requested
    if emit:
        emit_view_v2(metrics, output_path, manifest)

    return output


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main(
    manifest_path: str,
    receipts_path: str,
    output_format: str = "json",
    **kwargs: Any,
) -> int:
    """
    Main entry point for mesh view v2 aggregation.

    Args:
        manifest_path: Path to qed_run_manifest.json.
        receipts_path: Path to receipts file or directory.
        output_format: Output format - 'json' or 'table' (default: 'json').
        **kwargs: Additional paths for v7 data sources.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        result = generate_mesh_view(
            receipts_path=receipts_path,
            manifest_path=manifest_path,
            emit=False,  # Don't emit in main, we'll print
            **kwargs,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        return 1

    metrics = result.get("companies", {})

    if not metrics:
        print("Warning: No metrics computed", file=sys.stderr)

    if output_format == "table":
        print_table(metrics)
    else:
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mesh View v2 - QED v7 Telemetry Aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mesh_view_v2.py manifest.json receipts/
  python mesh_view_v2.py manifest.json receipts.jsonl --format table
  python mesh_view_v2.py manifest.json receipts/ --clarity data/clarity.jsonl
        """,
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to qed_run_manifest.json",
    )
    parser.add_argument(
        "receipts_path",
        type=str,
        help="Path to receipts file or directory",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "table"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--clarity",
        type=str,
        default="data/clarity_receipts.jsonl",
        help="Path to clarity receipts (default: data/clarity_receipts.jsonl)",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="data/sim_results.json",
        help="Path to sim results (default: data/sim_results.json)",
    )
    parser.add_argument(
        "--anomalies",
        type=str,
        default="data/shared_anomalies.jsonl",
        help="Path to anomaly library (default: data/shared_anomalies.jsonl)",
    )
    parser.add_argument(
        "--cross-domain",
        type=str,
        default="data/cross_domain_validations.jsonl",
        help="Path to cross-domain validations (default: data/cross_domain_validations.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/mesh_view.json",
        help="Path to output file (default: data/mesh_view.json)",
    )

    args = parser.parse_args()

    sys.exit(
        main(
            manifest_path=args.manifest_path,
            receipts_path=args.receipts_path,
            output_format=args.format,
            clarity_path=args.clarity,
            sim_results_path=args.sim,
            anomalies_path=args.anomalies,
            cross_domain_path=args.cross_domain,
            output_path=args.output,
        )
    )
