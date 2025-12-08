"""
NHTSA Recall Pipeline for QED v7

Converts public NHTSA recall data into AnomalyPattern proposals.
All patterns require human approval before being added to the library.
Hard cap of 10 new patterns per calendar month prevents library bloat.

API Reference: https://api.nhtsa.gov/recalls/recallsByVehicle
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import json
import uuid

from shared_anomalies import AnomalyPattern, append_pattern, generate_pattern_id


# --- Constants ---

MONTHLY_CAP = 10
MIN_DOLLAR_VALUE_THRESHOLD = 100_000

# Component to physics_domain mapping
COMPONENT_MAPPING = {
    "SERVICE BRAKES": "brake",
    "ELECTRICAL SYSTEM": "electrical",
    "STEERING": "steering",
    "ENGINE AND ENGINE COOLING": "thermal",
    "FUEL SYSTEM": "thermal",
    "VEHICLE SPEED CONTROL": "throttle",
    "POWER TRAIN": "thermal",
    "AIR BAGS": "safety",
    "SEAT BELTS": "safety",
}

# Safety-critical components requiring extra review
SAFETY_CRITICAL_COMPONENTS = {"brake", "steering", "throttle", "safety"}

# Failure probability by component type
FAILURE_PROBABILITY = {
    "brake": 0.02,
    "steering": 0.02,
    "throttle": 0.02,
    "safety": 0.02,
    "thermal": 0.01,
    "battery": 0.01,
    "electrical": 0.005,
    "sensor": 0.005,
}

# Cost per incident by severity
COST_PER_INCIDENT = {
    "HIGH": 500_000,      # Injury potential
    "MEDIUM": 25_000,     # Vehicle damage
    "LOW": 5_000,         # Inconvenience
}


# --- Enums and Dataclasses ---

class ProposalStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    PARTIALLY_APPROVED = "partially_approved"
    FULLY_APPROVED = "fully_approved"
    REJECTED = "rejected"


@dataclass
class PatternApproval:
    pattern_id: str
    approved: bool
    approver: Optional[str] = None
    reason: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ProposalReceipt:
    proposal_id: str
    timestamp: str
    patterns: List[Dict[str, Any]]
    status: str
    month_year: str
    approvals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rejections: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# --- Core Functions ---

def parse_nhtsa_recall(recall_data: dict) -> dict:
    """
    Parse raw NHTSA recall data into normalized format.

    Args:
        recall_data: Raw recall data from NHTSA API

    Returns:
        Normalized dict with: campaign_number, component, summary,
        consequence, units_affected, report_date, physics_domain
    """
    # Extract raw fields (handle various API field name formats)
    campaign_number = recall_data.get("NHTSACampaignNumber") or recall_data.get("campaign_number", "")
    component = recall_data.get("Component") or recall_data.get("component", "")
    summary = recall_data.get("Summary") or recall_data.get("summary", "")
    consequence = recall_data.get("Consequence") or recall_data.get("consequence", "")
    units_affected = recall_data.get("PotentialNumberOfUnitsAffected") or recall_data.get("units_affected", 0)
    report_date = recall_data.get("ReportReceivedDate") or recall_data.get("report_date", "")

    # Normalize component to physics_domain
    component_upper = component.upper().strip()
    physics_domain = COMPONENT_MAPPING.get(component_upper, "sensor")

    # Handle "ELECTRICAL SYSTEM" special case - could be battery
    if component_upper == "ELECTRICAL SYSTEM":
        summary_lower = summary.lower()
        if any(term in summary_lower for term in ["battery", "cell", "thermal runaway", "fire"]):
            physics_domain = "battery"

    # Ensure units_affected is int
    if isinstance(units_affected, str):
        units_affected = int(units_affected.replace(",", "")) if units_affected else 0

    return {
        "campaign_number": campaign_number,
        "component": component,
        "summary": summary,
        "consequence": consequence,
        "units_affected": units_affected,
        "report_date": report_date,
        "physics_domain": physics_domain,
    }


def estimate_dollar_value(units_affected: int, component: str, severity: str) -> float:
    """
    Estimate annual dollar value of a recall pattern.

    Formula: units_affected × failure_probability × cost_per_incident

    Args:
        units_affected: Number of vehicles affected
        component: Physics domain (brake, thermal, etc.)
        severity: HIGH, MEDIUM, or LOW

    Returns:
        Estimated annual dollar value
    """
    # Get failure probability for component
    failure_prob = FAILURE_PROBABILITY.get(component, 0.005)

    # Get cost per incident by severity
    severity_upper = severity.upper() if severity else "MEDIUM"
    cost = COST_PER_INCIDENT.get(severity_upper, COST_PER_INCIDENT["MEDIUM"])

    # Calculate dollar value
    dollar_value = units_affected * failure_prob * cost

    return dollar_value


def _extract_failure_mode(summary: str, consequence: str, physics_domain: str) -> str:
    """Extract failure mode from recall summary and consequence text."""
    text = (summary + " " + consequence).lower()

    # Domain-specific failure mode detection
    failure_mode_keywords = {
        "brake": {
            "loss": "brake_loss",
            "fail": "brake_failure",
            "reduce": "reduced_braking",
            "leak": "fluid_leak",
            "soft": "soft_pedal",
        },
        "steering": {
            "loss": "steering_loss",
            "lock": "steering_lock",
            "stiff": "steering_stiff",
            "assist": "power_assist_failure",
        },
        "thermal": {
            "overheat": "overheat",
            "fire": "thermal_runaway",
            "leak": "coolant_leak",
            "stall": "engine_stall",
        },
        "battery": {
            "fire": "thermal_runaway",
            "overheat": "overheat",
            "short": "short_circuit",
            "fail": "cell_failure",
        },
        "throttle": {
            "stuck": "stuck_throttle",
            "unintended": "unintended_acceleration",
            "loss": "throttle_loss",
        },
        "electrical": {
            "short": "short_circuit",
            "fail": "component_failure",
            "loss": "power_loss",
        },
    }

    # Check domain-specific keywords
    domain_keywords = failure_mode_keywords.get(physics_domain, {})
    for keyword, mode in domain_keywords.items():
        if keyword in text:
            return mode

    # Default failure modes by domain
    default_modes = {
        "brake": "brake_anomaly",
        "steering": "steering_anomaly",
        "thermal": "thermal_anomaly",
        "battery": "battery_anomaly",
        "throttle": "throttle_anomaly",
        "electrical": "electrical_anomaly",
        "sensor": "sensor_anomaly",
        "safety": "safety_anomaly",
    }

    return default_modes.get(physics_domain, "unknown_anomaly")


def _determine_severity(consequence: str) -> str:
    """Determine severity level from consequence text."""
    text = consequence.lower()

    # HIGH severity indicators (injury potential)
    high_keywords = ["crash", "injury", "fire", "death", "collision", "accident", "burn"]
    if any(kw in text for kw in high_keywords):
        return "HIGH"

    # LOW severity indicators (inconvenience only)
    low_keywords = ["inconvenience", "warning", "indicator", "display", "annoyance"]
    if any(kw in text for kw in low_keywords):
        return "LOW"

    # Default to MEDIUM
    return "MEDIUM"


def propose_patterns_from_nhtsa(recalls_data: List[dict]) -> List[AnomalyPattern]:
    """
    Convert NHTSA recalls to candidate AnomalyPatterns.

    Args:
        recalls_data: List of raw NHTSA recall records

    Returns:
        List of proposed AnomalyPatterns (not yet approved)
    """
    patterns = []

    for recall in recalls_data:
        # Parse the recall data
        parsed = parse_nhtsa_recall(recall)

        # Determine severity and failure mode
        severity = _determine_severity(parsed["consequence"])
        failure_mode = _extract_failure_mode(
            parsed["summary"],
            parsed["consequence"],
            parsed["physics_domain"]
        )

        # Estimate dollar value
        dollar_value = estimate_dollar_value(
            parsed["units_affected"],
            parsed["physics_domain"],
            severity
        )

        # Skip patterns below threshold
        if dollar_value < MIN_DOLLAR_VALUE_THRESHOLD:
            continue

        # Generate pattern_id from campaign_number + component
        id_content = f"{parsed['campaign_number']}:{parsed['physics_domain']}"
        pattern_id = hashlib.sha3_256(id_content.encode()).hexdigest()[:16]

        # Build params
        params = {
            "nhtsa_campaign": parsed["campaign_number"],
            "source": "nhtsa_recall",
            "units_affected": parsed["units_affected"],
            "severity": severity,
            "original_component": parsed["component"],
            "report_date": parsed["report_date"],
        }

        # Create AnomalyPattern
        pattern = AnomalyPattern(
            pattern_id=pattern_id,
            physics_domain=parsed["physics_domain"],
            failure_mode=failure_mode,
            params=params,
            hooks=["nhtsa"],  # Default hook, can be expanded during approval
            dollar_value_annual=dollar_value,
            validation_recall=0.0,  # Not yet validated
            false_positive_rate=1.0,  # Not yet validated
            cross_domain_targets=[],
            deprecated=False,
        )

        patterns.append(pattern)

    return patterns


def check_monthly_cap(proposals_path: str = "data/proposal_receipts.jsonl") -> Tuple[bool, int]:
    """
    Check if we can propose more patterns this month.

    Args:
        proposals_path: Path to proposal receipts JSONL

    Returns:
        Tuple of (can_propose: bool, remaining_slots: int)
    """
    current_month_year = datetime.now().strftime("%Y-%m")

    # Count proposals in current month
    proposals_this_month = 0
    p = Path(proposals_path)

    if p.exists():
        with open(p, 'r') as f:
            for line in f:
                if line.strip():
                    receipt = json.loads(line)
                    if receipt.get("month_year") == current_month_year:
                        # Count patterns in this proposal
                        proposals_this_month += len(receipt.get("patterns", []))

    remaining = MONTHLY_CAP - proposals_this_month
    can_propose = remaining > 0

    return (can_propose, max(0, remaining))


def emit_proposal_receipt(
    patterns: List[AnomalyPattern],
    output_path: str = "data/proposal_receipts.jsonl"
) -> Optional[str]:
    """
    Emit a proposal receipt for human review.

    Args:
        patterns: List of proposed AnomalyPatterns
        output_path: Path to write receipt

    Returns:
        proposal_id if successful, None if cap exceeded
    """
    # Check monthly cap
    can_propose, remaining = check_monthly_cap(output_path)

    if not can_propose:
        return None

    # Limit patterns to remaining slots
    patterns_to_propose = patterns[:remaining]

    if not patterns_to_propose:
        return None

    # Create receipt
    proposal_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    month_year = datetime.now().strftime("%Y-%m")

    # Convert patterns to dicts with extra proposal metadata
    pattern_dicts = []
    for p in patterns_to_propose:
        pattern_dict = asdict(p)
        pattern_dict["safety_critical"] = p.physics_domain in SAFETY_CRITICAL_COMPONENTS
        pattern_dict["requires_extra_review"] = pattern_dict["safety_critical"]
        pattern_dicts.append(pattern_dict)

    receipt = ProposalReceipt(
        proposal_id=proposal_id,
        timestamp=timestamp,
        patterns=pattern_dicts,
        status=ProposalStatus.PENDING_APPROVAL.value,
        month_year=month_year,
        approvals={},
        rejections={},
    )

    # Write to JSONL
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'a') as f:
        f.write(json.dumps(asdict(receipt)) + '\n')

    return proposal_id


def _load_receipts(proposals_path: str) -> List[dict]:
    """Load all proposal receipts."""
    receipts = []
    p = Path(proposals_path)
    if p.exists():
        with open(p, 'r') as f:
            for line in f:
                if line.strip():
                    receipts.append(json.loads(line))
    return receipts


def _save_receipts(receipts: List[dict], proposals_path: str) -> None:
    """Save all proposal receipts (overwrites file)."""
    p = Path(proposals_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        for receipt in receipts:
            f.write(json.dumps(receipt) + '\n')


def _update_receipt_status(receipt: dict) -> str:
    """Update receipt status based on approvals/rejections."""
    patterns = receipt.get("patterns", [])
    approvals = receipt.get("approvals", {})
    rejections = receipt.get("rejections", {})

    total = len(patterns)
    approved_count = len(approvals)
    rejected_count = len(rejections)
    decided_count = approved_count + rejected_count

    if decided_count == 0:
        return ProposalStatus.PENDING_APPROVAL.value
    elif rejected_count == total:
        return ProposalStatus.REJECTED.value
    elif decided_count == total:
        return ProposalStatus.FULLY_APPROVED.value
    else:
        return ProposalStatus.PARTIALLY_APPROVED.value


def approve_pattern(
    proposal_id: str,
    pattern_id: str,
    approver: str,
    proposals_path: str = "data/proposal_receipts.jsonl"
) -> bool:
    """
    Approve a specific pattern in a proposal.

    Args:
        proposal_id: UUID of the proposal
        pattern_id: ID of the pattern to approve
        approver: Name/ID of the human approver
        proposals_path: Path to proposal receipts

    Returns:
        True if approved successfully, False otherwise
    """
    receipts = _load_receipts(proposals_path)

    for receipt in receipts:
        if receipt["proposal_id"] == proposal_id:
            # Verify pattern exists in this proposal
            pattern_ids = [p["pattern_id"] for p in receipt.get("patterns", [])]
            if pattern_id not in pattern_ids:
                return False

            # Check not already rejected
            if pattern_id in receipt.get("rejections", {}):
                return False

            # Add approval
            if "approvals" not in receipt:
                receipt["approvals"] = {}

            receipt["approvals"][pattern_id] = {
                "approver": approver,
                "timestamp": datetime.now().isoformat(),
            }

            # Update status
            receipt["status"] = _update_receipt_status(receipt)

            _save_receipts(receipts, proposals_path)
            return True

    return False


def reject_pattern(
    proposal_id: str,
    pattern_id: str,
    reason: str,
    proposals_path: str = "data/proposal_receipts.jsonl"
) -> bool:
    """
    Reject a specific pattern in a proposal.

    Args:
        proposal_id: UUID of the proposal
        pattern_id: ID of the pattern to reject
        reason: Reason for rejection
        proposals_path: Path to proposal receipts

    Returns:
        True if rejected successfully, False otherwise
    """
    receipts = _load_receipts(proposals_path)

    for receipt in receipts:
        if receipt["proposal_id"] == proposal_id:
            # Verify pattern exists in this proposal
            pattern_ids = [p["pattern_id"] for p in receipt.get("patterns", [])]
            if pattern_id not in pattern_ids:
                return False

            # Check not already approved
            if pattern_id in receipt.get("approvals", {}):
                return False

            # Add rejection
            if "rejections" not in receipt:
                receipt["rejections"] = {}

            receipt["rejections"][pattern_id] = {
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            }

            # Update status
            receipt["status"] = _update_receipt_status(receipt)

            _save_receipts(receipts, proposals_path)
            return True

    return False


def append_approved_patterns(
    proposal_id: str,
    library_path: str = "data/shared_anomalies.jsonl",
    proposals_path: str = "data/proposal_receipts.jsonl"
) -> int:
    """
    Append approved patterns from a proposal to the shared library.

    Args:
        proposal_id: UUID of the proposal
        library_path: Path to shared anomalies JSONL
        proposals_path: Path to proposal receipts

    Returns:
        Count of patterns successfully added
    """
    receipts = _load_receipts(proposals_path)
    count = 0

    for receipt in receipts:
        if receipt["proposal_id"] == proposal_id:
            approvals = receipt.get("approvals", {})

            for pattern_dict in receipt.get("patterns", []):
                pattern_id = pattern_dict["pattern_id"]

                # Only add approved patterns
                if pattern_id not in approvals:
                    continue

                # Remove proposal-specific metadata
                clean_dict = {k: v for k, v in pattern_dict.items()
                             if k not in ["safety_critical", "requires_extra_review"]}

                # Create AnomalyPattern
                pattern = AnomalyPattern(**clean_dict)

                # Append to library
                if append_pattern(pattern, library_path):
                    count += 1

            break

    return count


def get_proposal_summary(
    proposal_id: str,
    proposals_path: str = "data/proposal_receipts.jsonl"
) -> Optional[dict]:
    """
    Get a summary of a proposal for review.

    Args:
        proposal_id: UUID of the proposal
        proposals_path: Path to proposal receipts

    Returns:
        Summary dict or None if not found
    """
    receipts = _load_receipts(proposals_path)

    for receipt in receipts:
        if receipt["proposal_id"] == proposal_id:
            patterns = receipt.get("patterns", [])
            approvals = receipt.get("approvals", {})
            rejections = receipt.get("rejections", {})

            return {
                "proposal_id": proposal_id,
                "timestamp": receipt["timestamp"],
                "status": receipt["status"],
                "total_patterns": len(patterns),
                "approved_count": len(approvals),
                "rejected_count": len(rejections),
                "pending_count": len(patterns) - len(approvals) - len(rejections),
                "total_dollar_value": sum(p.get("dollar_value_annual", 0) for p in patterns),
                "safety_critical_count": sum(1 for p in patterns if p.get("safety_critical", False)),
                "patterns": [
                    {
                        "pattern_id": p["pattern_id"],
                        "physics_domain": p["physics_domain"],
                        "failure_mode": p["failure_mode"],
                        "dollar_value_annual": p["dollar_value_annual"],
                        "safety_critical": p.get("safety_critical", False),
                        "status": (
                            "approved" if p["pattern_id"] in approvals
                            else "rejected" if p["pattern_id"] in rejections
                            else "pending"
                        ),
                    }
                    for p in patterns
                ],
            }

    return None


def list_pending_proposals(
    proposals_path: str = "data/proposal_receipts.jsonl"
) -> List[dict]:
    """
    List all proposals with pending patterns.

    Returns:
        List of proposal summaries with pending patterns
    """
    receipts = _load_receipts(proposals_path)
    pending = []

    for receipt in receipts:
        status = receipt.get("status", "")
        if status in [ProposalStatus.PENDING_APPROVAL.value,
                      ProposalStatus.PARTIALLY_APPROVED.value]:
            summary = get_proposal_summary(receipt["proposal_id"], proposals_path)
            if summary and summary["pending_count"] > 0:
                pending.append(summary)

    return pending
