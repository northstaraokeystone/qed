from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import hashlib
import json
from pathlib import Path


@dataclass
class AnomalyPattern:
    pattern_id: str                          # SHA3 hash of (physics_domain + failure_mode + params)
    physics_domain: str                      # e.g., "battery_thermal", "comms", "motion"
    failure_mode: str                        # e.g., "overheat", "signal_loss", "drift"
    params: Dict[str, Any]                   # domain-specific parameters
    hooks: List[str]                         # which hooks use this pattern, e.g., ["tesla", "spacex"]
    dollar_value_annual: float               # estimated annual savings in USD
    validation_recall: float = 0.0           # measured recall from sims (0.0 to 1.0)
    false_positive_rate: float = 1.0         # measured FP rate from sims (0.0 to 1.0)
    cross_domain_targets: List[str] = field(default_factory=list)  # hooks this can transfer to
    deprecated: bool = False

    @property
    def training_score(self) -> float:
        """Score for training priority: min(1, dollar_value / 10M)"""
        return min(1.0, self.dollar_value_annual / 10_000_000)

    @property
    def training_role(self) -> str:
        """train_cross_company if >$10M, else observe_only"""
        return "train_cross_company" if self.dollar_value_annual > 10_000_000 else "observe_only"

    @property
    def exploit_grade(self) -> bool:
        """True if high value AND high recall AND low FP"""
        return (
            self.dollar_value_annual > 1_000_000
            and self.validation_recall >= 0.99
            and self.false_positive_rate <= 0.01
        )


def generate_pattern_id(physics_domain: str, failure_mode: str, params: Dict) -> str:
    """Generate SHA3-256 hash for pattern immutability."""
    content = f"{physics_domain}:{failure_mode}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha3_256(content.encode()).hexdigest()[:16]


def load_library(path: str = "data/shared_anomalies.jsonl") -> List[AnomalyPattern]:
    """Load all patterns from JSONL file."""
    patterns = []
    p = Path(path)
    if not p.exists():
        return patterns
    with open(p, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                patterns.append(AnomalyPattern(**data))
    return patterns


def append_pattern(pattern: AnomalyPattern, path: str = "data/shared_anomalies.jsonl") -> bool:
    """Append pattern if pattern_id not already present (immutability check)."""
    existing = load_library(path)
    existing_ids = {p.pattern_id for p in existing}

    if pattern.pattern_id in existing_ids:
        return False  # already exists, immutable

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a') as f:
        f.write(json.dumps(asdict(pattern)) + '\n')
    return True


def get_patterns_for_hook(hook_name: str, path: str = "data/shared_anomalies.jsonl") -> List[AnomalyPattern]:
    """Return patterns where hook_name is in the hooks list."""
    all_patterns = load_library(path)
    return [p for p in all_patterns if hook_name in p.hooks and not p.deprecated]
