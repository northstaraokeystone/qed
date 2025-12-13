"""
sim/types_result.py - SimResult Dataclass

Immutable simulation result container.
CLAUDEME v3.1 Compliant: Frozen dataclass.
"""

from dataclasses import dataclass

from .types_config import SimConfig
from .types_state import SimState


@dataclass(frozen=True)
class SimResult:
    """Immutable simulation result."""
    final_state: SimState
    all_traces: dict
    violations: list
    statistics: dict
    config: SimConfig
