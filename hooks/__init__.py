# Make hooks a package so we can run `python -m hooks.<hook>`

from dataclasses import dataclass
from typing import Dict, List, Callable, Any


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware profile for deployment targeting."""
    platform: str      # "vehicle_ecu", "flight_computer", "satellite", "tunnel_plc", "neural_implant", "gpu_cluster"
    compute_class: str # "embedded", "edge", "server", "distributed"
    connectivity: str  # "cellular", "satellite", "fiber", "rf", "neural_link"
    safety_critical: bool


@dataclass(frozen=True)
class DeploymentDefaults:
    """Default deployment configuration for a hook."""
    hook: str
    recall_floor: float
    max_fp_rate: float
    slo_latency_ms: int
    slo_breach_budget: float
    compression_target: float
    enabled_patterns: List[str]
    regulatory_flags: Dict[str, bool]


# Registry populated lazily on first access to avoid circular imports
_HOOKS: Dict[str, Any] = {}
_HOOKS_LOADED: bool = False


def _load_hooks() -> None:
    """Lazily load hook modules into registry."""
    global _HOOKS, _HOOKS_LOADED
    if _HOOKS_LOADED:
        return

    from hooks import tesla, spacex, starlink, boring, neuralink, xai

    _HOOKS = {
        "tesla": tesla,
        "spacex": spacex,
        "starlink": starlink,
        "boring": boring,
        "neuralink": neuralink,
        "xai": xai,
    }
    _HOOKS_LOADED = True


def get_hook_config(name: str) -> DeploymentDefaults:
    """
    Get deployment configuration for a hook by name.

    Args:
        name: Hook name (tesla, spacex, starlink, boring, neuralink, xai)

    Returns:
        DeploymentDefaults for the requested hook

    Raises:
        ValueError: If hook name is not found
    """
    _load_hooks()
    if name not in _HOOKS:
        raise ValueError(f"unknown hook: {name}")
    return _HOOKS[name].get_deployment_config()


def get_hook_hardware(name: str) -> HardwareProfile:
    """
    Get hardware profile for a hook by name.

    Args:
        name: Hook name (tesla, spacex, starlink, boring, neuralink, xai)

    Returns:
        HardwareProfile for the requested hook

    Raises:
        ValueError: If hook name is not found
    """
    _load_hooks()
    if name not in _HOOKS:
        raise ValueError(f"unknown hook: {name}")
    return _HOOKS[name].get_hardware_profile()


def list_hooks() -> List[str]:
    """Return list of available hook names."""
    _load_hooks()
    return list(_HOOKS.keys())


# Re-export dataclasses for convenience
__all__ = [
    "HardwareProfile",
    "DeploymentDefaults",
    "get_hook_config",
    "get_hook_hardware",
    "list_hooks",
]
