"""
Hooks package for QED v5.0.

Provides a unified registry of all domain hooks with programmatic access
to normalize functions, cross-domain configs, deployment configs, and
hardware profiles.
"""

from typing import Any, Dict, List

# Import normalize functions (main entry point for each hook)
from hooks.tesla import main as tesla_normalize
from hooks.spacex import main as spacex_normalize
from hooks.starlink import main as starlink_normalize
from hooks.boring import main as boring_normalize
from hooks.neuralink import main as neuralink_normalize
from hooks.xai import main as xai_normalize

# Import cross-domain config functions
from hooks.tesla import get_cross_domain_config as tesla_get_cross_domain_config
from hooks.spacex import get_cross_domain_config as spacex_get_cross_domain_config
from hooks.starlink import get_cross_domain_config as starlink_get_cross_domain_config
from hooks.boring import get_cross_domain_config as boring_get_cross_domain_config
from hooks.neuralink import get_cross_domain_config as neuralink_get_cross_domain_config
from hooks.xai import get_cross_domain_config as xai_get_cross_domain_config

# Import deployment config functions
from hooks.tesla import get_deployment_config as tesla_get_deployment_config
from hooks.spacex import get_deployment_config as spacex_get_deployment_config
from hooks.starlink import get_deployment_config as starlink_get_deployment_config
from hooks.boring import get_deployment_config as boring_get_deployment_config
from hooks.neuralink import get_deployment_config as neuralink_get_deployment_config
from hooks.xai import get_deployment_config as xai_get_deployment_config

# Import hardware profile functions
from hooks.tesla import get_hardware_profile as tesla_get_hardware_profile
from hooks.spacex import get_hardware_profile as spacex_get_hardware_profile
from hooks.starlink import get_hardware_profile as starlink_get_hardware_profile
from hooks.boring import get_hardware_profile as boring_get_hardware_profile
from hooks.neuralink import get_hardware_profile as neuralink_get_hardware_profile
from hooks.xai import get_hardware_profile as xai_get_hardware_profile

# -----------------------------------------------------------------------------
# HOOKS registry: maps hook name to dict of functions
# -----------------------------------------------------------------------------
HOOKS: Dict[str, Dict[str, Any]] = {
    "tesla": {
        "normalize": tesla_normalize,
        "get_cross_domain_config": tesla_get_cross_domain_config,
        "get_deployment_config": tesla_get_deployment_config,
        "get_hardware_profile": tesla_get_hardware_profile,
    },
    "spacex": {
        "normalize": spacex_normalize,
        "get_cross_domain_config": spacex_get_cross_domain_config,
        "get_deployment_config": spacex_get_deployment_config,
        "get_hardware_profile": spacex_get_hardware_profile,
    },
    "starlink": {
        "normalize": starlink_normalize,
        "get_cross_domain_config": starlink_get_cross_domain_config,
        "get_deployment_config": starlink_get_deployment_config,
        "get_hardware_profile": starlink_get_hardware_profile,
    },
    "boring": {
        "normalize": boring_normalize,
        "get_cross_domain_config": boring_get_cross_domain_config,
        "get_deployment_config": boring_get_deployment_config,
        "get_hardware_profile": boring_get_hardware_profile,
    },
    "neuralink": {
        "normalize": neuralink_normalize,
        "get_cross_domain_config": neuralink_get_cross_domain_config,
        "get_deployment_config": neuralink_get_deployment_config,
        "get_hardware_profile": neuralink_get_hardware_profile,
    },
    "xai": {
        "normalize": xai_normalize,
        "get_cross_domain_config": xai_get_cross_domain_config,
        "get_deployment_config": xai_get_deployment_config,
        "get_hardware_profile": xai_get_hardware_profile,
    },
}


def get_hook(name: str) -> Dict[str, Any]:
    """
    Return function dict for the given hook name.

    Args:
        name: Hook name (tesla, spacex, starlink, boring, neuralink, xai)

    Returns:
        Dict with keys: normalize, get_cross_domain_config,
        get_deployment_config, get_hardware_profile

    Raises:
        KeyError: If hook name is not found in registry
    """
    if name not in HOOKS:
        raise KeyError(f"unknown hook: {name}")
    return HOOKS[name]


def list_hooks() -> List[str]:
    """
    Return list of all registered hook names.

    Returns:
        List of hook names in registry order
    """
    return list(HOOKS.keys())


def get_all_deployment_configs() -> Dict[str, Dict[str, Any]]:
    """
    Return deployment configs for all hooks.

    Returns:
        Dict mapping hook name to its deployment config
    """
    return {name: funcs["get_deployment_config"]() for name, funcs in HOOKS.items()}


def get_all_hardware_profiles() -> Dict[str, Dict[str, Any]]:
    """
    Return hardware profiles for all hooks.

    Returns:
        Dict mapping hook name to its hardware profile
    """
    return {name: funcs["get_hardware_profile"]() for name, funcs in HOOKS.items()}
