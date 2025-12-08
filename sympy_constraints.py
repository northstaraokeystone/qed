"""
Sympy-based constraint definitions for QED v6.

This module provides symbolic constraint definitions that can be evaluated
numerically. Constraints are defined per scenario/hook_name and express
bounds on telemetry parameters like amplitude.

The ∀t bounds (e.g., |A(t)| ≤ bound) are represented as dictionaries
with constraint metadata that can be evaluated numerically.
"""

from typing import Any, Dict, List

# Constraint definitions per scenario/hook_name
# Each constraint has:
#   id: unique identifier
#   bound: numeric bound value
#   description: human-readable description
#   sympy_expr: symbolic expression (for documentation/future symbolic eval)

_CONSTRAINTS: Dict[str, List[Dict[str, Any]]] = {
    "tesla_fsd": [
        {
            "id": "amplitude_bound_tesla",
            "bound": 14.7,
            "description": "∀t: |A(t)| ≤ 14.7 for Tesla FSD steering torque",
            "sympy_expr": "ForAll(t, Abs(A(t)) <= 14.7)",
        },
    ],
    "spacex_flight": [
        {
            "id": "amplitude_bound_spacex",
            "bound": 20.0,
            "description": "∀t: |A(t)| ≤ 20.0 for SpaceX thrust oscillation",
            "sympy_expr": "ForAll(t, Abs(A(t)) <= 20.0)",
        },
    ],
    "neuralink_stream": [
        {
            "id": "amplitude_bound_neuralink",
            "bound": 5.0,
            "description": "∀t: |A(t)| ≤ 5.0 for Neuralink neural spike bounds",
            "sympy_expr": "ForAll(t, Abs(A(t)) <= 5.0)",
        },
    ],
    "boring_tunnel": [
        {
            "id": "amplitude_bound_boring",
            "bound": 25.0,
            "description": "∀t: |A(t)| ≤ 25.0 for Boring tunnel vibration",
            "sympy_expr": "ForAll(t, Abs(A(t)) <= 25.0)",
        },
    ],
    "starlink_flow": [
        {
            "id": "amplitude_bound_starlink",
            "bound": 18.0,
            "description": "∀t: |A(t)| ≤ 18.0 for Starlink signal bounds",
            "sympy_expr": "ForAll(t, Abs(A(t)) <= 18.0)",
        },
    ],
    "xai_eval": [
        {
            "id": "amplitude_bound_xai",
            "bound": 10.0,
            "description": "∀t: |A(t)| ≤ 10.0 for xAI evaluation bounds",
            "sympy_expr": "ForAll(t, Abs(A(t)) <= 10.0)",
        },
    ],
    "generic": [
        {
            "id": "amplitude_bound_generic",
            "bound": 14.7,
            "description": "∀t: |A(t)| ≤ 14.7 default bound",
            "sympy_expr": "ForAll(t, Abs(A(t)) <= 14.7)",
        },
    ],
}


def get_constraints(hook_name: str) -> List[Dict[str, Any]]:
    """
    Get constraint definitions for a given hook_name/scenario.

    Args:
        hook_name: The scenario or hook identifier (e.g., 'tesla_fsd', 'generic')

    Returns:
        List of constraint dictionaries with id, bound, description, sympy_expr
    """
    return _CONSTRAINTS.get(hook_name, _CONSTRAINTS.get("generic", []))
