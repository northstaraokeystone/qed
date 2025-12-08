"""
Sympy-based constraint definitions for QED v6 telemetry compression.

This module provides a registry of symbolic constraints per hook/scenario that can
be evaluated numerically via lambdify. Constraints express physics bounds (amplitude
∀t) and ROI invariants (savings thresholds) that gate QED receipt verification.

Architecture:
    _CONSTRAINTS: Dict[hook_name, List[ConstraintDef]]
    get_constraints(hook_name) -> List[Callable]  # Returns lambdified evaluators
    get_constraint_specs(hook_name) -> List[Dict]  # Returns raw specs for introspection

Performance:
    - lambdify is called once at module load (~0.1ms per constraint)
    - Numeric evaluation: <0.01ms per call
    - Total: <1ms for typical hook with 2-3 constraints

Constraint Types:
    1. amplitude_bound: ∀t: |A*sin(2πft)| ≤ bound  (physics safety)
    2. ratio_min: compression_ratio ≥ threshold    (efficiency gate)
    3. savings_min: savings_M ≥ threshold          (ROI gate)
    4. mse_max: reconstruction_mse ≤ threshold     (quality gate)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# SymPy imports with fallback for lightweight environments
try:
    from sympy import symbols, Abs, lambdify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# -----------------------------------------------------------------------------
# Symbolic variable definitions (used when SymPy is available)
# -----------------------------------------------------------------------------
if SYMPY_AVAILABLE:
    # Time-domain variables for physics constraints
    t, A, f, phi = symbols('t A f phi', real=True)
    # Aggregate metrics for ROI/quality gates
    ratio, savings_M, mse, recall = symbols('ratio savings_M mse recall', real=True, positive=True)
    # Bound parameter (varies by scenario)
    bound = symbols('bound', real=True, positive=True)


# -----------------------------------------------------------------------------
# Constraint specification type
# -----------------------------------------------------------------------------
ConstraintSpec = Dict[str, Any]


def _make_amplitude_constraint(
    constraint_id: str,
    amplitude_bound: float,
    description: str,
) -> ConstraintSpec:
    """
    Create amplitude bound constraint: ∀t: |A*sin(2πft + φ)| ≤ bound.

    The peak amplitude of a sinusoidal signal is |A|, so this reduces to |A| ≤ bound.
    """
    spec: ConstraintSpec = {
        "id": constraint_id,
        "type": "amplitude_bound",
        "bound": amplitude_bound,
        "description": description,
        "sympy_expr": f"ForAll(t, Abs(A*sin(2*pi*f*t + phi)) <= {amplitude_bound})",
    }

    if SYMPY_AVAILABLE:
        # For sinusoidal signals, max|A*sin(...)| = |A|, so constraint is Abs(A) <= bound
        # Pre-compile to callable: (A_val) -> bool
        expr = Abs(A) <= amplitude_bound
        spec["_sympy_obj"] = expr
        spec["_evaluator"] = lambdify(A, expr, modules=["numpy", "math"])
    else:
        # Pure Python fallback
        spec["_evaluator"] = lambda A_val: abs(A_val) <= amplitude_bound

    return spec


def _make_ratio_constraint(
    constraint_id: str,
    min_ratio: float,
    description: str,
) -> ConstraintSpec:
    """Create compression ratio gate: ratio ≥ min_ratio."""
    spec: ConstraintSpec = {
        "id": constraint_id,
        "type": "ratio_min",
        "bound": min_ratio,
        "description": description,
        "sympy_expr": f"ratio >= {min_ratio}",
    }

    if SYMPY_AVAILABLE:
        expr = ratio >= min_ratio
        spec["_sympy_obj"] = expr
        spec["_evaluator"] = lambdify(ratio, expr, modules=["numpy", "math"])
    else:
        spec["_evaluator"] = lambda r: r >= min_ratio

    return spec


def _make_savings_constraint(
    constraint_id: str,
    min_savings_M: float,
    description: str,
) -> ConstraintSpec:
    """Create ROI gate: savings_M ≥ min_savings_M (in millions)."""
    spec: ConstraintSpec = {
        "id": constraint_id,
        "type": "savings_min",
        "bound": min_savings_M,
        "description": description,
        "sympy_expr": f"savings_M >= {min_savings_M}",
    }

    if SYMPY_AVAILABLE:
        expr = savings_M >= min_savings_M
        spec["_sympy_obj"] = expr
        spec["_evaluator"] = lambdify(savings_M, expr, modules=["numpy", "math"])
    else:
        spec["_evaluator"] = lambda s: s >= min_savings_M

    return spec


def _make_mse_constraint(
    constraint_id: str,
    max_mse: float,
    description: str,
) -> ConstraintSpec:
    """Create reconstruction quality gate: MSE ≤ max_mse."""
    spec: ConstraintSpec = {
        "id": constraint_id,
        "type": "mse_max",
        "bound": max_mse,
        "description": description,
        "sympy_expr": f"mse <= {max_mse}",
    }

    if SYMPY_AVAILABLE:
        expr = mse <= max_mse
        spec["_sympy_obj"] = expr
        spec["_evaluator"] = lambdify(mse, expr, modules=["numpy", "math"])
    else:
        spec["_evaluator"] = lambda m: m <= max_mse

    return spec


# -----------------------------------------------------------------------------
# Constraint registry: Dict[hook_name, List[ConstraintSpec]]
# -----------------------------------------------------------------------------
_CONSTRAINTS: Dict[str, List[ConstraintSpec]] = {
    # Tesla FSD: strict amplitude for steering torque safety
    "tesla_fsd": [
        _make_amplitude_constraint(
            "amplitude_bound_tesla",
            14.7,
            "∀t: |A(t)| ≤ 14.7g for Tesla FSD steering torque",
        ),
        _make_ratio_constraint(
            "ratio_min_tesla",
            20.0,
            "compression ratio ≥ 20:1 for fleet storage efficiency",
        ),
        _make_savings_constraint(
            "savings_min_tesla",
            1.0,
            "ROI ≥ $1M annual savings threshold",
        ),
    ],

    # SpaceX flight: higher amplitude tolerance, strict ROI
    "spacex_flight": [
        _make_amplitude_constraint(
            "amplitude_bound_spacex",
            20.0,
            "∀t: |A(t)| ≤ 20.0 for SpaceX thrust oscillation",
        ),
        _make_savings_constraint(
            "savings_min_spacex",
            10.0,
            "ROI ≥ $10M for launch telemetry value",
        ),
    ],

    # Neuralink: tight bounds for neural signal fidelity
    "neuralink_stream": [
        _make_amplitude_constraint(
            "amplitude_bound_neuralink",
            5.0,
            "∀t: |A(t)| ≤ 5.0 for Neuralink neural spike bounds",
        ),
        _make_mse_constraint(
            "mse_max_neuralink",
            0.001,
            "MSE ≤ 0.001 for high-fidelity neural reconstruction",
        ),
    ],

    # Boring tunnel: high vibration tolerance
    "boring_tunnel": [
        _make_amplitude_constraint(
            "amplitude_bound_boring",
            25.0,
            "∀t: |A(t)| ≤ 25.0 for Boring tunnel vibration",
        ),
        _make_ratio_constraint(
            "ratio_min_boring",
            30.0,
            "compression ratio ≥ 30:1 for tunnel sensor density",
        ),
    ],

    # Starlink flow: moderate bounds
    "starlink_flow": [
        _make_amplitude_constraint(
            "amplitude_bound_starlink",
            18.0,
            "∀t: |A(t)| ≤ 18.0 for Starlink signal bounds",
        ),
        _make_savings_constraint(
            "savings_min_starlink",
            5.0,
            "ROI ≥ $5M for constellation telemetry",
        ),
    ],

    # xAI evaluation: moderate bounds for model metrics
    "xai_eval": [
        _make_amplitude_constraint(
            "amplitude_bound_xai",
            10.0,
            "∀t: |A(t)| ≤ 10.0 for xAI evaluation bounds",
        ),
        _make_mse_constraint(
            "mse_max_xai",
            0.01,
            "MSE ≤ 0.01 for model eval accuracy",
        ),
    ],

    # Generic fallback: conservative defaults
    "generic": [
        _make_amplitude_constraint(
            "amplitude_bound_generic",
            14.7,
            "∀t: |A(t)| ≤ 14.7 default physics bound",
        ),
        _make_ratio_constraint(
            "ratio_min_generic",
            20.0,
            "compression ratio ≥ 20:1 default efficiency gate",
        ),
        _make_savings_constraint(
            "savings_min_generic",
            1.0,
            "ROI ≥ $1M default savings threshold",
        ),
    ],
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_constraints(hook_name: str) -> List[Dict[str, Any]]:
    """
    Get constraint definitions for a given hook_name/scenario.

    This function returns constraint specs (dicts) for backward compatibility
    with qed.check_constraints(). Each dict contains:
        - id: unique identifier
        - type: constraint type (amplitude_bound, ratio_min, savings_min, mse_max)
        - bound: numeric threshold
        - description: human-readable description
        - sympy_expr: symbolic expression string

    Args:
        hook_name: The scenario or hook identifier (e.g., 'tesla_fsd', 'generic')

    Returns:
        List of constraint dictionaries
    """
    specs = _CONSTRAINTS.get(hook_name, _CONSTRAINTS.get("generic", []))
    # Return public view without internal _evaluator/_sympy_obj
    return [
        {k: v for k, v in spec.items() if not k.startswith("_")}
        for spec in specs
    ]


def get_constraint_evaluators(hook_name: str) -> List[Tuple[str, str, Callable]]:
    """
    Get pre-compiled constraint evaluators for a hook.

    Returns list of (constraint_id, constraint_type, evaluator_callable) tuples.
    Each evaluator takes a single numeric argument and returns bool.

    Evaluator signatures by type:
        - amplitude_bound: (A: float) -> bool
        - ratio_min: (ratio: float) -> bool
        - savings_min: (savings_M: float) -> bool
        - mse_max: (mse: float) -> bool

    Example:
        >>> evals = get_constraint_evaluators("tesla_fsd")
        >>> for cid, ctype, fn in evals:
        ...     if ctype == "amplitude_bound":
        ...         passed = fn(12.5)  # Check if A=12.5 passes

    Args:
        hook_name: The scenario or hook identifier

    Returns:
        List of (id, type, callable) tuples for numeric evaluation
    """
    specs = _CONSTRAINTS.get(hook_name, _CONSTRAINTS.get("generic", []))
    return [
        (spec["id"], spec["type"], spec["_evaluator"])
        for spec in specs
    ]


def evaluate_all(
    hook_name: str,
    A: Optional[float] = None,
    ratio: Optional[float] = None,
    savings_M: Optional[float] = None,
    mse: Optional[float] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Evaluate all constraints for a hook with provided metric values.

    Only evaluates constraints for which the corresponding metric is provided.
    Missing metrics are skipped (not treated as violations).

    Args:
        hook_name: The scenario or hook identifier
        A: Amplitude value (for amplitude_bound constraints)
        ratio: Compression ratio (for ratio_min constraints)
        savings_M: Savings in millions (for savings_min constraints)
        mse: Mean squared error (for mse_max constraints)

    Returns:
        (all_passed: bool, violations: List[Dict])

        Each violation dict contains:
            - constraint_id: str
            - constraint_type: str
            - value: float (actual value)
            - bound: float (threshold)
    """
    specs = _CONSTRAINTS.get(hook_name, _CONSTRAINTS.get("generic", []))
    violations: List[Dict[str, Any]] = []

    for spec in specs:
        ctype = spec["type"]
        evaluator = spec["_evaluator"]
        cid = spec["id"]
        bound_val = spec["bound"]

        # Map constraint type to provided value
        value: Optional[float] = None
        if ctype == "amplitude_bound" and A is not None:
            value = A
        elif ctype == "ratio_min" and ratio is not None:
            value = ratio
        elif ctype == "savings_min" and savings_M is not None:
            value = savings_M
        elif ctype == "mse_max" and mse is not None:
            value = mse

        # Skip if metric not provided
        if value is None:
            continue

        # Evaluate constraint
        try:
            passed = bool(evaluator(value))
        except (TypeError, ValueError):
            passed = False

        if not passed:
            violations.append({
                "constraint_id": cid,
                "constraint_type": ctype,
                "value": float(value),
                "bound": float(bound_val),
            })

    return (len(violations) == 0, violations)


def list_hooks() -> List[str]:
    """Return all registered hook names."""
    return list(_CONSTRAINTS.keys())


def sympy_available() -> bool:
    """Check if SymPy is available for symbolic operations."""
    return SYMPY_AVAILABLE
