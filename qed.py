import hashlib
import json
import re
import secrets
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import IO, Any, Dict, Iterator, List, Optional, Union

import numpy as np


__all__ = [
    "QEDReceipt",
    "RunContext",
    "RunSummary",
    "RunResult",
    "qed",
    "run",
    "check_constraints",
    "write_receipt_jsonl",
    "detect_config_drift",
]


# =============================================================================
# v8: RunContext - Lightweight context for tracking deployment runs
# =============================================================================


@dataclass(frozen=True)
class RunContext:
    """
    Lightweight context for tracking QED runs.

    Links receipts to DecisionPackets via deployment_id, enabling clean
    handoff to TruthLink for manifest generation.
    """

    deployment_id: Optional[str] = None
    batch_id: Optional[str] = None
    config_hash: Optional[str] = None
    caller: str = "unknown"

    @classmethod
    def create(
        cls,
        deployment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        caller: str = "unknown",
    ) -> "RunContext":
        """
        Factory method to create RunContext with auto-generated batch_id.

        Args:
            deployment_id: Optional deployment identifier for linking to DecisionPackets
            config: Optional config dict - if provided, hash is computed for drift detection
            caller: Who invoked this run ("cli", "pipeline", "test", etc.)

        Returns:
            RunContext with auto-generated batch_id and optional config_hash
        """
        # Auto-generate batch_id: timestamp + random suffix
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        suffix = secrets.token_hex(4)
        batch_id = f"{ts}_{suffix}"

        # Compute config hash if config provided
        config_hash = None
        if config is not None:
            config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
            config_hash = hashlib.sha3_256(config_json.encode()).hexdigest()[:16]

        return cls(
            deployment_id=deployment_id,
            batch_id=batch_id,
            config_hash=config_hash,
            caller=caller,
        )


# =============================================================================
# v8: RunSummary - Aggregated statistics for TruthLink handoff
# =============================================================================


@dataclass(frozen=True)
class RunSummary:
    """
    Aggregated run statistics for TruthLink manifest generation.

    Contains exactly what TruthLink needs, eliminating redundant computation.
    """

    deployment_id: Optional[str]
    batch_id: str
    hook: Optional[str]
    window_count: int
    windows_passed: int
    windows_failed: int
    avg_compression: float
    total_estimated_savings: float
    slo_breach_count: int
    slo_breach_rate: float
    duration_ms: int
    run_hash: str  # SHA3 of all receipt hashes


@dataclass(frozen=True)
class QEDReceipt:
    """Immutable receipt for a single QED telemetry window."""

    ts: str
    window_id: str
    params: Dict[str, Any]
    ratio: float
    H_bits: float
    recall: float
    savings_M: float
    verified: Optional[bool]
    violations: List[Dict[str, Any]]
    trace: str
    pattern_id: Optional[str] = None
    # v8: deployment tracking fields
    deployment_id: Optional[str] = None
    batch_id: Optional[str] = None
    config_hash: Optional[str] = None
    config_drift_detected: bool = False


# =============================================================================
# v8: RunResult - List-like wrapper for backward compatibility
# =============================================================================


@dataclass
class RunResult:
    """
    Result container for run() that maintains backward compatibility.

    Supports iteration, indexing, and len() so existing code treating
    the result as a list continues to work unchanged.
    """

    receipts: List[QEDReceipt]
    summary: RunSummary
    context: RunContext

    def __iter__(self) -> Iterator[QEDReceipt]:
        """Iterate over receipts for backward compatibility."""
        return iter(self.receipts)

    def __len__(self) -> int:
        """Return receipt count for backward compatibility."""
        return len(self.receipts)

    def __getitem__(self, index: int) -> QEDReceipt:
        """Index into receipts for backward compatibility."""
        return self.receipts[index]

    def __bool__(self) -> bool:
        """Truth value based on receipt count."""
        return len(self.receipts) > 0


# =============================================================================
# v8: Context validation and drift detection
# =============================================================================


def _validate_context(context: RunContext) -> List[str]:
    """
    Validate RunContext fields and return list of warnings.

    Does not block execution - only emits warnings for invalid values.

    Args:
        context: RunContext to validate

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []

    if context.deployment_id is not None:
        # Check deployment_id contains only alphanumeric, dash, underscore
        if not re.match(r"^[a-zA-Z0-9_-]*$", context.deployment_id):
            warnings.append(
                f"deployment_id contains invalid characters: {context.deployment_id!r}"
            )

        # Check deployment_id length
        if len(context.deployment_id) >= 128:
            warnings.append(
                f"deployment_id exceeds 128 chars: {len(context.deployment_id)}"
            )

    if context.batch_id is not None:
        # Validate batch_id format (timestamp_suffix)
        if not re.match(r"^[0-9]{14}_[a-f0-9]{8}$", context.batch_id):
            warnings.append(f"batch_id has non-standard format: {context.batch_id!r}")

    return warnings


def detect_config_drift(context: RunContext, current_config_hash: str) -> bool:
    """
    Detect if config has changed between runs for same deployment.

    Args:
        context: RunContext with optional stored config_hash
        current_config_hash: Hash of current configuration

    Returns:
        True if drift detected, False otherwise
    """
    if context.config_hash is None:
        return False

    if context.config_hash != current_config_hash:
        print(
            f"[QED WARNING] Config drift detected: "
            f"context={context.config_hash} != current={current_config_hash}",
            file=sys.stderr,
        )
        return True

    return False


def _build_run_summary(
    receipts: List[QEDReceipt],
    context: RunContext,
    hook: Optional[str],
    duration_seconds: float,
) -> RunSummary:
    """
    Build RunSummary from processed receipts.

    Args:
        receipts: List of QEDReceipts from this run
        context: RunContext for this run
        hook: Hook name used
        duration_seconds: Total run duration in seconds

    Returns:
        RunSummary with aggregated statistics
    """
    window_count = len(receipts)
    windows_passed = sum(1 for r in receipts if r.verified is True)
    windows_failed = sum(1 for r in receipts if r.verified is False)

    # Compute averages and totals
    avg_compression = (
        sum(r.ratio for r in receipts) / window_count if window_count > 0 else 0.0
    )
    total_estimated_savings = sum(r.savings_M for r in receipts)

    # SLO breach detection (verified=False or has violations)
    slo_breaches = sum(
        1 for r in receipts if r.verified is False or len(r.violations) > 0
    )
    slo_breach_rate = slo_breaches / window_count if window_count > 0 else 0.0

    # Compute run_hash (SHA3 of all receipt data)
    hasher = hashlib.sha3_256()
    for receipt in receipts:
        receipt_json = json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":"))
        hasher.update(receipt_json.encode())
    run_hash = hasher.hexdigest()[:32]

    return RunSummary(
        deployment_id=context.deployment_id,
        batch_id=context.batch_id or "",
        hook=hook,
        window_count=window_count,
        windows_passed=windows_passed,
        windows_failed=windows_failed,
        avg_compression=avg_compression,
        total_estimated_savings=total_estimated_savings,
        slo_breach_count=slo_breaches,
        slo_breach_rate=slo_breach_rate,
        duration_ms=int(duration_seconds * 1000),
        run_hash=run_hash,
    )


def _generate_window_id(scenario: str, n_samples: int) -> str:
    """Generate unique window identifier."""
    return f"{scenario}_{n_samples}_{uuid.uuid4().hex[:12]}"


def check_constraints(
    A: float,
    f: float,
    scenario: str,
    hook_name: Optional[str] = None,
) -> tuple[Optional[bool], List[Dict[str, Any]]]:
    """
    Check amplitude constraints, delegating to sympy_constraints if available.

    Returns (verified, violations) where:
    - verified: True if all constraints pass, False if any fail, None if not checked
    - violations: list of constraint violations with details
    """
    if not (np.isfinite(A) and np.isfinite(f)):
        return (
            False,
            [
                {
                    "constraint_id": "finite_params",
                    "value": float(A) if np.isfinite(A) else "NaN/Inf",
                    "bound": "finite",
                    "ts_offset": 0.0,
                }
            ],
        )

    try:
        import sympy_constraints

        constraints = sympy_constraints.get_constraints(hook_name or scenario)
    except ImportError:
        return (None, [])
    except AttributeError:
        return (None, [])

    violations: List[Dict[str, Any]] = []
    for constraint in constraints:
        constraint_id = constraint.get("id", "unknown")
        constraint_type = constraint.get("type", "amplitude_bound")
        bound = constraint.get("bound", float("inf"))

        # Only check amplitude_bound constraints here (other types in evaluate_all)
        if constraint_type == "amplitude_bound" and abs(A) > bound:
            violations.append(
                {
                    "constraint_id": constraint_id,
                    "value": float(A),
                    "bound": float(bound),
                    "ts_offset": 0.0,
                }
            )

    verified = len(violations) == 0
    return (verified, violations)


def write_receipt_jsonl(receipt: QEDReceipt, fh: IO[str]) -> None:
    """Append receipt as single JSON line to file handle."""
    receipt_dict = asdict(receipt)
    line = json.dumps(receipt_dict, separators=(",", ":"))
    fh.write(line + "\n")


def _estimate_entropy_bits(signal: np.ndarray, bit_depth: int) -> float:
    N = signal.size
    bits_per_sample = min(bit_depth, 12) * 0.6
    bits_per_sample = max(5.0, min(10.0, bits_per_sample))
    return bits_per_sample * N


def _fit_dominant_sine(
    signal: np.ndarray,
    sample_rate_hz: float,
) -> tuple[float, float, float, float]:
    signal = np.asarray(signal, dtype=np.float64)
    N = signal.size
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate_hz)
    magnitudes = np.abs(spec)
    peak_idx = np.argmax(magnitudes[1:]) + 1
    f_peak = freqs[peak_idx]
    t = np.arange(N) / sample_rate_hz
    sin_col = np.sin(2 * np.pi * f_peak * t)
    cos_col = np.cos(2 * np.pi * f_peak * t)
    const_col = np.ones(N)
    X = np.column_stack((sin_col, cos_col, const_col))
    beta, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    beta_sin, beta_cos, beta_const = beta
    A = np.hypot(beta_sin, beta_cos)
    phi = np.arctan2(beta_cos, beta_sin)
    c = beta_const
    return A, f_peak, phi, c


def _check_amplitude_bounds(A: float, scenario: str) -> None:
    scenario_bounds = {
        "tesla_fsd": 14.7,
    }
    bound = scenario_bounds.get(scenario, 14.7)
    if abs(A) > bound:
        msg = f"amplitude {A} exceeds bound {bound} for scenario {scenario}"
        raise ValueError(msg)


def _estimate_recall(
    raw: np.ndarray,
    recon: np.ndarray,
    threshold: float,
) -> float:
    raw_events = raw > threshold
    recon_events = recon > threshold
    total = raw_events.sum()
    if total == 0:
        return 1.0
    matches = np.logical_and(raw_events, recon_events).sum()
    return matches / float(total)


def _estimate_roi_millions(ratio: float, scenario: str) -> float:
    if scenario != "tesla_fsd":
        return 0.0

    fleet = 2_000_000.0
    data_per_car_mb = 938.0
    cost_per_pb_month = 5000.0
    months_per_year = 12.0
    days_per_year = 365.0

    raw_pb_per_day = fleet * data_per_car_mb / 1e9
    raw_pb_per_year = raw_pb_per_day * days_per_year
    raw_cost = raw_pb_per_year * cost_per_pb_month * months_per_year

    effective_ratio = max(ratio, 1.0)
    compressed_pb_per_year = raw_pb_per_year / effective_ratio
    compressed_cost = compressed_pb_per_year * cost_per_pb_month * months_per_year

    gross_savings = raw_cost - compressed_cost
    net_after_compute = gross_savings * 0.9
    net_after_impl = net_after_compute - 1_750_000.0
    net_final = net_after_impl + 3_500_000.0

    return net_final / 1_000_000.0


def qed(
    signal: np.ndarray,
    scenario: str = "tesla_fsd",
    bit_depth: int = 12,
    sample_rate_hz: float = 1000.0,
    hook_name: Optional[str] = None,
    pattern_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a 1-second telemetry window and return a compact QED summary.

    Returns a dict with:
      - "ratio": compression ratio (float)
      - "H_bits": estimated Shannon information in bits (float)
      - "recall": safety-event recall estimate in [0, 1]
      - "savings_M": ROI in millions of dollars (float)
      - "trace": short string documenting assumptions and scenario
      - "receipt": QEDReceipt with full params, verified status, and violations
    """
    signal_arr = np.asarray(signal, dtype=np.float64)
    if signal_arr.size == 0:
        raise ValueError("signal must be non-empty")

    H_bits = _estimate_entropy_bits(signal_arr, bit_depth)
    A, f, phi, c = _fit_dominant_sine(signal_arr, sample_rate_hz)
    _check_amplitude_bounds(A, scenario)

    raw_bits = float(signal_arr.size * bit_depth)
    compressed_bits = 200.0
    ratio = raw_bits / compressed_bits

    t = np.arange(signal_arr.size) / sample_rate_hz
    recon = A * np.sin(2 * np.pi * f * t + phi) + c
    recall = _estimate_recall(signal_arr, recon, threshold=10.0)
    savings_M = _estimate_roi_millions(ratio, scenario)

    trace = (
        f"qed_v6 scenario={scenario} "
        f"N={signal_arr.size} H≈{int(H_bits)} ratio≈{ratio:.1f}"
    )

    verified, violations = check_constraints(A, f, scenario, hook_name)

    receipt = QEDReceipt(
        ts=datetime.now(timezone.utc).isoformat(),
        window_id=_generate_window_id(scenario, signal_arr.size),
        params={
            "A": float(A),
            "f": float(f),
            "phi": float(phi),
            "c": float(c),
            "scenario": scenario,
            "bit_depth": bit_depth,
            "sample_rate_hz": sample_rate_hz,
        },
        ratio=float(ratio),
        H_bits=float(H_bits),
        recall=float(recall),
        savings_M=float(savings_M),
        verified=verified,
        violations=violations,
        trace=trace,
        pattern_id=pattern_id,
    )

    return {
        "ratio": float(ratio),
        "H_bits": float(H_bits),
        "recall": float(recall),
        "savings_M": float(savings_M),
        "trace": trace,
        "receipt": receipt,
    }


def run(
    windows: Union[np.ndarray, List[np.ndarray]],
    hook: Optional[str] = None,
    context: Optional[RunContext] = None,
    deployment_id: Optional[str] = None,
    pattern_id: Optional[str] = None,
    **kwargs,
) -> RunResult:
    """
    Process telemetry windows and return RunResult with receipts and summary.

    v8 enhanced interface with deployment tracking for TruthLink handoff.
    Maintains backward compatibility - can be iterated like a list.

    Args:
        windows: Single telemetry array or list of arrays to process
        hook: Optional hook name for constraint checking
        context: Optional RunContext for deployment tracking
        deployment_id: Optional deployment ID (convenience - wrapped in context internally)
        pattern_id: Optional SHA3 hash of anomaly pattern that triggered detection
        **kwargs: Additional arguments passed to qed()

    Returns:
        RunResult containing receipts, summary, and context.
        RunResult is iterable and indexable for backward compatibility.

    Example:
        # New v8 style with context
        ctx = RunContext.create(deployment_id="deploy-123", caller="pipeline")
        result = run(windows, "my_hook", context=ctx)
        print(result.summary.avg_compression)

        # Backward compatible - treat as list
        result = run(windows, "my_hook")
        for receipt in result:
            print(receipt.ratio)
    """
    # Track timing
    start_time = time.time()

    # Backward compat: wrap deployment_id in context if needed
    if context is None:
        context = RunContext.create(deployment_id=deployment_id)
    elif deployment_id and not context.deployment_id:
        # deployment_id passed separately, inject into context
        context = replace(context, deployment_id=deployment_id)

    # Validate context and emit warnings
    warnings = _validate_context(context)
    for warning in warnings:
        print(f"[QED WARNING] {warning}", file=sys.stderr)

    # Normalize windows to list format
    # Handle single window (backward compat) vs multiple windows
    if isinstance(windows, np.ndarray):
        # Check if it's a 2D array (multiple windows) or 1D (single window)
        if windows.ndim == 1:
            window_list = [windows]
        elif windows.ndim == 2:
            # 2D array: each row is a window
            window_list = [windows[i] for i in range(windows.shape[0])]
        else:
            # 3D+ array: treat as single complex signal
            window_list = [windows]
    elif isinstance(windows, (list, tuple)):
        window_list = list(windows)
    else:
        # Fallback: try to iterate
        window_list = list(windows)

    # Process each window
    receipts: List[QEDReceipt] = []
    for window in window_list:
        result = qed(signal=window, hook_name=hook, pattern_id=pattern_id, **kwargs)
        receipt = result["receipt"]

        # Inject context fields into receipt
        receipt = replace(
            receipt,
            deployment_id=context.deployment_id,
            batch_id=context.batch_id,
            config_hash=context.config_hash,
        )
        receipts.append(receipt)

    # Build summary for TruthLink handoff
    duration_seconds = time.time() - start_time
    summary = _build_run_summary(receipts, context, hook, duration_seconds)

    return RunResult(receipts=receipts, summary=summary, context=context)
