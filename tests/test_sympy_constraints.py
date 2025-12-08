"""
Tests for sympy_constraints module.

Tests cover:
- Registry correctness per hook
- Lambdify evaluator functionality
- evaluate_all() for multiple constraint types
- Performance: <1ms evaluation
- Backward compatibility with get_constraints()
"""

import time
import sympy_constraints


class TestGetConstraints:
    """Test get_constraints() backward compatibility."""

    def test_tesla_returns_list(self):
        """Tesla hook returns list of constraint dicts."""
        cons = sympy_constraints.get_constraints("tesla_fsd")
        assert isinstance(cons, list)
        assert len(cons) >= 1

    def test_constraint_has_required_keys(self):
        """Each constraint dict has id, type, bound, description, sympy_expr."""
        cons = sympy_constraints.get_constraints("tesla_fsd")
        for c in cons:
            assert "id" in c
            assert "type" in c
            assert "bound" in c
            assert "description" in c
            assert "sympy_expr" in c
            # Internal keys should NOT be exposed
            assert "_evaluator" not in c
            assert "_sympy_obj" not in c

    def test_unknown_hook_returns_generic(self):
        """Unknown hook_name falls back to generic constraints."""
        cons = sympy_constraints.get_constraints("nonexistent_hook")
        generic = sympy_constraints.get_constraints("generic")
        assert len(cons) == len(generic)
        assert cons[0]["id"] == "amplitude_bound_generic"

    def test_all_hooks_return_valid_constraints(self):
        """All registered hooks return valid constraint lists."""
        for hook in sympy_constraints.list_hooks():
            cons = sympy_constraints.get_constraints(hook)
            assert isinstance(cons, list)
            assert len(cons) >= 1


class TestGetConstraintEvaluators:
    """Test get_constraint_evaluators() lambdified callables."""

    def test_returns_list_of_tuples(self):
        """Returns list of (id, type, callable) tuples."""
        evals = sympy_constraints.get_constraint_evaluators("tesla_fsd")
        assert isinstance(evals, list)
        for item in evals:
            assert len(item) == 3
            cid, ctype, fn = item
            assert isinstance(cid, str)
            assert isinstance(ctype, str)
            assert callable(fn)

    def test_amplitude_evaluator_tesla(self):
        """Amplitude evaluator correctly checks bounds."""
        evals = sympy_constraints.get_constraint_evaluators("tesla_fsd")
        amp_eval = None
        for cid, ctype, fn in evals:
            if ctype == "amplitude_bound":
                amp_eval = fn
                break
        assert amp_eval is not None
        # Tesla bound is 14.7
        assert amp_eval(10.0)       # Below bound
        assert amp_eval(14.7)       # At bound
        assert not amp_eval(15.0)   # Above bound

    def test_ratio_evaluator_generic(self):
        """Ratio evaluator correctly checks minimum."""
        evals = sympy_constraints.get_constraint_evaluators("generic")
        ratio_eval = None
        for cid, ctype, fn in evals:
            if ctype == "ratio_min":
                ratio_eval = fn
                break
        assert ratio_eval is not None
        # Generic ratio minimum is 20.0
        assert ratio_eval(25.0)       # Above min
        assert ratio_eval(20.0)       # At min
        assert not ratio_eval(15.0)   # Below min

    def test_savings_evaluator_spacex(self):
        """Savings evaluator correctly checks ROI threshold."""
        evals = sympy_constraints.get_constraint_evaluators("spacex_flight")
        savings_eval = None
        for cid, ctype, fn in evals:
            if ctype == "savings_min":
                savings_eval = fn
                break
        assert savings_eval is not None
        # SpaceX savings minimum is 10.0 (millions)
        assert savings_eval(15.0)       # Above min
        assert savings_eval(10.0)       # At min
        assert not savings_eval(5.0)    # Below min

    def test_mse_evaluator_neuralink(self):
        """MSE evaluator correctly checks maximum."""
        evals = sympy_constraints.get_constraint_evaluators("neuralink_stream")
        mse_eval = None
        for cid, ctype, fn in evals:
            if ctype == "mse_max":
                mse_eval = fn
                break
        assert mse_eval is not None
        # Neuralink MSE max is 0.001
        assert mse_eval(0.0005)       # Below max
        assert mse_eval(0.001)        # At max
        assert not mse_eval(0.002)    # Above max


class TestEvaluateAll:
    """Test evaluate_all() multi-constraint evaluation."""

    def test_all_pass_when_within_bounds(self):
        """All constraints pass when values are within bounds."""
        passed, violations = sympy_constraints.evaluate_all(
            "tesla_fsd",
            A=10.0,        # < 14.7
            ratio=60.0,    # >= 20
            savings_M=38.0 # >= 1M
        )
        assert passed is True
        assert violations == []

    def test_amplitude_violation_detected(self):
        """Amplitude violation is detected and reported."""
        passed, violations = sympy_constraints.evaluate_all(
            "tesla_fsd",
            A=15.0,        # > 14.7 VIOLATION
            ratio=60.0,
            savings_M=38.0
        )
        assert passed is False
        assert len(violations) == 1
        assert violations[0]["constraint_id"] == "amplitude_bound_tesla"
        assert violations[0]["constraint_type"] == "amplitude_bound"
        assert violations[0]["value"] == 15.0
        assert violations[0]["bound"] == 14.7

    def test_ratio_violation_detected(self):
        """Ratio violation is detected and reported."""
        passed, violations = sympy_constraints.evaluate_all(
            "generic",
            A=10.0,
            ratio=10.0,     # < 20 VIOLATION
            savings_M=5.0
        )
        assert passed is False
        # Should have ratio violation
        ratio_viol = [v for v in violations if v["constraint_type"] == "ratio_min"]
        assert len(ratio_viol) == 1
        assert ratio_viol[0]["value"] == 10.0
        assert ratio_viol[0]["bound"] == 20.0

    def test_savings_violation_detected(self):
        """Savings violation is detected and reported."""
        passed, violations = sympy_constraints.evaluate_all(
            "spacex_flight",
            A=15.0,
            savings_M=5.0   # < 10M VIOLATION
        )
        assert passed is False
        sav_viol = [v for v in violations if v["constraint_type"] == "savings_min"]
        assert len(sav_viol) == 1
        assert sav_viol[0]["value"] == 5.0
        assert sav_viol[0]["bound"] == 10.0

    def test_mse_violation_detected(self):
        """MSE violation is detected and reported."""
        passed, violations = sympy_constraints.evaluate_all(
            "neuralink_stream",
            A=3.0,
            mse=0.01        # > 0.001 VIOLATION
        )
        assert passed is False
        mse_viol = [v for v in violations if v["constraint_type"] == "mse_max"]
        assert len(mse_viol) == 1
        assert mse_viol[0]["value"] == 0.01
        assert mse_viol[0]["bound"] == 0.001

    def test_missing_metrics_skipped_not_violations(self):
        """Metrics not provided are skipped, not treated as violations."""
        # Only provide amplitude, skip ratio and savings
        passed, violations = sympy_constraints.evaluate_all(
            "tesla_fsd",
            A=10.0
        )
        assert passed is True
        assert violations == []

    def test_multiple_violations(self):
        """Multiple violations are detected together."""
        passed, violations = sympy_constraints.evaluate_all(
            "generic",
            A=20.0,         # > 14.7 VIOLATION
            ratio=10.0,     # < 20 VIOLATION
            savings_M=0.5   # < 1M VIOLATION
        )
        assert passed is False
        assert len(violations) == 3


class TestListHooks:
    """Test list_hooks() registry enumeration."""

    def test_returns_list_of_strings(self):
        """list_hooks() returns list of hook name strings."""
        hooks = sympy_constraints.list_hooks()
        assert isinstance(hooks, list)
        for h in hooks:
            assert isinstance(h, str)

    def test_includes_known_hooks(self):
        """Known hooks are present in the list."""
        hooks = sympy_constraints.list_hooks()
        assert "tesla_fsd" in hooks
        assert "spacex_flight" in hooks
        assert "neuralink_stream" in hooks
        assert "boring_tunnel" in hooks
        assert "starlink_flow" in hooks
        assert "xai_eval" in hooks
        assert "generic" in hooks

    def test_at_least_7_hooks(self):
        """At least 7 hooks are registered (6 scenarios + generic)."""
        hooks = sympy_constraints.list_hooks()
        assert len(hooks) >= 7


class TestSympyAvailable:
    """Test sympy_available() check."""

    def test_returns_bool(self):
        """sympy_available() returns boolean."""
        result = sympy_constraints.sympy_available()
        assert isinstance(result, bool)

    def test_sympy_available_when_imported(self):
        """SymPy should be available in test environment."""
        # Since we installed sympy for tests, this should be True
        assert sympy_constraints.sympy_available() is True


class TestPerformance:
    """Test evaluation performance requirements."""

    def test_evaluator_under_1ms(self):
        """Single evaluator call completes in <1ms."""
        evals = sympy_constraints.get_constraint_evaluators("tesla_fsd")
        _, _, fn = evals[0]

        start = time.perf_counter()
        for _ in range(1000):
            fn(12.5)
        elapsed = time.perf_counter() - start

        # 1000 calls should complete in <100ms (0.1ms per call average)
        assert elapsed < 0.1, f"1000 evals took {elapsed*1000:.2f}ms"

    def test_evaluate_all_under_1ms(self):
        """evaluate_all() completes in <1ms per call."""
        start = time.perf_counter()
        for _ in range(100):
            sympy_constraints.evaluate_all(
                "tesla_fsd",
                A=10.0,
                ratio=60.0,
                savings_M=38.0
            )
        elapsed = time.perf_counter() - start

        # 100 calls should complete in <100ms (1ms per call average)
        assert elapsed < 0.1, f"100 evaluate_all calls took {elapsed*1000:.2f}ms"

    def test_get_constraints_under_1ms(self):
        """get_constraints() completes in <1ms."""
        start = time.perf_counter()
        for _ in range(100):
            sympy_constraints.get_constraints("tesla_fsd")
        elapsed = time.perf_counter() - start

        # 100 calls should complete in <50ms (<0.5ms per call)
        assert elapsed < 0.05, f"100 get_constraints calls took {elapsed*1000:.2f}ms"


class TestConstraintScenarios:
    """Test specific scenario constraint values."""

    def test_tesla_amplitude_bound(self):
        """Tesla amplitude bound is 14.7."""
        cons = sympy_constraints.get_constraints("tesla_fsd")
        amp_cons = [c for c in cons if c["type"] == "amplitude_bound"][0]
        assert amp_cons["bound"] == 14.7

    def test_spacex_amplitude_bound(self):
        """SpaceX amplitude bound is 20.0."""
        cons = sympy_constraints.get_constraints("spacex_flight")
        amp_cons = [c for c in cons if c["type"] == "amplitude_bound"][0]
        assert amp_cons["bound"] == 20.0

    def test_neuralink_amplitude_bound(self):
        """Neuralink amplitude bound is 5.0 (tight for neural signals)."""
        cons = sympy_constraints.get_constraints("neuralink_stream")
        amp_cons = [c for c in cons if c["type"] == "amplitude_bound"][0]
        assert amp_cons["bound"] == 5.0

    def test_boring_amplitude_bound(self):
        """Boring amplitude bound is 25.0 (high tolerance for tunnel vibration)."""
        cons = sympy_constraints.get_constraints("boring_tunnel")
        amp_cons = [c for c in cons if c["type"] == "amplitude_bound"][0]
        assert amp_cons["bound"] == 25.0

    def test_spacex_savings_threshold(self):
        """SpaceX requires higher ROI threshold ($10M)."""
        cons = sympy_constraints.get_constraints("spacex_flight")
        sav_cons = [c for c in cons if c["type"] == "savings_min"][0]
        assert sav_cons["bound"] == 10.0

    def test_generic_savings_threshold(self):
        """Generic savings threshold is $1M."""
        cons = sympy_constraints.get_constraints("generic")
        sav_cons = [c for c in cons if c["type"] == "savings_min"][0]
        assert sav_cons["bound"] == 1.0
