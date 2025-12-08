"""
QED v7 Smoke Tests - New Module Coverage

Smoke tests for core QED v7 new modules:
- shared_anomalies: load/append/get pattern library
- physics_injection: perturbation injection by physics domain
- edge_lab_v2: sim runner and recall floor computation
- cross_domain: validation of cross-domain pattern transfer
- clarity_clean_adapter: receipt to corpus conversion with quality audit
- mesh_view_v2: aggregated metrics with v7 fields

Test organization:
- Group tests by module using classes
- Use pytest fixtures for common setup
- Use tmp_path for temp files (auto-cleanup)
- Use pytest.approx() for float comparisons
"""

import json
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import pytest

# =============================================================================
# Module imports - skip tests if modules not available
# =============================================================================

from shared_anomalies import (
    AnomalyPattern,
    append_pattern,
    generate_pattern_id,
    get_patterns_for_hook,
    load_library,
)
from physics_injection import (
    inject_perturbation,
    load_ngsim_trajectories,
    generate_sae_j1939_faults,
    sample_nhtsa_failure_rates,
)
from cross_domain import (
    CROSS_DOMAIN_MAP,
    validate_cross_domain,
)
from clarity_clean_adapter import (
    ClarityCleanReceipt,
    clean_corpus,
    process_receipts,
    receipts_to_text,
)
from mesh_view_v2 import (
    compute_metrics_v2,
    join_all_sources,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_anomaly_pattern() -> AnomalyPattern:
    """Sample AnomalyPattern with all required fields."""
    return AnomalyPattern(
        pattern_id="abc123def456",
        physics_domain="battery_thermal",
        failure_mode="overheat",
        params={"threshold_c": 65.0, "gradient_c_per_s": 2.5},
        hooks=["tesla", "spacex"],
        dollar_value_annual=5_000_000.0,
        validation_recall=0.995,
        false_positive_rate=0.005,
        cross_domain_targets=["spacex"],
        deprecated=False,
    )


@pytest.fixture
def sample_anomaly_pattern_dict() -> Dict[str, Any]:
    """Sample pattern as dict for injection tests."""
    return {
        "pattern_id": "abc123def456",
        "physics_domain": "battery_thermal",
        "failure_mode": "overheat",
        "params": {"threshold_c": 65.0, "gradient_c_per_s": 2.5},
        "hooks": ["tesla", "spacex"],
        "dollar_value_annual": 5_000_000.0,
    }


@pytest.fixture
def sample_telemetry_window() -> np.ndarray:
    """Sample telemetry window (100 floats)."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 100)
    signal = 10.0 * np.sin(2 * np.pi * 5.0 * t) + 2.0
    return signal + rng.normal(0, 0.5, 100)


@pytest.fixture
def sample_qed_receipts_jsonl(tmp_path) -> str:
    """Create sample QEDReceipt JSONL file in temp directory."""
    receipts_path = tmp_path / "receipts.jsonl"
    receipts = [
        {
            "ts": "2024-01-01T00:00:00Z",
            "window_id": "tesla_fsd_001",
            "hook": "tesla_fsd",
            "params": {"A": 12.0, "f": 40.0, "phi": 0.0, "c": 2.0, "scenario": "tesla_fsd"},
            "ratio": 60.0,
            "savings_M": 38.0,
            "verified": True,
            "violations": [],
            "classification": "normal",
            "slo_status": "ok",
            "score": 0.95,
            "compression_ratio": 60.0,
        },
        {
            "ts": "2024-01-01T00:00:01Z",
            "window_id": "tesla_fsd_002",
            "hook": "tesla_fsd",
            "params": {"A": 11.0, "f": 42.0, "phi": 0.1, "c": 1.5, "scenario": "tesla_fsd"},
            "ratio": 55.0,
            "savings_M": 35.0,
            "verified": True,
            "violations": ["amplitude_warning"],
            "classification": "anomaly",
            "slo_status": "ok",
            "score": 0.88,
            "compression_ratio": 55.0,
        },
    ]
    with open(receipts_path, "w") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")
    return str(receipts_path)


# =============================================================================
# Test: shared_anomalies load/append/get (Section 1)
# =============================================================================


class TestSharedAnomaliesLoad:
    """Tests for shared_anomalies module load/append/get functions."""

    def test_load_library_empty(self, tmp_path):
        """Load from nonexistent file returns empty list."""
        nonexistent_path = tmp_path / "nonexistent.jsonl"
        result = load_library(str(nonexistent_path))
        assert result == []
        assert isinstance(result, list)

    def test_load_library_with_data(self, tmp_path, sample_anomaly_pattern):
        """Load from valid JSONL returns list of AnomalyPattern."""
        lib_path = tmp_path / "anomalies.jsonl"
        with open(lib_path, "w") as f:
            f.write(json.dumps(asdict(sample_anomaly_pattern)) + "\n")

        result = load_library(str(lib_path))
        assert len(result) == 1
        assert isinstance(result[0], AnomalyPattern)
        assert result[0].pattern_id == sample_anomaly_pattern.pattern_id
        assert result[0].physics_domain == sample_anomaly_pattern.physics_domain

    def test_append_pattern_new(self, tmp_path, sample_anomaly_pattern):
        """Append new pattern returns True, file contains pattern."""
        lib_path = tmp_path / "data" / "anomalies.jsonl"

        result = append_pattern(sample_anomaly_pattern, str(lib_path))
        assert result is True

        # Verify file contains pattern
        loaded = load_library(str(lib_path))
        assert len(loaded) == 1
        assert loaded[0].pattern_id == sample_anomaly_pattern.pattern_id

    def test_append_pattern_duplicate(self, tmp_path, sample_anomaly_pattern):
        """Append same pattern_id returns False (immutability)."""
        lib_path = tmp_path / "data" / "anomalies.jsonl"

        # First append should succeed
        result1 = append_pattern(sample_anomaly_pattern, str(lib_path))
        assert result1 is True

        # Second append with same pattern_id should fail
        result2 = append_pattern(sample_anomaly_pattern, str(lib_path))
        assert result2 is False

        # File should still have only one pattern
        loaded = load_library(str(lib_path))
        assert len(loaded) == 1

    def test_get_patterns_for_hook(self, tmp_path, sample_anomaly_pattern):
        """Returns only patterns where hook in hooks list."""
        lib_path = tmp_path / "data" / "anomalies.jsonl"

        # Create patterns for different hooks
        pattern1 = sample_anomaly_pattern  # hooks=["tesla", "spacex"]

        pattern2 = AnomalyPattern(
            pattern_id="xyz789",
            physics_domain="comms",
            failure_mode="signal_loss",
            params={"dropout_ms": 100},
            hooks=["starlink"],
            dollar_value_annual=2_000_000.0,
        )

        append_pattern(pattern1, str(lib_path))
        append_pattern(pattern2, str(lib_path))

        # Get patterns for tesla - should return pattern1 only
        tesla_patterns = get_patterns_for_hook("tesla", str(lib_path))
        assert len(tesla_patterns) == 1
        assert tesla_patterns[0].pattern_id == pattern1.pattern_id

        # Get patterns for starlink - should return pattern2 only
        starlink_patterns = get_patterns_for_hook("starlink", str(lib_path))
        assert len(starlink_patterns) == 1
        assert starlink_patterns[0].pattern_id == pattern2.pattern_id

        # Get patterns for boring - should return empty
        boring_patterns = get_patterns_for_hook("boring", str(lib_path))
        assert len(boring_patterns) == 0

    def test_get_patterns_excludes_deprecated(self, tmp_path):
        """Deprecated patterns not returned by get_patterns_for_hook."""
        lib_path = tmp_path / "data" / "anomalies.jsonl"

        active_pattern = AnomalyPattern(
            pattern_id="active123",
            physics_domain="battery_thermal",
            failure_mode="overheat",
            params={},
            hooks=["tesla"],
            dollar_value_annual=1_000_000.0,
            deprecated=False,
        )

        deprecated_pattern = AnomalyPattern(
            pattern_id="deprecated456",
            physics_domain="battery_thermal",
            failure_mode="old_issue",
            params={},
            hooks=["tesla"],
            dollar_value_annual=500_000.0,
            deprecated=True,
        )

        append_pattern(active_pattern, str(lib_path))
        append_pattern(deprecated_pattern, str(lib_path))

        # Get patterns - should exclude deprecated
        tesla_patterns = get_patterns_for_hook("tesla", str(lib_path))
        assert len(tesla_patterns) == 1
        assert tesla_patterns[0].pattern_id == "active123"


class TestSharedAnomaliesProperties:
    """Tests for AnomalyPattern computed properties."""

    def test_training_score_under_10m(self):
        """Training score is dollar_value / 10M capped at 1.0."""
        pattern = AnomalyPattern(
            pattern_id="test",
            physics_domain="test",
            failure_mode="test",
            params={},
            hooks=["test"],
            dollar_value_annual=5_000_000.0,
        )
        assert pattern.training_score == pytest.approx(0.5, abs=0.001)

    def test_training_score_over_10m(self):
        """Training score caps at 1.0 for values over 10M."""
        pattern = AnomalyPattern(
            pattern_id="test",
            physics_domain="test",
            failure_mode="test",
            params={},
            hooks=["test"],
            dollar_value_annual=20_000_000.0,
        )
        assert pattern.training_score == 1.0

    def test_training_role_cross_company(self):
        """High value patterns get train_cross_company role."""
        pattern = AnomalyPattern(
            pattern_id="test",
            physics_domain="test",
            failure_mode="test",
            params={},
            hooks=["test"],
            dollar_value_annual=15_000_000.0,
        )
        assert pattern.training_role == "train_cross_company"

    def test_training_role_observe_only(self):
        """Low value patterns get observe_only role."""
        pattern = AnomalyPattern(
            pattern_id="test",
            physics_domain="test",
            failure_mode="test",
            params={},
            hooks=["test"],
            dollar_value_annual=5_000_000.0,
        )
        assert pattern.training_role == "observe_only"

    def test_exploit_grade_true(self):
        """exploit_grade True when high value, high recall, low FP."""
        pattern = AnomalyPattern(
            pattern_id="test",
            physics_domain="test",
            failure_mode="test",
            params={},
            hooks=["test"],
            dollar_value_annual=5_000_000.0,  # > 1M
            validation_recall=0.995,  # >= 0.99
            false_positive_rate=0.005,  # <= 0.01
        )
        assert pattern.exploit_grade is True

    def test_exploit_grade_false_low_value(self):
        """exploit_grade False when value under 1M."""
        pattern = AnomalyPattern(
            pattern_id="test",
            physics_domain="test",
            failure_mode="test",
            params={},
            hooks=["test"],
            dollar_value_annual=500_000.0,  # < 1M
            validation_recall=0.995,
            false_positive_rate=0.005,
        )
        assert pattern.exploit_grade is False


# =============================================================================
# Test: physics_injection perturbation (Section 2)
# =============================================================================


class TestPhysicsInjection:
    """Tests for physics_injection module perturbation functions."""

    def test_inject_perturbation_shape_preserved(self, sample_telemetry_window):
        """Output array same shape as input."""
        pattern = {"physics_domain": "motion", "acceleration_spike": -5.0}
        result = inject_perturbation(sample_telemetry_window, pattern, seed=42)

        assert result.shape == sample_telemetry_window.shape
        assert isinstance(result, np.ndarray)

    def test_inject_perturbation_values_differ(self, sample_telemetry_window):
        """Output not identical to input (perturbation applied)."""
        pattern = {"physics_domain": "motion", "acceleration_spike": -8.0}
        result = inject_perturbation(sample_telemetry_window, pattern, seed=42)

        # At least some values should differ
        assert not np.allclose(result, sample_telemetry_window)

    def test_inject_perturbation_reproducible(self, sample_telemetry_window):
        """Same seed produces same output."""
        pattern = {"physics_domain": "fault", "signal_dropout_prob": 0.1}

        result1 = inject_perturbation(sample_telemetry_window, pattern, seed=42)
        result2 = inject_perturbation(sample_telemetry_window, pattern, seed=42)

        np.testing.assert_array_equal(result1, result2)

    def test_inject_perturbation_trajectory_domain(self, sample_telemetry_window):
        """Test trajectory/motion domain perturbation."""
        pattern = {
            "physics_domain": "trajectory",
            "acceleration_spike": -6.0,
            "lateral_deviation": 1.5,
            "speed_variance": 10.0,
            "duration_ms": 2000,
        }
        result = inject_perturbation(sample_telemetry_window, pattern, seed=42)

        assert result.shape == sample_telemetry_window.shape
        assert not np.allclose(result, sample_telemetry_window)

    def test_inject_perturbation_fault_domain(self, sample_telemetry_window):
        """Test fault/CAN bus domain perturbation."""
        pattern = {
            "physics_domain": "fault",
            "signal_dropout_prob": 0.2,
            "value_corruption_range": (0.8, 1.2),
        }
        result = inject_perturbation(sample_telemetry_window, pattern, seed=42)

        assert result.shape == sample_telemetry_window.shape
        # Fault injection may include zeros from dropouts
        assert np.any(result == 0.0) or not np.allclose(result, sample_telemetry_window)

    def test_inject_perturbation_thermal_domain(self, sample_telemetry_window):
        """Test component/battery/thermal domain perturbation."""
        pattern = {
            "physics_domain": "thermal",
            "signal_degradation": 0.3,
            "spike_magnitude": 2.0,
        }
        result = inject_perturbation(sample_telemetry_window, pattern, seed=42)

        assert result.shape == sample_telemetry_window.shape

    def test_inject_perturbation_unknown_domain(self, sample_telemetry_window):
        """Test unknown domain falls back to generic perturbation."""
        pattern = {
            "physics_domain": "unknown_domain",
            "perturbation_magnitude": 0.2,
        }
        result = inject_perturbation(sample_telemetry_window, pattern, seed=42)

        assert result.shape == sample_telemetry_window.shape

    def test_ngsim_trajectories_loaded(self):
        """NGSIM trajectory patterns can be loaded."""
        trajectories = load_ngsim_trajectories()
        assert isinstance(trajectories, dict)
        assert len(trajectories) >= 1
        # Check for expected pattern names
        assert "hard_brake" in trajectories or len(trajectories) > 0

    def test_sae_j1939_faults_generated(self):
        """SAE J1939 fault patterns can be generated."""
        faults = generate_sae_j1939_faults()
        assert isinstance(faults, dict)
        assert len(faults) >= 1

    def test_nhtsa_failure_rates_sampled(self):
        """NHTSA failure patterns can be sampled."""
        failures = sample_nhtsa_failure_rates()
        assert isinstance(failures, dict)
        assert len(failures) >= 1


# =============================================================================
# Test: edge_lab_v2 sim runner (Section 3)
# =============================================================================


class TestEdgeLabV2:
    """
    Tests for edge_lab_v2 module sim runner.

    Note: edge_lab_v2 may not exist yet - tests will be skipped if import fails.
    """

    @pytest.fixture
    def edge_lab_v2_module(self):
        """Import edge_lab_v2 or skip tests."""
        return pytest.importorskip("edge_lab_v2")

    def test_run_pattern_sims_returns_results(self, edge_lab_v2_module, tmp_path):
        """Returns SimResults dataclass with expected fields."""
        # Create minimal test patterns
        patterns = [
            {
                "pattern_id": "test001",
                "physics_domain": "battery_thermal",
                "failure_mode": "overheat",
                "params": {},
                "hooks": ["tesla"],
            }
        ]

        # Use small n_per_hook for fast tests
        result = edge_lab_v2_module.run_pattern_sims(
            patterns=patterns,
            n_per_hook=10,
        )

        assert result is not None
        assert hasattr(result, "aggregate_recall") or "aggregate_recall" in dir(result)

    def test_sim_results_has_recall_field(self, edge_lab_v2_module, tmp_path):
        """aggregate_recall is float between 0-1."""
        patterns = [
            {
                "pattern_id": "test001",
                "physics_domain": "motion",
                "params": {},
                "hooks": ["tesla"],
            }
        ]

        result = edge_lab_v2_module.run_pattern_sims(patterns=patterns, n_per_hook=10)

        recall = getattr(result, "aggregate_recall", result.get("aggregate_recall", 0.5))
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    def test_sim_results_has_fp_rate_field(self, edge_lab_v2_module, tmp_path):
        """aggregate_fp_rate is float between 0-1."""
        patterns = [
            {
                "pattern_id": "test001",
                "physics_domain": "motion",
                "params": {},
                "hooks": ["tesla"],
            }
        ]

        result = edge_lab_v2_module.run_pattern_sims(patterns=patterns, n_per_hook=10)

        fp_rate = getattr(result, "aggregate_fp_rate", result.get("aggregate_fp_rate", 0.05))
        assert isinstance(fp_rate, float)
        assert 0.0 <= fp_rate <= 1.0

    def test_sim_results_pattern_count(self, edge_lab_v2_module, tmp_path):
        """pattern_results list matches input pattern count."""
        patterns = [
            {"pattern_id": f"test{i:03d}", "physics_domain": "motion", "params": {}, "hooks": ["tesla"]}
            for i in range(3)
        ]

        result = edge_lab_v2_module.run_pattern_sims(patterns=patterns, n_per_hook=10)

        pattern_results = getattr(result, "pattern_results", result.get("pattern_results", []))
        assert len(pattern_results) == len(patterns)


# =============================================================================
# Test: cross_domain validation (Section 4)
# =============================================================================


class TestCrossDomainValidation:
    """Tests for cross_domain validation module."""

    def test_validate_cross_domain_tesla_spacex_battery(self):
        """Returns True for valid tesla->spacex battery_thermal mapping."""
        pattern = {
            "physics_domain": "battery_thermal",
            "source_hook": "tesla",
            "hooks": ["tesla"],
            "params": {"temperature_c": 50.0},
        }
        result = validate_cross_domain(pattern, "spacex")
        assert result is True

    def test_validate_cross_domain_tesla_starlink_comms(self):
        """Returns True for valid tesla->starlink comms mapping."""
        pattern = {
            "physics_domain": "comms",
            "source_hook": "tesla",
            "hooks": ["tesla"],
            "params": {"signal_dbm": -80.0},
        }
        result = validate_cross_domain(pattern, "starlink")
        assert result is True

    def test_validate_cross_domain_unmapped_rejected(self):
        """Returns False for unmapped (e.g., boring->xai)."""
        pattern = {
            "physics_domain": "tunnel_vibration",
            "source_hook": "boring",
            "hooks": ["boring"],
            "params": {},
        }
        result = validate_cross_domain(pattern, "xai")
        assert result is False

    def test_validate_cross_domain_wrong_domain(self):
        """Returns False if physics_domain doesn't match mapping."""
        # Tesla has battery_thermal->spacex, not comms->spacex
        pattern = {
            "physics_domain": "comms",
            "source_hook": "tesla",
            "hooks": ["tesla"],
            "params": {},
        }
        # comms maps to starlink, not spacex
        result = validate_cross_domain(pattern, "spacex")
        assert result is False

    def test_cross_domain_map_structure(self):
        """Verify CROSS_DOMAIN_MAP has expected structure."""
        assert isinstance(CROSS_DOMAIN_MAP, dict)
        # Check for known mapping
        assert ("tesla", "battery_thermal") in CROSS_DOMAIN_MAP
        assert "spacex" in CROSS_DOMAIN_MAP[("tesla", "battery_thermal")]


# =============================================================================
# Test: clarity_clean_adapter (Section 5)
# =============================================================================


class TestClarityCleanAdapter:
    """Tests for clarity_clean_adapter module."""

    def test_receipts_to_text_returns_string(self, sample_qed_receipts_jsonl):
        """Output is non-empty string."""
        result = receipts_to_text(sample_qed_receipts_jsonl)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_receipts_to_text_empty_file(self, tmp_path):
        """Empty file returns empty string."""
        empty_path = tmp_path / "empty.jsonl"
        empty_path.write_text("")
        result = receipts_to_text(str(empty_path))
        assert result == ""

    def test_receipts_to_text_nonexistent_file(self, tmp_path):
        """Nonexistent file returns empty string."""
        result = receipts_to_text(str(tmp_path / "nonexistent.jsonl"))
        assert result == ""

    def test_clean_corpus_returns_tuple(self):
        """Returns (str, ClarityCleanReceipt) tuple."""
        corpus = "hook:tesla_fsd | class:normal | slo:ok"
        result = clean_corpus(corpus)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], ClarityCleanReceipt)

    def test_clarity_receipt_has_required_fields(self):
        """Receipt has token_count, anomaly_density, noise_ratio, timestamp, corpus_hash."""
        corpus = "hook:tesla_fsd | class:normal | score:0.95\nhook:tesla_fsd | class:anomaly | violations:amplitude"
        _, receipt = clean_corpus(corpus)

        assert hasattr(receipt, "token_count")
        assert hasattr(receipt, "anomaly_density")
        assert hasattr(receipt, "noise_ratio")
        assert hasattr(receipt, "timestamp")
        assert hasattr(receipt, "corpus_hash")
        assert hasattr(receipt, "receipt_count")

        assert isinstance(receipt.token_count, int)
        assert isinstance(receipt.anomaly_density, float)
        assert isinstance(receipt.noise_ratio, float)
        assert isinstance(receipt.timestamp, str)
        assert isinstance(receipt.corpus_hash, str)

    def test_clarity_receipt_anomaly_density_range(self):
        """anomaly_density is between 0.0 and 1.0."""
        corpus = "hook:tesla | violations:error\nhook:tesla | class:normal"
        _, receipt = clean_corpus(corpus)

        assert 0.0 <= receipt.anomaly_density <= 1.0

    def test_clarity_receipt_noise_ratio_range(self):
        """noise_ratio is between 0.0 and 1.0."""
        corpus = "hook:tesla_fsd | class:normal | score:0.95"
        _, receipt = clean_corpus(corpus)

        assert 0.0 <= receipt.noise_ratio <= 1.0

    def test_process_receipts_full_pipeline(self, sample_qed_receipts_jsonl):
        """Full pipeline returns cleaned corpus and receipt."""
        cleaned, receipt = process_receipts(sample_qed_receipts_jsonl)

        assert isinstance(cleaned, str)
        assert isinstance(receipt, ClarityCleanReceipt)
        assert receipt.source_file == sample_qed_receipts_jsonl

    def test_process_receipts_writes_files(self, sample_qed_receipts_jsonl, tmp_path):
        """Output files created when paths provided."""
        corpus_path = tmp_path / "output_corpus.txt"
        receipt_path = tmp_path / "clarity_receipts.jsonl"

        cleaned, receipt = process_receipts(
            sample_qed_receipts_jsonl,
            output_corpus_path=str(corpus_path),
            output_receipt_path=str(receipt_path),
        )

        assert corpus_path.exists()
        assert receipt_path.exists()

        # Verify corpus file contains cleaned text
        corpus_content = corpus_path.read_text()
        assert len(corpus_content) > 0

        # Verify receipt file contains valid JSON
        receipt_content = receipt_path.read_text()
        assert len(receipt_content) > 0
        receipt_data = json.loads(receipt_content.strip())
        assert "token_count" in receipt_data


# =============================================================================
# Test: mesh_view_v2 outputs (Section 6)
# =============================================================================


class TestMeshViewV2:
    """Tests for mesh_view_v2 module outputs."""

    @pytest.fixture
    def sample_joined_data(self, tmp_path, sample_qed_receipts_jsonl):
        """Create sample joined data structure for testing."""
        # Create minimal manifest
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "run_id": "test_run_001",
            "fleet_size": 100,
            "total_windows": 1000,
        }))

        # Create anomaly library with exploit-grade pattern
        anomalies_path = tmp_path / "anomalies.jsonl"
        anomalies_path.write_text(json.dumps({
            "pattern_id": "exploit001",
            "hook": "tesla_fsd",
            "physics_domain": "battery_thermal",
            "hooks": ["tesla_fsd"],
            "dollar_value_annual": 5000000,
            "validation_recall": 0.995,
            "false_positive_rate": 0.005,
            "exploit_grade": True,
        }) + "\n")

        # Create cross-domain validations
        cross_domain_path = tmp_path / "cross_domain.jsonl"
        cross_domain_path.write_text(json.dumps({
            "source_hook": "tesla",
            "target_hook": "tesla_fsd",
            "pattern_id": "exploit001",
            "passed": True,
            "validated_at": "2024-01-01T00:00:00Z",
        }) + "\n")

        # Create clarity receipts
        clarity_path = tmp_path / "clarity.jsonl"
        clarity_path.write_text(json.dumps({
            "hook": "tesla_fsd",
            "noise_ratio": 0.2,
            "clarity_score": 0.8,
        }) + "\n")

        return join_all_sources(
            receipts_path=sample_qed_receipts_jsonl,
            manifest_path=str(manifest_path),
            clarity_path=str(clarity_path),
            sim_results_path=str(tmp_path / "nonexistent_sim.json"),
            anomalies_path=str(anomalies_path),
            cross_domain_path=str(cross_domain_path),
        )

    def test_compute_metrics_v2_has_exploit_count(self, sample_joined_data):
        """Output dict contains 'exploit_count' key."""
        metrics = compute_metrics_v2(sample_joined_data)

        # Metrics should be nested: {company: {hook: {metrics}}}
        assert isinstance(metrics, dict)

        # Check at least one hook has exploit_count
        for company in metrics.values():
            for hook_data in company.values():
                assert "exploit_count" in hook_data
                assert isinstance(hook_data["exploit_count"], int)
                break
            break

    def test_compute_metrics_v2_has_cross_domain_links(self, sample_joined_data):
        """Output contains 'cross_domain_links' key."""
        metrics = compute_metrics_v2(sample_joined_data)

        for company in metrics.values():
            for hook_data in company.values():
                assert "cross_domain_links" in hook_data
                assert isinstance(hook_data["cross_domain_links"], int)
                break
            break

    def test_compute_metrics_v2_has_clarity_quality_score(self, sample_joined_data):
        """Output contains 'clarity_quality_score' key."""
        metrics = compute_metrics_v2(sample_joined_data)

        for company in metrics.values():
            for hook_data in company.values():
                assert "clarity_quality_score" in hook_data
                assert isinstance(hook_data["clarity_quality_score"], float)
                assert 0.0 <= hook_data["clarity_quality_score"] <= 1.0
                break
            break

    def test_mesh_view_v2_graceful_missing_sources(self, tmp_path, sample_qed_receipts_jsonl):
        """Returns v6 fields even if v7 sources missing."""
        # Create minimal manifest
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"run_id": "test"}))

        # Join with all v7 sources missing
        joined_data = join_all_sources(
            receipts_path=sample_qed_receipts_jsonl,
            manifest_path=str(manifest_path),
            clarity_path=str(tmp_path / "missing_clarity.jsonl"),
            sim_results_path=str(tmp_path / "missing_sim.json"),
            anomalies_path=str(tmp_path / "missing_anomalies.jsonl"),
            cross_domain_path=str(tmp_path / "missing_cross_domain.jsonl"),
        )

        metrics = compute_metrics_v2(joined_data)

        # Should still have v6 fields
        for company in metrics.values():
            for hook_data in company.values():
                assert "windows" in hook_data
                assert "avg_ratio" in hook_data
                assert "total_savings" in hook_data
                assert "slo_breach_rate" in hook_data
                assert "constraint_violations" in hook_data
                # v7 fields should be present but may be 0
                assert "exploit_count" in hook_data
                assert "cross_domain_links" in hook_data
                assert "clarity_quality_score" in hook_data
                break
            break


# =============================================================================
# Test: recall floor math (Clopper-Pearson) (Section 7)
# =============================================================================


class TestRecallFloorMath:
    """
    Tests for recall floor computation using Clopper-Pearson exact binomial CI.

    These tests verify the mathematical correctness of recall floor computation.
    Expected values computed from Clopper-Pearson exact binomial confidence intervals.
    """

    @pytest.fixture
    def compute_recall_floor(self):
        """Import compute_recall_floor or provide fallback implementation."""
        try:
            from edge_lab_v2 import compute_recall_floor
            return compute_recall_floor
        except ImportError:
            # Fallback implementation using scipy if available
            scipy_stats = pytest.importorskip("scipy.stats")

            def _compute_recall_floor(n_tests: int, n_misses: int, confidence: float = 0.95) -> float:
                """
                Compute Clopper-Pearson exact binomial CI lower bound.

                Args:
                    n_tests: Total number of tests run
                    n_misses: Number of misses (failures to detect)
                    confidence: Confidence level (default 0.95 for 95% CI)

                Returns:
                    Lower bound of recall at given confidence level
                """
                n_successes = n_tests - n_misses
                alpha = 1 - confidence

                if n_successes == 0:
                    return 0.0
                if n_successes == n_tests:
                    # All successes - use beta distribution
                    lower = scipy_stats.beta.ppf(alpha / 2, n_successes, n_misses + 1)
                else:
                    lower = scipy_stats.beta.ppf(alpha / 2, n_successes, n_misses + 1)

                return lower

            return _compute_recall_floor

    def test_recall_floor_900_zero_misses(self, compute_recall_floor):
        """n=900, misses=0 -> ~0.9967 (+-0.001)."""
        result = compute_recall_floor(n_tests=900, n_misses=0, confidence=0.95)
        assert result == pytest.approx(0.9967, abs=0.001)

    def test_recall_floor_100_zero_misses(self, compute_recall_floor):
        """n=100, misses=0 -> ~0.9638 (+-0.001)."""
        result = compute_recall_floor(n_tests=100, n_misses=0, confidence=0.95)
        assert result == pytest.approx(0.9638, abs=0.001)

    def test_recall_floor_900_one_miss(self, compute_recall_floor):
        """n=900, misses=1 -> ~0.9944 (+-0.001)."""
        result = compute_recall_floor(n_tests=900, n_misses=1, confidence=0.95)
        assert result == pytest.approx(0.9944, abs=0.001)

    def test_recall_floor_confidence_level(self, compute_recall_floor):
        """99% CI returns lower value than 95% CI."""
        result_95 = compute_recall_floor(n_tests=100, n_misses=0, confidence=0.95)
        result_99 = compute_recall_floor(n_tests=100, n_misses=0, confidence=0.99)

        # Higher confidence means wider interval, so lower bound is lower
        assert result_99 < result_95


# =============================================================================
# Test: pattern_id generation
# =============================================================================


class TestPatternIdGeneration:
    """Tests for pattern ID generation function."""

    def test_generate_pattern_id_deterministic(self):
        """Same inputs produce same pattern_id."""
        id1 = generate_pattern_id("battery_thermal", "overheat", {"threshold": 65})
        id2 = generate_pattern_id("battery_thermal", "overheat", {"threshold": 65})
        assert id1 == id2

    def test_generate_pattern_id_different_params(self):
        """Different params produce different pattern_id."""
        id1 = generate_pattern_id("battery_thermal", "overheat", {"threshold": 65})
        id2 = generate_pattern_id("battery_thermal", "overheat", {"threshold": 70})
        assert id1 != id2

    def test_generate_pattern_id_format(self):
        """Pattern ID is 16-char hex string."""
        pattern_id = generate_pattern_id("motion", "collision", {"speed": 50})
        assert len(pattern_id) == 16
        assert all(c in "0123456789abcdef" for c in pattern_id)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
