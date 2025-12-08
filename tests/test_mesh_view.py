"""
Tests for mesh_view_v1.py - Mesh View Aggregator

Tests cover:
  - Manifest loading and validation
  - Receipt sampling from JSONL
  - Hook/company parsing
  - Metrics computation (avg_ratio, savings, breach rate, violations)
  - JSON/table output formatting
  - Edge cases (empty files, missing fields, malformed data)
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from mesh_view_v1 import (
    load_manifest,
    sample_receipts,
    extract_hook_from_receipt,
    parse_company_from_hook,
    compute_metrics,
    emit_view,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_manifest():
    """Sample manifest dictionary."""
    return {
        "run_id": "test_run_001",
        "fleet_size": 1000,
        "total_windows": 500,
        "receipts_file": "receipts.jsonl",
        "timestamps": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T01:00:00Z",
        },
    }


@pytest.fixture
def sample_receipts_data():
    """Sample receipt data for testing."""
    return [
        {
            "ts": "2025-01-01T00:00:01Z",
            "window_id": "tesla_fsd_1000_abc123",
            "params": {"A": 12.0, "f": 40.0, "scenario": "tesla_fsd"},
            "ratio": 60.0,
            "H_bits": 7200.0,
            "recall": 0.99,
            "savings_M": 38.0,
            "verified": True,
            "violations": [],
            "trace": "qed_v6 scenario=tesla_fsd N=1000",
        },
        {
            "ts": "2025-01-01T00:00:02Z",
            "window_id": "tesla_fsd_1000_def456",
            "params": {"A": 13.0, "f": 40.0, "scenario": "tesla_fsd"},
            "ratio": 58.0,
            "H_bits": 7100.0,
            "recall": 0.98,
            "savings_M": 37.5,
            "verified": True,
            "violations": [],
            "trace": "qed_v6 scenario=tesla_fsd N=1000",
        },
        {
            "ts": "2025-01-01T00:00:03Z",
            "window_id": "spacex_flight_1000_ghi789",
            "params": {"A": 15.0, "f": 55.0, "scenario": "spacex_flight"},
            "ratio": 55.0,
            "H_bits": 6800.0,
            "recall": 0.97,
            "savings_M": 25.0,
            "verified": False,
            "violations": [{"constraint_id": "amplitude_bound", "value": 15.0}],
            "trace": "qed_v6 scenario=spacex_flight N=1000",
        },
    ]


@pytest.fixture
def manifest_file(sample_manifest):
    """Create temporary manifest file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(sample_manifest, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def receipts_file(sample_receipts_data):
    """Create temporary receipts JSONL file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for receipt in sample_receipts_data:
            f.write(json.dumps(receipt) + "\n")
        f.flush()
        yield f.name
    os.unlink(f.name)


# -----------------------------------------------------------------------------
# Test load_manifest
# -----------------------------------------------------------------------------

class TestLoadManifest:
    """Tests for load_manifest function."""

    def test_load_manifest_success(self, manifest_file, sample_manifest):
        """Test successful manifest loading."""
        result = load_manifest(manifest_file)
        assert result["run_id"] == sample_manifest["run_id"]
        assert result["fleet_size"] == sample_manifest["fleet_size"]
        assert result["total_windows"] == sample_manifest["total_windows"]

    def test_load_manifest_file_not_found(self):
        """Test FileNotFoundError for missing manifest."""
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/path/manifest.json")

    def test_load_manifest_invalid_json(self):
        """Test JSONDecodeError for malformed manifest."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{ invalid json }")
            f.flush()
            path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_manifest(path)
        finally:
            os.unlink(path)


# -----------------------------------------------------------------------------
# Test sample_receipts
# -----------------------------------------------------------------------------

class TestSampleReceipts:
    """Tests for sample_receipts function."""

    def test_sample_receipts_success(self, receipts_file):
        """Test successful receipt sampling."""
        receipts = sample_receipts(receipts_file, n=100)
        assert len(receipts) == 3
        assert receipts[0]["ratio"] == 60.0

    def test_sample_receipts_limit(self, receipts_file):
        """Test sample limit is respected."""
        receipts = sample_receipts(receipts_file, n=2)
        assert len(receipts) == 2

    def test_sample_receipts_file_not_found(self):
        """Test FileNotFoundError for missing receipts file."""
        with pytest.raises(FileNotFoundError):
            sample_receipts("/nonexistent/receipts.jsonl")

    def test_sample_receipts_empty_file(self):
        """Test handling of empty receipts file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.flush()
            path = f.name

        try:
            receipts = sample_receipts(path)
            assert receipts == []
        finally:
            os.unlink(path)

    def test_sample_receipts_skips_malformed_lines(self):
        """Test that malformed JSON lines are skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write('{"ratio": 60.0, "verified": true}\n')
            f.write("not valid json\n")
            f.write('{"ratio": 55.0, "verified": false}\n')
            f.flush()
            path = f.name

        try:
            receipts = sample_receipts(path)
            assert len(receipts) == 2
        finally:
            os.unlink(path)

    def test_sample_receipts_skips_comments(self):
        """Test that comment lines are skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write("# This is a comment\n")
            f.write('{"ratio": 60.0}\n')
            f.write("# Another comment\n")
            f.flush()
            path = f.name

        try:
            receipts = sample_receipts(path)
            assert len(receipts) == 1
        finally:
            os.unlink(path)


# -----------------------------------------------------------------------------
# Test extract_hook_from_receipt
# -----------------------------------------------------------------------------

class TestExtractHookFromReceipt:
    """Tests for extract_hook_from_receipt function."""

    def test_extract_from_direct_hook_field(self):
        """Test extraction from direct 'hook' field."""
        receipt = {"hook": "tesla_fsd", "ratio": 60.0}
        assert extract_hook_from_receipt(receipt) == "tesla_fsd"

    def test_extract_from_params_scenario(self):
        """Test extraction from params.scenario."""
        receipt = {"params": {"scenario": "spacex_flight"}, "ratio": 55.0}
        assert extract_hook_from_receipt(receipt) == "spacex_flight"

    def test_extract_from_params_hook_name(self):
        """Test extraction from params.hook_name."""
        receipt = {"params": {"hook_name": "neuralink_stream"}, "ratio": 50.0}
        assert extract_hook_from_receipt(receipt) == "neuralink_stream"

    def test_extract_from_window_id(self):
        """Test extraction from window_id prefix."""
        receipt = {"window_id": "boring_tunnel_1000_abc123", "ratio": 45.0}
        assert extract_hook_from_receipt(receipt) == "boring_tunnel"

    def test_extract_from_trace(self):
        """Test extraction from trace field."""
        receipt = {"trace": "qed_v6 scenario=xai_eval N=1000", "ratio": 40.0}
        assert extract_hook_from_receipt(receipt) == "xai_eval"

    def test_extract_fallback_generic(self):
        """Test fallback to 'generic' when no hook found."""
        receipt = {"ratio": 60.0}
        assert extract_hook_from_receipt(receipt) == "generic"


# -----------------------------------------------------------------------------
# Test parse_company_from_hook
# -----------------------------------------------------------------------------

class TestParseCompanyFromHook:
    """Tests for parse_company_from_hook function."""

    def test_parse_tesla(self):
        """Test parsing 'tesla' from 'tesla_fsd'."""
        assert parse_company_from_hook("tesla_fsd") == "tesla"

    def test_parse_spacex(self):
        """Test parsing 'spacex' from 'spacex_flight'."""
        assert parse_company_from_hook("spacex_flight") == "spacex"

    def test_parse_generic(self):
        """Test 'generic' returns 'generic'."""
        assert parse_company_from_hook("generic") == "generic"

    def test_parse_multiple_underscores(self):
        """Test parsing with multiple underscores."""
        assert parse_company_from_hook("boring_tunnel_test") == "boring"


# -----------------------------------------------------------------------------
# Test compute_metrics
# -----------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_basic_metrics(self, sample_receipts_data):
        """Test basic metrics computation."""
        metrics = compute_metrics(sample_receipts_data)

        # Should have 2 companies: tesla, spacex
        assert "tesla" in metrics
        assert "spacex" in metrics

        # Tesla metrics (2 receipts)
        tesla = metrics["tesla"]["tesla_fsd"]
        assert tesla["windows"] == 2
        assert tesla["avg_ratio"] == 59.0  # (60 + 58) / 2
        assert tesla["total_savings"] == 75.5  # 38 + 37.5
        assert tesla["slo_breach_rate"] == 0.0  # Both verified
        assert tesla["constraint_violations"] == 0

        # SpaceX metrics (1 receipt)
        spacex = metrics["spacex"]["spacex_flight"]
        assert spacex["windows"] == 1
        assert spacex["avg_ratio"] == 55.0
        assert spacex["total_savings"] == 25.0
        assert spacex["slo_breach_rate"] == 100.0  # Unverified
        assert spacex["constraint_violations"] == 1

    def test_compute_metrics_empty_list(self):
        """Test metrics computation with empty receipt list."""
        metrics = compute_metrics([])
        assert metrics == {}

    def test_compute_metrics_missing_fields(self):
        """Test handling of receipts with missing fields."""
        receipts = [
            {"hook": "tesla_fsd"},  # Minimal receipt
            {"hook": "tesla_fsd", "ratio": 50.0, "verified": None},
        ]
        metrics = compute_metrics(receipts)

        tesla = metrics["tesla"]["tesla_fsd"]
        assert tesla["windows"] == 2
        assert tesla["avg_ratio"] == 50.0  # Only one has ratio
        assert tesla["slo_breach_rate"] == 100.0  # Both unverified (missing or None)

    def test_compute_metrics_violations_as_int(self):
        """Test handling violations as int instead of list."""
        receipts = [
            {"hook": "tesla_fsd", "violations": 5, "verified": False},
        ]
        metrics = compute_metrics(receipts)
        assert metrics["tesla"]["tesla_fsd"]["constraint_violations"] == 5


# -----------------------------------------------------------------------------
# Test emit_view
# -----------------------------------------------------------------------------

class TestEmitView:
    """Tests for emit_view function."""

    def test_emit_json_format(self, sample_receipts_data, sample_manifest):
        """Test JSON output format."""
        metrics = compute_metrics(sample_receipts_data)
        output = emit_view(metrics, sample_manifest)

        # Should be valid JSON
        parsed = json.loads(output)
        assert "_meta" in parsed
        assert "companies" in parsed
        assert parsed["_meta"]["run_id"] == "test_run_001"

    def test_emit_without_manifest(self, sample_receipts_data):
        """Test JSON output without manifest metadata."""
        metrics = compute_metrics(sample_receipts_data)
        output = emit_view(metrics)

        parsed = json.loads(output)
        assert "_meta" not in parsed
        assert "companies" in parsed


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests from AI4 edge probe."""

    def test_edge_001_manifest_load(self, manifest_file):
        """EDGE_001: Manifest load returns dict with fleet_size."""
        manifest = load_manifest(manifest_file)
        assert isinstance(manifest, dict)
        assert "fleet_size" in manifest
        assert manifest["fleet_size"] == 1000

    def test_edge_002_sample_receipts(self, receipts_file):
        """EDGE_002: Sample 100 lines, no errors."""
        receipts = sample_receipts(receipts_file, n=100)
        assert isinstance(receipts, list)
        assert len(receipts) <= 100

    def test_edge_003_agg_metrics(self):
        """EDGE_003: avg_ratio=mean, savings=sum, breach_rate=mean(1-verified)*100."""
        receipts = [
            {"hook": "tesla_fsd", "ratio": 50, "savings_M": 19, "verified": True, "violations": []},
            {"hook": "tesla_fsd", "ratio": 50, "savings_M": 19, "verified": True, "violations": []},
        ]
        metrics = compute_metrics(receipts)

        tesla = metrics["tesla"]["tesla_fsd"]
        assert tesla["avg_ratio"] == 50.0
        assert tesla["total_savings"] == 38.0
        assert tesla["slo_breach_rate"] == 0.0
        assert tesla["constraint_violations"] == 0

    def test_mixed_verified_status(self):
        """Test breach rate with mixed verified status."""
        receipts = [
            {"hook": "tesla_fsd", "ratio": 60, "verified": True},
            {"hook": "tesla_fsd", "ratio": 60, "verified": False},
            {"hook": "tesla_fsd", "ratio": 60, "verified": True},
            {"hook": "tesla_fsd", "ratio": 60, "verified": None},  # Counts as breach
        ]
        metrics = compute_metrics(receipts)

        # 2 breaches out of 4 = 50%
        assert metrics["tesla"]["tesla_fsd"]["slo_breach_rate"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
