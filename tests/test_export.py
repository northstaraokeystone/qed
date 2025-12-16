"""
tests/test_export.py - Model Export Tests

Validates model export functionality for Grok collaboration.
CLAUDEME v3.1 Compliant: Tests with assertions.
"""

import pytest
import os
import tempfile
import json

from sim.export import export_model_details, model_to_grok_format


class TestExportModelDetails:
    """Test export_model_details function."""

    def test_returns_dict(self):
        """Export returns a dict."""
        result = export_model_details()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_has_required_keys(self):
        """Export dict contains all required keys."""
        result = export_model_details()
        required_keys = [
            "version",
            "name",
            "constants",
            "affinity_matrix",
            "scenarios",
            "receipt_schemas",
            "physics_notes",
            "dual_hash",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key '{key}'"

    def test_includes_constants(self):
        """Export includes threshold constants."""
        result = export_model_details()
        constants = result["constants"]
        assert "MIN_AFFINITY_THRESHOLD" in constants
        assert "STOCHASTIC_AFFINITY_THRESHOLD" in constants
        assert "DYNAMIC_THRESHOLD_SCALE" in constants
        assert "VARIANCE_SWEEP_RANGE" in constants

    def test_includes_affinity_matrix(self):
        """Export includes affinity matrix."""
        result = export_model_details()
        matrix = result["affinity_matrix"]
        assert len(matrix) > 0, "Affinity matrix should not be empty"
        # Check at least one known pair
        assert any("tesla" in key and "spacex" in key for key in matrix.keys()), (
            "Should include tesla_spacex pair"
        )

    def test_hash_present_and_valid_format(self):
        """Dual hash is present and in correct format."""
        result = export_model_details()
        dual_hash = result["dual_hash"]
        assert dual_hash is not None, "dual_hash should not be None"
        # Dual hash format is SHA256:BLAKE3
        assert ":" in dual_hash, "dual_hash should contain ':' separator"
        parts = dual_hash.split(":")
        assert len(parts) == 2, "dual_hash should have two parts"
        # Each part should be a hex string
        for part in parts:
            assert len(part) == 64, f"Hash part should be 64 chars, got {len(part)}"
            assert all(c in "0123456789abcdef" for c in part), "Hash should be hex"

    def test_file_written_when_path_provided(self):
        """Export writes to file when path is provided."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = export_model_details(output_path=temp_path)

            # File should exist
            assert os.path.exists(temp_path), "Output file should exist"

            # File should contain valid JSON
            with open(temp_path, "r") as f:
                loaded = json.load(f)

            # Loaded data should match returned dict (except dual_hash order)
            assert loaded["version"] == result["version"]
            assert loaded["name"] == result["name"]

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_scenarios_count_is_eight(self):
        """Export includes 8 mandatory scenarios."""
        result = export_model_details()
        scenarios = result["scenarios"]
        assert len(scenarios) == 8, f"Expected 8 scenarios, got {len(scenarios)}"
        assert "STOCHASTIC_AFFINITY" in scenarios, "Should include STOCHASTIC_AFFINITY"


class TestModelToGrokFormat:
    """Test model_to_grok_format function."""

    def test_returns_string(self):
        """Grok format returns a string."""
        model = export_model_details()
        result = model_to_grok_format(model)
        assert isinstance(result, str), f"Expected str, got {type(result)}"

    def test_within_character_limit(self):
        """Grok format is within 280 character limit."""
        model = export_model_details()
        result = model_to_grok_format(model)
        assert len(result) <= 280, f"Length {len(result)} exceeds 280 char limit"

    def test_includes_version(self):
        """Grok format includes version."""
        model = export_model_details()
        result = model_to_grok_format(model)
        assert "v12" in result or "QED" in result, "Should include version or name"

    def test_includes_thresholds(self):
        """Grok format includes threshold info."""
        model = export_model_details()
        result = model_to_grok_format(model)
        assert "0.48" in result or "Det=" in result, "Should include deterministic threshold"
        assert "0.52" in result or "Stoch=" in result, "Should include stochastic threshold"

    def test_includes_hash_prefix(self):
        """Grok format includes hash prefix."""
        model = export_model_details()
        result = model_to_grok_format(model)
        assert "Hash:" in result, "Should include hash prefix"

    def test_handles_empty_model(self):
        """Grok format handles empty/minimal model gracefully."""
        model = {}
        result = model_to_grok_format(model)
        assert isinstance(result, str)
        assert len(result) <= 280

    def test_handles_missing_keys(self):
        """Grok format handles missing keys with defaults."""
        model = {"version": "test", "constants": {}}
        result = model_to_grok_format(model)
        assert isinstance(result, str)
        assert len(result) <= 280
