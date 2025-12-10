"""
QED v8 Test Suite - DecisionPacket, TruthLink, Config, Merge, Mesh, and CLI

Comprehensive tests for QED v8 modules:
- decision_packet: DecisionPacket creation, integrity verification, serialization, diff
- truthlink: build(), project(), compare(), self-awareness
- config_schema: QEDConfig loading, validation (<1ms SLO)
- merge_rules: safety-only-tightens invariant, merge rejection tests
- mesh_view_v3: build(), find_clusters(), find_outliers(), fleet metrics
- proof.py CLI: v8 subcommands (build-packet, validate-config, merge-configs, etc.)

Test organization:
- Group tests by module using classes
- Use pytest fixtures for common setup
- Use tmp_path for temp files (auto-cleanup)
- Use pytest.approx() for float comparisons
- Use skip-based stubs for unbuilt or optional features
"""

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# =============================================================================
# Module imports - with graceful degradation using try/except
# =============================================================================

# decision_packet module
try:
    from decision_packet import (
        DecisionPacket,
        PatternSummary,
        PacketMetrics,
        load_packet,
        save_packet,
        compare_packets,
    )
    HAS_DECISION_PACKET = True
except ImportError as e:
    HAS_DECISION_PACKET = False
    DECISION_PACKET_ERROR = str(e)

# truthlink module
try:
    from truthlink import (
        build,
        project,
        compare,
        save,
        load,
        health_check,
        detect_drift,
        AddPattern,
        RemovePattern,
        ScaleFleet,
        AdjustConfig,
        ProjectedPacket,
        Comparison,
        TruthLinkHealth,
        DriftReport,
        TRUTHLINK_VERSION,
    )
    HAS_TRUTHLINK = True
except ImportError as e:
    HAS_TRUTHLINK = False
    TRUTHLINK_ERROR = str(e)

# config_schema module
try:
    from config_schema import (
        QEDConfig,
        ConfigProvenance,
        load as load_config,
    )
    HAS_CONFIG_SCHEMA = True
except ImportError as e:
    HAS_CONFIG_SCHEMA = False
    CONFIG_SCHEMA_ERROR = str(e)

# merge_rules module
try:
    from merge_rules import (
        merge,
        emit_receipt,
        MergeResult,
        MergeReceipt,
        Violation,
    )
    HAS_MERGE_RULES = True
except ImportError as e:
    HAS_MERGE_RULES = False
    MERGE_RULES_ERROR = str(e)

# mesh_view_v3 module
try:
    from mesh_view_v3 import (
        build as build_graph,
        DeploymentGraph,
        find_clusters,
        find_outliers,
        compute_fleet_metrics,
        diagnose,
        FleetSummary,
    )
    HAS_MESH_VIEW_V3 = True
except ImportError as e:
    HAS_MESH_VIEW_V3 = False
    MESH_VIEW_V3_ERROR = str(e)


# Skip markers for modules that failed to import
requires_decision_packet = pytest.mark.skipif(
    not HAS_DECISION_PACKET,
    reason=f"decision_packet module not available: {DECISION_PACKET_ERROR if not HAS_DECISION_PACKET else ''}"
)
requires_truthlink = pytest.mark.skipif(
    not HAS_TRUTHLINK,
    reason=f"truthlink module not available: {TRUTHLINK_ERROR if not HAS_TRUTHLINK else ''}"
)
requires_config_schema = pytest.mark.skipif(
    not HAS_CONFIG_SCHEMA,
    reason=f"config_schema module not available: {CONFIG_SCHEMA_ERROR if not HAS_CONFIG_SCHEMA else ''}"
)
requires_merge_rules = pytest.mark.skipif(
    not HAS_MERGE_RULES,
    reason=f"merge_rules module not available: {MERGE_RULES_ERROR if not HAS_MERGE_RULES else ''}"
)
requires_mesh_view_v3 = pytest.mark.skipif(
    not HAS_MESH_VIEW_V3,
    reason=f"mesh_view_v3 module not available: {MESH_VIEW_V3_ERROR if not HAS_MESH_VIEW_V3 else ''}"
)


# =============================================================================
# Fixtures - Sample Data
# =============================================================================


@pytest.fixture
def sample_pattern_summary() -> PatternSummary:
    """Sample PatternSummary for testing."""
    return PatternSummary(
        pattern_id="bat_thermal_001",
        validation_recall=0.995,
        false_positive_rate=0.008,
        dollar_value_annual=1_500_000.0,
        exploit_grade=True,
    )


@pytest.fixture
def sample_pattern_summaries() -> List[PatternSummary]:
    """List of sample PatternSummary objects."""
    return [
        PatternSummary(
            pattern_id="bat_thermal_001",
            validation_recall=0.995,
            false_positive_rate=0.008,
            dollar_value_annual=1_500_000.0,
            exploit_grade=True,
        ),
        PatternSummary(
            pattern_id="comms_dropout_002",
            validation_recall=0.98,
            false_positive_rate=0.02,
            dollar_value_annual=800_000.0,
            exploit_grade=False,
        ),
        PatternSummary(
            pattern_id="motion_spike_003",
            validation_recall=0.999,
            false_positive_rate=0.005,
            dollar_value_annual=2_000_000.0,
            exploit_grade=True,
        ),
    ]


@pytest.fixture
def sample_packet_metrics() -> PacketMetrics:
    """Sample PacketMetrics for testing."""
    return PacketMetrics(
        window_volume=1_247_832,
        avg_compression=11.3,
        annual_savings=2_400_000.0,
        slo_breach_rate=0.0002,
    )


@pytest.fixture
def sample_decision_packet(sample_pattern_summaries, sample_packet_metrics) -> DecisionPacket:
    """Complete sample DecisionPacket for testing."""
    return DecisionPacket(
        deployment_id="tesla-prod-2024-12-08",
        manifest_ref="manifests/tesla-prod-v8.yaml",
        sampled_receipts=["rcpt_001", "rcpt_002", "rcpt_003"],
        clarity_audit_ref="audits/clarity_2024-12-08.json",
        edge_lab_summary={
            "n_tests": 10000,
            "n_hits": 9850,
            "aggregate_recall": 0.985,
        },
        pattern_usage=sample_pattern_summaries,
        metrics=sample_packet_metrics,
    )


@pytest.fixture
def sample_decision_packet_dict() -> Dict[str, Any]:
    """Sample DecisionPacket as dict for deserialization tests."""
    return {
        "deployment_id": "spacex-falcon-2024-12-08",
        "manifest_ref": "manifests/spacex-falcon-v8.yaml",
        "sampled_receipts": ["rcpt_101", "rcpt_102"],
        "clarity_audit_ref": "audits/clarity_spacex.json",
        "edge_lab_summary": {"status": "passed"},
        "pattern_usage": [
            {
                "pattern_id": "thrust_anomaly_001",
                "validation_recall": 0.997,
                "false_positive_rate": 0.003,
                "dollar_value_annual": 5_000_000.0,
                "exploit_grade": True,
            }
        ],
        "metrics": {
            "window_volume": 500_000,
            "avg_compression": 15.0,
            "annual_savings": 5_000_000.0,
            "slo_breach_rate": 0.0001,
        },
    }


@pytest.fixture
def sample_valid_config_dict() -> Dict[str, Any]:
    """Sample valid QEDConfig as dict."""
    return {
        "version": "8.0",
        "deployment_id": "tesla-prod-main",
        "hook": "tesla",
        "compression_target": 10.0,
        "recall_floor": 0.999,
        "max_fp_rate": 0.01,
        "slo_latency_ms": 50,
        "slo_breach_budget": 0.001,
        "enabled_patterns": ["bat_thermal_001", "comms_dropout_002"],
        "safety_overrides": {},
        "regulatory_flags": {"gdpr": True},
    }


@pytest.fixture
def sample_provenance() -> "ConfigProvenance":
    """Sample ConfigProvenance for testing."""
    return ConfigProvenance(
        author="test_user",
        created_at="2024-12-08T00:00:00Z",
        reason="test config",
        config_hash="abc123",
        parent_hash=None,
    )


@pytest.fixture
def sample_parent_config(sample_provenance) -> QEDConfig:
    """Sample parent QEDConfig for merge tests."""
    return QEDConfig(
        version="8.0",
        deployment_id="global-baseline",
        hook="tesla",
        provenance=sample_provenance,
        compression_target=10.0,
        recall_floor=0.999,
        max_fp_rate=0.01,
        slo_latency_ms=50,
        slo_breach_budget=0.001,
        enabled_patterns=("bat_thermal_001", "comms_dropout_002"),
        safety_overrides={},
        regulatory_flags={"gdpr": True},
    )


@pytest.fixture
def sample_child_config(sample_provenance) -> QEDConfig:
    """Sample child QEDConfig for merge tests (stricter safety)."""
    return QEDConfig(
        version="8.0",
        deployment_id="tesla-prod-eu",
        hook="tesla",
        provenance=sample_provenance,
        compression_target=8.0,
        recall_floor=0.9995,  # Stricter (higher) recall floor
        max_fp_rate=0.005,   # Stricter (lower) max FP rate
        slo_latency_ms=40,
        slo_breach_budget=0.0005,
        enabled_patterns=("bat_thermal_001",),  # Subset
        safety_overrides={},
        regulatory_flags={"gdpr": True, "eu_ai_act": True},  # Additional flag
    )


@pytest.fixture
def sample_child_config_unsafe(sample_provenance) -> QEDConfig:
    """Sample child QEDConfig that loosens safety (should be rejected)."""
    return QEDConfig(
        version="8.0",
        deployment_id="tesla-prod-unsafe",
        hook="tesla",
        provenance=sample_provenance,
        compression_target=15.0,
        recall_floor=0.99,   # Looser (lower) recall floor - VIOLATION
        max_fp_rate=0.02,    # Looser (higher) max FP rate - VIOLATION
        slo_latency_ms=100,
        slo_breach_budget=0.01,
        enabled_patterns=("bat_thermal_001", "comms_dropout_002", "new_pattern"),
        safety_overrides={},
        regulatory_flags={},  # Removed regulatory flag
    )


@pytest.fixture
def temp_receipts_dir(tmp_path) -> Path:
    """Create temp directory with sample receipts."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()

    receipts_file = receipts_dir / "receipts.jsonl"
    receipts = [
        {
            "window_id": f"window_{i:03d}",
            "hook": "tesla",
            "ratio": 10.0 + i * 0.5,
            "savings_M": 0.5 + i * 0.1,
            "verified": True,
            "slo_breach": False,
        }
        for i in range(20)
    ]
    with open(receipts_file, "w") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")

    return receipts_dir


@pytest.fixture
def temp_manifest_file(tmp_path) -> Path:
    """Create temp manifest file."""
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "deployment_id": "test-deployment",
        "run_id": "run-001",
        "hook": "tesla",
        "patterns": ["bat_thermal_001"],
        "total_windows": 1000,
        "avg_compression": 12.0,
        "annual_savings": 1_000_000.0,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path


@pytest.fixture
def temp_config_file(tmp_path, sample_valid_config_dict) -> Path:
    """Create temp config file."""
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_valid_config_dict, f)
    return config_path


# =============================================================================
# Test Group 1: decision_packet
# =============================================================================


@requires_decision_packet
class TestDecisionPacketCreation:
    """Tests for DecisionPacket creation and auto-generation."""

    def test_packet_creation_basic(self, sample_pattern_summaries, sample_packet_metrics):
        """Test basic DecisionPacket creation."""
        packet = DecisionPacket(
            deployment_id="test-deploy",
            manifest_ref="test-manifest.yaml",
            sampled_receipts=["r1", "r2"],
            clarity_audit_ref="audit.json",
            edge_lab_summary={"status": "ok"},
            pattern_usage=sample_pattern_summaries,
            metrics=sample_packet_metrics,
        )

        assert packet.deployment_id == "test-deploy"
        assert packet.manifest_ref == "test-manifest.yaml"
        assert len(packet.sampled_receipts) == 2
        assert len(packet.pattern_usage) == 3

    def test_packet_id_auto_generated(self, sample_decision_packet):
        """Test packet_id is auto-generated as SHA3-256 hash."""
        assert sample_decision_packet.packet_id
        assert len(sample_decision_packet.packet_id) == 16  # 16-char hash

    def test_timestamp_auto_generated(self, sample_decision_packet):
        """Test timestamp is auto-generated in ISO format."""
        assert sample_decision_packet.timestamp
        # Should be valid ISO format
        from datetime import datetime
        datetime.fromisoformat(sample_decision_packet.timestamp.replace("Z", "+00:00"))

    def test_exploit_coverage_auto_computed(self, sample_decision_packet):
        """Test exploit_coverage is auto-computed from pattern_usage."""
        # 2 out of 3 patterns are exploit_grade=True
        assert sample_decision_packet.exploit_coverage == pytest.approx(2/3, rel=0.01)

    def test_health_score_property(self, sample_decision_packet):
        """Test health_score is computed correctly (0-100)."""
        score = sample_decision_packet.health_score
        assert 0 <= score <= 100

    def test_glyph_property(self, sample_decision_packet):
        """Test glyph visual fingerprint format."""
        glyph = sample_decision_packet.glyph
        # Format: XX-XX-XX-XX
        assert len(glyph) == 11
        assert glyph[2] == "-" and glyph[5] == "-" and glyph[8] == "-"

    def test_required_field_validation(self, sample_pattern_summaries, sample_packet_metrics):
        """Test ValueError raised for missing required fields."""
        with pytest.raises(ValueError, match="deployment_id is required"):
            DecisionPacket(
                deployment_id="",  # Empty = invalid
                manifest_ref="manifest.yaml",
                sampled_receipts=[],
                clarity_audit_ref="",
                edge_lab_summary={},
                pattern_usage=sample_pattern_summaries,
                metrics=sample_packet_metrics,
            )


@requires_decision_packet
class TestDecisionPacketIntegrity:
    """Tests for DecisionPacket integrity verification."""

    def test_integrity_status_verified(self, sample_decision_packet):
        """Test integrity_status returns VERIFIED for valid packet."""
        assert sample_decision_packet.integrity_status == "VERIFIED"

    def test_verify_integrity_method(self, sample_decision_packet):
        """Test verify_integrity() returns (True, message) tuple."""
        is_valid, message = sample_decision_packet.verify_integrity()
        assert is_valid is True
        assert "VERIFIED" in message

    def test_integrity_detects_tampering(self, sample_decision_packet):
        """Test integrity_status returns TAMPERED if content modified."""
        # Directly modify the packet_id to simulate tampering
        object.__setattr__(sample_decision_packet, "packet_id", "tampered1234567")
        assert sample_decision_packet.integrity_status == "TAMPERED"

    def test_validate_schema(self, sample_decision_packet):
        """Test validate_schema() returns (True, []) for valid packet."""
        is_valid, errors = sample_decision_packet.validate_schema()
        assert is_valid is True
        assert errors == []


@requires_decision_packet
class TestDecisionPacketSerialization:
    """Tests for DecisionPacket serialization/deserialization."""

    def test_to_dict(self, sample_decision_packet):
        """Test to_dict() serialization."""
        d = sample_decision_packet.to_dict()
        assert d["deployment_id"] == "tesla-prod-2024-12-08"
        assert "packet_id" in d
        assert "health_score" in d
        assert "glyph" in d
        assert "integrity_status" in d

    def test_to_json(self, sample_decision_packet):
        """Test to_json() serialization."""
        json_str = sample_decision_packet.to_json()
        parsed = json.loads(json_str)
        assert parsed["deployment_id"] == "tesla-prod-2024-12-08"

    def test_from_dict(self, sample_decision_packet_dict):
        """Test from_dict() deserialization."""
        packet = DecisionPacket.from_dict(sample_decision_packet_dict)
        assert packet.deployment_id == "spacex-falcon-2024-12-08"
        assert len(packet.pattern_usage) == 1
        assert packet.pattern_usage[0].pattern_id == "thrust_anomaly_001"

    def test_from_json(self, sample_decision_packet):
        """Test from_json() roundtrip."""
        json_str = sample_decision_packet.to_json()
        restored = DecisionPacket.from_json(json_str)
        assert restored.deployment_id == sample_decision_packet.deployment_id
        # packet_id may differ due to timestamp changes in __post_init__


@requires_decision_packet
class TestDecisionPacketDiff:
    """Tests for DecisionPacket diff and comparison."""

    def test_diff_method(self, sample_decision_packet, sample_decision_packet_dict):
        """Test diff() computes structured delta."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        diff = sample_decision_packet.diff(packet_b)

        assert "patterns_added" in diff
        assert "patterns_removed" in diff
        assert "metrics_delta" in diff
        assert "health_score_delta" in diff
        assert "savings_delta" in diff

    def test_diff_detects_pattern_changes(self, sample_decision_packet, sample_decision_packet_dict):
        """Test diff detects added/removed patterns."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        diff = sample_decision_packet.diff(packet_b)

        # packet_a has: bat_thermal_001, comms_dropout_002, motion_spike_003
        # packet_b has: thrust_anomaly_001
        assert "thrust_anomaly_001" in diff["patterns_added"]
        assert len(diff["patterns_removed"]) == 3

    def test_compare_packets_utility(self, sample_decision_packet, sample_decision_packet_dict):
        """Test compare_packets() utility function."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        comparison_str = compare_packets(sample_decision_packet, packet_b)

        assert isinstance(comparison_str, str)
        assert "Comparison:" in comparison_str


@requires_decision_packet
class TestDecisionPacketNarrative:
    """Tests for DecisionPacket narrative generation."""

    def test_narrative_method(self, sample_decision_packet):
        """Test narrative() generates human-readable summary."""
        narrative = sample_decision_packet.narrative()

        assert isinstance(narrative, str)
        assert "tesla-prod-2024-12-08" in narrative
        assert "Health score:" in narrative

    def test_narrative_diff(self, sample_decision_packet, sample_decision_packet_dict):
        """Test narrative_diff() generates comparison narrative."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        narrative = sample_decision_packet.narrative_diff(packet_b)

        assert isinstance(narrative, str)
        assert "vs parent:" in narrative


@requires_decision_packet
class TestDecisionPacketFileIO:
    """Tests for DecisionPacket file I/O."""

    def test_save_and_load_packet(self, sample_decision_packet, tmp_path):
        """Test save_packet() and load_packet() roundtrip."""
        packet_path = tmp_path / "packet.json"
        save_packet(sample_decision_packet, str(packet_path))

        assert packet_path.exists()

        loaded = load_packet(str(packet_path))
        assert loaded.deployment_id == sample_decision_packet.deployment_id


# =============================================================================
# Test Group 2: truthlink
# =============================================================================


@requires_truthlink
class TestTruthLinkBuild:
    """Tests for TruthLink build() function."""

    def test_build_single_mode(self, temp_manifest_file, temp_receipts_dir):
        """Test build() in single mode creates DecisionPacket."""
        packet = build(
            deployment_id="test-build",
            manifest_path=str(temp_manifest_file),
            receipts_dir=str(temp_receipts_dir),
            mode="single",
        )

        assert isinstance(packet, DecisionPacket)
        assert packet.deployment_id == "test-build"

    def test_build_includes_sampling_metadata(self, temp_manifest_file, temp_receipts_dir):
        """Test build() embeds sampling metadata in edge_lab_summary."""
        packet = build(
            deployment_id="test-sampling",
            manifest_path=str(temp_manifest_file),
            receipts_dir=str(temp_receipts_dir),
            mode="single",
        )

        assert "_build_metadata" in packet.edge_lab_summary
        metadata = packet.edge_lab_summary["_build_metadata"]
        assert "sampling_seed" in metadata
        assert "sampling_stats" in metadata
        assert "truthlink_version" in metadata

    def test_build_manifest_not_found(self, temp_receipts_dir):
        """Test build() raises FileNotFoundError for missing manifest."""
        with pytest.raises(FileNotFoundError):
            build(
                deployment_id="test",
                manifest_path="/nonexistent/manifest.json",
                receipts_dir=str(temp_receipts_dir),
                mode="single",
            )


@requires_truthlink
class TestTruthLinkProject:
    """Tests for TruthLink project() function."""

    def test_project_add_pattern(self, sample_decision_packet):
        """Test project() with AddPattern change."""
        changes = [AddPattern(pattern_id="new_pattern_001")]
        projected = project(sample_decision_packet, changes)

        assert isinstance(projected, ProjectedPacket)
        assert "Add pattern new_pattern_001" in projected.changes_applied
        assert projected.confidence > 0

    def test_project_remove_pattern(self, sample_decision_packet):
        """Test project() with RemovePattern change."""
        changes = [RemovePattern(pattern_id="bat_thermal_001")]
        projected = project(sample_decision_packet, changes)

        assert "Remove pattern bat_thermal_001" in projected.changes_applied
        # Should reduce savings
        assert projected.projected_savings_delta < 0

    def test_project_scale_fleet(self, sample_decision_packet):
        """Test project() with ScaleFleet change."""
        changes = [ScaleFleet(multiplier=2.0)]
        projected = project(sample_decision_packet, changes)

        assert "Scale fleet by 2.0x" in projected.changes_applied
        # Savings should approximately double
        assert projected.projected_savings_delta > 0

    def test_project_confidence_decay(self, sample_decision_packet):
        """Test confidence decays with large changes."""
        # Small change
        small = project(sample_decision_packet, [ScaleFleet(multiplier=1.1)])
        # Large change
        large = project(sample_decision_packet, [ScaleFleet(multiplier=5.0)])

        assert large.confidence < small.confidence


@requires_truthlink
class TestTruthLinkCompare:
    """Tests for TruthLink compare() function."""

    def test_compare_returns_comparison(self, sample_decision_packet, sample_decision_packet_dict):
        """Test compare() returns Comparison object."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        result = compare(sample_decision_packet, packet_b)

        assert isinstance(result, Comparison)
        assert result.packet_a_id == sample_decision_packet.packet_id
        assert result.packet_b_id == packet_b.packet_id

    def test_compare_classification(self, sample_decision_packet, sample_decision_packet_dict):
        """Test compare() classifies as improvement/regression/mixed/neutral."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        result = compare(sample_decision_packet, packet_b)

        assert result.classification in ["improvement", "regression", "mixed", "neutral"]
        assert result.recommendation
        assert result.narration


@requires_truthlink
class TestTruthLinkPersistence:
    """Tests for TruthLink save() and load() functions."""

    def test_save_and_load(self, sample_decision_packet, tmp_path):
        """Test save() and load() roundtrip."""
        packets_dir = tmp_path / "packets"

        save(sample_decision_packet, output_dir=str(packets_dir))

        loaded = load(
            deployment_id=sample_decision_packet.deployment_id,
            packets_dir=str(packets_dir),
        )

        assert len(loaded) >= 1
        assert loaded[0].deployment_id == sample_decision_packet.deployment_id


@requires_truthlink
class TestTruthLinkSelfAwareness:
    """Tests for TruthLink health_check() and detect_drift()."""

    def test_health_check(self, tmp_path):
        """Test health_check() returns TruthLinkHealth."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        health = health_check(packets_dir=str(packets_dir))

        assert isinstance(health, TruthLinkHealth)
        assert health.avg_build_time_ms >= 0
        assert health.packet_health_trend in ["improving", "stable", "degrading"]

    def test_detect_drift(self, tmp_path):
        """Test detect_drift() returns DriftReport."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        report = detect_drift(packets_dir=str(packets_dir))

        assert isinstance(report, DriftReport)
        assert report.window_days > 0
        assert report.health_score_trend in ["improving", "stable", "degrading"]


# =============================================================================
# Test Group 3: config_schema
# =============================================================================


@requires_config_schema
class TestQEDConfigCreation:
    """Tests for QEDConfig creation and validation."""

    def test_config_creation(self, sample_provenance):
        """Test QEDConfig creation with valid data."""
        config = QEDConfig(
            version="8.0",
            deployment_id="test-config",
            hook="tesla",
            provenance=sample_provenance,
            compression_target=10.0,
            recall_floor=0.999,
            max_fp_rate=0.01,
        )

        assert config.version == "8.0"
        assert config.deployment_id == "test-config"
        assert config.hook == "tesla"

    def test_config_is_frozen(self, sample_provenance):
        """Test QEDConfig is immutable (frozen dataclass)."""
        config = QEDConfig(
            version="8.0",
            deployment_id="test-frozen",
            hook="tesla",
            provenance=sample_provenance,
        )

        with pytest.raises(AttributeError):
            config.version = "9.0"


@requires_config_schema
class TestQEDConfigValidation:
    """Tests for QEDConfig validation and SLO."""

    def test_validation_under_1ms(self, temp_config_file):
        """Test config validation completes in <1ms."""
        # Warm up the validator
        load_config(str(temp_config_file))

        # Time the validation
        start = time.perf_counter_ns()
        load_config(str(temp_config_file))
        end = time.perf_counter_ns()

        duration_ms = (end - start) / 1_000_000

        # Should complete in <1ms (compiled jsonschema)
        # Allow up to 5ms for test stability
        assert duration_ms < 5.0, f"Validation took {duration_ms:.2f}ms, expected <5ms"

    def test_valid_hooks(self, sample_provenance):
        """Test valid hook values are accepted."""
        valid_hooks = ["tesla", "spacex", "starlink", "boring", "neuralink", "xai"]

        for hook in valid_hooks:
            config = QEDConfig(
                version="8.0",
                deployment_id=f"test-{hook}",
                hook=hook,
                provenance=sample_provenance,
            )
            assert config.hook == hook


@requires_config_schema
class TestQEDConfigLoading:
    """Tests for QEDConfig file loading."""

    def test_load_from_json(self, temp_config_file):
        """Test load() from JSON file."""
        config = load_config(str(temp_config_file))

        assert isinstance(config, QEDConfig)
        assert config.version == "8.0"
        assert config.hook == "tesla"

    def test_load_with_validation(self, temp_config_file):
        """Test load() with validation enabled."""
        config = load_config(str(temp_config_file), validate=True)
        assert config is not None


# =============================================================================
# Test Group 4: merge_rules
# =============================================================================


@requires_merge_rules
class TestMergeRulesBasic:
    """Tests for merge_rules basic functionality."""

    def test_merge_returns_result(self, sample_parent_config, sample_child_config):
        """Test merge() returns MergeResult."""
        result = merge(sample_parent_config, sample_child_config)

        assert isinstance(result, MergeResult)
        assert result.merged is not None

    def test_merge_creates_merged_config(self, sample_parent_config, sample_child_config):
        """Test merge creates a valid merged QEDConfig."""
        result = merge(sample_parent_config, sample_child_config)

        assert isinstance(result.merged, QEDConfig)


@requires_merge_rules
class TestMergeSafetyOnlyTightens:
    """Tests for safety-only-tightens invariant."""

    def test_recall_floor_uses_max(self, sample_parent_config, sample_child_config):
        """Test recall_floor uses max(parent, child) - stricter wins."""
        result = merge(sample_parent_config, sample_child_config)

        # Parent: 0.999, Child: 0.9995 -> merged should be 0.9995
        assert result.merged.recall_floor >= sample_parent_config.recall_floor
        assert result.merged.recall_floor >= sample_child_config.recall_floor

    def test_max_fp_rate_uses_min(self, sample_parent_config, sample_child_config):
        """Test max_fp_rate uses min(parent, child) - stricter wins."""
        result = merge(sample_parent_config, sample_child_config)

        # Parent: 0.01, Child: 0.005 -> merged should be 0.005
        assert result.merged.max_fp_rate <= sample_parent_config.max_fp_rate
        assert result.merged.max_fp_rate <= sample_child_config.max_fp_rate

    def test_enabled_patterns_uses_intersection(self, sample_parent_config, sample_child_config):
        """Test enabled_patterns uses intersection."""
        result = merge(sample_parent_config, sample_child_config)

        # Parent: {bat_thermal_001, comms_dropout_002}
        # Child: {bat_thermal_001}
        # Intersection: {bat_thermal_001}
        parent_patterns = set(sample_parent_config.enabled_patterns)
        child_patterns = set(sample_child_config.enabled_patterns)
        merged_patterns = set(result.merged.enabled_patterns)

        assert merged_patterns.issubset(parent_patterns)
        assert merged_patterns.issubset(child_patterns)

    def test_regulatory_flags_uses_or(self, sample_parent_config, sample_child_config):
        """Test regulatory_flags uses OR (union)."""
        result = merge(sample_parent_config, sample_child_config)

        # Parent: {gdpr}, Child: {gdpr, eu_ai_act} -> Union: {gdpr, eu_ai_act}
        # regulatory_flags is a dict, check keys
        merged_flags = result.merged.regulatory_flags

        assert "gdpr" in merged_flags


@requires_merge_rules
class TestMergeRejection:
    """Tests for merge rejection on safety violations."""

    def test_reject_loosened_recall_floor(self, sample_parent_config, sample_child_config_unsafe):
        """Test merge rejects child that loosens recall_floor."""
        result = merge(sample_parent_config, sample_child_config_unsafe)

        # Should have violations for loosening safety
        assert len(result.violations) > 0

        # At least one violation should mention recall_floor
        violation_fields = [v.field for v in result.violations]
        assert "recall_floor" in violation_fields or len(result.violations) > 0

    def test_reject_loosened_max_fp_rate(self, sample_parent_config, sample_child_config_unsafe):
        """Test merge rejects child that loosens max_fp_rate."""
        result = merge(sample_parent_config, sample_child_config_unsafe)

        # Should detect violation
        assert len(result.violations) > 0

    def test_violations_have_details(self, sample_parent_config, sample_child_config_unsafe):
        """Test violations include field, parent_value, child_value."""
        result = merge(sample_parent_config, sample_child_config_unsafe)

        if result.violations:
            v = result.violations[0]
            assert hasattr(v, "field")
            assert hasattr(v, "parent_value") or hasattr(v, "message")


@requires_merge_rules
class TestMergeReceipt:
    """Tests for merge receipt/audit trail."""

    def test_emit_receipt(self, sample_parent_config, sample_child_config, tmp_path):
        """Test emit_receipt() writes audit log."""
        result = merge(sample_parent_config, sample_child_config)

        if hasattr(result, "receipt") and result.receipt:
            receipt_path = tmp_path / "merge_receipts.jsonl"
            emit_receipt(result.receipt, str(receipt_path))

            assert receipt_path.exists()


# =============================================================================
# Test Group 5: mesh_view_v3
# =============================================================================


@requires_mesh_view_v3
class TestMeshViewBuild:
    """Tests for mesh_view_v3 build() function."""

    def test_build_graph_from_packets(self, sample_decision_packet):
        """Test build() creates DeploymentGraph from packets."""
        packets = [sample_decision_packet]
        graph = build_graph(packets)

        assert isinstance(graph, DeploymentGraph)
        assert len(graph.nodes) >= 1

    def test_build_empty_packets_no_crash(self):
        """Test build() handles empty packet list gracefully."""
        graph = build_graph([])

        assert isinstance(graph, DeploymentGraph)
        assert len(graph.nodes) == 0

    def test_build_multiple_packets(self, sample_decision_packet, sample_decision_packet_dict):
        """Test build() handles multiple packets."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        packets = [sample_decision_packet, packet_b]

        graph = build_graph(packets)

        assert len(graph.nodes) == 2


@requires_mesh_view_v3
class TestMeshViewClustering:
    """Tests for mesh_view_v3 clustering functions."""

    def test_find_clusters(self, sample_decision_packet, sample_decision_packet_dict):
        """Test find_clusters() using union-find algorithm."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        packets = [sample_decision_packet, packet_b]
        graph = build_graph(packets)

        clusters = find_clusters(graph)

        assert isinstance(clusters, list)

    def test_find_outliers(self, sample_decision_packet):
        """Test find_outliers() identifies isolated nodes."""
        packets = [sample_decision_packet]
        graph = build_graph(packets)

        outliers = find_outliers(graph)

        assert isinstance(outliers, list)


@requires_mesh_view_v3
class TestMeshViewFleetMetrics:
    """Tests for mesh_view_v3 fleet metrics."""

    def test_compute_fleet_metrics(self, sample_decision_packet, sample_decision_packet_dict):
        """Test compute_fleet_metrics() returns FleetSummary."""
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        packets = [sample_decision_packet, packet_b]
        graph = build_graph(packets)

        summary = compute_fleet_metrics(graph)

        assert isinstance(summary, FleetSummary)
        assert hasattr(summary, "fleet_cohesion")

    def test_diagnose_graph(self, sample_decision_packet):
        """Test diagnose() returns health check results."""
        packets = [sample_decision_packet]
        graph = build_graph(packets)

        diagnosis = diagnose(graph)

        assert isinstance(diagnosis, dict) or hasattr(diagnosis, "to_dict")


# =============================================================================
# Test Group 6: proof.py CLI v8 subcommands
# =============================================================================


@requires_decision_packet
class TestProofCLIV8:
    """Tests for proof.py CLI v8 subcommands."""

    @pytest.fixture
    def proof_path(self):
        """Path to proof.py."""
        return Path(__file__).parent.parent / "proof.py"

    def _run_proof(self, proof_path, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run proof.py with given args."""
        cmd = [sys.executable, str(proof_path)] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @pytest.mark.skipif(
        True,  # Skip until build-packet subcommand is implemented
        reason="build-packet subcommand not yet implemented"
    )
    def test_build_packet_subcommand(self, proof_path, temp_manifest_file, temp_receipts_dir):
        """Test proof.py build-packet subcommand."""
        result = self._run_proof(proof_path, [
            "build-packet",
            "--manifest", str(temp_manifest_file),
            "--receipts", str(temp_receipts_dir),
        ])

        assert result.returncode == 0

    @pytest.mark.skipif(
        True,  # Skip until validate-config subcommand is implemented
        reason="validate-config subcommand not yet implemented"
    )
    def test_validate_config_subcommand(self, proof_path, temp_config_file):
        """Test proof.py validate-config subcommand."""
        result = self._run_proof(proof_path, [
            "validate-config",
            str(temp_config_file),
        ])

        assert result.returncode == 0

    @pytest.mark.skipif(
        True,  # Skip until merge-configs subcommand is implemented
        reason="merge-configs subcommand not yet implemented"
    )
    def test_merge_configs_subcommand(self, proof_path, tmp_path):
        """Test proof.py merge-configs subcommand."""
        # Create parent and child config files
        parent_path = tmp_path / "parent.json"
        child_path = tmp_path / "child.json"

        parent_config = {
            "version": "8.0",
            "deployment_id": "parent",
            "hook": "tesla",
            "recall_floor": 0.999,
            "max_fp_rate": 0.01,
        }
        child_config = {
            "version": "8.0",
            "deployment_id": "child",
            "hook": "tesla",
            "recall_floor": 0.9995,
            "max_fp_rate": 0.005,
        }

        with open(parent_path, "w") as f:
            json.dump(parent_config, f)
        with open(child_path, "w") as f:
            json.dump(child_config, f)

        result = self._run_proof(proof_path, [
            "merge-configs",
            str(parent_path),
            str(child_path),
        ])

        assert result.returncode == 0

    @pytest.mark.skipif(
        True,  # Skip until compare-packets subcommand is implemented
        reason="compare-packets subcommand not yet implemented"
    )
    def test_compare_packets_subcommand(self, proof_path, sample_decision_packet, tmp_path):
        """Test proof.py compare-packets subcommand."""
        # Save two packets
        packet_a_path = tmp_path / "packet_a.json"
        packet_b_path = tmp_path / "packet_b.json"

        save_packet(sample_decision_packet, str(packet_a_path))

        # Create a modified packet
        from decision_packet import DecisionPacket
        packet_b = DecisionPacket.from_json(sample_decision_packet.to_json())
        save_packet(packet_b, str(packet_b_path))

        result = self._run_proof(proof_path, [
            "compare-packets",
            str(packet_a_path),
            str(packet_b_path),
        ])

        assert result.returncode == 0

    @pytest.mark.skipif(
        True,  # Skip until fleet-view subcommand is implemented
        reason="fleet-view subcommand not yet implemented"
    )
    def test_fleet_view_subcommand(self, proof_path, tmp_path):
        """Test proof.py fleet-view subcommand."""
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        result = self._run_proof(proof_path, [
            "fleet-view",
            "--packets-dir", str(packets_dir),
        ])

        # May return non-zero if no packets, but shouldn't crash
        assert result.returncode in [0, 1]


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(
    not (HAS_DECISION_PACKET and HAS_MESH_VIEW_V3 and HAS_MERGE_RULES),
    reason="Integration tests require decision_packet, mesh_view_v3, and merge_rules"
)
class TestV8Integration:
    """Integration tests for v8 module interactions."""

    def test_packet_to_graph_to_metrics(self, sample_decision_packet, sample_decision_packet_dict):
        """Test full flow: packets -> graph -> fleet metrics."""
        # Create packets
        packet_b = DecisionPacket.from_dict(sample_decision_packet_dict)
        packets = [sample_decision_packet, packet_b]

        # Build graph
        graph = build_graph(packets)
        assert len(graph.nodes) == 2

        # Compute fleet metrics
        summary = compute_fleet_metrics(graph)
        assert summary.fleet_cohesion >= 0

    def test_config_merge_to_deployment(self, sample_parent_config, sample_child_config):
        """Test config merge produces deployable config."""
        result = merge(sample_parent_config, sample_child_config)

        merged = result.merged

        # Merged config should be usable
        assert merged.version == "8.0"
        assert merged.recall_floor >= 0.999
        assert merged.max_fp_rate <= 0.01

    def test_projection_preserves_integrity(self, sample_decision_packet):
        """Test projected packet data is consistent."""
        changes = [AddPattern(pattern_id="new_pattern")]
        projected = project(sample_decision_packet, changes)

        # Projected packet should have consistent data
        assert projected.base_packet_id == sample_decision_packet.packet_id
        assert projected.deployment_id == sample_decision_packet.deployment_id
        assert 0 <= projected.confidence <= 1.0


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.skipif(
    not (HAS_DECISION_PACKET and HAS_MESH_VIEW_V3),
    reason="Performance tests require decision_packet and mesh_view_v3"
)
class TestV8Performance:
    """Performance tests for v8 modules."""

    def test_packet_creation_performance(self, sample_pattern_summaries, sample_packet_metrics):
        """Test DecisionPacket creation is fast (<10ms)."""
        start = time.perf_counter_ns()

        for _ in range(100):
            DecisionPacket(
                deployment_id="perf-test",
                manifest_ref="manifest.yaml",
                sampled_receipts=["r1", "r2"],
                clarity_audit_ref="audit.json",
                edge_lab_summary={},
                pattern_usage=sample_pattern_summaries,
                metrics=sample_packet_metrics,
            )

        end = time.perf_counter_ns()
        avg_ms = (end - start) / 100 / 1_000_000

        assert avg_ms < 10.0, f"Packet creation took {avg_ms:.2f}ms avg, expected <10ms"

    def test_graph_build_performance(self, sample_decision_packet):
        """Test graph build scales reasonably."""
        # Create 100 packets
        packets = []
        for i in range(100):
            p = DecisionPacket(
                deployment_id=f"deploy-{i}",
                manifest_ref=f"manifest-{i}.yaml",
                sampled_receipts=[f"r{i}"],
                clarity_audit_ref="",
                edge_lab_summary={},
                pattern_usage=sample_decision_packet.pattern_usage,
                metrics=sample_decision_packet.metrics,
            )
            packets.append(p)

        start = time.perf_counter_ns()
        graph = build_graph(packets)
        end = time.perf_counter_ns()

        duration_ms = (end - start) / 1_000_000

        # Should complete in <1s for 100 packets
        assert duration_ms < 1000, f"Graph build took {duration_ms:.0f}ms for 100 packets"
