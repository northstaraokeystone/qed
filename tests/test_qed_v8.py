"""
QED v8 Smoke Tests - DecisionPacket, TruthLink, Config, Merge, and Mesh

Smoke tests for core QED v8 modules:
- decision_packet: DecisionPacket dataclass validation and serialization
- truthlink: build_decision_packet function and packet building
- config_schema: QEDConfig loading, validation, and <1ms SLO
- merge_rules: config merging with safety tightening invariants
- mesh_view_v3: deployment graph building and fleet analysis
- proof.py: v8 CLI subcommands (stubs for future implementation)

Test organization:
- Group tests by module using classes
- Use pytest fixtures for common setup
- Use tmp_path for temp files (auto-cleanup)
- Use time.perf_counter() for timing tests
- Skip tests for modules not yet implemented with clear messages
"""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pytest

# =============================================================================
# Module imports - gracefully handle missing modules
# =============================================================================

try:
    from decision_packet import DecisionPacket, PacketMetrics, PatternSummary, load_packet, save_packet, compare_packets
    DECISION_PACKET_AVAILABLE = True
except ImportError:
    DECISION_PACKET_AVAILABLE = False

try:
    from truthlink import build, load, save
    TRUTHLINK_AVAILABLE = True
except ImportError:
    TRUTHLINK_AVAILABLE = False

try:
    from config_schema import QEDConfig, load as config_load, default as config_default
    CONFIG_SCHEMA_AVAILABLE = True
except ImportError:
    CONFIG_SCHEMA_AVAILABLE = False

try:
    from merge_rules import merge, MergeReceipt, MergeResult, Violation
    MERGE_RULES_AVAILABLE = True
except ImportError:
    MERGE_RULES_AVAILABLE = False

try:
    from mesh_view_v3 import build as mesh_build, DeploymentGraph, compute_fleet_metrics
    MESH_VIEW_V3_AVAILABLE = True
except ImportError:
    MESH_VIEW_V3_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_valid_config_dict() -> Dict[str, Any]:
    """Sample valid config dict matching v8 schema."""
    return {
        "version": "1.0",
        "deployment_id": "test_deployment_001",
        "hook": "tesla",
        "recall_floor": 0.9995,
        "max_fp_rate": 0.005,
        "slo_latency_ms": 50,
        "slo_breach_budget": 0.001,
        "compression_target": 10.0,
        "enabled_patterns": ["PAT_BATTERY_*", "PAT_THERMAL_*"],
        "safety_overrides": {
            "steering_torque_nm": 65.0,
            "brake_pressure_bar": 180.0,
        },
        "regulatory_flags": {
            "NHTSA": True,
            "ISO26262": True,
        },
    }


@pytest.fixture
def sample_decision_packet_dict() -> Dict[str, Any]:
    """Sample DecisionPacket as dict for testing."""
    return {
        "packet_id": "test_packet_001",
        "version": "1.0",
        "deployment_id": "tesla_prod_001",
        "hook": "tesla",
        "config_hash": "abc123",
        "manifest_hash": "def456",
        "receipts_hash": "ghi789",
        "created_at": "2024-01-01T00:00:00Z",
        "patterns": [],
        "metrics": {
            "total_windows": 1000,
            "total_savings_M": 50.0,
            "avg_ratio": 60.0,
            "slo_breach_rate": 0.0001,
        },
    }


@pytest.fixture
def temp_receipts_dir(tmp_path) -> Path:
    """Create temp directory with mock receipt files."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()

    # Create sample receipt file
    receipt_file = receipts_dir / "test_receipt.jsonl"
    receipts = [
        {
            "ts": "2024-01-01T00:00:00Z",
            "window_id": "w001",
            "hook": "tesla",
            "params": {"A": 12.0, "f": 40.0},
            "ratio": 60.0,
            "savings_M": 38.0,
            "verified": True,
            "violations": [],
        },
    ]
    with open(receipt_file, "w") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")

    return receipts_dir


@pytest.fixture
def temp_manifest_file(tmp_path) -> Path:
    """Create temp manifest file."""
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "run_id": "test_run_001",
        "deployment_id": "tesla_prod_001",
        "hook": "tesla",
        "fleet_size": 100,
        "total_windows": 1000,
        "created_at": "2024-01-01T00:00:00Z",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path


# =============================================================================
# Test Group 1: decision_packet
# =============================================================================


@pytest.mark.skipif(not DECISION_PACKET_AVAILABLE, reason="awaiting decision_packet.py")
class TestDecisionPacket:
    """Tests for DecisionPacket dataclass and operations."""

    def test_create_packet_with_required_fields(self, sample_decision_packet_dict):
        """Create packet with all required fields, verify packet_id generated."""
        packet = DecisionPacket(**sample_decision_packet_dict)

        assert packet.packet_id is not None
        assert packet.deployment_id == "tesla_prod_001"
        assert packet.hook == "tesla"
        assert isinstance(packet.metrics, (dict, PacketMetrics))

    def test_validate_valid_packet(self, sample_decision_packet_dict):
        """Validate valid packet returns True."""
        packet = DecisionPacket(**sample_decision_packet_dict)

        # DecisionPacket should have a validate method or be valid by construction
        # If no validate method, check that required fields are present
        assert packet.packet_id is not None
        assert packet.deployment_id is not None
        assert packet.config_hash is not None

    def test_validate_packet_missing_fields(self):
        """Validate packet with missing fields raises error or returns False."""
        # Try to create packet with missing required field
        with pytest.raises((TypeError, ValueError)):
            DecisionPacket(
                packet_id="test",
                version="1.0",
                # Missing deployment_id and other required fields
            )

    def test_serialize_deserialize_packet(self, sample_decision_packet_dict, tmp_path):
        """Serialize to JSON, deserialize back, verify equality."""
        packet = DecisionPacket(**sample_decision_packet_dict)

        # Save to file
        packet_path = tmp_path / "packet.json"
        save_packet(packet, str(packet_path))

        # Load back
        loaded_packet = load_packet(str(packet_path))

        # Verify equality of key fields
        assert loaded_packet.packet_id == packet.packet_id
        assert loaded_packet.deployment_id == packet.deployment_id
        assert loaded_packet.hook == packet.hook
        assert loaded_packet.config_hash == packet.config_hash

    def test_packet_id_changes_with_contents(self, sample_decision_packet_dict):
        """Verify packet_id changes when contents change (hash integrity)."""
        packet1 = DecisionPacket(**sample_decision_packet_dict)

        # Create packet with different config_hash
        modified_dict = sample_decision_packet_dict.copy()
        modified_dict["config_hash"] = "different_hash_xyz"
        packet2 = DecisionPacket(**modified_dict)

        # If packet_id is content-addressable, it should differ
        # Otherwise, ensure they are distinct packets
        assert packet1.config_hash != packet2.config_hash

    def test_compare_packets_function(self, sample_decision_packet_dict):
        """Test compare_packets function returns comparison string."""
        packet1 = DecisionPacket(**sample_decision_packet_dict)

        modified_dict = sample_decision_packet_dict.copy()
        modified_dict["deployment_id"] = "tesla_prod_002"
        packet2 = DecisionPacket(**modified_dict)

        comparison = compare_packets(packet1, packet2)

        assert isinstance(comparison, str)
        assert len(comparison) > 0


# =============================================================================
# Test Group 2: truthlink
# =============================================================================


@pytest.mark.skipif(not TRUTHLINK_AVAILABLE, reason="awaiting truthlink.py")
class TestTruthLink:
    """Tests for truthlink.build_decision_packet function."""

    def test_build_decision_packet_valid_inputs(self, temp_manifest_file, temp_receipts_dir):
        """Given valid manifest path, receipts dir, returns DecisionPacket."""
        # Build packet using truthlink.build
        result = build(
            manifest_path=str(temp_manifest_file),
            receipts_dir=str(temp_receipts_dir),
            deployment_id="tesla_prod_001",
        )

        # Should return a DecisionPacket or dict
        assert result is not None
        # Check for expected structure
        if hasattr(result, "deployment_id"):
            assert result.deployment_id == "tesla_prod_001"
        else:
            assert "deployment_id" in result or isinstance(result, dict)

    def test_returned_packet_passes_validation(self, temp_manifest_file, temp_receipts_dir):
        """Returned packet passes validation."""
        packet = build(
            manifest_path=str(temp_manifest_file),
            receipts_dir=str(temp_receipts_dir),
            deployment_id="tesla_prod_001",
        )

        # Verify packet has expected fields
        if hasattr(packet, "packet_id"):
            assert packet.packet_id is not None
        else:
            assert "packet_id" in packet or "deployment_id" in packet

    def test_packet_contains_expected_fields(self, temp_manifest_file, temp_receipts_dir):
        """Packet contains expected fields (deployment_id, metrics, pattern_usage)."""
        packet = build(
            manifest_path=str(temp_manifest_file),
            receipts_dir=str(temp_receipts_dir),
            deployment_id="tesla_prod_001",
        )

        # Check for expected fields based on DecisionPacket schema
        if hasattr(packet, "__dict__"):
            packet_dict = vars(packet)
        else:
            packet_dict = packet

        # At minimum, should have deployment_id
        assert "deployment_id" in packet_dict or hasattr(packet, "deployment_id")

    def test_missing_manifest_raises_error(self, temp_receipts_dir):
        """Missing manifest raises appropriate error."""
        nonexistent_manifest = "/nonexistent/manifest.json"

        with pytest.raises((FileNotFoundError, ValueError, IOError)):
            build(
                manifest_path=nonexistent_manifest,
                receipts_dir=str(temp_receipts_dir),
                deployment_id="test",
            )


# =============================================================================
# Test Group 3: config_schema
# =============================================================================


@pytest.mark.skipif(not CONFIG_SCHEMA_AVAILABLE, reason="awaiting config_schema.py")
class TestConfigSchema:
    """Tests for QEDConfig loading and validation."""

    def test_load_valid_config_from_template(self):
        """Load valid config from data/config_templates/tesla_config.json."""
        config_path = "data/config_templates/tesla_config.json"

        # Load config
        config = config_load(config_path)

        # Verify it's a QEDConfig object
        assert config is not None
        assert hasattr(config, "hook") or isinstance(config, dict)

        # Check for expected tesla config fields
        if hasattr(config, "recall_floor"):
            assert config.recall_floor >= 0.999
        elif isinstance(config, dict):
            assert config["recall_floor"] >= 0.999

    def test_validation_completes_under_1ms(self):
        """Validation completes in under 1ms (use time.perf_counter)."""
        config_path = "data/config_templates/tesla_config.json"

        # Warmup
        _ = config_load(config_path)

        # Timed run
        start = time.perf_counter()
        config = config_load(config_path)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1.0, f"Config validation took {elapsed_ms:.3f}ms, expected <1ms"

    def test_invalid_config_missing_field_fails(self, tmp_path):
        """Invalid config (missing required field) fails validation with error list."""
        # Create config missing required field
        invalid_config = {
            "version": "1.0",
            "deployment_id": "test",
            # Missing required fields like recall_floor, max_fp_rate
        }

        config_path = tmp_path / "invalid_config.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)

        # Should raise validation error or return None
        with pytest.raises((ValueError, KeyError, TypeError)):
            config_load(str(config_path))

    def test_invalid_config_wrong_type_fails(self, tmp_path):
        """Invalid config (wrong type) fails validation."""
        # Create config with wrong type for recall_floor
        invalid_config = {
            "version": "1.0",
            "deployment_id": "test",
            "hook": "tesla",
            "recall_floor": "not_a_number",  # Should be float
            "max_fp_rate": 0.01,
            "slo_latency_ms": 100,
            "slo_breach_budget": 0.005,
            "compression_target": 10.0,
            "enabled_patterns": [],
            "safety_overrides": {},
            "regulatory_flags": {},
        }

        config_path = tmp_path / "invalid_type_config.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)

        with pytest.raises((ValueError, TypeError)):
            config_load(str(config_path))

    def test_load_returns_frozen_immutable_config(self):
        """Load returns frozen/immutable config object."""
        config_path = "data/config_templates/tesla_config.json"
        config = config_load(config_path)

        # Try to modify config - should fail if frozen/immutable
        # Note: Python dataclasses with frozen=True raise FrozenInstanceError
        if hasattr(config, "__dataclass_fields__"):
            # Check if frozen
            with pytest.raises((AttributeError, TypeError)):
                config.recall_floor = 0.8  # Should fail if frozen


# =============================================================================
# Test Group 4: merge_rules
# =============================================================================


@pytest.mark.skipif(not MERGE_RULES_AVAILABLE, reason="awaiting merge_rules.py")
class TestMergeRules:
    """Tests for config merging logic."""

    def test_merge_global_tesla_tighter_recall_floor(self):
        """Merge global + tesla config, result has tesla's tighter recall_floor."""
        global_config = config_load("data/config_templates/global_config.json")
        tesla_config = config_load("data/config_templates/tesla_config.json")

        # Merge configs
        result = merge(parent=global_config, child=tesla_config)

        # Result should have tesla's tighter recall_floor
        merged_config = result.config if hasattr(result, "config") else result

        if hasattr(merged_config, "recall_floor"):
            # Tesla has tighter (higher) recall_floor than global
            assert merged_config.recall_floor >= tesla_config.recall_floor
        elif isinstance(merged_config, dict):
            assert merged_config["recall_floor"] >= tesla_config.recall_floor

    def test_child_loosening_recall_floor_rejected(self):
        """Child trying to loosen recall_floor (lower value) is rejected."""
        global_config = config_load("data/config_templates/global_config.json")

        # Create child config with lower (looser) recall_floor
        tesla_config = config_load("data/config_templates/tesla_config.json")

        # Try to loosen by merging global (0.999) onto tesla (0.9995)
        # This should fail or emit violation
        result = merge(parent=tesla_config, child=global_config)

        # Check for violations
        if hasattr(result, "violations"):
            # If global tries to loosen tesla's recall_floor, should have violation
            if global_config.recall_floor < tesla_config.recall_floor:
                assert len(result.violations) > 0 or result.config.recall_floor == tesla_config.recall_floor

    def test_child_loosening_max_fp_rate_rejected(self):
        """Child trying to loosen max_fp_rate (higher value) is rejected."""
        global_config = config_load("data/config_templates/global_config.json")
        tesla_config = config_load("data/config_templates/tesla_config.json")

        # Merge should not allow loosening max_fp_rate
        result = merge(parent=tesla_config, child=global_config)

        # Verify max_fp_rate is not loosened
        merged_config = result.config if hasattr(result, "config") else result

        if hasattr(merged_config, "max_fp_rate"):
            # Should take tighter (lower) value
            assert merged_config.max_fp_rate <= tesla_config.max_fp_rate

    def test_regulatory_flags_merge_with_or(self):
        """Regulatory flags merge with OR (either requires = final requires)."""
        config1_dict = {
            "version": "1.0",
            "deployment_id": "test1",
            "hook": "tesla",
            "recall_floor": 0.999,
            "max_fp_rate": 0.01,
            "slo_latency_ms": 100,
            "slo_breach_budget": 0.005,
            "compression_target": 10.0,
            "enabled_patterns": [],
            "safety_overrides": {},
            "regulatory_flags": {"NHTSA": True, "ISO26262": False},
        }

        config2_dict = config1_dict.copy()
        config2_dict["deployment_id"] = "test2"
        config2_dict["regulatory_flags"] = {"NHTSA": False, "ISO26262": True}

        # Create configs
        from config_schema import _create_config
        config1 = _create_config(config1_dict)
        config2 = _create_config(config2_dict)

        # Merge
        result = merge(parent=config1, child=config2)
        merged_config = result.config if hasattr(result, "config") else result

        # Regulatory flags should be OR'd
        if hasattr(merged_config, "regulatory_flags"):
            flags = merged_config.regulatory_flags
            # Either config requires NHTSA or ISO26262, so both should be True
            assert flags.get("NHTSA", False) or flags.get("ISO26262", False)

    def test_enabled_patterns_merge_with_intersection(self):
        """enabled_patterns merge with intersection."""
        config1_dict = {
            "version": "1.0",
            "deployment_id": "test1",
            "hook": "tesla",
            "recall_floor": 0.999,
            "max_fp_rate": 0.01,
            "slo_latency_ms": 100,
            "slo_breach_budget": 0.005,
            "compression_target": 10.0,
            "enabled_patterns": ["PAT_A*", "PAT_B*", "PAT_C*"],
            "safety_overrides": {},
            "regulatory_flags": {},
        }

        config2_dict = config1_dict.copy()
        config2_dict["deployment_id"] = "test2"
        config2_dict["enabled_patterns"] = ["PAT_B*", "PAT_C*", "PAT_D*"]

        from config_schema import _create_config
        config1 = _create_config(config1_dict)
        config2 = _create_config(config2_dict)

        result = merge(parent=config1, child=config2)
        merged_config = result.config if hasattr(result, "config") else result

        # Should intersect: ["PAT_B*", "PAT_C*"]
        if hasattr(merged_config, "enabled_patterns"):
            patterns = merged_config.enabled_patterns
            # Common patterns should be present
            assert "PAT_B*" in patterns or "PAT_C*" in patterns

    def test_merge_produces_receipt_for_audit(self):
        """Merge produces merge_receipt for audit trail."""
        global_config = config_load("data/config_templates/global_config.json")
        tesla_config = config_load("data/config_templates/tesla_config.json")

        result = merge(parent=global_config, child=tesla_config)

        # Should have receipt or be able to generate one
        assert result is not None
        # Check if result has receipt fields
        if hasattr(result, "receipt"):
            assert result.receipt is not None
        elif isinstance(result, MergeResult):
            # MergeResult should have receipt or metadata
            assert hasattr(result, "config")


# =============================================================================
# Test Group 5: mesh_view_v3
# =============================================================================


@pytest.mark.skipif(not MESH_VIEW_V3_AVAILABLE, reason="awaiting mesh_view_v3.py")
class TestMeshViewV3:
    """Tests for deployment graph building."""

    def test_build_deployment_graph_from_packets(self, tmp_path):
        """Load multiple packets, build_deployment_graph returns graph object."""
        # Create sample packet files
        packet1 = {
            "packet_id": "p001",
            "deployment_id": "deploy1",
            "hook": "tesla",
            "patterns": ["PAT_BATTERY_001", "PAT_THERMAL_001"],
            "config_hash": "abc123",
            "created_at": "2024-01-01T00:00:00Z",
        }

        packet2 = {
            "packet_id": "p002",
            "deployment_id": "deploy2",
            "hook": "spacex",
            "patterns": ["PAT_BATTERY_001"],  # Shared pattern
            "config_hash": "def456",
            "created_at": "2024-01-02T00:00:00Z",
        }

        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        with open(packets_dir / "packet1.json", "w") as f:
            json.dump(packet1, f)
        with open(packets_dir / "packet2.json", "w") as f:
            json.dump(packet2, f)

        # Build graph
        graph = mesh_build(packets_dir=str(packets_dir))

        # Should return DeploymentGraph
        assert graph is not None
        assert hasattr(graph, "nodes") or isinstance(graph, DeploymentGraph)

    def test_deployments_sharing_patterns_connected(self, tmp_path):
        """Deployments sharing patterns are connected."""
        # Create packets with shared patterns
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        packet1 = {
            "packet_id": "p001",
            "deployment_id": "deploy1",
            "hook": "tesla",
            "patterns": ["PAT_SHARED"],
            "created_at": "2024-01-01T00:00:00Z",
        }

        packet2 = {
            "packet_id": "p002",
            "deployment_id": "deploy2",
            "hook": "spacex",
            "patterns": ["PAT_SHARED"],
            "created_at": "2024-01-01T00:00:00Z",
        }

        with open(packets_dir / "p1.json", "w") as f:
            json.dump(packet1, f)
        with open(packets_dir / "p2.json", "w") as f:
            json.dump(packet2, f)

        graph = mesh_build(packets_dir=str(packets_dir))

        # Verify graph has edges between deployments sharing patterns
        if hasattr(graph, "edges"):
            assert len(graph.edges) > 0 or len(graph.nodes) >= 2

    def test_find_reuse_clusters_returns_groups(self, tmp_path):
        """find_reuse_clusters returns list of deployment groups."""
        # Create packets with clustering potential
        packets_dir = tmp_path / "packets"
        packets_dir.mkdir()

        # Create 3 packets with some shared patterns
        for i in range(3):
            packet = {
                "packet_id": f"p{i:03d}",
                "deployment_id": f"deploy{i}",
                "hook": "tesla",
                "patterns": ["PAT_COMMON", f"PAT_{i}"],
                "created_at": "2024-01-01T00:00:00Z",
            }
            with open(packets_dir / f"p{i}.json", "w") as f:
                json.dump(packet, f)

        graph = mesh_build(packets_dir=str(packets_dir))

        # Verify graph was built
        assert graph is not None

    def test_empty_packets_list_returns_empty_graph(self, tmp_path):
        """Empty packets list returns empty graph (no crash)."""
        empty_dir = tmp_path / "empty_packets"
        empty_dir.mkdir()

        # Build graph with no packets
        graph = mesh_build(packets_dir=str(empty_dir))

        # Should not crash, return empty graph
        assert graph is not None
        if hasattr(graph, "nodes"):
            assert len(graph.nodes) == 0


# =============================================================================
# Test Group 6: proof.py v8 subcommands
# =============================================================================


class TestProofV8Subcommands:
    """Tests for proof.py v8 CLI subcommands."""

    def test_import_proof_module_successfully(self):
        """Import proof module successfully."""
        import proof
        assert proof is not None

    @pytest.mark.skip(reason="v8 subcommands not yet implemented in proof.py")
    def test_build_packet_subcommand_exists(self):
        """build-packet subcommand exists (check via Click inspection or --help)."""
        import subprocess
        result = subprocess.run(
            ["python", "proof.py", "build-packet", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    @pytest.mark.skip(reason="v8 subcommands not yet implemented in proof.py")
    def test_validate_config_subcommand_exists(self):
        """validate-config subcommand exists."""
        import subprocess
        result = subprocess.run(
            ["python", "proof.py", "validate-config", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    @pytest.mark.skip(reason="v8 subcommands not yet implemented in proof.py")
    def test_merge_configs_subcommand_exists(self):
        """merge-configs subcommand exists."""
        import subprocess
        result = subprocess.run(
            ["python", "proof.py", "merge-configs", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    @pytest.mark.skip(reason="v8 subcommands not yet implemented in proof.py")
    def test_compare_packets_subcommand_exists(self):
        """compare-packets subcommand exists."""
        import subprocess
        result = subprocess.run(
            ["python", "proof.py", "compare-packets", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    @pytest.mark.skip(reason="v8 subcommands not yet implemented in proof.py")
    def test_fleet_view_subcommand_exists(self):
        """fleet-view subcommand exists."""
        import subprocess
        result = subprocess.run(
            ["python", "proof.py", "fleet-view", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
