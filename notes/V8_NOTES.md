# QED v8 Notes: DecisionPackets, TruthLink, and Fleet Governance

**Version**: 8.0 | **Class**: packets_and_governance | **Domain**: deployment | **Date**: 2025-12-10

## Overview

QED v8 introduces DecisionPackets as self-validating deployment artifacts that bundle manifests, receipts, audits, and patterns into signable, diffable units. TruthLink fuses these packets from raw artifacts with streaming sampling and projection capabilities. Config schemas with merge rules enforce "safety only tightens" governance across deployment layers. mesh_view_v3 transforms packets into a queryable fleet topology.

## 1. What's New in v8

QED v8 is organized around three pillars:

1. **TruthLink DecisionPackets** make each deployment signable and comparable
   - Self-validating packets via SHA3 hash integrity checks
   - Embedded narratives for compliance auditors
   - Parent-child lineage chains for audit trails

2. **Config schema with merge rules** enforce governance across layers
   - Only tighten safety: recall_floor can only increase, max_fp_rate can only decrease
   - Intersection-based pattern restrictions: child can only restrict further
   - Regulatory OR: if either config requires a flag, merged requires it
   - Automatic repair suggestions for violations

3. **mesh_view_v3 shows cross-fleet pattern reuse**
   - Deployment graph with similarity-weighted edges
   - Cluster detection for pattern propagation
   - Temporal evolution tracking and stale deployment detection
   - Predictive recommendations for pattern adoption

## 2. DecisionPacket Schema

**File**: `decision_packet.py`

The DecisionPacket is the core audit artifact for QED v8 deployments. It is self-validating (packet_id = hash of contents), self-describing (generates human-readable narratives), and diffable (structured comparison between packets).

### Core Fields

```
packet_id: str                          # SHA3-256 hash (16 chars) of packet contents
deployment_id: str                      # Unique deployment identifier
timestamp: str                          # ISO UTC timestamp, auto-set at creation
manifest_ref: str                       # Path to deployment manifest (required)
sampled_receipts: List[str]            # IDs of receipt samples (configurable count, default 100)
clarity_audit_ref: str                 # Reference to ClarityClean audit receipt
edge_lab_summary: Dict                 # Summary from edge_lab_v2 simulation
pattern_usage: List[PatternSummary]    # Patterns deployed with metrics
metrics: PacketMetrics                 # Aggregate deployment metrics
exploit_coverage: float                # Fraction of patterns with exploit_grade=true
signature: Optional[str]               # Cryptographic signature for verification
parent_packet_id: Optional[str]        # Link to previous packet for lineage chain
```

### PatternSummary Schema

```
pattern_id: str                         # Pattern identifier
validation_recall: float                # Validation recall [0.0-1.0]
false_positive_rate: float              # False positive rate [0.0-1.0]
dollar_value_annual: float              # Estimated annual savings in dollars
exploit_grade: bool                     # True if pattern exploits known vulnerability
```

### PacketMetrics Schema

```
window_volume: int                      # Total windows processed
avg_compression: float                  # Average compression ratio (X.X format)
annual_savings: float                   # Estimated annual savings in dollars
slo_breach_rate: float                 # SLO violation rate [0.0-1.0]
```

### Example DecisionPacket JSON

```json
{
  "packet_id": "a7b3c2d9e1f4a6b8",
  "deployment_id": "tesla-prod-us-west-001",
  "timestamp": "2025-12-10T14:30:00+00:00",
  "manifest_ref": "manifests/tesla-prod-v8.json",
  "sampled_receipts": ["rcpt_001", "rcpt_002", "rcpt_003"],
  "clarity_audit_ref": "audits/clarity_2025-12-10.json",
  "edge_lab_summary": {
    "n_tests": 10000,
    "n_hits": 9850,
    "aggregate_recall": 0.985,
    "_build_metadata": {
      "build_timestamp": "2025-12-10T14:25:00+00:00",
      "build_duration_ms": 127.45,
      "sources_loaded": ["manifest", "clarity", "edge_lab", "patterns"],
      "truthlink_version": "1.0.0"
    }
  },
  "pattern_usage": [
    {
      "pattern_id": "bat_thermal_001",
      "validation_recall": 0.995,
      "false_positive_rate": 0.008,
      "dollar_value_annual": 1500000.0,
      "exploit_grade": true
    },
    {
      "pattern_id": "motion_spike_003",
      "validation_recall": 0.999,
      "false_positive_rate": 0.005,
      "dollar_value_annual": 2000000.0,
      "exploit_grade": true
    }
  ],
  "metrics": {
    "window_volume": 1247832,
    "avg_compression": 11.3,
    "annual_savings": 2400000.0,
    "slo_breach_rate": 0.0002
  },
  "exploit_coverage": 0.67,
  "health_score": 92,
  "glyph": "A7-B3-C2-D9",
  "integrity_status": "VERIFIED",
  "parent_packet_id": "z9y8x7w6v5u4t3s2"
}
```

## 3. TruthLink Build Flow

**File**: `truthlink.py`

TruthLink is the fusion layer that builds DecisionPackets from raw components. It combines historian capability (builds current packet) with oracle capability (projects future packets).

### Build Process (Unified Entry Point)

```python
from truthlink import build
packet = build(
    deployment_id="tesla-prod-001",
    manifest_path="manifests/tesla.json",
    receipts_dir="receipts/",
    clarity_path="audits/clarity.json",
    edge_lab_path="edge_lab/results.json",
    patterns_path="patterns/library.json",
    mode="single",
    sample_size=100  # configurable receipt sampling
)
```

### Build Flow Numbered Steps

1. **Load manifest** via `load_manifest()` - required source
2. **Deterministic seed generation** for auditable receipt sampling
3. **Stream-sample receipts** with reservoir sampling + stratified extremes
   - O(1) memory regardless of receipt count
   - Captures top SLO breaches and top savings separately
   - Configurable sample size (default 100)
4. **Load optional sources** with graceful degradation (clarity, edge_lab, patterns)
5. **Compute aggregate metrics** from sampled receipts and/or manifest
6. **Load pattern library** and filter to deployment's enabled patterns
7. **Build PatternSummary list** with recall, FP rate, savings, exploit grade
8. **Compute exploit_coverage** as fraction of exploit-grade patterns
9. **Generate packet_id** (SHA3-256, first 16 chars) of canonical packet content
10. **Create DecisionPacket** with auto-generated timestamp and computed fields
11. **Return packet** ready for persistence or further processing

### Build Function Signature

```python
def build(
    deployment_id: str,
    manifest_path: str,
    receipts_dir: str,
    clarity_path: Optional[str] = None,
    edge_lab_path: Optional[str] = None,
    patterns_path: Optional[str] = None,
    mode: Literal["single", "batch", "watch"] = "single",
    parent_packet_id: Optional[str] = None,
    sample_size: int = 100
) -> Union[DecisionPacket, List[DecisionPacket], Generator[DecisionPacket]]
```

**Modes:**
- `single`: Build one packet from manifest path
- `batch`: Build multiple packets from directory of manifests
- `watch`: Generator that yields packets as new manifests appear (streaming)

### Packet Projection (Chef's Kiss)

TruthLink can also project future state of packets based on hypothetical changes:

```python
from truthlink import project, AddPattern, RemovePattern, ScaleFleet
projected = project(
    base_packet=current_packet,
    changes=[
        AddPattern("new_exploit_pattern_001"),
        ScaleFleet(1.5),  # Scale fleet by 50%
    ],
    historical_packets=historical_list
)
# Returns ProjectedPacket with:
# - projected_savings_delta, projected_health_delta
# - confidence score
# - assumptions and recommendations
# - similar deployments for validation
```

## 4. Config Schema and Validation

**File**: `config_schema.py`

QEDConfig is the deployment control plane—not just settings, but a predictive, self-validating, auditable contract that knows what will happen when applied.

### QEDConfig Fields

```
version: str                            # Schema version (e.g., "1.0")
deployment_id: str                      # Unique identifier for deployment
hook: str                               # Business unit (tesla, spacex, starlink, etc)
compression_target: float               # Target compression ratio (default: 10.0)
recall_floor: float                    # Minimum recall threshold [0.0-1.0] (default: 0.999)
max_fp_rate: float                     # Maximum false positive rate [0.0-1.0] (default: 0.01)
slo_latency_ms: int                    # P95 latency budget in milliseconds (default: 100)
slo_breach_budget: float               # Allowed SLO breach rate [0.0-1.0] (default: 0.001)
enabled_patterns: Tuple[str, ...]      # Pattern IDs allowed (empty = all) (default: ())
safety_overrides: Dict[str, Any]       # Safety settings (can only tighten)
regulatory_flags: Dict[str, bool]      # Compliance requirements
provenance: ConfigProvenance           # Who/when/why/parent tracking
```

### Key Behaviors

- **Validation < 1ms**: Hard SLO using jsonschema Draft2020-12 with compiled validators
- **Self-healing**: Invalid input → safe defaults + warnings (unless strict=True)
- **Immutable**: Frozen dataclass, no runtime mutation
- **Hashed**: config_hash computed and verified for integrity
- **Auditable**: Every config carries provenance with author, timestamp, reason, parent hash

### Config Loading and Validation

```python
from config_schema import load, QEDConfig

# Load from file (JSON or YAML)
config = load("config.json", validate=True, strict=False)  # auto-validates

# Create default config for hook
default = QEDConfig.default("tesla")

# Create from dict with validation
config = QEDConfig.from_dict(data, validate=True)

# Create derived config (auto-validates safety tightening)
child = QEDConfig.from_parent(
    parent=global_config,
    changes={"recall_floor": 0.9995, "max_fp_rate": 0.005},
    reason="regional tightening",
    author="ops-admin"
)
```

### Config Simulation (Predictive)

```python
# Simulate what will happen if config is applied
simulation = config.simulate(
    patterns_library=["pattern_1", "pattern_2", ...],
    fleet_size=1000
)
# Returns ConfigSimulation with:
# - patterns_enabled, patterns_blocked
# - estimated_coverage, estimated_breach_rate
# - risk_profile (conservative|moderate|aggressive)
# - warnings for unsafe configurations
# - comparison to default config
```

## 5. Merge Rules Engine

**File**: `merge_rules.py`

The merge rules engine enforces "safety only tightens" governance across configuration layers (global → company → region → deployment).

### Merge Rules (Four Principles)

1. **STRICTER for safety**: Take tighter value
   - `recall_floor`: max(parent, child) - higher is stricter
   - `max_fp_rate`: min(parent, child) - lower is stricter
   - `slo_latency_ms`: min(parent, child) - lower is tighter
   - `slo_breach_budget`: min(parent, child) - lower is tighter

2. **INTERSECTION for patterns**: Only patterns in BOTH
   - Child config can only restrict parent's pattern set further
   - Empty result triggers warning to prevent accidental "block all"

3. **OR for regulatory flags**: If either requires, merged requires
   - Flags are combined with OR logic
   - Cannot disable flags required by parent

4. **UNION for safety_overrides**: Parent wins on conflict

### Merge API

```python
from merge_rules import merge, merge_chain, MergeResult

# Merge two configs (parent -> child)
result = merge(
    parent=global_config,
    child=deployment_config,
    auto_repair=False,        # If True, auto-fix violations
    author="ops-team",
    reason="deployment override",
    emit_receipt_flag=True    # Write to merge_receipts.jsonl
)
# Returns MergeResult with:
# - merged: The validated merged config (or None if invalid)
# - is_valid: Whether merge succeeded
# - violations: List of safety violations (if any)
# - repairs_applied: List of auto-repairs (if auto_repair=True)
# - explanation: MergeExplanation (human-readable)
# - receipt: MergeReceipt (cryptographic audit trail)

# Merge chain: global -> company -> region -> deployment
result = merge_chain(
    [global_cfg, company_cfg, region_cfg, deployment_cfg],
    auto_repair=True
)
# Returns single MergeResult with:
# - trace: Per-level violations and repairs
# - cumulative_tightening: Original -> final values
```

### Simulation and Repair Suggestions

```python
# Dry run: preview merge without committing
simulation = simulate_merge(parent, child)
# Returns MergeSimulation with preview and available repairs

# Get repair suggestions without applying
repairs = suggest_repairs(parent, child)
# Returns List[Repair] with suggested fixes

# Apply repairs to child config
repaired_child = apply_repairs(child, repairs)
# Returns new valid config
```

### Pre-flight Conflict Detection

```python
# Check for conflicts before merge chain
conflicts = detect_conflicts([cfg1, cfg2, cfg3])
# Identifies pattern mismatches, flag conflicts, SLO incompatibilities
```

### Merge Receipt (Audit Trail)

Every merge emits a receipt to `merge_receipts.jsonl`:

```json
{
  "timestamp": "2025-12-10T15:00:00+00:00",
  "receipt_id": "f4a6b8c2d9e1a3b7",
  "parent_hash": "p1h2a3s4h5",
  "child_hash": "c6h7a8s9h0",
  "merged_hash": "m1h2a3s4h5",
  "is_valid": true,
  "violations_count": 0,
  "repairs_count": 0,
  "fields_tightened": ["recall_floor", "max_fp_rate"],
  "patterns_removed": ["pattern_x", "pattern_y"],
  "author": "ops-admin",
  "reason": "deployment override for regional compliance"
}
```

## 6. mesh_view_v3 Deployment Graph

**File**: `mesh_view_v3.py`

mesh_view_v3 transforms packets into a living fleet topology graph. It's not just visualization—it's a queryable graph that predicts pattern propagation and detects fleet anomalies.

### Core Graph Structures

**DeploymentNode**: Represents a single deployment

```
deployment_id: str                      # Deployment identifier
packet_id: str                          # Most recent packet
hook: str                               # Business unit
region: str                             # Deployment region
hardware_profile: str                   # Hardware configuration
patterns: Set[str]                      # Enabled patterns
exploit_patterns: Set[str]              # Exploit-grade patterns
metrics: NodeMetrics                    # Health/savings/breach_rate
last_updated: str                       # Timestamp
is_stale: bool                          # No packet in >7 days
```

**DeploymentEdge**: Connects nodes based on similarity

```
from_deployment: str                    # Source deployment
to_deployment: str                      # Target deployment
similarity_score: float                 # Pattern/config similarity [0.0-1.0]
weight: float                           # Edge weight for propagation
shared_patterns: Set[str]               # Common patterns
shared_hook: bool                       # Same business unit
shared_region: bool                     # Same geographic region
```

### Graph API

```python
from mesh_view_v3 import build, find_clusters, find_outliers, predict_propagation

# Build deployment graph from packets
graph = build(
    packets_dir="data/packets/",
    stale_threshold_days=7,
    configs_dir="configs/"  # optional
)
# Returns DeploymentGraph with nodes and edges

# Find pattern reuse clusters
clusters = find_clusters(graph)
# Returns Deployment groups sharing patterns/hooks/regions

# Find outlier deployments
outliers = find_outliers(graph)
# Returns deployments diverging from fleet norms

# Predict pattern propagation
recommendations = predict_propagation(
    graph=graph,
    source_pattern="bat_thermal_001",
    target_deployments=None  # None = recommend to all suitable
)
# Returns pattern adoption recommendations per deployment
```

### Fleet Health Diagnostics

```python
# Check graph health
health = graph.diagnose()
# Returns HealthDiagnosis with:
# - stale_deployments: List of nodes without recent packets
# - isolated_clusters: Groups with no inter-cluster patterns
# - coverage_gaps: Patterns missing in key regions
# - anomalies: Deployments deviating from norms

# Get fleet cohesion metrics
metrics = graph.compute_fleet_metrics()
# Returns FleetMetrics with overall health summary
```

### Output: deployment_graph.json

```json
{
  "nodes": [
    {
      "deployment_id": "tesla-prod-us-west-001",
      "packet_id": "a7b3c2d9e1f4a6b8",
      "hook": "tesla",
      "region": "us-west",
      "hardware_profile": "A100",
      "patterns": ["bat_thermal_001", "motion_spike_003"],
      "exploit_patterns": ["bat_thermal_001"],
      "metrics": {
        "health_score": 92,
        "savings": 2400000.0,
        "breach_rate": 0.0002
      },
      "last_updated": "2025-12-10T14:30:00+00:00",
      "is_stale": false
    }
  ],
  "edges": [
    {
      "from_deployment": "tesla-prod-us-west-001",
      "to_deployment": "tesla-prod-us-east-001",
      "similarity_score": 0.85,
      "weight": 0.85,
      "shared_patterns": ["bat_thermal_001", "motion_spike_003"],
      "shared_hook": true,
      "shared_region": false
    }
  ],
  "fleet_cohesion": 0.78
}
```

## 7. CLI Commands (proof.py)

**File**: `proof.py`

The proof CLI exposes v8 capabilities:

| Command | Usage | Purpose |
|---------|-------|---------|
| `proof build-packet` | `-d DEPLOYMENT_ID -m MANIFEST -r RECEIPTS_DIR` | Build DecisionPacket from artifacts |
| `proof validate-config` | `CONFIG_PATH` | Validate QED config file with detailed report |
| `proof merge-configs` | `-g GLOBAL_CFG -d DEPLOYMENT_CFG` | Merge and display result with explanation |
| `proof compare-packets` | `PACKET_A_ID PACKET_B_ID` | Diff two packets, show narrative |
| `proof fleet-view` | `PACKETS_DIR [--configs CONFIGS_DIR]` | Build deployment graph, show reuse clusters |
| `proof config-explain` | `CONFIG_PATH` | Show what config will do (simulation) |
| `proof merge-receipt` | `[--jsonl PATH]` | Tail merge receipts, show audit trail |

### Example Usage

```bash
# Build a packet
proof build-packet -d tesla-prod-001 \
  -m manifests/tesla.json \
  -r receipts/ \
  --clarity audits/clarity.json \
  --edge-lab edge_lab/results.json

# Validate a config
proof validate-config config.json --strict

# Merge two configs and see result
proof merge-configs -g config_global.json -d config_deployment.json --auto-repair

# Compare two packets
proof compare-packets a7b3c2d9e1f4a6b8 z9y8x7w6v5u4t3s2

# Build fleet topology and show pattern clusters
proof fleet-view data/packets/ --configs configs/ --output deployment_graph.json

# Show what a config will do when applied
proof config-explain config.json --patterns patterns.json --fleet-size 1000

# Tail merge receipt audit log
proof merge-receipt --jsonl merge_receipts.jsonl --follow
```

## 8. v8 KPIs

Success metrics for v8 deployments:

- **One valid DecisionPacket per live deployment**: Every active deployment has a current packet with health score ≥ 80
- **Growing share of exploit-grade patterns**: Each quarter, 10%+ more deployments adopt at least one exploit-grade pattern
- **Visible cross-fleet pattern reuse**: Fleet cohesion ≥ 0.75 measured by mesh_view_v3 graph

Additional operational KPIs:

- **Merge compliance**: 100% of config changes validated by merge rules (0 bypasses)
- **Governance effectiveness**: Average recall ≥ 99.9% across fleet, avg FP rate ≤ 1%
- **Deployment health**: Health scores stable (trend "stable" not "degrading")

## 9. Mapping to v4 Strategy

How v8 implements the original QED v4 strategy:

| Strategy Bullet | v8 Implementation | Evidence |
|-----------------|-------------------|----------|
| "TruthLink wraps receipts, manifests, audits into DecisionPackets" | truthlink.py builds packets from manifest, sampled receipts, clarity audit, edge_lab | decision_packet.py schema + build() flow |
| "Small configs with merge rules that only tighten safety" | QEDConfig + merge_rules.py enforce stricter/intersection/OR | config_schema.py + merge() function |
| "Packets show which high-value patterns each deployment runs" | PatternSummary lists pattern_id, recall, fp_rate, dollar_value, exploit_grade | decision_packet.py pattern_usage field |
| "Deployments comparable and signable" | packet_id = SHA3 hash; optional signature field; parent_packet_id chains | integrity_status property, verify_integrity() |
| "Governance across layers (global→deployment)" | merge_chain() validates N configs; auto-repair suggestions | merge_rules.py merge_chain() + suggest_repairs() |
| "Fleet topology with pattern reuse visibility" | mesh_view_v3 builds graph; find_clusters() identifies reuse; predict_propagation() | mesh_view_v3.py build() + clusters API |

## Verification Checklist

After deploying v8, verify:

- [ ] File renders correctly in GitHub markdown preview
- [ ] All section headers use consistent ## level
- [ ] Code examples are in fenced blocks with language tags
- [ ] All relative links to source files work (decision_packet.py, truthlink.py, etc)
- [ ] DecisionPacket JSON example is valid (can parse with json module)
- [ ] All CLI commands are documented with example usage
- [ ] KPI section is specific and measurable
- [ ] v4 strategy mapping covers all original requirements

## Related Files and Further Reading

- **decision_packet.py**: DecisionPacket dataclass with all properties and methods
- **truthlink.py**: TruthLink fusion layer, streaming sampling, projection
- **config_schema.py**: QEDConfig validation, self-healing, JSON schema
- **merge_rules.py**: Merge rules, repair suggestions, audit receipts
- **mesh_view_v3.py**: Deployment graph, cluster detection, propagation prediction
- **proof.py**: CLI commands for v8 operations
- **CLAUDEME.md**: Project conventions (receipts, glyphs, swarm patterns)
- **V6_NOTES.md**: Previous version documentation (for format reference)
