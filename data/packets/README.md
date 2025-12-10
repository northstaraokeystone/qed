# DecisionPacket Storage

## Purpose

Storage location for DecisionPacket JSONL outputs from TruthLink. Each packet represents one deployment's audit bundle, containing everything needed to verify a deployment's decision-health and pattern effectiveness.

## Packet Contents

Each DecisionPacket includes:

- **Links**: Run manifest, sampled receipts, ClarityClean audit references
- **Pattern metrics**: recall, fp_rate, dollar_value per pattern
- **Aggregate metrics**:
  - `window_volume`: Total events processed in the audit window
  - `avg_compression`: Data compression ratio achieved
  - `annual_savings`: Projected yearly cost savings
  - `slo_breach_rate`: Percentage of SLO violations
- **exploit_coverage**: Score measuring protection against known exploit patterns
- **Health metrics**: Decision-health scores and threshold compliance

## Naming Convention

**Format**: `{deployment_id}_{timestamp}.jsonl`

**Examples**:
- `tesla-prod_2025-12-10T00-00-00Z.jsonl`
- `spacex-staging_2025-12-09T12-30-00Z.jsonl`
- `starlink-edge-eu_2025-12-10T06-00-00Z.jsonl`

Timestamps use ISO 8601 format with colons replaced by hyphens for filesystem compatibility.

## Lifecycle

| Stage | Action |
|-------|--------|
| **Created by** | `proof build-packet` command |
| **Consumed by** | mesh_view_v3 (deployment graph), portfolio binder (aggregation) |
| **Retention** | 90 days active, archive to cold storage after |

### Retention Policy

- **Active (0-90 days)**: Packets remain in this directory for immediate access
- **Archive (90+ days)**: Move to S3-compatible blob storage with lifecycle receipts
- **Deletion**: Per data retention policy, with deletion_receipt logged

## Verification

Each packet includes a `packet_id` field containing the SHA3 hash of its contents.

**Validate a packet**:
```bash
proof validate-packet {file}
```

This command:
1. Recomputes SHA3 hash of packet contents
2. Compares against embedded `packet_id`
3. Verifies all linked receipts exist and match
4. Outputs validation_receipt on success

## Important Notes

- **Do not commit packet files to git** - packets are runtime artifacts, not source
- Packets accumulate over time - monitor disk usage
- Each packet is immutable once created - corrections require new packets
- Packets feed downstream systems (mesh_view, portfolio binder) - do not delete active packets without understanding dependencies
