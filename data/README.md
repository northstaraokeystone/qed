# QED v7 Data Directory

## Purpose

This directory holds anomaly patterns, scenarios, and seed data for QED v7. These files support the anomaly library, edge lab testing, and cross-domain pattern sharing.

## File Descriptions

### shared_anomalies.jsonl
Cross-company anomaly library containing validated failure patterns that can be shared across organizational hooks (Tesla, SpaceX, Starlink). This file is **append-only** to preserve audit integrity.

### edge_lab_scenarios.jsonl
Edge lab v2 test scenarios used for validating anomaly detection patterns. Contains labeled scenarios (anomaly, near_miss, normal) with expected loss estimates.

### nhtsa_sample.jsonl
Seed patterns derived from public NHTSA recall data. These patterns bootstrap the anomaly library with real-world failure modes from automotive safety recalls.

## Immutability Rules

1. **shared_anomalies.jsonl is append-only**
   - New patterns may only be appended to the end of the file
   - Existing entries must never be modified or deleted

2. **Entries keyed by SHA3 pattern_id**
   - Each pattern has a unique SHA3-derived pattern_id
   - Collisions are rejected - duplicate pattern_ids cannot be added

3. **Never edit existing entries**
   - To deprecate a pattern, set `deprecated=true` in a new entry
   - Never modify historical entries to preserve audit trail

4. **nhtsa_sample.jsonl is read-only seed data**
   - These patterns are static reference data
   - Do not modify - they serve as baseline seeds for the library

5. **All changes logged via receipts for audit trail**
   - Every modification generates a receipt
   - Receipts enable full auditability of pattern history

## Schema Reference

See `shared_anomalies.py` for the `AnomalyPattern` schema definition used by all pattern files in this directory.
