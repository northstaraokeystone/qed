# QED v7 Developer Notes

**Version:** 7.0 | **Date:** 2025-12-08
**Summary:** Physics injection, anomaly library, recall floor quantification

---

## shared_anomalies.jsonl

### Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `pattern_id` | str | SHA3 hash, 16 chars |
| `physics_domain` | str | Domain category |
| `failure_mode` | str | Failure type |
| `params` | dict | Pattern parameters |
| `hooks` | list | Associated hooks |
| `dollar_value_annual` | float | Annual loss value |
| `validation_recall` | float | 0.0-1.0 |
| `false_positive_rate` | float | 0.0-1.0 |
| `cross_domain_targets` | list | Valid target domains |
| `deprecated` | bool | Deprecation flag |

### Derived Properties

```python
training_score = min(1, dollar_value_annual / 10_000_000)
training_role = "train_cross_company" if dollar_value_annual > 10_000_000 else "observe_only"
exploit_grade = dollar_value_annual > 1_000_000 and validation_recall >= 0.99 and false_positive_rate <= 0.01
```

### Immutability Rules

- **Append-only** — keyed by SHA3 `pattern_id`
- **Duplicates rejected** — append returns `False`
- **Never edit entries** — only set `deprecated=true`
- **All changes logged** for audit

---

## Physics Injection Sources

| Source | Data | Patterns | Physics |
|--------|------|----------|---------|
| **NGSIM (FHWA)** | Vehicle trajectories (I-80, US-101) | hard_brake, sudden_accel, lane_drift, stop_and_go, near_collision | acceleration, lateral deviation, speed variance |
| **SAE J1939** | CAN bus fault standard | engine_overheat, oil_pressure_low, battery_voltage, sensor_fault, can_bus_error | signal dropout, value corruption |
| **NHTSA** | Public recall data | brake_fade, battery_thermal_runaway, steering_assist_loss, throttle_stuck, sensor_drift | component degradation, failure spikes |

*No proprietary fleet data used.*

---

## Recall Floor Math

**Method:** Clopper-Pearson exact binomial confidence interval

```python
lower_bound = beta.ppf(α/2, k, n - k + 1)
# n = total tests, k = successes (no misses), α = 1 - confidence
```

**Default sampling:** 300 vehicles × 3 anomalies = 900 tests

| Tests | Misses | Recall Floor (95% CI) |
|-------|--------|----------------------|
| 900 | 0 | 0.9967 |
| 100 | 0 | 0.9638 |
| 900 | 1 | 0.9944 |

*"With 900 tests and zero misses, we can claim at least 99.67% recall with 95% confidence."*

---

## Cross-Domain Mapping

### Approved Mappings (v7)

| Source | Target | Rationale |
|--------|--------|-----------|
| Tesla battery_thermal | SpaceX | Li-ion thermal physics align |
| Tesla comms | Starlink | RF signal quality aligns |

### Validation Flow

1. Check mapping exists in `CROSS_DOMAIN_MAP`
2. Run 1000-injection test on target domain
3. Compute recall and FP rate on target
4. Pass if `recall ≥ 0.99` AND `FP ≤ 0.01`
5. Log to `data/cross_domain_validations.jsonl`

**No auto-approval:** all cross-domain use requires explicit validation pass.

---

## Running edge_lab_v2 Sims

```bash
python proof.py run-sims --receipts-dir receipts/ --n-per-hook 1000
python proof.py recall-floor --sim-results data/sim_results.json
python proof.py pattern-report --sort-by dollar_value
```

**Output:** `data/sim_results.json`

| Metric | Description | Target |
|--------|-------------|--------|
| `sim_recall` | hits / (hits + misses) | ≥ 0.999 |
| `sim_fp_rate` | false positives / clean windows | ≤ 0.01 |
| `recall_floor` | Clopper-Pearson lower bound | 95% CI |

---

## v7 KPIs

From v4 strategy:
- Coverage of top loss scenarios in edge_lab v2
- Per-pattern recall and false positive rate after sims
- Sum of `dollar_value_annual` covered by exploit_grade patterns
- Number of validated cross-domain links

---

## v4 Strategy Mapping

| v4 Bullet | v7 File |
|-----------|---------|
| Clean adapter | `clarity_clean_adapter.py` |
| edge_lab v2 with physics injection | `edge_lab_v2.py`, `physics_injection.py` |
| shared_anomalies.jsonl library | `shared_anomalies.py`, `data/shared_anomalies.jsonl` |
| Rule-based sims on receipts | `edge_lab_v2.run_pattern_sims()` |
| mesh_view_v2 with exploit counts | `mesh_view_v2.py` |
| Recall floor quantification | `edge_lab_v2.compute_recall_floor()` |
| Cross-domain mappings | `cross_domain.py` |
| NHTSA pattern pipeline | `nhtsa_pipeline.py` |

---

## What's Unchanged from v6

- `QEDReceipt` schema (`pattern_id` added as optional field)
- `qed.run()` behavior (`pattern_id` param added, backward compatible)
- `sympy_constraints.py`
- `proof.py` v6 subcommands
- hooks structure (new functions added)
