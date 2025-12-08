# QED v6 Notes: Glue & CLI

**Version**: 6.0 | **Class**: glue_and_cli | **Domain**: generic | **Date**: 2025-12-08

## Overview

QED v6 introduces receipts-first telemetry compression with symbolic constraint verification. All flows emit immutable `QEDReceipt` records. Dollar/recall metrics drive KPIs.

## Receipts Schema

**QEDReceipt** (`qed.py:10-24`):
```
ts: str                    # ISO timestamp
window_id: str             # scenario_n_uuid
params: {A, f, phi, c, scenario, bit_depth, sample_rate_hz}
ratio: float               # Compression ratio
recall: float              # Safety-event recall [0,1]
savings_M: float           # ROI in millions USD
verified: bool|None        # Constraint pass/fail/unchecked
violations: [{constraint_id, value, bound, ts_offset}]
```

Output: `receipts.jsonl` via `write_receipt_jsonl()`.

## Sympy Constraints Registry

**File**: `sympy_constraints.py`

| Hook | Amplitude Bound | Ratio Min | Notes |
|------|-----------------|-----------|-------|
| tesla_fsd | A ≤ 14.7 | ≥ 20 | Steering torque |
| spacex_flight | A ≤ 20.0 | - | Thrust |
| neuralink_stream | A ≤ 5.0 | - | Neural spike |
| boring_tunnel | A ≤ 25.0 | ≥ 30 | Vibration |
| starlink_flow | A ≤ 18.0 | - | Signal |
| xai_eval | A ≤ 10.0 | - | Model metrics |

**API**: `evaluate_all(hook, A, ratio, savings_M, mse) → (passed, violations)`

## Edge Lab v1

**Scenario Schema** (`edge_lab_v1.py:41-50`):
```
scenario_id: str      # e.g., "tesla_spike_001"
hook: str             # tesla, spacex, neuralink, boring, starlink, xai
type: str             # spike, step, drift, normal, noise, saturation
expected_loss: float  # >0.1 for ROI significance
signal: list[float]   # Raw samples (1000+)
```

**KPI Gates**:
- Recall: ≥ 0.9967 (99.67%)
- Avg Ratio: ≥ 20.0
- Violation Rate: < 5%

## Mesh View v1

**Aggregation Output** (`mesh_view_v1.py:200-250`):
```json
{
  "companies": {
    "<company>": {
      "<hook>": {
        "windows": int,
        "avg_ratio": float,
        "total_savings": float,
        "slo_breach_rate": float,
        "constraint_violations": int
      }
    }
  }
}
```

## CLI Commands

| Tool | Command | Purpose |
|------|---------|---------|
| proof.py | `gates [--seed S]` | Run KPI gate checks |
| proof.py | `replay JSONL` | Replay scenarios |
| proof.py | `sympy_suite HOOK` | Validate hook constraints |
| proof.py | `summarize RESULTS` | Aggregate results |
| edge_lab_v1.py | `--jsonl PATH` | Run edge scenarios |
| mesh_view_v1.py | `MANIFEST RECEIPTS` | Generate mesh view |

## v4 Strategy Alignment

- **Receipts-first**: All telemetry → QEDReceipt JSONL
- **Dollar/recall driven**: savings_M tracked, recall ≥ 99.67%
- **One engine**: qed.py core, hook-specific bounds in registry
