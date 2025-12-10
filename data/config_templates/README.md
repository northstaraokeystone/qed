# QED Config Templates

Template configurations for QED deployments following the v8 spec.

## Purpose

This directory contains baseline and deployment-specific configuration templates that define:
- Safety thresholds (recall floors, false positive rates)
- SLO targets (latency, breach budgets)
- Enabled pattern families
- Regulatory compliance flags
- Safety-critical overrides

## Schema Reference

Config files follow the `QEDConfig` schema defined in `config_schema.py` (when built). All fields are validated at load time.

## Config Hierarchy

Configs merge in a strict hierarchy:

```
global → company → region → deployment
```

Each level inherits from its parent and can **only tighten** safety parameters:
- `recall_floor`: can only increase (more strict)
- `max_fp_rate`: can only decrease (fewer false positives)
- `slo_latency_ms`: can only decrease (faster response)
- `slo_breach_budget`: can only decrease (tighter budget)

Attempting to loosen any safety parameter from a parent config will raise a validation error.

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Config schema version (e.g., "1.0") |
| `deployment_id` | string | Unique identifier for this config |
| `hook` | string\|null | Hook name to apply, or null for global |
| `recall_floor` | float | Minimum recall threshold (0.0-1.0) |
| `max_fp_rate` | float | Maximum false positive rate (0.0-1.0) |
| `slo_latency_ms` | int | Target p95 latency in milliseconds |

## Optional Overrides

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `slo_breach_budget` | float | 0.005 | Allowed SLO breach fraction |
| `compression_target` | float | 10.0 | Target compression ratio |
| `enabled_patterns` | list | [] | Pattern families to enable (e.g., `PAT_CAN_*`) |
| `safety_overrides` | dict | {} | Domain-specific safety thresholds |
| `regulatory_flags` | dict | {} | Compliance flags (e.g., NHTSA, FAA) |

## Example: Loading and Merging Configs

```python
import json
from pathlib import Path

def load_config(path: Path) -> dict:
    """Load a single config file."""
    with open(path) as f:
        return json.load(f)

def merge_configs(parent: dict, child: dict) -> dict:
    """
    Merge child config into parent, enforcing tightening-only rules.

    Raises ValueError if child attempts to loosen safety parameters.
    """
    result = parent.copy()

    # Fields that can only tighten (increase)
    tighten_up = {"recall_floor"}
    # Fields that can only tighten (decrease)
    tighten_down = {"max_fp_rate", "slo_latency_ms", "slo_breach_budget"}

    for key, value in child.items():
        if key in tighten_up:
            if value < parent.get(key, 0):
                raise ValueError(
                    f"{key} can only increase: {value} < {parent[key]}"
                )
        elif key in tighten_down:
            if value > parent.get(key, float('inf')):
                raise ValueError(
                    f"{key} can only decrease: {value} > {parent[key]}"
                )
        result[key] = value

    # Merge dict fields (enabled_patterns, regulatory_flags, safety_overrides)
    for dict_key in ["enabled_patterns", "regulatory_flags", "safety_overrides"]:
        if dict_key in child:
            if isinstance(child[dict_key], list):
                # Lists extend (enabled_patterns)
                result[dict_key] = parent.get(dict_key, []) + child[dict_key]
            elif isinstance(child[dict_key], dict):
                # Dicts merge
                result[dict_key] = {**parent.get(dict_key, {}), **child[dict_key]}

    return result

# Usage example
templates = Path("data/config_templates")
global_cfg = load_config(templates / "global_config.json")
tesla_cfg = load_config(templates / "tesla_config.json")

# Merge Tesla into global baseline
final_cfg = merge_configs(global_cfg, tesla_cfg)
print(f"Final recall_floor: {final_cfg['recall_floor']}")  # 0.9995 (Tesla's tighter value)
print(f"Enabled patterns: {final_cfg['enabled_patterns']}")  # Tesla's patterns
```

## Files in This Directory

| File | Description |
|------|-------------|
| `global_config.json` | Fleet-wide baseline (conservative defaults) |
| `tesla_config.json` | Tesla automotive deployment (NHTSA, ISO26262) |
| `spacex_config.json` | SpaceX launch-critical deployment (FAA, ITAR, NASA_NPR) |

## Design Rationale

| Design Choice | Rationale |
|---------------|-----------|
| `enabled_patterns` empty in global | Forces explicit opt-in per deployment - no accidental pattern inheritance |
| Progressive strictness (global → tesla → spacex) | Demonstrates merge-only-tightens principle with real examples |
| `regulatory_flags` per deployment | Regulations are jurisdiction-specific, cannot assume fleet-wide |
| Hierarchy documented here | Auditors need to understand how final config is computed |
