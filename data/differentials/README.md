# data/differentials/ — Performance Cache Directory

## Purpose

This directory is an **optional cache** for computed state differentials. It is **NOT** a source of truth.

## Critical Constraints

### 1. Not Truth
State is the differential between receipt sets in the ledger, not stored in files in this directory.
Per CLAUDEME §5 Paradigm: State is represented by glyphs and receipts in the ledger, computed on-demand.

### 2. Performance Only
Files here are optimization artifacts. They speed up computation but have zero semantic authority.

### 3. Fully Deletable
The system MUST function identically if this directory is empty or deleted entirely.

```bash
# Verification test (must pass):
rm -rf data/differentials
python -m pytest  # All tests pass unchanged
```

### 4. Fully Recomputable
Any cached value can be recomputed from receipt history without loss of fidelity.

## What Goes Here

- Computed entropy deltas (optional caching)
- Integrity check intermediates (optional caching)
- Remediation state calculations (optional caching)
- Performance-sensitive differential computations

## What NEVER Goes Here

- ❌ Receipt state files (receipts go in receipts.json)
- ❌ Configuration (config belongs in qed_config.json)
- ❌ Truth of any kind (cache is optimization only)
- ❌ Anything that breaks tests when deleted

## Integration Points

These modules may write cached computations here:
- `entropy.py` — differential entropy measurements
- `integrity.py` — integrity check caches
- `remediate.py` — remediation state caches
- `unified_loop.py` — loop state differentials

## Deletion Safety

```python
# System must handle gracefully:
if not os.path.exists("data/differentials"):
    # Recompute from receipts, no failure
    pass
```

## Verification

Per smoke test H10: `rm -rf data/differentials && python -m pytest` must pass with all tests green.

---

**Status:** Cache directory (optimization only, not truth)
**Deletability:** 100% safe to delete and rebuild
**Authority:** Zero — receipts in ledger are sole authority
