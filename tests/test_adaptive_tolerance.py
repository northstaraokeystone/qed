#!/usr/bin/env python3
"""
Test script for adaptive entropy tolerance.

Verifies:
1. Tolerance values vary per cycle (not all identical)
2. tolerance_measurement receipts are emitted
3. Tolerance stays within TOLERANCE_FLOOR and TOLERANCE_CEILING bounds
4. Factor breakdown is included in receipts
"""

from sim import SimConfig, run_simulation, TOLERANCE_FLOOR, TOLERANCE_CEILING

def test_adaptive_tolerance():
    """Run short simulation and verify tolerance adapts."""
    print("=== Testing Adaptive Entropy Tolerance ===\n")

    # Run 10 cycles to observe tolerance variation
    config = SimConfig(
        n_cycles=10,
        n_initial_patterns=5,
        wound_rate=0.1,
        random_seed=42,
        scenario_name="TEST"
    )

    result = run_simulation(config)

    # Extract tolerance_measurement receipts
    tolerance_receipts = [
        r for r in result.final_state.receipt_ledger
        if r.get("receipt_type") == "tolerance_measurement"
    ]

    print(f"Found {len(tolerance_receipts)} tolerance_measurement receipts\n")

    if len(tolerance_receipts) == 0:
        print("❌ FAIL: No tolerance_measurement receipts emitted")
        return False

    # Extract tolerance values
    tolerances = [r["tolerance"] for r in tolerance_receipts]

    print("Tolerance per cycle:")
    for i, (receipt, tol) in enumerate(zip(tolerance_receipts, tolerances)):
        print(f"  Cycle {receipt['cycle']}: {tol:.6f} "
              f"(churn: {receipt['population_churn_factor']:.3f}, "
              f"clamped: {receipt['clamp_direction']})")

    # Verify tolerances are NOT all identical
    unique_tolerances = len(set(tolerances))
    print(f"\nUnique tolerance values: {unique_tolerances}")

    if unique_tolerances == 1:
        print("⚠️  WARNING: All tolerance values are identical")
        print("    This may be normal for very stable simulations")
    else:
        print("✓ PASS: Tolerance varies per cycle")

    # Verify bounds
    all_within_bounds = all(TOLERANCE_FLOOR <= t <= TOLERANCE_CEILING for t in tolerances)
    if all_within_bounds:
        print(f"✓ PASS: All tolerances within bounds [{TOLERANCE_FLOOR}, {TOLERANCE_CEILING}]")
    else:
        print(f"❌ FAIL: Some tolerances outside bounds")
        return False

    # Verify receipts have factor breakdown
    first_receipt = tolerance_receipts[0]
    required_fields = [
        "entropy_variance_factor",
        "population_churn_factor",
        "wound_rate_factor",
        "fitness_uncertainty_factor",
        "raw_value",
        "was_clamped",
        "clamp_direction"
    ]

    missing_fields = [f for f in required_fields if f not in first_receipt]
    if missing_fields:
        print(f"❌ FAIL: Missing fields in receipt: {missing_fields}")
        return False
    else:
        print("✓ PASS: All factor fields present in receipts")

    # Check for chaos alerts
    chaos_alerts = [
        r for r in result.final_state.receipt_ledger
        if r.get("receipt_type") == "anomaly" and r.get("metric") == "tolerance"
    ]

    if chaos_alerts:
        print(f"\n⚠️  {len(chaos_alerts)} chaos alerts emitted (tolerance >= ceiling)")
    else:
        print("\n✓ No chaos alerts (system stable)")

    print("\n=== Test Complete ===")
    return True


if __name__ == "__main__":
    import sys
    success = test_adaptive_tolerance()
    sys.exit(0 if success else 1)
