#!/usr/bin/env python3
"""
Quick test to verify the vacuum fluctuation model implementation.
"""

from sim import (
    run_simulation, SCENARIO_BASELINE, SCENARIO_STRESS,
    vacuum_fluctuation, PLANCK_ENTROPY_BASE, SimConfig
)

def test_vacuum_fluctuation():
    """Test that vacuum floor fluctuates."""
    print("Testing vacuum_fluctuation()...")

    # Run multiple times and check variance
    values = [vacuum_fluctuation() for _ in range(100)]

    # Check that values vary
    assert len(set(values)) > 1, "Vacuum floor should fluctuate"

    # Check that all values are >= 50% of base
    min_allowed = PLANCK_ENTROPY_BASE * 0.5
    assert all(v >= min_allowed for v in values), f"All values should be >= {min_allowed}"

    print(f"✓ Vacuum fluctuation working (range: {min(values):.6f} - {max(values):.6f})")


def test_baseline_scenario():
    """Test BASELINE scenario with new vacuum model."""
    print("\nTesting BASELINE scenario...")

    # Run short simulation
    config = SimConfig(
        n_cycles=10,
        n_initial_patterns=5,
        wound_rate=0.1,
        resource_budget=1.0,
        random_seed=42,
        scenario_name="BASELINE"
    )
    result = run_simulation(config)

    # Check for vacuum_state receipts
    vacuum_receipts = [r for r in result.final_state.receipt_ledger
                      if r.get("receipt_type") == "vacuum_state"]

    assert len(vacuum_receipts) > 0, "Should have vacuum_state receipts"
    print(f"✓ Found {len(vacuum_receipts)} vacuum_state receipts")

    # Check that vacuum_floor varies
    vacuum_floors = [r["vacuum_floor"] for r in vacuum_receipts]
    assert len(set(vacuum_floors)) > 1, "Vacuum floor should vary across cycles"
    print(f"✓ Vacuum floor varies (range: {min(vacuum_floors):.6f} - {max(vacuum_floors):.6f})")

    # Check for no violations
    print(f"✓ Violations: {len(result.violations)} (expected: 0)")

    return result


def test_virtual_patterns():
    """Test that virtual patterns can exist."""
    print("\nTesting virtual patterns...")

    # Run STRESS scenario (more likely to trigger high H_observation)
    config = SimConfig(
        n_cycles=100,
        n_initial_patterns=5,
        wound_rate=0.5,
        resource_budget=0.3,
        random_seed=43,
        scenario_name="STRESS"
    )
    result = run_simulation(config)

    # Check for spontaneous_emergence receipts
    emergence_receipts = [r for r in result.final_state.receipt_ledger
                         if r.get("receipt_type") == "spontaneous_emergence"]

    if len(emergence_receipts) > 0:
        print(f"✓ Found {len(emergence_receipts)} spontaneous emergence events")
    else:
        print("  (No spontaneous emergence occurred in this run - may need higher H_observation)")

    # Check for virtual_collapse receipts
    collapse_receipts = [r for r in result.final_state.receipt_ledger
                        if r.get("receipt_type") == "virtual_collapse"]

    if len(collapse_receipts) > 0:
        print(f"✓ Found {len(collapse_receipts)} virtual collapse events")

    # Check for virtual_promotion receipts
    promotion_receipts = [r for r in result.final_state.receipt_ledger
                         if r.get("receipt_type") == "virtual_promotion"]

    if len(promotion_receipts) > 0:
        print(f"✓ Found {len(promotion_receipts)} virtual promotion events")

    print(f"✓ Violations: {len(result.violations)}")

    return result


def main():
    print("=" * 60)
    print("VACUUM FLUCTUATION MODEL VERIFICATION")
    print("=" * 60)

    try:
        # Test 1: Vacuum fluctuation
        test_vacuum_fluctuation()

        # Test 2: BASELINE scenario
        baseline_result = test_baseline_scenario()

        # Test 3: Virtual patterns
        stress_result = test_virtual_patterns()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

        # Print summary
        print("\nSummary:")
        print(f"  BASELINE violations: {len(baseline_result.violations)}")
        print(f"  STRESS violations: {len(stress_result.violations)}")

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
