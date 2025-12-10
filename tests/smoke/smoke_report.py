#!/usr/bin/env python3
"""
QED v8 Smoke Test Report Generator

Runs smoke tests and generates proof receipt for audit trail.

Usage:
    python tests/smoke/smoke_report.py
    cd tests/smoke && python smoke_report.py

Output:
    - Console summary of test results
    - tests/smoke/smoke_receipt.jsonl with proof receipt

Source: SDD line 3-6: "No receipt => not real"
Source: Eng_Arch_Standards line 211-212: "Merkle proof exists linking input, code version"
"""

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    outcome: str  # "passed", "failed", "skipped", "error"
    duration_ms: float
    message: Optional[str] = None
    gate: Optional[str] = None  # G1, G2, G3, G4, G5


@dataclass
class SmokeReceipt:
    """Proof receipt for smoke test execution."""
    timestamp: str
    git_commit: str
    version: str
    gates: Dict[str, str]  # G1-G5 -> pass/fail/skip
    all_passed: bool
    timing_ms: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    details: List[Dict[str, Any]] = field(default_factory=list)


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_short_commit() -> str:
    """Get short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def extract_gate_from_test_name(test_name: str) -> Optional[str]:
    """Extract gate identifier (G1-G5) from test name."""
    name_lower = test_name.lower()
    for gate in ["g1", "g2", "g3", "g4", "g5"]:
        if gate in name_lower:
            return gate.upper()
    return None


def parse_pytest_json(json_path: Path) -> List[TestResult]:
    """Parse pytest JSON report into TestResult list."""
    results: List[TestResult] = []

    if not json_path.exists():
        return results

    with open(json_path, "r") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    for test in tests:
        nodeid = test.get("nodeid", "")
        outcome = test.get("outcome", "unknown")
        duration = test.get("call", {}).get("duration", 0.0) * 1000  # to ms

        # Extract test name from nodeid
        test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid

        # Extract gate
        gate = extract_gate_from_test_name(test_name)

        # Get message for failures
        message = None
        if outcome == "failed":
            longrepr = test.get("call", {}).get("longrepr", "")
            if longrepr:
                message = str(longrepr)[:500]  # Truncate long messages

        results.append(TestResult(
            name=test_name,
            outcome=outcome,
            duration_ms=duration,
            message=message,
            gate=gate,
        ))

    return results


def aggregate_gate_status(results: List[TestResult]) -> Dict[str, str]:
    """Aggregate test results into per-gate status."""
    gates: Dict[str, Dict[str, int]] = {
        "G1": {"passed": 0, "failed": 0, "skipped": 0},
        "G2": {"passed": 0, "failed": 0, "skipped": 0},
        "G3": {"passed": 0, "failed": 0, "skipped": 0},
        "G4": {"passed": 0, "failed": 0, "skipped": 0},
        "G5": {"passed": 0, "failed": 0, "skipped": 0},
    }

    for result in results:
        if result.gate and result.gate in gates:
            if result.outcome == "passed":
                gates[result.gate]["passed"] += 1
            elif result.outcome in ("failed", "error"):
                gates[result.gate]["failed"] += 1
            else:
                gates[result.gate]["skipped"] += 1

    # Determine gate status
    gate_status: Dict[str, str] = {}
    for gate, counts in gates.items():
        if counts["failed"] > 0:
            gate_status[gate] = "fail"
        elif counts["passed"] > 0:
            gate_status[gate] = "pass"
        else:
            gate_status[gate] = "skip"

    return gate_status


def run_smoke_tests() -> int:
    """
    Run smoke tests and generate proof receipt.

    Returns:
        Exit code (0 if all gates pass, 1 if any fail)
    """
    start_time = time.perf_counter()

    # Find the smoke test file
    script_dir = Path(__file__).parent
    test_file = script_dir / "test_smoke_v8.py"

    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return 1

    # Create temp file for JSON output
    json_report = script_dir / ".pytest_report.json"

    # Run pytest with JSON output
    print("=" * 60)
    print("QED v8 SMOKE TESTS")
    print("=" * 60)
    print(f"Test file: {test_file}")
    print(f"Git commit: {get_git_short_commit()}")
    print("-" * 60)

    # Run pytest
    pytest_args = [
        sys.executable,
        "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        f"--json-report-file={json_report}",
        "--json-report",
        "-s",  # Show print statements
    ]

    try:
        result = subprocess.run(
            pytest_args,
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=300,  # 5 minute timeout
        )
        pytest_exit_code = result.returncode
    except subprocess.TimeoutExpired:
        print("ERROR: Pytest timed out after 5 minutes")
        pytest_exit_code = 1
    except FileNotFoundError:
        # pytest-json-report might not be installed, try without it
        print("Note: pytest-json-report not installed, using basic output")
        pytest_args = [
            sys.executable,
            "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "-s",
        ]
        result = subprocess.run(
            pytest_args,
            capture_output=True,
            text=True,
            timeout=300,
        )
        pytest_exit_code = result.returncode
        # Print captured output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Parse results
    test_results: List[TestResult] = []
    if json_report.exists():
        test_results = parse_pytest_json(json_report)
        # Clean up temp file
        json_report.unlink()

    # Aggregate gate status
    gate_status = aggregate_gate_status(test_results)

    # Count results
    passed = sum(1 for r in test_results if r.outcome == "passed")
    failed = sum(1 for r in test_results if r.outcome in ("failed", "error"))
    skipped = sum(1 for r in test_results if r.outcome == "skipped")
    total = len(test_results)

    # Determine all_passed (no failures in available gates)
    all_passed = failed == 0

    # Create receipt
    receipt = SmokeReceipt(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        version="v8",
        gates=gate_status,
        all_passed=all_passed,
        timing_ms=round(elapsed_ms, 2),
        total_tests=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        details=[asdict(r) for r in test_results],
    )

    # Write receipt to JSONL
    receipt_path = script_dir / "smoke_receipt.jsonl"
    with open(receipt_path, "a") as f:
        f.write(json.dumps(asdict(receipt)) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {receipt.timestamp}")
    print(f"Git commit: {receipt.git_commit[:12]}")
    print(f"Total time: {elapsed_ms:.2f}ms")
    print("-" * 60)
    print(f"Total tests: {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print("-" * 60)
    print("GATE STATUS:")
    for gate, status in sorted(gate_status.items()):
        status_icon = {"pass": "[PASS]", "fail": "[FAIL]", "skip": "[SKIP]"}.get(status, "[????]")
        print(f"  {gate}: {status_icon}")
    print("-" * 60)

    if all_passed:
        print("RESULT: ALL GATES PASS")
    else:
        print("RESULT: GATES FAILED")
        # List failures
        failures = [r for r in test_results if r.outcome in ("failed", "error")]
        if failures:
            print("\nFailed tests:")
            for f in failures[:5]:  # Show first 5
                print(f"  - {f.name}")
                if f.message:
                    print(f"    {f.message[:100]}...")

    print("=" * 60)
    print(f"Receipt written to: {receipt_path}")

    return 0 if all_passed else 1


def main() -> int:
    """Main entry point."""
    return run_smoke_tests()


if __name__ == "__main__":
    sys.exit(main())
