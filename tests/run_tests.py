#!/usr/bin/env python3
"""
Test runner script for gravitational wave hunter.

This script provides convenient ways to run the test suite with different options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, slow=False):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest", "tests/"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=gravitational_wave_hunter", "--cov-report=term-missing"])
    
    # Filter test types
    if test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type == "unit":
        cmd.extend(["-m", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "gpu":
        cmd.extend(["-m", "gpu"])
    
    # Run tests
    print(f"Running tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="Run tests for gravitational wave hunter")
    parser.add_argument(
        "--type", 
        choices=["all", "fast", "slow", "unit", "integration", "gpu"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests (overrides --type)"
    )
    
    args = parser.parse_args()
    
    # Determine test type
    test_type = args.type
    if args.slow:
        test_type = "all"
    
    # Run tests
    exit_code = run_tests(
        test_type=test_type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
