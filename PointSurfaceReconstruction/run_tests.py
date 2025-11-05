#!/usr/bin/env python3
"""
Script to run all tests for the 3D Surface Reconstruction project
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all tests with pytest"""
    # Change to project root directory
    os.chdir(project_root)
    
    # Run pytest with specific options
    pytest_args = [
        "tests/",
        "-v",           # Verbose output
        "--tb=short",   # Short traceback
        "--durations=10",  # Show slowest 10 tests
        "-x",           # Stop on first failure
        "--disable-warnings",  # Disable warnings for cleaner output
    ]
    
    # Add coverage if available
    try:
        import coverage
        pytest_args.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report"
        ])
    except ImportError:
        print("Coverage not installed, running tests without coverage")
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    return exit_code


if __name__ == "__main__":
    print("Running 3D Surface Reconstruction Tests")
    print("=" * 50)
    
    exit_code = run_tests()
    
    sys.exit(exit_code)