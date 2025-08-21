#!/usr/bin/env python3
"""
Unified test runner for the gamma hedge project
Usage: python run_tests.py [--type=unit|integration|functional|all] [--verbose]
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add project root to path (script is now in scripts/ subdirectory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(cmd, cwd=None):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd or project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out after 5 minutes"
    except Exception as e:
        return False, "", str(e)

def discover_tests(test_type="all"):
    """Discover test files based on type"""
    test_files = []
    
    if test_type in ["unit", "all"]:
        test_files.extend(list(Path("tests/unit").glob("test_*.py")))
    
    if test_type in ["integration", "all"]:
        test_files.extend(list(Path("tests/integration").glob("test_*.py")))
        
    if test_type in ["functional", "all"]:
        test_files.extend(list(Path("tests/functional").glob("test_*.py")))
        test_files.extend(list(Path("tests/delta_hedge").glob("test_*.py")))
    
    return [str(f) for f in test_files if f.exists()]

def run_tests(test_type="all", verbose=False):
    """Run tests with the specified type"""
    print(f"Starting {test_type} tests...")
    print(f"Working directory: {project_root}")
    
    # Discover test files
    test_files = discover_tests(test_type)
    
    if not test_files:
        print(f"No {test_type} test files found")
        return False
    
    print(f"Found {len(test_files)} test files")
    if verbose:
        for test_file in test_files:
            print(f"   - {test_file}")
    
    # Run each test file
    passed = 0
    failed = 0
    errors = []
    
    for test_file in test_files:
        print(f"\nRunning: {test_file}")
        start_time = time.time()
        
        # Try with conda activate if available
        cmd = f'python "{test_file}"'
        
        success, stdout, stderr = run_command(cmd)
        elapsed = time.time() - start_time
        
        if success:
            print(f"PASSED ({elapsed:.2f}s)")
            if verbose and stdout:
                print(f"   Output: {stdout[:200]}{'...' if len(stdout) > 200 else ''}")
            passed += 1
        else:
            print(f"FAILED ({elapsed:.2f}s)")
            if stderr:
                print(f"   Error: {stderr[:300]}{'...' if len(stderr) > 300 else ''}")
            if verbose and stdout:
                print(f"   Output: {stdout[:200]}{'...' if len(stdout) > 200 else ''}")
            errors.append((test_file, stderr))
            failed += 1
    
    # Summary
    print(f"\nTest Summary:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if errors:
        print(f"\nFailed Tests:")
        for test_file, error in errors:
            print(f"   - {test_file}: {error[:100]}{'...' if len(error) > 100 else ''}")
    
    return failed == 0

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment...")
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("Error: Not in the gamma hedge project root directory")
        return False
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for required packages
    try:
        import torch
        import numpy as np
        print("Core dependencies available")
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run tests for gamma hedge project")
    parser.add_argument("--type", choices=["unit", "integration", "functional", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check environment, don't run tests")
    
    args = parser.parse_args()
    
    print("Gamma Hedge Test Runner")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    if args.check_only:
        print("Environment check passed")
        return
    
    # Run tests
    success = run_tests(args.type, args.verbose)
    
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()