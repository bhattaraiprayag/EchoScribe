# tests/run_tests.py

import os
import sys

import pytest


def main():
    """Runs the pytest test suite."""
    print("--- Running Transcription App Test Suite ---")

    test_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.dirname(test_dir)
    sys.path.insert(0, project_root)

    ## Run pytest on the test directory
    ## -v: verbose output
    ## -s: show print statements
    ## --disable-warnings: hides deprecation warnings from libraries
    retcode = pytest.main(["-v", "-s", "--disable-warnings", test_dir])

    print(f"--- Test Suite Finished with exit code: {retcode} ---")
    return retcode


if __name__ == "__main__":
    sys.exit(main())
