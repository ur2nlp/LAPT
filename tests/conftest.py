"""
Pytest configuration and shared fixtures.

conftest.py is a special pytest file that:
1. Provides configuration for the entire test suite
2. Defines fixtures that are available to all test files
3. Runs before any tests are collected
"""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

# Import paths are declared in pyproject.toml ([tool.pytest.ini_options]
# pythonpath), not manipulated here -- that setting is applied before collection,
# whereas a sys.path edit in this file only happens once conftest itself imports.


@pytest.fixture
def sample_jsonl_path():
    """
    Provides path to sample JSONL training data.

    Strategy: Use a small static fixture file rather than generating data dynamically.
    Why: Faster, reproducible, and sufficient for testing tokenizer training logic.
    """
    return Path(__file__).parent / "fixtures" / "sample_training.jsonl"


@pytest.fixture(scope="session")
def base_tokenizer():
    """
    Provides the XGLM base tokenizer for tests.

    Strategy decisions:
    - scope="session": Download once per test session, reuse across all tests
    - Use real tokenizer: Mocking the tokenizer object is complex and brittle
    - XGLM-564M: Small model, fast download, matches production usage

    The session scope means this fixture runs once at the start of the test session
    and the same tokenizer instance is shared across all tests that request it.
    This avoids downloading the tokenizer for every test.
    """
    return AutoTokenizer.from_pretrained("facebook/xglm-564M", use_fast=True)
