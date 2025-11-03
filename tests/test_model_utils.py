"""Tests for model_utils module."""

import pytest

from src.model_utils import format_number


class TestFormatNumber:
    """
    Test suite for the format_number() utility function.
    """

    def test_format_thousands(self):
        """
        Test that numbers >= 1000 are formatted with 'k' suffix.
        """
        assert format_number(50000) == "50k"
        assert format_number(1000) == "1k"
        assert format_number(1500) == "1k"  # Integer division

    def test_format_millions(self):
        """Test that numbers >= 1,000,000 are formatted with 'm' suffix."""
        assert format_number(1000000) == "1m"
        assert format_number(5000000) == "5m"
        assert format_number(1500000) == "1m"  # Integer division

    def test_format_small_numbers(self):
        """Test that numbers < 1000 are returned as strings unchanged."""
        assert format_number(999) == "999"
        assert format_number(100) == "100"
        assert format_number(1) == "1"
        assert format_number(0) == "0"
