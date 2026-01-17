"""Tests for model_utils module."""

from src.model_utils import format_number, get_model_shortname


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


class TestGetModelShortname:
    """
    Test suite for get_model_shortname() utility function.

    This function extracts a compact identifier from HuggingFace model names
    for use in directory paths.
    """

    def test_extract_from_org_path(self):
        """Test extracting model name from org/model format."""
        assert get_model_shortname("facebook/xglm-564M") == "xglm564m"
        assert get_model_shortname("meta-llama/Llama-2-7b") == "llama27b"

    def test_simple_model_name(self):
        """Test model names without organization prefix."""
        assert get_model_shortname("gpt2") == "gpt2"
        assert get_model_shortname("bert-base-uncased") == "bertbaseuncased"

    def test_removes_special_characters(self):
        """Test that dashes and dots are removed."""
        assert get_model_shortname("model-name-v2.1") == "modelnamev21"
        assert get_model_shortname("test.model.v1-beta") == "testmodelv1beta"

    def test_lowercases_output(self):
        """Test that output is lowercased."""
        assert get_model_shortname("GPT-2-XL") == "gpt2xl"
        assert get_model_shortname("BERT") == "bert"

    def test_complex_path(self):
        """Test with complex model path."""
        assert get_model_shortname("organization/sub-org/model-v1.2.3") == "modelv123"
