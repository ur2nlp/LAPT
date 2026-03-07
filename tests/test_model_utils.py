"""Tests for model_utils module."""

import pytest
from omegaconf import OmegaConf

from src.model_utils import (
    format_number, get_model_shortname, get_tokenized_path,
    is_local_model_path, get_init_model_identifier,
)
from src.__main__ import _validate_init_model_id


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


class TestIsLocalModelPath:
    """Test detection of local model paths vs HuggingFace model names."""

    def test_hf_model_name(self):
        assert not is_local_model_path("facebook/xglm-564M")

    def test_absolute_path(self):
        assert is_local_model_path("/scratch/user/models/checkpoint")

    def test_relative_path(self):
        assert is_local_model_path("./models/checkpoint")

    def test_home_path(self):
        assert is_local_model_path("~/models/checkpoint")


class TestGetInitModelIdentifier:
    """Test model identifier resolution with and without init_model_id."""

    def test_uses_init_model_id_when_provided(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-1.7B',
            'init_model_id': 'xglm1b',
        })
        assert get_init_model_identifier(args) == "xglm1b"

    def test_derives_from_hf_model_when_no_init_model_id(self):
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
        })
        assert get_init_model_identifier(args) == "xglm564m"


class TestValidateInitModelId:
    """Test validation that init_model_id is required for local model paths."""

    def test_passes_with_hf_model_name(self):
        """No error for standard HuggingFace model names."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
        })
        _validate_init_model_id(args)

    def test_passes_with_local_path_and_init_model_id(self):
        """No error when init_model_id is provided for a local path."""
        args = OmegaConf.create({
            'hf_model': '/scratch/user/models/v81/best-checkpoint',
            'init_model_id': 'v81',
        })
        _validate_init_model_id(args)

    def test_raises_for_local_path_without_init_model_id(self):
        """Error when hf_model is a local path but init_model_id is missing."""
        args = OmegaConf.create({
            'hf_model': '/scratch/user/models/checkpoint',
        })
        with pytest.raises(ValueError, match="init_model_id is required"):
            _validate_init_model_id(args)


class TestGetTokenizedPath:
    """Test tokenized dataset path generation."""

    def test_no_focus(self):
        """Without FOCUS, path includes model shortname."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {'enabled': False},
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_xglm564m"

    def test_no_focus_with_init_model_id(self):
        """Without FOCUS but with init_model_id, uses init_model_id."""
        args = OmegaConf.create({
            'hf_model': '/local/checkpoint',
            'init_model_id': 'v81',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {'enabled': False},
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_v81"

    def test_with_focus(self):
        """With FOCUS, path includes full focus suffix."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000,
            },
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_xglm564m_focus-v16k-s100k"

    def test_with_focus_and_init_model_id(self):
        """With FOCUS and init_model_id, uses init_model_id in path."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-1.7B',
            'init_model_id': 'xglm1b',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 32768,
                'num_samples': 5000000,
            },
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert path == "data/test/tokenized_xglm1b_focus-v32k-s5m"

    def test_no_redundant_focus_prefix(self):
        """Path should not contain 'tokenized_focus_' (old redundant format)."""
        args = OmegaConf.create({
            'hf_model': 'facebook/xglm-564M',
            'dataset': {'cache_dir': 'data/test'},
            'focus': {
                'enabled': True,
                'vocab_size': 16384,
                'num_samples': 100000,
            },
            'seed': 42,
        })
        path = get_tokenized_path(args)
        assert "tokenized_focus_" not in path
