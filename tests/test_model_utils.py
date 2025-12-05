"""Tests for model_utils module."""

import pytest
from omegaconf import DictConfig

from src.model_utils import format_number, get_model_shortname, get_focus_suffix


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


class TestGetFocusSuffix:
    """
    Test suite for get_focus_suffix() utility function.

    This function builds the path suffix that encodes FOCUS tokenizer parameters
    to avoid cache collisions when changing vocabulary settings.
    """

    def test_basic_suffix_no_optional_flags(self):
        """Test basic suffix with just model, vocab, and samples."""
        args = DictConfig({
            'hf_model': 'facebook/xglm-564M',
            'focus': {
                'vocab_size': 50000,
                'num_samples': 100000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': False
            }
        })
        assert get_focus_suffix(args) == "xglm564m_v50k_s100k"

    def test_suffix_with_no_additional_flag(self):
        """Test suffix when not inheriting additional special tokens."""
        args = DictConfig({
            'hf_model': 'facebook/xglm-564M',
            'focus': {
                'vocab_size': 32768,
                'num_samples': 1000000,
                'inherit_additional_special_tokens': False,
                'use_seed_vocabulary': False
            }
        })
        assert get_focus_suffix(args) == "xglm564m_v32k_s1m_no-additional"

    def test_suffix_with_seed_vocabulary_default_params(self):
        """Test suffix with seed vocabulary using default parameters."""
        args = DictConfig({
            'hf_model': 'facebook/xglm-564M',
            'focus': {
                'vocab_size': 16384,
                'num_samples': 50000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 1,
                'seed_lambda': 1.0
            }
        })
        assert get_focus_suffix(args) == "xglm564m_v16k_s50k_seeded"

    def test_suffix_with_seed_vocabulary_custom_min_frequency(self):
        """Test suffix with non-default seed min frequency."""
        args = DictConfig({
            'hf_model': 'facebook/xglm-564M',
            'focus': {
                'vocab_size': 16384,
                'num_samples': 50000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 5,
                'seed_lambda': 1.0
            }
        })
        assert get_focus_suffix(args) == "xglm564m_v16k_s50k_seeded-min5"

    def test_suffix_with_seed_lambda(self):
        """Test suffix with non-default seed lambda."""
        args = DictConfig({
            'hf_model': 'facebook/xglm-564M',
            'focus': {
                'vocab_size': 16384,
                'num_samples': 50000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 1,
                'seed_lambda': 0.5
            }
        })
        assert get_focus_suffix(args) == "xglm564m_v16k_s50k_seeded-lambda0.5"

    def test_suffix_with_all_flags(self):
        """Test suffix with all optional flags enabled."""
        args = DictConfig({
            'hf_model': 'facebook/xglm-564M',
            'focus': {
                'vocab_size': 32768,
                'num_samples': 1000000,
                'inherit_additional_special_tokens': False,
                'use_seed_vocabulary': True,
                'seed_min_frequency': 10,
                'seed_lambda': 0.7
            }
        })
        assert get_focus_suffix(args) == "xglm564m_v32k_s1m_no-additional_seeded-min10-lambda0.7"

    def test_suffix_respects_number_formatting(self):
        """Test that vocab and sample sizes use format_number() correctly."""
        args = DictConfig({
            'hf_model': 'gpt2',
            'focus': {
                'vocab_size': 128000,
                'num_samples': 5000000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': False
            }
        })
        assert get_focus_suffix(args) == "gpt2_v128k_s5m"

    def test_suffix_with_different_model(self):
        """Test suffix generation with different base model."""
        args = DictConfig({
            'hf_model': 'meta-llama/Llama-2-7b',
            'focus': {
                'vocab_size': 50000,
                'num_samples': 100000,
                'inherit_additional_special_tokens': True,
                'use_seed_vocabulary': False
            }
        })
        assert get_focus_suffix(args) == "llama27b_v50k_s100k"
