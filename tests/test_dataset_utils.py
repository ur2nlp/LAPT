"""
Tests for dataset_utils module.

Current coverage:
- _load_plaintext_dataset: Basic loading, empty line filtering, caching, error cases
- _load_concat_dataset: Multi-source concatenation, caching, error cases

Testing approach:
- Use real temporary files (via pytest's tmp_path fixture) for I/O testing
- Use mocking (unittest.mock) to isolate components and avoid expensive operations
- Mock = replace real function calls with fake ones that return test data
  Example: When testing concat, we mock load_untokenized_dataset() to return
  small test datasets instead of actually loading OSCAR (which is huge/slow)

TODO: Add tests for:
- _load_multinomial_dataset (most critical missing piece)
  - Test temperature-scaled sampling with different alpha values
  - Test per-language dev split creation
  - Test handling of empty sources
  - Test empirical vs uniform sampling modes
- _load_plaintext_dir (directory-based loading)
- load_untokenized_dataset (main dispatcher function)
  - Test routing to correct loader based on config type
  - Test OSCAR dataset loading
- load_or_tokenize_dataset (tokenization layer)
  - Test tokenization with different max_length values
  - Test dev split creation for non-multinomial datasets
- Edge cases:
  - Very large files (memory efficiency)
  - Unicode handling
  - Concurrent access to cache
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from datasets import Dataset, DatasetDict, load_from_disk
from omegaconf import DictConfig

from dataset_utils import (
    _load_plaintext_dataset,
    _load_concat_dataset,
    load_untokenized_dataset,
)


class TestPlaintextLoader:
    """
    Tests for loading plaintext files into dataset format.

    Testing strategy:
    - Use real files (via tmp_path) to test actual I/O
    - Use small synthetic data for speed
    - Test both fresh load and caching behavior
    - Test error cases (missing file, empty file)
    """

    def test_load_plaintext_basic(self, tmp_path):
        """
        Test loading a simple plaintext file with multiple lines.

        Strategy: Create real text file, verify it's converted to Dataset correctly.

        What we verify:
        1. Creates 'untokenized' subdirectory
        2. Loads all non-empty lines
        3. Returns correct path
        4. Dataset has correct structure ({'train': Dataset})
        """
        # Setup: Create a test plaintext file
        test_file = tmp_path / "data.txt"
        test_lines = [
            "First line of text",
            "Second line",
            "Third line with more words",
        ]
        test_file.write_text("\n".join(test_lines))

        cache_dir = tmp_path / "cache"

        # Act: Load the plaintext file
        result_path = _load_plaintext_dataset(
            cache_dir=str(cache_dir),
            file_path=str(test_file)
        )

        # Assert: Check return value
        expected_path = cache_dir / "untokenized"
        assert result_path == str(expected_path)
        assert expected_path.exists()

        # Assert: Load and verify dataset contents
        dataset_dict = load_from_disk(result_path)
        assert 'train' in dataset_dict
        assert len(dataset_dict['train']) == 3

        # Verify actual content
        texts = dataset_dict['train']['text']
        assert texts[0] == "First line of text"
        assert texts[1] == "Second line"
        assert texts[2] == "Third line with more words"

    def test_load_plaintext_strips_empty_lines(self, tmp_path):
        """
        Test that empty lines are filtered out.

        This is important for cleaning data - blank lines shouldn't become training examples.
        """
        test_file = tmp_path / "data.txt"
        test_file.write_text("Line 1\n\n\nLine 2\n   \nLine 3")

        cache_dir = tmp_path / "cache"
        result_path = _load_plaintext_dataset(str(cache_dir), str(test_file))

        dataset_dict = load_from_disk(result_path)
        # Should only have 3 non-empty lines
        assert len(dataset_dict['train']) == 3

    def test_load_plaintext_caching(self, tmp_path):
        """
        Test that calling twice doesn't reload - uses cached version.

        Strategy: Load once, modify source file, load again.
        Second load should still have original data (from cache).
        """
        test_file = tmp_path / "data.txt"
        test_file.write_text("Original line")

        cache_dir = tmp_path / "cache"

        # First load
        result_path1 = _load_plaintext_dataset(str(cache_dir), str(test_file))
        dataset1 = load_from_disk(result_path1)
        assert dataset1['train']['text'][0] == "Original line"

        # Modify source file
        test_file.write_text("Modified line")

        # Second load - should use cache
        result_path2 = _load_plaintext_dataset(str(cache_dir), str(test_file))
        dataset2 = load_from_disk(result_path2)

        # Should still have original data (cached)
        assert dataset2['train']['text'][0] == "Original line"

    def test_load_plaintext_missing_file(self, tmp_path):
        """
        Test that loading a non-existent file raises appropriate error.

        Error handling test - should fail fast with clear message.
        """
        cache_dir = tmp_path / "cache"
        nonexistent_file = tmp_path / "doesnt_exist.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_plaintext_dataset(str(cache_dir), str(nonexistent_file))

        assert "not found" in str(exc_info.value).lower()

    def test_load_plaintext_empty_file(self, tmp_path):
        """
        Test that loading a file with only empty lines raises error.

        Edge case: file exists but has no actual content.
        """
        test_file = tmp_path / "empty.txt"
        test_file.write_text("\n\n   \n\n")  # Only whitespace

        cache_dir = tmp_path / "cache"

        with pytest.raises(ValueError) as exc_info:
            _load_plaintext_dataset(str(cache_dir), str(test_file))

        assert "no non-empty lines" in str(exc_info.value).lower()


class TestConcatLoader:
    """
    Tests for concatenating multiple dataset sources.

    Testing strategy:
    - Mock the recursive load_untokenized_dataset() calls
    - Create synthetic datasets to concatenate
    - Verify correct concatenation logic
    - Test error cases (empty sources list)

    Key insight: We don't want to actually load OSCAR or other datasets,
    so we mock load_untokenized_dataset() to return synthetic data.
    """

    def test_concat_two_sources(self, tmp_path):
        """
        Test concatenating two plaintext sources.

        Mocking strategy:
        - We'll create real datasets in tmp_path for the sources
        - Mock load_untokenized_dataset to return those paths
        - Verify concat combines them correctly

        This is a "semi-mock" approach - we create real datasets but mock
        the recursive loading call.
        """
        # Create two synthetic source datasets
        source1_dir = tmp_path / "source1" / "untokenized"
        source1_dir.mkdir(parents=True)
        dataset1 = DatasetDict({
            'train': Dataset.from_dict({'text': ['Line 1', 'Line 2']})
        })
        dataset1.save_to_disk(str(source1_dir))

        source2_dir = tmp_path / "source2" / "untokenized"
        source2_dir.mkdir(parents=True)
        dataset2 = DatasetDict({
            'train': Dataset.from_dict({'text': ['Line 3', 'Line 4', 'Line 5']})
        })
        dataset2.save_to_disk(str(source2_dir))

        # Define sources configuration
        sources = [
            {'type': 'plaintext', 'path': 'dummy1.txt'},
            {'type': 'plaintext', 'path': 'dummy2.txt'},
        ]

        cache_dir = tmp_path / "concat_cache"

        # Mock the recursive calls to load_untokenized_dataset
        # It will be called twice (once per source), return our synthetic paths
        with patch('dataset_utils.load_untokenized_dataset') as mock_load:
            mock_load.side_effect = [str(source1_dir), str(source2_dir)]

            # Act: Concatenate the sources
            result_path = _load_concat_dataset(
                cache_dir=str(cache_dir),
                sources=sources
            )

        # Assert: Load and verify concatenated dataset
        expected_path = cache_dir / "untokenized"
        assert result_path == str(expected_path)

        dataset_dict = load_from_disk(result_path)
        assert 'train' in dataset_dict

        # Should have 2 + 3 = 5 lines total
        assert len(dataset_dict['train']) == 5

        # Verify order is preserved (source1 then source2)
        texts = dataset_dict['train']['text']
        assert texts[0] == 'Line 1'
        assert texts[1] == 'Line 2'
        assert texts[2] == 'Line 3'
        assert texts[3] == 'Line 4'
        assert texts[4] == 'Line 5'

        # Verify load_untokenized_dataset was called correctly
        assert mock_load.call_count == 2

    def test_concat_empty_sources(self, tmp_path):
        """
        Test that concatenating empty sources list raises error.

        Edge case: can't concatenate nothing.
        """
        cache_dir = tmp_path / "cache"
        sources = []

        with pytest.raises(ValueError) as exc_info:
            _load_concat_dataset(str(cache_dir), sources)

        assert "empty" in str(exc_info.value).lower()

    def test_concat_caching(self, tmp_path):
        """
        Test that concat respects caching - doesn't reload if cache exists.

        Strategy: Call once, verify sources were loaded.
        Call again, verify sources were NOT loaded again.
        """
        # Create a synthetic source dataset
        source_dir = tmp_path / "source" / "untokenized"
        source_dir.mkdir(parents=True)
        dataset = DatasetDict({
            'train': Dataset.from_dict({'text': ['Data']})
        })
        dataset.save_to_disk(str(source_dir))

        sources = [{'type': 'plaintext', 'path': 'dummy.txt'}]
        cache_dir = tmp_path / "concat_cache"

        # First call - should load sources
        with patch('dataset_utils.load_untokenized_dataset') as mock_load:
            mock_load.return_value = str(source_dir)
            result_path1 = _load_concat_dataset(str(cache_dir), sources)
            first_call_count = mock_load.call_count

        assert first_call_count == 1

        # Second call - should use cache, NOT call load_untokenized_dataset
        with patch('dataset_utils.load_untokenized_dataset') as mock_load:
            mock_load.return_value = str(source_dir)
            result_path2 = _load_concat_dataset(str(cache_dir), sources)
            second_call_count = mock_load.call_count

        # Should NOT have called load_untokenized_dataset because cache exists
        assert second_call_count == 0
        assert result_path1 == result_path2
