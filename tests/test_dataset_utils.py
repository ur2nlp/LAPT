"""
Tests for dataset_utils module.

Current coverage:
- _load_plaintext_dataset: Basic loading, empty line filtering, caching, error cases
- _load_concat_dataset: Multi-source concatenation, caching, error cases
- _tokenize_instruction_examples: Label masking, truncation, batching
- DataCollatorForInstructionTuning: Padding, label preservation, tensor output

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
- load_tokenized_dataset (tokenization layer)
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
    _load_plaintext_dir_dataset,
    _load_concat_dataset,
    _compute_sampling_probs,
    load_untokenized_dataset,
    load_external_eval_set,
    _tokenize_instruction_examples,
    DataCollatorForInstructionTuning,
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


class TestPlaintextDirLoader:
    """
    Tests for loading multiple plaintext files from a directory.

    Testing strategy:
    - Use real files (via tmp_path) to test actual I/O
    - Test pattern matching (glob pattern filtering)
    - Test error cases (no matching files, empty directory)
    """

    def test_load_plaintext_dir_basic(self, tmp_path):
        """
        Test loading multiple plaintext files from directory with glob pattern.

        Strategy: Create multiple .txt files, verify they're all loaded and concatenated.
        """
        # Setup: Create directory with multiple text files
        data_dir = tmp_path / "texts"
        data_dir.mkdir()

        (data_dir / "file1.txt").write_text("Line 1\nLine 2")
        (data_dir / "file2.txt").write_text("Line 3\nLine 4\nLine 5")
        (data_dir / "file3.txt").write_text("Line 6")
        (data_dir / "other.md").write_text("Ignore me")  # Wrong extension

        cache_dir = tmp_path / "cache"

        # Act: Load with *.txt pattern
        result_path = _load_plaintext_dir_dataset(
            cache_dir=str(cache_dir),
            directory=str(data_dir),
            pattern="*.txt"
        )

        # Assert: Check return value
        expected_path = cache_dir / "untokenized"
        assert result_path == str(expected_path)
        assert expected_path.exists()

        # Assert: Load and verify dataset contents
        dataset_dict = load_from_disk(result_path)
        assert 'train' in dataset_dict

        # Should have 6 lines from 3 .txt files (ignoring .md file)
        assert len(dataset_dict['train']) == 6

        # Verify all lines are present (order may vary based on file system)
        texts = set(dataset_dict['train']['text'])
        expected_lines = {f"Line {i}" for i in range(1, 7)}
        assert texts == expected_lines

    def test_load_plaintext_dir_different_pattern(self, tmp_path):
        """
        Test loading with custom glob pattern (e.g., *.on.txt for Old Norse).
        """
        data_dir = tmp_path / "sagas"
        data_dir.mkdir()

        (data_dir / "saga1.on.txt").write_text("Old Norse 1")
        (data_dir / "saga2.on.txt").write_text("Old Norse 2")
        (data_dir / "saga1.en.txt").write_text("English translation")  # Different pattern

        cache_dir = tmp_path / "cache"

        result_path = _load_plaintext_dir_dataset(
            cache_dir=str(cache_dir),
            directory=str(data_dir),
            pattern="*.on.txt"
        )

        dataset_dict = load_from_disk(result_path)

        # Should only have 2 lines from *.on.txt files
        assert len(dataset_dict['train']) == 2
        texts = set(dataset_dict['train']['text'])
        assert texts == {"Old Norse 1", "Old Norse 2"}

    def test_load_plaintext_dir_strips_empty_lines(self, tmp_path):
        """
        Test that empty lines are filtered out across all files.
        """
        data_dir = tmp_path / "texts"
        data_dir.mkdir()

        (data_dir / "file1.txt").write_text("Line 1\n\n\nLine 2")
        (data_dir / "file2.txt").write_text("   \nLine 3\n\n")

        cache_dir = tmp_path / "cache"

        result_path = _load_plaintext_dir_dataset(
            cache_dir=str(cache_dir),
            directory=str(data_dir),
            pattern="*.txt"
        )

        dataset_dict = load_from_disk(result_path)

        # Should only have 3 non-empty lines total
        assert len(dataset_dict['train']) == 3

    def test_load_plaintext_dir_no_matching_files(self, tmp_path):
        """
        Test that loading directory with no matching files raises error.

        Edge case: directory exists but has no files matching pattern.
        """
        data_dir = tmp_path / "empty"
        data_dir.mkdir()

        # Create files that don't match pattern
        (data_dir / "file.md").write_text("Markdown file")
        (data_dir / "file.py").write_text("Python file")

        cache_dir = tmp_path / "cache"

        with pytest.raises(ValueError) as exc_info:
            _load_plaintext_dir_dataset(
                cache_dir=str(cache_dir),
                directory=str(data_dir),
                pattern="*.txt"
            )

        assert "no files found" in str(exc_info.value).lower()

    def test_load_plaintext_dir_nonexistent_directory(self, tmp_path):
        """
        Test that loading non-existent directory raises appropriate error.
        """
        cache_dir = tmp_path / "cache"
        nonexistent_dir = tmp_path / "doesnt_exist"

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_plaintext_dir_dataset(
                cache_dir=str(cache_dir),
                directory=str(nonexistent_dir),
                pattern="*.txt"
            )

        assert "not found" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_load_plaintext_dir_caching(self, tmp_path):
        """
        Re-loading with the same file set returns cached data; adding a new
        file changes the concat source list and is caught as a source-cache
        mismatch rather than silently serving stale data.
        """
        data_dir = tmp_path / "texts"
        data_dir.mkdir()

        (data_dir / "file1.txt").write_text("Original line")

        cache_dir = tmp_path / "cache"

        result_path1 = _load_plaintext_dir_dataset(
            cache_dir=str(cache_dir),
            directory=str(data_dir),
            pattern="*.txt"
        )
        dataset1 = load_from_disk(result_path1)
        assert len(dataset1['train']) == 1
        assert dataset1['train']['text'][0] == "Original line"

        result_path2 = _load_plaintext_dir_dataset(
            cache_dir=str(cache_dir),
            directory=str(data_dir),
            pattern="*.txt"
        )
        dataset2 = load_from_disk(result_path2)
        assert len(dataset2['train']) == 1
        assert dataset2['train']['text'][0] == "Original line"

        (data_dir / "file2.txt").write_text("New line")
        with pytest.raises(ValueError, match="source cache"):
            _load_plaintext_dir_dataset(
                cache_dir=str(cache_dir),
                directory=str(data_dir),
                pattern="*.txt"
            )

    def test_load_plaintext_dir_all_files_empty(self, tmp_path):
        """
        Test error when all matching files contain only whitespace.

        Edge case: files exist and match pattern but have no content.
        """
        data_dir = tmp_path / "empty_files"
        data_dir.mkdir()

        (data_dir / "file1.txt").write_text("\n\n   \n")
        (data_dir / "file2.txt").write_text("    \n")

        cache_dir = tmp_path / "cache"

        with pytest.raises(ValueError) as exc_info:
            _load_plaintext_dir_dataset(
                cache_dir=str(cache_dir),
                directory=str(data_dir),
                pattern="*.txt"
            )

        assert "no non-empty lines" in str(exc_info.value).lower()


class TestExternalEvalSetLoader:
    """
    Tests for loading and tokenizing external evaluation datasets.

    Testing strategy:
    - Use real files (via tmp_path) to test actual I/O
    - Mock tokenizer for controlled tokenization behavior
    - Test both plaintext and JSONL formats
    - Test error cases (missing file, invalid format, missing columns)
    """

    def test_load_external_eval_plaintext(self, tmp_path):
        """
        Test loading a plaintext external eval set.

        Strategy: Create real text file, mock tokenizer, verify tokenization.
        """
        # Setup: Create test plaintext file
        test_file = tmp_path / "eval_data.txt"
        test_lines = [
            "First eval example",
            "Second eval example",
            "Third eval example",
        ]
        test_file.write_text("\n".join(test_lines))

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3]] * 3,
            'attention_mask': [[1, 1, 1]] * 3
        }

        eval_config = {
            'name': 'held_out',
            'path': str(test_file),
            'format': 'plaintext'
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=512
        )

        # Assert: Verify dataset structure
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 3

        # Verify tokenizer was called correctly
        assert mock_tokenizer.called
        call_args = mock_tokenizer.call_args
        assert call_args[1]['truncation'] is True
        assert call_args[1]['max_length'] == 512

    def test_load_external_eval_jsonl(self, tmp_path):
        """
        Test loading a JSONL external eval set.

        Strategy: Create JSONL file, verify correct field extraction.
        """
        import json

        # Setup: Create test JSONL file
        test_file = tmp_path / "eval_data.jsonl"
        test_data = [
            {"text": "Example 1", "label": "A"},
            {"text": "Example 2", "label": "B"},
        ]
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2]] * 2,
            'attention_mask': [[1, 1]] * 2
        }

        eval_config = {
            'name': 'test_set',
            'path': str(test_file),
            'format': 'jsonl'
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=256
        )

        # Assert: Verify dataset
        assert len(dataset) == 2
        assert mock_tokenizer.called

    def test_load_external_eval_jsonl_custom_column(self, tmp_path):
        """
        Test loading JSONL with custom text column name.

        Strategy: Use different field name, verify it's read correctly.
        """
        import json

        test_file = tmp_path / "eval_data.jsonl"
        test_data = [
            {"content": "Custom field 1"},
            {"content": "Custom field 2"},
        ]
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1]] * 2,
            'attention_mask': [[1]] * 2
        }

        eval_config = {
            'name': 'custom',
            'path': str(test_file),
            'format': 'jsonl',
            'text_column': 'content'
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=256
        )

        # Assert: Should succeed
        assert len(dataset) == 2

    def test_load_external_eval_missing_file(self, tmp_path):
        """
        Test that loading non-existent file raises appropriate error.
        """
        mock_tokenizer = MagicMock()

        eval_config = {
            'name': 'missing',
            'path': str(tmp_path / 'nonexistent.txt')
        }

        with pytest.raises(ValueError) as exc_info:
            load_external_eval_set(
                eval_config=eval_config,
                tokenizer=mock_tokenizer,
                max_length=512
            )

        assert "not found" in str(exc_info.value).lower()

    def test_load_external_eval_invalid_format(self, tmp_path):
        """
        Test that unsupported format raises appropriate error.
        """
        test_file = tmp_path / "data.csv"
        test_file.write_text("col1,col2\nval1,val2")

        mock_tokenizer = MagicMock()

        eval_config = {
            'name': 'csv_test',
            'path': str(test_file),
            'format': 'csv'  # Unsupported format
        }

        with pytest.raises(ValueError) as exc_info:
            load_external_eval_set(
                eval_config=eval_config,
                tokenizer=mock_tokenizer,
                max_length=512
            )

        assert "unsupported format" in str(exc_info.value).lower()

    def test_load_external_eval_jsonl_missing_column(self, tmp_path):
        """
        Test that JSONL with missing text column raises error.
        """
        import json

        test_file = tmp_path / "bad_data.jsonl"
        test_data = [{"wrong_field": "value"}]
        with open(test_file, 'w') as f:
            f.write(json.dumps(test_data[0]) + '\n')

        mock_tokenizer = MagicMock()

        eval_config = {
            'name': 'bad_jsonl',
            'path': str(test_file),
            'format': 'jsonl',
            'text_column': 'text'  # Expected column that doesn't exist
        }

        with pytest.raises(ValueError) as exc_info:
            load_external_eval_set(
                eval_config=eval_config,
                tokenizer=mock_tokenizer,
                max_length=512
            )

        assert "missing" in str(exc_info.value).lower()

    def test_load_external_eval_strips_empty_lines(self, tmp_path):
        """
        Test that empty lines are filtered out from plaintext files.
        """
        test_file = tmp_path / "data.txt"
        test_file.write_text("Line 1\n\n\nLine 2\n   \n")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': [[1]] * 2,
            'attention_mask': [[1]] * 2
        }

        eval_config = {
            'name': 'test',
            'path': str(test_file)
        }

        # Act: Load and tokenize
        dataset = load_external_eval_set(
            eval_config=eval_config,
            tokenizer=mock_tokenizer,
            max_length=512
        )

        # Assert: Should only have 2 non-empty lines
        assert len(dataset) == 2


class TestTokenizeInstructionExamples:
    """
    Tests for _tokenize_instruction_examples function.

    This function tokenizes instruction-tuning data with label masking:
    - Prompt tokens get label=-100 (ignored in loss)
    - Response tokens get their actual token IDs as labels

    Testing strategy:
    - Use real tokenizer (base_tokenizer fixture) for accurate token counts
    - Test various prompt/response combinations
    - Verify label masking is correct
    - Test truncation behavior
    """

    def test_basic_tokenization(self, base_tokenizer):
        """
        Test basic tokenization with simple prompt and response.

        Verifies:
        1. Output has correct keys (input_ids, attention_mask, labels)
        2. Prompt tokens are masked (-100 in labels)
        3. Response tokens have actual token IDs in labels
        4. input_ids and labels have same length
        """
        from dataset_utils import _tokenize_instruction_examples

        examples = {
            'prompt': ['Translate to Gothic: hello\nResponse:'],
            'response': [' world']
        }

        result = _tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

        # Check output structure
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result

        input_ids = result['input_ids'][0]
        labels = result['labels'][0]

        # Same length
        assert len(input_ids) == len(labels)

        # Count masked vs unmasked labels
        num_masked = sum(1 for l in labels if l == -100)
        num_unmasked = sum(1 for l in labels if l != -100)

        # Should have some masked (prompt) and some unmasked (response)
        assert num_masked > 0, "Should have masked prompt tokens"
        assert num_unmasked > 0, "Should have unmasked response tokens"

        # Unmasked labels should match corresponding input_ids
        for i, label in enumerate(labels):
            if label != -100:
                assert label == input_ids[i], f"Label at position {i} should match input_id"

    def test_prompt_fully_masked(self, base_tokenizer):
        """
        Test that the entire prompt portion is masked.

        Strategy: Tokenize prompt alone, count tokens, verify that many are masked.
        """
        from dataset_utils import _tokenize_instruction_examples

        prompt = "This is a test prompt with several words\nResponse:"
        response = " Yes"

        examples = {
            'prompt': [prompt],
            'response': [response]
        }

        result = _tokenize_instruction_examples(examples, base_tokenizer, max_length=512)
        labels = result['labels'][0]

        # Tokenize prompt separately to count its tokens
        prompt_tokens = base_tokenizer(prompt, add_special_tokens=True)
        prompt_length = len(prompt_tokens['input_ids'])

        # First prompt_length labels should all be -100
        for i in range(prompt_length):
            assert labels[i] == -100, f"Label at position {i} should be -100 (prompt portion)"

    def test_multiple_examples(self, base_tokenizer):
        """
        Test batched tokenization with multiple examples.

        Verifies each example is tokenized independently.
        """
        from dataset_utils import _tokenize_instruction_examples

        examples = {
            'prompt': [
                'Question: What is 2+2?\nResponse:',
                'Translate: hello\nResponse:'
            ],
            'response': [
                ' 4',
                ' hola'
            ]
        }

        result = _tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

        # Should have 2 examples
        assert len(result['input_ids']) == 2
        assert len(result['labels']) == 2
        assert len(result['attention_mask']) == 2

        # Each example should have different lengths (different prompts)
        len1 = len(result['input_ids'][0])
        len2 = len(result['input_ids'][1])
        # They could be same length by chance, but labels should differ
        assert result['labels'][0] != result['labels'][1]

    def test_truncation(self, base_tokenizer):
        """
        Test that sequences are truncated to max_length.

        Strategy: Use very short max_length, verify output is truncated.
        """
        from dataset_utils import _tokenize_instruction_examples

        # Long prompt and response
        examples = {
            'prompt': ['This is a very long prompt ' * 20 + '\nResponse:'],
            'response': [' This is a very long response ' * 20]
        }

        max_length = 50
        result = _tokenize_instruction_examples(examples, base_tokenizer, max_length=max_length)

        # Should be truncated to max_length
        assert len(result['input_ids'][0]) <= max_length
        assert len(result['labels'][0]) <= max_length
        assert len(result['attention_mask'][0]) <= max_length

    def test_empty_response(self, base_tokenizer):
        """
        Test handling of empty response (edge case).

        With empty response, all labels should be -100.
        """
        from dataset_utils import _tokenize_instruction_examples

        examples = {
            'prompt': ['Prompt text\nResponse:'],
            'response': ['']
        }

        result = _tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

        labels = result['labels'][0]

        # All labels should be -100 (no response tokens)
        assert all(l == -100 for l in labels), "All labels should be -100 for empty response"

    def test_response_starts_with_space(self, base_tokenizer):
        """
        Test that response with leading space is handled correctly.

        Our JSONL format uses ' response' (with leading space) to ensure
        proper tokenization as a continuation.
        """
        from dataset_utils import _tokenize_instruction_examples

        examples = {
            'prompt': ['Test\nResponse:'],
            'response': [' answer']  # Note leading space
        }

        result = _tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

        # Should tokenize without errors
        assert len(result['input_ids'][0]) > 0
        assert len(result['labels'][0]) > 0


class TestDataCollatorForInstructionTuning:
    """
    Tests for DataCollatorForInstructionTuning.

    This collator handles batching of instruction-tuning data:
    - Pads input_ids with pad_token_id
    - Pads attention_mask with 0
    - Pads labels with -100 (so padded positions don't contribute to loss)

    Testing strategy:
    - Test with features of different lengths to verify padding
    - Test single-example batch (no padding needed)
    - Verify tensor shapes and dtypes
    """

    def test_basic_padding(self, base_tokenizer):
        """
        Test that features of different lengths are padded correctly.

        Verifies:
        1. All sequences padded to same length
        2. input_ids padded with pad_token_id
        3. attention_mask padded with 0
        4. labels padded with -100
        """
        from dataset_utils import DataCollatorForInstructionTuning
        import torch

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        # Features with different lengths
        features = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                'attention_mask': [1, 1, 1, 1, 1],
                'labels': [-100, -100, 3, 4, 5]
            },
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, 2, 3]
            }
        ]

        batch = collator(features)

        # Check shapes - should be padded to longest (5)
        assert batch['input_ids'].shape == (2, 5)
        assert batch['attention_mask'].shape == (2, 5)
        assert batch['labels'].shape == (2, 5)

        # Check dtypes
        assert batch['input_ids'].dtype == torch.long
        assert batch['labels'].dtype == torch.long

        # Check padding values for second (shorter) sequence
        # Last 2 positions should be padded
        pad_token_id = base_tokenizer.pad_token_id or base_tokenizer.eos_token_id
        assert batch['input_ids'][1, 3].item() == pad_token_id
        assert batch['input_ids'][1, 4].item() == pad_token_id
        assert batch['attention_mask'][1, 3].item() == 0
        assert batch['attention_mask'][1, 4].item() == 0
        assert batch['labels'][1, 3].item() == -100
        assert batch['labels'][1, 4].item() == -100

    def test_single_example_batch(self, base_tokenizer):
        """
        Test batch with single example (no padding needed).
        """
        from dataset_utils import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, 2, 3]
            }
        ]

        batch = collator(features)

        # Should have batch size 1
        assert batch['input_ids'].shape == (1, 3)
        assert batch['labels'].shape == (1, 3)

        # Values should be unchanged
        assert batch['input_ids'][0].tolist() == [1, 2, 3]
        assert batch['labels'][0].tolist() == [-100, 2, 3]

    def test_preserves_masked_labels(self, base_tokenizer):
        """
        Test that -100 labels are preserved (not overwritten by padding logic).
        """
        from dataset_utils import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3, 4],
                'attention_mask': [1, 1, 1, 1],
                'labels': [-100, -100, 3, 4]  # First two are masked
            }
        ]

        batch = collator(features)

        # Original -100 values should be preserved
        assert batch['labels'][0, 0].item() == -100
        assert batch['labels'][0, 1].item() == -100
        assert batch['labels'][0, 2].item() == 3
        assert batch['labels'][0, 3].item() == 4

    def test_handles_all_masked_labels(self, base_tokenizer):
        """
        Test handling of sequence where all labels are -100.

        This can happen with empty responses or very long prompts.
        """
        from dataset_utils import DataCollatorForInstructionTuning

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, -100, -100]  # All masked
            }
        ]

        batch = collator(features)

        # Should work without errors
        assert batch['labels'].shape == (1, 3)
        assert batch['labels'][0].tolist() == [-100, -100, -100]

    def test_returns_pytorch_tensors(self, base_tokenizer):
        """
        Test that output is PyTorch tensors, not lists.
        """
        from dataset_utils import DataCollatorForInstructionTuning
        import torch

        collator = DataCollatorForInstructionTuning(base_tokenizer)

        features = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [-100, 2, 3]
            }
        ]

        batch = collator(features)

        assert isinstance(batch['input_ids'], torch.Tensor)
        assert isinstance(batch['attention_mask'], torch.Tensor)
        assert isinstance(batch['labels'], torch.Tensor)


class TestMixedInstructionPlaintextDatasets:
    """
    Tests for mixing instruction (prompt/response) and plaintext (text) datasets.

    This is important for instruction tuning where you might want to:
    - Include instruction data (translation, FLAN) with loss masking
    - Include LM data (monolingual text) with standard causal LM loss

    The challenge: these have different column structures that need to be
    handled during concatenation and tokenization.
    """

    def test_concatenate_different_column_schemas_unions_columns(self, tmp_path):
        """
        Test that concatenating datasets with different columns creates union.

        HuggingFace Datasets unions all columns and fills missing values with None.
        This means we can mix instruction and plaintext data, but need to handle
        the None values during tokenization.
        """
        from datasets import Dataset, concatenate_datasets

        instruction_data = Dataset.from_dict({
            'prompt': ['Translate: hello\nResponse:'],
            'response': [' hola']
        })

        plaintext_data = Dataset.from_dict({
            'text': ['This is plain text.']
        })

        # Concatenation works - creates union of columns
        combined = concatenate_datasets([instruction_data, plaintext_data])

        # Should have all three columns
        assert set(combined.column_names) == {'prompt', 'response', 'text'}

        # Instruction row: has prompt/response, text is None
        assert combined[0]['prompt'] == 'Translate: hello\nResponse:'
        assert combined[0]['response'] == ' hola'
        assert combined[0]['text'] is None

        # Plaintext row: has text, prompt/response are None
        assert combined[1]['text'] == 'This is plain text.'
        assert combined[1]['prompt'] is None
        assert combined[1]['response'] is None

    def test_mixed_tokenization_with_normalized_columns(self, tmp_path, base_tokenizer):
        """
        Test that mixed datasets work when columns are normalized.

        Strategy: Instruction data can include a 'text' column (full sequence)
        alongside 'prompt'/'response' for label masking during tokenization.
        """
        from datasets import Dataset, DatasetDict, concatenate_datasets
        from dataset_utils import load_tokenized_dataset

        # Create instruction data with all three columns
        instruction_data = Dataset.from_dict({
            'text': ['Translate: hello\nResponse: hola'],
            'prompt': ['Translate: hello\nResponse:'],
            'response': [' hola']
        })

        plaintext_data = Dataset.from_dict({
            'text': ['This is plain text for language modeling.']
        })

        # To concatenate, plaintext needs matching columns (even if empty/None)
        # This simulates what a unified loader might do
        plaintext_with_instruction_cols = Dataset.from_dict({
            'text': plaintext_data['text'],
            'prompt': [None] * len(plaintext_data),
            'response': [None] * len(plaintext_data)
        })

        # Now concatenation works
        combined = concatenate_datasets([instruction_data, plaintext_with_instruction_cols])
        assert len(combined) == 2

        # Create DatasetDict structure expected by load_tokenized_dataset
        dataset_dict = DatasetDict({'train': combined})

        # Save to disk
        untokenized_path = tmp_path / "untokenized"
        dataset_dict.save_to_disk(str(untokenized_path))

        tokenized_path = tmp_path / "tokenized"

        # This should work but instruction examples with None prompt/response
        # will need special handling
        # For now, test that we detect the instruction format
        from datasets import load_from_disk
        loaded = load_from_disk(str(untokenized_path))

        sample_split = list(loaded.keys())[0]
        has_prompt = 'prompt' in loaded[sample_split].column_names
        has_response = 'response' in loaded[sample_split].column_names

        assert has_prompt and has_response, "Dataset should have instruction columns"

    def test_plaintext_only_detection(self, tmp_path, base_tokenizer):
        """
        Test that pure plaintext datasets are correctly detected as non-instruction.
        """
        from datasets import Dataset, DatasetDict, load_from_disk
        from dataset_utils import load_tokenized_dataset

        plaintext_data = Dataset.from_dict({
            'text': ['Line 1', 'Line 2', 'Line 3']
        })

        dataset_dict = DatasetDict({'train': plaintext_data})
        untokenized_path = tmp_path / "untokenized"
        dataset_dict.save_to_disk(str(untokenized_path))

        tokenized_path = tmp_path / "tokenized"

        # Tokenize
        result = load_tokenized_dataset(
            str(untokenized_path),
            str(tokenized_path),
            base_tokenizer,
            max_length=128,
            dev_size=0.5
        )

        # Should NOT have labels column (standard causal LM)
        assert 'labels' not in result['train'].column_names

    def test_instruction_only_detection(self, tmp_path, base_tokenizer):
        """
        Test that pure instruction datasets are correctly detected and get labels.
        """
        from datasets import Dataset, DatasetDict
        from dataset_utils import load_tokenized_dataset

        instruction_data = Dataset.from_dict({
            'prompt': [
                'Translate: hello\nResponse:',
                'Translate: world\nResponse:'
            ],
            'response': [' hola', ' mundo']
        })

        dataset_dict = DatasetDict({'train': instruction_data})
        untokenized_path = tmp_path / "untokenized"
        dataset_dict.save_to_disk(str(untokenized_path))

        tokenized_path = tmp_path / "tokenized"

        result = load_tokenized_dataset(
            str(untokenized_path),
            str(tokenized_path),
            base_tokenizer,
            max_length=128,
            dev_size=0.5
        )

        # Should have labels column with masking
        assert 'labels' in result['train'].column_names

        # Check that some labels are -100 (masked prompt tokens)
        labels = result['train']['labels'][0]
        assert -100 in labels, "Instruction data should have masked prompt tokens"

    def test_mixed_instruction_and_plaintext_tokenization(self, tmp_path, base_tokenizer):
        """
        Test tokenizing a dataset that mixes instruction and plaintext examples.

        This is the key test for supporting mixed training data.
        Instruction examples should get label masking, plaintext should not.
        """
        from datasets import Dataset, DatasetDict, concatenate_datasets
        from dataset_utils import load_tokenized_dataset

        # Create instruction data (multiple examples to ensure some end up in train)
        instruction_data = Dataset.from_dict({
            'prompt': [
                'Question: What is 2+2?\nResponse:',
                'Translate: hello\nResponse:'
            ],
            'response': [' 4', ' hola']
        })

        # Create plaintext data
        plaintext_data = Dataset.from_dict({
            'text': [
                'This is regular language modeling text.',
                'Another plain text example for LM training.'
            ]
        })

        # Concatenate (creates union of columns with None for missing)
        combined = concatenate_datasets([instruction_data, plaintext_data])

        dataset_dict = DatasetDict({'train': combined})
        untokenized_path = tmp_path / "untokenized"
        dataset_dict.save_to_disk(str(untokenized_path))

        tokenized_path = tmp_path / "tokenized"

        result = load_tokenized_dataset(
            str(untokenized_path),
            str(tokenized_path),
            base_tokenizer,
            max_length=128,
            dev_size=0.5
        )

        # Both splits should have examples
        assert len(result['train']) >= 1
        assert len(result['test']) >= 1

        # Check the tokenized data has proper structure
        assert 'input_ids' in result['train'].column_names
        assert 'attention_mask' in result['train'].column_names
        assert 'labels' in result['train'].column_names

        # Gather all labels from both splits
        all_labels = list(result['train']['labels']) + list(result['test']['labels'])

        # Instruction examples should have -100 masking (at least some labels are -100)
        has_masking = any(-100 in labels for labels in all_labels)
        assert has_masking, "Instruction examples should have masked labels"

        # Plaintext examples should have no -100 (all labels are actual tokens)
        has_unmasked = any(-100 not in labels for labels in all_labels)
        assert has_unmasked, "Plaintext examples should have unmasked labels"


class TestComputeSamplingProbs:
    """
    Tests for _compute_sampling_probs, which computes per-source sampling probabilities
    for multinomial dataset sampling.

    Sources can optionally pin their probability via `sampling_prob`. Unpinned sources
    share the remaining budget using alpha-based temperature scaling.
    """

    def test_no_pinned_sources(self):
        """All sources use alpha-based reweighting (original behavior)."""
        sources = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'},
        ]
        train_sizes = [1000, 2000, 7000]
        alpha = 1.0

        probs = _compute_sampling_probs(sources, train_sizes, alpha)

        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-9
        # With alpha=1.0, probabilities should be proportional to sizes
        assert abs(probs[0] - 0.1) < 1e-9
        assert abs(probs[1] - 0.2) < 1e-9
        assert abs(probs[2] - 0.7) < 1e-9

    def test_one_pinned_source(self):
        """One source pinned, rest distributed by alpha."""
        sources = [
            {'id': 'got'},
            {'id': 'non'},
            {'id': 'eng', 'sampling_prob': 0.7},
        ]
        train_sizes = [1000, 3000, 500000]
        alpha = 1.0

        probs = _compute_sampling_probs(sources, train_sizes, alpha)

        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-9
        # eng is pinned at 0.7
        assert abs(probs[2] - 0.7) < 1e-9
        # Remaining 0.3 distributed proportionally (alpha=1) among got and non
        # got: 1000/(1000+3000) * 0.3 = 0.075
        # non: 3000/(1000+3000) * 0.3 = 0.225
        assert abs(probs[0] - 0.075) < 1e-9
        assert abs(probs[1] - 0.225) < 1e-9

    def test_multiple_pinned_sources(self):
        """Multiple sources pinned, rest distributed by alpha."""
        sources = [
            {'id': 'got', 'sampling_prob': 0.1},
            {'id': 'non'},
            {'id': 'ang'},
            {'id': 'eng', 'sampling_prob': 0.6},
        ]
        train_sizes = [1000, 2000, 2000, 500000]
        alpha = 1.0

        probs = _compute_sampling_probs(sources, train_sizes, alpha)

        assert abs(sum(probs) - 1.0) < 1e-9
        assert abs(probs[0] - 0.1) < 1e-9
        assert abs(probs[3] - 0.6) < 1e-9
        # Remaining 0.3 split equally (same size, alpha=1)
        assert abs(probs[1] - 0.15) < 1e-9
        assert abs(probs[2] - 0.15) < 1e-9

    def test_all_pinned_sources_sum_to_one(self):
        """All sources pinned with probs summing to 1.0."""
        sources = [
            {'id': 'a', 'sampling_prob': 0.3},
            {'id': 'b', 'sampling_prob': 0.7},
        ]
        train_sizes = [1000, 2000]
        alpha = 0.5

        probs = _compute_sampling_probs(sources, train_sizes, alpha)

        assert abs(probs[0] - 0.3) < 1e-9
        assert abs(probs[1] - 0.7) < 1e-9

    def test_all_pinned_sources_not_summing_to_one(self):
        """All sources pinned but not summing to 1.0 raises error."""
        sources = [
            {'id': 'a', 'sampling_prob': 0.3},
            {'id': 'b', 'sampling_prob': 0.5},
        ]
        train_sizes = [1000, 2000]

        with pytest.raises(ValueError, match="sum to"):
            _compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_prob_at_one_raises_error(self):
        """sampling_prob=1.0 on a single source is an error."""
        sources = [
            {'id': 'a', 'sampling_prob': 1.0},
            {'id': 'b'},
        ]
        train_sizes = [1000, 2000]

        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            _compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_prob_zero_raises_error(self):
        """sampling_prob=0 is an error."""
        sources = [
            {'id': 'a', 'sampling_prob': 0},
            {'id': 'b'},
        ]
        train_sizes = [1000, 2000]

        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            _compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_prob_negative_raises_error(self):
        """Negative sampling_prob is an error."""
        sources = [
            {'id': 'a', 'sampling_prob': -0.5},
            {'id': 'b'},
        ]
        train_sizes = [1000, 2000]

        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            _compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_sum_exceeds_one_raises_error(self):
        """Pinned probs summing to >= 1.0 raises error."""
        sources = [
            {'id': 'a', 'sampling_prob': 0.6},
            {'id': 'b', 'sampling_prob': 0.5},
            {'id': 'c'},
        ]
        train_sizes = [1000, 2000, 3000]

        with pytest.raises(ValueError, match="must be less than 1.0"):
            _compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_alpha_affects_unpinned_distribution(self):
        """Alpha reweighting applies only to unpinned sources."""
        sources = [
            {'id': 'small'},
            {'id': 'large'},
            {'id': 'pinned', 'sampling_prob': 0.5},
        ]
        train_sizes = [100, 10000, 999999]

        # With alpha=1.0, large source dominates unpinned budget
        probs_a1 = _compute_sampling_probs(sources, train_sizes, alpha=1.0)
        # With alpha=0.0001 (near 0), unpinned sources nearly equal
        probs_a0 = _compute_sampling_probs(sources, train_sizes, alpha=0.0001)

        # Pinned source unchanged in both
        assert abs(probs_a1[2] - 0.5) < 1e-9
        assert abs(probs_a0[2] - 0.5) < 1e-9

        # With alpha=1, large source gets most of the 0.5 budget
        assert probs_a1[1] > probs_a1[0]
        assert probs_a1[1] > 0.45

        # With alpha~0, both unpinned sources get ~0.25 each
        assert abs(probs_a0[0] - 0.25) < 0.01
        assert abs(probs_a0[1] - 0.25) < 0.01

    def test_unpinned_empty_source_raises_error(self):
        """All unpinned sources being empty raises error."""
        sources = [
            {'id': 'empty1'},
            {'id': 'empty2'},
            {'id': 'pinned', 'sampling_prob': 0.5},
        ]
        train_sizes = [0, 0, 1000]

        with pytest.raises(ValueError, match="unpinned sources are empty"):
            _compute_sampling_probs(sources, train_sizes, alpha=0.5)
