"""Tests for lapt.tokenized_data: the transformations and the tokenized stages.

Per-source loading is covered by `tests/test_source_*.py`, one file per
registered type, which exercise the artifact classes directly. This file
covers what sits above them: instruction tokenization and label masking, the
collator, sampling arithmetic, external eval sets, and the tokenized-dataset
stages including the multinomial mix.

Testing approach:
- Real temporary files (pytest's tmp_path) for I/O
- Mocking to avoid expensive downloads
"""

from pathlib import Path

import pytest
import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

from lapt.artifact_configs import DatasetConfig, TokenizedDatasetConfig
from lapt.tokenized_data import (
    TokenizedDatasetArtifact,
    TokenizedMultinomialMix,
    _partition_source_indices,
)
from lapt.sources.sampling import compute_sampling_probs




class TestTokenizeInstructionExamples:
    """
    Tests for tokenize_instruction_examples function.

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
        from lapt.tokenized_data import tokenize_instruction_examples

        examples = {
            'prompt': ['Translate to Gothic: hello\nResponse:'],
            'response': [' world']
        }

        result = tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

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
        from lapt.tokenized_data import tokenize_instruction_examples

        prompt = "This is a test prompt with several words\nResponse:"
        response = " Yes"

        examples = {
            'prompt': [prompt],
            'response': [response]
        }

        result = tokenize_instruction_examples(examples, base_tokenizer, max_length=512)
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
        from lapt.tokenized_data import tokenize_instruction_examples

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

        result = tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

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
        from lapt.tokenized_data import tokenize_instruction_examples

        # Long prompt and response
        examples = {
            'prompt': ['This is a very long prompt ' * 20 + '\nResponse:'],
            'response': [' This is a very long response ' * 20]
        }

        max_length = 50
        result = tokenize_instruction_examples(examples, base_tokenizer, max_length=max_length)

        # Should be truncated to max_length
        assert len(result['input_ids'][0]) <= max_length
        assert len(result['labels'][0]) <= max_length
        assert len(result['attention_mask'][0]) <= max_length

    def test_empty_response(self, base_tokenizer):
        """
        Test handling of empty response (edge case).

        The response contributes no tokens of its own, but an EOS is still
        appended and left trainable so the model learns to terminate
        immediately rather than continuing the prompt. So every label is -100
        except a final EOS.
        """
        from lapt.tokenized_data import tokenize_instruction_examples

        examples = {
            'prompt': ['Prompt text\nResponse:'],
            'response': ['']
        }

        result = tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

        labels = result['labels'][0]

        assert all(label == -100 for label in labels[:-1]), \
            "All prompt labels should be -100 for empty response"
        assert labels[-1] == base_tokenizer.eos_token_id, \
            "The appended EOS should stay trainable"

    def test_response_starts_with_space(self, base_tokenizer):
        """
        Test that response with leading space is handled correctly.

        Our JSONL format uses ' response' (with leading space) to ensure
        proper tokenization as a continuation.
        """
        from lapt.tokenized_data import tokenize_instruction_examples

        examples = {
            'prompt': ['Test\nResponse:'],
            'response': [' answer']  # Note leading space
        }

        result = tokenize_instruction_examples(examples, base_tokenizer, max_length=512)

        # Should tokenize without errors
        assert len(result['input_ids'][0]) > 0
        assert len(result['labels'][0]) > 0




def load_tokenized_dataset(untokenized_path, tokenized_path, tokenizer, max_length, dev_size):
    """Test-local shim over TokenizedDatasetArtifact, matching the retired
    function's signature. `tokenized_path`'s parent becomes the artifact's
    cache_dir; the artifact computes its own path from a minimal
    TokenizedDatasetConfig, which these tests don't otherwise need."""
    import os
    config = TokenizedDatasetConfig(
        max_length=max_length,
        dev_size=dev_size,
        dataset_config=DatasetConfig({'type': 'test'}),
        tokenizer_id=os.path.basename(tokenized_path),
    )
    return TokenizedDatasetArtifact(
        cache_dir=os.path.dirname(tokenized_path),
        tokenized_dataset_config=config,
        untokenized_path=untokenized_path,
        tokenizer=tokenizer,
        max_length=max_length,
        dev_size=dev_size,
    ).resolve()


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
        from datasets import Dataset, DatasetDict

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

        probs = compute_sampling_probs(sources, train_sizes, alpha)

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

        probs = compute_sampling_probs(sources, train_sizes, alpha)

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

        probs = compute_sampling_probs(sources, train_sizes, alpha)

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

        probs = compute_sampling_probs(sources, train_sizes, alpha)

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
            compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_prob_at_one_raises_error(self):
        """sampling_prob=1.0 on a single source is an error."""
        sources = [
            {'id': 'a', 'sampling_prob': 1.0},
            {'id': 'b'},
        ]
        train_sizes = [1000, 2000]

        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_prob_zero_raises_error(self):
        """sampling_prob=0 is an error."""
        sources = [
            {'id': 'a', 'sampling_prob': 0},
            {'id': 'b'},
        ]
        train_sizes = [1000, 2000]

        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_prob_negative_raises_error(self):
        """Negative sampling_prob is an error."""
        sources = [
            {'id': 'a', 'sampling_prob': -0.5},
            {'id': 'b'},
        ]
        train_sizes = [1000, 2000]

        with pytest.raises(ValueError, match="between 0 and 1 exclusive"):
            compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_pinned_sum_exceeds_one_raises_error(self):
        """Pinned probs summing to >= 1.0 raises error."""
        sources = [
            {'id': 'a', 'sampling_prob': 0.6},
            {'id': 'b', 'sampling_prob': 0.5},
            {'id': 'c'},
        ]
        train_sizes = [1000, 2000, 3000]

        with pytest.raises(ValueError, match="must be less than 1.0"):
            compute_sampling_probs(sources, train_sizes, alpha=0.5)

    def test_alpha_affects_unpinned_distribution(self):
        """Alpha reweighting applies only to unpinned sources."""
        sources = [
            {'id': 'small'},
            {'id': 'large'},
            {'id': 'pinned', 'sampling_prob': 0.5},
        ]
        train_sizes = [100, 10000, 999999]

        # With alpha=1.0, large source dominates unpinned budget
        probs_a1 = compute_sampling_probs(sources, train_sizes, alpha=1.0)
        # With alpha=0.0001 (near 0), unpinned sources nearly equal
        probs_a0 = compute_sampling_probs(sources, train_sizes, alpha=0.0001)

        # Pinned source unchanged in both
        assert abs(probs_a1[2] - 0.5) < 1e-9
        assert abs(probs_a0[2] - 0.5) < 1e-9

        # With alpha=1, large source gets most of the 0.5 budget
        assert probs_a1[1] > probs_a1[0]
        assert probs_a1[1] > 0.45

        # With alpha~0, both unpinned sources get ~0.25 each
        assert abs(probs_a0[0] - 0.25) < 0.01
        assert abs(probs_a0[1] - 0.25) < 0.01

    def test_alpha_optional_when_all_sources_pinned(self):
        """Alpha may be omitted when every source has a pinned probability."""
        sources = [
            {'id': 'a', 'sampling_prob': 0.3},
            {'id': 'b', 'sampling_prob': 0.7},
        ]
        train_sizes = [100, 10000]

        probs = compute_sampling_probs(sources, train_sizes, alpha=None)

        assert probs == [0.3, 0.7]

    def test_alpha_optional_with_single_unpinned_source(self):
        """A lone unpinned source takes the residual budget regardless of alpha."""
        sources = [
            {'id': 'a', 'sampling_prob': 0.3},
            {'id': 'b', 'sampling_prob': 0.4},
            {'id': 'filler'},
        ]
        train_sizes = [100, 200, 10000]

        probs_none = compute_sampling_probs(sources, train_sizes, alpha=None)
        probs_half = compute_sampling_probs(sources, train_sizes, alpha=0.5)

        assert abs(probs_none[2] - 0.3) < 1e-9
        assert probs_none == probs_half

    def test_missing_alpha_with_several_unpinned_raises_error(self):
        """Alpha is required as soon as two or more sources are unpinned."""
        sources = [
            {'id': 'a', 'sampling_prob': 0.5},
            {'id': 'unpinned1'},
            {'id': 'unpinned2'},
        ]
        train_sizes = [100, 1000, 10000]

        with pytest.raises(ValueError, match="alpha is required"):
            compute_sampling_probs(sources, train_sizes, alpha=None)

    def test_unpinned_empty_source_raises_error(self):
        """All unpinned sources being empty raises error."""
        sources = [
            {'id': 'empty1'},
            {'id': 'empty2'},
            {'id': 'pinned', 'sampling_prob': 0.5},
        ]
        train_sizes = [0, 0, 1000]

        with pytest.raises(ValueError, match="unpinned sources are empty"):
            compute_sampling_probs(sources, train_sizes, alpha=0.5)


class TestPartitionSourceIndices:
    """
    Tests for the deterministic train/dev row partition used by the plan-based
    multinomial path.
    """

    def test_fractional_dev_size(self):
        train, dev = _partition_source_indices(100, dev_size=0.1, seed=1)
        assert len(train) == 90
        assert len(dev) == 10
        # No overlap and full coverage
        combined = sorted(train.tolist() + dev.tolist())
        assert combined == list(range(100))

    def test_absolute_dev_size(self):
        train, dev = _partition_source_indices(100, dev_size=20, seed=1)
        assert len(train) == 80
        assert len(dev) == 20

    def test_skip_dev_split(self):
        train, dev = _partition_source_indices(50, dev_size=-1, seed=1)
        assert len(train) == 50
        assert len(dev) == 0

    def test_deterministic_seed(self):
        a_train, a_dev = _partition_source_indices(100, dev_size=0.2, seed=1)
        b_train, b_dev = _partition_source_indices(100, dev_size=0.2, seed=1)
        assert a_train.tolist() == b_train.tolist()
        assert a_dev.tolist() == b_dev.tolist()


def load_tokenized_multinomial_dataset(
    sources, alpha, total_samples, dev_size, base_cache_dir,
    tokenizer, tokenizer_id, max_length, seed=1,
):
    """Test-local shim over TokenizedMultinomialMix, matching the retired
    function's signature so the tests below stay close to what they're
    actually exercising."""
    return TokenizedMultinomialMix(
        base_cache_dir=base_cache_dir,
        sources=sources,
        alpha=alpha,
        total_samples=total_samples,
        dev_size=dev_size,
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        max_length=max_length,
        seed=seed,
    ).resolve()


class TestLoadTokenizedMultinomialDataset:
    """
    End-to-end tests for TokenizedMultinomialMix (formerly the standalone
    function load_tokenized_multinomial_dataset).

    Verifies that:
    - Each source is tokenized exactly once (no per-mix re-tokenization).
    - Upsampling is represented by repeated indices, not duplicated rows.
    - Changing alpha or total_samples does not invalidate per-source tokenized caches.
    - Dev splits are persisted under the mix directory.
    """

    @staticmethod
    def _write_source(
        cache_dir: Path,
        source_id: str,
        lines: list[str],
        source_path: str = 'unused',
    ) -> None:
        """Persist a tiny plaintext source as a valid untokenized cache.

        The config record is written alongside the data, matching what
        `PlaintextDataset.config()` produces, so the cache is reusable rather
        than merely present. Building the data without a record would be
        refused on read, and rightly so.
        """
        source_dir = cache_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        untokenized_dir = source_dir / "untokenized"
        ds = DatasetDict({'train': Dataset.from_dict({'text': lines})})
        ds.save_to_disk(str(untokenized_dir))
        with open(untokenized_dir / "config.yaml", 'w') as record:
            yaml.dump({'type': 'plaintext', 'path': source_path}, record)

    def test_basic_end_to_end(self, tmp_path, base_tokenizer):
        cache_dir = tmp_path / "cache"
        # Two sources of different sizes. Small one upsampled.
        self._write_source(cache_dir, "big", [f"big line {i}" for i in range(20)])
        self._write_source(cache_dir, "small", [f"small {i}" for i in range(5)])

        sources = [
            {'id': 'big', 'type': 'plaintext', 'path': 'unused'},
            {'id': 'small', 'type': 'plaintext', 'path': 'unused'},
        ]
        result = load_tokenized_multinomial_dataset(
            sources=sources,
            alpha=0.5,
            total_samples=40,
            dev_size=0.2,
            base_cache_dir=str(cache_dir),
            tokenizer=base_tokenizer,
            tokenizer_id="xglm564m",
            max_length=64,
        )

        assert 'train' in result
        assert len(result['train']) == 40
        # Per-source dev splits (sizes after partition: floor(0.2*20)=4, floor(0.2*5)=1)
        assert 'big' in result and 'small' in result
        assert len(result['big']) == 4
        assert len(result['small']) == 1

        # Per-source tokenized caches were created.
        assert (cache_dir / "big" / "tokenized_xglm564m_ml64_nolabels").exists()
        assert (cache_dir / "small" / "tokenized_xglm564m_ml64_nolabels").exists()

        # Plan artifact lives under the mix directory (its own subdirectory,
        # holding plan.npz plus the CachedArtifact config record).
        mix_dirs = [
            p for p in cache_dir.iterdir()
            if p.is_dir() and p.name.startswith("mix_")
        ]
        assert len(mix_dirs) == 1
        assert (mix_dirs[0] / "train_plan" / "plan.npz").exists()
        assert (mix_dirs[0] / "train_plan" / "config.yaml").exists()

    def test_alpha_change_reuses_tokenized_sources(self, tmp_path, base_tokenizer):
        """Sweeping alpha must NOT re-tokenize per-source data."""
        cache_dir = tmp_path / "cache"
        self._write_source(cache_dir, "a", [f"a {i}" for i in range(10)])
        self._write_source(cache_dir, "b", [f"b {i}" for i in range(10)])

        sources = [
            {'id': 'a', 'type': 'plaintext', 'path': 'unused'},
            {'id': 'b', 'type': 'plaintext', 'path': 'unused'},
        ]

        load_tokenized_multinomial_dataset(
            sources=sources, alpha=0.5, total_samples=20, dev_size=0.2,
            base_cache_dir=str(cache_dir), tokenizer=base_tokenizer,
            tokenizer_id="xglm564m", max_length=64,
        )
        tokenized_dir_a = cache_dir / "a" / "tokenized_xglm564m_ml64_nolabels"
        mtime_before = tokenized_dir_a.stat().st_mtime

        # Sweep alpha; tokenized source cache should stay untouched.
        load_tokenized_multinomial_dataset(
            sources=sources, alpha=1.0, total_samples=20, dev_size=0.2,
            base_cache_dir=str(cache_dir), tokenizer=base_tokenizer,
            tokenizer_id="xglm564m", max_length=64,
        )
        assert tokenized_dir_a.stat().st_mtime == mtime_before

        # Two distinct mix directories now exist (one per alpha value).
        mix_dirs = sorted(
            p for p in cache_dir.iterdir()
            if p.is_dir() and p.name.startswith("mix_")
        )
        assert len(mix_dirs) == 2

    def test_upsampling_uses_repeated_indices(self, tmp_path, base_tokenizer):
        """
        Train rows are an indices-mapped view; uniqueness in the underlying
        per-source caches matches the unique row counts (not the upsampled total).
        """
        cache_dir = tmp_path / "cache"
        self._write_source(cache_dir, "small", [f"line {i}" for i in range(3)])

        sources = [{'id': 'small', 'type': 'plaintext', 'path': 'unused'}]
        result = load_tokenized_multinomial_dataset(
            sources=sources, alpha=0.5, total_samples=30, dev_size=-1,
            base_cache_dir=str(cache_dir), tokenizer=base_tokenizer,
            tokenizer_id="xglm564m", max_length=64,
        )
        # 30 indices into a 3-row source means every row appears ~10 times.
        assert len(result['train']) == 30
        underlying = load_from_disk(
            str(cache_dir / "small" / "tokenized_xglm564m_ml64_nolabels")
        )
        assert len(underlying) == 3

    def test_substitutions_applied_before_tokenization(self, tmp_path, base_tokenizer):
        """
        A source declaring substitutions must be tokenized from the substituted
        variant, not the raw 'untokenized' dir. This is the regression that the
        index-based path originally missed: load_untokenized_dataset returns the
        substituted path, so tokenize_source must consume that exact path.
        """
        cache_dir = tmp_path / "cache"
        self._write_source(cache_dir, "chat", ["hello\nworld", "foo\n\nbar"])

        sources = [{
            'id': 'chat',
            'type': 'plaintext',
            'path': 'unused',
            'substitutions': [{'pattern': r'\s*\n+\s*', 'replacement': ' '}],
        }]
        load_tokenized_multinomial_dataset(
            sources=sources, alpha=0.5, total_samples=4, dev_size=-1,
            base_cache_dir=str(cache_dir), tokenizer=base_tokenizer,
            tokenizer_id="xglm564m", max_length=64,
        )

        chat_dir = cache_dir / "chat"
        # The substituted untokenized variant was materialized with newlines gone.
        sub_untok = [p for p in chat_dir.iterdir() if p.name.startswith("untokenized_sub_")]
        assert len(sub_untok) == 1
        sub_text = load_from_disk(str(sub_untok[0]))['train']['text']
        assert sub_text == ['hello world', 'foo bar']

        # Tokenization routed to the substituted variant's cache, NOT the raw name.
        tok_sub = [p for p in chat_dir.iterdir() if p.name.startswith("tokenized_sub_")]
        assert len(tok_sub) == 1
        assert not (chat_dir / "tokenized_xglm564m_ml64_nolabels").exists()

        # The tokenized rows encode the substituted (newline-free) text.
        tokenized = load_from_disk(str(tok_sub[0]))
        decoded = base_tokenizer.batch_decode(
            tokenized['input_ids'], skip_special_tokens=True
        )
        assert all("\n" not in d for d in decoded)

    def test_dev_splits_are_keyed_by_tokenizer(self, tmp_path, base_tokenizer):
        """
        The mix directory is tokenizer-agnostic, so the dev cache inside it must
        carry a tokenizer key. Without one, the second model to use a mix silently
        inherits the first model's dev token ids.
        """
        other_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

        cache_dir = tmp_path / "cache"
        lines = [f"dev line {i}" for i in range(10)]
        self._write_source(cache_dir, "src", lines)
        sources = [{'id': 'src', 'type': 'plaintext', 'path': 'unused'}]

        first = load_tokenized_multinomial_dataset(
            sources=sources, alpha=0.5, total_samples=10, dev_size=0.5,
            base_cache_dir=str(cache_dir), tokenizer=base_tokenizer,
            tokenizer_id="xglm564m", max_length=64,
        )
        second = load_tokenized_multinomial_dataset(
            sources=sources, alpha=0.5, total_samples=10, dev_size=0.5,
            base_cache_dir=str(cache_dir), tokenizer=other_tokenizer,
            tokenizer_id="gpt2", max_length=64,
        )

        mix_dirs = [
            p for p in cache_dir.iterdir()
            if p.is_dir() and p.name.startswith("mix_")
        ]
        assert len(mix_dirs) == 1
        dev_dirs = sorted(p.name for p in mix_dirs[0].iterdir() if p.name.startswith("dev"))
        assert dev_dirs == [
            "dev_gpt2_ml64_nolabels",
            "dev_xglm564m_ml64_nolabels",
        ]

        # Each model's dev rows decode back to the same text under its own tokenizer.
        first_decoded = base_tokenizer.batch_decode(
            first['src']['input_ids'], skip_special_tokens=True
        )
        second_decoded = other_tokenizer.batch_decode(
            second['src']['input_ids'], skip_special_tokens=True
        )
        assert [d.strip() for d in first_decoded] == [d.strip() for d in second_decoded]
        assert first['src']['input_ids'] != second['src']['input_ids']

    def test_pre_artifact_dev_cache_is_rebuilt_not_refused(self, tmp_path, base_tokenizer):
        """
        A dev cache written before this stage had config tracking carries no
        record. Every one of the 26 on the cluster is in that shape, so
        refusing them (CachedArtifact's default for an unverifiable cache)
        would break every existing mix until they were deleted by hand. They
        are a .select() over already-tokenized rows, so rebuilding is right.
        """
        cache_dir = tmp_path / "cache"
        self._write_source(cache_dir, "src", [f"line {i}" for i in range(10)])
        sources = [{'id': 'src', 'type': 'plaintext', 'path': 'unused'}]
        kwargs = dict(
            sources=sources, alpha=0.5, total_samples=10, dev_size=0.5,
            base_cache_dir=str(cache_dir), tokenizer=base_tokenizer,
            tokenizer_id="xglm564m", max_length=64,
        )

        first = load_tokenized_multinomial_dataset(**kwargs)
        dev_dir = next(
            p for p in (cache_dir.glob("mix_*/dev_*")) if p.is_dir()
        )

        # Regress the cache to its pre-artifact shape: data present, no record.
        (dev_dir / "config.yaml").unlink()
        assert (dev_dir / "dataset_dict.json").exists()

        second = load_tokenized_multinomial_dataset(**kwargs)

        assert (dev_dir / "config.yaml").exists(), "record should be written on rebuild"
        assert first['src']['input_ids'] == second['src']['input_ids']


def _make_msgs(*pairs):
    """Build a messages list from (role, content) tuples."""
    return [{'role': r, 'content': c} for r, c in pairs]


def _fake_hf_dataset(messages_list, column='messages'):
    """Build an in-memory Dataset mimicking an HF instruction dataset."""
    return Dataset.from_dict({column: messages_list})


