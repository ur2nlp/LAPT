#!/usr/bin/env python3
"""
Filter and sample from the FLAN Collection dataset.

Downloads a specific source+template subset from Open-Orca/FLAN (e.g., flan_zsnoopt_data/)
and filters by task name and length. Caches downloads for reuse.

Output format: One example per line with instruction and response.
"""

import argparse
import random
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


# Valid dataset sources (correspond to HuggingFace repo folders)
VALID_SOURCES = ['flan', 't0', 'niv2', 'cot', 'dialog']

# Valid template types:
# - zs = zero-shot, fs = few-shot
# - noopt = no options provided, opt = multiple choice options provided
VALID_TEMPLATES = ['zsnoopt', 'zsopt', 'fsnoopt', 'fsopt']

# Map folder names to readable source names for display
SOURCE_DISPLAY_NAMES = {
    'flan': 'FLAN 2021',
    't0': 'T0',
    'niv2': 'NIv2',
    'cot': 'CoT (Chain-of-Thought)',
    'dialog': 'Dialog'
}


class FlanFilter:
    """
    Callable filter for FLAN examples with statistics tracking.

    Filters by task name, instruction prefix, and length.
    Tracks how many examples were filtered by each criterion.

    Filters are applied in order:
    1. Task name filter (if specified)
    2. Prefix exclusion filter (if specified)
    3. Length filter (if specified)
    """

    def __init__(
        self,
        task_names: list = None,
        exclude_prefixes: list = None,
        max_length: int = None,
        length_metric: str = 'words'
    ):
        """
        Initialize the filter.

        Args:
            task_names: List of task names to include (e.g., ['anli/r1:0.1.0']).
                If None, all tasks are included.
            exclude_prefixes: List of instruction prefixes to exclude
                (e.g., ['Generate a context']). If None, no prefix filtering.
            max_length: Maximum length for combined inputs+targets. If None, no length filter.
            length_metric: How to measure length - 'words' or 'chars' (default: 'words')
        """
        self.task_names = task_names
        self.exclude_prefixes = exclude_prefixes
        self.max_length = max_length
        self.length_metric = length_metric

        # Statistics tracking
        self.stats = {
            'total_seen': 0,
            'filtered_by_task': 0,
            'filtered_by_prefix': 0,
            'filtered_by_length': 0,
            'passed_filters': 0
        }

    def __call__(self, example: dict) -> bool:
        """
        Filter a single example.

        Args:
            example: Dict with '_task_name', 'inputs', and 'targets' fields

        Returns:
            True if example passes all filters, False otherwise
        """
        self.stats['total_seen'] += 1

        # Filter by task name
        if self.task_names and example['_task_name'] not in self.task_names:
            self.stats['filtered_by_task'] += 1
            return False

        # Filter by excluded prefixes (case-insensitive)
        if self.exclude_prefixes:
            inputs_text = example['inputs']
            inputs_lower = inputs_text.lower()
            for prefix in self.exclude_prefixes:
                if inputs_lower.startswith(prefix.lower()):
                    self.stats['filtered_by_prefix'] += 1
                    return False

        # Filter by length (inputs + targets combined)
        if self.max_length:
            combined_text = example['inputs'] + ' ' + example['targets']

            if self.length_metric == 'words':
                length = len(combined_text.split())
            elif self.length_metric == 'chars':
                length = len(combined_text)
            else:
                raise ValueError(f"Unknown length_metric: {self.length_metric}")

            if length > self.max_length:
                self.stats['filtered_by_length'] += 1
                return False

        self.stats['passed_filters'] += 1
        return True

    def print_stats(self):
        """Print filtering statistics."""
        print("\nFiltering Statistics:", file=sys.stderr)
        print(f"  Total examples seen: {self.stats['total_seen']:,}", file=sys.stderr)
        print(f"  Filtered by task: {self.stats['filtered_by_task']:,}", file=sys.stderr)
        print(f"  Filtered by prefix: {self.stats['filtered_by_prefix']:,}", file=sys.stderr)
        print(f"  Filtered by length: {self.stats['filtered_by_length']:,}", file=sys.stderr)
        print(f"  Passed all filters: {self.stats['passed_filters']:,}", file=sys.stderr)


def format_example(example: dict, format_template: str) -> str:
    """
    Format a FLAN example as a single-line text string.

    Args:
        example: Dict with 'inputs' and 'targets' fields
        format_template: Template string with {inputs} and {targets} placeholders

    Returns:
        Formatted string
    """
    # Preprocess: replace HTML line breaks with spaces (case-insensitive)
    inputs = example['inputs'].strip().replace('<br>', ' ').replace('<BR>', ' ')
    targets = example['targets'].strip().replace('<br>', ' ').replace('<BR>', ' ')

    # Replace literal \n in template with actual newlines
    formatted = format_template.replace('\\n', '\n')
    formatted = formatted.format(inputs=inputs, targets=targets)

    # Collapse to single line (replace newlines with spaces)
    return ' '.join(formatted.split())


def download_flan_subset(
    source: str,
    template: str,
    cache_dir: str = "data/flan_cache"
) -> Path:
    """
    Download a specific FLAN subset (source + template combination).

    Downloads only the specified parquet files from the HuggingFace repo.
    Subsequent calls will reuse cached data (no re-download).

    Args:
        source: Source name (flan, t0, niv2, cot, dialog)
        template: Template type (zsnoopt, zsopt, fsnoopt, fsopt)
        cache_dir: Directory to cache downloads (default: data/flan_cache)

    Returns:
        Path to the downloaded parquet folder

    Raises:
        RuntimeError: If download fails or folder doesn't exist after download
    """
    folder_name = f"{source}_{template}_data"

    print(f"Downloading {SOURCE_DISPLAY_NAMES[source]} ({template}) subset...", file=sys.stderr)
    print(f"  Folder: {folder_name}/", file=sys.stderr)
    print(f"  Cache: {cache_dir}", file=sys.stderr)

    # Download only the specific folder using snapshot_download
    # HuggingFace will automatically skip already-downloaded files
    local_dir_path = Path(cache_dir)
    snapshot_download(
        repo_id="Open-Orca/FLAN",
        repo_type="dataset",
        allow_patterns=f"{folder_name}/*.parquet",
        local_dir=local_dir_path
    )

    parquet_dir = local_dir_path / folder_name

    if not parquet_dir.exists():
        raise RuntimeError(f"Download failed: {parquet_dir} does not exist")

    print(f"  Downloaded to: {parquet_dir}", file=sys.stderr)
    return parquet_dir


def filter_flan(
    output_file: str,
    num_samples: int,
    source: str,
    template: str,
    task_names: list = None,
    exclude_prefixes: list = None,
    max_length: int = None,
    length_metric: str = 'words',
    format_template: str = "{inputs} Response: {targets}",
    cache_dir: str = "data/flan_cache",
    seed: int = 1
):
    """
    Filter and sample from FLAN dataset subset.

    Args:
        output_file: Path to output text file
        num_samples: Number of examples to sample
        source: Dataset source (flan, t0, niv2, cot, dialog)
        template: Template type (zsnoopt, zsopt, fsnoopt, fsopt)
        task_names: List of specific task names to include (optional)
        exclude_prefixes: List of instruction prefixes to exclude (optional)
        max_length: Maximum length in words or characters (optional)
        length_metric: 'words' or 'chars' for length measurement
        format_template: Template for formatting examples
        cache_dir: Directory to cache downloads
        seed: Random seed for sampling
    """
    random.seed(seed)

    # Download the specific subset
    parquet_dir = download_flan_subset(source, template, cache_dir)

    # Load from local parquet files
    print(f"\nLoading dataset from {parquet_dir}...", file=sys.stderr)
    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_dir / "*.parquet"),
        split="train"
    )

    print(f"Loaded {len(dataset):,} examples", file=sys.stderr)

    # Create filter object
    print("\nApplying filters...", file=sys.stderr)
    flan_filter = FlanFilter(
        task_names=task_names,
        exclude_prefixes=exclude_prefixes,
        max_length=max_length,
        length_metric=length_metric
    )

    # Apply filter
    filtered_dataset = dataset.filter(flan_filter)

    # Print filtering statistics
    flan_filter.print_stats()

    if len(filtered_dataset) == 0:
        raise ValueError("No examples passed filters. Check your task names and filter criteria.")

    # Sample if needed (uniform random sampling without replacement)
    if len(filtered_dataset) <= num_samples:
        print(f"\nWarning: Only {len(filtered_dataset):,} examples available (requested {num_samples:,})", file=sys.stderr)
        print(f"Using all {len(filtered_dataset):,} examples", file=sys.stderr)
        sampled_dataset = filtered_dataset
    else:
        print(f"\nSampling {num_samples:,} examples from {len(filtered_dataset):,} filtered examples...", file=sys.stderr)
        # random.sample ensures uniform random sampling (no duplicates)
        indices = random.sample(range(len(filtered_dataset)), num_samples)
        sampled_dataset = filtered_dataset.select(indices)

    # Write output
    print(f"\nWriting {len(sampled_dataset):,} examples to {output_file}...", file=sys.stderr)

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write formatted examples
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in sampled_dataset:
            formatted = format_example(example, format_template)
            f.write(formatted + '\n')

    print(f"Done! Wrote {len(sampled_dataset):,} examples.", file=sys.stderr)


def main():
    """
    Main entry point for the script.

    Workflow:
    1. Parse command line arguments
    2. Download specific FLAN subset (cached if already exists)
    3. Load dataset from parquet files
    4. Apply filters (task name, prefix exclusion, length)
    5. Sample requested number of examples
    6. Format and write to output file
    """
    parser = argparse.ArgumentParser(
        description='Filter and sample from FLAN Collection dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get 50k FLAN zero-shot examples from NLI tasks, max 350 words
  python filter_flan.py --source flan --template zsnoopt --num-samples 50000 \\
      --task-names anli/r1:0.1.0 anli/r2:0.1.0 super_glue/rte:1.0.2 \\
      --max-length 350

  # Get 25k FLAN zero-shot examples from QA task, excluding generation prompts
  python filter_flan.py --source flan --template zsnoopt --num-samples 25000 \\
      --task-names bool_q:1.0.0 \\
      --exclude-prefixes "Generate a context"

  # Get NIv2 examples with character limit
  python filter_flan.py --source niv2 --template zsopt --num-samples 30000 \\
      --max-length 2000 --length-metric chars

Available sources: flan (FLAN 2021), t0 (T0), niv2 (NIv2), cot (Chain-of-Thought), dialog (Dialog)
Available templates: zsnoopt (zero-shot no option), zsopt (zero-shot with options),
                     fsnoopt (few-shot no option), fsopt (few-shot with options)

Note: Task names include version numbers (e.g., 'anli/r1:0.1.0', 'bool_q:1.0.0').
      Use tools/find_task_name.py to explore available task names in downloaded data.
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=VALID_SOURCES,
        help='Dataset source to download'
    )
    parser.add_argument(
        '--template',
        type=str,
        required=True,
        choices=VALID_TEMPLATES,
        help='Template type to download'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/flan_filtered/flan_filtered.txt',
        help='Output file path (default: data/flan_filtered/flan_filtered.txt)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        required=True,
        help='Number of examples to sample'
    )
    parser.add_argument(
        '--task-names',
        nargs='+',
        help='Specific task names to include with version numbers (e.g., anli/r1:0.1.0 bool_q:1.0.0). '
             'If not specified, all tasks from the source are included.'
    )
    parser.add_argument(
        '--exclude-prefixes',
        nargs='+',
        help='Instruction prefixes to exclude (e.g., "Generate a context"). '
             'Filters out examples where inputs start with any of these strings.'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        help='Maximum length for combined inputs + targets (in words or chars depending on --length-metric)'
    )
    parser.add_argument(
        '--length-metric',
        type=str,
        choices=['words', 'chars'],
        default='words',
        help='Metric for measuring length: "words" (default) or "chars"'
    )
    parser.add_argument(
        '--format-template',
        type=str,
        default='{inputs} Response: {targets}',
        help='Format template with {inputs} and {targets} placeholders. Use \\n for newlines.'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/flan_cache',
        help='Directory to cache downloaded data (default: data/flan_cache)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for sampling (default: 1)'
    )

    args = parser.parse_args()

    print(f"Filtering FLAN dataset with:", file=sys.stderr)
    print(f"  Source: {SOURCE_DISPLAY_NAMES[args.source]} ({args.source})", file=sys.stderr)
    print(f"  Template: {args.template}", file=sys.stderr)
    print(f"  Output: {args.output_file}", file=sys.stderr)
    print(f"  Samples: {args.num_samples:,}", file=sys.stderr)
    if args.task_names:
        print(f"  Task names: {args.task_names}", file=sys.stderr)
    if args.exclude_prefixes:
        print(f"  Exclude prefixes: {args.exclude_prefixes}", file=sys.stderr)
    if args.max_length:
        print(f"  Max length: {args.max_length} {args.length_metric}", file=sys.stderr)
    print(f"  Format: {args.format_template}", file=sys.stderr)
    print(f"  Seed: {args.seed}", file=sys.stderr)
    print()

    filter_flan(
        output_file=args.output_file,
        num_samples=args.num_samples,
        source=args.source,
        template=args.template,
        task_names=args.task_names,
        exclude_prefixes=args.exclude_prefixes,
        max_length=args.max_length,
        length_metric=args.length_metric,
        format_template=args.format_template,
        cache_dir=args.cache_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
