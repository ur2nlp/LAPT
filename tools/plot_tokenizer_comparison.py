"""
Plot tokenizer efficiency and overlap metrics across lambda values.

Creates plots showing how lambda parameter affects:
- Tokenization efficiency (characters per token) - left y-axis
- Vocabulary overlap with base tokenizer (percentage) - right y-axis

Each text file gets its own subplot for comparison. By default, uses dual y-axis
plot with compression on left (blue) and overlap on right (magenta).

Usage:
    # Basic usage (dual y-axis by default)
    python tools/plot_tokenizer_comparison.py lambda_efficiency.csv

    # Single y-axis with free scales per facet
    python tools/plot_tokenizer_comparison.py lambda_efficiency.csv --no-dual-axis

    # Separate plots for each metric
    python tools/plot_tokenizer_comparison.py lambda_efficiency.csv --separate --output comparison

    # Specify output file
    python tools/plot_tokenizer_comparison.py lambda_efficiency.csv --output comparison.png

    # Adjust plot dimensions
    python tools/plot_tokenizer_comparison.py lambda_efficiency.csv --width 12 --height 8

    # Show only specific metrics
    python tools/plot_tokenizer_comparison.py lambda_efficiency.csv --metrics compression overlap

    # Customize lambda extraction pattern
    python tools/plot_tokenizer_comparison.py lambda_efficiency.csv --lambda-pattern "lambda(\d+\.\d+)"
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_wrap,
    theme_minimal, theme, labs, scale_color_manual,
    element_text, element_rect
)


def format_file_name_as_language(file_name: str) -> str:
    """
    Convert file name to readable language name.

    Args:
        file_name: File name (e.g., "gothic_test.txt", "old_norse.txt")

    Returns:
        Readable language name (e.g., "Gothic", "Old Norse")
    """
    # Hard-coded lookup table for common cases
    lookup = {
        # Actual project files
        'gotica_clean_all_codices.txt': 'Gothic',
        'gotica_clean_all_codices_script.txt': 'Gothic (Romanized)',
        'icepahc_12th-18th_clean.txt': 'Old Norse',
        'flan_nli_qa_short.txt': 'English',
        # Common generic names
        'gothic_test.txt': 'Gothic',
        'gothic_romanized.txt': 'Gothic (Romanized)',
        'gothic_rom.txt': 'Gothic (Romanized)',
        'old_english.txt': 'Old English',
        'old_english_test.txt': 'Old English',
        'old_norse.txt': 'Old Norse',
        'old_norse_test.txt': 'Old Norse',
        'english.txt': 'English',
        'english_test.txt': 'English',
        'training_subset.jsonl': 'Training Data',
        'test.txt': 'Test Data',
        'val.txt': 'Validation Data',
    }

    # Try exact match first
    if file_name in lookup:
        return lookup[file_name]

    # Try case-insensitive match
    for key, value in lookup.items():
        if file_name.lower() == key.lower():
            return value

    # Intelligent parsing fallback
    # Remove common suffixes
    name = file_name.lower()
    for suffix in ['.txt', '.jsonl', '_test', '_val', '_validation', '_train', '_training']:
        name = name.replace(suffix, '')

    # Replace underscores with spaces and title case
    name = name.replace('_', ' ').strip()

    # Special handling for "old" prefix
    if name.startswith('old '):
        # "old english" -> "Old English"
        name = ' '.join(word.capitalize() for word in name.split())
    else:
        # "gothic" -> "Gothic"
        name = name.title()

    # If we got something reasonable, return it; otherwise use original
    if name and name != file_name:
        return name
    else:
        return file_name


def extract_lambda_from_tokenizer_name(tokenizer_name: str, pattern: str = None) -> float:
    """
    Extract lambda value from tokenizer path.

    Args:
        tokenizer_name: Tokenizer name/path
        pattern: Custom regex pattern to extract lambda (should have one capture group)

    Returns:
        Lambda value as float, or None if not found

    Examples:
        "xglm564m_focus-v16k-s200k_seeded-5.0x-lambda0.5" -> 0.5
        "xglm564m_focus-v16k-s200k_unseeded" -> 0.0 (unseeded assumed as lambda=0)
    """
    if pattern is None:
        # Default pattern: look for "lambda" followed by number
        pattern = r'lambda(\d+\.?\d*)'

    match = re.search(pattern, tokenizer_name)
    if match:
        return float(match.group(1))

    # If "unseeded" in name, assume lambda=0.0
    if 'unseeded' in tokenizer_name.lower():
        return 0.0

    # If base tokenizer (no FOCUS params), could be lambda=1.0 or N/A
    # Let's return None and filter it out
    return None


def prepare_data(csv_path: str, lambda_pattern: str = None) -> pd.DataFrame:
    """
    Load and prepare data for plotting.

    Args:
        csv_path: Path to CSV file from tokenizer_efficiency.py
        lambda_pattern: Custom regex pattern for extracting lambda

    Returns:
        DataFrame in long format with columns:
        - lambda_value: Extracted lambda value
        - file: Text file name
        - metric: Metric name ("Compression" or "Overlap")
        - value: Metric value
    """
    df = pd.read_csv(csv_path)

    # Format file names to readable language names
    df['file'] = df['file'].apply(format_file_name_as_language)

    # Extract lambda from tokenizer name
    df['lambda_value'] = df['tokenizer'].apply(
        lambda x: extract_lambda_from_tokenizer_name(x, lambda_pattern)
    )

    # Filter out rows where lambda couldn't be extracted
    df = df[df['lambda_value'].notna()].copy()

    if df.empty:
        raise ValueError("No lambda values could be extracted from tokenizer names. "
                        "Check that tokenizer names contain 'lambda' or 'unseeded'.")

    # Prepare data for plotting
    plot_data = []

    # Add compression metric (chars per token)
    compression_data = df[['lambda_value', 'file', 'chars_per_token']].copy()
    compression_data['metric'] = 'Compression (chars/token)'
    compression_data['value'] = compression_data['chars_per_token']
    plot_data.append(compression_data[['lambda_value', 'file', 'metric', 'value']])

    # Add overlap metric if present
    if 'base_overlap_pct' in df.columns:
        overlap_data = df[['lambda_value', 'file', 'base_overlap_pct']].copy()
        overlap_data['metric'] = 'Base Overlap (%)'
        overlap_data['value'] = overlap_data['base_overlap_pct']
        plot_data.append(overlap_data[['lambda_value', 'file', 'metric', 'value']])

    # Add fertility if requested
    if 'fertility' in df.columns:
        fertility_data = df[['lambda_value', 'file', 'fertility']].copy()
        fertility_data['metric'] = 'Fertility (tokens/word)'
        fertility_data['value'] = fertility_data['fertility']
        plot_data.append(fertility_data[['lambda_value', 'file', 'metric', 'value']])

    # Combine all metrics
    plot_df = pd.concat(plot_data, ignore_index=True)

    return plot_df


def create_dual_axis_plot(
    data: pd.DataFrame,
    width: float = 10,
    height: float = 6,
    title: str = None,
    output_path: str = None
):
    """
    Create dual-axis plot with compression and overlap on separate y-axes using matplotlib.

    Args:
        data: Prepared DataFrame with lambda_value, file, metric, value columns
        width: Plot width in inches
        height: Plot height in inches
        title: Plot title (optional)
        output_path: Output file path (required for matplotlib)

    Returns:
        None (saves plot directly)
    """
    # Check that we have both compression and overlap
    has_compression = 'Compression (chars/token)' in data['metric'].values
    has_overlap = 'Base Overlap (%)' in data['metric'].values

    if not (has_compression and has_overlap):
        raise ValueError("Dual-axis plot requires both compression and overlap metrics")

    # Get unique files
    files = sorted(data['file'].unique())
    n_files = len(files)

    # Determine optimal subplot layout (prefer ~3 cols for readability)
    if n_files <= 3:
        ncol = n_files
        nrow = 1
    else:
        ncol = 3
        nrow = (n_files + ncol - 1) // ncol

    # Separate the two metrics
    compression_data = data[data['metric'] == 'Compression (chars/token)'].copy()
    overlap_data = data[data['metric'] == 'Base Overlap (%)'].copy()

    # Calculate global axis limits (across all files)
    comp_min = compression_data['value'].min()
    comp_max = compression_data['value'].max()
    comp_range = comp_max - comp_min
    comp_ylim = (comp_min - 0.05 * comp_range, comp_max + 0.05 * comp_range)

    overlap_min = overlap_data['value'].min()
    overlap_max = overlap_data['value'].max()
    overlap_range = overlap_max - overlap_min
    overlap_ylim = (overlap_min - 0.05 * overlap_range, overlap_max + 0.05 * overlap_range)

    # Create figure and subplots with symmetrical layout
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)
    axes = []

    # Calculate how many plots in last row
    plots_in_last_row = n_files % ncol if n_files % ncol != 0 else ncol

    for i in range(n_files):
        row = i // ncol
        col = i % ncol

        # If this is the last row and it's not full, center the plots
        if row == nrow - 1 and plots_in_last_row < ncol:
            # Calculate offset to center the plots
            offset = (ncol - plots_in_last_row) // 2
            ax = fig.add_subplot(gs[row, col + offset])
        else:
            ax = fig.add_subplot(gs[row, col])

        axes.append(ax)

    # Colors
    comp_color = '#2E86AB'
    overlap_color = '#A23B72'

    # Plot each file
    for idx, file in enumerate(files):
        ax1 = axes[idx]

        # Get data for this file
        comp_file = compression_data[compression_data['file'] == file].sort_values('lambda_value')
        over_file = overlap_data[overlap_data['file'] == file].sort_values('lambda_value')

        # Plot compression on left y-axis
        line1 = ax1.plot(comp_file['lambda_value'], comp_file['value'],
                        color=comp_color, linewidth=2, marker='o', markersize=6,
                        label='Compression (chars/token)')
        ax1.set_xlabel('Lambda', fontsize=10)
        ax1.set_ylabel('Characters per Token', fontsize=10, color=comp_color)
        ax1.tick_params(axis='y', labelcolor=comp_color, labelsize=9)
        ax1.tick_params(axis='x', labelsize=9)
        ax1.set_title(format_file_name_as_language(file), fontsize=11, fontweight='bold', pad=10)
        ax1.set_ylim(comp_ylim)
        ax1.grid(True, alpha=0.3)

        # Create second y-axis for overlap
        ax2 = ax1.twinx()
        line2 = ax2.plot(over_file['lambda_value'], over_file['value'],
                        color=overlap_color, linewidth=2, marker='o', markersize=6,
                        label='Base Overlap (%)')
        ax2.set_ylabel('Base Overlap (%)', fontsize=10, color=overlap_color)
        ax2.tick_params(axis='y', labelcolor=overlap_color, labelsize=9)
        ax2.set_ylim(overlap_ylim)

    # Add overall title
    fig.suptitle(title or 'Tokenizer Efficiency and Base Overlap vs. Lambda',
                 fontsize=14, fontweight='bold', y=0.98)

    # Add legend (manually create it)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=comp_color, linewidth=2, marker='o',
               markersize=6, label='Compression (chars/token)'),
        Line2D([0], [0], color=overlap_color, linewidth=2, marker='o',
               markersize=6, label='Base Overlap (%)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()


def create_plot(
    data: pd.DataFrame,
    metrics: list = None,
    width: float = 10,
    height: float = 6,
    title: str = None
):
    """
    Create plotnine plot of lambda vs metrics (single scale).

    Args:
        data: Prepared DataFrame with lambda_value, file, metric, value columns
        metrics: List of metric names to include (default: all available)
        width: Plot width in inches
        height: Plot height in inches
        title: Plot title (optional)

    Returns:
        plotnine ggplot object
    """
    # Filter metrics if specified
    if metrics:
        data = data[data['metric'].isin(metrics)].copy()

    if data.empty:
        raise ValueError(f"No data remaining after filtering metrics: {metrics}")

    # Get unique files for determining subplot layout
    n_files = data['file'].nunique()
    n_metrics = data['metric'].nunique()

    # Determine number of columns for faceting
    if n_files == 1:
        ncol = n_metrics
    elif n_files <= 3:
        ncol = n_files
    else:
        ncol = 3

    # Create color palette
    colors = {
        'Compression (chars/token)': '#2E86AB',
        'Base Overlap (%)': '#A23B72',
        'Fertility (tokens/word)': '#F18F01'
    }

    # Build plot
    p = (
        ggplot(data, aes(x='lambda_value', y='value', color='metric', group='metric'))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + facet_wrap('~file', ncol=ncol, scales='free_y')
        + scale_color_manual(
            values=colors,
            name='Metric'
        )
        + labs(
            x='Lambda (0 = corpus-focused, 1 = base-focused)',
            y='Metric Value',
            title=title or 'Tokenizer Efficiency and Base Overlap vs. Lambda'
        )
        + theme_minimal()
        + theme(
            figure_size=(width, height),
            plot_title=element_text(size=14, face='bold'),
            axis_title=element_text(size=11),
            axis_text=element_text(size=9),
            legend_position='bottom',
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
            strip_text=element_text(size=10, face='bold'),
            strip_background=element_rect(fill='lightgray', alpha=0.3)
        )
    )

    return p


def create_separate_plots(
    data: pd.DataFrame,
    width: float = 10,
    height: float = 4
):
    """
    Create separate plots for compression and overlap.

    Args:
        data: Prepared DataFrame
        width: Plot width in inches
        height: Plot height per metric in inches

    Returns:
        Tuple of (compression_plot, overlap_plot) or (compression_plot, None)
    """
    # Get unique files for determining subplot layout
    n_files = data['file'].nunique()
    ncol = min(n_files, 3)

    plots = {}

    # Compression plot
    compression_data = data[data['metric'] == 'Compression (chars/token)'].copy()
    if not compression_data.empty:
        p_compression = (
            ggplot(compression_data, aes(x='lambda_value', y='value'))
            + geom_line(size=1.2, color='#2E86AB')
            + geom_point(size=3, color='#2E86AB')
            + facet_wrap('~file', ncol=ncol)
            + labs(
                x='Lambda (0 = corpus-focused, 1 = base-focused)',
                y='Characters per Token',
                title='Tokenization Compression vs. Lambda'
            )
            + theme_minimal()
            + theme(
                figure_size=(width, height),
                plot_title=element_text(size=14, face='bold'),
                axis_title=element_text(size=11),
                strip_text=element_text(size=10, face='bold'),
                strip_background=element_rect(fill='lightgray', alpha=0.3)
            )
        )
        plots['compression'] = p_compression

    # Overlap plot
    overlap_data = data[data['metric'] == 'Base Overlap (%)'].copy()
    if not overlap_data.empty:
        p_overlap = (
            ggplot(overlap_data, aes(x='lambda_value', y='value'))
            + geom_line(size=1.2, color='#A23B72')
            + geom_point(size=3, color='#A23B72')
            + facet_wrap('~file', ncol=ncol)
            + labs(
                x='Lambda (0 = corpus-focused, 1 = base-focused)',
                y='Base Vocabulary Overlap (%)',
                title='Vocabulary Overlap with Base Tokenizer vs. Lambda'
            )
            + theme_minimal()
            + theme(
                figure_size=(width, height),
                plot_title=element_text(size=14, face='bold'),
                axis_title=element_text(size=11),
                strip_text=element_text(size=10, face='bold'),
                strip_background=element_rect(fill='lightgray', alpha=0.3)
            )
        )
        plots['overlap'] = p_overlap

    return plots


def main():
    parser = argparse.ArgumentParser(
        description="Plot tokenizer comparison metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'csv_file',
        help='CSV file from tokenizer_efficiency.py'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: tokenizer_comparison.png)'
    )
    parser.add_argument(
        '--width',
        type=float,
        default=10,
        help='Plot width in inches (default: 10)'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=6,
        help='Plot height in inches (default: 6)'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['compression', 'overlap', 'fertility'],
        help='Specific metrics to plot (default: all available)'
    )
    parser.add_argument(
        '--separate',
        action='store_true',
        help='Create separate plots for each metric'
    )
    parser.add_argument(
        '--dual-axis',
        action='store_true',
        default=True,
        help='Use dual y-axis for compression and overlap (default: True, use --no-dual-axis to disable)'
    )
    parser.add_argument(
        '--no-dual-axis',
        dest='dual_axis',
        action='store_false',
        help='Disable dual y-axis (use single scale with free_y per facet)'
    )
    parser.add_argument(
        '--lambda-pattern',
        help='Custom regex pattern to extract lambda (must have one capture group)'
    )
    parser.add_argument(
        '--title',
        help='Custom plot title'
    )

    args = parser.parse_args()

    # Check that CSV exists
    if not Path(args.csv_file).exists():
        print(f"ERROR: CSV file not found: {args.csv_file}")
        sys.exit(1)

    # Load and prepare data
    print(f"Loading data from {args.csv_file}...")
    try:
        data = prepare_data(args.csv_file, args.lambda_pattern)
    except Exception as e:
        print(f"ERROR preparing data: {e}")
        sys.exit(1)

    print(f"Found {len(data['lambda_value'].unique())} lambda values")
    print(f"Found {len(data['file'].unique())} text files")
    print(f"Found {len(data['metric'].unique())} metrics: {', '.join(data['metric'].unique())}")
    print()

    # Filter metrics if specified
    if args.metrics:
        metric_map = {
            'compression': 'Compression (chars/token)',
            'overlap': 'Base Overlap (%)',
            'fertility': 'Fertility (tokens/word)'
        }
        metric_names = [metric_map[m] for m in args.metrics if metric_map.get(m) in data['metric'].unique()]
        if not metric_names:
            print(f"ERROR: None of the requested metrics are available in the data")
            print(f"Available metrics: {', '.join(data['metric'].unique())}")
            sys.exit(1)
    else:
        metric_names = None

    # Create plot(s)
    try:
        if args.separate:
            print("Creating separate plots for each metric...")
            plots = create_separate_plots(data, args.width, args.height)

            # Save each plot
            output_base = Path(args.output or 'tokenizer_comparison')
            output_dir = output_base.parent
            output_stem = output_base.stem
            output_suffix = output_base.suffix or '.png'

            for metric_name, plot in plots.items():
                output_path = output_dir / f"{output_stem}_{metric_name}{output_suffix}"
                print(f"Saving {metric_name} plot to {output_path}...")
                plot.save(output_path, dpi=300, verbose=False)
        elif args.dual_axis:
            # Check if we have both metrics needed for dual-axis
            has_compression = 'Compression (chars/token)' in data['metric'].values
            has_overlap = 'Base Overlap (%)' in data['metric'].values

            output_path = args.output or 'tokenizer_comparison.png'

            if has_compression and has_overlap:
                print("Creating dual-axis plot...")
                print(f"Saving plot to {output_path}...")
                create_dual_axis_plot(
                    data,
                    width=args.width,
                    height=args.height,
                    title=args.title,
                    output_path=output_path
                )
            else:
                print("WARNING: Dual-axis requires both compression and overlap metrics.")
                print("Falling back to single-scale plot...")
                plot = create_plot(
                    data,
                    metrics=metric_names,
                    width=args.width,
                    height=args.height,
                    title=args.title
                )
                print(f"Saving plot to {output_path}...")
                plot.save(output_path, dpi=300, verbose=False)
        else:
            print("Creating combined plot...")
            plot = create_plot(
                data,
                metrics=metric_names,
                width=args.width,
                height=args.height,
                title=args.title
            )

            # Save plot
            output_path = args.output or 'tokenizer_comparison.png'
            print(f"Saving plot to {output_path}...")
            plot.save(output_path, dpi=300, verbose=False)

    except Exception as e:
        print(f"ERROR creating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
