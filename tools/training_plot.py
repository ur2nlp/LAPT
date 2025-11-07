"""
Plot training metrics from HuggingFace Trainer logs.

Supports two input formats:
1. trainer_state.json files (recommended - proper JSON)
2. Raw log output (legacy - requires string manipulation)

Usage:
    # Single run
    python tools/training_plot.py --metric loss --state-file models/run1/checkpoint-1000/trainer_state.json

    # Compare multiple runs
    python tools/training_plot.py --metric eval_loss --state-pattern "models/*/checkpoint-*/trainer_state.json"

    # Use raw logs (legacy)
    python tools/training_plot.py --metric loss --log-file training.log --skip-lines 5

    # Multiple metrics
    python tools/training_plot.py --metrics loss eval_loss learning_rate --state-file path/to/trainer_state.json

    # Save to file instead of showing
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --output plot.png
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_rect,
    element_text,
    facet_wrap,
    geom_line,
    ggplot,
    labs,
    theme,
    theme_bw,
    theme_minimal,
)


def load_from_trainer_state(filepath):
    """Load log history from trainer_state.json (proper JSON format)."""
    with open(filepath, 'r') as f:
        state = json.load(f)
    return pd.DataFrame(state['log_history'])


def load_from_raw_log(filepath, skip_lines=0):
    """Load from raw log output (legacy - almost-JSON format)."""
    lines = open(filepath, 'r').readlines()[skip_lines:]
    # HF logs use single quotes, need to convert to double quotes for JSON
    jsonl_text = '\n'.join([line.replace('\'', '\"') for line in lines])
    return pd.read_json(jsonl_text, lines=True)


def load_data(state_file=None, state_pattern=None, log_file=None, log_pattern=None, skip_lines=0):
    """Load training data from various sources."""
    dataframes = []

    if state_pattern:
        # Multiple trainer_state.json files
        files = glob.glob(state_pattern)
        if not files:
            print(f"Warning: No files found matching pattern: {state_pattern}", file=sys.stderr)
        for filepath in files:
            df = load_from_trainer_state(filepath)
            df['run'] = str(Path(filepath).parent.parent.name)  # Extract run name from path
            dataframes.append(df)

    elif state_file:
        # Single trainer_state.json file
        df = load_from_trainer_state(state_file)
        df['run'] = Path(state_file).parent.parent.name
        dataframes.append(df)

    elif log_pattern:
        # Multiple raw log files
        files = glob.glob(log_pattern)
        if not files:
            print(f"Warning: No files found matching pattern: {log_pattern}", file=sys.stderr)
        for filepath in files:
            df = load_from_raw_log(filepath, skip_lines)
            df['run'] = str(Path(filepath).stem)
            dataframes.append(df)

    elif log_file:
        # Single raw log file
        df = load_from_raw_log(log_file, skip_lines)
        df['run'] = Path(log_file).stem
        dataframes.append(df)

    else:
        raise ValueError("Must provide one of: --state-file, --state-pattern, --log-file, or --log-pattern")

    if not dataframes:
        raise ValueError("No data loaded. Check file paths.")

    return pd.concat(dataframes, ignore_index=True)


def plot_metric(data, metric, x_axis='step', output=None, title=None):
    """Create a plot for a single metric."""
    # Filter to rows where metric exists
    metric_data = data[data[metric].notna()].copy()

    if len(metric_data) == 0:
        print(f"Warning: No data found for metric '{metric}'", file=sys.stderr)
        print(f"Available metrics: {[col for col in data.columns if data[col].notna().any()]}", file=sys.stderr)
        return None

    # Determine if we're comparing multiple runs
    multiple_runs = len(metric_data['run'].unique()) > 1

    # Choose x-axis (prefer 'step' over 'epoch' if available)
    if x_axis not in metric_data.columns or metric_data[x_axis].isna().all():
        # Fallback to epoch if step not available
        x_axis = 'epoch' if 'epoch' in metric_data.columns else 'step'

    plot = (
        ggplot(metric_data, aes(x=x_axis, y=metric)) +
        (geom_line(aes(color='run'), size=1.2) if multiple_runs else geom_line(size=1.2)) +
        labs(
            title=title or f'{metric} over training',
            x=x_axis.capitalize(),
            y=metric
        ) +
        theme_bw() +
        theme(
            legend_position="bottom" if multiple_runs else "none",
            axis_title=element_text(size=14),
            legend_title=element_text(size=12),
            legend_text=element_text(size=10),
            axis_text=element_text(size=10),
            figure_size=(10, 6),
            plot_background=element_rect(fill='white'),
            panel_background=element_rect(fill='white')
        )
    )

    if output:
        plot.save(output, dpi=300, verbose=False, transparent=False)
        print(f"Saved plot to {output}")
    else:
        plot.show()

    return plot


def plot_multiple_metrics(data, metrics, x_axis='step', output=None):
    """Create subplots for multiple metrics."""
    # Reshape data for faceting
    plot_data = []
    for metric in metrics:
        metric_data = data[data[metric].notna()].copy()
        metric_data['metric_name'] = metric
        metric_data['metric_value'] = metric_data[metric]
        plot_data.append(metric_data[[x_axis, 'run', 'metric_name', 'metric_value']])

    plot_data = pd.concat(plot_data, ignore_index=True)

    multiple_runs = len(plot_data['run'].unique()) > 1

    plot = (
        ggplot(plot_data, aes(x=x_axis, y='metric_value')) +
        (geom_line(aes(color='run'), size=1.0) if multiple_runs else geom_line(size=1.0)) +
        facet_wrap('~metric_name', scales='free_y', ncol=2) +
        labs(x=x_axis.capitalize(), y='Value') +
        theme_minimal() +
        theme(
            legend_position="bottom" if multiple_runs else "none",
            axis_title=element_text(size=12),
            legend_title=element_text(size=10),
            legend_text=element_text(size=8),
            axis_text=element_text(size=8),
            figure_size=(12, 4 * ((len(metrics) + 1) // 2)),
            plot_background=element_rect(fill='white'),
            panel_background=element_rect(fill='white')
        )
    )

    if output:
        plot.save(output, dpi=300, verbose=False, transparent=False)
        print(f"Saved plot to {output}")
    else:
        plot.show()

    return plot


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from HuggingFace Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input sources (mutually exclusive groups would be nice, but keeping it simple)
    parser.add_argument('--state-file', type=str, help='Path to trainer_state.json (recommended)')
    parser.add_argument('--state-pattern', type=str, help='Glob pattern for multiple trainer_state.json files')
    parser.add_argument('--log-file', type=str, help='Path to raw log file (legacy)')
    parser.add_argument('--log-pattern', type=str, help='Glob pattern for multiple raw log files (legacy)')
    parser.add_argument('--skip-lines', type=int, default=0, help='Skip N lines from raw logs (default: 0)')

    # Metrics to plot
    parser.add_argument('--metric', type=str, help='Single metric to plot (e.g., loss, eval_loss)')
    parser.add_argument('--metrics', nargs='+', help='Multiple metrics to plot as subplots')

    # Plot options
    parser.add_argument('--x-axis', type=str, default='step', choices=['step', 'epoch'],
                       help='X-axis variable (default: step)')
    parser.add_argument('--output', type=str, help='Save plot to file instead of showing')
    parser.add_argument('--title', type=str, help='Custom plot title')
    parser.add_argument('--list-metrics', action='store_true', help='List available metrics and exit')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.state_file, args.state_pattern, args.log_file, args.log_pattern]):
        parser.error("Must provide one of: --state-file, --state-pattern, --log-file, or --log-pattern")

    if not args.list_metrics and not args.metric and not args.metrics:
        parser.error("Must provide either --metric or --metrics (or use --list-metrics)")

    # Load data
    data = load_data(
        state_file=args.state_file,
        state_pattern=args.state_pattern,
        log_file=args.log_file,
        log_pattern=args.log_pattern,
        skip_lines=args.skip_lines
    )

    # List metrics if requested
    if args.list_metrics:
        available_metrics = [col for col in data.columns if data[col].notna().any() and col not in ['run', 'step', 'epoch']]
        print("Available metrics:")
        for metric in sorted(available_metrics):
            count = data[metric].notna().sum()
            print(f"  {metric} ({count} values)")
        return

    # Plot
    if args.metrics:
        plot_multiple_metrics(data, args.metrics, x_axis=args.x_axis, output=args.output)
    else:
        plot_metric(data, args.metric, x_axis=args.x_axis, output=args.output, title=args.title)


if __name__ == '__main__':
    main()
