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

    # Glob patterns for metrics
    python tools/training_plot.py --metrics "eval_*" --state-file path/to/trainer_state.json
    python tools/training_plot.py --metric "eval_distinct_*" --state-file path/to/trainer_state.json

    # Mix literal names and patterns
    python tools/training_plot.py --metrics loss "eval_*" learning_rate --state-file path/to/trainer_state.json

    # Save to file instead of showing
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --output plot.png

    # Set y-axis limits
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --ylim 0 5
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --ylim 0  # lower bound only
"""

import argparse
import fnmatch
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
    geom_point,
    ggplot,
    labs,
    theme,
    theme_bw,
    theme_minimal,
    ylim,
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


def load_data(state_file=None, state_pattern=None, log_file=None, log_pattern=None, skip_lines=0, run_names=None, exclude_pattern=None):
    """Load training data from various sources."""
    dataframes = []
    files = []

    if state_pattern:
        # Multiple trainer_state.json files
        files = sorted(glob.glob(state_pattern))

        # Filter out excluded files
        if exclude_pattern:
            files = [f for f in files if not fnmatch.fnmatch(f, exclude_pattern)]

        if not files:
            print(f"Warning: No files found matching pattern: {state_pattern}", file=sys.stderr)
        for idx, filepath in enumerate(files):
            df = load_from_trainer_state(filepath)
            # Use custom name if provided, otherwise use full filepath
            if run_names and idx < len(run_names):
                df['run'] = run_names[idx]
            else:
                df['run'] = filepath
            dataframes.append(df)

    elif state_file:
        # Single trainer_state.json file
        df = load_from_trainer_state(state_file)
        df['run'] = run_names[0] if run_names else state_file
        dataframes.append(df)

    elif log_pattern:
        # Multiple raw log files
        files = sorted(glob.glob(log_pattern))

        # Filter out excluded files
        if exclude_pattern:
            files = [f for f in files if not fnmatch.fnmatch(f, exclude_pattern)]

        if not files:
            print(f"Warning: No files found matching pattern: {log_pattern}", file=sys.stderr)
        for idx, filepath in enumerate(files):
            df = load_from_raw_log(filepath, skip_lines)
            # Use custom name if provided, otherwise use full filepath
            if run_names and idx < len(run_names):
                df['run'] = run_names[idx]
            else:
                df['run'] = filepath
            dataframes.append(df)

    elif log_file:
        # Single raw log file
        df = load_from_raw_log(log_file, skip_lines)
        df['run'] = run_names[0] if run_names else log_file
        dataframes.append(df)

    else:
        raise ValueError("Must provide one of: --state-file, --state-pattern, --log-file, or --log-pattern")

    if not dataframes:
        raise ValueError("No data loaded. Check file paths.")

    return pd.concat(dataframes, ignore_index=True)


def plot_metric(data, metric, x_axis='step', output=None, title=None, y_limits=None):
    """Create a plot for a single metric.

    Args:
        y_limits: Tuple of (lower, upper) for y-axis. Either can be None for auto.
    """
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

    # Find min and max points to mark
    min_idx = metric_data[metric].idxmin()
    max_idx = metric_data[metric].idxmax()
    extrema_data = metric_data.loc[[min_idx, max_idx]].copy()

    plot = (
        ggplot(metric_data, aes(x=x_axis, y=metric)) +
        (geom_line(aes(color='run'), size=1.2) if multiple_runs else geom_line(size=1.2)) +
        (geom_point(aes(color='run'), data=extrema_data, size=4, shape='x') if multiple_runs
         else geom_point(data=extrema_data, size=4, shape='x')) +
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

    if y_limits:
        plot = plot + ylim(y_limits)

    if output:
        plot.save(output, dpi=300, verbose=False, transparent=False)
        print(f"Saved plot to {output}")
    else:
        plot.show()

    return plot


def expand_metric_patterns(patterns, available_metrics):
    """
    Expand glob patterns to matching metric names.

    Supports literal metric names, glob patterns (*, ?, []), and mixed usage.

    Args:
        patterns: List of metric names or glob patterns
        available_metrics: List of all available metric names

    Returns:
        List of matched metric names (preserving order, no duplicates)

    Examples:
        expand_metric_patterns(['eval_*'], ['eval_loss', 'loss', 'eval_acc'])
        # Returns: ['eval_acc', 'eval_loss']

        expand_metric_patterns(['loss', 'eval_*'], ['eval_loss', 'loss'])
        # Returns: ['loss', 'eval_loss']
    """
    expanded = []
    seen = set()

    for pattern in patterns:
        if any(char in pattern for char in ['*', '?', '[', ']']):
            # It's a glob pattern - find matches
            matches = fnmatch.filter(available_metrics, pattern)
            for match in sorted(matches):
                if match not in seen:
                    expanded.append(match)
                    seen.add(match)
        else:
            # It's a literal metric name
            if pattern not in seen:
                expanded.append(pattern)
                seen.add(pattern)

    return expanded


def print_metric_summary(data, metrics, x_axis='step'):
    """Print summary statistics for metrics (min/max values and where they occurred)."""
    print("\n" + "="*80)
    print("METRIC SUMMARY")
    print("="*80)

    for metric in metrics:
        metric_data = data[data[metric].notna()].copy()

        if len(metric_data) == 0:
            continue

        print(f"\n{metric}:")
        print("-" * 80)

        # Find min and max
        min_idx = metric_data[metric].idxmin()
        max_idx = metric_data[metric].idxmax()

        min_val = metric_data.loc[min_idx, metric]
        max_val = metric_data.loc[max_idx, metric]
        min_step = metric_data.loc[min_idx, x_axis] if x_axis in metric_data.columns else 'N/A'
        max_step = metric_data.loc[max_idx, x_axis] if x_axis in metric_data.columns else 'N/A'
        min_run = metric_data.loc[min_idx, 'run']
        max_run = metric_data.loc[max_idx, 'run']

        print(f"  Min: {min_val:.6f} at {x_axis}={min_step} (run: {min_run})")
        print(f"  Max: {max_val:.6f} at {x_axis}={max_step} (run: {max_run})")

    print("\n" + "="*80 + "\n")


def plot_multiple_metrics(data, metrics, x_axis='step', output=None, y_limits=None):
    """Create subplots for multiple metrics.

    Args:
        y_limits: Tuple of (lower, upper) for y-axis. Either can be None for auto.
    """
    # Reshape data for faceting
    plot_data = []
    extrema_data = []

    for metric in metrics:
        metric_data = data[data[metric].notna()].copy()
        metric_data['metric_name'] = metric
        metric_data['metric_value'] = metric_data[metric]
        plot_data.append(metric_data[[x_axis, 'run', 'metric_name', 'metric_value']])

        # Find min and max points for this metric
        min_idx = metric_data[metric].idxmin()
        max_idx = metric_data[metric].idxmax()
        extrema = metric_data.loc[[min_idx, max_idx]].copy()
        extrema['metric_name'] = metric
        extrema['metric_value'] = extrema[metric]
        extrema_data.append(extrema[[x_axis, 'run', 'metric_name', 'metric_value']])

    plot_data = pd.concat(plot_data, ignore_index=True)
    extrema_data = pd.concat(extrema_data, ignore_index=True)

    multiple_runs = len(plot_data['run'].unique()) > 1

    plot = (
        ggplot(plot_data, aes(x=x_axis, y='metric_value')) +
        (geom_line(aes(color='run'), size=1.0) if multiple_runs else geom_line(size=1.0)) +
        (geom_point(aes(color='run'), data=extrema_data, size=3, shape='x') if multiple_runs
         else geom_point(data=extrema_data, size=3, shape='x')) +
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

    if y_limits:
        plot = plot + ylim(y_limits)

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
    parser.add_argument('--exclude-pattern', type=str, help='Glob pattern to exclude files (e.g., "*v14*")')

    # Metrics to plot
    parser.add_argument('--metric', type=str, help='Single metric to plot (e.g., loss, eval_loss)')
    parser.add_argument('--metrics', nargs='+', help='Multiple metrics to plot as subplots')

    # Plot options
    parser.add_argument('--x-axis', type=str, default='step', choices=['step', 'epoch'],
                       help='X-axis variable (default: step)')
    parser.add_argument('--output', type=str, help='Save plot to file instead of showing')
    parser.add_argument('--title', type=str, help='Custom plot title')
    parser.add_argument('--run-names', nargs='+', help='Custom names for runs (in order of matched files)')
    parser.add_argument('--list-metrics', action='store_true', help='List available metrics and exit')
    parser.add_argument('--ylim', nargs='+', type=float, metavar='VALUE',
                       help='Y-axis limits: one value for lower bound, two for (lower, upper). Use "none" for auto.')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.state_file, args.state_pattern, args.log_file, args.log_pattern]):
        parser.error("Must provide one of: --state-file, --state-pattern, --log-file, or --log-pattern")

    if not args.list_metrics and not args.metric and not args.metrics:
        parser.error("Must provide either --metric or --metrics (or use --list-metrics)")

    # Parse y-axis limits
    y_limits = None
    if args.ylim:
        if len(args.ylim) == 1:
            y_limits = (args.ylim[0], None)  # Lower bound only
        elif len(args.ylim) == 2:
            y_limits = (args.ylim[0], args.ylim[1])
        else:
            parser.error("--ylim accepts 1 or 2 values")

    # Load data
    data = load_data(
        state_file=args.state_file,
        state_pattern=args.state_pattern,
        log_file=args.log_file,
        log_pattern=args.log_pattern,
        skip_lines=args.skip_lines,
        run_names=args.run_names,
        exclude_pattern=args.exclude_pattern
    )

    # Get available metrics
    available_metrics = [col for col in data.columns if data[col].notna().any() and col not in ['run', 'step', 'epoch']]

    # List metrics if requested
    if args.list_metrics:
        print("Available metrics:")
        for metric in sorted(available_metrics):
            count = data[metric].notna().sum()
            print(f"  {metric} ({count} values)")
        return

    # Expand metric patterns
    if args.metrics:
        metrics = expand_metric_patterns(args.metrics, available_metrics)
        if not metrics:
            print(f"Error: No metrics matched the patterns: {args.metrics}", file=sys.stderr)
            print(f"Available metrics: {sorted(available_metrics)}", file=sys.stderr)
            sys.exit(1)
        print(f"Plotting {len(metrics)} metric(s): {', '.join(metrics)}", file=sys.stderr)
        plot_multiple_metrics(data, metrics, x_axis=args.x_axis, output=args.output, y_limits=y_limits)
        print_metric_summary(data, metrics, x_axis=args.x_axis)
    else:
        # Single metric - also support pattern matching
        if any(char in args.metric for char in ['*', '?', '[', ']']):
            metrics = expand_metric_patterns([args.metric], available_metrics)
            if not metrics:
                print(f"Error: No metrics matched the pattern: {args.metric}", file=sys.stderr)
                print(f"Available metrics: {sorted(available_metrics)}", file=sys.stderr)
                sys.exit(1)
            if len(metrics) > 1:
                print(f"Pattern matched {len(metrics)} metrics, plotting all: {', '.join(metrics)}", file=sys.stderr)
                plot_multiple_metrics(data, metrics, x_axis=args.x_axis, output=args.output, y_limits=y_limits)
                print_metric_summary(data, metrics, x_axis=args.x_axis)
            else:
                print(f"Plotting: {metrics[0]}", file=sys.stderr)
                plot_metric(data, metrics[0], x_axis=args.x_axis, output=args.output, title=args.title, y_limits=y_limits)
                print_metric_summary(data, [metrics[0]], x_axis=args.x_axis)
        else:
            plot_metric(data, args.metric, x_axis=args.x_axis, output=args.output, title=args.title, y_limits=y_limits)
            print_metric_summary(data, [args.metric], x_axis=args.x_axis)


if __name__ == '__main__':
    main()
