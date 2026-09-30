"""
Plot training metrics from HuggingFace Trainer logs.

Supports two input formats:
1. trainer_state.json files (recommended - proper JSON)
2. Raw log output (legacy - requires string manipulation)

Usage:
    # Single run
    python tools/training_plot.py --metric loss --state-file models/run1/checkpoint-1000/trainer_state.json

    # Compare multiple runs (regex matched against file paths under current directory)
    python tools/training_plot.py --metric eval_loss --state-pattern "models/.*/trainer_state\\.json"

    # Use raw logs (legacy)
    python tools/training_plot.py --metric loss --log-file training.log --skip-lines 5

    # Multiple metrics
    python tools/training_plot.py --metrics loss eval_loss learning_rate --state-file path/to/trainer_state.json

    # Regex patterns for metrics (matched with re.fullmatch)
    python tools/training_plot.py --metrics "eval_.*" --state-file path/to/trainer_state.json
    python tools/training_plot.py --metric "eval_.*loss" --state-file path/to/trainer_state.json

    # Mix literal names and patterns
    python tools/training_plot.py --metrics loss "eval_.*" learning_rate --state-file path/to/trainer_state.json

    # Save to file instead of showing
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --output plot.png

    # Set y-axis limits shared by all subplots
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --ylim 0 5
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --ylim 0  # lower bound only

    # Set per-subplot y-axis limits (METRIC:LOWER[:UPPER]; "none" for an auto bound)
    python tools/training_plot.py --metrics loss eval_loss --state-file path/to/trainer_state.json --ylims loss:0:5 eval_loss:none:3

    # Mix a shared default (--ylim) with per-metric overrides (--ylims wins for named metrics)
    python tools/training_plot.py --metrics loss eval_loss grad_norm --state-file path/to/trainer_state.json --ylim 0 10 --ylims eval_loss:1:3
"""

from lapt_core.plotting import main


if __name__ == '__main__':
    main(epilog=__doc__)
