"""Inject bits-per-character (BPC) metrics into a trainer_state.json file.

For runs that pre-date native BPC logging, BPC can be back-computed from
eval loss (nats) given the average characters-per-token for each eval set:

    bpc = loss / (chars_per_token * math.log(2))

The conversion from nats to bits is exact; the only approximation is the
per-set average chars/token coefficient, which the user must supply.

For each eval set NAME with a provided coefficient, the script looks for
``eval_{NAME}_loss`` entries in log_history and writes ``eval_{NAME}_bpc``
alongside them. The file is updated in-place.

Usage:
    python tools/inject_bpc.py outputs/trainer_states/v8L.json \\
        --coeff got_holdout 3.42 \\
        --coeff got-eng_holdout 3.15 \\
        --coeff ang 3.81 \\
        --coeff non 4.02 \\
        --coeff eng 4.55

    # Dry-run: print modified entries without writing
    python tools/inject_bpc.py outputs/trainer_states/v8L.json \\
        --coeff got_holdout 3.42 --dry-run

    # Process multiple files with the same coefficients
    python tools/inject_bpc.py outputs/trainer_states/v*.json \\
        --coeff got_holdout 3.42 --coeff got-eng_holdout 3.15
"""

import argparse
import json
import math
import sys
from pathlib import Path


def compute_bpc(loss_nats: float, chars_per_token: float) -> float:
    """Convert cross-entropy loss in nats to bits per character."""
    return loss_nats / (chars_per_token * math.log(2))


def inject_bpc_into_state(
    state: dict,
    coefficients: dict[str, float],
) -> tuple[dict, int]:
    """Add BPC fields to log_history entries in-place.

    Args:
        state: Parsed trainer_state dict (modified in-place).
        coefficients: Mapping from eval-set name to avg chars/token.

    Returns:
        The modified state and count of entries written.
    """
    written = 0
    for entry in state.get("log_history", []):
        for name, chars_per_token in coefficients.items():
            loss_key = f"eval_{name}_loss"
            bpc_key = f"eval_{name}_bpc"
            if loss_key in entry and bpc_key not in entry:
                entry[bpc_key] = compute_bpc(entry[loss_key], chars_per_token)
                written += 1
    return state, written


def process_file(
    path: Path,
    coefficients: dict[str, float],
    dry_run: bool,
) -> None:
    with path.open() as f:
        state = json.load(f)

    state, written = inject_bpc_into_state(state, coefficients)

    if dry_run:
        # Print only the modified log entries for inspection.
        relevant_keys = {f"eval_{name}_bpc" for name in coefficients}
        for entry in state.get("log_history", []):
            if relevant_keys & entry.keys():
                print(json.dumps(entry, indent=2))
        print(f"\n[dry-run] Would write {written} BPC values to {path}", file=sys.stderr)
        return

    with path.open("w") as f:
        json.dump(state, f, indent=2)
    print(f"Wrote {written} BPC values to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject BPC metrics into trainer_state.json files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        metavar="trainer_state.json",
        help="One or more trainer_state.json files to update.",
    )
    parser.add_argument(
        "--coeff",
        nargs=2,
        action="append",
        metavar=("NAME", "CHARS_PER_TOKEN"),
        required=True,
        help=(
            "Eval-set name and its average characters-per-token coefficient. "
            "May be repeated for multiple sets."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print modified entries without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    coefficients = {}
    for name, value in args.coeff:
        try:
            coefficients[name] = float(value)
        except ValueError:
            print(f"Error: coefficient for '{name}' must be a number, got '{value}'", file=sys.stderr)
            sys.exit(1)

    for path in args.files:
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.", file=sys.stderr)
            continue
        process_file(path, coefficients, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
