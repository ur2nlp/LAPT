"""Compare remote run inventory against local files to decide what to fetch.

Reads tab-separated lines from stdin:
    experiment_id \t dir_name \t trainer_state_path \t config_path

Checks local trainer_states/ and configs/ directories to determine what
already exists. Prints scp commands for runs that need fetching.

Usage:
    ssh circ 'bash -s' < tools/remote_inventory.sh | python tools/fetch_diff.py
    ssh circ 'bash -s' < tools/remote_inventory.sh | python tools/fetch_diff.py --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml


TRAINER_STATES_DIR = Path("outputs/trainer_states")
CONFIGS_DIR = Path("outputs/configs")
REGISTRY_PATH = Path("outputs/registry.yaml")


def normalize_exp_id(exp_id: str) -> str:
    """Normalize experiment ID to canonical zero-padded form.

    Pads single-digit numbers to two digits so that 'v8L' and 'v08L' are
    treated as the same experiment and stored consistently as 'v08L'.
    """
    match = re.match(r'^(v)(\d+)(.*)$', exp_id)
    if match:
        num = int(match.group(2))
        return f"{match.group(1)}{num:02d}{match.group(3)}"
    return exp_id


def get_local_status(filepath: Path) -> str:
    """Determine run status from a local trainer_state.json file."""
    with open(filepath) as f:
        data = json.load(f)
    last = data["log_history"][-1] if data.get("log_history") else {}
    has_runtime = "train_runtime" in last
    global_step = data.get("global_step", 0)
    max_steps = data.get("max_steps", 0)
    if has_runtime and global_step >= max_steps:
        return "complete"
    elif has_runtime:
        return "early_stopped"
    return "training"


def load_manually_closed(path: Path) -> set[str]:
    """Return the set of experiment IDs manually marked as closed in the registry."""
    if not path.exists():
        return set()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {exp_id for exp_id, entry in data.items() if entry.get("status") == "manually_closed"}


def main():
    parser = argparse.ArgumentParser(
        description="Diff remote runs against local outputs and emit scp commands",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without emitting scp commands",
    )
    args = parser.parse_args()

    manually_closed = load_manually_closed(REGISTRY_PATH)

    # build local inventory from outputs/trainer_states/
    local_runs: dict[str, str] = {}
    if TRAINER_STATES_DIR.exists():
        for filepath in TRAINER_STATES_DIR.glob("*.json"):
            try:
                local_runs[normalize_exp_id(filepath.stem)] = get_local_status(filepath)
            except (json.JSONDecodeError, KeyError):
                local_runs[normalize_exp_id(filepath.stem)] = "unknown"

    # track which configs already exist locally
    local_configs: set[str] = set()
    if CONFIGS_DIR.exists():
        for filepath in CONFIGS_DIR.glob("*.yaml"):
            local_configs.add(normalize_exp_id(filepath.stem))

    # read remote inventory from stdin
    to_fetch: list[tuple[str, str, str | None, str]] = []
    skipped = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            print(f"Warning: malformed line: {line}", file=sys.stderr)
            continue
        exp_id = normalize_exp_id(parts[0])
        remote_trainer_state = parts[2]
        remote_config = parts[3] if len(parts) >= 4 else None

        local_status = local_runs.get(exp_id)
        if local_status in ("complete", "early_stopped") or exp_id in manually_closed:
            skipped += 1
            continue

        # decide what to fetch
        if local_status == "training":
            # config won't change mid-training, only fetch updated trainer_state
            to_fetch.append((exp_id, remote_trainer_state, None, "update"))
        else:
            # new run: fetch trainer_state, and config if not already local
            need_config = remote_config if exp_id not in local_configs else None
            to_fetch.append((exp_id, remote_trainer_state, need_config, "new"))

    # report to stderr
    print(f"# {skipped} skipped (complete/early_stopped/manually_closed)", file=sys.stderr)
    print(f"# {len(to_fetch)} to fetch", file=sys.stderr)

    if not to_fetch:
        print("# Nothing to fetch", file=sys.stderr)
        sys.exit(0)

    if not args.dry_run:
        print(f'mkdir -p "{TRAINER_STATES_DIR}" "{CONFIGS_DIR}"')

    for exp_id, remote_ts, remote_cfg, reason in to_fetch:
        ts_dest = TRAINER_STATES_DIR / f"{exp_id}.json"
        if args.dry_run:
            cfg_note = " +config" if remote_cfg else ""
            print(f"  {exp_id}: {reason}{cfg_note}", file=sys.stderr)
        else:
            print(f'scp "circ:{remote_ts}" "{ts_dest}"')
            if remote_cfg:
                cfg_dest = CONFIGS_DIR / f"{exp_id}.yaml"
                print(f'scp "circ:{remote_cfg}" "{cfg_dest}"')


if __name__ == "__main__":
    main()
