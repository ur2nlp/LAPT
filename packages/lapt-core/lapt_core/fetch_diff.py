"""Compare a remote run inventory against local files to decide what to fetch.

Reads tab-separated lines on stdin, one per remote run:

    experiment_id \t dir_name \t trainer_state_path \t config_path

and prints the `scp` commands for the runs that are missing or stale. A run
already recorded locally as finished is never re-fetched; one still training has
only its trainer state refreshed, since a config cannot change mid-run.

The remote is an ssh destination -- a hostname, or an alias from your ssh config
-- given by `--remote` or by `$<PREFIX>_REMOTE`, where the prefix is whatever the
calling project passes to `main()`. Nothing is hardcoded: the host and the remote
paths belong to whoever runs this.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml


def normalize_exp_id(exp_id: str) -> str:
    """Zero-pad the numeric run of a `v`-prefixed experiment id.

    Turns `v8L` into `v08L` so that hand-typed ids sort the way they read --
    `v1` before `v10` rather than after it -- and so the two spellings resolve
    to one run rather than two.

    Only useful where ids are typed by hand and the same run can be written
    more than one way. A project whose ids come from a config verbatim should
    leave this off: padding would then rewrite an id its own records do not use.

    Args:
        exp_id: The experiment id as declared remotely.

    Returns:
        The padded id, or `exp_id` unchanged if it does not match `v<digits>`.
    """
    match = re.match(r"^(v)(\d+)(.*)$", exp_id)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}{match.group(3)}"
    return exp_id


def _unchanged(exp_id: str) -> str:
    """Return the id as given; the default when normalization is off."""
    return exp_id


def get_local_status(filepath: Path) -> str:
    """Determine run status from a local trainer_state.json file.

    Args:
        filepath: Path to a trainer_state.json file.

    Returns:
        One of 'complete', 'early_stopped' or 'training'.
    """
    with open(filepath) as state_file:
        data = json.load(state_file)
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
    """Return the experiment IDs manually marked as closed in the registry.

    Args:
        path: Path to registry.yaml. A missing file yields an empty set.

    Returns:
        The set of experiment IDs whose status is 'manually_closed'.
    """
    if not path.exists():
        return set()
    with open(path) as registry_file:
        data = yaml.safe_load(registry_file) or {}
    return {
        exp_id for exp_id, entry in data.items()
        if entry.get("status") == "manually_closed"
    }


def main(
    env_prefix: str,
    normalize_ids: bool = False,
    epilog: str | None = None,
) -> None:
    """Diff a remote inventory against local outputs and emit scp commands.

    Args:
        env_prefix: Prefix for the environment variables this reads, so `LAPT`
            gives `$LAPT_REMOTE` and `$LAPT_OUTPUTS_DIR`. Separate variables are
            what let two checkouts share one shell.
        normalize_ids: Whether to zero-pad `v`-prefixed ids by default. Off
            unless a project's ids are hand-typed; `--normalize-ids` and
            `--no-normalize-ids` override it per invocation.
        epilog: Usage examples for `--help`, normally the calling script's
            `__doc__`.
    """
    remote_var = f"{env_prefix}_REMOTE"
    outputs_var = f"{env_prefix}_OUTPUTS_DIR"
    default_outputs_dir = Path(os.environ.get(outputs_var, "outputs"))

    parser = argparse.ArgumentParser(
        description="Diff remote runs against local outputs and emit scp commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without emitting scp commands",
    )
    parser.add_argument(
        "--remote",
        default=os.environ.get(remote_var),
        help=(
            "ssh destination the runs are fetched from, e.g. a hostname or an "
            f"alias from your ssh config. Defaults to ${remote_var}."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=default_outputs_dir,
        help=(
            "Directory holding registry.yaml, configs/ and trainer_states/ "
            f"(default: {default_outputs_dir}, or ${outputs_var})"
        ),
    )
    parser.add_argument(
        "--normalize-ids",
        action=argparse.BooleanOptionalAction,
        default=normalize_ids,
        help=(
            "Zero-pad v-prefixed experiment ids (v8L -> v08L) so they sort as "
            f"they read (default: {normalize_ids})"
        ),
    )
    args = parser.parse_args()
    normalize = normalize_exp_id if args.normalize_ids else _unchanged

    if not args.remote:
        parser.error(f"no remote host: pass --remote HOST or set ${remote_var}")

    trainer_states_dir = args.outputs_dir / "trainer_states"
    configs_dir = args.outputs_dir / "configs"
    registry_path = args.outputs_dir / "registry.yaml"

    manually_closed = load_manually_closed(registry_path)

    # build local inventory from outputs/trainer_states/
    local_runs: dict[str, str] = {}
    if trainer_states_dir.exists():
        for filepath in trainer_states_dir.glob("*.json"):
            try:
                local_runs[normalize(filepath.stem)] = get_local_status(filepath)
            except (json.JSONDecodeError, KeyError):
                local_runs[normalize(filepath.stem)] = "unknown"

    # track which configs already exist locally
    local_configs: set[str] = set()
    if configs_dir.exists():
        for filepath in configs_dir.glob("*.yaml"):
            local_configs.add(normalize(filepath.stem))

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
        exp_id = normalize(parts[0])
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
        print(f'mkdir -p "{trainer_states_dir}" "{configs_dir}"')

    for exp_id, remote_ts, remote_cfg, reason in to_fetch:
        ts_dest = trainer_states_dir / f"{exp_id}.json"
        if args.dry_run:
            cfg_note = " +config" if remote_cfg else ""
            print(f"  {exp_id}: {reason}{cfg_note}", file=sys.stderr)
        else:
            print(f'scp "{args.remote}:{remote_ts}" "{ts_dest}"')
            if remote_cfg:
                cfg_dest = configs_dir / f"{exp_id}.yaml"
                print(f'scp "{args.remote}:{remote_cfg}" "{cfg_dest}"')
