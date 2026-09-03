"""Migrate per-source cache records from `source_config.yaml` to `config.yaml`.

Source caches written before the sources became artifacts carry their tracked
parameters in `source_config.yaml`. The artifact layer looks for `config.yaml`,
and refuses a cache it cannot verify, so those caches need a record under the
new name before they can be reused.

The record is *copied*, not renamed, so both layers keep working: the artifact
layer validates against `config.yaml` while the older code path still finds
`source_config.yaml`. That makes this reversible -- delete the copies and
nothing has changed -- and means it can be run before switching branches.

Two record shapes need more than a copy. A cache predating a tracked field has
no entry for it, and one predating seed tracking has no seed. Rather than
hardcode which fields those are, the target record is built by handing the
cached one to the source class itself, so this cannot drift from the code it is
migrating toward. A field whose cached value *disagrees* with the class is
reported as a conflict and left alone: that is a cache describing something
else, not an old record.

Run from a checkout containing `lapt.sources`, on the machine holding the cache
tree. Prints a plan and changes nothing unless `--apply` is passed.
"""

import argparse
import os
import sys

import yaml

# Put the repository root on sys.path so `lapt` imports without an
# editable install, for running this script straight from a checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lapt.sources.base import LEGACY_CONFIG_FILENAME, SOURCE_TYPES
from lapt_core.artifacts import CONFIG_FILENAME


def find_legacy_records(root: str) -> list[str]:
    """Find every cache directory holding a legacy record.

    Args:
        root: Directory to walk.

    Returns:
        Sorted paths of the directories, not the records themselves.
    """
    found = []
    for dirpath, _, filenames in os.walk(root):
        if LEGACY_CONFIG_FILENAME in filenames:
            found.append(dirpath)
    return sorted(found)


def read_record(cache_dir: str, filename: str) -> dict | None:
    """Read a YAML record from a cache directory.

    Args:
        cache_dir: Directory holding the record.
        filename: Record filename.

    Returns:
        The parsed record, or None if it is absent or empty.
    """
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as record_file:
        return yaml.safe_load(record_file)


def expected_record(cache_dir: str, cached: dict, seed: int) -> dict:
    """Build the record the current code would write for this cache.

    Hands the cached parameters back to the source class the record names, so
    the target is whatever that class tracks today -- including fields added
    since the cache was built, which take their historical defaults.

    Args:
        cache_dir: The cache directory, whose parent the source is rooted at.
        cached: The legacy record.
        seed: Seed to record for sources that subsample.

    Returns:
        The target record.

    Raises:
        ValueError: If the record names a type the registry does not know.
    """
    source_type = cached.get('type')
    if source_type == 'substituted':
        return dict(cached)

    source_class = SOURCE_TYPES.get(source_type)
    source = source_class.from_config(os.path.dirname(cache_dir), cached, seed)
    return source.config()


def plan_for(cache_dir: str, seed: int) -> dict:
    """Decide what this cache directory needs.

    Args:
        cache_dir: Directory holding a legacy record.
        seed: Seed to record for sources that subsample.

    Returns:
        A dict with `status`, `additions`, `conflicts`, and `record` keys.
        Status is one of: `write`, `already-migrated`, `differs`, `conflict`,
        `unknown-type`.
    """
    cached = read_record(cache_dir, LEGACY_CONFIG_FILENAME) or {}

    try:
        target = expected_record(cache_dir, cached, seed)
    except ValueError as lookup_error:
        return {
            'status': 'unknown-type',
            'additions': {},
            'conflicts': {},
            'record': None,
            'detail': str(lookup_error),
        }

    conflicts = {
        key: (cached[key], target[key])
        for key in cached.keys() & target.keys()
        if cached[key] != target[key]
    }
    additions = {key: target[key] for key in target.keys() - cached.keys()}

    existing = read_record(cache_dir, CONFIG_FILENAME)
    if existing is not None:
        status = 'already-migrated' if existing == target else 'differs'
    elif conflicts:
        status = 'conflict'
    else:
        status = 'write'

    return {
        'status': status,
        'additions': additions,
        'conflicts': conflicts,
        'record': target,
        'detail': '',
    }


def describe(cache_dir: str, plan: dict, root: str) -> None:
    """Print one directory's plan to stdout.

    Args:
        cache_dir: The directory the plan is for.
        plan: The plan from `plan_for`.
        root: Root the path is reported relative to.
    """
    relative = os.path.relpath(cache_dir, root)
    print(f"  [{plan['status']}] {relative}")
    if plan['status'] == 'already-migrated':
        return
    for key, value in sorted(plan['additions'].items()):
        print(f"      + {key}: {value!r}")
    for key, (was, now) in sorted(plan['conflicts'].items()):
        print(f"      ! {key}: cached {was!r} vs current {now!r}")
    if plan['detail']:
        print(f"      ! {plan['detail']}")


def main() -> int:
    """Plan, and optionally apply, the record migration.

    Returns:
        Process exit status: non-zero when a directory needs attention.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('root', help="Cache tree to walk, e.g. a project's data/ directory")
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Seed to record for caches that subsample and predate seed tracking. Must be "
             "the seed those caches were actually built with (default: 1)",
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help="Write the records. Without this, only the plan is printed",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        return 2

    directories = find_legacy_records(args.root)
    if not directories:
        print(f"No {LEGACY_CONFIG_FILENAME} found under {args.root}; nothing to migrate")
        return 0

    print(f"Found {len(directories)} cache(s) with {LEGACY_CONFIG_FILENAME} under {args.root}")
    print(f"Recording seed={args.seed} for caches that subsample and predate seed tracking\n")

    plans = {directory: plan_for(directory, args.seed) for directory in directories}
    for directory, plan in plans.items():
        describe(directory, plan, args.root)

    counts = {}
    for plan in plans.values():
        counts[plan['status']] = counts.get(plan['status'], 0) + 1
    print("\nSummary: " + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())))

    needs_attention = counts.get('conflict', 0) + counts.get('differs', 0) \
        + counts.get('unknown-type', 0)

    if not args.apply:
        writable = counts.get('write', 0)
        print(f"\nDry run. Pass --apply to write {writable} record(s).")
        print(
            f"The legacy {LEGACY_CONFIG_FILENAME} files are left in place, so the older code "
            "path keeps validating too and this stays reversible."
        )
        return 1 if needs_attention else 0

    written = 0
    for directory, plan in plans.items():
        if plan['status'] != 'write':
            continue
        with open(os.path.join(directory, CONFIG_FILENAME), 'w') as record_file:
            yaml.dump(plan['record'], record_file, default_flow_style=False, sort_keys=False)
        written += 1

    print(f"\nWrote {written} record(s).")
    if needs_attention:
        print(
            f"{needs_attention} cache(s) were left untouched and are listed above; "
            "a conflict means the cached record describes different parameters, so it "
            "needs a rebuild rather than a migration.",
            file=sys.stderr,
        )
    return 1 if needs_attention else 0


if __name__ == '__main__':
    sys.exit(main())
