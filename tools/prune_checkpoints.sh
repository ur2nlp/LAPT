#!/usr/bin/env bash
# Delete intermediate `checkpoint-<step>` directories from finished training runs.
#
# A run directory is pruned only when its `best-checkpoint` is finalized, i.e.
# `best-checkpoint/training_config.yaml` exists. LAPT writes that file last, after
# `trainer.train()` returns and the best model is saved, so its presence means the
# run completed and the weights in `best-checkpoint` are whole. Runs that are still
# training, crashed, or were preempted keep their checkpoints so they can resume.
#
# Only directories named exactly `checkpoint-<digits>` are deleted; `best-checkpoint`,
# `snapshots/`, and everything else are left untouched. As an extra guard, a
# checkpoint modified *after* the best-checkpoint was finalized is kept (this can
# happen if a finished run's output directory is reused by a new job).
#
# Dry run by default: lists what would be deleted and how much space it would free.
# Pass -x to actually delete.
#
# Like remote_inventory.sh, this is meant to run on the remote host. The parent
# directories holding run folders are passed as repeated -b flags, or as a
# colon-separated LAPT_MODEL_DIRS exported by the remote shell.
#
# Usage:
#   ssh "$LAPT_REMOTE" 'bash -s' < tools/prune_checkpoints.sh -- -b /path/to/models/gothic_instruct
#   ssh "$LAPT_REMOTE" 'bash -s' < tools/prune_checkpoints.sh -- -b /a -b /b -x
#
# When run directly on the remote host, omit the `--` (it is only needed to end
# `bash -s`'s own option parsing):
#   bash tools/prune_checkpoints.sh -b /path/to/models/gothic_instruct -x

set -euo pipefail

BASE_DIRS=()
EXECUTE=0

while getopts "b:x" opt; do
    case "$opt" in
        b) BASE_DIRS+=("$OPTARG") ;;
        x) EXECUTE=1 ;;
        *) echo "Usage: $0 [-b DIR]... [-x]" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# reject leftover arguments: a stray `--` makes getopts stop early, silently
# dropping -b/-x and falling back to LAPT_MODEL_DIRS
if [ $# -gt 0 ]; then
    echo "Unexpected arguments: $*" >&2
    echo "Usage: $0 [-b DIR]... [-x]" >&2
    exit 1
fi

# fall back to the remote environment when no -b was given
if [ ${#BASE_DIRS[@]} -eq 0 ] && [ -n "${LAPT_MODEL_DIRS:-}" ]; then
    IFS=':' read -r -a BASE_DIRS <<< "$LAPT_MODEL_DIRS"
fi

if [ ${#BASE_DIRS[@]} -eq 0 ]; then
    echo "No model directories given. Pass -b DIR (repeatable), or export" >&2
    echo "LAPT_MODEL_DIRS as a colon-separated list on the remote host." >&2
    exit 1
fi

if [ "$EXECUTE" -eq 1 ]; then
    action="deleted"
else
    action="would delete"
fi

total_kb=0
num_deleted=0

for base_dir in "${BASE_DIRS[@]}"; do
    if [ ! -d "$base_dir" ]; then
        echo "skip: $base_dir is not a directory" >&2
        continue
    fi

    for run_dir in "$base_dir"/*/; do
        run_dir="${run_dir%/}"
        [ -d "$run_dir" ] || continue

        # collect directories named exactly checkpoint-<step>
        checkpoints=()
        for candidate in "$run_dir"/checkpoint-*; do
            [ -d "$candidate" ] || continue
            if [[ "$(basename "$candidate")" =~ ^checkpoint-[0-9]+$ ]]; then
                checkpoints+=("$candidate")
            fi
        done
        [ ${#checkpoints[@]} -eq 0 ] && continue

        best_marker="$run_dir/best-checkpoint/training_config.yaml"
        if [ ! -f "$best_marker" ]; then
            echo "keep: $run_dir (no finalized best-checkpoint)"
            continue
        fi

        for checkpoint in "${checkpoints[@]}"; do
            if [ "$checkpoint" -nt "$best_marker" ]; then
                echo "keep: $checkpoint (newer than best-checkpoint)"
                continue
            fi

            size_kb=$(du -sk "$checkpoint" | cut -f1)
            total_kb=$((total_kb + size_kb))
            num_deleted=$((num_deleted + 1))

            if [ "$EXECUTE" -eq 1 ]; then
                rm -rf -- "$checkpoint"
            fi
            echo "$action: $checkpoint ($(numfmt --to=iec --from-unit=1024 "$size_kb"))"
        done
    done
done

echo "$action $num_deleted checkpoint(s), $(numfmt --to=iec --from-unit=1024 "$total_kb") total"
if [ "$EXECUTE" -eq 0 ] && [ "$num_deleted" -gt 0 ]; then
    echo "(dry run; pass -x to delete)"
fi
