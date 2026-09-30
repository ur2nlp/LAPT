#!/usr/bin/env bash
# Inventory training runs on a remote cluster.
#
# Outputs TSV to stdout: experiment_id \t dir_name \t trainer_state_path \t config_path
# Intended to be piped into tools/fetch_diff.py.
#
# This script executes on the *remote* host, so it cannot read the local
# environment. The directories to scan are therefore arguments, passed either as
# repeated -b flags or, if the remote shell exports it, as a colon-separated
# LAPT_MODEL_DIRS. There is no built-in default: a path that is right for one
# cluster account is wrong for every other one.
#
# Usage:
#   ssh "$LAPT_REMOTE" 'bash -s' < tools/remote_inventory.sh -- -b /path/to/models
#   ssh "$LAPT_REMOTE" 'bash -s' < tools/remote_inventory.sh -- -b /a -b /b -d 14
#   ssh "$LAPT_REMOTE" 'bash -s' < tools/remote_inventory.sh -- -d 14 -f seeded

set -euo pipefail

BASE_DIRS=()
DAYS=7
FILTER=""

while getopts "b:d:f:" opt; do
    case "$opt" in
        b) BASE_DIRS+=("$OPTARG") ;;
        d) DAYS="$OPTARG" ;;
        f) FILTER="$OPTARG" ;;
        *) echo "Usage: $0 [-b DIR]... [-d DAYS] [-f FILTER]" >&2; exit 1 ;;
    esac
done

# fall back to the remote environment when no -b was given
if [ ${#BASE_DIRS[@]} -eq 0 ] && [ -n "${LAPT_MODEL_DIRS:-}" ]; then
    IFS=':' read -r -a BASE_DIRS <<< "$LAPT_MODEL_DIRS"
fi

if [ ${#BASE_DIRS[@]} -eq 0 ]; then
    echo "No model directories given. Pass -b DIR (repeatable), or export" >&2
    echo "LAPT_MODEL_DIRS as a colon-separated list on the remote host." >&2
    exit 1
fi

dirs=""
for BASE_DIR in "${BASE_DIRS[@]}"; do
    [ -d "$BASE_DIR" ] || continue
    if [ "$DAYS" -eq 0 ]; then
        dirs="$dirs $(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)"
    else
        dirs="$dirs $(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -mtime "-${DAYS}")"
    fi
done

for dir in $dirs; do
    dir_name=$(basename "$dir")

    # apply name filter if given
    if [ -n "$FILTER" ] && [[ "$dir_name" != *"$FILTER"* ]]; then
        continue
    fi

    config="$dir/training_config.yaml"
    if [ ! -f "$config" ]; then
        continue
    fi

    exp_id=$(grep "^experiment_id:" "$config" | awk '{print $2}') || true
    if [ -z "$exp_id" ]; then
        continue
    fi

    # find trainer_state.json: top-level if finished, latest checkpoint otherwise
    if [ -f "$dir/trainer_state.json" ]; then
        echo -e "${exp_id}\t${dir_name}\t${dir}/trainer_state.json\t${config}"
    else
        latest_ckpt=$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1) || true
        if [ -n "$latest_ckpt" ] && [ -f "$latest_ckpt/trainer_state.json" ]; then
            echo -e "${exp_id}\t${dir_name}\t${latest_ckpt}/trainer_state.json\t${config}"
        fi
    fi
done
