#!/usr/bin/env bash
# Inventory training runs on the CIRC cluster.
#
# Outputs TSV to stdout: experiment_id \t dir_name \t trainer_state_path \t config_path
# Intended to be piped into tools/fetch_diff.py.
#
# Usage:
#   ssh circ 'bash -s' < tools/remote_inventory.sh
#   ssh circ 'bash -s' < tools/remote_inventory.sh -- -d 14
#   ssh circ 'bash -s' < tools/remote_inventory.sh -- -d 14 -f seeded

set -euo pipefail

BASE_DIR="/scratch/cdowney4/LAPT/models/old_germanic"
DAYS=7
FILTER=""

while getopts "d:f:" opt; do
    case "$opt" in
        d) DAYS="$OPTARG" ;;
        f) FILTER="$OPTARG" ;;
        *) echo "Usage: $0 [-d DAYS] [-f FILTER]" >&2; exit 1 ;;
    esac
done

if [ "$DAYS" -eq 0 ]; then
    dirs=$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
else
    dirs=$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -mtime "-${DAYS}")
fi

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
