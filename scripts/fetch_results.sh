#!/bin/bash
# Fetch trainer states and configs for this project's runs into outputs/.
#
# The tools this calls are framework-general and carry no site configuration,
# so the cluster details come from the environment. Set these in your shell
# profile:
#
#   export LAPT_REMOTE=<ssh host or alias>
#   export LAPT_MODEL_DIRS=<remote model dir>[:<remote model dir>...]
#
# LAPT_MODEL_DIRS is passed through to the remote side explicitly, because
# remote_inventory.sh runs there via `bash -s` and does not inherit this shell.

set -euo pipefail

: "${LAPT_REMOTE:?set LAPT_REMOTE to the ssh host holding the runs}"
: "${LAPT_MODEL_DIRS:?set LAPT_MODEL_DIRS to a colon-separated list of remote model dirs}"

inventory_args=()
while IFS= read -r -d ':' dir; do
    inventory_args+=(-b "$dir")
done <<< "${LAPT_MODEL_DIRS}:"

ssh "$LAPT_REMOTE" 'bash -s' < tools/remote_inventory.sh -- "${inventory_args[@]}" \
    | python tools/fetch_diff.py \
    | bash
