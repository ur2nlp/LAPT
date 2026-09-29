"""Compare remote run inventory against local files to decide what to fetch.

Reads tab-separated lines from stdin:
    experiment_id \t dir_name \t trainer_state_path \t config_path

Checks local trainer_states/ and configs/ directories to determine what
already exists. Prints scp commands for runs that need fetching.

The remote host is an ssh destination (a hostname, or an alias from your
ssh config) given by --remote or the LAPT_REMOTE environment variable. Nothing
is hardcoded: the host and the remote paths belong to whoever runs this.

Usage:
    INV="ssh $LAPT_REMOTE 'bash -s' < tools/remote_inventory.sh -- -b /path/to/models"
    eval "$INV" | python tools/fetch_diff.py --dry-run
    eval "$INV" | python tools/fetch_diff.py | bash
"""

from lapt_core.fetch_diff import main


if __name__ == "__main__":
    # LAPT ids are hand-typed, so v8L and v08L are the same run and must resolve
    # to one record. Pass --no-normalize-ids to turn that off for one invocation.
    main(env_prefix="LAPT", normalize_ids=True, epilog=__doc__)
