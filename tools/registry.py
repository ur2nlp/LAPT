r"""Experiment registry for tracking run parameters and annotations.

Maintains a lightweight YAML registry (outputs/registry.yaml) that maps
experiment IDs to their key hyperparameters and human-written annotations.
Resolved training configs are stored in outputs/configs/{run_id}.yaml and
trainer states in outputs/trainer_states/{run_id}.json.

Subcommands:
    extract   - Parse training_config.yaml → extract key params → upsert into registry
    show      - Display registry entries
    diff      - Show only parameters that vary between selected runs
    verify    - Check registry params match local config files in outputs/configs/
    annotate  - Set era/group/note/observation for a run
    debt      - Report annotation debt (un-registered runs, empty note/observation)
    sort      - Rewrite registry.yaml ordered by experiment ID or timestamp

Usage:
    # Extract from local file
    python tools/registry.py extract models/v81/training_config.yaml

    # Extract every config matching a regex (re.search against file paths)
    python tools/registry.py extract --pattern 'outputs/configs/v139-i[0-9a-z]\.yaml'

    # Bulk extract from stdin (multiple YAML docs separated by ---)
    ssh "$LAPT_REMOTE" '...' | python tools/registry.py extract --stdin --multi

    # Show runs in a group
    python tools/registry.py show --group dropout-sweep

    # Show all runs ordered by when they were extracted
    python tools/registry.py show --sort timestamp

    # Permanently reorder the registry file by extraction timestamp
    python tools/registry.py sort --by timestamp

    # Show runs hiding specific params (in addition to defaults)
    python tools/registry.py show v29L v30L --hide save_total_limit

    # What differs between these three runs?
    python tools/registry.py diff v81 v82 v83

    # Quick annotation
    python tools/registry.py annotate v89 --note "2x seed multiplier test"
"""

from lapt_core.registry import RegistrySchema, main


LAPT_SCHEMA = RegistrySchema(
    env_prefix="LAPT",
    # sections of the config to flatten into params, with optional prefix stripping
    extract_sections=(
        ("training", "training"),
        ("dataset", "dataset"),
        ("focus", "focus"),
    ),
    # non-scalar, path-like, or redundant with something already recorded
    skip_keys=frozenset({
        "name",
        "sources",
        "path",
        "cache_dir",
        "config",
        "tokenizer_path",
        "model_freeze_prefix",
        "metric_for_best_model",
        "optim",
        "lr_scheduler_type",
        "type",
        "language",
        "split",
        "format",
        "external_eval_sets",
    }),
    # well-known params first; everything else sorts alphabetically after
    preferred_order=(
        "hf_model",
        "tokenizer",
        "lr",
        "effective_batch",
        "dropout",
        "weight_decay",
        "alpha",
        "total_samples",
        "max_steps",
        "focus_enabled",
        "vocab_size",
        "seed_lambda",
        "seed_vocab_multiplier",
    ),
)


if __name__ == "__main__":
    main(LAPT_SCHEMA, epilog=__doc__)
