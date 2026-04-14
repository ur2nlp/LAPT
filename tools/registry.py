"""Experiment registry for tracking run parameters and annotations.

Maintains a lightweight YAML registry (outputs/registry.yaml) that maps
experiment IDs to their key hyperparameters and human-written annotations.
Resolved training configs are stored in outputs/configs/{run_id}.yaml and
trainer states in outputs/trainer_states/{run_id}.json.

Subcommands:
    extract   - Parse training_config.yaml → extract key params → upsert into registry
    show      - Display registry entries, optionally joined with metrics
    diff      - Show only parameters that vary between selected runs
    verify    - Check registry params match local config files in outputs/configs/
    annotate  - Set era/group/note/observation for a run

Usage:
    # Extract from local file
    python tools/registry.py extract models/v81/training_config.yaml

    # Bulk extract from stdin (multiple YAML docs separated by ---)
    ssh circ '...' | python tools/registry.py extract --stdin --multi

    # Show runs in a group with metrics
    python tools/registry.py show --group dropout-sweep --metrics

    # Show runs hiding specific params (in addition to defaults)
    python tools/registry.py show v29L v30L --hide save_total_limit

    # What differs between these three runs?
    python tools/registry.py diff v81 v82 v83

    # Quick annotation
    python tools/registry.py annotate v89 --note "2x seed multiplier test"
"""

import argparse
import sys
from pathlib import Path

import yaml


REGISTRY_PATH = Path("outputs/registry.yaml")

# params hidden from display by default (usually redundant with max_steps)
DEFAULT_HIDDEN_PARAMS = {"eval_steps", "logging_steps", "save_steps"}

# preferred display order for well-known params (others sorted alphabetically after)
PREFERRED_ORDER = [
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
]

# sections of the config to flatten into params, with optional prefix stripping
_EXTRACT_SECTIONS = [
    ("training", "training"),
    ("dataset", "dataset"),
    ("focus", "focus"),
]

# keys to skip during extraction (non-scalar, path-like, or redundant)
_SKIP_KEYS = {
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
}

HUMAN_FIELDS = ["era", "group", "note", "observation", "status"]


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    """Load the registry YAML file, returning an empty dict if missing."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def save_registry(registry: dict, path: Path = REGISTRY_PATH) -> None:
    """Write the registry to YAML, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            registry,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _is_scalar(value: object) -> bool:
    """Check if a value is a scalar suitable for registry storage."""
    return isinstance(value, (int, float, bool, str)) and not isinstance(value, dict)


def _extract_source_sampling_params(sources: list) -> dict:
    """Extract per-source sampling_prob and upsampling_factor from dataset sources list.

    For each source with an `id`, produces keys like `sampling_prob_got` and
    `upsampling_factor_got-eng` so these can be tracked and diffed across runs.

    Args:
        sources: The `dataset.sources` list from a training config.

    Returns:
        Dict of flattened per-source sampling parameters.
    """
    params = {}
    if not isinstance(sources, list):
        return params
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = source.get("id")
        if not source_id:
            continue
        for key in ("sampling_prob", "upsampling_factor"):
            if key in source and _is_scalar(source[key]):
                params[f"{key}_{source_id}"] = source[key]
    return params


def extract_params(config: dict) -> tuple[str, dict]:
    """Extract all scalar parameters from a training_config.yaml dict.

    Flattens training/dataset/focus sections into a single dict. Keys that
    are paths, non-scalar, or redundant are skipped. A few synthetic params
    are computed (effective_batch, lr as alias for learning_rate, focus_enabled
    as alias for focus.enabled).

    Args:
        config: Parsed YAML config dictionary.

    Returns:
        Tuple of (experiment_id, params_dict).

    Raises:
        ValueError: If experiment_id is missing from config.
    """
    experiment_id = config.get("experiment_id")
    if not experiment_id:
        raise ValueError("Config missing 'experiment_id' field")
    experiment_id = str(experiment_id)

    params = {}

    # base model lives at the top level of the training config
    hf_model = config.get("hf_model")
    if isinstance(hf_model, str):
        params["hf_model"] = hf_model

    # record only the basename of the provided tokenizer to keep displays compact
    tokenizer_path = config.get("focus", {}).get("tokenizer_path")
    if isinstance(tokenizer_path, str) and tokenizer_path:
        params["tokenizer"] = Path(tokenizer_path).name

    for section_name, _ in _EXTRACT_SECTIONS:
        section = config.get(section_name, {})
        for key, value in section.items():
            if key in _SKIP_KEYS:
                continue
            if not _is_scalar(value):
                continue
            # resolve Hydra interpolations stored as strings like "${divide:...}"
            if isinstance(value, str) and value.startswith("${"):
                continue
            params[key] = value

    # extract per-source sampling params from dataset.sources
    dataset_section = config.get("dataset", {})
    sources = dataset_section.get("sources", [])
    params.update(_extract_source_sampling_params(sources))

    # compute synthetic params
    training = config.get("training", {})
    focus = config.get("focus", {})

    batch_size = training.get("train_batch_size", 1)
    grad_accum = training.get("gradient_accumulation_steps", 1)
    params["effective_batch"] = batch_size * grad_accum

    # create short aliases for commonly referenced params
    if "learning_rate" in params:
        params["lr"] = params.pop("learning_rate")
    if "enabled" in params:
        params["focus_enabled"] = params.pop("enabled")

    return experiment_id, params


def upsert_entry(registry: dict, experiment_id: str, params: dict) -> None:
    """Insert or update a registry entry's params without touching human fields.

    Args:
        registry: The full registry dict (modified in place).
        experiment_id: Run identifier (e.g. "v81").
        params: Extracted parameter dict.
    """
    if experiment_id not in registry:
        registry[experiment_id] = {
            "params": params,
            "era": "",
            "group": "",
            "note": "",
            "observation": "",
        }
    else:
        registry[experiment_id]["params"] = params


def diff_runs(registry: dict, run_ids: list[str]) -> tuple[dict[str, list], dict[str, object]]:
    """Identify parameters that vary vs. stay constant across runs.

    Args:
        registry: The full registry dict.
        run_ids: List of experiment IDs to compare.

    Returns:
        Tuple of (varying_params, constant_params) where:
        - varying_params maps param name → list of values (one per run)
        - constant_params maps param name → the shared value
    """
    all_params = {}
    for run_id in run_ids:
        entry = registry.get(run_id, {})
        all_params[run_id] = entry.get("params", {})

    # collect all param keys across runs, ordered by preference then alphabetically
    seen_keys = set()
    for rid in run_ids:
        seen_keys.update(all_params[rid].keys())

    preferred_set = set(PREFERRED_ORDER)
    all_keys = [k for k in PREFERRED_ORDER if k in seen_keys]
    all_keys += sorted(k for k in seen_keys if k not in preferred_set)

    varying = {}
    constant = {}

    for key in all_keys:
        values = [all_params[rid].get(key) for rid in run_ids]
        unique = set(str(v) for v in values)
        if len(unique) > 1:
            varying[key] = values
        else:
            constant[key] = values[0]

    return varying, constant


def format_param_value(value: object) -> str:
    """Format a parameter value for display."""
    if isinstance(value, float):
        if value < 0.001:
            return f"{value:.1e}"
        return f"{value:g}"
    if isinstance(value, int):
        if value >= 1_000_000 and value % 1_000_000 == 0:
            return f"{value // 1_000_000}M"
        if value >= 1_000 and value % 1_000 == 0:
            return f"{value // 1_000}k"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def cmd_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand."""
    registry = load_registry()
    configs_to_process = []

    if args.stdin:
        raw = sys.stdin.read()
        if args.multi:
            # split on YAML document separator
            docs = raw.split("\n---\n")
            # also handle --- at start of stream
            if docs and docs[0].strip() == "---":
                docs = docs[1:]
            elif docs and docs[0].startswith("---\n"):
                docs[0] = docs[0][4:]
        else:
            docs = [raw]

        for doc in docs:
            doc = doc.strip()
            if not doc:
                continue
            try:
                config = yaml.safe_load(doc)
                if config:
                    configs_to_process.append(config)
            except yaml.YAMLError as e:
                print(f"Warning: failed to parse YAML doc: {e}", file=sys.stderr)
    else:
        for filepath in args.files:
            path = Path(filepath)
            if not path.exists():
                print(f"Warning: {path} not found, skipping", file=sys.stderr)
                continue
            try:
                with open(path) as f:
                    config = yaml.safe_load(f)
                if config:
                    configs_to_process.append(config)
            except yaml.YAMLError as e:
                print(f"Warning: failed to parse {path}: {e}", file=sys.stderr)

    extracted_count = 0
    for config in configs_to_process:
        try:
            experiment_id, params = extract_params(config)
            upsert_entry(registry, experiment_id, params)
            extracted_count += 1
            print(f"Extracted {experiment_id}", file=sys.stderr)
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)

    if extracted_count > 0:
        save_registry(registry)
        print(
            f"Updated {REGISTRY_PATH} ({extracted_count} run(s))",
            file=sys.stderr,
        )
    else:
        print("No configs extracted.", file=sys.stderr)


def cmd_show(args: argparse.Namespace) -> None:
    """Handle the 'show' subcommand."""
    registry = load_registry()
    if not registry:
        print("Registry is empty.", file=sys.stderr)
        return

    # filter to requested runs
    if args.runs:
        run_ids = [rid for rid in args.runs if rid in registry]
        missing = [rid for rid in args.runs if rid not in registry]
        if missing:
            print(f"Warning: not in registry: {', '.join(missing)}", file=sys.stderr)
    else:
        run_ids = list(registry.keys())

    # filter by era or group
    if args.era:
        run_ids = [rid for rid in run_ids if registry[rid].get("era") == args.era]
    if args.group:
        run_ids = [rid for rid in run_ids if registry[rid].get("group") == args.group]

    if not run_ids:
        print("No matching runs found.", file=sys.stderr)
        return

    # sort by experiment id (natural sort: v1, v2, ..., v10, v81)
    run_ids.sort(key=_natural_sort_key)

    # load metrics if requested
    metrics = {}
    if args.metrics:
        try:
            from tools.summarize_results import parse_trainer_state
        except ImportError:
            # try relative import for direct script execution
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from tools.summarize_results import parse_trainer_state

        outputs_dir = Path("outputs/trainer_states")
        for rid in run_ids:
            json_path = outputs_dir / f"{rid}.json"
            if json_path.exists():
                try:
                    summary = parse_trainer_state(json_path)
                    metrics[rid] = summary
                except Exception as e:
                    print(
                        f"Warning: failed to parse metrics for {rid}: {e}",
                        file=sys.stderr,
                    )

    # display
    for rid in run_ids:
        entry = registry[rid]
        params = entry.get("params", {})
        print(f"--- {rid} ---")

        # params — preferred order first, then remaining alphabetically
        hidden = DEFAULT_HIDDEN_PARAMS | set(args.hide)
        preferred_set = set(PREFERRED_ORDER)
        ordered_keys = [k for k in PREFERRED_ORDER if k in params and k not in hidden]
        ordered_keys += sorted(k for k in params if k not in preferred_set and k not in hidden)
        for key in ordered_keys:
            print(f"  {key}: {format_param_value(params[key])}")

        # human fields
        for field_name in HUMAN_FIELDS:
            value = entry.get(field_name, "")
            if value:
                print(f"  {field_name}: {value}")

        # metrics
        if rid in metrics:
            summary = metrics[rid]
            step_str = f"{summary.global_step}/{summary.max_steps}"
            print(f"  status: {summary.status} ({step_str})")
            if summary.best_eval:
                best_parts = []
                for lang, (loss, step) in sorted(summary.best_eval.items()):
                    best_parts.append(f"{lang}={loss:.4f}@{step}")
                print(f"  best_eval: {', '.join(best_parts)}")

        print()


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the 'diff' subcommand."""
    registry = load_registry()

    run_ids = args.runs
    missing = [rid for rid in run_ids if rid not in registry]
    if missing:
        print(f"Error: not in registry: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    if len(run_ids) < 2:
        print("Only 1 run specified, nothing to diff.", file=sys.stderr)
        return

    varying, constant = diff_runs(registry, run_ids)

    hidden = DEFAULT_HIDDEN_PARAMS | set(args.hide)
    varying = {k: v for k, v in varying.items() if k not in hidden}
    constant = {k: v for k, v in constant.items() if k not in hidden}

    if not varying:
        print(f"All key params are identical across {', '.join(run_ids)}.")
        if constant:
            parts = [f"{k}={format_param_value(v)}" for k, v in constant.items()]
            print(f"Values: {', '.join(parts)}")
        return

    # print table of varying params
    print(f"Params that vary across {', '.join(run_ids)}:")

    # compute column widths
    run_width = max(len(rid) for rid in run_ids)
    run_width = max(run_width, 3)
    col_widths = {}
    for key, values in varying.items():
        formatted = [format_param_value(v) for v in values]
        col_widths[key] = max(len(key), max(len(f) for f in formatted))

    # header
    header_parts = [f"{'Run':<{run_width}}"]
    for key in varying:
        header_parts.append(f"{key:>{col_widths[key]}}")
    print("  ".join(header_parts))

    # rows
    for i, rid in enumerate(run_ids):
        row_parts = [f"{rid:<{run_width}}"]
        for key, values in varying.items():
            formatted = format_param_value(values[i])
            row_parts.append(f"{formatted:>{col_widths[key]}}")
        print("  ".join(row_parts))

    # constant params
    if constant:
        parts = [f"{k}={format_param_value(v)}" for k, v in constant.items()]
        print(f"\nConstant: {', '.join(parts)}")


def cmd_annotate(args: argparse.Namespace) -> None:
    """Handle the 'annotate' subcommand."""
    registry = load_registry()
    run_id = args.run

    if run_id not in registry:
        print(f"Error: {run_id} not in registry.", file=sys.stderr)
        sys.exit(1)

    updated = False
    for field_name in HUMAN_FIELDS:
        value = getattr(args, field_name, None)
        if value is not None:
            registry[run_id][field_name] = value
            updated = True

    if updated:
        save_registry(registry)
        print(f"Updated annotations for {run_id}", file=sys.stderr)
    else:
        print("No annotations specified. Use --era, --group, --note, or --observation.")


def cmd_verify(args: argparse.Namespace) -> None:
    """Handle the 'verify' subcommand.

    For each run, looks for a config file at outputs/configs/{run_id}.yaml,
    re-extracts params using the same logic as 'extract', and diffs against
    what is stored in the registry.
    """
    registry = load_registry()

    if args.runs:
        run_ids = list(args.runs)
        missing_from_registry = [rid for rid in run_ids if rid not in registry]
        if missing_from_registry:
            print(
                f"Error: not in registry: {', '.join(missing_from_registry)}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        run_ids = sorted(registry.keys(), key=_natural_sort_key)

    ok_count = 0
    missing_count = 0
    mismatch_count = 0

    for run_id in run_ids:
        config_path = Path("outputs/configs") / f"{run_id}.yaml"

        if not config_path.exists():
            print(f"{run_id}: MISSING ({config_path})")
            missing_count += 1
            continue

        try:
            with open(config_path) as config_file:
                config = yaml.safe_load(config_file)
            _, extracted_params = extract_params(config)
        except yaml.YAMLError as exc:
            print(f"{run_id}: ERROR (invalid YAML in {config_path}: {exc})")
            mismatch_count += 1
            continue
        except ValueError as exc:
            print(f"{run_id}: ERROR ({exc})")
            mismatch_count += 1
            continue

        registry_params = registry[run_id].get("params", {})

        all_keys = sorted(set(extracted_params) | set(registry_params))
        differences = []
        for key in all_keys:
            in_extracted = key in extracted_params
            in_registry = key in registry_params
            if in_extracted and in_registry:
                if extracted_params[key] != registry_params[key]:
                    differences.append((
                        key,
                        format_param_value(registry_params[key]),
                        format_param_value(extracted_params[key]),
                    ))
            elif in_registry:
                differences.append((key, format_param_value(registry_params[key]), "<missing>"))
            else:
                differences.append((key, "<missing>", format_param_value(extracted_params[key])))

        if differences:
            print(f"{run_id}: MISMATCH ({len(differences)} param(s) differ)")
            key_width = max(len(key) for key, _, _ in differences)
            for key, registry_val, config_val in differences:
                print(f"  {key:<{key_width}}  registry={registry_val!r}  config={config_val!r}")
            mismatch_count += 1
        else:
            print(f"{run_id}: OK")
            ok_count += 1

    total = ok_count + missing_count + mismatch_count
    print(
        f"\nSummary ({total} run(s)): {ok_count} OK, "
        f"{missing_count} missing config, {mismatch_count} mismatch(es)"
    )


def _natural_sort_key(run_id: str) -> tuple:
    """Sort key that handles 'v81' style IDs numerically."""
    import re
    parts = re.split(r"(\d+)", run_id)
    result = []
    for part in parts:
        if part.isdigit():
            result.append((1, int(part)))
        else:
            result.append((0, part))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment registry management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract
    extract_parser = subparsers.add_parser(
        "extract",
        help="Parse training_config.yaml and upsert into registry",
    )
    extract_parser.add_argument(
        "files",
        nargs="*",
        help="Paths to training_config.yaml files",
    )
    extract_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read YAML from stdin instead of files",
    )
    extract_parser.add_argument(
        "--multi",
        action="store_true",
        help="Parse multiple YAML documents separated by ---",
    )

    # show
    show_parser = subparsers.add_parser(
        "show",
        help="Display registry entries",
    )
    show_parser.add_argument(
        "runs",
        nargs="*",
        help="Specific run IDs to show (default: all)",
    )
    show_parser.add_argument(
        "--era",
        help="Filter by era",
    )
    show_parser.add_argument(
        "--group",
        help="Filter by group",
    )
    show_parser.add_argument(
        "--metrics",
        action="store_true",
        help="Join with metrics from outputs/trainer_states/*.json",
    )
    show_parser.add_argument(
        "--hide",
        nargs="*",
        default=[],
        metavar="PARAM",
        help=(
            "Additional params to hide from display "
            f"(always hides: {sorted(DEFAULT_HIDDEN_PARAMS)})"
        ),
    )

    # diff
    diff_parser = subparsers.add_parser(
        "diff",
        help="Show parameters that vary between runs",
    )
    diff_parser.add_argument(
        "runs",
        nargs="+",
        help="Run IDs to compare",
    )
    diff_parser.add_argument(
        "--hide",
        nargs="*",
        default=[],
        metavar="PARAM",
        help=(
            "Additional params to hide from display "
            f"(always hides: {sorted(DEFAULT_HIDDEN_PARAMS)})"
        ),
    )

    # verify
    verify_parser = subparsers.add_parser(
        "verify",
        help="Check registry params match local config files in outputs/configs/",
    )
    verify_parser.add_argument(
        "runs",
        nargs="*",
        help="Specific run IDs to verify (default: all)",
    )

    # annotate
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Set era/group/note/observation for a run",
    )
    annotate_parser.add_argument(
        "run",
        help="Run ID to annotate",
    )
    annotate_parser.add_argument("--era", help="Set era tag")
    annotate_parser.add_argument("--group", help="Set group tag")
    annotate_parser.add_argument("--note", help="Set note")
    annotate_parser.add_argument("--observation", help="Set observation")
    annotate_parser.add_argument(
        "--status",
        help="Set run status (e.g. 'manually_closed' to prevent re-fetching)",
    )

    args = parser.parse_args()

    commands = {
        "extract": cmd_extract,
        "show": cmd_show,
        "diff": cmd_diff,
        "verify": cmd_verify,
        "annotate": cmd_annotate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
