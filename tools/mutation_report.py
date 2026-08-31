"""Summarize a mutmut run as a line-oriented report.

`mutmut results` lists one entry per mutant, named after the function it lives
in (`dataset_utils.x__apply_substitutions__mutmut_2`). That is the wrong unit
for reading: mutmut generates several mutants per source line, so a function
with 25 surviving mutants may represent only 6 distinct untested lines, and a
per-function count reads as far more alarming than the code warrants.

This regroups the same data by source line, so each entry is one place in the
file rather than one generated variant, and prints `path:line` for editors that
make those clickable.

Run `mutmut run` first; this reads the state it leaves in `mutants/`.
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        default="survived",
        help="mutmut status to report (survived, killed, 'no tests', timeout, ...)",
    )
    parser.add_argument(
        "--show-mutations",
        action="store_true",
        help="print the replacement text of every mutant on each reported line",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="report only the N lines with the most mutants (0 = all)",
    )
    return parser.parse_args()


def build_function_start_lines(source_path: Path) -> dict[tuple[str | None, str], int]:
    """Map (class_name, function_name) to the first line of that function's source.

    mutmut renders a mutant diff from the function alone, so diff offsets are
    relative to the function's first line. Decorators count: mutmut rebuilds the
    function through libcst, which keeps them attached, while ast reports the
    `def` line, so take the earliest decorator line where there is one.

    Args:
        source_path: Path to the unmutated source file.

    Returns:
        Mapping from (class name or None, function name) to a 1-based line number.
    """
    tree = ast.parse(source_path.read_text())
    start_lines: dict[tuple[str | None, str], int] = {}

    def visit(node: ast.AST, class_name: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_line = child.lineno
                for decorator in child.decorator_list:
                    first_line = min(first_line, decorator.lineno)
                start_lines[(class_name, child.name)] = first_line
                visit(child, class_name)

    visit(tree, None)
    return start_lines


def extract_changes(diff_text: str) -> list[tuple[int, str, str]]:
    """Pull (function-relative line, removed text, added text) out of a unified diff.

    Args:
        diff_text: Unified diff of one function, original versus mutant.

    Returns:
        One tuple per removed line, in file order.
    """
    changes: list[tuple[int, str, str]] = []
    original_line = 0
    pending_removals: list[tuple[int, str]] = []

    for line in diff_text.split("\n"):
        if line.startswith("@@"):
            # "@@ -12,7 +12,7 @@" -- take the start of the original-side range
            original_range = line.split(" ")[1]
            original_line = int(original_range[1:].split(",")[0])
            pending_removals = []
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            pending_removals.append((original_line, line[1:]))
            original_line += 1
        elif line.startswith("+"):
            if pending_removals:
                removal_line, removed_text = pending_removals.pop(0)
                changes.append((removal_line, removed_text, line[1:]))
        elif line.startswith(" "):
            original_line += 1

    return changes


def main() -> int:
    arguments = parse_arguments()

    if not Path("mutants").is_dir():
        print("No mutants/ directory. Run `mutmut run` first.", file=sys.stderr)
        return 1

    from mutmut.__main__ import (
        Config,
        get_diff_for_mutant,
        orig_function_and_class_names_from_key,
        status_by_exit_code,
        walk_mutatable_files,
    )
    from mutmut.mutation.data import SourceFileMutationData

    Config.ensure_loaded()

    # mutants_by_location maps "path:line" to the mutations seen on that line
    mutants_by_location: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    unmapped_count = 0
    total_matching = 0

    for path in walk_mutatable_files():
        if not (Path("mutants") / f"{path}.meta").exists():
            continue

        mutation_data = SourceFileMutationData(path=path)
        mutation_data.load()
        if not mutation_data.exit_code_by_key:
            continue

        source_path = Path(path)
        if not source_path.exists():
            continue
        source_lines = source_path.read_text().split("\n")
        function_start_lines = build_function_start_lines(source_path)

        for mutant_name, exit_code in mutation_data.exit_code_by_key.items():
            if status_by_exit_code[exit_code] != arguments.status:
                continue
            total_matching += 1

            function_name, class_name = orig_function_and_class_names_from_key(mutant_name)
            function_start = function_start_lines.get((class_name, function_name))
            if function_start is None:
                unmapped_count += 1
                continue

            diff_text = get_diff_for_mutant(mutant_name, path=path)
            for relative_line, removed_text, added_text in extract_changes(diff_text):
                absolute_line = function_start + relative_line - 1

                # The diff is rebuilt by libcst and may not be byte-identical to
                # the file, so confirm the mapping before trusting the number.
                index = absolute_line - 1
                if not (0 <= index < len(source_lines)):
                    unmapped_count += 1
                    continue
                if source_lines[index].strip() != removed_text.strip():
                    matches = [
                        candidate
                        for candidate in range(function_start - 1, len(source_lines))
                        if source_lines[candidate].strip() == removed_text.strip()
                    ]
                    if len(matches) != 1:
                        unmapped_count += 1
                        continue
                    absolute_line = matches[0] + 1

                mutants_by_location[(str(path), absolute_line)].append(
                    (removed_text.strip(), added_text.strip())
                )

    if not mutants_by_location:
        print(f"No mutants with status {arguments.status!r}.")
        return 0

    locations = sorted(
        mutants_by_location.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )
    if arguments.limit:
        locations = locations[: arguments.limit]

    distinct_lines = len(mutants_by_location)
    print(f"{total_matching} {arguments.status} mutants on {distinct_lines} distinct lines")
    if unmapped_count:
        print(f"({unmapped_count} could not be mapped to a line and are omitted)")
    print()

    for (path, line_number), mutations in locations:
        original_text = mutations[0][0]
        print(f"{path}:{line_number}  ({len(mutations)} mutants)")
        print(f"    {original_text}")
        if arguments.show_mutations:
            for _, added_text in mutations:
                print(f"      -> {added_text}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
