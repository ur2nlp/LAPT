"""Tests that every source type is reachable the way a config reaches it."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IMPORT_THE_PACKAGE = """
import json
from lapt.sources import SOURCE_TYPES
print(json.dumps(SOURCE_TYPES.known_types()))
"""

IMPORT_EVERY_MODULE = """
import importlib, json, pkgutil
import lapt.sources
from lapt.sources import SOURCE_TYPES
for module in pkgutil.iter_modules(lapt.sources.__path__):
    importlib.import_module(f'lapt.sources.{module.name}')
print(json.dumps(SOURCE_TYPES.known_types()))
"""


def _registered_types(script: str) -> list[str]:
    """Run `script` in a clean interpreter and return the types it registered.

    A subprocess is not incidental here. `SOURCE_TYPES` is process-global
    mutable state and pytest runs everything in one process, so a sibling test
    importing `lapt.sources.oscar` directly would register that type as a side
    effect -- masking exactly the omission these tests exist to catch. Only a
    fresh interpreter answers the question honestly.
    """
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=REPO_ROOT,
        env={**os.environ, 'PYTHONPATH': str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestEveryTypeModuleIsWiredIn:
    """Registration is an import side effect, so a module absent from
    `lapt/sources/__init__.py` is invisible to config dispatch -- while its own
    test file still passes, because importing the module registers it.

    That is not hypothetical. `plaintext_dir` was a documented type with a
    working implementation that no config could reach for several commits,
    because it was never registered; nothing caught it, since the
    implementation stayed reachable from its own tests.
    """

    def test_importing_the_package_registers_every_type_module(self):
        from_package = _registered_types(IMPORT_THE_PACKAGE)
        from_every_module = _registered_types(IMPORT_EVERY_MODULE)

        missing = sorted(set(from_every_module) - set(from_package))
        assert not missing, (
            f"{missing} define a type_name but are not imported by "
            f"lapt/sources/__init__.py, so no config can reach them"
        )

    def test_the_registry_is_not_empty(self):
        """Guards the comparison above from passing by both sides being empty."""
        assert len(_registered_types(IMPORT_THE_PACKAGE)) >= 8

    def test_the_types_configs_reference_are_present(self):
        """A few names pinned explicitly, since configs and docs use them."""
        registered = _registered_types(IMPORT_THE_PACKAGE)

        for type_name in ('plaintext', 'plaintext_dir', 'concat', 'multinomial'):
            assert type_name in registered
