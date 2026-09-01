"""A `CachedArtifact` specialization for corpora stored as HuggingFace datasets.

`lapt.core.artifacts` is deliberately agnostic about what an artifact *is*: it
knows how to decide whether a cached copy is valid and how to rebuild one, but
`write` and `read` are left abstract. Most stages of a corpus pipeline answer
those two the same way — serialize a `DatasetDict` with `save_to_disk`, restore
it with `load_from_disk` — so answering them once here removes the boilerplate
from every concrete loader.

What belongs here versus in a project's own package is not "does it have a heavy
dependency" but "does the concept survive a change of domain". Storing a
`DatasetDict` under a tracked config record is domain-neutral: a speech corpus
and a text corpus are cached identically. The loaders that produce them are not,
so concrete subclasses live with the project that defines their sources.

Note that this module imports `datasets`, which is roughly twenty times more
expensive to import than `lapt.core.artifacts` itself. That is why
`lapt/core/__init__.py` re-exports nothing: importing this module must not be a
side effect of reaching for the caching primitives. Import what you need
directly.

Deliberately absent: any dependency on `omegaconf`. `config()` returns a plain
dict, and unwrapping a `DictConfig` (or a dataclass, or an argparse namespace)
is the concrete subclass's job. A base that reached into `DictConfig` would only
serve projects configured with Hydra in one particular idiom.
"""

from typing import Any

from datasets import DatasetDict, load_from_disk

from lapt.core.artifacts import CachedArtifact


class DatasetArtifact(CachedArtifact):
    """A cached corpus stage whose value is a `DatasetDict` on disk.

    Subclasses supply `config()` — the tracked parameters, which double as the
    cache-validation record — and `build()`, which returns the dataset. The
    round trip to disk is handled here.

    Attributes:
        name: Directory name for the artifact, defaulting to the stage this
            class was written for.
        type_name: Key this class is registered under in a `DatasetRegistry`,
            matching the `type` field of a source configuration. Left None on
            intermediate base classes that should not be instantiable by name.
    """

    name = "untokenized"
    type_name: str | None = None

    def write(self, value: DatasetDict, path: str) -> None:
        """Serialize the dataset into the artifact directory.

        Args:
            value: The dataset returned by `build`.
            path: Directory to write into; already created by `resolve`.
        """
        value.save_to_disk(path)

    def read(self, path: str) -> DatasetDict:
        """Restore a previously written dataset.

        Args:
            path: Directory a prior `write` populated.

        Returns:
            The cached dataset.
        """
        return load_from_disk(path)


class DatasetRegistry:
    """Maps the `type` field of a source configuration to a `DatasetArtifact`.

    This replaces the `if/elif` dispatch chain that a loader module otherwise
    grows, so adding a source type is a new class rather than an edit to a
    branch in the middle of an existing function.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._types: dict[str, type[DatasetArtifact]] = {}

    def register(self, artifact_class: type[DatasetArtifact]) -> type[DatasetArtifact]:
        """Register a dataset class under its `type_name`.

        Usable as a decorator, since the class is returned unchanged.

        Args:
            artifact_class: The class to register. Its `type_name` must be set
                and not already taken.

        Returns:
            `artifact_class`, unchanged.

        Raises:
            ValueError: If `type_name` is unset or already registered.
        """
        type_name = artifact_class.type_name
        if not type_name:
            raise ValueError(
                f"{artifact_class.__name__} cannot be registered without a type_name"
            )
        if type_name in self._types:
            existing = self._types[type_name].__name__
            raise ValueError(
                f"Dataset type '{type_name}' is already registered to {existing}"
            )
        self._types[type_name] = artifact_class
        return artifact_class

    def __contains__(self, type_name: str) -> bool:
        """Whether a class is registered for `type_name`."""
        return type_name in self._types

    def get(self, type_name: str) -> type[DatasetArtifact]:
        """Look up the class registered for a source type.

        Args:
            type_name: The `type` field of a source configuration.

        Returns:
            The registered class.

        Raises:
            ValueError: If nothing is registered under that name. The message
                lists the known types, since this usually means a typo.
        """
        if type_name not in self._types:
            known = ", ".join(sorted(self._types)) or "(none registered)"
            raise ValueError(
                f"Unsupported dataset type: {type_name}. Known types: {known}"
            )
        return self._types[type_name]

    def create(self, type_name: str, *args: Any, **kwargs: Any) -> DatasetArtifact:
        """Instantiate the class registered for a source type.

        Args:
            type_name: The `type` field of a source configuration.
            *args: Positional arguments for the class.
            **kwargs: Keyword arguments for the class.

        Returns:
            The constructed artifact.
        """
        return self.get(type_name)(*args, **kwargs)

    def known_types(self) -> list[str]:
        """Return the registered type names, sorted."""
        return sorted(self._types)
