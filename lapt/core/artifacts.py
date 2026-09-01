"""Content-tracked caching for multi-stage training pipelines.

A training pipeline is a chain of expensive, deterministic stages: download and
normalize a corpus, train a tokenizer, tokenize the corpus, adapt a model. Each
stage is worth caching, and each cache is only valid for the configuration that
produced it.

The idiom this module replaces had three parts living in three different places:
the code that decided *where* an artifact lived, the code that checked *whether*
it was still valid, and the code that *built* it. Nothing bound them together, so
a path digest and the config it was supposed to describe could drift apart, and
every pipeline stage re-implemented the same "does it exist / is it stale / build
or load" block by hand.

`CachedArtifact` binds all three to one declaration. A subclass supplies its
config, how to build it, and how to read and write it; the base class derives the
path, validates the cached config, and decides between loading and rebuilding.
`ArtifactGraph` then wires stages together so that invalidating one stage
invalidates everything downstream of it, rather than requiring a hand-maintained
cascade.

Typical use::

    class Corpus(CachedArtifact):
        name = "corpus"

        def __init__(self, root, language):
            super().__init__(root)
            self.language = language

        def config(self):
            return {"language": self.language}

        def build(self, deps):
            return download_corpus(self.language)

        def write(self, value, path):
            value.save_to_disk(path)

        def read(self, path):
            return DatasetDict.load_from_disk(path)

    graph = ArtifactGraph(Corpus("data/got", "gothic"), Tokenized("data/got"))
    dataset = graph.get("tokenized")

This module depends only on the standard library and PyYAML. It must stay that
way: keeping it free of torch, transformers, and omegaconf is what makes it cheap
to import from a CLI helper and straightforward to unit-test.
"""

import hashlib
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import yaml

CONFIG_FILENAME = "config.yaml"


def dict_diff(cached: dict, current: dict, path: str = "") -> list[str]:
    """Recursively describe the differences between two config dicts.

    Args:
        cached: The configuration read back from a cached artifact.
        current: The configuration the caller is requesting now.
        path: Dotted path to the enclosing key, used when recursing.

    Returns:
        Human-readable difference descriptions, one per differing key. Empty
        when the two dicts are equivalent.
    """
    diffs = []

    only_in_cached = set(cached.keys()) - set(current.keys())
    for key in sorted(only_in_cached):
        full_path = f"{path}.{key}" if path else key
        diffs.append(f"{full_path}: present in cached config but not in current")

    only_in_current = set(current.keys()) - set(cached.keys())
    for key in sorted(only_in_current):
        full_path = f"{path}.{key}" if path else key
        diffs.append(f"{full_path}: present in current config but not in cached")

    for key in sorted(set(cached.keys()) & set(current.keys())):
        cached_value = cached[key]
        current_value = current[key]
        full_path = f"{path}.{key}" if path else key

        if isinstance(cached_value, dict) and isinstance(current_value, dict):
            diffs.extend(dict_diff(cached_value, current_value, full_path))
        elif cached_value != current_value:
            diffs.append(f"{full_path}: {cached_value} (cached) != {current_value} (current)")

    return diffs


def config_digest(payload: Any, length: int = 8) -> str:
    """Hash a JSON-serializable config payload into a short, stable digest.

    Every digest that ends up in a cache path should come from this function.
    Hand-rolled `sha256(json.dumps(...))` calls are how a path and the config it
    claims to describe drift apart: the two get edited at different times, and
    nothing detects that the directory name no longer reflects its contents.

    Keys are sorted so that a config's dict ordering cannot change the digest,
    and non-serializable values fall back to their `str()` form so that a stray
    enum or Path does not raise.

    Args:
        payload: Any JSON-serializable structure. Mappings are canonicalized by
            sorting their keys at every level.
        length: Number of leading hex characters to keep.

    Returns:
        The leading `length` hex characters of the SHA-256 digest.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def format_number(n: int) -> str:
    """Format a count with a k/m suffix for use in directory names.

    Uses integer division so that a given count always produces the same string;
    a decimal form would let rounding changes silently repoint a cache path.

    Args:
        n: The count to format.

    Returns:
        A string like "50k" or "1m", or the plain number below 1000.
    """
    if n >= 1_000_000:
        return f"{n // 1_000_000}m"
    elif n >= 1000:
        return f"{n // 1000}k"
    else:
        return str(n)


class ConfigMismatchError(ValueError):
    """Raised when a cached artifact was built with a different configuration."""


class ArtifactConfig:
    """Base class for the config record saved alongside a cached artifact.

    Subclasses implement `to_dict` and set `artifact_name`. The saved YAML is the
    record of what actually produced the artifact, so a mismatch against the
    current config means the cache is stale, not that the config is wrong.
    """

    artifact_name: str = "Artifact"

    def to_dict(self) -> dict:
        """Return the parameters that determine this artifact's contents."""
        raise NotImplementedError

    def save(self, config_path: str) -> None:
        """Write this config to YAML for later verification.

        Args:
            config_path: Full path of the config file to write. Parent
                directories are created if missing.
        """
        parent = os.path.dirname(config_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(config_path, 'w') as config_file:
            yaml.dump(self.to_dict(), config_file, default_flow_style=False, sort_keys=False)
        print(f"Saved {self.artifact_name} config to {config_path}", file=sys.stderr)

    def check_cached(self, config_path: str, error_on_mismatch: bool = True) -> bool:
        """Verify that the config cached at a path matches this one.

        A missing or empty config file is treated as a match, so that artifacts
        created before config tracking existed are not needlessly invalidated.

        Args:
            config_path: Full path to the cached config file.
            error_on_mismatch: Raise on mismatch when True; print to stderr and
                return False when False.

        Returns:
            True when the configs match or no cached config exists.

        Raises:
            ConfigMismatchError: If the configs differ and `error_on_mismatch`.
        """
        if not os.path.exists(config_path):
            return True

        with open(config_path) as config_file:
            cached_config = yaml.safe_load(config_file)

        if cached_config is None:
            return True

        diffs = dict_diff(cached_config, self.to_dict())
        if not diffs:
            return True

        error_msg = self._format_mismatch(config_path, diffs)
        if error_on_mismatch:
            raise ConfigMismatchError(error_msg)

        print(error_msg, file=sys.stderr)
        return False

    def _format_mismatch(self, config_path: str, diffs: list[str]) -> str:
        """Build the operator-facing message for a config mismatch.

        Args:
            config_path: Path of the cached config that failed to match.
            diffs: Difference descriptions from `dict_diff`.

        Returns:
            A multi-line message naming the differences and the ways out.
        """
        divider = '=' * 70
        return (
            f"\n{divider}\n"
            f"CONFIG MISMATCH: {self.artifact_name}\n"
            f"{divider}\n"
            f"The cached artifact was created with different parameters:\n\n"
            + "\n".join(f"  {diff}" for diff in diffs)
            + f"\n\n"
            f"The cached config at {config_path} records what actually created\n"
            f"this artifact. To proceed, either:\n\n"
            f"  1. Rebuild with the current config, by passing the matching\n"
            f"     fresh_* flag for this stage\n"
            f"  2. Change your config back to match the cached version\n"
            f"{divider}\n"
        )


class _DictArtifactConfig(ArtifactConfig):
    """Adapts a plain dict into the `ArtifactConfig` interface.

    Lets `CachedArtifact` subclasses return a dict from `config()` without
    each of them having to declare a config class.
    """

    def __init__(self, payload: dict, artifact_name: str):
        self.payload = payload
        self.artifact_name = artifact_name

    def to_dict(self) -> dict:
        return self.payload


class CachedArtifact(ABC):
    """One reproducible pipeline stage: config, path, build, load.

    Subclasses declare four things — what configuration determines the artifact,
    how to build it, and how to write and read it — and inherit the caching
    logic. `resolve` is the entry point: it returns the cached value when one
    exists and is valid, and otherwise builds, writes, and records the config.

    Two path policies are available. By default the path is fixed and a config
    change is reported as a mismatch, which is the right behavior for a large
    artifact that should be rebuilt in place. Setting `path_includes_digest`
    makes the path content-addressed instead, so configurations coexist rather
    than invalidating each other; because both the digest and the saved config
    derive from `config()`, they cannot disagree.

    Attributes:
        name: Stable identifier for this stage, used as the default directory
            name and as the key in an `ArtifactGraph`.
        depends_on: Names of the stages whose values `build` requires.
        path_includes_digest: Whether to append a config digest to the path.
        config_filename: Name of the YAML config record written inside the
            artifact directory. Override when adopting `CachedArtifact` for
            caches that already carry a record under a different name, so the
            existing records are read rather than silently ignored.
    """

    name: str = "artifact"
    depends_on: tuple[str, ...] = ()
    path_includes_digest: bool = False
    config_filename: str = CONFIG_FILENAME

    def __init__(self, root: str):
        """Initialize the artifact.

        Args:
            root: Directory that this artifact's cache directory lives inside.
        """
        self.root = root

    @abstractmethod
    def config(self) -> dict:
        """Return every parameter that affects this artifact's contents.

        Anything omitted here is invisible to both cache validation and the path
        digest, so a change to it will silently reuse a stale artifact.
        """

    @abstractmethod
    def build(self, deps: Mapping[str, Any]) -> Any:
        """Produce the artifact from scratch.

        Args:
            deps: Resolved values of the stages named in `depends_on`, keyed by
                stage name.

        Returns:
            The built artifact, which is then passed to `write`.
        """

    @abstractmethod
    def write(self, value: Any, path: str) -> None:
        """Persist a freshly built artifact.

        Implement as a no-op for builders that write their own output as a side
        effect, but note that `path` must still exist afterwards, since its
        presence is what marks the artifact as cached.

        Args:
            value: The return value of `build`.
            path: Directory to write into. Created before this is called.
        """

    @abstractmethod
    def read(self, path: str) -> Any:
        """Load a previously cached artifact.

        Args:
            path: Directory previously passed to `write`.

        Returns:
            The artifact, equivalent to what `build` would have returned.
        """

    @property
    def digest(self) -> str:
        """Short, stable hash of this artifact's configuration."""
        return config_digest(self.config())

    @property
    def path(self) -> str:
        """Directory this artifact is cached in.

        Override to impose a different layout, but derive the result from
        `config()` so that the path cannot describe a configuration other than
        the one recorded beside it.
        """
        if self.path_includes_digest:
            return os.path.join(self.root, f"{self.name}_{self.digest}")
        return os.path.join(self.root, self.name)

    @property
    def config_path(self) -> str:
        """Path of the YAML config record inside this artifact's directory."""
        return os.path.join(self.path, self.config_filename)

    def exists(self) -> bool:
        """Whether a cached copy of this artifact is present on disk."""
        return os.path.exists(self.path)

    def artifact_config(self) -> ArtifactConfig:
        """Wrap `config()` in the `ArtifactConfig` validation interface."""
        return _DictArtifactConfig(self.config(), self.name)

    def validate(self, error_on_mismatch: bool = True) -> bool:
        """Check a cached artifact's recorded config against the current one.

        An artifact directory with no config record predates config tracking (or
        was written by hand); that is reported once and accepted rather than
        treated as a mismatch.

        Args:
            error_on_mismatch: Raise on mismatch when True.

        Returns:
            True when the cached config matches, is absent, or the artifact does
            not exist yet.

        Raises:
            ConfigMismatchError: If the configs differ and `error_on_mismatch`.
        """
        if not self.exists():
            return True

        if not os.path.exists(self.config_path):
            print(
                f"Note: using cached {self.name} at {self.path} without config tracking\n"
                f"      (artifact predates config tracking; its parameters cannot be verified)",
                file=sys.stderr,
            )
            return True

        return self.artifact_config().check_cached(
            self.config_path,
            error_on_mismatch=error_on_mismatch,
        )

    def resolve(self, deps: Mapping[str, Any] | None = None, fresh: bool = False) -> Any:
        """Return the artifact, building it only if there is no valid cache.

        Args:
            deps: Resolved dependency values, keyed by stage name. Required if
                `depends_on` is non-empty.
            fresh: Discard any cached copy and rebuild unconditionally.

        Returns:
            The cached or freshly built artifact.

        Raises:
            ConfigMismatchError: If a cached artifact was built with a different
                configuration and `fresh` is False.
        """
        deps = deps or {}
        missing = [name for name in self.depends_on if name not in deps]
        if missing:
            raise KeyError(
                f"{self.name} depends on {list(self.depends_on)} but was not given: {missing}"
            )

        if fresh:
            self.clear()
        elif self.exists():
            self.validate()
            print(f"Loading cached {self.name} from {self.path}", file=sys.stderr)
            return self.read(self.path)

        value = self.build(deps)
        os.makedirs(self.path, exist_ok=True)
        self.write(value, self.path)
        self.artifact_config().save(self.config_path)
        print(f"Saved {self.name} to {self.path}", file=sys.stderr)
        return value

    def clear(self) -> None:
        """Delete this artifact's cache directory if it exists."""
        if os.path.exists(self.path):
            print(f"Clearing cached {self.name} at {self.path}", file=sys.stderr)
            shutil.rmtree(self.path)


class ArtifactGraph:
    """A dependency graph of `CachedArtifact` stages.

    Resolving a stage resolves its dependencies first and memoizes each value, so
    a stage shared by two downstream consumers is built once. Invalidating a
    stage also clears everything reachable from it, which replaces the
    hand-maintained cascade that each pipeline otherwise grows.
    """

    def __init__(self, *artifacts: CachedArtifact):
        """Initialize the graph.

        Args:
            *artifacts: Stages to register. Names must be unique.

        Raises:
            ValueError: If two artifacts share a name.
        """
        self.artifacts: dict[str, CachedArtifact] = {}
        for artifact in artifacts:
            self.add(artifact)
        self._resolved: dict[str, Any] = {}

    def add(self, artifact: CachedArtifact) -> None:
        """Register a stage.

        Args:
            artifact: The stage to add.

        Raises:
            ValueError: If a stage with the same name is already registered.
        """
        if artifact.name in self.artifacts:
            raise ValueError(f"Duplicate artifact name: {artifact.name!r}")
        self.artifacts[artifact.name] = artifact

    def __contains__(self, name: str) -> bool:
        return name in self.artifacts

    def _require(self, name: str) -> CachedArtifact:
        """Look up a stage by name.

        Args:
            name: Registered stage name.

        Returns:
            The stage.

        Raises:
            KeyError: If no stage with that name is registered.
        """
        if name not in self.artifacts:
            known = ", ".join(sorted(self.artifacts)) or "<none>"
            raise KeyError(f"Unknown artifact {name!r}. Registered: {known}")
        return self.artifacts[name]

    def get(self, name: str, fresh: bool = False) -> Any:
        """Resolve a stage, building its dependencies first as needed.

        Args:
            name: Stage to resolve.
            fresh: Rebuild this stage unconditionally. Dependencies are still
                served from cache; use `invalidate` to rebuild a subtree.

        Returns:
            The stage's value.

        Raises:
            KeyError: If the stage or one of its dependencies is unregistered.
            ValueError: If the dependencies contain a cycle.
        """
        return self._get(name, fresh=fresh, visiting=())

    def _get(self, name: str, fresh: bool, visiting: tuple[str, ...]) -> Any:
        """Resolve a stage while tracking the in-progress chain for cycles.

        Args:
            name: Stage to resolve.
            fresh: Rebuild this stage unconditionally.
            visiting: Names currently being resolved, nearest last.

        Returns:
            The stage's value.

        Raises:
            ValueError: If `name` is already in `visiting`.
        """
        if name in visiting:
            cycle = " -> ".join(visiting + (name,))
            raise ValueError(f"Cyclic artifact dependency: {cycle}")

        if name in self._resolved and not fresh:
            return self._resolved[name]

        artifact = self._require(name)
        deps = {
            dep_name: self._get(dep_name, fresh=False, visiting=visiting + (name,))
            for dep_name in artifact.depends_on
        }

        value = artifact.resolve(deps, fresh=fresh)
        self._resolved[name] = value
        return value

    def dependents(self, name: str) -> list[str]:
        """List the stages that depend on a stage, directly or transitively.

        Args:
            name: The stage to start from.

        Returns:
            Names of all downstream stages, in registration order.

        Raises:
            KeyError: If the stage is not registered.
        """
        self._require(name)
        downstream: set[str] = set()

        # iterate to a fixed point so that indirect dependents are included
        # regardless of the order stages were registered in
        changed = True
        while changed:
            changed = False
            for candidate_name, candidate in self.artifacts.items():
                if candidate_name in downstream:
                    continue
                triggers = {name} | downstream
                if triggers & set(candidate.depends_on):
                    downstream.add(candidate_name)
                    changed = True

        return [key for key in self.artifacts if key in downstream]

    def invalidate(self, name: str) -> list[str]:
        """Clear a stage's cache along with every stage downstream of it.

        This is the declarative replacement for a hand-written `fresh_*` cascade:
        the set of things a change invalidates follows from `depends_on` rather
        than from a chain of conditionals kept in sync by hand.

        Args:
            name: The stage to invalidate.

        Returns:
            Names of every cleared stage, upstream-first.

        Raises:
            KeyError: If the stage is not registered.
        """
        cleared = [name] + self.dependents(name)
        for artifact_name in cleared:
            self.artifacts[artifact_name].clear()
            self._resolved.pop(artifact_name, None)
        return cleared
