"""Pack-owned source references and the step source resolver.

A benchmark condition reads named step facts through an explicit namespace.
This module owns that vocabulary and its resolution; the core boundary
(:class:`unigym.evaluation.EnvStepContext`) only provides the primitive readers
``view.record()``, ``view.obs()`` and ``values.get()``.

Namespaces are canonical and explicit.  There is no shorthand and no fallback
search order; the documented grammar is ``kind:key``:

``record:<key>``
    A borrowed leaf from :meth:`unigym.evaluation.EnvView.record`.  Record
    leaves must resolve to tensors here because the record also carries
    non-tensor bookkeeping such as body-name tuples.
``obs:<key>``
    A borrowed tensor from the step's published observation snapshot.
``value:<key>``
    A run-scoped managed value from :class:`unigym.evaluation.EnvStepContext`.
    Tensor and JSON values are both accepted, including a stored JSON ``null``.
``query:<key>``
    Reserved.  The core query face is a typed facade (poses, rigid states,
    histories), not a path grammar, so this resolver refuses it in v1 until an
    explicit typed registry exists.

A key may itself contain ``/`` (for example
``record:articulation/arm/joint_pos``); only the first ``:`` separates the
namespace from the key.  A slash prefix such as ``record/foo`` is not an alias.

The resolver is stateless: it never caches a resolved value, a view or a
context.  Record and observation tensors are borrowed and must not be retained
past the owning callback; managed values are isolated copies produced by the
value view.  A missing ``record:`` or ``obs:`` key is an error naming the
namespace and the sorted candidate keys.  ``value:`` distinguishes an absent
key (error) from a stored JSON ``null`` (a valid ``None``).  No source falls
back to a zero tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from torch import Tensor
from unigym.evaluation import EnvStepContext

from .json import JsonValue

__all__ = [
    "SourceError",
    "SourceKind",
    "SourceParseError",
    "SourceRef",
    "SourceResolveError",
    "SourceResolver",
    "SourceUnavailableError",
]

#: The four source namespaces understood by the pack.
SourceKind: TypeAlias = Literal["record", "obs", "value", "query"]

_KINDS: tuple[SourceKind, ...] = ("record", "obs", "value", "query")
_KIND_SET: frozenset[str] = frozenset(_KINDS)


class SourceError(Exception):
    """Base class for source-reference and source-resolution failures."""


class SourceParseError(SourceError):
    """A source reference does not have the canonical ``kind:key`` shape."""


class SourceResolveError(SourceError):
    """A source reference cannot be resolved at this step boundary."""


class SourceUnavailableError(SourceResolveError, NotImplementedError):
    """A well-formed source namespace is not implemented in pack v1."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """One canonical source reference.

    Args:
        kind: One of ``record``, ``obs``, ``value`` or ``query``.
        key: Non-empty namespace-local key.

    Keys may contain ``/`` because record and observation keys use hierarchical
    names such as ``articulation/arm/joint_pos``; only the first **colon**
    separates the namespace from the key, as in
    ``record:articulation/arm/joint_pos``.  A slash form such as ``record/foo``
    is not an alias.
    """

    kind: SourceKind
    key: str

    def __post_init__(self) -> None:
        """Validate the namespace and key at construction."""
        if not isinstance(self.kind, str) or self.kind not in _KIND_SET:
            raise SourceParseError(f"source kind must be one of {list(_KINDS)}, got {self.kind!r}")
        if not isinstance(self.key, str):
            raise SourceParseError(
                f"source key must be a non-empty string, got {type(self.key).__name__}"
            )
        if not self.key:
            raise SourceParseError(f"source kind {self.kind!r} requires a non-empty key")

    @classmethod
    def parse(cls, text: str) -> SourceRef:
        """Parse a canonical ``kind:key`` reference.

        Args:
            text: Reference text such as
                ``"record:articulation/arm/joint_pos"``.

        Returns:
            The validated reference.

        Raises:
            SourceParseError: ``text`` is not a string, has no known namespace
                prefix, or has an empty key.  Shorthand without a namespace and
                the non-canonical slash form are rejected rather than resolved
                by guessing.
        """
        if not isinstance(text, str):
            raise SourceParseError(
                "source reference must be a string of the form 'kind:key', "
                f"got {type(text).__name__}"
            )
        kind, separator, key = text.partition(":")
        if not separator:
            if kind in _KIND_SET:
                raise SourceParseError(
                    f"source reference {text!r} is missing its key; write {kind + ':<key>'!r}"
                )
            raise SourceParseError(
                f"source reference {text!r} has no namespace prefix; expected one of "
                f"{list(_KINDS)} followed by ':<key>' (shorthand is not accepted)"
            )
        if kind not in _KIND_SET:
            raise SourceParseError(
                f"source reference {text!r} has unknown namespace prefix {kind!r}; "
                f"expected one of {list(_KINDS)} followed by ':<key>' "
                "(shorthand and slash aliases are not accepted)"
            )
        if not key:
            raise SourceParseError(
                f"source reference {text!r} is missing its key after {kind + ':'!r}"
            )
        # ``kind`` was validated against _KIND_SET above.
        return cls(kind=cast(SourceKind, kind), key=key)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> SourceRef:
        """Build a reference from a closed ``{"kind": ..., "key": ...}`` mapping.

        This is the JSON-safe form used by task documents and the condition
        parser.  Unknown keys are rejected; ``kind`` and ``key`` are required.

        Args:
            mapping: A mapping with exactly the ``kind`` and ``key`` entries.

        Returns:
            The validated reference.

        Raises:
            SourceParseError: ``mapping`` is not a mapping, has unknown or
                missing keys, or fails the constructor validation.
        """
        if not isinstance(mapping, Mapping):
            raise SourceParseError(
                f"source mapping must be a mapping, got {type(mapping).__name__}"
            )
        expected = {"kind", "key"}
        unknown = sorted(repr(key) for key in mapping if key not in expected)
        if unknown:
            raise SourceParseError(
                f"source mapping has unknown keys {unknown}; expected exactly 'kind' and 'key'"
            )
        missing = sorted(expected - set(mapping))
        if missing:
            raise SourceParseError(
                f"source mapping is missing required keys {missing}; "
                "expected exactly 'kind' and 'key'"
            )
        kind = mapping["kind"]
        key = mapping["key"]
        if not isinstance(kind, str):
            raise SourceParseError(
                f"source mapping 'kind' must be a string, got {type(kind).__name__}"
            )
        if kind not in _KIND_SET:
            raise SourceParseError(
                f"source mapping 'kind' must be one of {list(_KINDS)}, got {kind!r}"
            )
        # ``kind`` was validated against _KIND_SET above.
        return cls(kind=cast(SourceKind, kind), key=key)

    def __str__(self) -> str:
        """Return the canonical ``kind:key`` form."""
        return f"{self.kind}:{self.key}"


class SourceResolver:
    """Stateless resolver from source references to step-boundary values.

    The resolver holds no state, so calling :meth:`resolve` cannot cache or
    alias a previous step.  Instances are cheap and interchangeable.
    """

    __slots__ = ()

    def resolve(self, step: EnvStepContext, ref: SourceRef) -> Tensor | JsonValue:
        """Resolve ``ref`` against one step boundary.

        Args:
            step: The core step context owned by the caller for this callback.
            ref: The canonical source reference to resolve.

        Returns:
            The referenced tensor (record/obs) or managed value (value).  JSON
            values, including ``None``, are only produced for the ``value``
            namespace.

        Raises:
            TypeError: ``step`` is not an :class:`EnvStepContext` or ``ref`` is
                not a :class:`SourceRef`.
            SourceResolveError: The namespace has no snapshot, the key is
                absent, or a record/obs leaf is not a tensor.
            SourceUnavailableError: The reference uses the reserved ``query``
                namespace.
        """
        if not isinstance(step, EnvStepContext):
            raise TypeError(
                "SourceResolver.resolve: expected an EnvStepContext, " f"got {type(step).__name__}"
            )
        if not isinstance(ref, SourceRef):
            raise TypeError(
                "SourceResolver.resolve: expected a SourceRef; "
                "use SourceRef.parse('kind:key') for text references, "
                f"got {type(ref).__name__}"
            )
        if ref.kind == "record":
            return self._resolve_record(step, ref)
        if ref.kind == "obs":
            return self._resolve_obs(step, ref)
        if ref.kind == "value":
            return self._resolve_value(step, ref)
        return self._resolve_query(ref)

    def _resolve_record(self, step: EnvStepContext, ref: SourceRef) -> Tensor:
        record = step.view.record()
        if ref.key not in record:
            raise SourceResolveError(self._missing("record", ref.key, sorted(record)))
        value = record[ref.key]
        if value is None:
            raise SourceResolveError(self._missing("record", ref.key, sorted(record)))
        if not isinstance(value, Tensor):
            raise SourceResolveError(
                f"record source {ref.key!r} resolved to {type(value).__name__}, "
                "not a torch.Tensor; record sources must be tensors "
                "(use a value source for JSON data)"
            )
        return value

    def _resolve_obs(self, step: EnvStepContext, ref: SourceRef) -> Tensor:
        snapshot = step.view.obs()
        if snapshot is None:
            raise SourceResolveError(
                f"obs source {ref.key!r} is unavailable: this step boundary "
                "published no observation snapshot"
            )
        published = sorted(snapshot.values)
        if ref.key not in snapshot.values:
            raise SourceResolveError(self._missing("obs", ref.key, published))
        value = snapshot.values[ref.key]
        if value is None:
            raise SourceResolveError(self._missing("obs", ref.key, published))
        if not isinstance(value, Tensor):
            raise SourceResolveError(
                f"obs source {ref.key!r} resolved to {type(value).__name__}, "
                "not a torch.Tensor; observation sources must be tensors"
            )
        return value

    def _resolve_value(self, step: EnvStepContext, ref: SourceRef) -> Tensor | JsonValue:
        if ref.key not in step.values:
            candidates = sorted(step.values.snapshot())
            raise SourceResolveError(self._missing("value", ref.key, candidates))
        return step.values.get(ref.key)

    def _resolve_query(self, ref: SourceRef) -> Tensor:
        raise SourceUnavailableError(
            f"query source {ref.key!r} is unavailable: query sources require an "
            "explicit typed QuerySourceRegistry and are not implemented in pack v1; "
            "use record:, obs: or value: instead (there is no query path grammar)"
        )

    @staticmethod
    def _missing(kind: str, key: str, candidates: list[str]) -> str:
        return (
            f"{kind} source {key!r} is not present at this step boundary; "
            f"available {kind} keys: {candidates!r}"
        )
