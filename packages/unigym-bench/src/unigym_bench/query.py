"""Pack-private, one-tick memoized query read snapshot.

A :class:`TickQuery` wraps the borrowed :class:`~unigym.evaluation.EnvView` of
one benchmark boundary.  It acquires the core
:class:`~unigym.evaluation.EvalQuery` lazily on the first read, so attaching a
snapshot to a tick never forces a backend query.  Contact scans and manifolds
are memoized for the tick by normalized env selection (and group); state and
geometry are read-only passthroughs to the core adapter.  The returned query
values are the core query layer's detached, frozen copies, so callers must not
mutate them.

A snapshot is invalid after :meth:`TickQuery.close` and after the owning view
closes.  It must never be reused across ticks: the engine builds one snapshot
per boundary and closes it when the monitor callbacks return.  There is no
module-global cache.

The ``state`` and ``geometry`` accessors return the core adapter's read-only
subfaces.  Reading them through this snapshot fails after close, but a subface
extracted before close stays bound to the core view lifetime; this pack wrapper
does not add a second lifetime check around an extracted core subface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from unigym.evaluation import EnvView, EvalQuery, EvalViewStaleError
from unigym.query import (
    AABB,
    ContactJacobian,
    ContactManifold,
    ContactPoint,
    ContactQuery,
    DistanceResult,
    DistanceScan,
    EntityRef,
    EnvSelection,
    StateQuery,
)

__all__ = ["TickQuery"]

#: Cache-key marker for a selection that must be delegated without memoizing.
#: It keeps a permissive core-teaching error intact instead of raising a pack
#: error while building the cache key.
_UNCACHEABLE = object()


class TickGeometryQuery(Protocol):
    """The read-only geometry subset of the evaluation adapter.

    This mirrors the core ``EvalQuery.geometry`` face deliberately: it has no
    scratch world and no write operation.
    """

    def distance(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
    ) -> DistanceResult:
        """Return the signed distance between two geoms."""
        ...

    def aabb(self, entity: EntityRef, *, env: EnvSelection = None) -> AABB:
        """Return the world AABB of one geom or body."""
        ...

    def all_pairs(
        self,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
    ) -> DistanceScan:
        """Return every collision-geom pair's signed distance."""
        ...


class _TickContacts:
    """Contact subface that memoizes reads for one tick.

    ``scan`` and ``manifold`` return the same detached
    :class:`~unigym.query.ContactManifold` object for repeated identical
    selections.  A manifold is a frozen core value, so callers must treat it
    as read-only.  ``jacobian`` and ``wrench_to_joint`` stay passthroughs
    because they consume a contact point rather than an env selection.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: TickQuery) -> None:
        self._owner = owner

    def scan(self, *, env: EnvSelection = None) -> ContactManifold:
        """Return every active contact for *env*, memoized for this tick."""
        key = self._owner._env_key(env)
        if key is _UNCACHEABLE:
            return self._owner._eval_query().contacts.scan(env=env)
        cached = self._owner._scans.get(key)
        if cached is None:
            cached = self._owner._eval_query().contacts.scan(env=env)
            self._owner._scans[key] = cached
        return cached

    def manifold(self, group: str, *, env: EnvSelection = None) -> ContactManifold:
        """Return one contact group's active points, memoized for this tick."""
        env_key = self._owner._env_key(env)
        if not isinstance(group, str) or env_key is _UNCACHEABLE:
            return self._owner._eval_query().contacts.manifold(group, env=env)
        key = (group, env_key)
        cached = self._owner._manifolds.get(key)
        if cached is None:
            cached = self._owner._eval_query().contacts.manifold(group, env=env)
            self._owner._manifolds[key] = cached
        return cached

    def jacobian(self, point: ContactPoint, articulation: str) -> ContactJacobian:
        """Return the contact-frame Jacobian rows for *point*; passthrough."""
        return self._owner._eval_query().contacts.jacobian(point, articulation)

    def wrench_to_joint(self, point: ContactPoint, articulation: str) -> torch.Tensor:
        """Return the contact force mapped to joint space; passthrough."""
        return self._owner._eval_query().contacts.wrench_to_joint(point, articulation)


class TickQuery:
    """One boundary's memoized, view-bound query read snapshot.

    The snapshot is lazy: constructing it never touches the backend, so a tick
    with no runtime query stays inert until a consumer reads a subface.  When a
    subface is read, the core ``EnvView.query`` teaching errors propagate
    unchanged (``EvalQueryUnavailableError`` for a missing backend and
    ``EvalViewStaleError`` for a closed view or snapshot).
    """

    __slots__ = ("_closed", "_contacts", "_manifolds", "_scans", "_scope", "_view")

    def __init__(self, view: EnvView) -> None:
        if not isinstance(view, EnvView):
            raise TypeError(f"TickQuery.view must be an EnvView, got {type(view).__name__}")
        self._view = view
        self._scope = view.env_id
        self._closed = False
        self._scans: dict[object, ContactManifold] = {}
        self._manifolds: dict[tuple[str, object], ContactManifold] = {}
        self._contacts = _TickContacts(self)

    @property
    def contacts(self) -> ContactQuery:
        """The contact subface, memoized for this tick."""
        self._eval_query()
        return self._contacts

    @property
    def state(self) -> StateQuery:
        """The live state subface; a read-only passthrough."""
        return self._eval_query().state

    @property
    def geometry(self) -> TickGeometryQuery:
        """The geometry subface; a read-only passthrough without scratch."""
        return self._eval_query().geometry

    def close(self) -> None:
        """Invalidate this snapshot and drop its caches (idempotent)."""
        self._closed = True
        self._scans.clear()
        self._manifolds.clear()

    def _eval_query(self) -> EvalQuery:
        """Return the borrowed core adapter, rejecting a closed snapshot."""
        if self._closed:
            raise EvalViewStaleError(
                "TickQuery is closed; query reads are only valid inside the owning tick"
            )
        return self._view.query

    def _env_key(self, env: EnvSelection) -> object:
        """Return a hashable, normalized cache key for one env selection.

        ``None`` means every env on a vector view; on a per-env view it is the
        view's fixed env.  An explicit int and a one-row sequence of the same
        env share one key.  A selection that cannot be normalized is delegated
        without memoizing so the core error remains the contract.
        """
        if env is None:
            return None if self._scope is None else (self._scope,)
        if isinstance(env, bool):
            return _UNCACHEABLE
        if isinstance(env, int):
            return (env,)
        if isinstance(env, torch.Tensor):
            return tuple(int(value) for value in env.detach().reshape(-1).tolist())
        if isinstance(env, (str, bytes, bytearray)):
            return _UNCACHEABLE
        if isinstance(env, Sequence):
            return tuple(int(value) for value in env)
        return _UNCACHEABLE
