"""Pack-private tick query snapshot: laziness, memoization and lifetime."""

from __future__ import annotations

import pytest
import torch
from unigym.evaluation import EvalQueryUnavailableError, EvalViewStaleError
from unigym_bench.engine import StreamingEngine
from unigym_bench.query import TickQuery
from _support import (
    QueryErrorSpyMonitor,
    QueryFailMonitor,
    QuerySpyMonitor,
    make_bundle,
    make_query_view,
    make_view,
)


def test_building_a_tick_query_does_not_resolve_the_backend() -> None:
    view, _, factory_calls = make_query_view(2)

    tick = TickQuery(view)

    assert tick is not None
    assert factory_calls == []


def test_scan_is_memoized_by_normalized_env_selection() -> None:
    view, fake, factory_calls = make_query_view(2)
    tick = TickQuery(view)

    first = tick.contacts.scan()
    again = tick.contacts.scan()
    assert first is again
    assert fake.contacts.scan_calls == [None]

    zero = tick.contacts.scan(env=0)
    zero_again = tick.contacts.scan(env=[0])
    one = tick.contacts.scan(env=torch.tensor([1]))
    assert zero is zero_again
    assert one is not zero
    assert fake.contacts.scan_calls == [None, (0,), (1,)]
    assert factory_calls == [1]


def test_manifold_is_memoized_by_group_and_env_selection() -> None:
    view, fake, _ = make_query_view(2)
    tick = TickQuery(view)

    feet = tick.contacts.manifold("feet")
    assert tick.contacts.manifold("feet") is feet
    hands = tick.contacts.manifold("hands")
    tick.contacts.manifold("feet", env=1)

    assert hands is not feet
    assert fake.contacts.manifold_calls == [
        ("feet", None),
        ("hands", None),
        ("feet", (1,)),
    ]
    assert feet.points == ()


def test_state_and_geometry_are_read_only_passthroughs() -> None:
    view, fake, _ = make_query_view(2)
    tick = TickQuery(view)

    assert tick.state is tick.state
    assert tick.geometry is tick.geometry
    tick.state.body_pose("base")
    tick.state.body_pose("base")
    tick.geometry.aabb("base")

    assert fake.state.body_pose_calls == [("base", None, None), ("base", None, None)]
    assert fake.geometry.aabb_calls == [("base", None)]


def test_missing_backend_raises_the_core_teaching_error() -> None:
    tick = TickQuery(make_view(2))

    with pytest.raises(EvalQueryUnavailableError, match="exposes no runtime query"):
        _ = tick.contacts
    with pytest.raises(EvalQueryUnavailableError):
        _ = tick.state
    with pytest.raises(EvalQueryUnavailableError):
        _ = tick.geometry


def test_close_invalidates_reads_and_is_idempotent() -> None:
    view, _, _ = make_query_view(2)
    tick = TickQuery(view)
    tick.contacts.scan()

    tick.close()
    tick.close()

    with pytest.raises(EvalViewStaleError, match="TickQuery is closed"):
        _ = tick.contacts
    with pytest.raises(EvalViewStaleError, match="TickQuery is closed"):
        _ = tick.state
    with pytest.raises(EvalViewStaleError, match="TickQuery is closed"):
        _ = tick.geometry


def test_view_close_makes_reads_fail() -> None:
    view, _, _ = make_query_view(2)
    tick = TickQuery(view)
    tick.contacts.scan()

    view.close()

    with pytest.raises(EvalViewStaleError):
        _ = tick.contacts
    with pytest.raises(EvalViewStaleError):
        _ = tick.geometry


def test_engine_shares_one_tick_query_across_monitors_and_replaces_it() -> None:
    view, fake, factory_calls = make_query_view(1)
    first = QuerySpyMonitor("first")
    second = QuerySpyMonitor("second")
    engine = StreamingEngine([first, second])

    engine.on_step(make_bundle(num_envs=1, t=0, view=view))

    first_query = first.contexts[0].query
    assert first_query is not None
    assert second.contexts[0].query is first_query
    assert len(fake.contacts.scan_calls) == 1
    assert factory_calls == [1]
    with pytest.raises(EvalViewStaleError, match="TickQuery is closed"):
        _ = first_query.contacts

    engine.on_step(make_bundle(num_envs=1, t=1, view=view))

    next_query = first.contexts[1].query
    assert next_query is not None
    assert next_query is not first_query
    assert second.contexts[1].query is next_query
    assert len(fake.contacts.scan_calls) == 2


def test_engine_closes_the_tick_snapshot_when_a_callback_raises() -> None:
    view, _, _ = make_query_view(1)
    monitor = QueryFailMonitor(fail_on_tick=True)
    engine = StreamingEngine([monitor])

    with pytest.raises(RuntimeError, match="query fail tick"):
        engine.on_step(make_bundle(num_envs=1, t=0, view=view))

    query = monitor.contexts[0].query
    assert query is not None
    with pytest.raises(EvalViewStaleError, match="TickQuery is closed"):
        _ = query.contacts


def test_engine_closes_the_final_snapshot_when_a_callback_raises() -> None:
    view, _, _ = make_query_view(1)
    monitor = QueryFailMonitor(fail_on_episode_end=True)
    engine = StreamingEngine([monitor])
    engine.on_step(make_bundle(num_envs=1, t=0, view=view))

    with pytest.raises(RuntimeError, match="query fail episode_end"):
        engine.on_episode_end([0], make_bundle(num_envs=1, t=0, view=view))

    query = monitor.final_contexts[0].query
    assert query is not None
    with pytest.raises(EvalViewStaleError, match="TickQuery is closed"):
        _ = query.contacts


def test_engine_tick_without_a_query_backend_stays_lazy_until_a_read() -> None:
    monitor = QueryErrorSpyMonitor()
    engine = StreamingEngine([monitor])

    engine.on_step(make_bundle(num_envs=1, t=0, view=make_view(1)))

    assert len(monitor.errors) == 1
    assert isinstance(monitor.errors[0], EvalQueryUnavailableError)


def test_query_cache_does_not_survive_reset() -> None:
    view, fake, _ = make_query_view(1)
    monitor = QuerySpyMonitor()
    engine = StreamingEngine([monitor])
    engine.on_step(make_bundle(num_envs=1, t=0, view=view))
    first_query = monitor.contexts[0].query
    assert first_query is not None

    engine.on_reset([0])
    engine.on_step(make_bundle(num_envs=1, t=1, view=view))

    next_query = monitor.contexts[1].query
    assert next_query is not None
    assert next_query is not first_query
    assert len(fake.contacts.scan_calls) == 2
    with pytest.raises(EvalViewStaleError):
        _ = first_query.contacts


def test_query_cache_does_not_survive_rewind() -> None:
    view, fake, _ = make_query_view(1)
    monitor = QuerySpyMonitor()
    engine = StreamingEngine([monitor])
    engine.on_step(make_bundle(num_envs=1, t=0, view=view))
    snapshot = engine.rewind_state()
    engine.on_step(make_bundle(num_envs=1, t=1, view=view))
    before_rewind = monitor.contexts[1].query
    assert before_rewind is not None

    engine.apply_rewind_state(snapshot)
    engine.on_step(make_bundle(num_envs=1, t=2, view=view))

    after_rewind = monitor.contexts[2].query
    assert after_rewind is not None
    assert after_rewind is not before_rewind
    assert len(fake.contacts.scan_calls) == 3
    with pytest.raises(EvalViewStaleError):
        _ = before_rewind.contacts
