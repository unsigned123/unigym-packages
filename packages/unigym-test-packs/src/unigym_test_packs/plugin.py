"""Pytest plugin for installed unigym test packs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from unigym.pack_api import PackRuntime, discover

from .discovery import PackDiscoveryError, PackIndex, discover_installed_packs
from .markers import (
    MARKER_NAME,
    PackMarkerError,
    marker_requirement,
    validate_pack_name,
)

_STRICT_OPTION = "--strict-test-packs"
_STRICT_DEST = "strict_test_packs"
_INDEX_KEY = pytest.StashKey[PackIndex]()
_RUNTIME_KEY = pytest.StashKey[dict[tuple[str, ...], PackRuntime]]()


def _install_hint(name: str) -> str:
    """Return the distribution hint for one missing pack."""
    return f"unigym-test-pack-{name.replace('_', '-')}"


def _missing_message(missing: tuple[str, ...]) -> str:
    """Build the shared missing-pack teaching message."""
    hints = ", ".join(_install_hint(name) for name in missing)
    return (
        f"missing test pack(s): {', '.join(missing)}; "
        f"install {hints} or provide an equivalent pack distribution"
    )


def _index(config: pytest.Config) -> PackIndex:
    """Return the configuration's installed pack index."""
    try:
        return config.stash[_INDEX_KEY]
    except KeyError:
        raise pytest.UsageError(
            "unigym-test-packs was not configured; the plugin did not run pytest_configure"
        ) from None


def _runtime_cache(config: pytest.Config) -> dict[tuple[str, ...], PackRuntime]:
    """Return the configuration's per-tuple runtime cache."""
    try:
        return config.stash[_RUNTIME_KEY]
    except KeyError:
        raise pytest.UsageError(
            "unigym-test-packs was not configured; the plugin did not run pytest_configure"
        ) from None


def _declared(item: pytest.Item) -> tuple[str, ...]:
    """Return the item's declared packs, turning marker defects into failures."""
    try:
        return marker_requirement(item)
    except PackMarkerError as err:
        pytest.fail(str(err), pytrace=False)


def _require_declared(
    item: pytest.Item, declared: tuple[str, ...], requested: tuple[str, ...]
) -> None:
    """Fail when a fixture requests a pack the marker did not declare."""
    undeclared = tuple(name for name in requested if name not in declared)
    if undeclared:
        pytest.fail(
            f"{item.nodeid}: fixture request for pack(s) {undeclared} not declared by "
            f"{MARKER_NAME}{declared}",
            pytrace=False,
        )


def _require_installed(
    item: pytest.Item, config: pytest.Config, index: PackIndex, requested: tuple[str, ...]
) -> None:
    """Skip or fail when a declared pack is not installed."""
    missing = index.missing(requested)
    if not missing:
        return
    message = _missing_message(missing)
    if config.getoption(_STRICT_DEST):
        pytest.fail(f"{item.nodeid}: {message}", pytrace=False)
    pytest.skip(message)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the strict test-pack availability switch."""
    parser.addoption(
        _STRICT_OPTION,
        action="store_true",
        default=False,
        dest=_STRICT_DEST,
        help="fail at test setup when a declared test pack is not installed",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker and load every installed test pack once."""
    registered = {entry.split(":", 1)[0].strip() for entry in config.getini("markers")}
    if MARKER_NAME not in registered:
        config.addinivalue_line(
            "markers",
            f"{MARKER_NAME}: declare installed test packs required by the test",
        )
    try:
        config.stash[_INDEX_KEY] = discover_installed_packs()
    except PackDiscoveryError as err:
        raise pytest.UsageError(str(err)) from err
    config.stash[_RUNTIME_KEY] = {}


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Validate a pack-marked item and enforce availability at setup."""
    declared = _declared(item)
    if not declared:
        return
    _require_installed(item, item.config, _index(item.config), declared)


@pytest.fixture
def pack_root(request: pytest.FixtureRequest) -> Callable[[str], Path]:
    """Return a factory for a declared pack's installed root directory."""
    item = request.node
    if not isinstance(item, pytest.Item):
        pytest.fail("pack_root is only available to collected test items", pytrace=False)
    index = _index(request.config)

    def resolve(name: str) -> Path:
        validated = validate_pack_name(name, where=f"{item.nodeid}: pack_root argument")
        declared = _declared(item)
        _require_declared(item, declared, (validated,))
        _require_installed(item, request.config, index, (validated,))
        return index.root(validated)

    return resolve


@pytest.fixture
def pack_runtime(request: pytest.FixtureRequest) -> Callable[..., PackRuntime]:
    """Return a factory for an explicit core pack runtime over declared packs."""
    item = request.node
    if not isinstance(item, pytest.Item):
        pytest.fail("pack_runtime is only available to collected test items", pytrace=False)
    index = _index(request.config)
    cache = _runtime_cache(request.config)

    def build(*names: str) -> PackRuntime:
        if not names:
            pytest.fail("pack_runtime requires at least one pack name", pytrace=False)
        validated = tuple(
            validate_pack_name(name, where=f"{item.nodeid}: pack_runtime argument")
            for name in names
        )
        if len(set(validated)) != len(validated):
            pytest.fail(
                f"{item.nodeid}: pack_runtime repeats a pack name: {validated}",
                pytrace=False,
            )
        declared = _declared(item)
        _require_declared(item, declared, validated)
        _require_installed(item, request.config, index, validated)
        if validated not in cache:
            cache[validated] = discover(roots=[index.root(name) for name in validated])
        return cache[validated]

    return build
