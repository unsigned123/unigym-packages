"""Download and safely unpack the approved robot asset release tarballs."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .manifest import RELEASE_BASE_URL, Asset, get_asset


class RobotsError(Exception):
    """Base class for unigym-robots errors."""


class UnknownSetError(RobotsError, KeyError):
    """Raised when a set name is not recognised."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class DownloadError(RobotsError):
    """Raised when a release asset cannot be downloaded or unpacked."""


class ChecksumError(RobotsError):
    """Raised when a downloaded archive does not match its pinned sha256."""


class UnsafeArchiveError(RobotsError):
    """Raised when an archive contains paths or members that are not allowed."""


@dataclass(frozen=True, slots=True)
class AssetSet:
    """A release tarball and the approved assets it contains."""

    name: str
    release_asset: str
    install_dir: str
    asset_ids: tuple[str, ...]

    @property
    def assets(self) -> tuple[Asset, ...]:
        return tuple(get_asset(asset_id) for asset_id in self.asset_ids)

    @property
    def sha256(self) -> str:
        """Return the pinned tarball hash shared by every member asset."""
        hashes = {asset.sha256 for asset in self.assets}
        if len(hashes) != 1:
            raise RobotsError(f"inconsistent sha256 values in set {self.name!r}")
        return hashes.pop()

    @property
    def bytes(self) -> int:
        """Approximate uncompressed size of the set's assets."""
        return sum(asset.bytes for asset in self.assets)

    @property
    def download_url(self) -> str:
        return f"{RELEASE_BASE_URL}/{self.release_asset}"


ASSET_SETS: Final[dict[str, AssetSet]] = {
    "mjcf-v0": AssetSet(
        name="mjcf-v0",
        release_asset="unigym-robots-mjcf-v0.1.0.tar.gz",
        install_dir="mjcf-v0.1.0",
        asset_ids=("go1-mjcf", "go2-mjcf", "g1-mjcf"),
    ),
    "usd-go2-v0": AssetSet(
        name="usd-go2-v0",
        release_asset="unigym-robots-usd-go2-v0.1.0.tar.gz",
        install_dir="usd-go2-v0.1.0",
        asset_ids=("go2-usd",),
    ),
    "usd-g1-29dof-v0": AssetSet(
        name="usd-g1-29dof-v0",
        release_asset="unigym-robots-usd-g1-29dof-v0.1.0.tar.gz",
        install_dir="usd-g1-29dof-v0.1.0",
        asset_ids=("g1-29dof-usd",),
    ),
    "usd-g1-43dof-v0": AssetSet(
        name="usd-g1-43dof-v0",
        release_asset="unigym-robots-usd-g1-43dof-v0.1.0.tar.gz",
        install_dir="usd-g1-43dof-v0.1.0",
        asset_ids=("g1-43dof-usd",),
    ),
}

_SET_ALIASES: Final[dict[str, str]] = {
    "mjcf": "mjcf-v0",
    "mjcf-v0": "mjcf-v0",
    "usd-go2": "usd-go2-v0",
    "usd-go2-v0": "usd-go2-v0",
    "usd-g1-29dof": "usd-g1-29dof-v0",
    "g1-29dof-usd": "usd-g1-29dof-v0",
    "usd-g1-29dof-v0": "usd-g1-29dof-v0",
    "usd-g1-43dof": "usd-g1-43dof-v0",
    "g1-43dof-usd": "usd-g1-43dof-v0",
    "usd-g1-43dof-v0": "usd-g1-43dof-v0",
}
for _set in ASSET_SETS.values():
    _SET_ALIASES[_set.release_asset] = _set.name
    _SET_ALIASES[_set.release_asset.removesuffix(".tar.gz")] = _set.name

ROOT_ENV_VAR: Final = "UNIGYM_ROBOTS_ROOT"
DEFAULT_ROOT: Final = Path.home() / ".cache" / "unigym" / "robots"


def default_root() -> Path:
    """Return the extraction root, honouring ``UNIGYM_ROBOTS_ROOT``."""
    configured = os.environ.get(ROOT_ENV_VAR)
    return Path(configured).expanduser() if configured else DEFAULT_ROOT


def resolve_set(name: str) -> AssetSet:
    """Resolve a canonical set name, short alias, or tarball filename."""
    key = name.strip()
    canonical = _SET_ALIASES.get(key) or _SET_ALIASES.get(key.lower())
    if canonical is None:
        known = ", ".join(ASSET_SETS)
        raise UnknownSetError(f"unknown set {name!r}; known sets: {known}")
    return ASSET_SETS[canonical]


def asset_path(asset_id: str, root: str | Path | None = None) -> Path:
    """Return the expected on-disk path of an asset's primary model file."""
    asset = get_asset(asset_id)
    base = Path(root).expanduser() if root is not None else default_root()
    return base / ASSET_SETS[asset.set_name].install_dir / asset.archive_path


def sha256_file(path: str | Path) -> str:
    """Compute the sha256 hex digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise :class:`ChecksumError` unless ``path`` matches ``expected``."""
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ChecksumError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")


def _validate_members(members: Iterable[tarfile.TarInfo], destination: Path) -> None:
    destination = destination.resolve()
    for member in members:
        name = member.name
        if not name or name.startswith(("/", "\\")) or "\\" in name:
            raise UnsafeArchiveError(f"unsafe archive member path: {name!r}")
        parts = Path(name).parts
        if any(part in {"", ".", ".."} for part in parts):
            raise UnsafeArchiveError(f"unsafe archive member path: {name!r}")
        if not (member.isfile() or member.isdir()):
            raise UnsafeArchiveError(f"unsupported archive member type: {name!r}")
        target = (destination / name).resolve()
        if os.path.commonpath((str(destination), str(target))) != str(destination):
            raise UnsafeArchiveError(f"archive member escapes destination: {name!r}")


def extract_archive(archive: str | Path, destination: str | Path) -> Path:
    """Safely extract ``archive`` into ``destination``.

    Only regular files and directories are accepted, and every member must
    stay inside ``destination``. Extraction uses the ``data`` filter when the
    running Python provides it.
    """
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            _validate_members(tar.getmembers(), destination)
            try:
                tar.extractall(destination, filter="data")
            except TypeError:  # pragma: no cover - Python without extraction filters
                tar.extractall(destination)
    except (tarfile.TarError, OSError) as exc:
        raise DownloadError(f"failed to extract {archive}: {exc}") from exc
    return destination


def _download(url: str, destination: Path, timeout: float) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "unigym-robots/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(destination, "wb") as stream:
                shutil.copyfileobj(response, stream)
    except OSError as exc:
        raise DownloadError(f"failed to download {url}: {exc}") from exc
    return destination


def download_set(
    name: str,
    root: str | Path | None = None,
    *,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Download, verify and extract one asset set.

    Args:
        name: Canonical set name, alias, or release tarball filename.
        root: Extraction root. Defaults to ``UNIGYM_ROBOTS_ROOT`` or
            ``~/.cache/unigym/robots``.
        force: Re-download even when the install directory already exists.
        timeout: Per-request network timeout in seconds.

    Returns:
        The directory that contains the extracted set.
    """
    asset_set = resolve_set(name)
    base = Path(root).expanduser() if root is not None else default_root()
    base.mkdir(parents=True, exist_ok=True)
    install_dir = base / asset_set.install_dir

    if install_dir.is_dir() and any(install_dir.iterdir()) and not force:
        return install_dir

    with tempfile.TemporaryDirectory(dir=base, prefix=".unigym-robots-") as tmp:
        temp_root = Path(tmp)
        archive = _download(asset_set.download_url, temp_root / asset_set.release_asset, timeout)
        verify_sha256(archive, asset_set.sha256)
        extracted = extract_archive(archive, temp_root / "extract")
        for asset in asset_set.assets:
            expected = extracted / asset.archive_path
            if not expected.is_file():
                raise DownloadError(
                    f"{asset_set.release_asset} does not contain {asset.archive_path}"
                )
        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(extracted), str(install_dir))
    return install_dir


def download_asset(
    asset_id: str,
    root: str | Path | None = None,
    *,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Download the set that contains ``asset_id`` and return its install dir."""
    asset = get_asset(asset_id)
    return download_set(asset.set_name, root, force=force, timeout=timeout)


def download_all(
    root: str | Path | None = None,
    *,
    force: bool = False,
    timeout: float = 120.0,
) -> dict[str, Path]:
    """Download every approved asset set and return install dirs by set name."""
    return {name: download_set(name, root, force=force, timeout=timeout) for name in ASSET_SETS}


__all__ = [
    "ASSET_SETS",
    "ROOT_ENV_VAR",
    "AssetSet",
    "ChecksumError",
    "DownloadError",
    "RobotsError",
    "UnknownSetError",
    "UnsafeArchiveError",
    "asset_path",
    "default_root",
    "download_all",
    "download_asset",
    "download_set",
    "extract_archive",
    "resolve_set",
    "sha256_file",
    "verify_sha256",
]
