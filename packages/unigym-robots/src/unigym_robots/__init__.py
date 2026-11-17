"""Pinned Unitree robot assets for unigym.

The installed package contains metadata and download helpers only. Robot
binaries live in GitHub Release tarballs; see :mod:`unigym_robots.download`.
"""

from __future__ import annotations

from pathlib import Path

from .download import (
    ASSET_SETS,
    AssetSet,
    ChecksumError,
    DownloadError,
    RobotsError,
    UnknownSetError,
    UnsafeArchiveError,
    asset_path,
    default_root,
    download_all,
    download_asset,
    download_set,
)
from .manifest import ASSETS, ASSETS_BY_ID, RELEASE_TAG, Asset, get_asset

__version__ = "0.1.0"


def pack_root() -> Path:
    """Return the ``unigym.packs`` root shipped inside this package.

    The directory carries ``unigym.pack.yaml`` plus the Go1/Go2/G1 robot
    profiles.  No robot binary lives here; the profiles reference the pinned
    GitHub Release tarballs by ``release:`` id.
    """
    return Path(__file__).resolve().parent / "pack"


__all__ = [
    "ASSETS",
    "ASSETS_BY_ID",
    "ASSET_SETS",
    "RELEASE_TAG",
    "Asset",
    "AssetSet",
    "ChecksumError",
    "DownloadError",
    "RobotsError",
    "UnknownSetError",
    "UnsafeArchiveError",
    "__version__",
    "asset_path",
    "default_root",
    "download_all",
    "download_asset",
    "download_set",
    "get_asset",
    "pack_root",
]
