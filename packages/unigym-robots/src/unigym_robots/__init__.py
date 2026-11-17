"""Pinned Unitree robot assets for unigym.

The installed package contains metadata and download helpers only. Robot
binaries live in GitHub Release tarballs; see :mod:`unigym_robots.download`.
"""

from __future__ import annotations

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
]
