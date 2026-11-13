"""The basic content pack and its installed pack root."""

from __future__ import annotations

from pathlib import Path

__all__ = ["pack_root"]


def pack_root() -> Path:
    """Return the persistent pack root shipped inside this distribution."""
    return Path(__file__).resolve().parent / "pack"
