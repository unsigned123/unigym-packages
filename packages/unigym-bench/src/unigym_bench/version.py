"""Version constants for the external benchmark pack.

``CORE_SURFACE_VERSION`` is a pack-local revision of the required ``unigym``
public-surface contract.  It is not a value exported by ``unigym``; bump it
when the pack starts relying on a different set of public seams.
"""

from __future__ import annotations

__all__ = ["CORE_SURFACE_VERSION", "REQUIRED_CORE_VERSION", "__version__"]

__version__ = "0.1.0"

CORE_SURFACE_VERSION: int = 2

REQUIRED_CORE_VERSION = "0.106.0"
