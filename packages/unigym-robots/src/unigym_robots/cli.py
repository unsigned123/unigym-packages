"""Command line interface for unigym-robots."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .download import (
    ASSET_SETS,
    RobotsError,
    asset_path,
    default_root,
    download_all,
    download_asset,
    download_set,
    resolve_set,
)
from .manifest import ASSETS_BY_ID, Asset, get_asset


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    raise AssertionError("unreachable")


def _asset_dict(asset: Asset, root: Path | None = None) -> dict[str, object]:
    data = dataclasses.asdict(asset)
    if root is not None:
        data["local_path"] = str(asset_path(asset.asset_id, root))
    return data


def _cmd_list(args: argparse.Namespace) -> int:
    root = default_root()
    if args.json:
        payload = {
            "root": str(root),
            "sets": [
                {
                    "name": asset_set.name,
                    "release_asset": asset_set.release_asset,
                    "sha256": asset_set.sha256,
                    "assets": [_asset_dict(asset, root) for asset in asset_set.assets],
                }
                for asset_set in ASSET_SETS.values()
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"root: {root}")
    for asset_set in ASSET_SETS.values():
        print(f"\n{asset_set.name}  ({_format_bytes(asset_set.bytes)}, sha256={asset_set.sha256})")
        print(f"  tarball: {asset_set.release_asset}")
        for asset in asset_set.assets:
            print(
                f"  - {asset.asset_id:<16} {asset.robot:<4} {asset.format:<4} {asset.archive_path}"
            )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    root = default_root()
    target = args.target
    if target in ASSETS_BY_ID:
        payload: dict[str, object] = _asset_dict(get_asset(target), root)
    else:
        asset_set = resolve_set(target)
        payload = {
            "set": asset_set.name,
            "release_asset": asset_set.release_asset,
            "download_url": asset_set.download_url,
            "sha256": asset_set.sha256,
            "root": str(root),
            "assets": [_asset_dict(asset, root) for asset in asset_set.assets],
        }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for key, value in payload.items():
        if key == "assets":
            print(f"{key}:")
            for asset in value:  # type: ignore[union-attr]
                print(f"  - {asset['asset_id']}: {asset['archive_path']}")
        else:
            print(f"{key}: {value}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser() if args.root else None
    try:
        if args.all:
            installed = download_all(root, force=args.force)
            for name, path in installed.items():
                print(f"{name}: {path}")
            return 0
        if args.asset:
            asset = get_asset(args.asset)
            installed_dir = download_asset(asset.asset_id, root, force=args.force)
            print(f"{asset.asset_id}: {installed_dir / asset.archive_path}")
            return 0
        if args.set_name:
            asset_set = resolve_set(args.set_name)
            installed_dir = download_set(asset_set.name, root, force=args.force)
            print(f"{asset_set.name}: {installed_dir}")
            return 0
    except RobotsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("error: pass --set, --asset or --all", file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unigym-robots",
        description="Download pinned Unitree MJCF and USD assets for unigym.",
    )
    parser.add_argument("--version", action="version", version="unigym-robots 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list sets and assets")
    list_parser.add_argument("--json", action="store_true", help="emit JSON")
    list_parser.set_defaults(func=_cmd_list)

    info_parser = subparsers.add_parser("info", help="show one asset or set")
    info_parser.add_argument("target", help="asset id (for example go2-usd) or set name")
    info_parser.add_argument("--json", action="store_true", help="emit JSON")
    info_parser.set_defaults(func=_cmd_info)

    download_parser = subparsers.add_parser("download", help="download and extract assets")
    group = download_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", dest="set_name", help="set name, for example mjcf-v0")
    group.add_argument("--asset", help="single asset id")
    group.add_argument("--all", action="store_true", help="download every set")
    download_parser.add_argument("--root", help="extraction root (default: ~/.cache/unigym/robots)")
    download_parser.add_argument("--force", action="store_true", help="re-download existing sets")
    download_parser.set_defaults(func=_cmd_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyError as exc:
        if exc.args and isinstance(exc.args[0], str):
            print(f"error: {exc.args[0]}", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
