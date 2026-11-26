"""Offline tests for the asset manifest and download helpers."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import unigym_robots
from unigym_robots import cli, download, manifest
from unigym_robots.download import (
    ASSET_SETS,
    ChecksumError,
    UnknownSetError,
    UnsafeArchiveError,
    asset_path,
    extract_archive,
    resolve_set,
    sha256_file,
    verify_sha256,
)
from unigym_robots.manifest import ASSETS, ASSETS_BY_ID, RELEASE_TAG, get_asset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = {
    "go1-mjcf",
    "go2-mjcf",
    "g1-mjcf",
    "go2-usd",
    "g1-29dof-usd",
    "g1-43dof-usd",
}
EXPECTED_SETS = {"mjcf-v0", "usd-go2-v0", "usd-g1-29dof-v0", "usd-g1-43dof-v0"}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _make_tar(path: Path, files: dict[str, bytes], extra: list[tarfile.TarInfo] | None = None):
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        for info in extra or []:
            tar.addfile(info)


def test_exactly_the_approved_six_assets() -> None:
    assert {asset.asset_id for asset in ASSETS} == EXPECTED_IDS
    assert set(ASSETS_BY_ID) == EXPECTED_IDS
    assert len(ASSETS) == 6


def test_asset_metadata_is_complete_and_pinned() -> None:
    for asset in ASSETS:
        assert asset.robot in {"Go1", "Go2", "G1"}
        assert asset.format in {"mjcf", "usd"}
        assert asset.set_name in EXPECTED_SETS
        assert asset.upstream_repo
        assert len(asset.upstream_commit) == 40
        assert asset.upstream_path
        assert asset.license_id in {"BSD-3-Clause", "Apache-2.0"}
        assert asset.license_file.startswith("LICENSES/")
        assert _is_sha256(asset.sha256)
        assert asset.bytes > 0
        assert asset.archive_path
        assert asset.transformations


def test_release_tag_and_urls_are_pinned() -> None:
    assert RELEASE_TAG == "unigym-robots-assets-v0.1.0"
    for asset in ASSETS:
        assert asset.download_url.startswith(manifest.RELEASE_BASE_URL)
        assert asset.download_url.endswith(asset.release_asset)
        assert "/releases/download/unigym-robots-assets-v0.1.0/" in asset.download_url


def test_license_files_exist_and_match_manifest() -> None:
    for asset in ASSETS:
        license_path = PACKAGE_ROOT / asset.license_file
        assert license_path.is_file(), asset.license_file
        assert license_path.stat().st_size > 0
    assert (PACKAGE_ROOT / "LICENSES" / "G1-Apache-2.0.txt").is_file()


def test_docs_and_no_binaries_in_package() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "PROVENANCE.md").is_file()
    assert (PACKAGE_ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    binary_suffixes = {".usd", ".usda", ".usdc", ".stl", ".obj", ".dae", ".xml"}
    for directory in (PACKAGE_ROOT / "src", PACKAGE_ROOT / "tests"):
        for path in directory.rglob("*"):
            assert path.suffix.lower() not in binary_suffixes, path


def test_sets_group_assets_and_share_tarball_hash() -> None:
    assert set(ASSET_SETS) == EXPECTED_SETS
    covered = [asset_id for asset_set in ASSET_SETS.values() for asset_id in asset_set.asset_ids]
    assert sorted(covered) == sorted(EXPECTED_IDS)
    for asset_set in ASSET_SETS.values():
        assert asset_set.assets
        assert len({asset.sha256 for asset in asset_set.assets}) == 1
        assert asset_set.sha256 == asset_set.assets[0].sha256
        assert asset_set.release_asset in {asset.release_asset for asset in asset_set.assets}
        assert asset_set.bytes == sum(asset.bytes for asset in asset_set.assets)


def test_set_asset_paths_live_under_their_install_dir() -> None:
    for asset in ASSETS:
        prefix = ASSET_SETS[asset.set_name].install_dir
        assert asset.archive_path.split("/", 1)[0] in {"go1", "go2", "g1", "g1-29dof", "g1-43dof"}
        path = asset_path(asset.asset_id, "/tmp/robots")
        assert str(path) == f"/tmp/robots/{prefix}/{asset.archive_path}"


def test_resolve_set_accepts_names_aliases_and_filenames() -> None:
    assert resolve_set("mjcf-v0").name == "mjcf-v0"
    assert resolve_set("mjcf").name == "mjcf-v0"
    assert resolve_set("MJCF").name == "mjcf-v0"
    assert resolve_set("unigym-robots-mjcf-v0.1.0.tar.gz").name == "mjcf-v0"
    assert resolve_set("g1-43dof-usd").name == "usd-g1-43dof-v0"
    with pytest.raises(UnknownSetError):
        resolve_set("nope")


def test_get_asset_rejects_unknown_id() -> None:
    with pytest.raises(KeyError):
        get_asset("go3-mjcf")


def test_default_root_honours_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UNIGYM_ROBOTS_ROOT", str(tmp_path / "custom"))
    assert download.default_root() == tmp_path / "custom"
    monkeypatch.delenv("UNIGYM_ROBOTS_ROOT")
    assert download.default_root() == Path.home() / ".cache" / "unigym" / "robots"


def test_sha256_helpers(tmp_path: Path) -> None:
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"unigym-robots")
    expected = hashlib.sha256(b"unigym-robots").hexdigest()
    assert sha256_file(sample) == expected
    verify_sha256(sample, expected)
    verify_sha256(sample, expected.upper())
    with pytest.raises(ChecksumError):
        verify_sha256(sample, "0" * 64)


def test_extract_archive_roundtrip(tmp_path: Path) -> None:
    archive = tmp_path / "safe.tar.gz"
    _make_tar(archive, {"go1/go1.xml": b"<mujoco/>", "go1/LICENSE": b"BSD"})
    destination = extract_archive(archive, tmp_path / "out")
    assert (destination / "go1" / "go1.xml").read_bytes() == b"<mujoco/>"
    assert (destination / "go1" / "LICENSE").read_bytes() == b"BSD"


@pytest.mark.parametrize(
    "name",
    ["../escape.xml", "/absolute.xml", "a/../../escape.xml", "a\\b.xml"],
)
def test_extract_archive_rejects_unsafe_paths(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _make_tar(archive, {name: b"nope"})
    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escape.xml").exists()


def test_extract_archive_rejects_links(tmp_path: Path) -> None:
    link = tarfile.TarInfo("go1/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    archive = tmp_path / "link.tar.gz"
    _make_tar(archive, {"go1/go1.xml": b"<mujoco/>"}, extra=[link])
    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out")


def test_download_set_verifies_and_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_download(url: str, destination: Path, timeout: float) -> Path:
        assert url.endswith("unigym-robots-mjcf-v0.1.0.tar.gz")
        calls.append(destination)
        _make_tar(
            destination,
            {
                "go1/go1.xml": b"<mujoco/>",
                "go2/go2.xml": b"<mujoco/>",
                "g1/g1.xml": b"<mujoco/>",
            },
        )
        return destination

    monkeypatch.setattr(download, "_download", fake_download)
    monkeypatch.setattr(download, "verify_sha256", lambda *args: None)

    install_dir = download.download_set("mjcf-v0", tmp_path)
    assert install_dir == tmp_path / "mjcf-v0.1.0"
    for asset in ASSET_SETS["mjcf-v0"].assets:
        assert (install_dir / asset.archive_path).is_file()
    assert len(calls) == 1

    download.download_set("mjcf-v0", tmp_path)
    assert len(calls) == 1, "existing install should be reused"

    download.download_set("mjcf-v0", tmp_path, force=True)
    assert len(calls) == 2


def test_download_set_rejects_incomplete_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(url: str, destination: Path, timeout: float) -> Path:
        _make_tar(destination, {"go1/go1.xml": b"<mujoco/>"})
        return destination

    monkeypatch.setattr(download, "_download", fake_download)
    monkeypatch.setattr(download, "verify_sha256", lambda *args: None)
    with pytest.raises(download.DownloadError):
        download.download_set("mjcf-v0", tmp_path)


def test_cli_list_and_info_are_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["sets"]) == 4
    assert payload["sets"][0]["name"] == "mjcf-v0"

    assert cli.main(["info", "go2-usd", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["asset_id"] == "go2-usd"
    assert info["sha256"] == get_asset("go2-usd").sha256

    assert cli.main(["info", "mjcf-v0"]) == 0
    assert "mjcf-v0" in capsys.readouterr().out


def test_cli_version_matches_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"unigym-robots {unigym_robots.__version__}"


def test_cli_errors_return_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info", "not-a-thing"]) == 2
    assert "error:" in capsys.readouterr().err
    assert cli.main(["download", "--set", "not-a-set"]) == 2
    assert "error:" in capsys.readouterr().err
