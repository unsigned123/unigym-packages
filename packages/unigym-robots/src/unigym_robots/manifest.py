"""Immutable manifest of the approved Unitree robot assets.

The package itself ships no robot binaries. Each ``Asset`` points at a
GitHub Release tarball published by ``unsigned123/unigym-packages`` and records
the exact upstream source that tarball was built from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

AssetFormat = Literal["mjcf", "usd"]

RELEASE_TAG: Final = "unigym-robots-assets-v0.1.0"
RELEASE_REPO: Final = "unsigned123/unigym-packages"
RELEASE_BASE_URL: Final = f"https://github.com/{RELEASE_REPO}/releases/download/{RELEASE_TAG}"


@dataclass(frozen=True, slots=True)
class Asset:
    """One approved robot asset and the release archive that carries it."""

    asset_id: str
    robot: str
    format: AssetFormat
    set_name: str
    upstream_repo: str
    upstream_commit: str
    upstream_path: str
    license_id: str
    license_file: str
    release_asset: str
    sha256: str
    bytes: int
    archive_path: str
    transformations: tuple[str, ...]

    @property
    def filename(self) -> str:
        """Basename of the primary model file inside the archive."""
        return self.archive_path.rsplit("/", 1)[-1]

    @property
    def download_url(self) -> str:
        """Direct URL of the release tarball that carries this asset."""
        return f"{RELEASE_BASE_URL}/{self.release_asset}"


ASSETS: Final[tuple[Asset, ...]] = (
    Asset(
        asset_id="go1-mjcf",
        robot="Go1",
        format="mjcf",
        set_name="mjcf-v0",
        upstream_repo="google-deepmind/mujoco_menagerie",
        upstream_commit="370c43965323c0944b433b202aac5058b5f907a9",
        upstream_path="unitree_go1",
        license_id="BSD-3-Clause",
        license_file="LICENSES/Go1-BSD-3-Clause.txt",
        release_asset="unigym-robots-mjcf-v0.1.0.tar.gz",
        sha256="32577b386e2e01f8732649c5d7456cc55472b1402589d8932d1dbbbe28f97964",
        bytes=10_333_137,
        archive_path="go1/go1.xml",
        transformations=(
            "Copied the MuJoCo Menagerie unitree_go1 tree as-is.",
            "Removed the upstream PNG render (go1.png).",
            "Renamed the directory from unitree_go1/ to go1/.",
            "Added ORIGIN.json with source, license and per-file hashes.",
        ),
    ),
    Asset(
        asset_id="go2-mjcf",
        robot="Go2",
        format="mjcf",
        set_name="mjcf-v0",
        upstream_repo="google-deepmind/mujoco_menagerie",
        upstream_commit="370c43965323c0944b433b202aac5058b5f907a9",
        upstream_path="unitree_go2",
        license_id="BSD-3-Clause",
        license_file="LICENSES/Go2-BSD-3-Clause.txt",
        release_asset="unigym-robots-mjcf-v0.1.0.tar.gz",
        sha256="32577b386e2e01f8732649c5d7456cc55472b1402589d8932d1dbbbe28f97964",
        bytes=28_428_930,
        archive_path="go2/go2.xml",
        transformations=(
            "Copied the MuJoCo Menagerie unitree_go2 tree as-is.",
            "Dropped the MJX variant (go2_mjx.xml, scene_mjx.xml).",
            "Removed the upstream PNG renders (go2.png, go2_mjx.png).",
            "Renamed the directory from unitree_go2/ to go2/.",
            "Added ORIGIN.json with source, license and per-file hashes.",
        ),
    ),
    Asset(
        asset_id="g1-mjcf",
        robot="G1",
        format="mjcf",
        set_name="mjcf-v0",
        upstream_repo="google-deepmind/mujoco_menagerie",
        upstream_commit="370c43965323c0944b433b202aac5058b5f907a9",
        upstream_path="unitree_g1",
        license_id="BSD-3-Clause",
        license_file="LICENSES/G1-BSD-3-Clause.txt",
        release_asset="unigym-robots-mjcf-v0.1.0.tar.gz",
        sha256="32577b386e2e01f8732649c5d7456cc55472b1402589d8932d1dbbbe28f97964",
        bytes=35_987_049,
        archive_path="g1/g1.xml",
        transformations=(
            "Kept the approved 29-DoF g1.xml and its scene.xml.",
            "Dropped the hands MJCF (g1_with_hands.xml, scene_with_hands.xml).",
            "Dropped the MJX/MJLab variants (g1_mjx.xml, scene_mjx.xml, g1_mjlab.xml).",
            "Removed the upstream PNG renders (g1.png, g1_mjx_colliders.png, g1_with_hands.png).",
            "Renamed the directory from unitree_g1/ to g1/.",
            "Added ORIGIN.json with source, license and per-file hashes.",
        ),
    ),
    Asset(
        asset_id="go2-usd",
        robot="Go2",
        format="usd",
        set_name="usd-go2-v0",
        upstream_repo="unitreerobotics/unitree_model",
        upstream_commit="b6a8942b0803b6c137e58cef12beb4b03e4a2fa7",
        upstream_path="Go2/usd",
        license_id="BSD-3-Clause",
        license_file="LICENSES/Go2-BSD-3-Clause.txt",
        release_asset="unigym-robots-usd-go2-v0.1.0.tar.gz",
        sha256="f0c1df2e0766eb13d9ba500cc3c2841e02257578ebd277ed09f95ae5979ff09d",
        bytes=18_818_131,
        archive_path="go2/usd/go2.usd",
        transformations=(
            "Selected the approved Go2/usd tree at the pinned commit.",
            "Renamed the top-level directory from Go2/ to go2/.",
            "Added the upstream repository LICENSE at go2/LICENSE.",
            "Added ORIGIN.json with source, license and per-file hashes.",
        ),
    ),
    Asset(
        asset_id="g1-29dof-usd",
        robot="G1",
        format="usd",
        set_name="usd-g1-29dof-v0",
        upstream_repo="unitreerobotics/unitree_model",
        upstream_commit="b6a8942b0803b6c137e58cef12beb4b03e4a2fa7",
        upstream_path="G1/29dof/usd",
        license_id="BSD-3-Clause",
        license_file="LICENSES/G1-BSD-3-Clause.txt",
        release_asset="unigym-robots-usd-g1-29dof-v0.1.0.tar.gz",
        sha256="4526abafeb69e60c27a89c9aa5a8a421a13964d788032461e09bc38b304c4e44",
        bytes=28_353_150,
        archive_path="g1-29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        transformations=(
            "Selected the approved G1/29dof/usd tree at the pinned commit.",
            "Renamed the top-level directory from G1/ to g1-29dof/.",
            "Added the upstream repository LICENSE at g1-29dof/LICENSE.",
            "Added ORIGIN.json with source, license and per-file hashes.",
        ),
    ),
    Asset(
        asset_id="g1-43dof-usd",
        robot="G1",
        format="usd",
        set_name="usd-g1-43dof-v0",
        upstream_repo="unitreerobotics/unitree_sim_isaaclab_usds",
        upstream_commit="27b13d380531c0c6ea6ab109511ee4872594e271",
        upstream_path="assets/robots/g1-29dof_wholebody_dex3",
        license_id="Apache-2.0",
        license_file="LICENSES/G1-Apache-2.0.txt",
        release_asset="unigym-robots-usd-g1-43dof-v0.1.0.tar.gz",
        sha256="2174f20ec206518cd5b7118145e88b8b93c910e1f738ff948c2fbf83f57e26b8",
        bytes=81_158_317,
        archive_path="g1-43dof/g1_29dof_with_dex3_rev_1_0.usd",
        transformations=(
            "Selected only the approved 43-DoF asset directory from the HF dataset.",
            "Removed the HF dataset thumbnail PNG (.thumbs/).",
            "Flattened the assets/robots/g1-29dof_wholebody_dex3/ prefix to g1-43dof/.",
            "Kept the generated config.yaml for provenance.",
            "Added the Apache-2.0 license text declared by the dataset card.",
            "Added ORIGIN.json with source, license and per-file hashes.",
        ),
    ),
)

ASSETS_BY_ID: Final[dict[str, Asset]] = {asset.asset_id: asset for asset in ASSETS}


def get_asset(asset_id: str) -> Asset:
    """Return the manifest entry for ``asset_id``.

    Raises:
        KeyError: If the asset is not part of the approved six.
    """
    try:
        return ASSETS_BY_ID[asset_id]
    except KeyError:
        known = ", ".join(sorted(ASSETS_BY_ID))
        raise KeyError(f"unknown asset {asset_id!r}; known assets: {known}") from None
