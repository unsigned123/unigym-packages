# Provenance

This file records the exact upstream sources used to build the
`unigym-robots-assets-v0.1.0` release. All commits are pinned; asset content is
copied byte-for-byte unless a transformation is listed.

Release: <https://github.com/unsigned123/unigym-packages/releases/tag/unigym-robots-assets-v0.1.0>

## 1. Go1 MJCF

| Field | Value |
| --- | --- |
| Asset id | `go1-mjcf` |
| Set | `mjcf-v0` |
| Upstream repository | <https://github.com/google-deepmind/mujoco_menagerie> |
| Commit | `370c43965323c0944b433b202aac5058b5f907a9` |
| Commit date | 2026-11-10 |
| Upstream path | <https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_go1> |
| Primary file | `go1.xml` (with `scene.xml` and `assets/*.stl`) |
| License | BSD-3-Clause, Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") |
| License copy | `LICENSES/Go1-BSD-3-Clause.txt` |

Transformations: copied the MJCF tree; removed `go1.png`; renamed
`unitree_go1/` to `go1/`; added `ORIGIN.json`.

## 2. Go2 MJCF

| Field | Value |
| --- | --- |
| Asset id | `go2-mjcf` |
| Set | `mjcf-v0` |
| Upstream repository | <https://github.com/google-deepmind/mujoco_menagerie> |
| Commit | `370c43965323c0944b433b202aac5058b5f907a9` |
| Commit date | 2026-11-10 |
| Upstream path | <https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_go2> |
| Primary file | `go2.xml` (with `scene.xml` and `assets/*.obj`) |
| License | BSD-3-Clause, Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") |
| License copy | `LICENSES/Go2-BSD-3-Clause.txt` |

Transformations: copied the MJCF tree; dropped the MJX variant
(`go2_mjx.xml`, `scene_mjx.xml`); removed `go2.png` and `go2_mjx.png`; renamed
`unitree_go2/` to `go2/`; added `ORIGIN.json`.

## 3. G1 29-DoF MJCF

| Field | Value |
| --- | --- |
| Asset id | `g1-mjcf` |
| Set | `mjcf-v0` |
| Upstream repository | <https://github.com/google-deepmind/mujoco_menagerie> |
| Commit | `370c43965323c0944b433b202aac5058b5f907a9` |
| Commit date | 2026-11-10 |
| Upstream path | <https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_g1> |
| Primary file | `g1.xml` (29 actuators, with `scene.xml` and `assets/*`) |
| License | BSD-3-Clause, Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") |
| License copy | `LICENSES/G1-BSD-3-Clause.txt` |

Transformations: kept the 29-DoF `g1.xml` and `scene.xml`; dropped the hands
model (`g1_with_hands.xml`, `scene_with_hands.xml`); dropped the MJX/MJLab
variants (`g1_mjx.xml`, `scene_mjx.xml`, `g1_mjlab.xml`); removed PNG renders;
renamed `unitree_g1/` to `g1/`; added `ORIGIN.json`. The upstream `assets/`
directory is copied in full, including hand meshes that the 29-DoF
`g1.xml` does not reference.

## 4. Go2 USD

| Field | Value |
| --- | --- |
| Asset id | `go2-usd` |
| Set | `usd-go2-v0` |
| Upstream repository | <https://github.com/unitreerobotics/unitree_model> |
| Commit | `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7` |
| Commit date | 2026-06-01 |
| Upstream path | <https://github.com/unitreerobotics/unitree_model/tree/b6a8942b0803b6c137e58cef12beb4b03e4a2fa7/Go2/usd> |
| Primary file | `go2.usd` (with `configuration/go2_description_*.usd`) |
| License | BSD-3-Clause, Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") |
| License copy | `LICENSES/Go2-BSD-3-Clause.txt`; upstream `LICENSE` also included in the tarball |

Transformations: selected `Go2/usd`; renamed the top-level directory from
`Go2/` to `go2/`; added the upstream `LICENSE` and `ORIGIN.json`. USD layer
contents are unchanged.

## 5. G1 29-DoF USD

| Field | Value |
| --- | --- |
| Asset id | `g1-29dof-usd` |
| Set | `usd-g1-29dof-v0` |
| Upstream repository | <https://github.com/unitreerobotics/unitree_model> |
| Commit | `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7` |
| Commit date | 2026-06-01 |
| Upstream path | <https://github.com/unitreerobotics/unitree_model/tree/b6a8942b0803b6c137e58cef12beb4b03e4a2fa7/G1/29dof/usd> |
| Primary file | `g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd` (with `configuration/g1_29dof_rev_1_0_{base,physics,sensor}.usd`) |
| License | BSD-3-Clause, Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") |
| License copy | `LICENSES/G1-BSD-3-Clause.txt`; upstream `LICENSE` also included in the tarball |

Transformations: selected `G1/29dof/usd`; renamed the top-level directory from
`G1/` to `g1-29dof/`; added the upstream `LICENSE` and `ORIGIN.json`. USD layer
contents are unchanged.

## 6. G1 43-DoF USD (whole-body with Dex3 hands)

| Field | Value |
| --- | --- |
| Asset id | `g1-43dof-usd` |
| Set | `usd-g1-43dof-v0` |
| Upstream repository | <https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds> |
| Commit | `27b13d380531c0c6ea6ab109511ee4872594e271` |
| Commit date | 2026-11-11 |
| Upstream path | <https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds/tree/27b13d380531c0c6ea6ab109511ee4872594e271/assets/robots/g1-29dof_wholebody_dex3> |
| Primary file | `g1_29dof_with_dex3_rev_1_0.usd` (with `configuration/g1_29dof_with_dex3_rev_1_0_{base,physics,sensor}.usd`) |
| License | Apache-2.0, declared by the dataset card (`license: apache-2.0`) |
| License copy | `LICENSES/G1-Apache-2.0.txt` |

Transformations: selected only the approved asset directory; removed the HF
dataset thumbnail PNG (`.thumbs/`); flattened
`assets/robots/g1-29dof_wholebody_dex3/` to `g1-43dof/`; kept the generated
`config.yaml` for provenance; added the Apache-2.0 text and `ORIGIN.json`. USD
layer contents are unchanged.

The dataset repository does not contain a separate `LICENSE` file at this
revision; the Apache-2.0 declaration is in the dataset card (`README.md`).

Note: the upstream directory name says `29dof`, while the file includes the
Dex3 hands and is referred to as the 43-DoF G1 asset in this project. The name
is preserved in `ORIGIN.json` for traceability.

## Release tarballs

| Set | Tarball | Bytes | sha256 |
| --- | --- | ---: | --- |
| `mjcf-v0` | `unigym-robots-mjcf-v0.1.0.tar.gz` | 23500205 | `32577b386e2e01f8732649c5d7456cc55472b1402589d8932d1dbbbe28f97964` |
| `usd-go2-v0` | `unigym-robots-usd-go2-v0.1.0.tar.gz` | 6045544 | `f0c1df2e0766eb13d9ba500cc3c2841e02257578ebd277ed09f95ae5979ff09d` |
| `usd-g1-29dof-v0` | `unigym-robots-usd-g1-29dof-v0.1.0.tar.gz` | 8219295 | `4526abafeb69e60c27a89c9aa5a8a421a13964d788032461e09bc38b304c4e44` |
| `usd-g1-43dof-v0` | `unigym-robots-usd-g1-43dof-v0.1.0.tar.gz` | 22987993 | `2174f20ec206518cd5b7118145e88b8b93c910e1f738ff948c2fbf83f57e26b8` |

## Build and verification

Tarballs were built from a clean checkout at the pinned commits with sorted
member names, `mtime=0`, `uid=gid=0`, empty owner names and `gzip mtime=0`, so
rebuilds are byte-stable for the same inputs.

- MJCF load check: MuJoCo 4.0.0 via `mujoco.MjModel.from_xml_path` on
  `go1.xml`, `go2.xml`, `g1.xml` and their `scene.xml` variants.
  `go1`: nq=19, nv=18, nu=12, nmesh=5; `go2`: nq=19, nv=18, nu=12, nmesh=16;
  `g1`: nq=36, nv=35, nu=29, nmesh=35.
- USD: stages opened successfully with OpenUSD 0.24.5 (the build bundled with
  Isaac Sim 5.1) via `pxr.Usd.Stage.Open`: `go2.usd` (default prim
  `/go2_description`, 115 prims), `g1_29dof_rev_1_0.usd` (default prim
  `/g1_29dof_rev_1_0`, 139 prims) and
  `g1_29dof_with_dex3_rev_1_0.usd` (default prim
  `/g1_29dof_with_hand_rev_1_0`, 330 prims). IsaacLab spawning was **not** run
  and remains unverified; layer contents were not modified.
- Every downloaded tarball is sha256-checked by `download_set` before
  extraction, and `ORIGIN.json` records per-file hashes for auditing.

## Excluded upstream content

- NVIDIA Isaac Sim / Nucleus assets, Unitree PNG renders and preview images,
  MJX/MJLab MJCF variants, the G1 hands MJCF, and test packs are not included.
- No NVIDIA-licensed or Isaac Sim proprietary content is redistributed here.
