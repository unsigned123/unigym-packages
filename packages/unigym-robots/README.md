# unigym-robots

Pinned Unitree robot assets for [`unigym`](https://github.com/unsigned123/unigym),
distributed as GitHub Release tarballs instead of wheel binaries.

This package contains **no** meshes, MJCF, USD or other robot binaries. It ships
a manifest, a small downloader and a CLI. The assets themselves are attached to
the [`unigym-robots-assets-v0.1.0`](https://github.com/unsigned123/unigym-packages/releases/tag/unigym-robots-assets-v0.1.0)
GitHub Release, with pinned sha256 hashes.

## Install

```bash
pip install unigym-robots
```

Python 3.11 or newer. There are no runtime dependencies.

## Usage

List the sets and assets without touching the network:

```bash
unigym-robots list
unigym-robots info go2-usd
unigym-robots info mjcf-v0 --json
```

Download and extract one set:

```bash
unigym-robots download --set mjcf-v0
unigym-robots download --set usd-go2-v0
unigym-robots download --asset g1-43dof-usd
unigym-robots download --all
```

Files are extracted to `~/.cache/unigym/robots` by default. Override with
`--root` or the `UNIGYM_ROBOTS_ROOT` environment variable:

```bash
UNIGYM_ROBOTS_ROOT=/data/assets unigym-robots download --set mjcf-v0
```

Python API:

```python
from unigym_robots import asset_path, download_asset, download_set

install_dir = download_set("mjcf-v0")
model_path = asset_path("g1-mjcf")          # .../mjcf-v0.1.0/g1/g1.xml
usd_path = download_asset("g1-43dof-usd") / "g1-43dof/g1_29dof_with_dex3_rev_1_0.usd"
```

`download_set` verifies the tarball sha256 before extraction and uses a
restricted tar extraction (regular files and directories only, no path
traversal, no links). Existing installs are reused unless `--force` is passed.

## Pack profiles

The package installs an `unigym.packs` entry point, so `require` can see it
directly:

```python
import unigym.packs

packs = unigym.packs.require("unigym_robots", entry_points=True)
go2 = packs.pack("unigym_robots").robot("go2")   # RobotProfile
go2.assets.mjcf                                  # "release:go2-mjcf"
go2.asset_backends                               # {"mjcf": ("mujoco", "mjwarp"), "usd": ("isaaclab2",)}
go2.actuation[0].joint_set                       # "body"
```

`pack.robot(id)` loads the complete profile from `pack/robots/<id>.yaml`
without activating the pack or mutating the robot registry. Profiles carry
joint sets, the standby pose, `release:` asset references (never local
binaries), per-format `asset_backends` allow-lists, body bindings and
actuation groups. Building a world remains
explicit: these profiles do not construct one. Without an installed entry
point, discover the shipped root with
`unigym.packs.discover_set([unigym_robots.pack_root()])`.

## Approved assets

| Asset id | Robot | Format | Set | Upstream source |
| --- | --- | --- | --- | --- |
| `go1-mjcf` | Go1 | MJCF | `mjcf-v0` | MuJoCo Menagerie `unitree_go1` @ `370c439` |
| `go2-mjcf` | Go2 | MJCF | `mjcf-v0` | MuJoCo Menagerie `unitree_go2` @ `370c439` |
| `g1-mjcf` | G1 29-DoF | MJCF | `mjcf-v0` | MuJoCo Menagerie `unitree_g1` @ `370c439` (`g1.xml`) |
| `go2-usd` | Go2 | USD | `usd-go2-v0` | `unitreerobotics/unitree_model` @ `b6a8942`, `Go2/usd/` |
| `g1-29dof-usd` | G1 29-DoF | USD | `usd-g1-29dof-v0` | `unitreerobotics/unitree_model` @ `b6a8942`, `G1/29dof/usd/` |
| `g1-43dof-usd` | G1 43-DoF | USD | `usd-g1-43dof-v0` | HF `unitree_sim_isaaclab_usds` @ `27b13d3`, `assets/robots/g1-29dof_wholebody_dex3/` |

Release tarballs:

| Set | Tarball | Compressed size |
| --- | --- | --- |
| `mjcf-v0` | `unigym-robots-mjcf-v0.1.0.tar.gz` | ~23 MB |
| `usd-go2-v0` | `unigym-robots-usd-go2-v0.1.0.tar.gz` | ~6 MB |
| `usd-g1-29dof-v0` | `unigym-robots-usd-g1-29dof-v0.1.0.tar.gz` | ~8 MB |
| `usd-g1-43dof-v0` | `unigym-robots-usd-g1-43dof-v0.1.0.tar.gz` | ~22 MB |

Each asset directory contains an `ORIGIN.json` recording the upstream
repository, commit, path, license, transformations and a sha256 for every
extracted file.

## Provenance and licensing

- MuJoCo Menagerie Go1, Go2 and G1 files are BSD-3-Clause, copyright Unitree
  Robotics. See `LICENSES/Go1-BSD-3-Clause.txt`, `Go2-BSD-3-Clause.txt` and
  `G1-BSD-3-Clause.txt`.
- `unitreerobotics/unitree_model` Go2 and G1 USD files are BSD-3-Clause,
  copyright Unitree Robotics. Each USD tarball also carries its upstream
  `LICENSE` file.
- The Hugging Face dataset `unitree_sim_isaaclab_usds` declares Apache-2.0. See
  `LICENSES/G1-Apache-2.0.txt`.

The full source URLs, commits and modifications are in
[`PROVENANCE.md`](PROVENANCE.md); copyright and redistribution notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The package code in this repository is MIT licensed. The bundled upstream
assets remain under their own licenses.

### Trademarks

Unitree, Go1, Go2 and G1 are trademarks or product names of HangZhou YuShu
TECHNOLOGY CO., LTD. ("Unitree Robotics"). They are used here only to identify
the robot models whose assets are redistributed. This project is not affiliated
with, sponsored by or endorsed by Unitree Robotics. MuJoCo and MuJoCo Menagerie
are projects of Google DeepMind; this project is not affiliated with or endorsed
by Google DeepMind either.

## Not on PyPI as assets

The code (`unigym-robots`) is a normal Python distribution, but the robot
assets are **not** published to PyPI and are never included in a wheel or sdist.
They are GitHub Release artifacts only. No binaries are tracked in git.

## Development

```bash
python -m pytest packages/unigym-robots/tests -q
python -m ruff check packages/unigym-robots/src packages/unigym-robots/tests
python -m ruff format --check packages/unigym-robots/src packages/unigym-robots/tests
```

Tests are offline; they never contact GitHub or the network.
