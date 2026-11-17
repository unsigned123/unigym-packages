# Third-party notices

The `unigym-robots` Python package is MIT licensed. The robot assets it
downloads are third-party works and remain under the licenses below. No
third-party asset binaries are included in the package wheel or sdist; they are
distributed only as GitHub Release tarballs.

## MuJoCo Menagerie — Go1, Go2, G1 (MJCF)

- Upstream: <https://github.com/google-deepmind/mujoco_menagerie>
- Files: `unitree_go1/`, `unitree_go2/`, `unitree_g1/` at commit
  `370c43965323c0944b433b202aac5058b5f907a9`
- Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree
  Robotics") for Go1 and Go2
- Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree
  Robotics") for G1
- License: BSD-3-Clause
- License texts: `LICENSES/Go1-BSD-3-Clause.txt`,
  `LICENSES/Go2-BSD-3-Clause.txt`, `LICENSES/G1-BSD-3-Clause.txt`

MuJoCo Menagerie is maintained by Google DeepMind. Its repository-level license
does not replace the per-model license retained here.

## unitreerobotics/unitree_model — Go2 and G1 29-DoF (USD)

- Upstream: <https://github.com/unitreerobotics/unitree_model>
- Files: `Go2/usd/` and `G1/29dof/usd/` at commit
  `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7`
- Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree
  Robotics")
- License: BSD-3-Clause
- License text: `LICENSES/Go2-BSD-3-Clause.txt` and
  `LICENSES/G1-BSD-3-Clause.txt`; the upstream `LICENSE` is also included in
  each USD tarball next to the asset tree.

## unitree_sim_isaaclab_usds — G1 43-DoF / Dex3 (USD)

- Upstream:
  <https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds>
- Files: `assets/robots/g1-29dof_wholebody_dex3/` at commit
  `27b13d380531c0c6ea6ab109511ee4872594e271`
- License: Apache-2.0, as declared by the dataset card
  (`license: apache-2.0`). The dataset repository has no separate `LICENSE`
  file at this revision.
- License text: `LICENSES/G1-Apache-2.0.txt`

Copyright in the converted USD files is held by their upstream authors. The
Apache-2.0 text is reproduced in full; no additional copyright holder is
asserted by this project.

## Notes on consolidated copyright lines

`LICENSES/Go2-BSD-3-Clause.txt` and `LICENSES/G1-BSD-3-Clause.txt` list the
copyright notices from both upstream sources for that robot (MuJoCo Menagerie
and Unitree Robotics). The license terms are unchanged; only the copyright
lines were consolidated so that every upstream notice is retained.

## Trademarks

Unitree, Go1, Go2 and G1 are trademarks or product names of HangZhou YuShu
TECHNOLOGY CO., LTD. ("Unitree Robotics"). This project uses those names only
descriptively, to identify the robot models whose assets are redistributed. It
is not affiliated with, sponsored by or endorsed by Unitree Robotics. MuJoCo and
MuJoCo Menagerie are projects of Google DeepMind; this project is not
affiliated with or endorsed by Google DeepMind.

NVIDIA Isaac Sim and Isaac Lab are trademarks of NVIDIA Corporation. No NVIDIA
Isaac Sim/Nucleus assets are redistributed by this project.
