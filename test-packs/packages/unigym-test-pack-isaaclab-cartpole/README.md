# unigym-test-pack-isaaclab-cartpole

Content pack for the IsaacLab 2 Cartpole backend fixture.

The pack ships `assets/cartpole.usd` and its relative companion
`assets/Props/instanceable_meshes.usd`. It declares the USD as the
`isaaclab_cartpole.usd` asset resource; core tests resolve it through the
`pack_runtime("isaaclab_cartpole")` fixture (a core `PackLoader`).

The Cartpole asset is small enough to ship directly. G1 is a separate M2
large-asset decision.

## Asset-scene fixtures

The pack also carries the tiny file-backed asset fixtures used by the
IsaacLab asset-scene dog lane:

- `assets/static_cup.usda`: one collision-authored cube, no
  `PhysicsRigidBodyAPI` and no `ArticulationRootAPI` (a static asset);
- `assets/dynamic_cube.usda`: one `PhysicsRigidBodyAPI` cube with collision
  and mass (a passive dynamic rigid body);
- `assets/jointless_cube.urdf`: one link with inertial, collision and visual
  data and no `<joint>` (the passive-rigid-body URDF route);
- `assets/jointed_cube.urdf`: two links joined by one `<joint>` (the
  articulation-in-scene refusal fixture).

They are declared as `assets` resources and resolve through
`pack_runtime("isaaclab_cartpole").resolve("assets", <id>)`. The tests
never download assets and never read an asset environment variable.

