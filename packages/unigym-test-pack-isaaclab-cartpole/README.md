# unigym-test-pack-isaaclab-cartpole

Content pack for the IsaacLab 2 Cartpole backend fixture.

The pack ships `assets/cartpole.usd` and its relative companion
`assets/Props/instanceable_meshes.usd`. It declares the USD as the
`isaaclab_cartpole.usd` asset resource; core tests resolve it through
`pack_runtime("isaaclab_cartpole")`.

The Cartpole asset is small enough to ship directly. G1 is a separate M2
large-asset decision.
