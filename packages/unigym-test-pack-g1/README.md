# unigym-test-pack-g1

Content pack for the 43-DoF G1 IsaacLab fixture.

The pack ships the complete `g1.usd` root and its `configuration/**` companion
tree: 20 files, about 77.2MB on disk and about 22MB compressed. It also declares
the data-only `g1` robot provider. The wheel is the selective distribution
boundary; every environment that installs this pack gets the complete tree, and
tests select declared resources through `pack_runtime("g1")`.

The pack is intentionally not split by hand variant. G1 is a test-only asset
fixture; splitting the shared USD closure would duplicate large files.
