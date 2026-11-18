# Test fixtures

`tiny_arm.xml` is a self-contained MuJoCo primitives-only model copied
byte-for-byte from the MIT-licensed `unigym-core` repository:

```text
unigym-core/tests/_support/data/tiny_arm.xml
```

It contains no mesh, texture, heightfield, include, or network reference. It
is used only by the `unigym-rl` manual-rollout test lane.
