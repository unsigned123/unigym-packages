# unigym-test-packs

Pytest plugin for installed `unigym` test packs.

Pack distributions expose one entry point under the `unigym.test_packs` group.
The entry-point name is the pack name and its target is the pack root path or a
zero-argument callable returning it.

The plugin provides the `requires_test_packs` marker, the `pack_root(name)` and
`pack_runtime(*names)` fixtures, and the `--strict-test-packs` switch. It reads
only installed entry points; `UNIGYM_PACKS` and `UNIGYM_DEPS_ROOT` are ignored.
