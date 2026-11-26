from __future__ import annotations

from pathlib import Path

import pytest


def _write_pack(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        f"name: {name}\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides: {}\n",
        encoding="utf-8",
    )


def test_fixtures_read_installed_pack_and_runtime(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_consumer="""
        import pytest
        from unigym.pack_api import load_manifest_v1

        pytestmark = pytest.mark.requires_test_packs("basic")

        def test_installed_pack(pack_root, pack_runtime):
            root = pack_root("basic")
            assert load_manifest_v1(root).name == "basic"
            runtime = pack_runtime("basic")
            assert tuple(pack.name for pack in runtime.pack_set.packs()) == ("basic",)
            assert runtime is pack_runtime("basic")
            assert runtime.resolve("providers", "basic_arm").is_file()
        """
    )

    result = pytester.runpytest_subprocess("-q")

    result.assert_outcomes(passed=1)


def test_missing_pack_skips_by_default(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_missing="""
        import pytest

        pytestmark = pytest.mark.requires_test_packs("missing")

        def test_needs_missing_pack():
            raise AssertionError("must not execute")
        """
    )

    result = pytester.runpytest_subprocess("-q", "-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*missing test pack(s): missing*"])


def test_missing_pack_fails_under_strict_mode(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_missing="""
        import pytest

        pytestmark = pytest.mark.requires_test_packs("missing")

        def test_needs_missing_pack():
            raise AssertionError("must not execute")
        """
    )

    result = pytester.runpytest_subprocess("-q", "--strict-test-packs")

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*missing test pack(s): missing*"])


def test_undeclared_fixture_request_fails(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_undeclared="""
        import pytest

        pytestmark = pytest.mark.requires_test_packs("basic")

        def test_uses_undeclared_pack(pack_root):
            pack_root("missing")
        """
    )

    result = pytester.runpytest_subprocess("-q")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*not declared*"])


def test_marker_without_arguments_fails(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_marker="""
        import pytest

        pytestmark = pytest.mark.requires_test_packs()

        def test_marker():
            pass
        """
    )

    result = pytester.runpytest_subprocess("-q")

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*requires at least one pack name*"])


def test_duplicate_markers_fail(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_marker="""
        import pytest

        pytestmark = [
            pytest.mark.requires_test_packs("basic"),
            pytest.mark.requires_test_packs("basic"),
        ]

        def test_marker():
            pass
        """
    )

    result = pytester.runpytest_subprocess("-q")

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*may appear at most once*"])


def test_marker_selects_all_pack_dependent_tests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_selection="""
        import pytest

        @pytest.mark.requires_test_packs("basic")
        def test_pack_marked():
            pass

        def test_unmarked():
            raise AssertionError("must be deselected")
        """
    )

    result = pytester.runpytest_subprocess("-q", "-m", "requires_test_packs")

    result.assert_outcomes(passed=1)


def test_plugin_ignores_pack_environment(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_env_root = pytester.path / "fake_env_pack"
    _write_pack(fake_env_root, "basic")
    (fake_env_root / "poison.txt").write_text("ambient", encoding="utf-8")

    deps_root = pytester.path / "deps"
    _write_pack(deps_root / "unigym_packs" / "poison", "poison")

    monkeypatch.setenv("UNIGYM_PACKS", str(fake_env_root))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(deps_root))
    pytester.makepyfile(
        test_env="""
        import pytest

        pytestmark = pytest.mark.requires_test_packs("basic")

        def test_installed_only(pack_root, pack_runtime):
            root = pack_root("basic")
            assert not (root / "poison.txt").exists()
            assert tuple(pack.name for pack in pack_runtime("basic").pack_set.packs()) == ("basic",)
        """
    )

    result = pytester.runpytest_subprocess("-q")

    result.assert_outcomes(passed=1)
