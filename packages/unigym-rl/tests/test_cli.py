"""R2.6 tests for the optional :mod:`unigym_rl.cli` option sugar."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from unigym_rl.cli import EnvOptions, Options, TrainOptions, parse_args


def test_defaults_from_empty_argv() -> None:
    opts = parse_args([])

    assert opts.env == EnvOptions(num_envs=8, max_episode_steps=500, device="cpu", seed=0)
    assert opts.train == TrainOptions(max_iterations=1000, rollout_steps=64)


def test_hyphenated_flags_override_every_field() -> None:
    opts = parse_args(
        [
            "--num-envs",
            "16",
            "--max-episode-steps",
            "250",
            "--device",
            "cuda:0",
            "--seed",
            "-5",
            "--max-iterations",
            "20",
            "--rollout-steps",
            "32",
        ]
    )

    assert opts.env == EnvOptions(num_envs=16, max_episode_steps=250, device="cuda:0", seed=-5)
    assert opts.train == TrainOptions(max_iterations=20, rollout_steps=32)


def test_underscore_aliases_match_the_hyphenated_flags() -> None:
    hyphenated = parse_args(
        [
            "--num-envs",
            "16",
            "--max-episode-steps",
            "250",
            "--max-iterations",
            "20",
            "--rollout-steps",
            "32",
        ]
    )
    underscored = parse_args(
        [
            "--num_envs",
            "16",
            "--max_episode_steps",
            "250",
            "--max_iterations",
            "20",
            "--rollout_steps",
            "32",
        ]
    )

    assert underscored == hyphenated


def test_supplied_argv_is_used_instead_of_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--num-envs", "999", "--device", "cuda:9"])

    opts = parse_args(["--num-envs", "3"])

    assert opts.env.num_envs == 3
    assert opts.env.device == "cpu"


def test_none_argv_falls_back_to_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--seed", "-7"])

    assert parse_args().env.seed == -7


def test_unknown_argument_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--not-a-flag"])


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--num-envs", "0"),
        ("--num-envs", "-3"),
        ("--num-envs", "eight"),
        ("--num-envs", "1.5"),
        ("--max-episode-steps", "0"),
        ("--max-episode-steps", "-3"),
        ("--max-episode-steps", "eight"),
        ("--max-episode-steps", "1.5"),
        ("--max-iterations", "0"),
        ("--max-iterations", "-3"),
        ("--max-iterations", "eight"),
        ("--max-iterations", "1.5"),
        ("--rollout-steps", "0"),
        ("--rollout-steps", "-3"),
        ("--rollout-steps", "eight"),
        ("--rollout-steps", "1.5"),
    ],
)
def test_positive_int_flags_reject_bad_text(flag: str, bad_value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([flag, bad_value])


def test_empty_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="device"):
        parse_args(["--device", ""])

    with pytest.raises(ValueError, match="device"):
        EnvOptions(device="")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("num_envs", 0),
        ("num_envs", -1),
        ("max_episode_steps", 0),
        ("max_episode_steps", -1),
    ],
)
def test_env_options_reject_non_positive_ints(field: str, bad_value: int) -> None:
    with pytest.raises(ValueError, match=field):
        EnvOptions(**{field: bad_value})


@pytest.mark.parametrize("field", ["max_iterations", "rollout_steps"])
@pytest.mark.parametrize("bad_value", [0, -1])
def test_train_options_reject_non_positive_ints(field: str, bad_value: int) -> None:
    with pytest.raises(ValueError, match=field):
        TrainOptions(**{field: bad_value})


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("num_envs", "8"),
        ("num_envs", True),
        ("max_episode_steps", 8.0),
        ("seed", "0"),
        ("device", 0),
    ],
)
def test_env_options_reject_wrong_types(field: str, bad_value: object) -> None:
    with pytest.raises(ValueError, match=field):
        EnvOptions(**{field: bad_value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("max_iterations", "20"), ("rollout_steps", None)],
)
def test_train_options_reject_wrong_types(field: str, bad_value: object) -> None:
    with pytest.raises(ValueError, match=field):
        TrainOptions(**{field: bad_value})  # type: ignore[arg-type]


def test_options_are_frozen() -> None:
    opts = parse_args([])

    with pytest.raises(FrozenInstanceError):
        opts.env.num_envs = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        opts.train.rollout_steps = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        opts.env = EnvOptions()  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        EnvOptions().device = "cuda:0"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        TrainOptions().max_iterations = 1  # type: ignore[misc]


def test_options_reject_wrong_parts() -> None:
    with pytest.raises(ValueError, match="env"):
        Options(env=object(), train=TrainOptions())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train"):
        Options(env=EnvOptions(), train=object())  # type: ignore[arg-type]


def test_importing_package_does_not_import_cli() -> None:
    code = "import sys, unigym_rl; print('unigym_rl.cli' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_importing_cli_does_not_import_rl_frameworks() -> None:
    code = (
        "import sys, unigym_rl.cli; "
        "print(sorted(m for m in ('gymnasium', 'skrl') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"
