"""Optional CLI option sugar (R2.6).

Turns a small, explicit flag set into frozen option records.  This module is
sugar only: it builds no ``EnvSpec``, ``World``, robot, task, or trainer, and
the package root never imports it.  Import it explicitly::

    from unigym_rl.cli import parse_args

    opts = parse_args()  # or parse_args(["--num-envs", "4"])
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

__all__ = ["EnvOptions", "Options", "TrainOptions", "parse_args"]


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value}")
    return value


def _require_int_ge_one(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


@dataclass(frozen=True)
class EnvOptions:
    """Environment-facing CLI options, consumed by ``EnvSpec`` and the task."""

    num_envs: int = 8
    max_episode_steps: int = 500
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        _require_int_ge_one(self.num_envs, "EnvOptions.num_envs")
        _require_int_ge_one(self.max_episode_steps, "EnvOptions.max_episode_steps")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"EnvOptions.device must be a non-empty string, got {self.device!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"EnvOptions.seed must be an int, got {type(self.seed).__name__}")


@dataclass(frozen=True)
class TrainOptions:
    """Training-loop options reserved for a future trainer."""

    max_iterations: int = 1000
    rollout_steps: int = 64

    def __post_init__(self) -> None:
        _require_int_ge_one(self.max_iterations, "TrainOptions.max_iterations")
        _require_int_ge_one(self.rollout_steps, "TrainOptions.rollout_steps")


@dataclass(frozen=True)
class Options:
    """The frozen handoff point between CLI parsing and run construction."""

    env: EnvOptions
    train: TrainOptions

    def __post_init__(self) -> None:
        if not isinstance(self.env, EnvOptions):
            raise ValueError(f"Options.env must be EnvOptions, got {type(self.env).__name__}")
        if not isinstance(self.train, TrainOptions):
            raise ValueError(f"Options.train must be TrainOptions, got {type(self.train).__name__}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unigym-rl",
        description="Parse unigym-rl run options (sugar only; no trainer).",
    )
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=_positive_int,
        default=8,
        metavar="N",
        help="number of vectorized environments (default: 8)",
    )
    parser.add_argument(
        "--max-episode-steps",
        "--max_episode_steps",
        dest="max_episode_steps",
        type=_positive_int,
        default=500,
        metavar="N",
        help="per-episode truncation budget (default: 500)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help='torch device string, e.g. "cpu" or "cuda:0" (default: cpu)',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="base seed for run construction (default: 0)",
    )
    parser.add_argument(
        "--max-iterations",
        "--max_iterations",
        dest="max_iterations",
        type=_positive_int,
        default=1000,
        metavar="N",
        help="training iterations reserved for a future trainer (default: 1000)",
    )
    parser.add_argument(
        "--rollout-steps",
        "--rollout_steps",
        dest="rollout_steps",
        type=_positive_int,
        default=64,
        metavar="N",
        help="rollout length reserved for a future trainer (default: 64)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> Options:
    """Parse *argv* into frozen :class:`Options`.

    ``argv=None`` means ``sys.argv[1:]``.  A supplied list is used verbatim and
    is never merged with ``sys.argv``.  Unknown flags are rejected by
    ``argparse`` (``SystemExit``); ``parse_args`` itself never calls
    ``sys.exit``.
    """
    args = _build_parser().parse_args(argv)
    return Options(
        env=EnvOptions(
            num_envs=args.num_envs,
            max_episode_steps=args.max_episode_steps,
            device=args.device,
            seed=args.seed,
        ),
        train=TrainOptions(
            max_iterations=args.max_iterations,
            rollout_steps=args.rollout_steps,
        ),
    )
