"""The user-facing L2 task environment contract (R0).

``TaskEnv`` sits directly on the core ``ExternalRunner``: it owns the
Gymnasium-shaped ``reset`` / ``step`` driver and nothing else.  Subclasses
provide task logic only (observation, reward, termination), never the loop,
never framework glue.

R0 is deliberately strict:

* the run must declare ``termination_mode: external``;
* the caller must own reset policy (``auto_reset: false`` and
  ``reset_on_termination: false``);
* observations are flat ``(num_envs, obs_dim)`` tensors;
* actions are flat ``(num_envs, action_dim)`` tensors and are converted to
  joint-addressed ``NamedAction`` objects by ``OrderedActionSpec.unflatten``;
* ``reset`` must be called before the first ``step``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import torch
from unigym.runner.state import read_extra

from unigym_rl.context import EpisodeEnd, TaskContext, TaskState, Transition
from unigym_rl.errors import TaskConfigError, TaskContractError, TaskEnvError

if TYPE_CHECKING:
    from unigym.run.api import Run
    from unigym.runtime.actions import OrderedActionSpec

__all__ = ["TaskEnv"]


class Buffer(Protocol):
    """The only surface ``TaskEnv.register_buffer`` requires."""

    def reset(self, env_ids: torch.Tensor | None = None) -> None: ...


def _resolve_control_dt(spec: object, instance: object) -> float:
    """Resolve the policy period from the run's bindings, then the adapter."""
    from unigym.runtime.actions import resolve_control_dt as resolve

    adapter = getattr(instance, "adapter", None)
    fallback = getattr(adapter, "control_dt", None)
    policies = tuple(getattr(spec, "policies", ()) or ())
    name = getattr(adapter, "_unigym_name", None) or getattr(spec, "adapter_name", None)
    if not policies or not name:
        resolved = fallback
    else:
        resolved = resolve(policies, str(name), fallback)
    if resolved is None:
        raise TaskConfigError(
            "TaskEnv: no control period is declared; set a policy binding "
            "control_dt or the adapter control_dt"
        )
    try:
        value = float(resolved)
    except (TypeError, ValueError) as exc:
        raise TaskConfigError(f"TaskEnv: control_dt must be a number, got {resolved!r}") from exc
    if value <= 0.0:
        raise TaskConfigError(f"TaskEnv: control_dt must be > 0, got {value!r}")
    return value


def _empty_ids(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(0, dtype=torch.int64, device=device)


class TaskEnv:
    """Base class for a task-owned, externally-terminated vector environment.

    Subclasses implement :meth:`observe` and :meth:`reward`; every other hook
    has a safe default.  See the module docstring for the R0 contract.
    """

    #: Observations are flat; the dimension may be declared here or inferred
    #: from the first ``observe`` output.
    obs_dim: int | None = None

    def __init__(
        self,
        run: Run,
        *,
        action_spec: OrderedActionSpec | None = None,
    ) -> None:
        spec = getattr(run, "spec", None)
        if spec is None:
            raise TaskConfigError("TaskEnv requires a run exposing .spec")
        mode = getattr(spec, "termination_mode", None)
        if mode != "external":
            raise TaskConfigError(
                "TaskEnv requires termination_mode: external so the task owns "
                f"terminated/truncated masks; got termination_mode={mode!r} | "
                "hint: build the Run with termination_mode='external'"
            )
        runner_cfg = getattr(spec, "runner", {}) or {}
        if not isinstance(runner_cfg, Mapping):
            raise TaskConfigError(
                f"TaskEnv: spec.runner must be a mapping, got {type(runner_cfg).__name__}"
            )
        if runner_cfg.get("auto_reset") is not False:
            raise TaskConfigError(
                "TaskEnv requires runner.auto_reset: false (the task layer "
                f"resets done ids); got runner.auto_reset={runner_cfg.get('auto_reset')!r}"
            )
        if runner_cfg.get("reset_on_termination", False) is not False:
            raise TaskConfigError(
                "TaskEnv requires runner.reset_on_termination: false; got "
                f"{runner_cfg.get('reset_on_termination')!r}"
            )

        self.run = run
        self.instance = run.build()
        self.runner = run.external()
        self.env = self.instance.env
        self.controls = getattr(self.env, "controls", {}) or {}
        if not isinstance(self.controls, Mapping):
            raise TaskConfigError(
                f"TaskEnv: env.controls must be a mapping, got {type(self.controls).__name__}"
            )

        self.num_envs = int(self.env.num_envs)
        if self.num_envs < 1:
            raise TaskConfigError(f"TaskEnv: num_envs must be >= 1, got {self.num_envs}")
        self.device = torch.device(self.env.device)

        resolved_spec = (
            action_spec if action_spec is not None else getattr(spec, "action_spec", None)
        )
        if resolved_spec is None:
            raise TaskConfigError(
                "TaskEnv: no action_spec is available; pass action_spec=... or "
                "declare action_spec on the run"
            )
        self.action_spec = resolved_spec
        self.action_dim = int(self.action_spec.dim)

        self.control_dt = _resolve_control_dt(spec, self.instance)
        self.context = TaskContext(
            run=run,
            instance=self.instance,
            env=self.env,
            controls=dict(self.controls),
            action_spec=self.action_spec,
            num_envs=self.num_envs,
            device=self.device,
            control_dt=self.control_dt,
        )

        self._last_obs: torch.Tensor | None = None
        self._closed = False
        self._started = False
        self._buffers: list[Buffer] = []
        self.on_bind(self.context)

    # ------------------------------------------------------------------ #
    # final API (users do not override)
    # ------------------------------------------------------------------ #
    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Reset every env and return ``(obs, {"state": TaskState})``.

        ``seed`` and ``options`` are accepted for the Gymnasium-shaped
        signature.  R0 has no core seeding seam, so both are currently
        ignored rather than silently faked.
        """
        del seed, options
        self._ensure_open()
        result = self.runner.reset(None)
        all_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        state = self._snapshot(t=int(result.t))
        self._reset_buffers(all_ids)
        self.on_reset(all_ids, state)
        obs = self._validate_obs(self.observe(state))
        self._last_obs = obs
        self._started = True
        return obs, {"state": state}

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        """Advance one policy tick and return the Gymnasium vector tuple."""
        self._ensure_open()
        if not self._started or self._last_obs is None:
            raise TaskEnvError("TaskEnv.step requires reset() first (no previous observation)")
        flat = self._validate_actions(actions)

        named = self.action_spec.unflatten(flat)
        result = self.runner.step(named)
        self._validate_core_termination(result)

        pre = self._snapshot(t=int(result.t), extra=result.extra)
        self.on_step(flat, pre)
        next_obs = self._validate_obs(self.observe(pre))
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        transition = Transition(
            obs=self._last_obs,
            action=flat,
            next_obs=next_obs,
            terminated=zeros,
            truncated=zeros,
            info=pre,
        )

        task_terminated = self._validate_mask(self.terminated(transition), "terminated()")
        task_truncated = self._validate_mask(self.truncated(transition), "truncated()")

        due_mask = self._due_mask(result.reset_env_ids)
        truncated = (due_mask | task_truncated) & ~task_terminated
        done = task_terminated | truncated
        transition = replace(
            transition,
            terminated=task_terminated,
            truncated=truncated,
        )

        reasons = self.termination_reasons(transition, task_terminated, truncated) or {}
        reasons = self._validate_reasons(reasons)

        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1).to(torch.int64)
        final_observation = next_obs[done] if bool(done.any()) else None
        episode_end = None
        final_info: tuple[EpisodeEnd | None, ...] = tuple([None] * self.num_envs)
        if bool(done.any()):
            episode_end = EpisodeEnd(
                env_ids=done_ids,
                terminated=task_terminated[done],
                truncated=truncated[done],
                reasons={name: mask[done] for name, mask in reasons.items()},
                state=pre,
            )
            final_info = self._per_env_episode_end(
                done, done_ids, task_terminated, truncated, reasons, pre
            )

        reward = self._validate_reward(self.reward(transition))

        if bool(done.any()):
            self.on_episode_end(done_ids, pre, transition)
            self.runner.reset(done_ids)
            post = self._snapshot(t=int(result.t) + 1)
            self._reset_buffers(done_ids)
            self.on_reset(done_ids, post)
            obs = self._validate_obs(self.observe(post))
        else:
            post = pre
            obs = next_obs

        self._last_obs = obs
        info: dict[str, object] = {
            "state": post,
            "termination_reasons": reasons,
            "final_observation": final_observation,
            "final_info": final_info,
            "episode_end": episode_end,
        }
        return obs, reward, task_terminated, truncated, info

    def close(self) -> None:
        """Run ``on_close`` once; the borrowed Run/runner are left open."""
        if self._closed:
            return
        self._closed = True
        self.on_close()

    # ------------------------------------------------------------------ #
    # user hooks (safe defaults)
    # ------------------------------------------------------------------ #
    def on_bind(self, context: TaskContext) -> None:
        """Called once after the run is bound; default no-op."""

    def register_buffer(self, buffer: Buffer) -> Buffer:
        """Register a history buffer reset on every env reset; return it.

        Idempotent by object identity; the buffer only needs a callable
        ``reset(env_ids=None)``.  Nothing is pushed automatically.
        """
        if not callable(getattr(buffer, "reset", None)):
            raise TaskConfigError(
                "register_buffer expects an object with a callable reset(env_ids=None), "
                f"got {type(buffer).__name__}"
            )
        if any(existing is buffer for existing in self._buffers):
            return buffer
        self._buffers.append(buffer)
        return buffer

    def observe(self, state: TaskState) -> torch.Tensor:
        """Return the flat ``(num_envs, obs_dim)`` observation for *state*."""
        raise NotImplementedError(
            f"{type(self).__name__}.observe(state) must return a (num_envs, obs_dim) tensor"
        )

    def reward(self, transition: Transition) -> torch.Tensor:
        """Return the per-env reward ``(num_envs,)`` for *transition*."""
        raise NotImplementedError(
            f"{type(self).__name__}.reward(transition) must return a (num_envs,) tensor"
        )

    def terminated(self, transition: Transition) -> torch.Tensor:
        """Return the task's terminal mask ``(num_envs,)`` bool; default False."""
        del transition
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def truncated(self, transition: Transition) -> torch.Tensor:
        """Return the task's truncation mask ``(num_envs,)`` bool; default False."""
        del transition
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def termination_reasons(
        self,
        transition: Transition,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> Mapping[str, torch.Tensor] | None:
        """Return log-friendly reason masks (``str -> (num_envs,) bool``)."""
        del transition, terminated, truncated
        return None

    def on_step(self, action: torch.Tensor, state: TaskState) -> None:
        """Called once per step before ``observe``; default no-op."""
        del action, state

    def on_reset(self, env_ids: torch.Tensor, state: TaskState) -> None:
        """Called after every reset (all ids, then done ids); default no-op."""

    def on_episode_end(
        self,
        env_ids: torch.Tensor,
        state: TaskState,
        transition: Transition,
    ) -> None:
        """Called for done ids before they are reset; default no-op."""

    def on_close(self) -> None:
        """Called once by :meth:`close`; default no-op."""

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def observation_shape(self) -> tuple[int, ...] | None:
        """The flat observation shape once known (``(obs_dim,)``)."""
        if self.obs_dim is None:
            return None
        return (int(self.obs_dim),)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _ensure_open(self) -> None:
        if self._closed:
            raise TaskEnvError(f"{type(self).__name__} is closed")

    def _reset_buffers(self, env_ids: torch.Tensor) -> None:
        for buffer in self._buffers:
            buffer.reset(env_ids)

    def _snapshot(
        self,
        *,
        t: int,
        extra: Mapping[str, object] | None = None,
    ) -> TaskState:
        if extra is None:
            extra = read_extra(self.env)
        obs_cache = getattr(self.env, "obs_cache", None)
        if not isinstance(obs_cache, Mapping):
            raise TaskContractError(
                f"TaskEnv: env.obs_cache must be a mapping, got {type(obs_cache).__name__}"
            )
        return TaskState(
            t=int(t),
            num_envs=self.num_envs,
            extra=extra,
            obs_cache=dict(obs_cache),
        )

    def _validate_obs(self, obs: object) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            raise TaskContractError(
                f"observe() must return a torch.Tensor, got {type(obs).__name__}"
            )
        if obs.ndim != 2:
            raise TaskContractError(
                f"observe() must return a 2-D (num_envs, obs_dim) tensor, got {tuple(obs.shape)}"
            )
        if obs.shape[0] != self.num_envs:
            raise TaskContractError(
                f"observe() must return {self.num_envs} rows, got {obs.shape[0]}"
            )
        if obs.shape[1] < 1:
            raise TaskContractError("observe() must return at least one feature column")
        if self.obs_dim is not None and obs.shape[1] != self.obs_dim:
            raise TaskContractError(
                f"observe() returned obs_dim={obs.shape[1]}, expected {self.obs_dim}"
            )
        self.obs_dim = int(obs.shape[1])
        return obs

    def _validate_actions(self, actions: object) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            raise TaskContractError(
                f"step(actions) expects a torch.Tensor, got {type(actions).__name__}"
            )
        if actions.ndim != 2:
            raise TaskContractError(
                f"step(actions) expects a 2-D (num_envs, action_dim) tensor, "
                f"got {tuple(actions.shape)}"
            )
        if actions.shape[0] != self.num_envs:
            raise TaskContractError(
                f"step(actions) expects {self.num_envs} rows, got {actions.shape[0]}"
            )
        if actions.shape[1] != self.action_dim:
            raise TaskContractError(
                f"step(actions) expects {self.action_dim} columns, got {actions.shape[1]}"
            )
        if not actions.is_floating_point():
            raise TaskContractError(f"step(actions) must be a float tensor, got {actions.dtype}")
        if actions.device != self.device:
            actions = actions.to(self.device)
        return actions

    def _validate_mask(self, mask: object, name: str) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise TaskContractError(f"{name} must return a torch.Tensor, got {type(mask).__name__}")
        if mask.dtype is not torch.bool:
            raise TaskContractError(f"{name} must return bool, got {mask.dtype}")
        if mask.shape != (self.num_envs,):
            raise TaskContractError(
                f"{name} must return shape ({self.num_envs},), got {tuple(mask.shape)}"
            )
        return mask.to(self.device)

    def _validate_reward(self, reward: object) -> torch.Tensor:
        if not isinstance(reward, torch.Tensor):
            raise TaskContractError(
                f"reward() must return a torch.Tensor, got {type(reward).__name__}"
            )
        if reward.shape != (self.num_envs,):
            raise TaskContractError(
                f"reward() must return shape ({self.num_envs},), got {tuple(reward.shape)}"
            )
        if not reward.is_floating_point():
            raise TaskContractError(f"reward() must be a float tensor, got {reward.dtype}")
        return reward.to(self.device)

    def _validate_core_termination(self, result: object) -> None:
        core = getattr(result, "terminated", None)
        if core is None:
            return
        if not isinstance(core, torch.Tensor):
            raise TaskContractError(
                "core StepResult.terminated must be a tensor or None, " f"got {type(core).__name__}"
            )
        if core.shape != (self.num_envs,):
            raise TaskContractError(
                "core StepResult.terminated must be all-False with shape "
                f"({self.num_envs},), got {tuple(core.shape)}"
            )
        if bool(core.any().item()):
            raise TaskContractError(
                "core StepResult.terminated fired in termination_mode: external; "
                "core terminations must be empty (the task layer owns them)"
            )

    def _due_mask(self, reset_env_ids: object) -> torch.Tensor:
        empty = _empty_ids(self.num_envs, self.device)
        if reset_env_ids is None:
            ids = empty
        else:
            if not isinstance(reset_env_ids, torch.Tensor):
                raise TaskContractError(
                    "core StepResult.reset_env_ids must be a tensor or None, "
                    f"got {type(reset_env_ids).__name__}"
                )
            if reset_env_ids.ndim != 1 or reset_env_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TaskContractError(
                    "core StepResult.reset_env_ids must be a 1-D integer tensor, "
                    f"got {reset_env_ids.dtype} shape {tuple(reset_env_ids.shape)}"
                )
            ids = reset_env_ids.to(device=self.device, dtype=torch.int64)
            if ids.numel():
                if int(ids.min()) < 0 or int(ids.max()) >= self.num_envs:
                    raise TaskContractError(
                        f"core StepResult.reset_env_ids out of range [0, {self.num_envs}): "
                        f"{ids.tolist()}"
                    )
                if int(torch.unique(ids).numel()) != int(ids.numel()):
                    raise TaskContractError(
                        "core StepResult.reset_env_ids must be unique, " f"got {ids.tolist()}"
                    )
        due = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if ids.numel():
            due[ids] = True
        return due

    def _validate_reasons(self, reasons: object) -> dict[str, torch.Tensor]:
        if not isinstance(reasons, Mapping):
            raise TaskContractError(
                "termination_reasons() must return a mapping or None, "
                f"got {type(reasons).__name__}"
            )
        validated: dict[str, torch.Tensor] = {}
        for name, mask in reasons.items():
            if not isinstance(name, str):
                raise TaskContractError(
                    f"termination_reasons() keys must be str, got {type(name).__name__}"
                )
            validated[name] = self._validate_mask(mask, f"termination_reasons()[{name!r}]")
        return validated

    def _per_env_episode_end(
        self,
        done: torch.Tensor,
        done_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        reasons: Mapping[str, torch.Tensor],
        state: TaskState,
    ) -> tuple[EpisodeEnd | None, ...]:
        per_env: list[EpisodeEnd | None] = [None] * self.num_envs
        for offset, env_id in enumerate(done_ids.tolist()):
            env_ids = torch.tensor([env_id], dtype=torch.int64, device=self.device)
            per_env[env_id] = EpisodeEnd(
                env_ids=env_ids,
                terminated=terminated[done][offset : offset + 1],
                truncated=truncated[done][offset : offset + 1],
                reasons={name: mask[done][offset : offset + 1] for name, mask in reasons.items()},
                state=state,
            )
        return tuple(per_env)
