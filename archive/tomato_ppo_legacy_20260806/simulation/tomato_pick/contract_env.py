"""Deterministic C2 contract environment used before Isaac assets are selected.

This environment tests state, reward, termination, observation, and action
contracts.  It is deliberately not a physics simulator and must never be used
to produce a deployable PPO checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

from module.skills.tomato_pick import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PHASE_ORDER,
    TomatoPickAction,
    TomatoPickObservation,
)

from .config import load_c2_profile


@dataclass(frozen=True, slots=True)
class SpaceSpec:
    """Dependency-free shape/range declaration for a future Gym tensor space."""

    shape: tuple[int, ...]
    dtype: str
    low: float | None
    high: float | None


@dataclass(slots=True)
class _EpisodeState:
    seed: int
    phase_index: int
    phase_step: int
    episode_step: int
    phase_progress_steps: float
    phase_target_steps: int
    contact_confirmed: bool
    grasp_stable: bool
    stem_detached: bool
    retained_without_slip: bool
    slip_detected: bool
    dropped: bool
    verifier_valid: bool
    stem_break_event_count: int
    hand_joint_positions: list[float]
    previous_action: TomatoPickAction
    failure_reason: str | None = None
    success: bool = False

    @property
    def phase(self) -> str:
        return PHASE_ORDER[self.phase_index]


class ContractTomatoPickEnv:
    """Gym-style scalar environment for frozen-interface acceptance tests.

    ``reset`` and ``step`` intentionally return the frozen DTO instead of a
    NumPy array, keeping this gate runnable in the standard-library test
    environment.  The Isaac wrapper will own tensor conversion and batching.
    """

    backend_kind = "contract_smoke_not_physics"
    suitable_for_ppo_training = False
    observation_space_spec = SpaceSpec((OBSERVATION_DIM,), "float32", None, None)
    action_space_spec = SpaceSpec((ACTION_DIM,), "float32", -1.0, 1.0)

    def __init__(self, *, profile: Mapping[str, Any] | None = None) -> None:
        loaded = dict(profile) if profile is not None else load_c2_profile()
        policy = loaded.get("policy")
        environment = loaded.get("environment")
        if not isinstance(policy, Mapping) or not isinstance(environment, Mapping):
            raise TypeError("profile requires policy and environment mappings")
        limits = tuple(policy.get("phase_step_limits", ()))
        if len(limits) != len(PHASE_ORDER) or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in limits
        ):
            raise ValueError("phase_step_limits must contain four positive integers")
        maximum = environment.get("termination", {}).get("maximum_episode_steps")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("maximum_episode_steps must be a positive integer")
        self._phase_limits = limits
        self._maximum_episode_steps = maximum
        self._random = random.Random()
        self._state: _EpisodeState | None = None
        self._closed = False

    @property
    def state_summary(self) -> Mapping[str, Any]:
        """Return small episode state for tests and structured metrics."""

        state = self._require_state()
        return {
            "seed": state.seed,
            "phase": state.phase,
            "phase_step": state.phase_step,
            "episode_step": state.episode_step,
            "contact_confirmed": state.contact_confirmed,
            "grasp_stable": state.grasp_stable,
            "stem_detached": state.stem_detached,
            "retained_without_slip": state.retained_without_slip,
            "slip_detected": state.slip_detected,
            "dropped": state.dropped,
            "verifier_valid": state.verifier_valid,
            "stem_break_event_count": state.stem_break_event_count,
            "failure_reason": state.failure_reason,
            "success": state.success,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[TomatoPickObservation, Mapping[str, Any]]:
        """Start a deterministic episode and clear every verifier/metric field."""

        if self._closed:
            raise RuntimeError("environment is closed")
        if options not in (None, {}):
            raise ValueError("contract environment does not accept reset options")
        resolved_seed = 0 if seed is None else seed
        if isinstance(resolved_seed, bool) or not isinstance(resolved_seed, int):
            raise TypeError("seed must be an integer")
        self._random.seed(resolved_seed)
        self._state = _EpisodeState(
            seed=resolved_seed,
            phase_index=0,
            phase_step=0,
            episode_step=0,
            phase_progress_steps=0.0,
            phase_target_steps=self._target_steps(0),
            contact_confirmed=False,
            grasp_stable=False,
            stem_detached=False,
            retained_without_slip=False,
            slip_detected=False,
            dropped=False,
            verifier_valid=True,
            stem_break_event_count=0,
            hand_joint_positions=[0.0] * 16,
            previous_action=TomatoPickAction.zero(),
        )
        observation = self._observation()
        return observation, self._info(stem_break_event=False)

    def step(
        self,
        action: TomatoPickAction | Sequence[float],
    ) -> tuple[TomatoPickObservation, float, bool, bool, Mapping[str, Any]]:
        """Apply one normalized 20 Hz action and update contract-only dynamics."""

        state = self._require_state()
        if state.success or state.failure_reason is not None:
            raise RuntimeError("episode is terminal; call reset before step")
        normalized = (
            action if isinstance(action, TomatoPickAction) else TomatoPickAction(tuple(action))
        )
        old_action = state.previous_action
        phase_before = state.phase
        state.episode_step += 1
        state.phase_step += 1
        self._update_hand(normalized)

        stem_break_event = False
        reward = self._action_rate_penalty(old_action, normalized)
        if phase_before == "contact":
            reward += self._step_contact(normalized)
        elif phase_before == "grasp":
            reward += self._step_grasp(normalized)
        elif phase_before == "detach":
            phase_reward, stem_break_event = self._step_detach(normalized)
            reward += phase_reward
        else:
            reward += self._step_hold(normalized)

        state.previous_action = normalized
        if state.dropped:
            state.retained_without_slip = False
            state.failure_reason = "drop"
            reward -= 10.0
        elif state.slip_detected:
            state.retained_without_slip = False
            state.failure_reason = "slip"
            reward -= 1.0

        if (
            state.failure_reason is None
            and state.phase_step >= self._phase_limits[state.phase_index]
        ):
            state.failure_reason = "phase_timeout"

        if (
            state.failure_reason is None
            and state.episode_step >= self._maximum_episode_steps
            and not state.success
        ):
            state.failure_reason = "episode_timeout"

        if state.success:
            reward += 10.0
        terminated = state.success or state.failure_reason in {"drop", "slip"}
        truncated = state.failure_reason in {"phase_timeout", "episode_timeout"}
        observation = self._observation()
        return observation, reward, terminated, truncated, self._info(
            stem_break_event=stem_break_event
        )

    def close(self) -> None:
        """Close idempotently; no simulator resources exist in this backend."""

        self._closed = True
        self._state = None

    def _require_state(self) -> _EpisodeState:
        if self._closed:
            raise RuntimeError("environment is closed")
        if self._state is None:
            raise RuntimeError("call reset before using the environment")
        return self._state

    def _target_steps(self, phase_index: int) -> int:
        high = min(self._phase_limits[phase_index] - 1, (4, 5, 7, 5)[phase_index])
        low = min(high, (3, 4, 5, 3)[phase_index])
        return self._random.randint(max(1, low), max(1, high))

    def _advance_phase(self) -> None:
        state = self._require_state()
        if state.phase_index >= len(PHASE_ORDER) - 1:
            return
        state.phase_index += 1
        state.phase_step = 0
        state.phase_progress_steps = 0.0
        state.phase_target_steps = self._target_steps(state.phase_index)

    def _update_hand(self, action: TomatoPickAction) -> None:
        state = self._require_state()
        for index, delta in enumerate(action.delta_hand_joints):
            state.hand_joint_positions[index] = min(
                1.0,
                max(-1.0, state.hand_joint_positions[index] + 0.08 * delta),
            )

    @staticmethod
    def _flexion_action(action: TomatoPickAction) -> float:
        indices = (0, 1, 2, 3, 4, 11, 12, 13, 14, 15)
        return sum(action.delta_hand_joints[index] for index in indices) / len(indices)

    def _step_contact(self, action: TomatoPickAction) -> float:
        state = self._require_state()
        if action.delta_xyz[2] <= -0.45:
            state.phase_progress_steps += 1.0
            reward = 1.0 / state.phase_target_steps
        else:
            state.phase_progress_steps = max(0.0, state.phase_progress_steps - 0.25)
            reward = 0.0
        if state.phase_progress_steps >= state.phase_target_steps:
            state.contact_confirmed = True
            reward += 0.5
            self._advance_phase()
        return reward

    def _step_grasp(self, action: TomatoPickAction) -> float:
        state = self._require_state()
        flexion = self._flexion_action(action)
        if flexion < -0.5:
            state.dropped = True
            return 0.0
        if flexion >= 0.5:
            state.phase_progress_steps += 1.0
            reward = 1.0 / state.phase_target_steps
        else:
            reward = 0.0
        if state.phase_progress_steps >= state.phase_target_steps:
            state.grasp_stable = True
            state.retained_without_slip = True
            reward += 1.0
            self._advance_phase()
        return reward

    def _step_detach(self, action: TomatoPickAction) -> tuple[float, bool]:
        state = self._require_state()
        if self._flexion_action(action) < -0.25:
            state.slip_detected = True
            return 0.0, False
        if action.delta_xyz[2] >= 0.45:
            increment = 1.0 + 0.25 * abs(action.delta_rotation_vector[2])
            state.phase_progress_steps += increment
            reward = increment / state.phase_target_steps
        else:
            reward = 0.0
        if state.phase_progress_steps >= state.phase_target_steps:
            if not state.stem_detached:
                state.stem_detached = True
                state.stem_break_event_count += 1
                reward += 5.0
            self._advance_phase()
            return reward, True
        return reward, False

    def _step_hold(self, action: TomatoPickAction) -> float:
        state = self._require_state()
        if self._flexion_action(action) < -0.25:
            state.slip_detected = True
            return 0.0
        if max(abs(action.delta_xyz[0]), abs(action.delta_xyz[1])) > 0.9:
            state.dropped = True
            return 0.0
        state.phase_progress_steps += 1.0
        reward = 2.0 / state.phase_target_steps
        if state.phase_progress_steps >= state.phase_target_steps:
            state.retained_without_slip = True
            state.success = bool(
                state.verifier_valid
                and state.stem_detached
                and state.retained_without_slip
                and not state.slip_detected
                and not state.dropped
            )
        return reward

    @staticmethod
    def _action_rate_penalty(
        previous: TomatoPickAction,
        current: TomatoPickAction,
    ) -> float:
        squared = sum(
            (right - left) ** 2
            for left, right in zip(previous.values, current.values)
        )
        return -0.01 * squared

    def _observation(self) -> TomatoPickObservation:
        state = self._require_state()
        progress = min(1.0, state.phase_progress_steps / state.phase_target_steps)
        pressure = 0.0
        if state.contact_confirmed:
            pressure = 0.25
        if state.grasp_stable:
            pressure = 0.60
        if state.slip_detected:
            pressure = 0.10
        reference_z = 0.0 if state.phase != "contact" else 1.0 - progress
        observation = TomatoPickObservation.pack(
            target_pose_error=(0.0, 0.0, reference_z, 0.0, 0.0, 0.0),
            ee_twist=(0.0,) * 6,
            arm_joint_position_norm=(0.0,) * 7,
            arm_joint_velocity_norm=(0.0,) * 7,
            hand_joint_position_norm=tuple(state.hand_joint_positions),
            hand_joint_velocity_norm=(0.0,) * 16,
            finger_pressure_norm=(pressure,) * 5,
            finger_pressure_rate_norm=(0.0,) * 5,
            gravity_direction=(0.0, 0.0, -1.0),
            previous_action=state.previous_action.values,
            target_arm_hand_valid=(1.0, 1.0, 1.0),
            force_valid=(1.0,) * 5,
            target_arm_hand_age_norm=(0.0, 0.0, 0.0),
            force_age_norm=(0.0,) * 5,
            fruit_axes_norm=(0.0, -0.2, 0.2),
            episode_progress=min(
                1.0, state.episode_step / self._maximum_episode_steps
            ),
        )
        if any(not math.isfinite(value) for value in observation.values):
            raise RuntimeError("contract observation contains a non-finite value")
        return observation

    def _info(self, *, stem_break_event: bool) -> Mapping[str, Any]:
        state = self._require_state()
        return {
            "schema_version": 1,
            "backend_kind": self.backend_kind,
            "physics_valid": False,
            "suitable_for_ppo_training": self.suitable_for_ppo_training,
            "simulated_tactile_proxy": True,
            "actor_contains_privileged_state": False,
            "phase": state.phase,
            "episode_step": state.episode_step,
            "stem_break_event": stem_break_event,
            "stem_break_event_count": state.stem_break_event_count,
            "stem_detached": state.stem_detached,
            "retained_without_slip": state.retained_without_slip,
            "slip_detected": state.slip_detected,
            "dropped": state.dropped,
            "verifier_valid": state.verifier_valid,
            "success": state.success,
            "failure_reason": state.failure_reason,
        }


__all__ = ["ContractTomatoPickEnv", "SpaceSpec"]
