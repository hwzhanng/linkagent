"""One auditable sim-to-real boundary for the tomato-pick actor.

The physics environment owns ground truth.  This object is the only place
allowed to perturb deployable observations, policy commands, fruit material,
and stem break thresholds.  Keeping those transformations together prevents
training-time noise from leaking into rewards or becoming scattered wrappers.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp


class Sim2RealRuntime:
    """Sample the full deployment distribution once per environment reset."""

    FORCE_MAX_AGE_S = 0.15
    TARGET_MAX_AGE_S = 5.0
    MATERIAL_BUCKET_COUNT = 32

    def __init__(
        self,
        *,
        profile: dict[str, Any],
        num_envs: int,
        action_dim: int,
        policy_rate_hz: int,
        device: str,
        enabled: bool,
    ) -> None:
        self.profile = profile
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.policy_dt = 1.0 / policy_rate_hz
        self.device = device
        self.enabled = enabled

        perception = profile["perception_bridge"]["randomization"]
        control = profile["robot_and_control_randomization"]
        tactile = profile["tactile_measurement_model"]
        contact = profile["contact_and_stem_randomization"]
        self._ranges = {
            "target_position_std": perception["center_position_noise_std_m"],
            "target_rotation_std_deg": perception["orientation_noise_std_deg"],
            "axis_relative_std": perception["semi_axis_relative_noise_std"],
            "target_position_bias": perception["frame_calibration_translation_bias_m"],
            "target_rotation_bias_deg": perception["frame_calibration_rotation_bias_deg"],
            "target_delay_s": perception["measurement_delay_s"],
            "action_delay": control["action_delay_policy_steps"],
            "action_scale": control["action_scale_multiplier"],
            "joint_encoder_offset": control["joint_encoder_offset_rad"],
            "hand_gain": control.get("hand_joint_gain", control.get("hand_synergy_gain")),
            "hand_deadband": control.get(
                "hand_joint_deadband", control.get("hand_synergy_deadband")
            ),
            "force_gain": tactile["normal_force_gain"],
            "force_bias": tactile["normal_force_bias_n"],
            "force_delay_s": tactile["force_delay_s"],
            "static_friction": contact["fruit_static_friction"],
            "dynamic_friction": contact["fruit_dynamic_friction"],
            "restitution": contact["fruit_restitution"],
            "break_force": contact["break_force_n"],
            "break_torque": contact["break_torque_nm"],
        }
        self.contact_threshold_n = float(tactile["contact_threshold_n"])
        self._validate_ranges()

        maximum_action_delay = int(self._ranges["action_delay"][1])
        maximum_force_delay = math.ceil(
            float(self._ranges["force_delay_s"][1]) / self.policy_dt
        )
        self._action_history = torch.zeros(
            (maximum_action_delay + 1, num_envs, action_dim), device=device
        )
        self._force_history = torch.zeros(
            (maximum_force_delay + 1, num_envs, 5), device=device
        )
        self.action_delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.action_scale = torch.ones((num_envs, 1), device=device)
        self.hand_gain = torch.ones((num_envs, 1), device=device)
        self.hand_deadband = torch.zeros((num_envs, 1), device=device)
        self.joint_encoder_offset = torch.zeros((num_envs, 7), device=device)
        self.measured_axes_m = torch.zeros((num_envs, 3), device=device)
        self.target_valid = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.target_age_s = torch.zeros(num_envs, device=device)
        self.force_valid = torch.ones((num_envs, 5), dtype=torch.bool, device=device)
        self.force_age_s = torch.zeros((num_envs, 5), device=device)
        self._target_position_std = torch.zeros((num_envs, 1), device=device)
        self._target_rotation_std = torch.zeros((num_envs, 1), device=device)
        self._target_position_bias = torch.zeros((num_envs, 3), device=device)
        self._target_rotation_bias = torch.zeros((num_envs, 3), device=device)
        self._target_base_age_s = torch.zeros(num_envs, device=device)
        self.target_position_error_m = torch.zeros((num_envs, 3), device=device)
        self.target_rotation_error_rad = torch.zeros((num_envs, 3), device=device)
        self._force_gain = torch.ones((num_envs, 5), device=device)
        self._force_bias = torch.zeros((num_envs, 5), device=device)
        self._force_delay_steps = torch.zeros((num_envs, 5), dtype=torch.long, device=device)
        self.break_force_n = torch.zeros(num_envs, device=device)
        self.break_torque_nm = torch.zeros(num_envs, device=device)
        self._material_bucket_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._material_buckets = self._build_material_buckets()

    def _validate_ranges(self) -> None:
        for name, bounds in self._ranges.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"sim2real range {name!r} must contain two values")
            lower, upper = bounds
            if isinstance(lower, bool) or isinstance(upper, bool) or lower > upper:
                raise ValueError(f"invalid sim2real range {name!r}: {bounds!r}")

    def _uniform(self, name: str, shape: tuple[int, ...], *, nominal: float) -> torch.Tensor:
        if not self.enabled:
            return torch.full(shape, nominal, device=self.device)
        lower, upper = self._ranges[name]
        return torch.empty(shape, device=self.device).uniform_(float(lower), float(upper))

    def _integers(self, name: str, shape: tuple[int, ...], *, nominal: int = 0) -> torch.Tensor:
        if not self.enabled:
            return torch.full(shape, nominal, dtype=torch.long, device=self.device)
        lower, upper = self._ranges[name]
        return torch.randint(int(lower), int(upper) + 1, shape, device=self.device)

    def _build_material_buckets(self) -> torch.Tensor:
        count = self.MATERIAL_BUCKET_COUNT if self.enabled else 1
        static = self._uniform("static_friction", (count,), nominal=0.55)
        dynamic = self._uniform("dynamic_friction", (count,), nominal=0.45)
        dynamic = torch.minimum(dynamic, static)
        restitution = self._uniform("restitution", (count,), nominal=0.02)
        return torch.stack((static, dynamic, restitution), dim=1)

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        true_axes_m: torch.Tensor,
        gravity_force_n: torch.Tensor,
    ) -> None:
        """Reset histories and sample every supported deployment uncertainty."""

        ids = env_ids.long()
        count = len(ids)
        self._action_history[:, ids] = 0.0
        self._force_history[:, ids] = 0.0

        self.action_delay_steps[ids] = self._integers("action_delay", (count,))
        self.action_scale[ids] = self._uniform("action_scale", (count, 1), nominal=1.0)
        self.hand_gain[ids] = self._uniform("hand_gain", (count, 1), nominal=1.0)
        self.hand_deadband[ids] = self._uniform("hand_deadband", (count, 1), nominal=0.0)
        self.joint_encoder_offset[ids] = self._uniform(
            "joint_encoder_offset", (count, 7), nominal=0.0
        )

        axis_std = self._uniform("axis_relative_std", (count, 1), nominal=0.0)
        axis_noise = torch.randn((count, 3), device=self.device) * axis_std
        lower = torch.tensor(
            self.profile["correlated_tomato_sampling"]["semi_axes_m"]["x"],
            device=self.device,
        )[0]
        upper = torch.tensor(
            self.profile["correlated_tomato_sampling"]["semi_axes_m"]["x"],
            device=self.device,
        )[1]
        self.measured_axes_m[ids] = (true_axes_m * (1.0 + axis_noise)).clamp(lower, upper)

        self._target_position_std[ids] = self._uniform(
            "target_position_std", (count, 1), nominal=0.0
        )
        self._target_rotation_std[ids] = torch.deg2rad(
            self._uniform("target_rotation_std_deg", (count, 1), nominal=0.0)
        )
        self._target_position_bias[ids] = self._uniform(
            "target_position_bias", (count, 3), nominal=0.0
        )
        self._target_rotation_bias[ids] = torch.deg2rad(
            self._uniform("target_rotation_bias_deg", (count, 3), nominal=0.0)
        )
        self._target_base_age_s[ids] = self._uniform(
            "target_delay_s", (count,), nominal=0.0
        )
        self.target_position_error_m[ids] = self._target_position_bias[ids] + (
            torch.randn((count, 3), device=self.device) * self._target_position_std[ids]
        )
        self.target_rotation_error_rad[ids] = self._target_rotation_bias[ids] + (
            torch.randn((count, 3), device=self.device) * self._target_rotation_std[ids]
        )
        self.target_valid[ids] = True
        self.target_age_s[ids] = self._target_base_age_s[ids]

        self._force_gain[ids] = self._uniform("force_gain", (count, 5), nominal=1.0)
        self._force_bias[ids] = self._uniform("force_bias", (count, 5), nominal=0.0)
        force_delay_s = self._uniform("force_delay_s", (count, 5), nominal=0.0)
        self._force_delay_steps[ids] = torch.round(force_delay_s / self.policy_dt).long()
        self.force_valid[ids] = True
        self.force_age_s[ids] = self._force_delay_steps[ids].float() * self.policy_dt

        raw_break = self._uniform("break_force", (count,), nominal=5.0)
        self.break_force_n[ids] = torch.maximum(raw_break, gravity_force_n * 1.5)
        self.break_torque_nm[ids] = self._uniform("break_torque", (count,), nominal=0.008)
        bucket_count = len(self._material_buckets)
        self._material_bucket_ids[ids] = torch.randint(
            0, bucket_count, (count,), device=self.device
        )

    def process_actions(
        self,
        actions: torch.Tensor,
        *,
        base_scales: torch.Tensor,
        blocked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the delayed policy action and the physical scaled command."""

        incoming = torch.where(blocked.unsqueeze(1), 0.0, actions)
        self._action_history[1:] = self._action_history[:-1].clone()
        self._action_history[0] = incoming
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed = self._action_history[self.action_delay_steps, env_ids]
        scaled = delayed * base_scales * self.action_scale
        hand_active = delayed[:, 6:].abs() >= self.hand_deadband
        scaled[:, 6:] = torch.where(
            hand_active, scaled[:, 6:] * self.hand_gain, torch.zeros_like(scaled[:, 6:])
        )
        return delayed, scaled

    def clear_action_history(self, env_ids: torch.Tensor) -> None:
        """Discard commands that belong to a completed control phase.

        Deployment phase transitions are feedback barriers: once GRASP,
        DETACH, or HOLD is confirmed, queued commands from the previous phase
        are no longer valid.  Keeping them in the simulated delay line can,
        for example, replay a DETACH release command after entering HOLD.
        """

        ids = env_ids.long()
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        self._action_history[:, ids] = 0.0

    def measure_tactile(
        self, true_force_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return measured normal force, validity, and normalized age."""

        self._force_history[1:] = self._force_history[:-1].clone()
        self._force_history[0] = true_force_n
        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        finger_ids = torch.arange(5, device=self.device).unsqueeze(0)
        delayed = self._force_history[self._force_delay_steps, env_ids, finger_ids]
        candidate = (delayed * self._force_gain + self._force_bias).clamp(0.0, 25.0)
        # The deployed runner aborts before policy execution when a force
        # sample is invalid.  V1 therefore trains only on valid samples rather
        # than teaching recovery from a state the real policy never receives.
        valid = torch.ones((self.num_envs, 5), dtype=torch.bool, device=self.device)
        self.force_valid.copy_(valid)
        base_age = self._force_delay_steps.float() * self.policy_dt
        self.force_age_s = torch.where(
            valid,
            base_age,
            (self.force_age_s + self.policy_dt).clamp_max(self.FORCE_MAX_AGE_S),
        )
        return (
            candidate,
            valid,
            (self.force_age_s / self.FORCE_MAX_AGE_S).clamp(0.0, 1.0),
        )

    def apply_fruit_material(self, asset: Any, env_ids: torch.Tensor) -> None:
        """Assign one of 32 bounded material buckets through the PhysX tensor API."""

        ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
        materials = wp.to_torch(asset.root_view.get_material_properties())
        samples = self._material_buckets[self._material_bucket_ids[env_ids.long()]].to(
            materials.device
        )
        materials[ids_cpu] = samples.to(materials.dtype).unsqueeze(1).expand(
            -1, materials.shape[1], -1
        )
        asset.root_view.set_material_properties(
            wp.from_torch(materials.contiguous(), dtype=wp.float32),
            wp.from_torch(ids_cpu.to(torch.int32), dtype=wp.int32),
        )


__all__ = ["Sim2RealRuntime"]
