"""V3 strict-telemetry adapter over the frozen v2 PhysX environment."""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from simulation.ppo_tomato_grasp.enveloping_verifier import _contact_geometry_metrics
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)
from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoSceneCfg
from simulation.ppo_tomato_grasp_v3.strict_telemetry import GATE_ORDER, encode_failed_gates, update_streak
from simulation.ppo_tomato_grasp_v3.direct_joint_action import (
    V3_DIRECT_LOWER_RAD,
    V3_DIRECT_UPPER_RAD,
    v3_direct_action_to_joint_targets,
)
from simulation.ppo_tomato_grasp_v3.collision_proxies import author_fingertip_collision_proxies
from simulation.ppo_tomato_grasp_v3.mimic import (
    URDF_MIMIC_SPECS,
    advance_mimic_positions,
    apply_urdf_mimic_targets,
)


@configclass
class V3PhysxTomatoSceneCfg(PhysxTomatoSceneCfg):
    """Add one contact view per fixed v3 fingertip pad rigid body."""

    v3_thumb_pad_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/v3_fixed_pads/thumb_link5_pad",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"], update_period=0.0, history_length=6,
    )
    v3_index_pad_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/v3_fixed_pads/index_link4_pad",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"], update_period=0.0, history_length=6,
    )
    v3_middle_pad_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/v3_fixed_pads/middle_link4_pad",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"], update_period=0.0, history_length=6,
    )
    v3_ring_pad_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/v3_fixed_pads/ring_link4_pad",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"], update_period=0.0, history_length=6,
    )
    v3_little_pad_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/v3_fixed_pads/little_link4_pad",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"], update_period=0.0, history_length=6,
    )


@configclass
class V3EnvelopingPhysxTomatoGraspEnvCfg(V2EnvelopingPhysxTomatoGraspEnvCfg):
    """V3 keeps v2 dynamics but makes current physical contact authoritative."""

    v3_require_current_physical_five_digit_contact = True
    v3_sequential_contact_gate = False
    v3_contact_persistence_steps = 8
    v3_urdf_terminal_mimic_enabled = False
    v3_urdf_mimic_servo_rate_rad_s = 1.25
    v3_action_contract = "legacy_enveloping14"
    v3_direct_servo_zero_velocity = True
    v3_filter_fingertip_proxy_support_collision = True
    v3_disable_palm_cup_rim_collision = True
    v3_joint_drive_effort_limit_nm = 0.50
    v3_joint_drive_stiffness_nm_rad = 5.00
    v3_joint_drive_damping_nm_s_rad = 2.00
    v3_mimic_drive_effort_limit_nm = 0.10
    v3_mimic_drive_stiffness_nm_rad = 2.00
    v3_mimic_drive_damping_nm_s_rad = 0.05
    v3_mimic_drive_armature_kg_m2 = 0.0001
    v3_hand_disable_gravity = True
    v3_use_tomato_filtered_contact_force = True
    v3_contact_history_substeps = 6
    v3_proxy_attachment_mode = "fixed_body"
    scene: V3PhysxTomatoSceneCfg = V3PhysxTomatoSceneCfg(
        num_envs=16, env_spacing=1.5, replicate_physics=False, clone_in_fabric=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot_cfg.actuators["left_hand"] = ImplicitActuatorCfg(
            joint_names_expr=[
                r"thumb_joint[0-3]",
                r"(index|middle|ring|little)_joint[0-2]",
            ],
            effort_limit_sim=float(self.v3_joint_drive_effort_limit_nm),
            velocity_limit_sim=0.85,
            stiffness=float(self.v3_joint_drive_stiffness_nm_rad),
            damping=float(self.v3_joint_drive_damping_nm_s_rad),
        )
        self.robot_cfg.actuators["terminal_mimic"] = ImplicitActuatorCfg(
            joint_names_expr=[
                r"thumb_joint4",
                r"(index|middle|ring|little)_joint3",
            ],
            effort_limit_sim=float(self.v3_mimic_drive_effort_limit_nm),
            velocity_limit_sim=0.85,
            stiffness=float(self.v3_mimic_drive_stiffness_nm_rad),
            damping=float(self.v3_mimic_drive_damping_nm_s_rad),
            armature=float(self.v3_mimic_drive_armature_kg_m2),
        )
        self.robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=bool(self.v3_hand_disable_gravity)
        )
        event = self.events["v2_fingertip_collision_proxies"]
        event.func = author_fingertip_collision_proxies
        event.params["filter_palm_support"] = bool(
            self.v3_filter_fingertip_proxy_support_collision
        )
        event.params["disable_palm_cup_rims"] = bool(
            self.v3_disable_palm_cup_rim_collision
        )
        event.params["attachment_mode"] = str(self.v3_proxy_attachment_mode)


class V3EnvelopingPhysxTomatoGraspEnv(V2EnvelopingPhysxTomatoGraspEnv):
    """Add reset-safe gate streaks and physical-contact strict success."""

    cfg: V3EnvelopingPhysxTomatoGraspEnvCfg

    def _setup_scene(self) -> None:
        """Create v3 fingertip shapes before Isaac Lab initializes sensors."""

        super()._setup_scene()
        event = self.cfg.events.get("v2_fingertip_collision_proxies")
        if event is None:
            raise RuntimeError("v3 fingertip collision event is missing")
        env_ids = torch.arange(self.cfg.scene.num_envs, device=self.device, dtype=torch.long)
        event.func(self, env_ids, **event.params)

    def __init__(self, cfg: V3EnvelopingPhysxTomatoGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self._v3_telemetry_ready = False
        event = cfg.events.get("v2_fingertip_collision_proxies")
        if event is not None:
            event.params["filter_palm_support"] = bool(
                cfg.v3_filter_fingertip_proxy_support_collision
            )
            event.params["disable_palm_cup_rims"] = bool(
                cfg.v3_disable_palm_cup_rim_collision
            )
            event.params["attachment_mode"] = str(cfg.v3_proxy_attachment_mode)
        super().__init__(cfg, render_mode, **kwargs)
        # Widen the policy joint envelope for the simulation-only direct
        # flexion contract: the hardware-calibrated LEFT_SAFE_* ranges truncate
        # finger curl ~0.3 rad per flexion layer below the vendor URDF limits,
        # which blocks a 90 mm enclosure.  Kinematic prior only, never sent to
        # hardware.
        if cfg.v3_action_contract == "direct_flexion14_v1":
            self._lower = torch.tensor(V3_DIRECT_LOWER_RAD, device=self.device)
            self._upper = torch.tensor(V3_DIRECT_UPPER_RAD, device=self.device)
        # The inherited explicit-servo environment clears PhysX drive gains.
        # V3 uses an implicit drive for stable gravity compensation, so
        # restore its audited gains after the inherited constructor returns.
        stiffness = self.robot.root_physx_view.get_dof_stiffnesses().clone()
        damping = self.robot.root_physx_view.get_dof_dampings().clone()
        stiffness[:, self.active_joint_ids] = float(cfg.v3_joint_drive_stiffness_nm_rad)
        damping[:, self.active_joint_ids] = float(cfg.v3_joint_drive_damping_nm_s_rad)
        mimic_ids = list(self._v3_mimic_joint_ids.values()) if hasattr(self, "_v3_mimic_joint_ids") else [
            self.robot.joint_names.index(name) for name in URDF_MIMIC_SPECS
        ]
        stiffness[:, mimic_ids] = float(cfg.v3_mimic_drive_stiffness_nm_rad)
        damping[:, mimic_ids] = float(cfg.v3_mimic_drive_damping_nm_s_rad)
        physx_env_ids = self.robot._ALL_INDICES.detach().cpu()
        self.robot.root_physx_view.set_dof_stiffnesses(stiffness.detach().cpu(), physx_env_ids)
        self.robot.root_physx_view.set_dof_dampings(damping.detach().cpu(), physx_env_ids)
        self._v3_mimic_joint_ids = {
            name: self.robot.joint_names.index(name) for name in URDF_MIMIC_SPECS
        }
        self._v3_pad_body_ids = [
            self.robot.body_names.index(f"{name}_pad")
            for name in ("thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")
            if f"{name}_pad" in self.robot.body_names
        ]
        envs, device = self.num_envs, self.device
        self._v3_gate_streak = {
            name: torch.zeros(envs, dtype=torch.long, device=device) for name in GATE_ORDER
        }
        self._v3_gate_max_streak = {
            name: torch.zeros(envs, dtype=torch.long, device=device) for name in GATE_ORDER
        }
        self._v3_leave_one_out_streak = {
            name: torch.zeros(envs, dtype=torch.long, device=device) for name in GATE_ORDER
        }
        self._v3_leave_one_out_max_streak = {
            name: torch.zeros(envs, dtype=torch.long, device=device) for name in GATE_ORDER
        }
        self._v3_all_conditions_streak = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_all_conditions_max_streak = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_first_three_physical = torch.full((envs,), -1, dtype=torch.long, device=device)
        self._v3_first_five_physical = torch.full((envs,), -1, dtype=torch.long, device=device)
        self._v3_first_strict_step = torch.full((envs,), -1, dtype=torch.long, device=device)
        self._v3_first_break_step = torch.full((envs,), -1, dtype=torch.long, device=device)
        self._v3_first_break_code = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_terminal_failure_code = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_first_five_failure_code = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_best_gate_pass_count = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_best_gate_step = torch.full((envs,), -1, dtype=torch.long, device=device)
        self._v3_best_gate_failure_code = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_previous_all_conditions = torch.zeros(envs, dtype=torch.bool, device=device)
        self._v3_gate_current = {
            name: torch.zeros(envs, dtype=torch.bool, device=device) for name in GATE_ORDER
        }
        self._v3_physical_coverage_current = torch.zeros(envs, device=device)
        self._v3_physical_coverage_max = torch.zeros(envs, device=device)
        self._v3_physical_opposition_current = torch.ones(envs, device=device)
        self._v3_physical_opposition_min = torch.ones(envs, device=device)
        self._v3_physical_hemisphere_current = torch.ones(envs, device=device)
        self._v3_physical_hemisphere_min = torch.ones(envs, device=device)
        self._v3_physical_palm_load_share = torch.zeros(envs, device=device)
        self._v3_physical_five_current = torch.zeros(envs, dtype=torch.bool, device=device)
        self._v3_digit_recent = torch.zeros(envs, 5, dtype=torch.long, device=device)
        self._v3_physical_digit_streak = torch.zeros((envs, 5), dtype=torch.long, device=device)
        self._v3_physical_digit_max_streak = torch.zeros((envs, 5), dtype=torch.long, device=device)
        self._v3_first_physical_digit_step = torch.full((envs, 5), -1, dtype=torch.long, device=device)
        self._v3_max_simultaneous_physical = torch.zeros(envs, dtype=torch.long, device=device)
        self._v3_physical_peak_force = torch.zeros((envs, 5), device=device)
        self._v3_unfiltered_link4_force = torch.zeros((envs, 5), device=device)
        self._v3_strict_success_latched = torch.zeros(envs, dtype=torch.bool, device=device)
        fields: dict[str, tuple[str, torch.dtype]] = {
            "v3_all_conditions_max_streak": ("_v3_all_conditions_max_streak", torch.long),
            "v3_first_three_physical": ("_v3_first_three_physical", torch.long),
            "v3_first_five_physical": ("_v3_first_five_physical", torch.long),
            "v3_first_strict_step": ("_v3_first_strict_step", torch.long),
            "v3_first_break_step": ("_v3_first_break_step", torch.long),
            "v3_first_break_code": ("_v3_first_break_code", torch.long),
            "v3_terminal_failure_code": ("_v3_terminal_failure_code", torch.long),
            "v3_first_five_failure_code": ("_v3_first_five_failure_code", torch.long),
            "v3_best_gate_pass_count": ("_v3_best_gate_pass_count", torch.long),
            "v3_best_gate_step": ("_v3_best_gate_step", torch.long),
            "v3_best_gate_failure_code": ("_v3_best_gate_failure_code", torch.long),
            "v3_physical_coverage_max": ("_v3_physical_coverage_max", torch.float32),
            "v3_physical_opposition_min": ("_v3_physical_opposition_min", torch.float32),
            "v3_physical_hemisphere_min": ("_v3_physical_hemisphere_min", torch.float32),
            "v3_physical_palm_load_share": ("_v3_physical_palm_load_share", torch.float32),
            "v3_physical_five_current": ("_v3_physical_five_current", torch.bool),
            "v3_physical_digit_max_streak": ("_v3_physical_digit_max_streak", torch.long),
            "v3_first_physical_digit_step": ("_v3_first_physical_digit_step", torch.long),
            "v3_max_simultaneous_physical": ("_v3_max_simultaneous_physical", torch.long),
            "v3_physical_peak_force": ("_v3_physical_peak_force", torch.float32),
            "v3_strict_success": ("_v3_strict_success_latched", torch.bool),
        }
        for name in GATE_ORDER:
            fields[f"v3_{name}_max_streak"] = (f"_v3_snapshot_{name}_max_streak", torch.long)
            fields[f"v3_without_{name}_max_streak"] = (f"_v3_snapshot_without_{name}_max_streak", torch.long)
            setattr(self, f"_v3_snapshot_{name}_max_streak", self._v3_gate_max_streak[name])
            setattr(self, f"_v3_snapshot_without_{name}_max_streak", self._v3_leave_one_out_max_streak[name])
        inherited_snapshot_fields = dict(getattr(self, "_registered_snapshot_fields", {}))
        self.register_snapshot_fields(fields)
        # The v2 adapter stores its dynamic field table separately from the
        # already allocated destination buffers.  Preserve that table when
        # v3 appends telemetry, otherwise v2 fields would stay zero at reset.
        self._registered_snapshot_fields = {**inherited_snapshot_fields, **fields}
        self._v3_telemetry_ready = True
        self._reset_idx(self.robot._ALL_INDICES)

    @torch.inference_mode()
    def _apply_action(self) -> None:
        if self.cfg.v3_action_contract == "direct_flexion14_v1":
            self._apply_direct_flexion_action()
        elif self.cfg.v3_action_contract == "legacy_enveloping14":
            super()._apply_action()
        else:
            raise RuntimeError(f"unsupported v3 action contract {self.cfg.v3_action_contract!r}")
        if not bool(self.cfg.v3_urdf_terminal_mimic_enabled):
            return
        full_pos = self.robot.root_physx_view.get_dof_positions().clone()
        full_vel = self.robot.root_physx_view.get_dof_velocities().clone()
        current = full_pos.to(self.device)
        limits = self.robot.data.soft_joint_pos_limits
        coupled = apply_urdf_mimic_targets(
            current, self.robot.joint_names, limits[..., 0], limits[..., 1]
        )
        mimic_ids = list(self._v3_mimic_joint_ids.values())
        if self.cfg.v3_action_contract == "direct_flexion14_v1":
            self.robot.set_joint_position_target(coupled[:, mimic_ids], joint_ids=mimic_ids)
            return
        next_pos, delta = advance_mimic_positions(
            current,
            coupled,
            mimic_ids,
            float(self.cfg.v3_urdf_mimic_servo_rate_rad_s) * self.physics_dt,
        )
        full_pos[:, mimic_ids] = next_pos[:, mimic_ids].to(full_pos.device)
        if self.cfg.v3_action_contract == "direct_flexion14_v1" and self.cfg.v3_direct_servo_zero_velocity:
            full_vel[:, mimic_ids] = 0.0
        else:
            full_vel[:, mimic_ids] = (delta / self.physics_dt).to(full_vel.device)
        self.robot.root_physx_view.set_dof_positions(full_pos, self.robot._ALL_INDICES)
        self.robot.root_physx_view.set_dof_velocities(full_vel, self.robot._ALL_INDICES)
        self.robot.set_joint_position_target(next_pos[:, mimic_ids], joint_ids=mimic_ids)
        self.robot.root_physx_view.set_dof_position_targets(
            next_pos.to(full_pos.device), self.robot._ALL_INDICES
        )

    @torch.inference_mode()
    def _apply_direct_flexion_action(self) -> None:
        """Stage-1 low-level path for the independent v3 flexion contract.

        It deliberately reuses the inherited calibrated limits, backlash,
        rate limit, projection and position servo.  Stage-2 tactile and stem
        reflexes remain gated until P7; silently applying their legacy action
        indices to the new semantics would be unsafe.
        """

        if self.cfg.tactile_reflex_enabled or self.cfg.adaptive_retention_reflex_enabled:
            raise RuntimeError("legacy tactile reflexes are not mapped to direct_flexion14_v1")
        desired = v3_direct_action_to_joint_targets(self.actions) + self.actuator_bias
        if self.cfg.post_break_target_latch:
            new_break = self.stem_broke & ~self.post_break_targets_latched
            if bool(new_break.any()):
                self.post_break_targets[new_break] = self.prev_targets[new_break]
                self.post_break_targets_latched[new_break] = True
            post_break = self.stem_broke & self.post_break_targets_latched
            if bool(post_break.any()):
                mix = float(self.cfg.post_break_command_mix)
                desired = torch.where(
                    post_break.unsqueeze(-1),
                    self.post_break_targets * (1.0 - mix) + desired * mix,
                    desired,
                )
        requested = desired - self.prev_targets
        direction = requested.sign()
        reversing = (
            (direction != 0.0)
            & (self.last_target_direction != 0.0)
            & (direction != self.last_target_direction)
        )
        requested = torch.where(
            reversing,
            direction * (requested.abs() - self.actuator_backlash).clamp_min(0.0),
            requested,
        )
        requested = torch.where(
            requested.abs() < self.actuator_deadband,
            torch.zeros_like(requested),
            requested,
        )
        self.last_target_direction = torch.where(
            direction != 0.0, direction, self.last_target_direction
        )
        max_delta = (
            self._velocity.clamp_max(self.cfg.command_rate_limit_rad_s)
            * self.cfg.policy_dt_s
            * self.actuator_response_scale
        )
        targets = (self.prev_targets + requested.clamp(-max_delta, max_delta)).maximum(
            self._lower
        ).minimum(self._upper)
        self.prev_targets = targets
        dof_pos = self.robot.root_physx_view.get_dof_positions()[:, self.active_joint_ids].to(self.device)
        dof_vel = self.robot.root_physx_view.get_dof_velocities()[:, self.active_joint_ids].to(self.device)
        invalid = (dof_pos < self._lower) | (dof_pos > self._upper)
        if not self.cfg.joint_projection_after_break:
            invalid = invalid & (~self.stem_broke).unsqueeze(-1)
        if self.cfg.joint_projection_enabled and bool(invalid.any()):
            full_pos = self.robot.root_physx_view.get_dof_positions().clone()
            full_vel = self.robot.root_physx_view.get_dof_velocities().clone()
            corrected = dof_pos.maximum(self._lower).minimum(self._upper)
            full_pos[:, self.active_joint_ids] = corrected.to(full_pos.device)
            full_vel[:, self.active_joint_ids] = torch.where(
                invalid, torch.zeros_like(dof_vel), dof_vel
            ).to(full_vel.device)
            self.robot.root_physx_view.set_dof_positions(full_pos, self.robot._ALL_INDICES)
            self.robot.root_physx_view.set_dof_velocities(full_vel, self.robot._ALL_INDICES)
            self.joint_safety_projection += invalid.float().sum(dim=-1)
            dof_pos = corrected
        if self.cfg.position_servo_enabled:
            servo_step = self._velocity.clamp_max(self.cfg.position_servo_rate_rad_s) * self.physics_dt
            next_pos = dof_pos + (targets - dof_pos).clamp(-servo_step, servo_step)
            full_pos = self.robot.root_physx_view.get_dof_positions().clone()
            full_vel = self.robot.root_physx_view.get_dof_velocities().clone()
            full_pos[:, self.active_joint_ids] = next_pos.to(full_pos.device)
            if self.cfg.v3_direct_servo_zero_velocity:
                full_vel[:, self.active_joint_ids] = 0.0
            else:
                full_vel[:, self.active_joint_ids] = (
                    (next_pos - dof_pos) / self.physics_dt
                ).to(full_vel.device)
            self.robot.root_physx_view.set_dof_positions(full_pos, self.robot._ALL_INDICES)
            self.robot.root_physx_view.set_dof_velocities(full_vel, self.robot._ALL_INDICES)
            self.robot.root_physx_view.set_dof_position_targets(full_pos, self.robot._ALL_INDICES)
            self.robot.set_joint_effort_target(
                torch.zeros_like(targets), joint_ids=self.active_joint_ids
            )
        else:
            self.robot.set_joint_position_target(targets, joint_ids=self.active_joint_ids)
        self._apply_stem_wrench()

    @torch.inference_mode()
    def _read_contact_forces(self) -> torch.Tensor:
        """Keep shaping channels, but make tomato-filtered PhysX force strict.

        ``net_forces_w`` includes contacts with the palm insert and other hand
        geometry.  V3's physical gate must instead use ``force_matrix_w``,
        whose only configured filter is this environment's Tomato rigid body.
        """

        forces = super()._read_contact_forces()
        self._v3_unfiltered_link4_force = self._v2_measured_link4_force.clone()
        if not bool(self.cfg.v3_use_tomato_filtered_contact_force):
            return forces

        def filtered_force(name: str) -> torch.Tensor:
            history = self.scene.sensors[name].data.force_matrix_w_history
            if history is None:
                raise RuntimeError(f"contact sensor {name!r} has no tomato-filtered force matrix")
            # Preserve the peak physical force across the PhysX substeps in
            # the current 20 Hz control interval.  Reading only its final
            # substep aliases compliant contacts into isolated impulses.
            vector = history.sum(dim=(2, 3))
            return torch.linalg.vector_norm(vector, dim=-1).amax(dim=1).nan_to_num(0.0).clamp_min(0.0)

        pad_names = (
            "v3_thumb_pad_contact", "v3_index_pad_contact", "v3_middle_pad_contact",
            "v3_ring_pad_contact", "v3_little_pad_contact",
        )
        self._v2_measured_link4_force = torch.stack(
            tuple(
                filtered_force(name)
                for name in pad_names
            ),
            dim=-1,
        )
        return forces

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not getattr(self, "_v3_telemetry_ready", False):
            return
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
        with torch.inference_mode(False):
            for name in GATE_ORDER:
                self._v3_gate_streak[name] = self._v3_gate_streak[name].clone()
                self._v3_gate_max_streak[name] = self._v3_gate_max_streak[name].clone()
                self._v3_leave_one_out_streak[name] = self._v3_leave_one_out_streak[name].clone()
                self._v3_leave_one_out_max_streak[name] = self._v3_leave_one_out_max_streak[name].clone()
                self._v3_gate_streak[name][ids] = 0
                self._v3_gate_max_streak[name][ids] = 0
                self._v3_leave_one_out_streak[name][ids] = 0
                self._v3_leave_one_out_max_streak[name][ids] = 0
                setattr(self, f"_v3_snapshot_{name}_max_streak", self._v3_gate_max_streak[name])
                setattr(self, f"_v3_snapshot_without_{name}_max_streak", self._v3_leave_one_out_max_streak[name])
                self._v3_gate_current[name] = self._v3_gate_current[name].clone()
                self._v3_gate_current[name][ids] = False
            for name, fill in (
                ("_v3_all_conditions_streak", 0), ("_v3_all_conditions_max_streak", 0),
                ("_v3_first_three_physical", -1), ("_v3_first_five_physical", -1),
                ("_v3_first_strict_step", -1), ("_v3_first_break_step", -1),
                ("_v3_first_break_code", 0), ("_v3_terminal_failure_code", 0),
                ("_v3_first_five_failure_code", 0), ("_v3_best_gate_pass_count", 0),
                ("_v3_best_gate_step", -1), ("_v3_best_gate_failure_code", 0),
                ("_v3_physical_coverage_current", 0.0), ("_v3_physical_coverage_max", 0.0),
                ("_v3_physical_opposition_current", 1.0), ("_v3_physical_opposition_min", 1.0),
                ("_v3_physical_hemisphere_current", 1.0), ("_v3_physical_hemisphere_min", 1.0),
                ("_v3_physical_palm_load_share", 0.0), ("_v3_physical_five_current", False),
                ("_v3_strict_success_latched", False), ("_v3_previous_all_conditions", False),
            ):
                value = getattr(self, name).clone()
                value[ids] = fill
                setattr(self, name, value)
            for name, fill in (
                ("_v3_physical_digit_streak", 0),
                ("_v3_physical_digit_max_streak", 0),
                ("_v3_first_physical_digit_step", -1),
                ("_v3_max_simultaneous_physical", 0),
                ("_v3_physical_peak_force", 0.0),
                ("_v3_digit_recent", 0),
            ):
                value = getattr(self, name).clone()
                value[ids] = fill
                setattr(self, name, value)

    @torch.inference_mode()
    def _update_strict_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        previous_success = (
            self._v3_strict_success_latched.clone()
            if getattr(self, "_v3_telemetry_ready", False)
            else self.strict_success_latched.clone()
        )
        _legacy_strict, _legacy_new = super()._update_strict_geometry()
        if not getattr(self, "_v3_telemetry_ready", False):
            return _legacy_strict, _legacy_new

        threshold = float(self.cfg.contact_threshold_n)
        physical_active = self._v2_measured_link4_force >= threshold
        simultaneous = physical_active.sum(-1)
        if self.cfg.v3_sequential_contact_gate:
            # User-authorized v3.4.1 threshold adjustment: each finger counts
            # while it touched the fruit within the last persistence window,
            # so contacts may arrive sequentially instead of in one frame.
            # The hold streak still requires this union to persist step after
            # step, so a single transient touch cannot satisfy the window.
            self._v3_digit_recent = torch.where(
                physical_active,
                torch.full_like(self._v3_digit_recent, int(self.cfg.v3_contact_persistence_steps)),
                (self._v3_digit_recent - 1).clamp_min(0),
            )
            all_five = self._v3_digit_recent.gt(0).all(-1)
        else:
            all_five = simultaneous == 5
        step = self.episode_length_buf.to(torch.long)
        self._v3_physical_digit_streak = torch.where(
            physical_active,
            self._v3_physical_digit_streak + 1,
            torch.zeros_like(self._v3_physical_digit_streak),
        )
        self._v3_physical_digit_max_streak = torch.maximum(
            self._v3_physical_digit_max_streak, self._v3_physical_digit_streak
        )
        self._v3_first_physical_digit_step = torch.where(
            (self._v3_first_physical_digit_step < 0) & physical_active,
            step.unsqueeze(-1).expand_as(self._v3_first_physical_digit_step),
            self._v3_first_physical_digit_step,
        )
        self._v3_max_simultaneous_physical = torch.maximum(
            self._v3_max_simultaneous_physical, simultaneous
        )
        self._v3_physical_peak_force = torch.maximum(
            self._v3_physical_peak_force, self._v2_measured_link4_force
        )
        first_three_now = (self._v3_first_three_physical < 0) & (simultaneous >= 3)
        first_five_now = (self._v3_first_five_physical < 0) & all_five
        self._v3_first_three_physical = torch.where(first_three_now, step, self._v3_first_three_physical)
        self._v3_first_five_physical = torch.where(first_five_now, step, self._v3_first_five_physical)

        points, measured_points_valid = self._geometry_contact_points()
        center = self.tomato.data.root_pos_w
        normal = torch.zeros((self.num_envs, 3), device=self.device)
        normal[:, 2] = 1.0
        coverage, opposition, hemisphere = _contact_geometry_metrics(
            points.unsqueeze(0), physical_active.unsqueeze(0), center.unsqueeze(0), normal.unsqueeze(0)
        )
        coverage, opposition, hemisphere = coverage[0], opposition[0], hemisphere[0]
        eligible = simultaneous >= 3
        self._v3_physical_coverage_current = coverage
        self._v3_physical_coverage_max = torch.maximum(self._v3_physical_coverage_max, coverage)
        self._v3_physical_opposition_current = opposition
        opposition_eligible = eligible & physical_active[:, 0]
        self._v3_physical_opposition_min = torch.where(
            opposition_eligible, torch.minimum(self._v3_physical_opposition_min, opposition), self._v3_physical_opposition_min
        )
        self._v3_physical_hemisphere_current = hemisphere
        self._v3_physical_hemisphere_min = torch.where(
            eligible, torch.minimum(self._v3_physical_hemisphere_min, hemisphere), self._v3_physical_hemisphere_min
        )
        physical_force = self._v2_measured_link4_force
        load_share = self.palm_force / (self.palm_force + physical_force.sum(-1)).clamp_min(1.0e-6)
        self._v3_physical_palm_load_share = load_share
        tomato_pos = self.tomato.data.root_pos_w
        valid_support = (
            torch.linalg.vector_norm(tomato_pos - self.grasp_center, dim=-1) <= self.cfg.max_center_distance_m
        ) & (tomato_pos[:, 2] >= self.cfg.drop_height_m)
        margin = self._v3_first_three_physical - self.strict_first_palm
        palm_first = (
            (self.strict_first_palm >= 0)
            & (self._v3_first_three_physical >= 0)
            & (margin >= self.cfg.strict_palm_first_margin_steps)
        )
        gates = {
            "palm_up": normal[:, 2] >= self.cfg.strict_palm_up_cos_min,
            "palm_support": self.palm_force >= self.cfg.palm_support_force_n,
            "all_five_physical": all_five,
            "coverage": coverage >= self.cfg.strict_angular_coverage_min_deg,
            "opposition": physical_active[:, 0] & (opposition <= self.cfg.strict_thumb_opposition_cos_max),
            "hemisphere": hemisphere <= self.cfg.strict_same_hemisphere_score_max,
            "palm_load_share": (load_share >= self.cfg.strict_palm_load_share_min) & (load_share <= self.cfg.strict_palm_load_share_max),
            "force_safe": (physical_force <= self._ceiling.unsqueeze(0)).all(-1) & (self.palm_force <= 1.25 * self.cfg.safe_force_n),
            "valid_support": valid_support,
            "not_dropped": ~self.dropped,
            "distal_participation": self.strict_distal_participation.all(-1) & ~self.strict_proximal_only,
            "palm_first": palm_first,
        }
        for name, condition in gates.items():
            self._v3_gate_current[name] = condition.clone()
            current, maximum = update_streak(condition, self._v3_gate_streak[name], self._v3_gate_max_streak[name])
            self._v3_gate_streak[name], self._v3_gate_max_streak[name] = current, maximum
            setattr(self, f"_v3_snapshot_{name}_max_streak", maximum)

        gate_matrix = torch.stack([gates[name] for name in GATE_ORDER], dim=-1)
        for index, name in enumerate(GATE_ORDER):
            without = torch.cat((gate_matrix[:, :index], gate_matrix[:, index + 1 :]), dim=-1).all(-1)
            current, maximum = update_streak(
                without,
                self._v3_leave_one_out_streak[name],
                self._v3_leave_one_out_max_streak[name],
            )
            self._v3_leave_one_out_streak[name], self._v3_leave_one_out_max_streak[name] = current, maximum
            setattr(self, f"_v3_snapshot_without_{name}_max_streak", maximum)

        step_conditions = tuple(name for name in GATE_ORDER if name not in ("distal_participation", "palm_first"))
        all_conditions = torch.stack([gates[name] for name in step_conditions], dim=-1).all(-1)
        self._v3_all_conditions_streak, self._v3_all_conditions_max_streak = update_streak(
            all_conditions, self._v3_all_conditions_streak, self._v3_all_conditions_max_streak
        )
        first_strict_now = (self._v3_first_strict_step < 0) & all_conditions
        self._v3_first_strict_step = torch.where(first_strict_now, step, self._v3_first_strict_step)
        failure_code = encode_failed_gates(gates)
        self._v3_first_five_failure_code = torch.where(
            first_five_now, failure_code, self._v3_first_five_failure_code
        )
        pass_count = gate_matrix.sum(-1).to(torch.long)
        better = pass_count > self._v3_best_gate_pass_count
        self._v3_best_gate_pass_count = torch.where(better, pass_count, self._v3_best_gate_pass_count)
        self._v3_best_gate_step = torch.where(better, step, self._v3_best_gate_step)
        self._v3_best_gate_failure_code = torch.where(better, failure_code, self._v3_best_gate_failure_code)
        broke_now = (
            self._v3_previous_all_conditions
            & ~all_conditions
            & (self._v3_first_break_step < 0)
        )
        self._v3_first_break_step = torch.where(broke_now, step, self._v3_first_break_step)
        self._v3_first_break_code = torch.where(broke_now, failure_code, self._v3_first_break_code)
        self._v3_terminal_failure_code = failure_code
        self._v3_previous_all_conditions = all_conditions
        self._v3_physical_five_current = all_five

        ready = (
            (self._v3_all_conditions_streak >= self.cfg.strict_hold_steps)
            & gates["distal_participation"]
            & gates["palm_first"]
        )
        if self.cfg.strict_enveloping_require_stem_break:
            ready &= self.stem_broke
        newly = ready & ~previous_success
        self._v3_strict_success_latched = previous_success | ready
        # V3 is authoritative in this class.  Keep the common buffers aligned
        # so inherited termination/reward code cannot publish the old latch.
        self.strict_success_latched = self._v3_strict_success_latched.clone()
        self.strict_hold_streak = self._v3_all_conditions_streak.clone()
        self.strict_hold_max = self._v3_all_conditions_max_streak.clone()
        self.strict_coverage_current = coverage
        self.strict_coverage_max = self._v3_physical_coverage_max.clone()
        self.strict_opposition_current = opposition
        self.strict_opposition_min = self._v3_physical_opposition_min.clone()
        self.strict_hemisphere_current = hemisphere
        self.strict_hemisphere_min = self._v3_physical_hemisphere_min.clone()
        self.strict_contact_geometry_measured = measured_points_valid & physical_active
        self.strict_first_three = self._v3_first_three_physical.clone()
        return all_conditions, newly


__all__ = ["V3EnvelopingPhysxTomatoGraspEnv", "V3EnvelopingPhysxTomatoGraspEnvCfg"]
