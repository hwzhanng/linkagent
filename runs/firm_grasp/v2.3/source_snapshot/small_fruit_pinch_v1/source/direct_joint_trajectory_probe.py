#!/usr/bin/env python3
"""连续验证三指直控候选的首接触顺序与保持物理。

仅左手仿真：先移动拇指对向轴、食中根部及侧摆，再同步闭合三个远端。
该脚本不训练 PPO，直控结果只用于决定动作空间如何最小解耦。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=320)
parser.add_argument("--coarse-steps", type=int, default=90)
parser.add_argument("--thumb-lead-steps", type=int, default=260)
parser.add_argument("--distal-steps", type=int, default=130)
parser.add_argument("--print-every", type=int, default=10)
parser.add_argument("--root-scale", type=float, default=1.0,
                    help="三指根部目标相对张开姿态的比例（0–1），用于碰撞限幅扫描")
parser.add_argument("--axis-only", type=int, default=None,
                    help="仅驱动一个 canonical 轴（0–15），用于单轴自碰撞回归")
parser.add_argument("--diameter-mm", type=float, default=70.0)
parser.add_argument("--center-base-mm", nargs=3, type=float, default=(75.736, -41.634, 164.405))
parser.add_argument(
    "--control", nargs=10, type=float,
    default=(-0.91586, -0.23098, 0.22568, 0.65065, 0.56853, 0.32217, -0.22014, 0.56886, 0.94374, 0.89589),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.legacy_bridge import load_legacy_hand_manifold
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets


class DirectTrajectoryProbeEnv(NativeSmallFruitPinchEnv):
    def _postprocess_joint_targets(self, targets: torch.Tensor) -> torch.Tensor:
        override = getattr(self, "_probe_joint_targets", None)
        return targets if override is None else override

    def _get_dones(self):
        super()._get_dones()
        return (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

    def _apply_action(self) -> None:
        super()._apply_action()
        pose = getattr(self, "_probe_fruit_pose", None)
        if pose is not None:
            ids = torch.arange(self.num_envs, device=self.device)
            self._tomato.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            self._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((self.num_envs, 6), device=self.device), env_ids=ids
            )


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    if args.coarse_steps <= 0 or args.thumb_lead_steps <= 0 or args.distal_steps <= 0:
        raise ValueError("阶段步数必须为正数")
    if not 0.0 <= args.root_scale <= 1.0:
        raise ValueError("root-scale 必须位于 [0,1]")
    if args.axis_only is not None and not 0 <= args.axis_only < 16:
        raise ValueError("axis-only 必须位于 0–15")
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.episode_length_s = max(30.0, args.steps / 20.0 + 5.0)
    radius = float(args.diameter_mm) / 2000.0
    cfg.events.fruit_geometry.params["axis_ranges_m"] = {
        axis: (radius, radius) for axis in ("x", "y", "z")
    }
    env = DirectTrajectoryProbeEnv(cfg=cfg)
    try:
        env.reset(seed=20260933)
        device = env.device
        ids = torch.arange(1, device=device)
        center = torch.tensor(args.center_base_mm, device=device) / 1000.0
        old_center = env._tomato.data.root_pos_w.torch.clone()
        fruit_pose = env._tomato.data.default_root_pose.torch.clone()
        fruit_pose[:, :3] = env.scene.env_origins + center
        fruit_pose[:, 3:7] = fruit_pose.new_tensor((0.0, 0.0, 0.0, 1.0))
        env._probe_fruit_pose = fruit_pose
        env._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=ids)
        env._tomato.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=device), env_ids=ids
        )
        delta = fruit_pose[:, :3] - old_center
        env._stem_rest_attachment_pos_w += delta
        env._stem_plant_anchor_pos_w += delta
        env._pinch_reference_center_w.copy_(fruit_pose[:, :3])

        legacy = load_legacy_hand_manifold()
        lower = torch.tensor(legacy.LEFT_SAFE_LOWER_RAD, device=device)
        upper = torch.tensor(legacy.LEFT_SAFE_UPPER_RAD, device=device)
        selected = torch.tensor((0, 1, 2, 5, 6, 7, 10, 11, 12, 13), device=device)
        control = torch.tensor(args.control, device=device)
        final = pinch_action_to_joint_targets(torch.zeros((1, 14), device=device))
        final[:, selected] = lower[selected] + 0.5 * (control + 1.0) * (
            upper[selected] - lower[selected]
        )
        opened = pinch_action_to_joint_targets(torch.zeros((1, 14), device=device))
        root_indices = torch.tensor((0, 1, 2, 5, 6, 7, 10, 11), device=device)
        final[:, root_indices] = opened[:, root_indices] + args.root_scale * (
            final[:, root_indices] - opened[:, root_indices]
        )
        if args.axis_only is not None:
            isolated = opened.clone()
            isolated[:, args.axis_only] = final[:, args.axis_only]
            final = isolated
        preopposed = opened.clone()
        # 根部/侧摆先到位；三个远端仍保持张开，防止指中段扫过果面。
        coarse_indices = torch.tensor((0, 2, 4, 5, 7, 9), device=device)
        preopposed[:, coarse_indices] = final[:, coarse_indices]
        thumb_ready = preopposed.clone()
        thumb_ready[:, (0, 10, 11)] = final[:, (0, 10, 11)]
        zero_action = torch.zeros((1, 14), device=device)
        peak = torch.zeros((1, 3), device=device)

        for step in range(args.steps):
            if step < args.coarse_steps:
                alpha = smoothstep((step + 1) / args.coarse_steps)
                target = opened + alpha * (preopposed - opened)
                phase = "preopposition"
            elif step < args.coarse_steps + args.thumb_lead_steps:
                alpha = smoothstep((step - args.coarse_steps + 1) / args.thumb_lead_steps)
                target = preopposed + alpha * (thumb_ready - preopposed)
                phase = "thumb_lead"
            elif step < args.coarse_steps + args.thumb_lead_steps + args.distal_steps:
                alpha = smoothstep(
                    (step - args.coarse_steps - args.thumb_lead_steps + 1)
                    / args.distal_steps
                )
                target = thumb_ready + alpha * (final - thumb_ready)
                phase = "finger_distal_close"
            else:
                target = final
                phase = "hold"
            env._probe_joint_targets = target
            _, _, terminated, truncated, _ = env.step(zero_action)
            force = env._last_physical_force_n[:, :3]
            peak = torch.maximum(peak, force)
            if step % max(1, args.print_every) == 0 or bool(terminated[0] or truncated[0]):
                current = env._robot.data.joint_pos.torch[:, env._hand_joint_ids]
                max_target_error = torch.abs(current - target).amax(dim=1)
                print(
                    f"step={step:03d} phase={phase} "
                    f"force={[round(float(v),4) for v in force[0]]} "
                    f"raw={[round(float(v),4) for v in env._raw_finger_force_n[0,:3]]} "
                    f"axial={[round(float(v),4) for v in env._contact_region_axial_progress[0,:3]]} "
                    f"proximal={[round(float(v),4) for v in env._forbidden_proximal_force_n[0]]} "
                    f"forbidden_first={bool(env._forbidden_first_contact[0])} "
                    f"thumb_index_margin_mm={round(float(env._thumb_index_surface_margin_m[0])*1000.0,3)} "
                    f"max_target_error_rad={round(float(max_target_error[0]),4)} "
                    f"term={bool(terminated[0])} trunc={bool(truncated[0])}",
                    flush=True,
                )
            if bool(terminated[0] or truncated[0]):
                break
        print(f"peak_force_n={[round(float(v),4) for v in peak[0]]}", flush=True)
        tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids[:3]] - env.scene.env_origins[:, None, :]
        print("final_tip_positions_mm", (tips[0] * 1000.0).detach().cpu().tolist(), "fruit_center_mm", (center * 1000.0).detach().cpu().tolist(), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
