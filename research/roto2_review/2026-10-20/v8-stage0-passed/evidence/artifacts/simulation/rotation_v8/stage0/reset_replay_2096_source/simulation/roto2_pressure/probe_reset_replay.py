"""Controlled diagnostic of settled-state reset, never a policy controller."""
from pathlib import Path
import json
import numpy as np
import torch
from .calibrate_redundancy import step_physics, save
from .core import quat_inv_rotate


@torch.no_grad()
def probe_reset_replay(env, output, horizon=160):
    raw = env.unwrapped
    if raw.num_envs % 4:
        raise ValueError("Reset replay probe requires a multiple of four environments")
    output = Path(output)
    sources = list(raw._reset_grasps)
    groups = torch.arange(raw.num_envs, device=raw.device) % 4
    originals = []
    for i in range(raw.num_envs):
        c = sources[(i // 4) % len(sources)]
        originals.append(dict(
            id=c.get("id", str(i // 4)),
            joint_positions=c.get("source_initial_joint_positions", c["joint_positions"]),
            fruit_position_hand=c.get("source_initial_fruit_position_hand", c["fruit_position_hand"]),
            hand_posture="palm_up"))
    raw._reset_grasps = originals
    env.reset()
    initial_center = raw._center_reference_w.clone()
    target = raw._q_target.clone()

    def hold(target):
        traces = {k: [] for k in ("pad", "stable", "failed", "pose", "whole_finger_force", "speed", "peak_pressure")}
        for _ in range(horizon):
            d = step_physics(raw, target)
            for k in traces:
                traces[k].append(d[k].cpu().numpy().copy())
        return {k: np.asarray(v) for k, v in traces.items()}

    before = hold(target)
    q_all = raw._robot.data.joint_pos.torch.clone()
    qd_all = raw._robot.data.joint_vel.torch.clone()
    q_target = raw._q_target.clone()
    fruit_pose = raw._fruit.data.root_pose_w.torch.clone()
    fruit_velocity = raw._fruit.data.root_vel_w.torch.clone()
    robot_pose = raw._robot.data.root_pose_w.torch
    fruit_local = quat_inv_rotate(raw._wxyz(robot_pose[:, 3:]), fruit_pose[:, :3] - robot_pose[:, :3])
    fruit_quat = fruit_pose[:, 3:] if raw._native_xyzw else fruit_pose[:, [4, 5, 6, 3]]
    rows = []
    for i in range(raw.num_envs):
        row = dict(id=f"replay_{i}", joint_positions=q_all[i, raw._active_ids].cpu().tolist(),
                   joint_targets=q_target[i].cpu().tolist(),
                   fruit_position_hand=fruit_local[i].cpu().tolist(),
                   fruit_quaternion_xyzw=fruit_quat[i].cpu().tolist())
        rows.append(originals[i] if int(groups[i]) == 0 else row)
    raw._reset_grasps = rows
    env.reset()
    full_ids = torch.where(groups >= 2)[0]
    velocity_ids = torch.where(groups == 3)[0]
    # Diagnostic reset only. During holds, unchanged actuators/physics apply.
    raw._robot.write_joint_position_to_sim_index(
        position=q_all[full_ids], joint_ids=raw._all_joint_ids, env_ids=full_ids)
    raw._robot.write_joint_velocity_to_sim_index(
        velocity=qd_all[velocity_ids], joint_ids=raw._all_joint_ids, env_ids=velocity_ids)
    raw._fruit.write_root_velocity_to_sim_index(
        root_velocity=fruit_velocity[velocity_ids], env_ids=velocity_ids)
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(dt=0.0)
    target = raw._q_target.clone()
    after = hold(target)
    variants = ["original_proposal_again", "settled_active_zero_velocity",
                "settled_all_joints_zero_velocity", "settled_all_joints_measured_velocity"]
    summaries = []
    group_np = groups.cpu().numpy()
    for i, variant in enumerate(variants):
        mask = group_np == i
        eligible = (before["pad"][-20:, mask].mean(0) >= .8).sum(-1) >= 4
        replay_four = (after["pad"][-20:, mask].mean(0) >= .8).sum(-1) >= 4
        replay_safe = after["stable"][20:, mask].all(0) & ~after["failed"][:, mask].any(0)
        summaries.append(dict(variant=variant, environments=int(mask.sum()),
                              original_terminal_four_pad=int(eligible.sum()),
                              replay_terminal_four_pad=int(replay_four.sum()),
                              retained_four_pad_among_original=int((eligible & replay_four).sum()),
                              retained_four_pad_and_safe_among_original=int((eligible & replay_four & replay_safe).sum())))
    np.savez_compressed(output.with_suffix(".npz"),
                        **{"before_" + k: v for k, v in before.items()},
                        **{"after_" + k: v for k, v in after.items()},
                        groups=group_np, settled_all_joint_positions=q_all.cpu().numpy(),
                        settled_all_joint_velocities=qd_all.cpu().numpy(),
                        settled_joint_targets=q_target.cpu().numpy(),
                        settled_fruit_pose=fruit_pose.cpu().numpy(),
                        settled_fruit_velocity=fruit_velocity.cpu().numpy(),
                        initial_center=initial_center.cpu().numpy())
    save(output, dict(seed=raw.cfg.seed, horizon=horizon, num_envs=raw.num_envs,
                      source_grasps=len(sources), variants=summaries,
                      joint_names=list(raw._robot.joint_names), probe_completed=True,
                      stage0_passed=False, note="Matched reset-state ablation; not a release test or PPO training"))
    print(json.dumps(summaries), flush=True)
