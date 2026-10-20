"""Real-physics diagnostics; these do not establish learned policy quality."""
import json
import time
from pathlib import Path

import torch


@torch.no_grad()
def free_hand_probe(env, output):
    raw = env.unwrapped
    env.reset()
    initial = raw._robot.data.joint_pos.torch.clone()
    pose = raw._fruit.data.root_pose_w.torch.clone()
    pose[:, 2] += 3
    raw._fruit.write_root_pose_to_sim_index(root_pose=pose)
    for _ in range(120):
        raw._apply_action()
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(raw.cfg.sim.dt)
    final = raw._robot.data.joint_pos.torch.clone()
    result = {"joint_names":raw._robot.joint_names,
              "initial":initial.tolist(),"final":final.tolist(),
              "target":raw._q_target.tolist(),
              "max_joint_deviation_per_env":(final-initial).abs().amax(-1).tolist(),
              "net_contact_forces":{k:v.data.net_forces_w.torch.tolist() for k,v in raw._sensors.items()}}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"free_hand_max_deviation": result["max_joint_deviation_per_env"]}),flush=True)


@torch.no_grad()
def validate_wrapper(wrapped):
    """An eval-only reset must preserve training physics and observation history."""
    wrapped.reset(hard=True)
    raw = wrapped._unwrapped
    obs, *_ = wrapped.step(torch.zeros_like(raw._action))
    start = wrapped.num_eval_envs
    before = {k: v[:][start:].clone() for k, v in obs["policy"].items()}
    joints = raw._robot.data.joint_pos.torch[start:].clone()
    fruit = raw._fruit.data.root_pose_w.torch[start:].clone()
    pressure_history = raw._pressure_prev[start:].clone()
    after, _ = wrapped.reset_eval_envs()
    for k, v in before.items():
        if not torch.equal(v, after["policy"][k][:][start:]):
            raise RuntimeError(f"Evaluation reset changed training history: {k}")
    if not (torch.equal(joints, raw._robot.data.joint_pos.torch[start:])
            and torch.equal(fruit, raw._fruit.data.root_pose_w.torch[start:])
            and torch.equal(pressure_history, raw._pressure_prev[start:])):
        raise RuntimeError("Evaluation reset changed training physics or pressure history")
    result = {"eval_reset_isolation_passed": True,
              "stacked_shapes": {k: list(v[:].shape) for k, v in after["policy"].items()}}
    print(json.dumps(result), flush=True)
    return result


@torch.no_grad()
def physical_probe(env, steps, output=None):
    raw = env.unwrapped
    reports = {}
    start = time.monotonic()
    for mode in ("hold", "random"):
        obs, _ = env.reset()
        initial = {
            "fruit": raw._fruit.data.root_pos_w.torch[:1].tolist(),
            "contact_bodies": raw._robot.data.body_pos_w.torch[:1, raw._contact_body_ids].tolist(),
            "active_joints": raw._robot.data.joint_pos.torch[:1, raw._active_ids].tolist(),
        }
        age = torch.zeros(raw.num_envs, device=raw.device)
        support_per_env = torch.zeros_like(age)
        endings_per_env = torch.zeros_like(age)
        max_age = torch.zeros_like(age)
        lifetimes, samples, snapshots = [], [], []
        count = 0
        for step in range(steps):
            action = torch.zeros_like(raw._action)
            if mode == "random":
                action.uniform_(-1, 1)
            obs, reward, failed, timeout, info = env.step(action)
            if not all(torch.isfinite(x).all() for x in (
                *obs["policy"].values(), reward, raw._dense, raw._sparse,
                raw._robot.data.joint_pos.torch, raw._fruit.data.root_pose_w.torch,
                raw._track.J, raw._track.spin_J,
            )):
                raise RuntimeError(f"Nonfinite physical probe {mode} step {step}")
            age += 1
            max_age = torch.maximum(max_age, age)
            done = failed | timeout
            endings_per_env += done.float()
            lifetimes.extend(age[done].cpu().tolist())
            age[done] = 0
            count += int(done.sum())
            metrics = info["step"]
            support_per_env += metrics["secure_support"].float()
            samples.append(torch.stack((
                metrics["secure_support"].float().mean(),
                metrics["peak_pressure_pa"].mean(),
                metrics["rotation_rate"].abs().mean(), reward.mean(),
                raw._track.ready.float().mean(),
                raw._track.last_accepted.float().mean(),
                raw._forces.amax(-1).mean(),
            )).cpu())
            if step in (0, 1, 4, 9, 19, 49, 99, steps - 1):
                snapshots.append({"step": step, "forces_env0": raw._forces[0].tolist(),
                                  "joints_env0": raw._robot.data.joint_pos.torch[0].tolist(),
                                  "fruit_velocity_env0": raw._fruit.data.root_vel_w.torch[0].tolist(),
                                  "fruit_env0": raw._fruit.data.root_pos_w.torch[0].tolist(),
                                  "failed_count": int(failed.sum())})
        mean = torch.stack(samples).mean(0).tolist()
        reports[mode] = dict(zip(("secure_support_fraction", "mean_peak_pressure_pa",
                                 "mean_abs_spin_rad_s", "mean_reward", "track_ready_fraction",
                                 "track_acceptance_fraction", "mean_max_contact_force_n"), mean))
        reports[mode].update({"steps_per_env": steps, "num_envs": raw.num_envs,
                              "completed_episodes": count,
                              "mean_completed_episode_steps": sum(lifetimes) / max(1, len(lifetimes)),
                              "longest_completed_episode_steps": max(lifetimes, default=0),
                              "longest_ongoing_episode_steps": float(age.max()),
                              "initial": initial, "snapshots": snapshots})
        reports[mode]["per_env"] = {"support_fraction": (support_per_env / steps).tolist(),
                                   "endings": endings_per_env.tolist(), "longest_episode_steps": max_age.tolist()}
        print(json.dumps({mode: reports[mode]}, indent=2), flush=True)
    reports["elapsed_seconds"] = time.monotonic() - start
    reports["finite_rollout_passed"] = True
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2) + "\n")
    return reports
