"""Deterministic contact/geometry smoke test for the palm-up tomato env.

Three phases, printed as a compact report:

1. rest: zero action; the tomato must settle on the palm (palm contact force
   above the support gate, tomato z near the rest height, no penetration).
2. close: a scripted ramp to a full enclosure action; per-digit PhysX contact
   forces and the secure latch are recorded.
3. pick: while secure (or forced with ``--force-pick``), the fixed slow pick
   trajectory runs and the stem tension ratio, damage, break event, and
   post-break hold are recorded.

The probe validates step-A acceptance: scene loads, contact sensors read
non-zero physical forces, the palm supports the resting tomato, and the stem
wrench/break chain is physically coherent.  It is not a success-rate claim;
that is the step-B scripted feasibility gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120, help="Policy steps in the rest phase")
parser.add_argument("--close-steps", type=int, default=160, help="Policy steps in the close phase (joint slew needs ~110 steps for full flexion)")
parser.add_argument("--pick-steps", type=int, default=120, help="Policy steps in the pick phase")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--force-pick", action="store_true", help="Latch secure so the pick trajectory runs even if the scripted close does not reach it")
parser.add_argument("--ramp-to", type=float, default=1.0, help="Close action value ramped in the close phase")
parser.add_argument(
    "--close-action",
    type=str,
    default=None,
    help="Optional ten-value final action, e.g. '0.8,0.7,0,0,-0.6,0,0,0,0,0'; linearly ramped",
)
parser.add_argument(
    "--sequence",
    type=str,
    default=None,
    help="Optional staged close sequence 'a0,...,a9:steps;...'; each target is slew-ramped",
)
parser.add_argument("--deterministic", action="store_true", help="Disable sensor/object randomization and position jitter")
parser.add_argument("--diagnose", action="store_true", help="Dump base_link collision state and sensor buffers, then exit")
parser.add_argument("--trace", action="store_true", help="Print a state line after every pick-phase step")
parser.add_argument("--placements", type=str, default=None, help="Semicolon-separated tomato xy placements 'x1,y1;x2,y2'; for each, reset and run a scripted close, then report")
parser.add_argument("--close-value", type=float, default=1.0, help="Scripted close action used by --placements")
parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=1, help="Curriculum stage: 0 physics, 1 enclosure hold, 2 stem pick")
parser.add_argument("--adaptive-close", action="store_true", help="Use force-aware residual closure instead of one fixed action")
parser.add_argument("--search-actions", type=int, default=0, help="Evaluate this many deterministic 10D action candidates in parallel")
parser.add_argument("--search-seed", type=int, default=20260812, help="Deterministic candidate-search seed")
parser.add_argument("--search-top-k", type=int, default=10, help="Maximum ranked candidates printed by action search")
parser.add_argument("--local-search", action="store_true", help="Perturb the known reachable five-digit prior instead of sampling the full action box")
parser.add_argument("--candidate-actions", type=str, default=None, help="Semicolon-separated explicit 10D candidates; count must equal --num_envs")
parser.add_argument("--rest-xy", type=str, default=None, help="Override task-frame tomato centre as 'x,y'")
parser.add_argument("--axis-m", type=float, default=None, help="Use one deterministic tomato semi-axis for geometry feasibility")
parser.add_argument("--actuator-effort-nm", type=float, default=None, help="Override the explicit hand actuator effort limit for an auditable probe")
parser.add_argument("--actuator-stiffness", type=float, default=None, help="Override explicit hand actuator stiffness for an auditable probe")
parser.add_argument("--actuator-damping", type=float, default=None, help="Override explicit hand actuator damping for an auditable probe")
parser.add_argument("--tomato-angular-damping", type=float, default=None, help="Override tomato rigid-body angular damping for an auditable probe")
parser.add_argument("--tomato-linear-damping", type=float, default=None, help="Override tomato rigid-body linear damping for an auditable probe")
parser.add_argument("--retention-seek-increment", type=float, default=None, help="Override per-digit post-break tactile closure increment")
parser.add_argument("--retention-global-increment", type=float, default=None, help="Override post-break global closure increment")
parser.add_argument("--retention-global-limit", type=float, default=None, help="Override maximum post-break global closure boost")
parser.add_argument("--controller-progress-per-step", type=float, default=None, help="Override coarse adaptive-controller approach rate")
parser.add_argument("--force-feedback-gain", type=float, default=None, help="Override per-digit target-force feedback gain")
parser.add_argument("--residual-span-fraction", type=float, default=None, help="Override per-digit residual joint-range fraction")
parser.add_argument("--friction", type=float, default=None, help="Override deterministic tomato static friction")
parser.add_argument("--original-palm-collision", action="store_true", help="Audit the vendor palm collision without the flat support replacement")
parser.add_argument("--stem-linear-frequency", type=float, default=None, help="Override stem linear natural frequency in rad/s")
parser.add_argument("--support-only-debug", action="store_true", help="Disable imported hand colliders while retaining the palm support insert")
parser.add_argument("--collision-digit", choices=("thumb", "index", "middle", "ring", "little"), default=None, help="With --support-only-debug, re-enable one digit's imported colliders")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoGraspEnv, PhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp.stage0_scripted_controller import (
    Stage0ControllerCfg,
    Stage0ScriptedController,
)


def main() -> None:
    print("[tomato-probe] building config", flush=True)
    cfg = PhysxTomatoGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.training_stage = args.stage
    if args.stem_linear_frequency is not None:
        if args.stem_linear_frequency <= 0.0:
            raise ValueError("--stem-linear-frequency must be positive")
        cfg.stem_linear_natural_frequency_rad_s = args.stem_linear_frequency
    actuator = cfg.robot_cfg.actuators["left_hand"]
    if args.actuator_effort_nm is not None:
        if args.actuator_effort_nm <= 0.0:
            raise ValueError("--actuator-effort-nm must be positive")
        actuator.effort_limit_sim = args.actuator_effort_nm
    if args.actuator_stiffness is not None:
        if args.actuator_stiffness <= 0.0:
            raise ValueError("--actuator-stiffness must be positive")
        actuator.stiffness = args.actuator_stiffness
    if args.actuator_damping is not None:
        if args.actuator_damping <= 0.0:
            raise ValueError("--actuator-damping must be positive")
        actuator.damping = args.actuator_damping
    if args.tomato_angular_damping is not None:
        if args.tomato_angular_damping < 0.0:
            raise ValueError("--tomato-angular-damping must be non-negative")
        cfg.tomato_cfg.spawn.rigid_props.angular_damping = args.tomato_angular_damping
    if args.tomato_linear_damping is not None:
        if args.tomato_linear_damping < 0.0:
            raise ValueError("--tomato-linear-damping must be non-negative")
        cfg.tomato_cfg.spawn.rigid_props.linear_damping = args.tomato_linear_damping
    if args.residual_span_fraction is not None:
        if not 0.0 <= args.residual_span_fraction <= 0.25:
            raise ValueError("--residual-span-fraction must lie in [0, 0.25]")
        cfg.residual_span_fraction = args.residual_span_fraction
    if args.axis_m is not None:
        if not 0.025 <= args.axis_m <= 0.035:
            raise ValueError("--axis-m must lie in [0.025, 0.035]")
        cfg.events["tomato_geometry"].params["axis_range_m"] = (args.axis_m, args.axis_m)
    cfg.debug_support_only_collision = args.support_only_debug
    cfg.debug_collision_digit = args.collision_digit or ""
    cfg.use_original_palm_collision = args.original_palm_collision
    if args.rest_xy is not None:
        rest_values = tuple(float(value) for value in args.rest_xy.split(","))
        if len(rest_values) != 2:
            raise ValueError("--rest-xy requires x,y")
        cfg.tomato_rest_xy = rest_values
    if args.deterministic:
        cfg.position_jitter = (0.0, 0.0, 0.0)
        cfg.sensor_delay_steps_max = 0
        cfg.sensor_dropout_probability = 0.0
        cfg.sensor_stale_probability = 0.0
        cfg.tactile_gain_range = (1.0, 1.0)
        cfg.tactile_bias_range_n = (0.0, 0.0)
        cfg.friction_range = (0.70, 0.70)
        cfg.restitution_range = (0.0, 0.0)
        cfg.stiffness_range_n_m = (18000.0, 18000.0)
        cfg.stem_tensile_limit_n = (0.70, 0.70)
        cfg.stem_critical_duration_s = (0.20, 0.20)
    if args.friction is not None:
        if args.friction <= 0.0:
            raise ValueError("--friction must be positive")
        cfg.friction_range = (args.friction, args.friction)
    env = PhysxTomatoGraspEnv(cfg)
    n = args.num_envs
    print("[tomato-probe] env created", flush=True)
    if args.placements is not None:
        results = []
        if args.close_action is None:
            action = torch.full((n, cfg.action_space), args.close_value, device=env.device)
        else:
            values = [float(value.strip()) for value in args.close_action.split(",")]
            if len(values) != cfg.action_space:
                raise ValueError(f"--close-action requires {cfg.action_space} comma-separated values")
            action = torch.tensor(values, device=env.device).expand(n, -1).clone().clamp(-1.0, 1.0)
        for label, placement in enumerate(args.placements.split(";")):
            x, y = (float(v) for v in placement.split(","))
            env.cfg.tomato_rest_xy = (x, y)
            env.reset()
            peak = torch.zeros((n, 5), device=env.device)
            palm_max = torch.zeros(n, device=env.device)
            for step in range(args.close_steps):
                env.step(action)
                peak = torch.maximum(peak, env.contact_forces)
                palm_max = torch.maximum(palm_max, env.palm_force)
            results.append({
                "placement": (x, y),
                "peak_digit_n": peak.mean(dim=0).detach().cpu().tolist(),
                "digits_contacting": int((peak >= cfg.contact_threshold_n).float().sum(dim=1).max()),
                "palm_max_n": float(palm_max.max()),
                "tomato_z": float(env.tomato.data.root_pos_w[:, 2].mean()),
                "tomato_xyz": env.tomato.data.root_pos_w[0].detach().cpu().tolist(),
                "xy_displacement_cm": float(
                    torch.linalg.vector_norm(
                        env.tomato.data.root_pos_w[0, :2]
                        - env.scene.env_origins[0, :2]
                        - torch.tensor((x, y), device=env.device)
                    ) * 100.0
                ),
                "speed_m_s": float(torch.linalg.vector_norm(env.tomato.data.root_lin_vel_w[0])),
                "secure": bool(env.secure_latched.any()),
            })
            print(f"[tomato-probe] placement {label} xy=({x},{y}): {results[-1]}", flush=True)
        print({"placements": results}, flush=True)
        raise SystemExit(0)
    if args.search_actions or args.candidate_actions is not None:
        search_count = args.search_actions or n
        if search_count != n:
            raise ValueError("--search-actions must equal --num_envs so each environment evaluates one candidate")
        generator = torch.Generator(device=env.device).manual_seed(args.search_seed)
        # Bias the search toward the physically plausible closure band while
        # retaining independent spread/shape/residual variation.  The coarse
        # thumb coordinate is deliberately sampled at the opposition end;
        # this is a feasibility search, not a policy shortcut.
        if args.candidate_actions is not None:
            rows = [[float(value) for value in item.split(",")] for item in args.candidate_actions.split(";")]
            if len(rows) != n or any(len(row) != cfg.action_space for row in rows):
                raise ValueError("--candidate-actions must provide one 10D row per environment")
            candidates = torch.tensor(rows, device=env.device).clamp(-1.0, 1.0)
        else:
            if args.local_search:
                centre = torch.tensor(
                    (0.30, 0.00, -0.03, -0.79, 1.00, 0.00, -0.79, 0.54, -0.89, -0.02),
                    device=env.device,
                )
                scale = torch.tensor(
                    (0.10, 0.12, 0.12, 0.12, 0.04, 0.20, 0.15, 0.15, 0.10, 0.20),
                    device=env.device,
                )
                noise = torch.randn((n, cfg.action_space), device=env.device, generator=generator)
                candidates = (centre + scale * noise).clamp(-1.0, 1.0)
                candidates[0] = centre
            else:
                candidates = torch.empty((n, cfg.action_space), device=env.device).uniform_(-0.15, 0.85, generator=generator)
                candidates[:, 0] = torch.empty(n, device=env.device).uniform_(0.20, 0.75, generator=generator)
                candidates[:, 1] = torch.empty(n, device=env.device).uniform_(-0.35, 0.65, generator=generator)
                candidates[:, 2] = torch.empty(n, device=env.device).uniform_(-0.65, 0.65, generator=generator)
                candidates[:, 3] = torch.empty(n, device=env.device).uniform_(-0.80, 0.80, generator=generator)
                candidates[:, 4] = torch.empty(n, device=env.device).uniform_(0.55, 1.0, generator=generator)
                candidates[:, 5:] = torch.empty((n, 5), device=env.device).uniform_(-0.95, 0.95, generator=generator)
        env.reset()
        zero = torch.zeros_like(candidates)
        for _ in range(args.steps):
            env.step(zero)
        peak = torch.zeros((n, 5), device=env.device)
        contact_seen = torch.zeros((n, 5), dtype=torch.bool, device=env.device)
        max_simultaneous = torch.zeros(n, dtype=torch.long, device=env.device)
        all_five_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        max_all_five_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        palm_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
        success_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        first_termination_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
        first_termination_code = torch.zeros(n, dtype=torch.long, device=env.device)
        first_all_five_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
        palm_peak = torch.zeros(n, device=env.device)
        for step in range(args.close_steps):
            fraction = min(1.0, float(step + 1) / max(1, args.close_steps // 2))
            _, _, terminated, truncated, _ = env.step(candidates * fraction)
            done = terminated | truncated
            first_done = alive & done
            first_termination_step = torch.where(
                first_done,
                torch.full_like(first_termination_step, step + 1),
                first_termination_step,
            )
            first_termination_code = torch.where(
                first_done, env.last_termination_code, first_termination_code
            )
            # DirectRLEnv automatically resets completed environments.  Keep
            # each candidate's statistics confined to its first episode so a
            # reset/open hand cannot be mistaken for the candidate's final
            # grasp state.  The terminal step is retained; later steps are not.
            sample_mask = alive.unsqueeze(-1)
            peak = torch.where(sample_mask, torch.maximum(peak, env.contact_forces), peak)
            active = env.contact_forces >= cfg.contact_threshold_n
            contact_seen |= active & sample_mask
            simultaneous = active.sum(dim=-1)
            first_all_five = alive & (simultaneous == 5) & (first_all_five_step < 0)
            first_all_five_step = torch.where(
                first_all_five,
                torch.full_like(first_all_five_step, step + 1),
                first_all_five_step,
            )
            max_simultaneous = torch.where(
                alive, torch.maximum(max_simultaneous, simultaneous), max_simultaneous
            )
            all_five_streak = torch.where(
                alive & (simultaneous == 5) & (env.palm_force >= cfg.palm_support_force_n),
                all_five_streak + 1,
                torch.zeros_like(all_five_streak),
            )
            max_all_five_streak = torch.maximum(max_all_five_streak, all_five_streak)
            palm_seen |= alive & (env.palm_force >= cfg.palm_support_force_n)
            palm_peak = torch.where(alive, torch.maximum(palm_peak, env.palm_force), palm_peak)
            success_seen |= alive & env.success
            alive &= ~done
        safe = peak <= torch.tensor(cfg.force_ceiling_n, device=env.device)
        score = (
            max_simultaneous.float()
            + 0.1 * max_all_five_streak.float()
            + 5.0 * success_seen.float()
            + 0.25 * palm_seen.float()
            - 0.25 * peak.clamp_min(0.0).sum(dim=-1) / 5.0
            - 2.0 * (first_termination_step >= 0).float()
        )
        order = torch.argsort(score, descending=True)
        results = []
        for index in order.tolist():
            results.append({
                "env": int(index),
                "score": float(score[index]),
                "action": candidates[index].detach().cpu().tolist(),
                "contact_seen": contact_seen[index].detach().cpu().tolist(),
                "max_simultaneous_digits": int(max_simultaneous[index]),
                "max_all_five_streak": int(max_all_five_streak[index]),
                "palm_support_seen": bool(palm_seen[index]),
                "terminated_during_search": bool(first_termination_step[index] >= 0),
                "first_termination_step": int(first_termination_step[index]),
                "first_termination_code": int(first_termination_code[index]),
                "first_all_five_step": int(first_all_five_step[index]),
                "final_state_is_same_episode": bool(alive[index]),
                "peak_palm_n": float(palm_peak[index]),
                "peak_digit_n": peak[index].detach().cpu().tolist(),
                "final_joint_pos": env.robot.data.joint_pos[index, env.active_joint_ids].detach().cpu().tolist(),
                "final_joint_target": env.prev_targets[index].detach().cpu().tolist(),
                "final_tip_distance_cm": (
                    torch.linalg.vector_norm(
                        env._tip_positions()[index] - env.tomato.data.root_pos_w[index], dim=-1
                    ) * 100.0
                ).detach().cpu().tolist(),
                "final_tip_xyz_local": (
                    env._tip_positions()[index]
                    - env.scene.env_origins[index].unsqueeze(0)
                ).detach().cpu().tolist(),
                "final_tomato_xyz": env.tomato.data.root_pos_w[index].detach().cpu().tolist(),
                "safe_peak": bool(safe[index].all()),
                "success": bool(success_seen[index]),
            })
        if args.search_top_k <= 0:
            raise ValueError("--search-top-k must be positive")
        print({"action_search": results[: args.search_top_k]}, flush=True)
        raise SystemExit(0)
    if args.diagnose:
        from pxr import Usd, UsdPhysics, UsdGeom

        stage = env.sim.stage
        for name in ("base_link", "thumb_link5", "index_link4"):
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
            info = {"schemas": sorted(prim.GetAppliedSchemas())}
            children = []
            for child in prim.GetChildren():
                if "ollision" in child.GetName():
                    mc = UsdPhysics.MeshCollisionAPI(child)
                    approx = None
                    try:
                        approx = mc.GetApproximationAttr().Get()
                    except Exception:
                        pass
                    children.append(
                        {"path": str(child.GetPath()), "type": child.GetTypeName(),
                         "schemas": sorted(child.GetAppliedSchemas()),
                         "approximation": approx,
                         "collision_enabled": UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()}
                    )
            info["collision_children"] = children
            print(f"[tomato-probe] diagnose {name}: {info}", flush=True)
        palm = env.scene.sensors["palm_contact"]
        print(f"[tomato-probe] diagnose palm sensor net_forces_w shape={tuple(palm.data.net_forces_w.shape)} sample={palm.data.net_forces_w[0,0].detach().cpu().tolist()}", flush=True)
        names = ("base_link", "thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")
        ids = [env.robot.body_names.index(name) for name in names]
        print(f"[tomato-probe] diagnose body_pos_w={env.robot.data.body_pos_w[0, ids].detach().cpu().tolist()}", flush=True)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for name in names:
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
            if prim.IsValid():
                bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                print(f"[tomato-probe] diagnose {name} bbox_min={bbox.GetMin()} bbox_max={bbox.GetMax()}", flush=True)
        print(f"[tomato-probe] diagnose tomato prim path={env.tomato.cfg.prim_path}", flush=True)
        tomato = stage.GetPrimAtPath("/World/envs/env_0/Tomato")
        from pxr import UsdGeom
        xf = UsdGeom.Xformable(tomato)
        print(f"[tomato-probe] diagnose tomato xform ops: {[op.GetOpType().name for op in xf.GetOrderedXformOps()]}", flush=True)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                print(f"[tomato-probe] diagnose tomato scale op = {op.Get()}", flush=True)
        for child in tomato.GetChildren():
            if "ollision" in child.GetName() or "mesh" in child.GetName():
                mc = UsdPhysics.MeshCollisionAPI(child)
                approx = None
                try:
                    approx = mc.GetApproximationAttr().Get()
                except Exception:
                    pass
                print(f"[tomato-probe] diagnose tomato {child.GetName()}: type={child.GetTypeName()} schemas={sorted(child.GetAppliedSchemas())} approximation={approx}", flush=True)
                # World bounds of the collision mesh extent.
                extent = child.GetAttribute("extent").Get()
                print(f"[tomato-probe] diagnose tomato {child.GetName()} extent={extent}", flush=True)
        raise SystemExit(0)
    obs, _ = env.reset()
    print("[tomato-probe] env reset done", flush=True)
    reset_transforms = env.tomato.root_physx_view.get_transforms().detach().cpu()
    print(
        f"[tomato-probe] reset root_data={env.tomato.data.root_pos_w.detach().cpu().tolist()} "
        f"physx_transform={reset_transforms.tolist()}",
        flush=True,
    )
    report: dict[str, object] = {"num_envs": n, "phases": {}}

    # Phase 1: rest.  Zero action; the tomato settles on the palm.
    zero = torch.zeros((n, cfg.action_space), device=env.device)
    palm_force_samples = torch.zeros((n, args.steps), device=env.device)
    tomato_z = torch.zeros((n, args.steps), device=env.device)
    for step in range(args.steps):
        env.step(zero)
        palm_force_samples[:, step] = env.palm_force
        tomato_z[:, step] = env.tomato.data.root_pos_w[:, 2]
        if step % 10 == 0:
            pos = env.tomato.data.root_pos_w.detach().cpu()
            vel = getattr(env.tomato.data, "root_lin_vel_w", torch.zeros((n, 3), device=env.device)).detach().cpu()
            quat = env.tomato.data.root_quat_w.detach().cpu()
            print(f"[tomato-probe] rest step {step}/{args.steps} palm={env.palm_force.detach().cpu().tolist()} "
                  f"pos={[f'({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})' for p in pos]} "
                  f"quat={[f'({q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f},{q[3]:+.2f})' for q in quat]} "
                  f"speed={[f'{v:.3f}' for v in torch.linalg.vector_norm(vel, dim=-1).tolist()]} "
                  f"stem={[f'{v:.2f}' for v in env.stem_load_ratios['combined'].detach().cpu().tolist()]} "
                  f"stem_t={env.stem_load_ratios['tension'].detach().cpu().tolist()} "
                  f"anchor={env.anchor_pos.detach().cpu().tolist()}", flush=True)
    report["phases"]["rest"] = {
        "palm_force_mean_n": float(palm_force_samples.mean()),
        "palm_force_max_n": float(palm_force_samples.max()),
        "palm_support_ok": bool((palm_force_samples.max(dim=-1).values >= cfg.palm_support_force_n).all()),
        "tomato_z_initial": float(tomato_z[:, 0].mean()),
        "tomato_z_final": float(tomato_z[:, -1].mean()),
        "tomato_drift_cm": float((tomato_z[:, -1] - tomato_z[:, 0]).abs().max() * 100.0),
        "stem_tension_ratio_max": float(env.stem_load_ratios["tension"].max()),
    }

    # Phase 2: close.  Ramp to a full enclosure action.
    action = torch.zeros((n, cfg.action_space), device=env.device)
    sequence = None
    if args.sequence is not None:
        sequence = []
        for item in args.sequence.split(";"):
            values_text, steps_text = item.rsplit(":", 1)
            values = [float(value.strip()) for value in values_text.split(",")]
            if len(values) != cfg.action_space:
                raise ValueError(f"--sequence targets require {cfg.action_space} values")
            steps = int(steps_text)
            if steps <= 0:
                raise ValueError("--sequence phase lengths must be positive")
            sequence.append((torch.tensor(values, device=env.device).clamp(-1.0, 1.0), steps))
        close_steps = sum(steps for _, steps in sequence)
    else:
        close_steps = args.close_steps
    if args.close_action is None:
        close_target = torch.full((n, cfg.action_space), args.ramp_to, device=env.device)
    else:
        values = [float(value.strip()) for value in args.close_action.split(",")]
        if len(values) != cfg.action_space:
            raise ValueError(f"--close-action requires {cfg.action_space} comma-separated values")
        close_target = torch.tensor(values, device=env.device).expand(n, -1).clone().clamp(-1.0, 1.0)
    peak = torch.zeros((n, 5), device=env.device)
    close_contact = torch.zeros((n, 5), dtype=torch.bool, device=env.device)
    close_success_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
    close_max_stable_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    close_first_done_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    close_first_done_code = torch.zeros(n, dtype=torch.long, device=env.device)
    close_alive = torch.ones(n, dtype=torch.bool, device=env.device)
    controller_overrides = {}
    if args.retention_seek_increment is not None:
        controller_overrides["retention_seek_increment"] = args.retention_seek_increment
    if args.retention_global_increment is not None:
        controller_overrides["retention_global_increment"] = args.retention_global_increment
    if args.retention_global_limit is not None:
        controller_overrides["retention_global_limit"] = args.retention_global_limit
    if args.controller_progress_per_step is not None:
        controller_overrides["progress_per_step"] = args.controller_progress_per_step
    if args.force_feedback_gain is not None:
        controller_overrides["force_feedback_gain_per_n"] = args.force_feedback_gain
    adaptive_controller = Stage0ScriptedController(
        n,
        env.device,
        Stage0ControllerCfg(
            contact_threshold_n=cfg.contact_threshold_n,
            palm_support_force_n=cfg.palm_support_force_n,
            target_force_n=cfg.target_force_n,
            **controller_overrides,
        ),
    )
    previous_target = torch.zeros(cfg.action_space, device=env.device)
    phase_index = 0
    phase_step = 0
    for step in range(close_steps):
        if args.adaptive_close and sequence is None:
            action.copy_(adaptive_controller.command(
                env.contact_forces,
                env.palm_force,
                retention_mode=env.stem_broke,
                center_error_m=env.tomato.data.root_pos_w - env.grasp_center,
            ))
        elif sequence is not None:
            target, phase_length = sequence[phase_index]
            fraction = min(1.0, float(phase_step + 1) / max(1, min(30, phase_length // 3)))
            action.copy_(previous_target + (target - previous_target) * fraction)
        else:
            fraction = min(1.0, float(step + 1) / max(1, close_steps // 2))
            action.copy_(close_target * fraction)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        first_done = close_alive & done
        close_first_done_step = torch.where(
            first_done, torch.full_like(close_first_done_step, step + 1), close_first_done_step
        )
        close_first_done_code = torch.where(
            first_done, env.last_termination_code, close_first_done_code
        )
        sample = close_alive.unsqueeze(-1)
        peak = torch.where(sample, torch.maximum(peak, env.contact_forces), peak)
        close_contact |= (env.contact_forces >= cfg.contact_threshold_n) & sample
        close_success_seen |= close_alive & env.success
        close_success_seen |= first_done & (env.last_termination_code == 1)
        close_max_stable_steps = torch.where(
            close_alive,
            torch.maximum(close_max_stable_steps, env.stable_steps),
            close_max_stable_steps,
        )
        close_alive &= ~done
        if step % 20 == 0:
            print(f"[tomato-probe] close step {step}/{close_steps} contact={env.contact_forces.mean(dim=0).detach().cpu().tolist()}", flush=True)
            if args.trace:
                tip_pos = env.robot.data.body_pos_w[0, env.tip_ids].detach().cpu().tolist()
                print(f"[tomato-probe] close trace targets={env.prev_targets[0].detach().cpu().tolist()} tips={tip_pos}", flush=True)
        if sequence is not None:
            phase_step += 1
            if phase_step >= sequence[phase_index][1]:
                previous_target = sequence[phase_index][0]
                phase_index += 1
                phase_step = 0
        # Stage 2 owns the fixed pick trajectory.  Once every still-live
        # environment has latched its verified enclosure, hand control must
        # continue in the pick/retention loop immediately; running the
        # arbitrary remainder of the close horizon used to hide a break and
        # auto-reset before the probe began measuring the pick phase.
        if args.stage >= 2 and bool((env.pick_started | ~close_alive).all()):
            break
    report["phases"]["close"] = {
        "action_value": args.ramp_to,
        "peak_contact_n": peak.mean(dim=0).detach().cpu().tolist(),
        "digits_contacting_mean": float(close_contact.float().mean()),
        "secure_latched": bool(env.secure_latched.all()),
        "success_seen": bool(close_success_seen.all()),
        "max_stable_steps": close_max_stable_steps.detach().cpu().tolist(),
        "first_done_step": close_first_done_step.detach().cpu().tolist(),
        "first_done_code": close_first_done_code.detach().cpu().tolist(),
        "pick_started": bool(env.pick_started.all()),
        "palm_force_n": float(env.palm_force.mean()),
        "final_tip_distance_cm": (
            torch.linalg.vector_norm(
                env._tip_positions() - env.tomato.data.root_pos_w[:, None, :], dim=-1
            ).mean(dim=0) * 100.0
        ).detach().cpu().tolist(),
        "final_tomato_xyz": env.tomato.data.root_pos_w.mean(dim=0).detach().cpu().tolist(),
    }
    if args.force_pick and not bool(env.secure_latched.all()):
        env.secure_latched[:] = True
        print("[tomato-probe] secure forced for the pick phase", flush=True)

    # Phase 3: pick.  The anchor rises; stem loads and break are recorded.
    pick_ratio = torch.zeros((n, args.pick_steps), device=env.device)
    pick_damage = torch.zeros((n, args.pick_steps), device=env.device)
    break_at = torch.full((n,), -1, dtype=torch.long, device=env.device)
    pick_alive = torch.ones(n, dtype=torch.bool, device=env.device)
    pick_success_seen = close_success_seen.clone()
    pick_break_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
    pick_max_hold = torch.zeros(n, dtype=torch.long, device=env.device)
    pick_first_done_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    pick_first_done_code = torch.zeros(n, dtype=torch.long, device=env.device)
    for step in range(args.pick_steps):
        if args.adaptive_close and sequence is None:
            action.copy_(adaptive_controller.command(
                env.contact_forces,
                env.palm_force,
                retention_mode=env.stem_broke,
                center_error_m=env.tomato.data.root_pos_w - env.grasp_center,
            ))
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        first_done = pick_alive & done
        pick_first_done_step = torch.where(
            first_done, torch.full_like(pick_first_done_step, step), pick_first_done_step
        )
        pick_first_done_code = torch.where(
            first_done, env.last_termination_code, pick_first_done_code
        )
        pick_ratio[:, step] = torch.where(
            pick_alive, env.stem_load_ratios["combined"], torch.zeros_like(env.stem_damage)
        )
        pick_damage[:, step] = torch.where(
            pick_alive, env.stem_damage, torch.zeros_like(env.stem_damage)
        )
        new_break = pick_alive & env.last_break_event
        break_at = torch.where(new_break & (break_at < 0), torch.full_like(break_at, step), break_at)
        pick_break_seen |= new_break
        pick_success_seen |= pick_alive & env.success
        pick_success_seen |= first_done & (env.last_termination_code == 1)
        pick_max_hold = torch.where(
            pick_alive, torch.maximum(pick_max_hold, env.stable_steps), pick_max_hold
        )
        pick_alive &= ~done
        if args.trace and (step % 10 == 0 or bool(first_done.any())):
            print({
                "pick_step": step,
                "anchor_z": float(env.anchor_pos[0, 2]),
                "tomato_xyz": env.tomato.data.root_pos_w[0].detach().cpu().tolist(),
                "tension_ratio": float(env.stem_load_ratios["tension"][0]),
                "combined_ratio": float(env.stem_load_ratios["combined"][0]),
                "stem_damage": float(env.stem_damage[0]),
                "attached": bool(env.stem_attached[0]),
                "palm_force": float(env.palm_force[0]),
                "digit_force": env.contact_forces[0].detach().cpu().tolist(),
                "secure": bool(env.secure_latched[0]),
                "stable_steps": int(env.stable_steps[0]),
                "success": bool(env.success[0]),
                "damage_score": float(env.damage_score[0]),
                "relative_speed": float(env._relative_speed()[0]),
                "center_distance_cm": float(
                    torch.linalg.vector_norm(
                        env.tomato.data.root_pos_w[0] - env.grasp_center[0]
                    ) * 100.0
                ),
                "done": bool(done[0]),
                "termination_code": int(env.last_termination_code[0]),
            }, flush=True)
    report["phases"]["pick"] = {
        "anchor_rise_cm": float(env.pick_progress_m.max() * 100.0),
        "peak_combined_ratio": float(pick_ratio.max()),
        "stem_broke_any": bool(pick_break_seen.any()),
        "break_policy_step": break_at.detach().cpu().tolist(),
        "max_stem_damage": float(pick_damage.max()),
        "post_break_hold_steps_max": int(pick_max_hold.max()),
        "first_done_step": pick_first_done_step.detach().cpu().tolist(),
        "first_done_code": pick_first_done_code.detach().cpu().tolist(),
        "success_any": bool(pick_success_seen.any()),
        "final_palm_force_n": float(env.palm_force.mean()),
        "final_center_distance_cm": float(
            torch.linalg.vector_norm(env.tomato.data.root_pos_w - env.grasp_center, dim=-1).mean() * 100.0
        ),
    }
    print(report, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f"[tomato-probe] fatal: {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        app.close()
