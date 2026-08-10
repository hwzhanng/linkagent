"""Run isolated Physics-v1 contact and stem qualification gates."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "logs/qualification/tomato_physics_v1/isolated_physics_gate.json"
)


def _quat_y(angle_rad: float) -> tuple[float, float, float, float]:
    # Isaac Lab asset/spawner configuration uses quaternion order (x,y,z,w).
    return (0.0, math.sin(0.5 * angle_rad), 0.0, math.cos(0.5 * angle_rad))


def _rotate_y(vector: tuple[float, float, float], angle_rad: float):
    torch = importlib.import_module("torch")
    x, y, z = vector
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor((c * x + s * z, y, -s * x + c * z))


def main() -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    sim = None
    try:
        torch = importlib.import_module("torch")
        sim_utils = importlib.import_module("isaaclab.sim")
        assets = importlib.import_module("isaaclab.assets")
        physics = importlib.import_module("simulation.tomato_pick.physics_v1")
        profile = physics.load_physics_v1_profile()
        contact = profile["contact"]

        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=contact["tomato_static_friction"],
            dynamic_friction=contact["tomato_dynamic_friction"],
            restitution=contact["restitution"],
            friction_combine_mode=contact["friction_combine_mode"],
            restitution_combine_mode=contact["restitution_combine_mode"],
            compliant_contact_stiffness=contact[
                "compliant_contact_stiffness_n_m"
            ],
            compliant_contact_damping=contact[
                "compliant_contact_damping_n_s_m"
            ],
        )
        collision = sim_utils.CollisionPropertiesCfg(
            contact_offset=contact["contact_offset_m"],
            rest_offset=contact["rest_offset_m"],
        )
        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=contact[
                "maximum_depenetration_velocity_m_s"
            ],
            linear_damping=0.08,
            angular_damping=0.08,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        )
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
        )

        ramp_size = (1.2, 0.32, 0.02)
        puck_size = (0.05, 0.05, 0.02)
        ramp_angles_deg = (20.0, 35.0)
        pucks = []
        initial_positions = []
        downhill_axes = []
        for lane, angle_deg in enumerate(ramp_angles_deg):
            angle = math.radians(angle_deg)
            center = torch.tensor((0.0, float(lane) * 0.55, 0.45))
            ramp_cfg = sim_utils.CuboidCfg(
                size=ramp_size,
                collision_props=collision,
                physics_material=material,
            )
            ramp_cfg.func(
                f"/World/Incline{lane}",
                ramp_cfg,
                translation=tuple(float(v) for v in center),
                orientation=_quat_y(angle),
            )
            local_puck = (-0.25, 0.0, 0.5 * (ramp_size[2] + puck_size[2]))
            initial = center + _rotate_y(local_puck, angle)
            puck_cfg = assets.RigidObjectCfg(
                prim_path=f"/World/InclinePuck{lane}",
                spawn=sim_utils.CuboidCfg(
                    size=puck_size,
                    rigid_props=rigid,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                    collision_props=collision,
                    physics_material=material,
                ),
                init_state=assets.RigidObjectCfg.InitialStateCfg(
                    pos=tuple(float(v) for v in initial),
                    rot=_quat_y(angle),
                ),
            )
            pucks.append(assets.RigidObject(puck_cfg))
            initial_positions.append(initial.to(args.device))
            downhill_axes.append(_rotate_y((1.0, 0.0, 0.0), angle).to(args.device))

        radius = 0.03
        mass = 0.12
        floor_top = 0.25
        floor_cfg = sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.02),
            collision_props=collision,
            physics_material=material,
        )
        floor_cfg.func(
            "/World/ReboundFloor",
            floor_cfg,
            translation=(1.0, 0.0, floor_top - 0.01),
        )
        sphere_initial_z = floor_top + radius + 0.02
        sphere_cfg = assets.RigidObjectCfg(
            prim_path="/World/ReboundTomato",
            spawn=sim_utils.SphereCfg(
                radius=radius,
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=collision,
                physics_material=material,
            ),
            init_state=assets.RigidObjectCfg.InitialStateCfg(
                pos=(1.0, 0.0, sphere_initial_z)
            ),
        )
        sphere = assets.RigidObject(sphere_cfg)
        sim.reset()

        minimum_bottom_height = math.inf
        # The ratio includes the release state by definition; subsequent
        # samples must not exceed it by more than the numerical tolerance.
        maximum_energy_ratio = 1.0
        post_contact_peak_bottom_height = -math.inf
        contact_seen = False
        gravity = 9.81
        initial_energy = mass * gravity * (sphere_initial_z - floor_top)
        incline_displacements = None
        for step in range(240):
            sim.step(render=False)
            for puck in pucks:
                puck.update(sim.cfg.dt)
            sphere.update(sim.cfg.dt)
            position = sphere.data.root_pos_w.torch[0]
            linear = sphere.data.root_lin_vel_w.torch[0]
            angular = sphere.data.root_ang_vel_w.torch[0]
            bottom_height = float(position[2]) - radius - floor_top
            minimum_bottom_height = min(minimum_bottom_height, bottom_height)
            if bottom_height <= contact["contact_offset_m"]:
                contact_seen = True
            elif contact_seen:
                post_contact_peak_bottom_height = max(
                    post_contact_peak_bottom_height, bottom_height
                )
            translational = 0.5 * mass * float(torch.dot(linear, linear))
            inertia = 0.4 * mass * radius * radius
            rotational = 0.5 * inertia * float(torch.dot(angular, angular))
            potential = mass * gravity * max(float(position[2]) - floor_top, 0.0)
            # Subtract the constant energy at first geometric contact.
            mechanical = translational + rotational + potential - mass * gravity * radius
            maximum_energy_ratio = max(
                maximum_energy_ratio, mechanical / max(initial_energy, 1.0e-8)
            )
            # The incline test is a short onset test.  Leaving the pucks on a
            # finite ramp for the full rebound settling window would measure
            # edge departure rather than friction behavior.
            if step == 59:
                incline_displacements = []
                for initial, axis, puck in zip(
                    initial_positions, downhill_axes, pucks
                ):
                    delta = puck.data.root_pos_w.torch[0] - initial
                    incline_displacements.append(float(torch.dot(delta, axis)))
        if incline_displacements is None:
            raise RuntimeError("incline snapshot was not recorded")

        friction = {
            "static_friction": contact["tomato_static_friction"],
            "dynamic_friction": contact["tomato_dynamic_friction"],
            "analytic_static_onset_angle_deg": math.degrees(
                math.atan(contact["tomato_static_friction"])
            ),
            "low_angle_deg": ramp_angles_deg[0],
            "low_angle_displacement_m": incline_displacements[0],
            "high_angle_deg": ramp_angles_deg[1],
            "high_angle_displacement_m": incline_displacements[1],
        }
        rebound_height = (
            max(post_contact_peak_bottom_height, 0.0)
            if math.isfinite(post_contact_peak_bottom_height)
            else 0.0
        )
        contact_result = {
            "drop_height_m": 0.02,
            "minimum_bottom_height_m": minimum_bottom_height,
            "maximum_penetration_m": max(-minimum_bottom_height, 0.0),
            "post_contact_rebound_height_m": rebound_height,
            "maximum_mechanical_energy_ratio": maximum_energy_ratio,
        }

        samples = physics.sample_correlated_fruit_profiles(
            4096, generator=torch.Generator().manual_seed(42)
        )
        dt = 1.0 / 120.0
        critical = samples["stem_critical_duration_s"]
        attached = torch.ones(critical.shape, dtype=torch.bool)
        damage = torch.zeros_like(critical)
        damage, spike_broken = physics.update_stem_damage(
            damage,
            torch.ones_like(critical),
            dt_s=dt,
            critical_duration_s=critical,
            onset_ratio=profile["stem_prior"]["damage_onset_ratio"],
            recovery_time_s=profile["stem_prior"]["damage_recovery_time_s"],
            attached=attached,
        )
        break_step = torch.full(critical.shape, -1, dtype=torch.long)
        for step in range(1, 24):
            damage, broken = physics.update_stem_damage(
                damage,
                torch.ones_like(critical),
                dt_s=dt,
                critical_duration_s=critical,
                onset_ratio=profile["stem_prior"]["damage_onset_ratio"],
                recovery_time_s=profile["stem_prior"]["damage_recovery_time_s"],
                attached=attached,
            )
            newly_broken = broken & (break_step < 0)
            break_step[newly_broken] = step + 1
        recovery_damage, recovery_broken = physics.update_stem_damage(
            torch.full_like(critical, 0.5),
            torch.zeros_like(critical),
            dt_s=dt,
            critical_duration_s=critical,
            onset_ratio=profile["stem_prior"]["damage_onset_ratio"],
            recovery_time_s=profile["stem_prior"]["damage_recovery_time_s"],
            attached=attached,
        )
        stem = {
            "sample_count": int(critical.numel()),
            "one_substep_break_count": int(spike_broken.sum()),
            "sustained_break_count": int((break_step > 0).sum()),
            "minimum_break_step": int(break_step[break_step > 0].min()),
            "maximum_break_step": int(break_step.max()),
            "minimum_break_time_s": float(break_step[break_step > 0].min()) * dt,
            "maximum_break_time_s": float(break_step.max()) * dt,
            "subcritical_recovery_fraction": float(
                (recovery_damage < 0.5).float().mean()
            ),
            "subcritical_break_count": int(recovery_broken.sum()),
        }

        checks = {
            "low_angle_does_not_slide": abs(incline_displacements[0]) < 0.005,
            "high_angle_slides": incline_displacements[1] > 0.02,
            "contact_does_not_penetrate_excessively": (
                contact_result["maximum_penetration_m"] < 0.005
            ),
            "rebound_is_strongly_damped": rebound_height < 0.01,
            "contact_does_not_gain_energy": maximum_energy_ratio <= 1.05,
            "one_substep_does_not_break_stem": int(spike_broken.sum()) == 0,
            "sustained_load_breaks_all_stems": int((break_step > 0).sum())
            == int(critical.numel()),
            "subcritical_load_recovers": bool((recovery_damage < 0.5).all()),
            "subcritical_load_does_not_break": int(recovery_broken.sum()) == 0,
        }
        result = {
            "schema_version": 1,
            "profile_id": profile["profile_id"],
            "qualification": "literature_prior_provisional",
            "friction": friction,
            "compression_and_rebound": contact_result,
            "stem_duration": stem,
            "checks": checks,
            "passed": all(checks.values()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("TOMATO_PHYSICS_V1_ISOLATED=" + json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 2
    finally:
        if sim is not None:
            sim.clear_instance()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
