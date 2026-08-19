"""Convert the vendor left-hand URDF with the Isaac Sim 4.5 importer.

IsaacLab 2.3's ``UrdfFileCfg`` targets the newer 2.4 importer API, while the
installed Isaac Sim 4.5 runtime exposes 2.3.10.  This small compatibility
entry point uses only the methods present in the installed importer and writes
an ordinary USD artifact that the PhysX environment can load without runtime
conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import omni.kit.commands


def main() -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, import_cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    import_cfg.set_distance_scale(1.0)
    import_cfg.set_make_default_prim(True)
    import_cfg.set_create_physics_scene(False)
    import_cfg.set_density(0.0)
    import_cfg.set_convex_decomp(True)
    import_cfg.set_collision_from_visuals(False)
    import_cfg.set_merge_fixed_joints(False)
    import_cfg.set_fix_base(True)
    import_cfg.set_self_collision(False)
    import_cfg.set_parse_mimic(False)
    import_cfg.set_override_joint_dynamics(False)
    import_cfg.set_import_inertia_tensor(True)
    import_cfg.set_default_drive_strength(60.0)
    import_cfg.set_default_position_drive_damping(5.0)

    ok, robot_model = omni.kit.commands.execute(
        "URDFParseFile", urdf_path=str(args.urdf.resolve()), import_config=import_cfg
    )
    if not ok:
        raise RuntimeError(f"URDFParseFile failed: {args.urdf}")
    ok, _ = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_path=str(args.urdf.resolve()),
        urdf_robot=robot_model,
        import_config=import_cfg,
        dest_path=str(args.output.resolve()),
    )
    if not ok:
        raise RuntimeError(f"URDFImportRobot failed: {args.output}")
    print(f"[physx-asset] wrote {args.output.resolve()}", flush=True)


try:
    main()
except BaseException as exc:
    print(f"[physx-asset] fatal: {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    app.close()
