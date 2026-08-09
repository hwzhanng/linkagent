"""Overlay the Physics-v1 reset workspace point cloud on the real Isaac scene."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DIAGNOSTIC_DIR = (
    PROJECT_ROOT / "logs/diagnostics/tomato_physics_v1_reset_workspace"
)
DEFAULT_CSV = (
    DEFAULT_DIAGNOSTIC_DIR / "physics_v1_reset_workspace_seed42_1024.csv"
)
DEFAULT_SCREENSHOT = (
    DEFAULT_DIAGNOSTIC_DIR / "physics_v1_reset_workspace_kit_overlay.png"
)


def _load_cloud(path: Path, maximum_points: int) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    palms: list[tuple[float, ...]] = []
    fruits: list[tuple[float, ...]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            palms.append(
                (float(row["palm_x_m"]), float(row["palm_y_m"]), float(row["palm_z_m"]))
            )
            fruits.append(
                (float(row["fruit_x_m"]), float(row["fruit_y_m"]), float(row["fruit_z_m"]))
            )
            if len(palms) >= maximum_points:
                break
    if not palms:
        raise RuntimeError(f"point cloud is empty: {path}")
    return palms, fruits


def _translated(
    points: list[tuple[float, ...]], origin: tuple[float, float, float]
) -> list[tuple[float, float, float]]:
    return [
        (point[0] + origin[0], point[1] + origin[1], point[2] + origin[2])
        for point in points
    ]


def main() -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--pair-line-stride", type=int, default=16)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument(
        "--stay-open",
        action="store_true",
        help="keep the interactive Kit viewport open after taking the screenshot",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless and not args.enable_cameras:
        raise ValueError("headless capture requires --enable_cameras")
    if args.max_points < 64:
        raise ValueError("at least 64 points are required to show the workspace envelope")
    if args.pair_line_stride < 1:
        raise ValueError("--pair-line-stride must be positive")

    cloud_csv = args.cloud_csv.expanduser().resolve()
    screenshot = args.screenshot.expanduser().resolve()
    if not cloud_csv.is_file():
        raise FileNotFoundError(cloud_csv)
    local_palms, local_fruits = _load_cloud(cloud_csv, args.max_points)

    launcher = AppLauncher(args)
    app = launcher.app
    raw_env = None
    rgb_annotator = None
    render_product = None
    try:
        from isaacsim.core.rendering_manager import ViewportManager
        import omni.replicator.core as rep
        import omni.usd
        import numpy as np
        from PIL import Image
        from pxr import Gf, UsdGeom, Vt

        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        task = importlib.import_module("simulation.tomato_pick.isaac.physics_v1")
        cfg_module = importlib.import_module(
            "simulation.tomato_pick.isaac.physics_v1.env_cfg"
        )

        cfg = cfg_module.TomatoPickPhysicsV1EnvCfg()
        cfg.scene.num_envs = 1
        cfg.seed = args.seed
        cfg.sim.device = args.device
        raw_env = gym.make(task.TASK_ID, cfg=cfg, render_mode="human")
        env = raw_env.unwrapped
        observation, _ = raw_env.reset(seed=args.seed)
        if observation["policy"].shape != (1, 66):
            raise RuntimeError("Physics-v1 actor observation shape drifted")
        # One zero-delta policy step publishes the reset PhysX/Fabric transforms
        # to RTX.  Without it, the tensors are already reset while the viewport
        # can still show the nominal pre-reset robot pose.
        observation, _, _, _, _ = raw_env.step(
            torch.zeros((1, 11), device=env.device)
        )
        if observation["policy"].shape != (1, 66):
            raise RuntimeError("Physics-v1 actor observation shape drifted after sync")

        env_origin_tensor = env.scene.env_origins[0]
        origin = tuple(float(value) for value in env_origin_tensor)
        palms = _translated(local_palms, origin)
        fruits = _translated(local_fruits, origin)
        actual_palm = tuple(
            float(value)
            for value in env._robot.data.body_pos_w.torch[0, env._ee_body_id]
        )
        actual_fruit = tuple(float(value) for value in env._tomato.data.root_pos_w.torch[0])

        all_points = torch.tensor(palms + fruits, device=env.device)
        lower = all_points.min(dim=0).values
        upper = all_points.max(dim=0).values
        target = 0.5 * (lower + upper)
        extent = upper - lower
        eye = target + torch.tensor(
            (
                max(0.90, float(extent[0]) * 1.75),
                max(0.95, float(extent[1]) * 1.90),
                max(0.42, float(extent[2]) * 0.82),
            ),
            device=env.device,
        )
        camera_eye = tuple(float(value) for value in eye)
        camera_target = tuple(float(value) for value in target)
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=list(camera_eye),
            target=list(camera_target),
        )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Kit USD stage is unavailable")
        root = UsdGeom.Xform.Define(stage, "/World/Diagnostics/ResetWorkspace")
        root.GetPrim().SetMetadata("comment", "Physics-v1 reset workspace overlay")

        def define_points(path, positions, rgb, width, opacity):
            points_prim = UsdGeom.Points.Define(stage, path)
            points_prim.CreatePointsAttr(
                Vt.Vec3fArray([Gf.Vec3f(*position) for position in positions])
            )
            points_prim.CreateWidthsAttr(Vt.FloatArray([width] * len(positions)))
            points_prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
            points_prim.CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))
            extent = UsdGeom.PointBased.ComputeExtent(
                Vt.Vec3fArray([Gf.Vec3f(*position) for position in positions])
            )
            points_prim.CreateExtentAttr(extent)

        define_points(
            "/World/Diagnostics/ResetWorkspace/PalmCandidates",
            palms,
            (0.05, 0.55, 1.00),
            0.010,
            0.58,
        )
        define_points(
            "/World/Diagnostics/ResetWorkspace/FruitCandidates",
            fruits,
            (1.00, 0.30, 0.03),
            0.014,
            0.70,
        )
        pair_indices = range(0, len(palms), args.pair_line_stride)
        line_starts = [palms[index] for index in pair_indices]
        line_ends = [fruits[index] for index in pair_indices]
        lines = UsdGeom.BasisCurves.Define(
            stage, "/World/Diagnostics/ResetWorkspace/SamplePairs"
        )
        line_positions = [
            Gf.Vec3f(*position)
            for pair in zip(line_starts, line_ends)
            for position in pair
        ]
        lines.CreateTypeAttr(UsdGeom.Tokens.linear)
        lines.CreateBasisAttr(UsdGeom.Tokens.bezier)
        lines.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        lines.CreateCurveVertexCountsAttr(Vt.IntArray([2] * len(line_starts)))
        lines.CreatePointsAttr(Vt.Vec3fArray(line_positions))
        lines.CreateWidthsAttr(Vt.FloatArray([0.0015]))
        lines.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        lines.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.90, 0.90, 0.95)]))
        lines.CreateDisplayOpacityAttr(Vt.FloatArray([0.25]))
        define_points(
            "/World/Diagnostics/ResetWorkspace/CurrentPalm",
            [actual_palm],
            (0.10, 1.00, 0.20),
            0.030,
            1.0,
        )
        define_points(
            "/World/Diagnostics/ResetWorkspace/CurrentFruit",
            [actual_fruit],
            (1.00, 0.95, 0.05),
            0.030,
            1.0,
        )
        current_pair = UsdGeom.BasisCurves.Define(
            stage, "/World/Diagnostics/ResetWorkspace/CurrentPair"
        )
        current_pair.CreateTypeAttr(UsdGeom.Tokens.linear)
        current_pair.CreateBasisAttr(UsdGeom.Tokens.bezier)
        current_pair.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        current_pair.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
        current_pair.CreatePointsAttr(
            Vt.Vec3fArray([Gf.Vec3f(*actual_palm), Gf.Vec3f(*actual_fruit)])
        )
        current_pair.CreateWidthsAttr(Vt.FloatArray([0.004]))
        current_pair.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        current_pair.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.20, 1.00, 0.25)]))

        render_product = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach([render_product])
        captured_rgb = None
        for _ in range(args.warmup_frames):
            env.sim.render()
            app.update()
            raw_rgb = rgb_annotator.get_data()
            if isinstance(raw_rgb, dict):
                raw_rgb = raw_rgb.get("data", np.array([], dtype=np.uint8))
            raw_rgb = np.asarray(raw_rgb, dtype=np.uint8)
            if raw_rgb.size:
                if raw_rgb.ndim == 1:
                    raw_rgb = raw_rgb.reshape(720, 1280, -1)
                candidate = raw_rgb[:, :, :3]
                if float(candidate.std()) > 2.0:
                    captured_rgb = candidate.copy()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        screenshot.unlink(missing_ok=True)
        deadline = time.monotonic() + 20.0
        while captured_rgb is None and time.monotonic() < deadline:
            env.sim.render()
            app.update()
            raw_rgb = rgb_annotator.get_data()
            if isinstance(raw_rgb, dict):
                raw_rgb = raw_rgb.get("data", np.array([], dtype=np.uint8))
            raw_rgb = np.asarray(raw_rgb, dtype=np.uint8)
            if raw_rgb.size:
                if raw_rgb.ndim == 1:
                    raw_rgb = raw_rgb.reshape(720, 1280, -1)
                candidate = raw_rgb[:, :, :3]
                if float(candidate.std()) > 2.0:
                    captured_rgb = candidate.copy()
        if captured_rgb is None:
            raise RuntimeError("RTX camera stayed blank after the warmup window")
        Image.fromarray(captured_rgb).save(screenshot)

        print(f"KIT_RESET_CLOUD_SCREENSHOT={screenshot}", flush=True)
        print(
            "KIT_RESET_CLOUD_LEGEND="
            "blue=palm candidates, orange=fruit candidates, "
            "green/yellow=current palm/fruit",
            flush=True,
        )
        if args.stay_open and not args.headless:
            print("[INFO] Interactive Kit viewport is open; close the window to exit.", flush=True)
            while app.is_running():
                app.update()
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if rgb_annotator is not None and render_product is not None:
            rgb_annotator.detach([render_product])
        if render_product is not None:
            render_product.destroy()
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
