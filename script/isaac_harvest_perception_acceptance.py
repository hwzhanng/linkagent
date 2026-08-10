"""Validate Ark-box plus rendered RGB-D target geometry in a live Isaac scene."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/simulation/harvest_cycle_acceptance/perception"),
    )
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _load_local_credential(path: Path, name: str) -> None:
    """Load one named local value without logging or copying it to artifacts."""

    if os.environ.get(name):
        return
    if not path.is_file():
        raise RuntimeError(f"{name} is not configured and {path.name} is missing")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        prepared = value.strip()
        if len(prepared) >= 2 and prepared[0] == prepared[-1] and prepared[0] in {'"', "'"}:
            prepared = prepared[1:-1]
        if prepared:
            os.environ[name] = prepared
            return
    raise RuntimeError(f"{name} is not configured")


def main() -> int:
    args = _parser().parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import gymnasium as gym
        import torch

        from module.skills.tomato_vision import ReplayTomatoVisionPerceptionPort
        from proto import Observation
        from script.tomato_vision_worker_ark import ArkVlmBoxDepthBackend
        from simulation.harvest_cycle import HarvestCycleIsaacEnvCfg
        from simulation.harvest_cycle.perception import publish_isaac_rgbd
        from simulation.harvest_cycle.robotics import IsaacHarvestRoboticsBundle

        repo_root = Path(__file__).resolve().parents[1]
        _load_local_credential(repo_root / ".env.local", "ARK_API_KEY")
        output_root = args.output_root.expanduser().resolve()
        spool_root = output_root / "spool"
        spool_root.mkdir(parents=True, exist_ok=True)

        cfg = HarvestCycleIsaacEnvCfg()
        cfg.scene.num_envs = 1
        cfg.seed = args.seed
        env = gym.make("LinkAgent-HarvestCycle-Acceptance-v1", cfg=cfg)
        backend = None
        perception = None
        bundle = None
        try:
            env.reset(seed=args.seed)
            base = env.unwrapped
            zeros = torch.zeros((1, 11), device=base.device)
            for _ in range(4):
                env.step(zeros)
            bundle = IsaacHarvestRoboticsBundle(base)
            latest_path = spool_root / "latest.json"
            if latest_path.is_file():
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                prior_ns = int(latest["sample_monotonic_ns"])
                now_ns = bundle.clock.monotonic_ns()
                if now_ns <= prior_ns:
                    bundle.advance((prior_ns - now_ns) / 1.0e9 + 0.05)
            publication = publish_isaac_rgbd(
                bundle,
                spool_root,
                capture_id=f"isaac-seed-{args.seed}-view-000",
            )
            backend = ArkVlmBoxDepthBackend(
                capture_root=spool_root,
                config_dir=repo_root / "config",
                ark_vlm={
                    "type": "ark_responses",
                    "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
                    "model_id": "doubao-seed-2.0-pro",
                    "api_key_env": "ARK_API_KEY",
                    "timeout_s": 60.0,
                    "max_response_bytes": 256_000,
                    "max_output_tokens": 4096,
                    "thinking_type": "disabled",
                },
                minimum_confidence=0.35,
                depth_band_m=0.08,
            )
            try:
                vision_payload = backend.process(
                    bundle_path=publication.bundle_path,
                    observation_sequence=1,
                    requested_track_id=None,
                )
            except Exception as exc:
                print(
                    "ISAAC_HARVEST_PERCEPTION_FAILURE="
                    + json.dumps(
                        {"type": type(exc).__name__, "message": str(exc)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise
            observation = Observation(
                sequence=1,
                timestamp_s=bundle.clock.monotonic_ns() / 1.0e9,
                metadata={"tomato_vision": vision_payload},
            )
            perception = ReplayTomatoVisionPerceptionPort(
                clock=bundle.clock,
                expected_calibration_id="isaac_top_rgbd_v1",
                expected_device_id="isaac-top-rgbd-01",
                minimum_detector_confidence=0.35,
                minimum_selection_confidence=0.35,
                max_fit_residual_m=0.020,
                silhouette_axis_floor_ratio=0.86,
                visible_surface_center_correction_ratio=0.82,
                pull_axis_prior_world=(0.0, 0.0, -1.0),
                max_pull_axis_deviation_deg=3.0,
            )
            detection = perception.detect(observation)
            target = detection.targets[0]
            fruit_truth = tuple(
                float(value) for value in base._tomato.data.root_pos_w.torch[0]
            )
            position_error_m = math.dist(target.center_pose.position_m, fruit_truth)
            instance = vision_payload["instances"][0]
            result = {
                "schema_version": 1,
                "passed": position_error_m <= 0.055,
                "backend": "isaac_lab_cuda_physx",
                "semantic_detector": backend.backend_id,
                "model_id": backend.health()["model_id"],
                "ark_response_id": backend._detector.last_response_id,
                "capture_id": publication.pointer.capture_id,
                "rgb_sha256": publication.rgb_sha256,
                "aligned_depth_sha256": publication.aligned_depth_sha256,
                "target": target.summary(),
                "stem_direction_xyz": list(target.stem_direction_xyz),
                "simulator_fruit_center_world_m": list(fruit_truth),
                "metric_center_error_m": position_error_m,
                "bbox_xyxy_px": instance["bbox_xyxy_px"],
                "depth_support_pixels": instance["valid_depth_pixel_count"],
                "evidence_sources": vision_payload["camera"]["evidence_sources"],
            }
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "acceptance.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print("ISAAC_HARVEST_PERCEPTION_ACCEPTANCE=" + json.dumps(result, sort_keys=True))
            return 0 if result["passed"] else 2
        finally:
            if perception is not None:
                perception.close()
            if backend is not None:
                backend.close()
            if bundle is not None:
                bundle.close()
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
