from __future__ import annotations

import argparse
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from pxr import UsdUtils

from .contract import ACTION_DIM, ACTOR_OBSERVATION_DIM, CRITIC_STATE_DIM, TASK_VERSION, VERIFICATION_CONTRACT_VERSION
from .legacy_bridge import load_legacy_hand_manifold
from .run_manifest import build_run_manifest, write_run_manifest


def freeze_native_run(env, agent_cfg, args: argparse.Namespace) -> tuple[Path, str]:
    root = Path(__file__).resolve().parents[3]
    if args.run_manifest is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.run_manifest = root / "logs" / "verification" / stamp / "run_manifest.json"
    scene_path = args.run_manifest.parent / "scene.usda"
    if args.run_manifest.exists() or scene_path.exists():
        raise FileExistsError(f"运行证据目录不可覆盖: {args.run_manifest.parent}")
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    if not env.sim.stage.Flatten().Export(str(scene_path)):
        raise RuntimeError(f"无法冻结实际场景: {scene_path}")
    controls = {}
    for cls in reversed(type(env).__mro__):
        if cls.__module__.startswith("research.small_fruit_pinch_v1"):
            for name, value in vars(cls).items():
                if not name.startswith("_") and not inspect.isroutine(value) and not isinstance(value, property) and name != "cfg":
                    controls[name] = value
    controls["canonical_joint_names"] = [env._robot.joint_names[index] for index in env._hand_joint_ids]
    controls["fingertip_contact_body_names"] = [env._robot.body_names[index] for index in env._fingertip_contact_body_ids]
    controls["fingertip_frame_body_names"] = [env._robot.body_names[index] for index in env._fingertip_frame_body_ids]
    source_paths = list((root / "research/small_fruit_pinch_v1/source").glob("*.py"))
    source_paths.extend((root / "simulation/tomato_pick").rglob("*.py"))
    legacy = load_legacy_hand_manifold()
    source_paths.extend(Path(legacy.__file__).parent.glob("*.py"))
    local_layers = [Path(layer.realPath) for layer in env.sim.stage.GetUsedLayers() if layer.realPath]
    asset_paths = {*local_layers, scene_path.resolve()}
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(env.cfg.robot.spawn.usd_path))
    if unresolved:
        raise RuntimeError(f"USD 依赖无法冻结: {unresolved}")
    asset_paths.update(Path(layer.realPath) for layer in layers if layer.realPath)
    asset_paths.update(Path(path) for path in assets)
    checkpoint = getattr(args, "checkpoint", None) or getattr(args, "resume", None) or getattr(args, "actor_init", None)
    manifest = build_run_manifest(
        project_root=root, cli_argv=sys.argv, cli_args=vars(args),
        env_config=env.cfg.to_dict(), agent_config=agent_cfg.to_dict(), env_controls=controls,
        source_paths=source_paths, asset_paths=tuple(asset_paths), checkpoint_path=checkpoint,
        task_version=TASK_VERSION, validation_contract_version=VERIFICATION_CONTRACT_VERSION,
        action_dimension=ACTION_DIM, observation_dimension=ACTOR_OBSERVATION_DIM,
        critic_dimension=CRITIC_STATE_DIM,
        runtime_versions={"torch": torch.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(env.device) if str(env.device).startswith("cuda") else "cpu"},
    )
    write_run_manifest(args.run_manifest, manifest)
    print(f"run_manifest path={args.run_manifest} sha256={manifest['run_sha256']}", flush=True)
    return args.run_manifest, str(manifest["run_sha256"])
