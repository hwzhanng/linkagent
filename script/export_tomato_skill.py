"""Freeze one Clean-v5 RSL-RL Actor as an Agent skill artifact.

This script only reads a local checkpoint and writes deterministic CPU
``torch.export`` and TorchScript artifacts.  It never imports the robot SDK
or opens a device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from module.skills.tomato_pick import (
    ACTION_LAYOUT_ID,
    HAND_LOWER_RAD,
    HAND_UPPER_RAD,
    INTERFACE_VERSION,
    OBSERVATION_LAYOUT_ID,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actor_from_checkpoint(checkpoint: Path) -> tuple[MLPModel, dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "actor_state_dict" not in payload:
        raise ValueError("checkpoint does not contain an RSL-RL actor_state_dict")
    observation = TensorDict(
        {
            "policy": torch.zeros((1, 66), dtype=torch.float32),
            "critic": torch.zeros((1, 83), dtype=torch.float32),
        },
        batch_size=[1],
    )
    actor = MLPModel(
        observation,
        {"actor": ["policy"], "critic": ["critic"]},
        "actor",
        11,
        [512, 256, 128],
        "elu",
        True,
        {
            "class_name": "GaussianDistribution",
            "init_std": 0.60,
            "std_type": "log",
        },
    )
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.eval()
    return actor, payload


def _canaries(
    actor: MLPModel,
    scripted: torch.jit.ScriptModule,
    exported: torch.nn.Module,
) -> list[dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    inputs = (
        torch.zeros((1, 66), dtype=torch.float32),
        torch.linspace(-1.0, 1.0, 66, dtype=torch.float32).reshape(1, 66),
        torch.tensor(([1.0, -1.0] * 33,), dtype=torch.float32),
        torch.linspace(1.0, -1.0, 66, dtype=torch.float32).reshape(1, 66),
        *tuple(
            torch.randn((1, 66), generator=generator, dtype=torch.float32) * scale
            for scale in (0.05, 0.25, 0.50, 1.0, 2.0)
            for _ in range(8)
        ),
    )
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for index, value in enumerate(inputs):
            reference = actor(
                TensorDict(
                    {"policy": value, "critic": torch.zeros((1, 83))},
                    batch_size=[1],
                )
            )
            scripted_action = scripted(value)
            exported_action = exported(value)
            script_error = float((reference - scripted_action).abs().max().item())
            export_error = float((reference - exported_action).abs().max().item())
            if max(script_error, export_error) > 1.0e-6:
                raise RuntimeError(
                    "export parity failed with max errors "
                    f"TorchScript={script_error}, torch.export={export_error}"
                )
            records.append(
                {
                    "id": f"canary_{index}",
                    "observation": [float(item) for item in value.reshape(-1).tolist()],
                    "raw_action": [float(item) for item in exported_action.reshape(-1).tolist()],
                    "clipped_action": [
                        float(item)
                        for item in exported_action.clamp(-1.0, 1.0).reshape(-1).tolist()
                    ],
                    "max_abs_torchscript_error": script_error,
                    "max_abs_torch_export_error": export_error,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skill-id")
    parser.add_argument(
        "--status",
        default="candidate_pending_fixed_multiseed_evaluation",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    actor, payload = _actor_from_checkpoint(checkpoint)
    exportable = actor.as_jit().cpu().eval()
    scripted = torch.jit.script(exportable)
    actor_path = output / "actor_deterministic_jit.pt"
    scripted.save(str(actor_path))
    exported_program = torch.export.export(
        exportable,
        (torch.zeros((1, 66), dtype=torch.float32),),
    )
    exported_path = output / "actor_deterministic_export.pt2"
    torch.export.save(exported_program, str(exported_path))
    exported_module = exported_program.module()
    source_path = output / "source_checkpoint.pt"
    shutil.copy2(checkpoint, source_path)
    params_dir = checkpoint.parent / "params"
    if params_dir.is_dir():
        shutil.copytree(params_dir, output / "training_params", dirs_exist_ok=True)

    canaries = _canaries(actor, scripted, exported_module)
    canary_path = output / "inference_canaries.json"
    canary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "equivalence_tolerance": 1.0e-6,
                "seed": 20260808,
                "cases": canaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_sha = _sha256(source_path)
    iteration = int(payload.get("iter", -1))
    skill_id = args.skill_id or f"tomato_pick_clean_v5_model{iteration}_candidate"
    manifest = {
        "schema_version": 1,
        "skill_id": skill_id,
        "status": args.status,
        "policy_manifest": {
            "policy_id": skill_id,
            "checkpoint_sha256": checkpoint_sha,
            "hand_joint_mapper_id": "g20_eigengrasp_5d_v3_thumb_roll_fixed",
            "tactile_feature_extractor_id": "g20_pressure_5d_v1",
            "training_asset_id": "linkbot_bimanual_sdk_training_mass1kg_tomato_clean_v5",
            "action_scales": [0.004, 0.004, 0.006, 0.04, 0.04, 0.06],
            "hand_position_lower_rad": list(HAND_LOWER_RAD),
            "hand_position_upper_rad": list(HAND_UPPER_RAD),
            "hand_velocity_scale_rad_s": [2.0] * 16,
            "fruit_axis_lower_m": [0.025, 0.025, 0.025],
            "fruit_axis_upper_m": [0.035, 0.035, 0.035],
            "max_state_age_s": [0.50, 0.10, 0.10],
            "max_force_age_s": 0.15,
            "normalization_stats": {
                "relative_position_scale_m": 0.15,
                "linear_velocity_scale_m_s": 0.5,
                "angular_velocity_scale_rad_s": 2.0,
                "pressure_full_scale_n": 20.0,
                "action_clip": 1.0,
            },
            "interface_version": INTERFACE_VERSION,
            "observation_layout_id": OBSERVATION_LAYOUT_ID,
            "action_layout_id": ACTION_LAYOUT_ID,
            "policy_rate_hz": 20,
            "grasp_frame_convention": "z_plus_away_from_plant",
        },
        "training": {
            "task": "LinkAgent-TomatoPick-Clean-v5",
            "iteration": iteration,
            "seed": 42,
            "actor_architecture": [66, 512, 256, 128, 11],
            "activation": "elu",
            "observation_normalization": True,
            "deterministic_inference": True,
            "source_run": str(checkpoint.parent),
        },
        "artifacts": {
            "deterministic_actor": {
                "path": actor_path.name,
                "sha256": _sha256(actor_path),
            },
            "deterministic_actor_export": {
                "path": exported_path.name,
                "sha256": _sha256(exported_path),
            },
            "source_checkpoint": {
                "path": source_path.name,
                "sha256": checkpoint_sha,
            },
            "inference_canaries": {
                "path": canary_path.name,
                "sha256": _sha256(canary_path),
            },
        },
        "deployment": {
            "profile": "simulation_evaluation_only",
            "hardware_approved": False,
            "requires_independent_pick_verifier": True,
            "requires_motion_safety": True,
            "requires_control_lease": True,
            "blocked_until": [
                "fixed deterministic multi-seed evaluation selects this checkpoint",
                "G20 fingertip force calibration is complete",
                "vendor gateways pass read-only shadow validation",
                "graded hardware release is explicitly approved",
            ],
        },
        "evaluation": {
            "selection_status": "pending",
            "training_window_is_not_release_evidence": True,
        },
    }
    manifest_path = output / "skill_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
