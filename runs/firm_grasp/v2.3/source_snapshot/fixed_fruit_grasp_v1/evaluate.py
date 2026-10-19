#!/usr/bin/env python3
"""独立评估策略检查点或环境 neutral 目标张开手基线。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from research.fixed_fruit_grasp_v1.run_utils import (
    CONTRACT_ID,
    MAX_EVALUATION_EPISODES,
    STAGES,
    append_jsonl,
    agent_config_sha256,
    atomic_write_json,
    checkpoint_metadata,
    configure_fixed_fruit_agent_cfg,
    completed_episode_delta,
    contract_sha256,
    fsync_directory,
    fsync_file,
    hash_run_inputs,
    fixed_fruit_agent_settings,
    json_safe,
    open_hand_action_provider,
    sha256_file,
    sha256_json,
    resolve_checkpoint_pose,
    summarize_episode_rows,
    score_evaluation_batch,
    sync_evidence,
    validate_checkpoint_compatibility,
    validate_open_action_shape,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("checkpoint", "open"), default="checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="汇总 JSON 路径；逐回合记录写入同名 JSONL 附属文件。")
    parser.add_argument("--stage", choices=STAGES, default=None)
    parser.add_argument(
        "--pregrasp-distance-mm",
        type=float,
        default=None,
        help="沿接近轴固定的预抓取距离，单位 mm；有 checkpoint 时省略则读取冻结身份。",
    )
    parser.add_argument(
        "--center-local-mm",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="固定果心相对抓取基坐标的位置，单位 mm；有 checkpoint 时省略则读取冻结身份。",
    )
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episodes > MAX_EVALUATION_EPISODES:
        parser.error(f"--episodes must be at most {MAX_EVALUATION_EPISODES}")
    if args.num_envs < 1 or args.num_envs > 4:
        parser.error("--num-envs must be between 1 and 4")
    if args.pregrasp_distance_mm is not None and not math.isfinite(args.pregrasp_distance_mm):
        parser.error("--pregrasp-distance-mm must be finite")
    if args.pregrasp_distance_mm is not None and not 0.0 < args.pregrasp_distance_mm <= 200.0:
        parser.error("--pregrasp-distance-mm must be in (0, 200]")
    if args.center_local_mm is not None and not all(
        math.isfinite(value) for value in args.center_local_mm
    ):
        parser.error("--center-local-mm values must be finite")
    if args.policy == "checkpoint":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --policy checkpoint")
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
    elif args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"reference checkpoint does not exist: {args.checkpoint}")


def _sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}.{suffix}")


def _prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    reserved = (
        path,
        _sidecar(path, "episodes.jsonl"),
        _sidecar(path, "unscored_episodes.jsonl"),
        _sidecar(path, "failure.json"),
        _sidecar(path, "receipt.json"),
    )
    collisions = [str(item) for item in reserved if item.exists()]
    if collisions:
        raise FileExistsError(f"evaluation outputs already exist: {', '.join(collisions)}")
    return path


def _atomic_torch_load(torch: Any, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("RSL-RL checkpoint must contain a mapping")
    return payload


def _architecture_config(agent_config: dict[str, Any], num_actions: int, policy: Any) -> dict[str, Any]:
    result = {
        key: agent_config.get(key)
        for key in ("actor", "critic", "obs_groups", "clip_actions", "check_for_nan")
    }
    result["num_actions"] = int(num_actions)
    result["actor_obs_dim"] = json_safe(getattr(policy, "obs_dim", None))
    return result


def _write_failure(output: Path, exc: BaseException, phase: str, details: dict[str, Any]) -> None:
    failure_path = _sidecar(output, "failure.json")
    failure = {
        "contract_id": CONTRACT_ID,
        "phase": phase,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "details": json_safe(details),
        "written_before_simulation_close": True,
    }
    atomic_write_json(failure_path, failure)
    prior_summary = {}
    if output.is_file():
        try:
            prior_summary = json.loads(output.read_text(encoding="utf-8"))
        except Exception:
            prior_summary = {}
    prior_summary.update(
        {
            "valid": False,
            "result_status": "FAILED",
            "contract_id": CONTRACT_ID,
            "failure_path": str(failure_path),
            "failure": failure,
            "written_before_simulation_close": True,
        }
    )
    atomic_write_json(output, prior_summary)
    atomic_write_json(
        _sidecar(output, "receipt.json"),
        {
            "valid": False,
            "status": "FAILED",
            "summary_path": str(output),
            "failure_path": str(failure_path),
        },
    )


def _run(args: argparse.Namespace, output: Path) -> tuple[int, list[Path]]:
    import torch
    from isaaclab.utils.seed import configure_seed
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv, make_config
    from research.small_fruit_pinch_v1.source.agents import SmallFruitPinchPPORunnerCfg

    episode_path = _sidecar(output, "episodes.jsonl")
    unscored_episode_path = _sidecar(output, "unscored_episodes.jsonl")
    receipt_path = _sidecar(output, "receipt.json")
    failure_path = _sidecar(output, "failure.json")
    files_to_sync = [output, episode_path, unscored_episode_path, receipt_path, failure_path]
    env = None
    wrapped = None
    runner = None
    checkpoint_meta = None
    checkpoint_hash = None
    episode_rows: list[dict[str, Any]] = []
    unscored_episode_rows: list[dict[str, Any]] = []
    episode_cursor = 0
    open_action_info = None
    exit_code = 1

    try:
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint.expanduser().resolve()
            checkpoint_hash = sha256_file(checkpoint_path)
            payload = _atomic_torch_load(torch, checkpoint_path)
            checkpoint_meta = checkpoint_metadata(payload)
            del payload

        stage = args.stage or (
            checkpoint_meta.get("stage") if checkpoint_meta is not None else "contact"
        )
        if stage not in STAGES:
            raise ValueError("evaluation stage must be contact, hold, or lift")

        requested_distance_m = (
            None
            if args.pregrasp_distance_mm is None
            else args.pregrasp_distance_mm / 1000.0
        )
        requested_center_m = (
            None
            if args.center_local_mm is None
            else tuple(value / 1000.0 for value in args.center_local_mm)
        )
        if checkpoint_meta is not None:
            frozen_identity = checkpoint_meta.get("task_identity")
            if not isinstance(frozen_identity, dict):
                raise ValueError("checkpoint task identity must be an object")
            pregrasp_distance_m, fruit_center_local_m = resolve_checkpoint_pose(
                frozen_identity,
                requested_distance_m,
                requested_center_m,
            )
            pose_resolution_source = "checkpoint_task_identity"
        else:
            pregrasp_distance_m = requested_distance_m
            fruit_center_local_m = requested_center_m
            pose_resolution_source = (
                "cli" if requested_distance_m is not None or requested_center_m is not None
                else "environment_defaults"
            )

        configure_seed(args.seed)
        cfg = make_config(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            stage=stage,
            pregrasp_distance_m=pregrasp_distance_m,
            fruit_center_local_m=fruit_center_local_m,
        )
        env = FixedFruitGraspEnv(cfg=cfg)
        identity = json_safe(env.task_identity())
        if not isinstance(identity, dict):
            raise TypeError("env.task_identity() must return a JSON-safe object")
        hashes = hash_run_inputs(Path(__file__).parent, identity)
        initial_pose = identity.get("initial_pose", {})
        pose_resolution = {
            "source": pose_resolution_source,
            "requested_pregrasp_distance_mm": args.pregrasp_distance_mm,
            "requested_center_local_mm": args.center_local_mm,
            "resolved_pregrasp_distance_m": initial_pose.get("pregrasp_distance_m"),
            "resolved_fruit_center_local_m": initial_pose.get("closed_grasp_fruit_center_m"),
        }

        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        if int(wrapped.num_actions) != 14:
            raise ValueError(f"fixed-fruit policy contract requires 14 actions; got {wrapped.num_actions}")
        if not isinstance(getattr(env, "completed_episodes", None), list):
            raise TypeError("env.completed_episodes must be an append-only list")

        agent_cfg = SmallFruitPinchPPORunnerCfg()
        agent_cfg.logger = "tensorboard"
        agent_cfg.run_name = "fixed_fruit_grasp_v1"
        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, importlib.metadata.version("rsl-rl-lib")
        )
        agent_cfg = configure_fixed_fruit_agent_cfg(agent_cfg)
        agent_settings = fixed_fruit_agent_settings(agent_cfg, wrapped.num_actions)
        agent_config = json_safe(agent_cfg.to_dict())
        runner = OnPolicyRunner(
            wrapped,
            agent_config,
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.logger.writer = None
        policy_model = runner.alg.get_policy()
        architecture = _architecture_config(agent_config, wrapped.num_actions, policy_model)

        inference_policy = None
        if checkpoint_meta is not None:
            mode = validate_checkpoint_compatibility(
                checkpoint_meta,
                identity,
                stage,
                source_and_asset_hashes=hashes,
            )
            if mode != "full_resume":
                raise ValueError("evaluation stage must match checkpoint stage")
            if checkpoint_meta.get("agent_architecture_sha256") != sha256_json(architecture):
                raise ValueError("checkpoint agent architecture does not match this evaluator")
            if checkpoint_meta.get("agent_config_comparison_sha256") != agent_config_sha256(agent_config):
                raise ValueError("checkpoint agent configuration does not match this evaluator")
            if args.policy == "checkpoint":
                runner.load(str(args.checkpoint), map_location=args.device)
                inference_policy = runner.get_inference_policy(device=args.device)

        fixed_open_action = None
        if args.policy == "open":
            provider_name, provider_value = open_hand_action_provider(env)
            if provider_name == "set_open_hand_baseline" and provider_value is None:
                # 张量仅用于满足 runner.step 接口；环境旁路会忽略它并保持 reset 的 neutral 目标。
                baseline_enabled = getattr(env, "open_hand_baseline_enabled", None)
                fixed_open_action = torch.zeros(
                    (args.num_envs, wrapped.num_actions),
                    dtype=torch.float32,
                    device=args.device,
                )
                open_action_info = {
                    "provider": provider_name,
                    "mechanism": "environment_neutral_target_override",
                    "shape": list(fixed_open_action.shape),
                    "zero_payload_is_open_action": False,
                    "payload_ignored_by_environment": True,
                    "zero_fallback_used": False,
                    "environment_override_enabled": (
                        bool(baseline_enabled) if baseline_enabled is not None else None
                    ),
                    "environment_override_enabled_missing_reason": (
                        None
                        if baseline_enabled is not None
                        else "environment exposes no open_hand_baseline_enabled status"
                    ),
                    "neutral_target_source": "environment reset default_joint_pos",
                }
            else:
                fixed_open_action = validate_open_action_shape(
                    provider_value,
                    args.num_envs,
                    wrapped.num_actions,
                    torch,
                    args.device,
                )
                open_action_info = {
                    "provider": provider_name,
                    "mechanism": "explicit_inverse_mapped_action",
                    "shape": list(fixed_open_action.shape),
                    "meaning": "environment-provided inverse-mapped open/default-joint-position target",
                    "zero_fallback_used": False,
                    "nonzero_element_count": int(torch.count_nonzero(fixed_open_action).item()),
                }

        run_metadata = {
            "contract_id": CONTRACT_ID,
            "policy": args.policy,
            "checkpoint_path": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_task_identity_sha256": (
                checkpoint_meta.get("task_identity_sha256") if checkpoint_meta else None
            ),
            "task_identity": identity,
            "task_identity_sha256": sha256_json(identity),
            "contract_sha256": contract_sha256(identity),
            "source_and_asset_hashes": hashes,
            "agent_config": agent_config,
            "agent_settings": agent_settings,
            "agent_config_sha256": sha256_json(agent_config),
            "agent_config_comparison_sha256": agent_config_sha256(agent_config),
            "agent_architecture": architecture,
            "agent_architecture_sha256": sha256_json(architecture),
            "stage": stage,
            "pose_resolution": pose_resolution,
            "num_envs": args.num_envs,
            "episodes_target": args.episodes,
            "seed": args.seed,
            "device": args.device,
            "open_action": open_action_info,
            "reset_source": "RslRlVecEnvWrapper constructor performs a complete env.reset()",
            "episode_accounting": (
                "first requested completed rows are scored in batch arrival order, with env_id ascending "
                "within each vectorized step; excess completed rows are retained as unscored"
            ),
            "scoring_target": args.episodes,
            "scoring_target_maximum": MAX_EVALUATION_EPISODES,
            "unscored_episodes_jsonl_path": str(unscored_episode_path),
            "video_recording": {
                "recorded": False,
                "missing_reason": "no verified camera capture adapter is configured for this task",
            },
            "command": list(sys.argv),
        }

        obs = wrapped.get_observations()
        start_time = time.monotonic()
        while len(episode_rows) < args.episodes:
            with torch.inference_mode():
                if args.policy == "checkpoint":
                    action = inference_policy(obs)
                else:
                    action = fixed_open_action
            obs, _, dones, _ = wrapped.step(action)
            rows, episode_cursor = completed_episode_delta(env, episode_cursor)
            done_count = int(dones.sum().item())
            if done_count != len(rows):
                raise RuntimeError(
                    "environment done count and completed_episodes append count differ: "
                    f"{done_count} != {len(rows)}"
                )
            scored_batch, unscored_batch = score_evaluation_batch(
                rows, len(episode_rows), args.episodes
            )
            for row in scored_batch:
                episode_rows.append(row)
                append_jsonl(episode_path, row)
            for row in unscored_batch:
                unscored_episode_rows.append(row)
                append_jsonl(unscored_episode_path, row)

        episode_summary = summarize_episode_rows(episode_rows)
        success_count = episode_summary.get("success_count")
        denominator = int(episode_summary["episodes_completed"])
        success_rate = episode_summary.get("success_rate")
        complete_success_denominator = episode_summary["success_observed"] == denominator
        goal_observed = sum(isinstance(row.get("goal_success"), bool) for row in episode_rows)
        goal_success_count = sum(row.get("goal_success") is True for row in episode_rows)
        target_met = (
            denominator == args.episodes
            and complete_success_denominator
            and success_rate is not None
            and success_rate >= 0.90
        )
        summary = {
            "valid": denominator == args.episodes and complete_success_denominator,
            "result_status": (
                "EVALUATED"
                if denominator == args.episodes and complete_success_denominator
                else "INCOMPLETE"
            ),
            "contract_id": CONTRACT_ID,
            "policy": args.policy,
            "checkpoint_path": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_sha256": checkpoint_hash,
            "task_identity": identity,
            "task_identity_sha256": sha256_json(identity),
            "contract_sha256": contract_sha256(identity),
            "source_and_asset_hashes": hashes,
            "agent_config": agent_config,
            "agent_settings": agent_settings,
            "agent_config_sha256": sha256_json(agent_config),
            "agent_config_comparison_sha256": agent_config_sha256(agent_config),
            "agent_architecture_sha256": sha256_json(architecture),
            "stage": stage,
            "pose_resolution": pose_resolution,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "requested_episode_target": args.episodes,
            "scoring_target_maximum": MAX_EVALUATION_EPISODES,
            "scoring_order": "batch arrival order; env_id ascending within each vectorized step",
            "actual_denominator": denominator,
            "scored_episode_count": denominator,
            "completed_episode_rows": denominator + len(unscored_episode_rows),
            "unscored_episode_count": len(unscored_episode_rows),
            "unscored_episodes_jsonl_path": str(unscored_episode_path),
            "success_count": success_count,
            "success_rate": success_rate,
            "goal_success_count": goal_success_count if goal_observed == denominator else None,
            "goal_success_observed": goal_observed,
            "meets_stage_90_percent_target": target_met,
            "meets_90_percent_target": (
                denominator == args.episodes
                and goal_observed == denominator
                and goal_success_count / denominator >= 0.90
            ),
            "episode_summary": episode_summary,
            "episodes": episode_rows,
            "episodes_jsonl_path": str(episode_path),
            "unscored_episodes_jsonl_path": str(unscored_episode_path),
            "elapsed_wall_s": time.monotonic() - start_time,
            "open_action": open_action_info,
            "reset_source": run_metadata["reset_source"],
            "episode_accounting": run_metadata["episode_accounting"],
            "video_recording": run_metadata["video_recording"],
            "written_before_simulation_close": True,
        }
        atomic_write_json(output, summary)
        atomic_write_json(
            receipt_path,
            {
                "valid": summary["valid"],
                "status": summary["result_status"],
                "summary_path": str(output),
                "episodes_jsonl_path": str(episode_path),
                "unscored_episodes_jsonl_path": str(unscored_episode_path),
                "policy": args.policy,
                "checkpoint_sha256": checkpoint_hash,
                "actual_denominator": denominator,
                "unscored_episode_count": len(unscored_episode_rows),
                "contract_id": CONTRACT_ID,
            },
        )
        exit_code = 0 if summary["valid"] else 1

    except BaseException as exc:
        _write_failure(
            output,
            exc,
            "evaluation",
            {
                "policy": args.policy,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "episodes_completed": len(episode_rows),
                "stage": args.stage,
            },
        )
        exit_code = 1
    finally:
        sync_evidence(files_to_sync)
        if wrapped is not None:
            try:
                wrapped.close()
            except BaseException as close_exc:
                _write_failure(
                    output,
                    close_exc,
                    "environment_close",
                    {"episodes_completed": len(episode_rows)},
                )
                exit_code = 1
        elif env is not None:
            try:
                env.close()
            except BaseException as close_exc:
                _write_failure(output, close_exc, "environment_close", {})
                exit_code = 1

    return exit_code, files_to_sync


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_cli(args, parser)
    try:
        output = _prepare_output(args.output)
    except Exception as exc:
        parser.error(str(exc))

    app = None
    exit_code = 1
    try:
        app = AppLauncher(args).app
        exit_code, _ = _run(args, output)
    except BaseException as exc:
        _write_failure(
            output,
            exc,
            "app_or_entrypoint",
            {"policy": args.policy, "checkpoint": str(args.checkpoint) if args.checkpoint else None},
        )
        exit_code = 1

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if app is not None:
        app.close(exit_code=exit_code)
    os._exit(exit_code)


if __name__ == "__main__":
    main()
