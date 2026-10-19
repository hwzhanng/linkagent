#!/usr/bin/env python3
"""使用 RSL-RL 训练固定果实 14 维抓握策略。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from research.fixed_fruit_grasp_v1.run_utils import (
    CONTRACT_ID,
    STAGES,
    EpisodeAccumulator,
    append_jsonl,
    agent_config_sha256,
    atomic_write_json,
    checkpoint_metadata,
    configure_fixed_fruit_agent_cfg,
    completed_episode_delta,
    contract_sha256,
    fixed_fruit_agent_settings,
    fsync_directory,
    fsync_file,
    hash_run_inputs,
    InspectionWindowStopState,
    json_safe,
    initialize_actor_output_bias,
    parse_initial_action_bias_json,
    reset_actor_exploration_std,
    sha256_file,
    sha256_json,
    summarize_inspection_window,
    sync_evidence,
    validate_checkpoint_compatibility,
    validate_explicit_actor_transfer,
)


class _StopTraining(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="contact")
    parser.add_argument("--actor-transfer", type=Path, default=None, help="显式actor迁移（v1.3→v1.4、奖励变更v1.4→v1.5或保守PPO v1.5→v1.6）；critic/optimizer重启。")
    parser.add_argument("--actor-transfer-source-sha256", default=None, help="迁移源checkpoint的已核验64位SHA256，必须显式提供。")
    parser.add_argument("--resume", type=Path, default=None, help="仅接受本任务新合同的 RSL-RL 检查点。")
    parser.add_argument(
        "--initial-action-bias-json",
        default=None,
        help="fresh run 的 14D JSON 动作均值先验；每个值必须在 [-1, 1]。",
    )
    parser.add_argument(
        "--pregrasp-distance-mm",
        type=float,
        default=None,
        help="沿接近轴固定的预抓取距离，单位 mm；省略时使用环境默认值。",
    )
    parser.add_argument(
        "--center-local-mm",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="固定果心相对抓取基坐标的位置，单位 mm；省略时使用环境默认值。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=None, help="本进程墙钟上限；supervisor 仍应设置外层 timeout。")
    parser.add_argument("--initialize-only", action="store_true", help="只构建新策略并写出 initial.pt，不执行 PPO 更新。")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> bool:
    if args.iterations < 0:
        parser.error("--iterations must be non-negative")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0.0:
        parser.error("--max-seconds must be positive")
    if args.pregrasp_distance_mm is not None and not math.isfinite(args.pregrasp_distance_mm):
        parser.error("--pregrasp-distance-mm must be finite")
    if args.pregrasp_distance_mm is not None and not 0.0 < args.pregrasp_distance_mm <= 200.0:
        parser.error("--pregrasp-distance-mm must be in (0, 200]")
    if args.center_local_mm is not None and not all(
        math.isfinite(value) for value in args.center_local_mm
    ):
        parser.error("--center-local-mm values must be finite")
    if args.actor_transfer is not None:
        if args.resume is not None:
            parser.error("--actor-transfer and --resume are mutually exclusive")
        if not args.actor_transfer.is_file():
            parser.error("actor-transfer checkpoint does not exist")
        pin = args.actor_transfer_source_sha256
        if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            parser.error("--actor-transfer-source-sha256 requires a lowercase 64-character SHA256")
    elif args.actor_transfer_source_sha256 is not None:
        parser.error("--actor-transfer-source-sha256 requires --actor-transfer")
    try:
        args.initial_action_bias = parse_initial_action_bias_json(
            args.initial_action_bias_json,
            resume=args.resume is not None or args.actor_transfer is not None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"resume checkpoint does not exist: {args.resume}")
    if args.initialize_only and args.resume is not None:
        parser.error("--initialize-only cannot be combined with --resume")
    initialize_only = bool(args.initialize_only or args.iterations == 0)
    if args.iterations == 0 and args.resume is not None:
        parser.error("--iterations 0 cannot be combined with --resume")
    return initialize_only


def _prepare_log_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    reserved = {
        "initial.pt",
        "actor_initial.pt",
        "final.pt",
        "actor_final.pt",
        "metrics.jsonl",
        "episodes.jsonl",
        "inspections.jsonl",
        "checkpoints.jsonl",
        "summary.json",
        "failure.json",
        "receipt.json",
        "run_metadata.json",
    }
    collisions = sorted(item.name for item in path.iterdir() if item.name in reserved)
    if collisions:
        raise FileExistsError(
            f"log directory already contains this run's reserved outputs: {', '.join(collisions)}"
        )
    return path


def _atomic_torch_save(torch: Any, value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        fsync_file(temporary)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _clone_parameters(policy: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
    }


def _parameter_delta(torch: Any, before: dict[str, Any], policy: Any) -> dict[str, Any]:
    after = dict(policy.named_parameters())
    if set(before) != set(after):
        return {
            "changed": None,
            "changed_parameter_tensors": None,
            "changed_parameter_elements": None,
            "max_abs_delta": None,
            "l2_delta": None,
            "missing_reason": "actor parameter names changed during training",
        }
    changed_tensors = 0
    changed_elements = 0
    max_abs_delta = 0.0
    squared_delta = 0.0
    finite = True
    for name, old in before.items():
        current = after[name].detach().cpu()
        difference = current - old
        finite = finite and bool(torch.isfinite(current).all().item())
        nonzero = difference != 0
        if bool(nonzero.any().item()):
            changed_tensors += 1
            changed_elements += int(nonzero.sum().item())
            max_abs_delta = max(max_abs_delta, float(difference.abs().max().item()))
            squared_delta += float(difference.double().square().sum().item())
    return {
        "changed": bool(changed_elements > 0),
        "parameters_finite": finite,
        "changed_parameter_tensors": changed_tensors,
        "changed_parameter_elements": changed_elements,
        "max_abs_delta": max_abs_delta,
        "l2_delta": math.sqrt(squared_delta),
    }


def _actor_state(policy: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for name, value in policy.state_dict().items()
    }


def _architecture_config(agent_config: dict[str, Any], num_actions: int, policy: Any) -> dict[str, Any]:
    policy_config = {
        key: agent_config.get(key)
        for key in ("actor", "critic", "obs_groups", "clip_actions", "check_for_nan")
    }
    policy_config["num_actions"] = int(num_actions)
    policy_config["actor_obs_dim"] = json_safe(getattr(policy, "obs_dim", None))
    return policy_config


def _checkpoint_info(
    *,
    identity: dict[str, Any],
    hashes: dict[str, Any],
    stage: str,
    agent_config: dict[str, Any],
    architecture: dict[str, Any],
    progress: dict[str, Any],
    resume_info: dict[str, Any],
    initialization_kind: str,
    policy_initialization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256(identity),
        "task_identity": identity,
        "task_identity_sha256": sha256_json(identity),
        "source_and_asset_hashes": hashes,
        "stage": stage,
        "agent_config": agent_config,
        "agent_config_sha256": sha256_json(agent_config),
        "agent_config_comparison_sha256": agent_config_sha256(agent_config),
        "agent_architecture": architecture,
        "agent_architecture_sha256": sha256_json(architecture),
        "progress": progress,
        "resume": resume_info,
        "initialization_kind": initialization_kind,
        "policy_initialization": policy_initialization,
    }


def _save_checkpoint(
    runner: Any,
    torch: Any,
    path: Path,
    info: dict[str, Any],
) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        runner.save(str(temporary), infos={CONTRACT_ID: json_safe(info)})
        fsync_file(temporary)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _check_loss_finite(torch: Any, loss_dict: Any) -> bool:
    if not isinstance(loss_dict, dict) or not loss_dict:
        return False
    for value in loss_dict.values():
        if hasattr(value, "detach"):
            value = value.detach()
            if not bool(torch.isfinite(value).all().item()):
                return False
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def _write_failure(log_dir: Path, exc: BaseException, phase: str, details: dict[str, Any]) -> None:
    failure = {
        "contract_id": CONTRACT_ID,
        "phase": phase,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "details": json_safe(details),
        "written_before_simulation_close": True,
    }
    atomic_write_json(log_dir / "failure.json", failure)
    prior_summary = {}
    summary_path = log_dir / "summary.json"
    if summary_path.is_file():
        try:
            prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            prior_summary = {}
    prior_summary.update(
        {
            "valid": False,
            "result_status": "FAILED",
            "contract_id": CONTRACT_ID,
            "failure_path": str(log_dir / "failure.json"),
            "failure": failure,
            "written_before_simulation_close": True,
        }
    )
    atomic_write_json(summary_path, prior_summary)
    atomic_write_json(
        log_dir / "receipt.json",
        {
            "valid": False,
            "status": "FAILED",
            "summary_path": str(log_dir / "summary.json"),
            "failure_path": str(log_dir / "failure.json"),
        },
    )


def _run(
    args: argparse.Namespace,
    log_dir: Path,
    initialize_only: bool,
    stop_request: dict[str, int | None],
) -> tuple[int, list[Path]]:
    import torch
    from isaaclab.utils.seed import configure_seed
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv, make_config
    from research.small_fruit_pinch_v1.source.agents import SmallFruitPinchPPORunnerCfg

    files_to_sync = [
        log_dir / "metrics.jsonl",
        log_dir / "episodes.jsonl",
        log_dir / "inspections.jsonl",
        log_dir / "checkpoints.jsonl",
        log_dir / "summary.json",
        log_dir / "failure.json",
        log_dir / "receipt.json",
        log_dir / "run_metadata.json",
        log_dir / "initial.pt",
        log_dir / "actor_initial.pt",
        log_dir / "final.pt",
        log_dir / "actor_final.pt",
    ]
    env = None
    wrapped = None
    runner = None
    episode_accumulator = EpisodeAccumulator()
    episode_cursor = 0
    inspection_window_pending: list[dict[str, Any]] = []
    inspection_window_rows_consumed = 0
    inspection_stop_state = InspectionWindowStopState()
    update_state = {
        "optimizer_update_calls": 0,
        "valid_update_count_this_process": 0,
        "stage_updates_before": 0,
        "total_updates_before": 0,
        "stage_updates": 0,
        "total_updates": 0,
        "start_iteration": 0,
        "last_iteration": None,
        "last_update_losses": None,
        "losses_finite": True,
        "stop_reason": "iterations_completed",
        "started_monotonic": time.monotonic(),
        "deadline": None,
    }
    checkpoints: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    resume_info: dict[str, Any] = {"mode": "fresh_actor", "source_checkpoint": None}
    initialized = False
    exit_code = 1

    def _signal_stop_reason() -> str | None:
        signum = stop_request.get("signum")
        return None if signum is None else f"termination_signal_{int(signum)}"

    def _raise_for_signal() -> None:
        reason = _signal_stop_reason()
        if reason is not None:
            update_state["stop_reason"] = reason
            raise _StopTraining(reason)

    try:
        if args.max_seconds is not None:
            update_state["deadline"] = update_state["started_monotonic"] + args.max_seconds

        configure_seed(args.seed)
        cfg = make_config(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            stage=args.stage,
            pregrasp_distance_m=(
                None
                if args.pregrasp_distance_mm is None
                else args.pregrasp_distance_mm / 1000.0
            ),
            fruit_center_local_m=(
                None
                if args.center_local_mm is None
                else tuple(value / 1000.0 for value in args.center_local_mm)
            ),
        )
        env = FixedFruitGraspEnv(cfg=cfg)
        identity = json_safe(env.task_identity())
        if not isinstance(identity, dict):
            raise TypeError("env.task_identity() must return a JSON-safe object")
        hashes = hash_run_inputs(Path(__file__).parent, identity)

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
        # RSL writer 保持关闭，训练指标由本入口写入本地 JSONL；此时也可安全保存初始检查点。
        runner.logger.writer = None
        policy = runner.alg.get_policy()
        if args.resume is None:
            if args.initial_action_bias is None:
                initialization_kind = "fresh_default_policy_initialization"
                policy_initialization = {
                    "kind": initialization_kind,
                    "classification": "untrained_initialization_not_learned_success",
                    "bias": None,
                    "changed": False,
                    "actor_output_layer": None,
                    "weight_sha256_before": None,
                    "weight_sha256_after": None,
                }
            else:
                policy_initialization = initialize_actor_output_bias(
                    torch, policy, args.initial_action_bias
                )
                initialization_kind = str(policy_initialization["kind"])
        else:
            initialization_kind = "checkpoint_resume"
            policy_initialization = {
                "kind": "checkpoint_initialization_not_loaded_yet",
                "classification": "initialization_provenance_pending_checkpoint_metadata",
                "bias": None,
                "changed": False,
                "actor_output_layer": None,
                "weight_sha256_before": None,
                "weight_sha256_after": None,
            }
        architecture = _architecture_config(agent_config, wrapped.num_actions, policy)

        resume_payload = None
        if args.resume is not None:
            resume_path = args.resume.expanduser().resolve()
            resume_payload = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            metadata = checkpoint_metadata(resume_payload)
            stored_initialization = metadata.get("policy_initialization")
            if isinstance(stored_initialization, dict):
                policy_initialization = json_safe(stored_initialization)
            else:
                policy_initialization = {
                    "kind": "legacy_checkpoint_initialization",
                    "classification": "initialization_provenance_not_recorded_in_source_checkpoint",
                    "bias": None,
                    "changed": None,
                    "actor_output_layer": None,
                    "weight_sha256_before": None,
                    "weight_sha256_after": None,
                }
            resume_mode = validate_checkpoint_compatibility(
                metadata, identity, args.stage, source_and_asset_hashes=hashes
            )
            if metadata.get("agent_architecture_sha256") != sha256_json(architecture):
                raise ValueError("checkpoint actor/critic architecture does not match the current agent config")
            if metadata.get("agent_config_comparison_sha256") != agent_config_sha256(agent_config):
                raise ValueError("checkpoint agent configuration does not match the current agent config")
            if resume_mode == "full_resume":
                runner.load(str(resume_path), map_location=args.device)
                progress = metadata.get("progress") or {}
                update_state["stage_updates_before"] = int(
                    progress.get("stage_updates_completed", 0)
                )
                update_state["total_updates_before"] = int(
                    progress.get("total_updates_completed", 0)
                )
                next_iteration = int(progress.get("next_iteration", 0))
                if next_iteration < 0:
                    raise ValueError("checkpoint next_iteration must be non-negative")
                runner.current_learning_iteration = next_iteration
                update_state["start_iteration"] = next_iteration
                resume_info = {
                    "mode": resume_mode,
                    "source_checkpoint": str(resume_path),
                    "source_checkpoint_sha256": sha256_file(resume_path),
                    "source_stage": metadata["stage"],
                }
            else:
                actor_state = resume_payload.get("actor_state_dict")
                if not isinstance(actor_state, dict):
                    raise ValueError("stage transition checkpoint has no actor_state_dict")
                missing, unexpected = policy.load_state_dict(actor_state, strict=True)
                if missing or unexpected:
                    raise ValueError(
                        f"stage transition actor mismatch: missing={missing}, unexpected={unexpected}"
                    )
                progress = metadata.get("progress") or {}
                update_state["total_updates_before"] = int(
                    progress.get("total_updates_completed", 0)
                )
                update_state["start_iteration"] = 0
                runner.current_learning_iteration = 0
                resume_info = {
                    "mode": resume_mode,
                    "source_checkpoint": str(resume_path),
                    "source_checkpoint_sha256": sha256_file(resume_path),
                    "source_stage": metadata["stage"],
                    "critic_optimizer_and_iteration_reset": True,
                }
            update_state["stage_updates"] = update_state["stage_updates_before"]
            update_state["total_updates"] = update_state["total_updates_before"]
            del resume_payload

        if args.actor_transfer is not None:
            transfer_path = args.actor_transfer.expanduser().resolve()
            transfer_hash = sha256_file(transfer_path)
            if transfer_hash != args.actor_transfer_source_sha256:
                raise ValueError("actor-transfer checkpoint does not match the explicit SHA256 pin")
            transfer_payload = torch.load(transfer_path, map_location="cpu", weights_only=False)
            transfer_metadata = checkpoint_metadata(transfer_payload)
            resume_info = validate_explicit_actor_transfer(
                transfer_metadata, identity, args.stage, hashes,
                sha256_json(architecture), agent_config_sha256(agent_config),
                transfer_hash, args.actor_transfer_source_sha256,
                actual_agent_config=agent_config,
                actual_architecture=architecture,
            )
            actor_state = transfer_payload.get("actor_state_dict")
            if not isinstance(actor_state, dict) or not actor_state:
                raise ValueError("actor transfer has no actor_state_dict")
            for value in actor_state.values():
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise ValueError("actor transfer contains nonfinite parameters or normalization buffers")
            missing, unexpected = policy.load_state_dict(actor_state, strict=True)
            if missing or unexpected:
                raise ValueError("actor transfer state dictionary does not exactly match the actor")
            if bool(resume_info.get("exploration_std_reset_required")):
                # 仅已审查的 v1.5.0 -> v1.6.0 迁移：actor 均值权重与 obs 归一化
                # 缓冲保持逐字节不变，只把唯一 distribution.log_std_param 显式
                # 重置为 log(0.08)。这是显式探索重初始化，不是学习成功证据，
                # 且每策略只允许一次（helper 内部 fail-closed）。
                resume_info["exploration_std_reset"] = reset_actor_exploration_std(
                    torch, policy, transferred_state=actor_state
                )
            resume_info["source_checkpoint"] = str(transfer_path)
            update_state["total_updates_before"] = resume_info["source_total_updates_completed"]
            update_state["total_updates"] = update_state["total_updates_before"]
            update_state["stage_updates_before"] = 0
            update_state["stage_updates"] = 0
            update_state["start_iteration"] = 0
            runner.current_learning_iteration = 0
            initialization_kind = "actor_only_task_semantics_transfer"
            policy_initialization = {
                **resume_info, "kind": initialization_kind,
                "classification": "pretrained_actor_prior_not_new_semantics_success",
                "source_initialization": transfer_metadata.get("policy_initialization"),
            }
            del transfer_payload

        run_metadata = {
            "contract_id": CONTRACT_ID,
            "task_identity": identity,
            "task_identity_sha256": sha256_json(identity),
            "contract_sha256": contract_sha256(identity),
            "source_and_asset_hashes": hashes,
            "stage": args.stage,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": args.device,
            "requested_pose_mm": {
                "pregrasp_distance_mm": args.pregrasp_distance_mm,
                "center_local_mm": args.center_local_mm,
            },
            "agent_config": agent_config,
            "agent_settings": agent_settings,
            "agent_config_sha256": sha256_json(agent_config),
            "agent_config_comparison_sha256": agent_config_sha256(agent_config),
            "agent_architecture": architecture,
            "agent_architecture_sha256": sha256_json(architecture),
            "requested_iterations": args.iterations,
            "initialize_only": initialize_only,
            "initialization_kind": initialization_kind,
            "policy_initialization": policy_initialization,
            "command": list(sys.argv),
            "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
            "reset_source": "RslRlVecEnvWrapper constructor performs a complete env.reset()",
            "video_recording": {
                "recorded": False,
                "missing_reason": "training entrypoint has no camera capture path",
            },
        }
        atomic_write_json(log_dir / "run_metadata.json", run_metadata)

        actor_before = _clone_parameters(policy)
        actor_state_initial = _actor_state(policy)
        _atomic_torch_save(
            torch,
            {
                "state_dict": actor_state_initial,
                "initialization": initialization_kind,
                "policy_initialization": policy_initialization,
            },
            log_dir / "actor_initial.pt",
        )
        info = _checkpoint_info(
            identity=identity,
            hashes=hashes,
            stage=args.stage,
            agent_config=agent_config,
            architecture=architecture,
            progress={
                "stage_updates_completed": update_state["stage_updates"],
                "total_updates_completed": update_state["total_updates"],
                "next_iteration": update_state["start_iteration"],
                "current_learning_iteration_semantics": "zero-based label; not an update count",
            },
            resume_info=resume_info,
            initialization_kind=initialization_kind,
            policy_initialization=policy_initialization,
        )
        initial_checkpoint = _save_checkpoint(
            runner, torch, log_dir / "initial.pt", info
        )
        checkpoints.append({"kind": "initial", **initial_checkpoint})
        append_jsonl(
            log_dir / "checkpoints.jsonl",
            {"kind": "initial", **initial_checkpoint, "updates_completed": update_state["stage_updates"]},
        )
        initialized = True

        atomic_write_json(
            log_dir / "receipt.json",
            {
                "valid": True,
                "status": "INITIALIZED",
                "summary_path": str(log_dir / "summary.json"),
                "checkpoint_path": initial_checkpoint["path"],
                "checkpoint_sha256": initial_checkpoint["sha256"],
                "contract_id": CONTRACT_ID,
                "stage": args.stage,
            },
        )

        if initialize_only:
            parameter_delta = _parameter_delta(torch, actor_before, policy)
            summary = {
                "valid": True,
                "result_status": "INITIALIZED_ONLY",
                "learning_status": "NOT_RUN",
                "contract_id": CONTRACT_ID,
                "task_identity": identity,
                "task_identity_sha256": sha256_json(identity),
                "source_and_asset_hashes": hashes,
                "stage": args.stage,
                "requested_iterations": args.iterations,
                "optimizer_update_calls": 0,
                "valid_update_count": 0,
                "actor_parameters_changed": False,
                "actor_parameter_delta": parameter_delta,
                "initialization_kind": initialization_kind,
                "policy_initialization": policy_initialization,
                "initial_checkpoint": initial_checkpoint,
                "final_checkpoint": None,
                "resume": resume_info,
                "episode_summary": episode_accumulator.summary(),
                "inspections": [],
                "inspection_window_status": {
                    "minimum_completed_episodes_per_window": 32,
                    "window_policy": "non-overlapping windows checked throughout training",
                    "windows_completed": 0,
                    "pending_completed_episodes": 0,
                    "consecutive_zero_contact_windows": 0,
                    "zero_contact_metric": None,
                    "early_stop_requires_two_consecutive_windows": True,
                },
                "video_recording": run_metadata["video_recording"],
                "independent_evaluation_required": True,
                "agent_config": agent_config,
                "agent_settings": agent_settings,
                "written_before_simulation_close": True,
            }
            atomic_write_json(log_dir / "summary.json", summary)
            atomic_write_json(
                log_dir / "receipt.json",
                {
                    "valid": True,
                    "status": "INITIALIZED_ONLY",
                    "summary_path": str(log_dir / "summary.json"),
                    "checkpoint_path": initial_checkpoint["path"],
                    "checkpoint_sha256": initial_checkpoint["sha256"],
                    "contract_id": CONTRACT_ID,
                    "stage": args.stage,
                },
            )
            exit_code = 0
            return exit_code, files_to_sync

        if args.iterations <= 0:
            raise ValueError("PPO training requires --iterations > 0 unless --initialize-only is used")

        atomic_write_json(
            log_dir / "summary.json",
            {
                "valid": False,
                "result_status": "RUNNING",
                "learning_status": "RUNNING",
                "contract_id": CONTRACT_ID,
                "stage": args.stage,
                "initial_checkpoint": initial_checkpoint,
                "written_before_simulation_close": True,
            },
        )
        append_jsonl(
            log_dir / "inspections.jsonl",
            {
                "inspection": "pre_training",
                "completed_episodes": 0,
                "contact_fraction_zero": None,
                "same_immediate_failure": None,
                "missing_reason": "no training episode has completed yet",
            },
        )

        stage = args.stage
        policy_for_logging = policy
        last_periodic_save = time.monotonic()
        episode_summary = episode_accumulator.summary()
        original_update = runner.alg.update
        original_log = runner.logger.log
        original_step = wrapped.step

        def tracked_update(*update_args: Any, **update_kwargs: Any) -> Any:
            _raise_for_signal()
            result = original_update(*update_args, **update_kwargs)
            update_state["optimizer_update_calls"] += 1
            update_state["last_update_losses"] = json_safe(result)
            finite = _check_loss_finite(torch, result)
            update_state["losses_finite"] = update_state["losses_finite"] and finite
            if finite:
                update_state["valid_update_count_this_process"] += 1
                update_state["stage_updates"] += 1
                update_state["total_updates"] += 1
            return result

        runner.alg.update = tracked_update

        def _progress_info() -> dict[str, Any]:
            return _checkpoint_info(
                identity=identity,
                hashes=hashes,
                stage=stage,
                agent_config=agent_config,
                architecture=architecture,
                progress={
                    "optimizer_update_calls_this_process": update_state["optimizer_update_calls"],
                    "valid_update_count_this_process": update_state["valid_update_count_this_process"],
                    "stage_updates_completed": update_state["stage_updates"],
                    "total_updates_completed": update_state["total_updates"],
                    "next_iteration": (
                        update_state["start_iteration"]
                        + update_state["valid_update_count_this_process"]
                    ),
                    "current_learning_iteration_semantics": "zero-based label; not an update count",
                },
                resume_info=resume_info,
                initialization_kind=initialization_kind,
                policy_initialization=policy_initialization,
            )

        def _save_periodic(elapsed: float, iteration: int | None) -> None:
            nonlocal last_periodic_save
            elapsed_minutes = max(30, int(elapsed // 1800) * 30)
            name = f"checkpoint_elapsed_{elapsed_minutes:04d}m_update_{update_state['total_updates']:07d}.pt"
            result = _save_checkpoint(
                runner, torch, log_dir / name, _progress_info()
            )
            checkpoints.append({"kind": "periodic", **result})
            append_jsonl(
                log_dir / "checkpoints.jsonl",
                {
                    "kind": "periodic",
                    **result,
                    "elapsed_wall_s": elapsed,
                    "iteration_index": iteration,
                    "updates_completed": update_state["stage_updates"],
                },
            )
            last_periodic_save = time.monotonic()

        def _drain_episode_rows() -> list[dict[str, Any]]:
            nonlocal episode_cursor, episode_summary
            rows, episode_cursor = completed_episode_delta(env, episode_cursor)
            for row in rows:
                append_jsonl(log_dir / "episodes.jsonl", row)
                episode_accumulator.add(row)
            inspection_window_pending.extend(rows)
            episode_summary = episode_accumulator.summary()
            return rows

        def _inspect_ready_windows(iteration: int) -> None:
            nonlocal inspection_window_rows_consumed
            while len(inspection_window_pending) >= 32:
                completed_windows = [
                    item for item in inspections if item.get("window_episode_count") is not None
                ]
                window_rows = inspection_window_pending[:32]
                del inspection_window_pending[:32]
                window_number = len(completed_windows) + 1
                start_episode_index = inspection_window_rows_consumed
                inspection_window_rows_consumed += len(window_rows)
                window_summary = summarize_inspection_window(window_rows)
                stop_reason = inspection_stop_state.observe(window_summary)
                inspection = {
                    "inspection": f"post_episode_window_{window_number}",
                    "iteration_index": iteration,
                    "update_count_this_process": update_state["valid_update_count_this_process"],
                    "window_start_completed_episode_index": start_episode_index,
                    "window_end_completed_episode_index": inspection_window_rows_consumed,
                    "window_episode_count": len(window_rows),
                    "zero_contact_metric": inspection_stop_state.zero_contact_metric,
                    "consecutive_zero_contact_windows": (
                        inspection_stop_state.consecutive_zero_contact_windows
                    ),
                    "stop_reason": stop_reason,
                    "episode_summary": window_summary,
                }
                # 前60次真实更新允许从无接触起步；所有空窗口仍如实记录。
                warmup = update_state["valid_update_count_this_process"] < 60
                inspection["contact_learning_warmup"] = warmup
                inspection["warmup_minimum_actual_updates"] = 60
                if warmup and stop_reason is not None:
                    inspection["deferred_stop_candidate"] = stop_reason
                    inspection["stop_reason"] = None
                inspections.append(inspection)
                append_jsonl(log_dir / "inspections.jsonl", inspection)
                if stop_reason is not None and not warmup:
                    update_state["stop_reason"] = stop_reason
                    raise _StopTraining(stop_reason)

        def logged_update(*log_args: Any, **log_kwargs: Any) -> None:
            original_log(*log_args, **log_kwargs)
            iteration = int(log_kwargs.get("it", -1))
            update_state["last_iteration"] = iteration
            rows = _drain_episode_rows()
            elapsed = time.monotonic() - update_state["started_monotonic"]
            action_std = log_kwargs.get("action_std")
            try:
                action_std_mean = float(action_std.mean().item())
            except Exception:
                action_std_mean = None
            losses = log_kwargs.get("loss_dict", update_state["last_update_losses"])
            record = {
                "iteration_index": iteration,
                "iteration_index_semantics": "zero-based RSL-RL label",
                "optimizer_update_calls": update_state["optimizer_update_calls"],
                "valid_update_count_this_process": update_state["valid_update_count_this_process"],
                "stage_updates_completed": update_state["stage_updates"],
                "total_updates_completed": update_state["total_updates"],
                "losses_finite": update_state["losses_finite"],
                "losses": json_safe(losses),
                "learning_rate": json_safe(log_kwargs.get("learning_rate")),
                "action_std_mean": action_std_mean,
                "collection_time_s": json_safe(log_kwargs.get("collect_time")),
                "learning_time_s": json_safe(log_kwargs.get("learn_time")),
                "elapsed_wall_s": elapsed,
                "new_completed_episodes": len(rows),
                "episode_summary": episode_summary,
                # 只保存末策略步的4-env实时均值快照；不是整回合或整更新总回报。
                "reward_component_last_policy_step": json_safe(env.extras.get("fixed_grasp_reward_terms")),
            }
            append_jsonl(log_dir / "metrics.jsonl", record)
            print(
                f"fixed_fruit_update iteration_index={iteration} "
                f"valid_updates={update_state['valid_update_count_this_process']} "
                f"episodes={episode_summary['episodes_completed']} "
                f"contact_fraction_mean={episode_summary['contact_fraction_mean']}",
                flush=True,
            )

            _raise_for_signal()

            _inspect_ready_windows(iteration)

            now = time.monotonic()
            elapsed = now - update_state["started_monotonic"]
            if elapsed - (last_periodic_save - update_state["started_monotonic"]) >= 1800.0:
                _save_periodic(elapsed, iteration)
            if update_state["deadline"] is not None and now >= update_state["deadline"]:
                update_state["stop_reason"] = "max_seconds"
                raise _StopTraining("max_seconds")
            if not update_state["losses_finite"]:
                update_state["stop_reason"] = "non_finite_update"
                raise _StopTraining("non_finite_update")

        runner.logger.log = logged_update

        def bounded_step(actions: Any) -> Any:
            _raise_for_signal()
            deadline = update_state["deadline"]
            if deadline is not None and time.monotonic() >= deadline:
                update_state["stop_reason"] = "max_seconds"
                raise _StopTraining("max_seconds")
            result = original_step(actions)
            _raise_for_signal()
            return result

        wrapped.step = bounded_step

        try:
            runner.learn(
                num_learning_iterations=args.iterations,
                init_at_random_ep_len=False,
            )
        except _StopTraining as stop:
            update_state["stop_reason"] = stop.reason

        try:
            _drain_episode_rows()
            _inspect_ready_windows(int(update_state["last_iteration"] or 0))
        except _StopTraining as stop:
            update_state["stop_reason"] = stop.reason

        if update_state["valid_update_count_this_process"] > 0:
            last_it = update_state["last_iteration"]
            if last_it is not None:
                runner.current_learning_iteration = int(last_it)

        final_actor_state = _actor_state(policy_for_logging)
        actor_final_path = log_dir / "actor_final.pt"
        _atomic_torch_save(torch, {"state_dict": final_actor_state}, actor_final_path)
        parameter_delta = _parameter_delta(torch, actor_before, policy_for_logging)
        stage_updates = update_state["stage_updates"]
        total_updates = update_state["total_updates"]
        next_iteration = (
            update_state["start_iteration"]
            + update_state["valid_update_count_this_process"]
        )
        final_info = _checkpoint_info(
            identity=identity,
            hashes=hashes,
            stage=stage,
            agent_config=agent_config,
            architecture=architecture,
            progress={
                "optimizer_update_calls_this_process": update_state["optimizer_update_calls"],
                "valid_update_count_this_process": update_state["valid_update_count_this_process"],
                "stage_updates_completed": stage_updates,
                "total_updates_completed": total_updates,
                "next_iteration": next_iteration,
                "current_learning_iteration_semantics": "zero-based label; not an update count",
            },
            resume_info=resume_info,
            initialization_kind=initialization_kind,
            policy_initialization=policy_initialization,
        )
        final_checkpoint = _save_checkpoint(
            runner, torch, log_dir / "final.pt", final_info
        )
        checkpoints.append({"kind": "final", **final_checkpoint})
        append_jsonl(
            log_dir / "checkpoints.jsonl",
            {
                "kind": "final",
                **final_checkpoint,
                "updates_completed": stage_updates,
                "optimizer_update_calls": update_state["optimizer_update_calls"],
            },
        )

        update_valid = (
            update_state["valid_update_count_this_process"] > 0
            and parameter_delta.get("changed") is True
            and parameter_delta.get("parameters_finite") is True
            and update_state["losses_finite"]
        )
        result_status = "TRAINED" if update_valid else "NOT_UPDATED"
        summary = {
            "valid": bool(update_valid),
            "result_status": result_status,
            "learning_status": "COMPLETE" if update_state["stop_reason"] == "iterations_completed" else "STOPPED",
            "stop_reason": update_state["stop_reason"],
            "contract_id": CONTRACT_ID,
            "task_identity": identity,
            "task_identity_sha256": sha256_json(identity),
            "contract_sha256": contract_sha256(identity),
            "source_and_asset_hashes": hashes,
            "stage": args.stage,
            "agent_config": agent_config,
            "agent_settings": agent_settings,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "requested_iterations": args.iterations,
            "optimizer_update_calls": update_state["optimizer_update_calls"],
            "valid_update_count": update_state["valid_update_count_this_process"],
            "stage_updates_completed": stage_updates,
            "total_updates_completed": total_updates,
            "start_iteration_index": update_state["start_iteration"],
            "last_iteration_index": update_state["last_iteration"],
            "iteration_index_semantics": "current_learning_iteration is a zero-based iteration label, not an update count",
            "actor_parameters_changed": parameter_delta.get("changed"),
            "actor_parameter_delta": parameter_delta,
            "initialization_kind": initialization_kind,
            "policy_initialization": policy_initialization,
            "actor_initial_path": str(log_dir / "actor_initial.pt"),
            "actor_initial_sha256": sha256_file(log_dir / "actor_initial.pt"),
            "actor_final_path": str(actor_final_path),
            "actor_final_sha256": sha256_file(actor_final_path),
            "initial_checkpoint": initial_checkpoint,
            "final_checkpoint": final_checkpoint,
            "periodic_checkpoints": checkpoints,
            "episode_summary": episode_summary,
            "inspections": inspections,
            "inspection_window_status": {
                "minimum_completed_episodes_per_window": 32,
                "window_policy": "non-overlapping windows checked throughout training",
                "windows_completed": sum(
                    item.get("window_episode_count") is not None for item in inspections
                ),
                "pending_completed_episodes": len(inspection_window_pending),
                "consecutive_zero_contact_windows": (
                    inspection_stop_state.consecutive_zero_contact_windows
                ),
                "zero_contact_metric": inspection_stop_state.zero_contact_metric,
                "early_stop_requires_two_consecutive_windows": True,
            },
            "resume": resume_info,
            "video_recording": run_metadata["video_recording"],
            "independent_evaluation_required": True,
            "written_before_simulation_close": True,
        }
        atomic_write_json(log_dir / "summary.json", summary)
        atomic_write_json(
            log_dir / "receipt.json",
            {
                "valid": bool(update_valid),
                "status": result_status,
                "summary_path": str(log_dir / "summary.json"),
                "checkpoint_path": final_checkpoint["path"],
                "checkpoint_sha256": final_checkpoint["sha256"],
                "contract_id": CONTRACT_ID,
                "stage": args.stage,
                "valid_update_count": update_state["valid_update_count_this_process"],
                "actor_parameters_changed": parameter_delta.get("changed"),
            },
        )
        if update_valid:
            exit_code = 0
        else:
            error = RuntimeError(
                "PPO did not produce a finite update with nonzero actor parameter change"
            )
            _write_failure(
                log_dir,
                error,
                "post_training_validation",
                {
                    "valid_update_count": update_state["valid_update_count_this_process"],
                    "parameter_delta": parameter_delta,
                    "stop_reason": update_state["stop_reason"],
                },
            )
            exit_code = 1

    except BaseException as exc:
        drain_provider = locals().get("_drain_episode_rows")
        if callable(drain_provider):
            try:
                drain_provider()
            except BaseException:
                pass
        if initialized and runner is not None and env is not None:
            try:
                policy = runner.alg.get_policy()
                actor_path = log_dir / "actor_final.pt"
                _atomic_torch_save(torch, {"state_dict": _actor_state(policy)}, actor_path)
                if not (log_dir / "final.pt").exists():
                    progress_provider = locals().get("_progress_info")
                    progress_info = (
                        progress_provider()
                        if callable(progress_provider)
                        else locals().get("info", {})
                    )
                    if progress_info:
                        failure_checkpoint = _save_checkpoint(
                            runner, torch, log_dir / "final.pt", progress_info
                        )
                        checkpoints.append({"kind": "failure_final", **failure_checkpoint})
            except BaseException:
                pass
        _write_failure(
            log_dir,
            exc,
            "training",
            {
                "initialized": initialized,
                "optimizer_update_calls": update_state["optimizer_update_calls"],
                "valid_update_count": update_state["valid_update_count_this_process"],
                "stage_updates_completed": update_state["stage_updates"],
                "stop_reason": update_state["stop_reason"],
                "checkpoints": checkpoints,
                "resume": resume_info,
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
                    log_dir,
                    close_exc,
                    "environment_close",
                    {"valid_summary_written_before_close": (log_dir / "summary.json").exists()},
                )
                sync_evidence(files_to_sync)
                exit_code = 1
        elif env is not None:
            try:
                env.close()
            except BaseException as close_exc:
                _write_failure(log_dir, close_exc, "environment_close", {})
                sync_evidence(files_to_sync)
                exit_code = 1
    return exit_code, files_to_sync


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    initialize_only = _validate_cli(args, parser)
    try:
        log_dir = _prepare_log_dir(args.log_dir)
    except Exception as exc:
        parser.error(str(exc))

    app = None
    exit_code = 1
    stop_request: dict[str, int | None] = {"signum": None}
    try:
        app = AppLauncher(args).app

        def request_stop(signum: int, _frame: Any) -> None:
            # 不在 signal handler 内执行 I/O 或 Kit 操作，只记录退出请求。
            stop_request["signum"] = int(signum)

        # AppLauncher 安装的默认 SIGTERM 回调会直接关闭 Kit 并以 0 退出。
        # 这里在启动完成后替换它，让训练循环先保存 checkpoint 和回执。
        signal.signal(signal.SIGTERM, request_stop)
        exit_code, _ = _run(args, log_dir, initialize_only, stop_request)
    except BaseException as exc:
        _write_failure(
            log_dir,
            exc,
            "app_or_entrypoint",
            {"initialize_only": initialize_only, "command": list(sys.argv)},
        )
        exit_code = 1
    finally:
        # 所有回执、JSONL 记录和检查点均在关闭环境或 App 前完成同步。
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        if app is not None:
            try:
                app.close(exit_code=exit_code)
            finally:
                # close 可能直接 os._exit；若它返回或抛错，也不再运行清理逻辑。
                os._exit(exit_code)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
