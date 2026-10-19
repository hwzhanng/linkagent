#!/usr/bin/env python3
"""F3/F4 firm-grasp PPO 训练与确定性评估入口。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .release_profile import DEFAULT_SUPPORT_RAMP_POLICY_STEPS
from .oriented_reset import (DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ,
    make_oriented_reset_condition)
from .actor_seed import (
    DEFAULT_F2_ACTOR_SEED,
    DEFAULT_F2_ACTOR_SEED_SHA256,
    DEFAULT_LEGACY_THUMB_SEED,
)


TASK_ID = "firm_grasp_v1"
TASK_NAME = "FirmGrasp-v1"
TASK_VERSION = "firm_grasp_v2.3.0"
PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parents[1]
EXPECTED_ACTION_DIM = 14
EXPECTED_ACTOR_DIM = 162
EXPECTED_CRITIC_DIM = 223
ROLLOUT_STEPS = 256
CHECKPOINT_INTERVAL = 1
# 预检上界：任务的固定释放步为 120 策略步（6 s）。权威校验仍在
# firm_grasp_v1.env.make_config 内按实际 cfg.support_release_step 执行。
SUPPORT_RELEASE_STEP_UPPER_BOUND = 120
ACTOR_WARMSTART_DISABLED_MESSAGE = (
    "--actor-warmstart 已禁用：firm_grasp_v1.7.0 的动作历史槽8/9已从"
    "ring/little residual改为index/middle abduction，旧actor不具备完整新动作语义；"
    "请使用 fresh 或严格同版本 S8 checkpoint"
)


class _StopRun(Exception):
    """在策略步边界结束当前运行并先写出可恢复材料。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _json_ready(value: Any) -> Any:
    """把张量、NumPy 标量和非有限浮点值转换为严格 JSON 值。"""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return _json_ready(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    raise TypeError(f"值无法转换为 JSON：{type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_identity_paths(value: Any) -> Any:
    """用源码根相对路径保存身份，便于同一源码树换挂载点后复核。"""

    if isinstance(value, Mapping):
        return {str(key): _normalize_identity_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_identity_paths(child) for child in value]
    if isinstance(value, str) and value.startswith("/"):
        candidate = Path(value)
        try:
            relative = candidate.resolve(strict=False).relative_to(SOURCE_ROOT.resolve())
        except (OSError, ValueError):
            return value
        return "$SOURCE/" + relative.as_posix()
    return value


def normalize_task_identity(raw_identity: Any) -> dict[str, Any]:
    """移除运行参数并保留完整物理、观测、奖励和校准合同。"""

    if not isinstance(raw_identity, Mapping):
        raise TypeError("env.task_identity() 必须返回 mapping")
    identity = _json_ready(raw_identity)
    if identity.get("contract_id") != TASK_ID or identity.get("task_id") != TASK_NAME:
        raise ValueError("环境身份不属于 firm_grasp_v1")
    identity.pop("run", None)
    return _normalize_identity_paths(identity)


def _identity_from_env(env: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provider = getattr(env, "task_identity", None)
    raw = provider() if callable(provider) else provider
    if not isinstance(raw, Mapping):
        raise TypeError("FirmGraspEnv 必须提供 task_identity()")
    raw_identity = _json_ready(raw)
    run_fields = raw_identity.get("run", {})
    return (
        normalize_task_identity(raw_identity),
        run_fields if isinstance(run_fields, dict) else {},
        raw_identity,
    )


def _critical_source_manifest(identity: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    """绑定新任务源码、父环境实现、agent 配置助手和其已声明依赖。"""

    files: dict[str, str] = {}
    task_exclusions = {"__pycache__", "tests"}
    excluded_task_files = {
        "launch_probe.py",
        "launch.py",
        "probe.py",
        "verify_window.py",
        "reachability_scan.py",
        "probe_contacts.py",
        "analyze_probe.py",
        "analyze_pad_surface.py",
        "thumb_reposition.py",
        "reachability.py",
        "reachability_central.py",
        "record_video.py",
        "smoke_new_task.py",
    }
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE_DIR)
        if any(part in task_exclusions for part in relative.parts) or path.name in excluded_task_files:
            continue
        if path.suffix not in {".py", ".json", ".npz", ".usd", ".usda"}:
            continue
        files["firm_grasp_v1/" + relative.as_posix()] = _sha256_file(path)

    # 旧PAD模块为只读继承的实际运行依赖，一并绑定，避免新包遗漏其分类器。
    for path in sorted((SOURCE_ROOT / "research/thumb_pad_grasp_v1").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".npz"} and "tests" not in path.parts:
            files[path.relative_to(SOURCE_ROOT).as_posix()] = _sha256_file(path)

    fixed_sources = (
        "research/fixed_fruit_grasp_v1/env.py",
        "research/fixed_fruit_grasp_v1/contract.py",
        "research/fixed_fruit_grasp_v1/physical_contacts.py",
        "research/fixed_fruit_grasp_v1/physical_hold.py",
        "research/fixed_fruit_grasp_v1/collision_cooking.py",
        "research/fixed_fruit_grasp_v1/run_utils.py",
        "research/small_fruit_pinch_v1/source/agents.py",
    )
    for relative in fixed_sources:
        path = SOURCE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"关键任务源码缺失：{relative}")
        files[relative] = _sha256_file(path)

    dependencies = identity.get("source_dependencies")
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise ValueError("任务身份缺少父环境 source_dependencies")
    for name, dependency in sorted(dependencies.items()):
        if not isinstance(dependency, Mapping):
            raise ValueError(f"source_dependencies.{name} 格式错误")
        path_value = dependency.get("source_path")
        declared_hash = dependency.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"source_dependencies.{name} 缺少 source_path")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"父环境依赖源码缺失：{name}: {path}")
        actual_hash = _sha256_file(path)
        if not isinstance(declared_hash, str) or actual_hash != declared_hash:
            raise ValueError(f"父环境 source_dependencies 哈希不匹配：{name}")
        files["environment_dependency/" + str(name)] = actual_hash

    return files, _sha256_json(files)


def _parse_action_vector(raw: str | None, option_name: str) -> list[float] | None:
    """解析仅用于策略初始化的 14 维有限有界动作向量。"""

    if raw is None:
        return None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} 必须是 JSON 数组") from exc
    if not isinstance(values, list) or len(values) != EXPECTED_ACTION_DIM:
        raise ValueError(f"{option_name} 必须恰有 {EXPECTED_ACTION_DIM} 个数值")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{option_name} 第 {index} 项不是数值")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{option_name} 第 {index} 项不是有限数值") from exc
        if not math.isfinite(number) or not -1.0 <= number <= 1.0:
            raise ValueError(f"{option_name} 第 {index} 项必须在 [-1, 1] 内且为有限值")
        result.append(number)
    return result


def _parse_action_bias(raw: str | None) -> list[float] | None:
    """解析仅用于 fresh 策略初始化的 14 维有界动作均值。"""

    return _parse_action_vector(raw, "--initial-action-bias-json")


def _environment_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """把 CLI 中的 reset、F3 goal 与 pulse 条件传给任务配置。"""

    return {
        "thumb_mapping": args.thumb_mapping,
        "num_envs": args.num_envs,
        "device": args.device,
        "seed": args.seed,
        "stage": "contact",
        "support_ramp_policy_steps": args.support_ramp_policy_steps,
        "wrist_tilt_deg": args.wrist_tilt_deg,
        "wrist_tilt_axis_world_xyz": tuple(args.wrist_tilt_axis_world_xyz),
        "reset_fruit_reference_hand_m": args.reset_fruit_reference_hand_m,
        "base_lift_m": args.base_lift_m,
        "reset_reference_offset_hand_m": (
            None if args.reset_reference_offset_hand_mm is None
            else tuple(value / 1000.0 for value in args.reset_reference_offset_hand_mm)
        ),
        "pregrasp_distance_m": (
            None if args.pregrasp_distance_mm is None else args.pregrasp_distance_mm / 1000.0
        ),
        "fruit_center_local_m": (
            None if args.center_local_mm is None
            else tuple(value / 1000.0 for value in args.center_local_mm)
        ),
        "goal": args.goal,
        "hold_seconds": args.hold_seconds,
        "disturbance_mode": args.disturbance_mode,
        "disturbance_axis": args.disturbance_axis,
        "disturbance_sign": args.disturbance_sign,
        "force_ratio": args.force_ratio,
        "torque_ratio": args.torque_ratio,
        "legacy_thumb_baseline": args.legacy_thumb_baseline,
        "support_release_power": args.support_release_power,
        "preparation_reference_offset_hand_m": (None if args.preparation_reference_offset_hand_mm is None
            else tuple(v/1000. for v in args.preparation_reference_offset_hand_mm)),
    }


def validate_checkpoint_info(
    info: Any,
    *,
    task_identity: Mapping[str, Any],
    identity_sha256: str,
    critical_source_sha256: str,
    observation_dimensions: Mapping[str, int],
) -> int:
    """只接受同一新任务、相同源码和相同策略维度的完整 checkpoint。"""

    if not isinstance(info, Mapping):
        raise ValueError("checkpoint 缺少 firm_grasp_v1 元数据")
    stored_identity = info.get("task_identity")
    if not isinstance(stored_identity, Mapping):
        raise ValueError("checkpoint 缺少 task_identity；禁止旧任务权重迁移")
    normalized_stored = normalize_task_identity(stored_identity)
    stored_identity_hash = info.get("identity_sha256")
    if stored_identity_hash != _sha256_json(normalized_stored):
        raise ValueError("checkpoint task_identity SHA256 自校验失败")
    if normalized_stored != dict(task_identity) or stored_identity_hash != identity_sha256:
        raise ValueError("checkpoint 物理/观测/奖励/校准身份与当前任务不匹配")
    if info.get("critical_source_sha256") != critical_source_sha256:
        raise ValueError("checkpoint critical_source_sha256 与当前源码不匹配")
    stored_dimensions = info.get("observation_dimensions")
    if not isinstance(stored_dimensions, Mapping):
        raise ValueError("checkpoint 缺少 actor/critic/action 维度")
    if {key: int(stored_dimensions.get(key, -1)) for key in observation_dimensions} != dict(
        observation_dimensions
    ):
        raise ValueError("checkpoint actor/critic/action 维度与当前 wrapper 不匹配")
    completed = info.get("completed_updates")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise ValueError("checkpoint completed_updates 必须是非负整数")
    transitions = info.get("interaction_transitions")
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
        raise ValueError("checkpoint interaction_transitions 必须是非负整数")
    return completed


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    run_root = SOURCE_ROOT.parent.resolve()
    if not output.is_relative_to(run_root / "evidence"):
        raise ValueError("新任务输出只允许位于独立run/evidence中")
    link = json.loads((run_root / "BASELINE_LINK.json").read_text())
    for forbidden in link["protected_paths"]:
        protected = Path(forbidden).resolve()
        if output == protected or output.is_relative_to(protected):
            raise ValueError("拒绝写入原版或留存目录")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    return output


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("x", encoding="utf-8")

    def write(self, row: Any) -> None:
        self.stream.write(json.dumps(_json_ready(row), ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def _write_failure(output: Path, exc: BaseException, phase: str, context: Mapping[str, Any] | None = None) -> None:
    try:
        _write_json(
            output / "failure.json",
            {
                "status": "FAILED",
                "task_id": TASK_ID,
                "phase": phase,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "context": dict(context or {}),
                "written_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except BaseException as write_exc:
        print(f"无法写出 failure.json：{write_exc!r}", file=sys.stderr, flush=True)


def _make_parser(app_launcher: Any | None = None) -> argparse.ArgumentParser:
    if app_launcher is None:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    mode_argument = parser.add_argument("--mode", choices=("train", "evaluate"))
    output_argument = parser.add_argument("--output", type=Path, help="必须是尚不存在的唯一输出目录")
    parser.add_argument(
        "--actor-warmstart",
        type=Path,
        default=None,
        help="已禁用：S7 动作8/9改为index/middle abduction，需fresh或同S8 checkpoint",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="同一 firm_grasp_v1 的显式续训或评估权重")
    parser.add_argument(
        "--actor-seed-checkpoint",
        type=Path,
        default=None,
        help="只导入 actor 与 normalizer；新建身份、critic 和 optimizer",
    )
    parser.add_argument("--legacy-thumb-baseline", action="store_true",
        help="恢复冻结 thumb-pad v1.6 mapper/filter；仅该模式允许 v1.6 actor seed")
    parser.add_argument("--iterations", type=int, default=10, help="本次训练新增 PPO 更新数")
    parser.add_argument("--episodes", type=int, default=10, help="评估分母回合数")
    parser.add_argument("--policy", choices=("checkpoint", "open", "fresh"), default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--thumb-mapping",choices=("legacy","independent"),default="independent")
    parser.add_argument("--goal", choices=("hold", "g2"), default="g2")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--disturbance-mode", choices=("quiet", "force", "torque"), default="quiet")
    parser.add_argument("--disturbance-axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--disturbance-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--force-ratio", type=float, default=0.1)
    parser.add_argument("--torque-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--support-ramp-policy-steps",
        type=int,
        default=DEFAULT_SUPPORT_RAMP_POLICY_STEPS,
        help="撤支撑线性渐退的策略步数（20 Hz）；默认最后20步=1 s退出，0保留原突撤对照",
    )
    parser.add_argument("--center-local-mm", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--wrist-tilt-deg", type=float, default=0.0,
        help="S5 reset 基座整体倾斜角，|deg|<=45；非零时必须显式 reference")
    parser.add_argument("--wrist-tilt-axis-world-xyz", nargs=3, type=float,
        default=list(DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ), metavar=("X", "Y", "Z"),
        help="S5 世界系旋转轴；内部单位化并拒绝零向量")
    parser.add_argument("--base-lift-m", type=float, default=0.0,
        help="reset 时固定基座的竖直平移（米，<=0.10），仅用于排除定向 reset 手部碰地")
    parser.add_argument("--reset-reference-offset-hand-mm", nargs=3, type=float, default=None,
        metavar=("X", "Y", "Z"),
        help="reset reference 的手系平移（mm，每轴<=5），仅初始化条件")
    parser.add_argument("--preparation-reference-offset-hand-mm", nargs=3, type=float, default=None,
        help="准备支撑弹簧参考的手系位移(mm)，35→80策略步线性引入，非状态写入")
    parser.add_argument("--reset-fruit-reference-hand-mm", nargs=3, type=float, default=None,
        metavar=("X", "Y", "Z"), help="S5 base_link 坐标系果实中心 reference（mm）")
    parser.add_argument("--pregrasp-distance-mm", type=float, default=None)
    parser.add_argument("--support-release-power",type=float,default=1.,help="原线性退出比例的指数[1,4]；6秒仍严格为零")
    parser.add_argument("--initial-action-bias-json", default=None, help="fresh 策略的 14 個 [-1,1] JSON 動作均值")
    parser.add_argument("--actor-mean-offset-json", default=None,
        help="新任务初始化：在已迁移actor输出偏置上加14D有界偏移，不改原权重文件")
    parser.add_argument("--exploration-std", type=float, default=0.002,
        help="fresh 策略所有动作维度的实际 Gaussian std")
    parser.add_argument("--learning-rate",type=float,default=2e-5,
        help="新任务PPO固定学习率；实际值进入agent_config")
    parser.add_argument("--freeze-actor-normalizer", action="store_true",
        help="冻结 actor 观测归一化统计；critic normalizer 仍按其自身样本更新")
    parser.add_argument("--freeze-actor-log-std", action="store_true",
        help="冻结 actor 的 Gaussian log_std；保持显式探索方差，不让 PPO 自行放大")
    parser.add_argument("--actor-mean-shift-max", type=float, default=None,
        help=("受约束PPO：每个update后在本次rollout观测上比较当前actor与初始化prior的"
              "平均绝对mean动作偏移；超过该值时把actor参数向prior线性收缩到边界。"
              "仅训练模式可用，需配合 --freeze-actor-normalizer。"))
    parser.add_argument(
        "--warmstart-action-offset-json",
        default=None,
        help="已禁用：它依赖已禁用的 --actor-warmstart",
    )
    parser.add_argument("--max-env-steps", type=int, default=None, help="評估总交互步数上限；省略时按回合上限和 env 数计算")
    parser.add_argument("--max-seconds", type=float, default=None, help="训练墙钟上限；只在策略步边界停止")
    app_launcher.add_app_launcher_args(parser)
    mode_argument.required = True
    output_argument.required = True
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.policy is None:
        args.policy = "fresh" if args.mode == "train" else "checkpoint"
    args.actor_mean_offset = None
    if args.actor_mean_offset_json is not None:
        offset = json.loads(args.actor_mean_offset_json)
        if (not isinstance(offset, list) or len(offset) != 14
                or any(isinstance(x, bool) or not isinstance(x, (float, int))
                       or not math.isfinite(x) or abs(x) > .05 for x in offset)):
            raise ValueError("actor mean offset须为14D有限数值，每维绝对值<=0.05")
        if args.policy != 'fresh' or args.checkpoint is not None or args.legacy_thumb_baseline:
            raise ValueError("mean offset仅用于新候选初始化，不用于严格重载或旧版对照")
        args.actor_mean_offset = offset
    if args.num_envs <= 0:
        raise ValueError("--num-envs 必须为正数")
    if args.seed < 0:
        raise ValueError("--seed 必须为非负整数")
    if not math.isfinite(args.hold_seconds) or args.hold_seconds <= 0.0:
        raise ValueError("--hold-seconds 必须为正的有限值")
    if not math.isfinite(args.disturbance_sign) or args.disturbance_sign not in (-1.0, 1.0):
        raise ValueError("--disturbance-sign 必须为 -1 或 1")
    for option_name in ("force_ratio", "torque_ratio"):
        value = getattr(args, option_name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{option_name.replace('_', '-')} 必须为非负有限值")
    if not math.isfinite(args.exploration_std) or args.exploration_std <= 0.0:
        raise ValueError("--exploration-std 必须为正的有限值")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.:
        raise ValueError('--learning-rate必须为正有限值')
    constrained_actor = (args.freeze_actor_normalizer or args.freeze_actor_log_std
                         or args.actor_mean_shift_max is not None)
    if constrained_actor and args.mode != 'train':
        raise ValueError('actor 归一化/探索方差/mean偏移约束仅用于 train 模式')
    if args.actor_mean_shift_max is not None:
        if not math.isfinite(args.actor_mean_shift_max) or not (0.0 < args.actor_mean_shift_max <= 0.05):
            raise ValueError('--actor-mean-shift-max 必须位于 (0, 0.05]')
        if not args.freeze_actor_normalizer:
            raise ValueError('--actor-mean-shift-max 必须配合 --freeze-actor-normalizer，'
                             '否则prior与当前actor的归一化统计会漂移')
    if args.iterations <= 0:
        raise ValueError("--iterations 必须为正数")
    if args.episodes <= 0:
        raise ValueError("--episodes 必须为正数")
    if args.max_env_steps is not None and args.max_env_steps <= 0:
        raise ValueError("--max-env-steps 必须为正数")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("--max-seconds 必须为正的有限值")
    if args.support_ramp_policy_steps < 0:
        raise ValueError("--support-ramp-policy-steps 不能为负；0 表示原突撤对照")
    if args.support_ramp_policy_steps > SUPPORT_RELEASE_STEP_UPPER_BOUND:
        raise ValueError(
            f"--support-ramp-policy-steps 不得超过释放步 {SUPPORT_RELEASE_STEP_UPPER_BOUND}"
        )
    if args.pregrasp_distance_mm is not None and (
        not math.isfinite(args.pregrasp_distance_mm) or args.pregrasp_distance_mm <= 0
    ):
        raise ValueError("--pregrasp-distance-mm 必须为正的有限值")
    if not math.isfinite(args.base_lift_m) or not (0.0 <= args.base_lift_m <= 0.10):
        raise ValueError("--base-lift-m 必须位于 [0, 0.10] 米")
    if args.reset_reference_offset_hand_mm is not None:
        if not all(math.isfinite(value) and abs(value) <= 5.0
                   for value in args.reset_reference_offset_hand_mm):
            raise ValueError("--reset-reference-offset-hand-mm 每轴必须位于 [-5, 5] mm")
        if args.reset_fruit_reference_hand_mm is None or args.wrist_tilt_deg == 0.0:
            raise ValueError("--reset-reference-offset-hand-mm 仅用于 reference reset 条件")
    if args.center_local_mm is not None and not all(math.isfinite(value) for value in args.center_local_mm):
        raise ValueError("--center-local-mm 必须由三个有限值组成")
    if args.reset_fruit_reference_hand_mm is not None and args.center_local_mm is not None:
        raise ValueError("reference 模式禁止同时显式给出 --center-local-mm")
    condition = make_oriented_reset_condition(args.wrist_tilt_deg,
        args.wrist_tilt_axis_world_xyz,
        (None if args.reset_fruit_reference_hand_mm is None
         else tuple(value / 1000.0 for value in args.reset_fruit_reference_hand_mm)))
    args.reset_fruit_reference_hand_m = condition.reset_fruit_reference_hand_m
    args.wrist_tilt_axis_world_xyz = list(condition.wrist_tilt_axis_world_xyz)
    args.initial_action_bias = _parse_action_bias(args.initial_action_bias_json)
    args.warmstart_action_offset = _parse_action_vector(
        args.warmstart_action_offset_json, "--warmstart-action-offset-json"
    )
    if args.actor_warmstart is not None:
        raise ValueError(ACTOR_WARMSTART_DISABLED_MESSAGE)
    if args.warmstart_action_offset is not None:
        raise ValueError("--warmstart-action-offset-json 已随 --actor-warmstart 一并禁用")

    if args.mode == "train":
        if args.policy == "open":
            raise ValueError("train 模式不能使用 open 对照策略")
        if args.policy == "checkpoint" and args.checkpoint is None:
            raise ValueError("train --policy checkpoint 必须显式提供同任务 --checkpoint")
        if args.policy == "fresh" and args.checkpoint is not None:
            raise ValueError("fresh 训练不能同时提供 --checkpoint")
        if args.max_env_steps is not None:
            raise ValueError("--max-env-steps 仅用于 evaluate 模式")
    else:
        if args.policy == "checkpoint" and args.checkpoint is None:
            raise ValueError("evaluate --policy checkpoint 必须显式提供 --checkpoint")
        if args.policy in {"open", "fresh"} and args.checkpoint is not None:
            raise ValueError("open/fresh 评估不接受 --checkpoint")
        if args.max_seconds is not None:
            raise ValueError("--max-seconds 仅用于 train 模式")
    if args.initial_action_bias is not None and args.policy != "fresh":
        raise ValueError("--initial-action-bias-json 仅用于 fresh 策略")
    if args.actor_seed_checkpoint is not None and args.policy != "fresh":
        raise ValueError("--actor-seed-checkpoint 仅用于 fresh policy；同任务续训请用 --checkpoint")
    if args.actor_seed_checkpoint is not None and args.initial_action_bias is not None:
        raise ValueError("--actor-seed-checkpoint 与 --initial-action-bias-json 互斥")
    if args.legacy_thumb_baseline and args.initial_action_bias is not None:
        raise ValueError("--legacy-thumb-baseline 必须使用冻结 v1.6 actor seed，不能叠加动作均值偏置")
    if args.actor_seed_checkpoint is None and args.policy == "fresh":
        if args.legacy_thumb_baseline:
            args.actor_seed_checkpoint = DEFAULT_LEGACY_THUMB_SEED
        elif args.mode == "train" and args.initial_action_bias is None:
            args.actor_seed_checkpoint = DEFAULT_F2_ACTOR_SEED
    if args.actor_seed_checkpoint is not None:
        if args.checkpoint is not None:
            raise ValueError("--actor-seed-checkpoint 与 strict --checkpoint 互斥")
        if not args.actor_seed_checkpoint.is_file():
            raise ValueError(f"actor seed checkpoint 不存在：{args.actor_seed_checkpoint}")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise ValueError(f"checkpoint 不存在：{args.checkpoint}")


def _tensor_dimensions(observations: Any) -> dict[str, list[int]]:
    dimensions: dict[str, list[int]] = {}

    def visit(value: Any, prefix: str) -> None:
        if hasattr(value, "shape") and hasattr(value, "ndim"):
            dimensions[prefix] = [int(item) for item in value.shape]
            return
        if isinstance(value, Mapping) or hasattr(value, "items"):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                visit(child, name)

    if hasattr(observations, "items"):
        for key, value in observations.items():
            visit(value, str(key))
    else:
        raise TypeError("RslRlVecEnvWrapper.get_observations() 必须返回 TensorDict/mapping")
    return dimensions


def _state_sha256(torch: Any, module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(_canonical_json(list(tensor.shape)))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(_canonical_json(value))
    return digest.hexdigest()


def _trainable_sha256(torch: Any, module: Any) -> str:
    """只绑定可训练参数，排除每次rollout均可能变化的归一化缓冲。"""
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_info(
    *,
    identity: Mapping[str, Any],
    identity_sha256: str,
    critical_source_sha256: str,
    source_files: Mapping[str, str],
    observation_dimensions: Mapping[str, int],
    completed_updates: int,
    interaction_transitions: int,
    observation_shapes: Mapping[str, list[int]],
    agent_config_sha256: str,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_identity": dict(identity),
        "identity_sha256": identity_sha256,
        "critical_source_sha256": critical_source_sha256,
        "critical_source_files": dict(source_files),
        "completed_updates": int(completed_updates),
        "interaction_transitions": int(interaction_transitions),
        "observation_dimensions": dict(observation_dimensions),
        "observation_shapes": dict(observation_shapes),
        "agent_config_sha256": agent_config_sha256,
        "policy_initialization": dict(initialization),
    }


def _save_checkpoint(runner: Any, output: Path, filename: str, info: Mapping[str, Any]) -> dict[str, Any]:
    destination = output / filename
    temporary = output / f".{filename}.{os.getpid()}.tmp"
    try:
        runner.save(str(temporary), infos={TASK_ID: _json_ready(info)})
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(destination), "sha256": _sha256_file(destination), "bytes": destination.stat().st_size}


def _success_label(row: Mapping[str, Any]) -> str:
    value = row.get("success")
    if value is True:
        return "SUCCESS"
    if value is False:
        return "FAILURE"
    return "UNKNOWN"


def _count_dones(dones: Any) -> int:
    if hasattr(dones, "detach"):
        return int(dones.detach().to(dtype=getattr(dones, "dtype", None)).sum().item())
    if isinstance(dones, (list, tuple)):
        return sum(bool(value) for value in dones)
    return int(bool(dones))


def _run(
    args: argparse.Namespace,
    output: Path,
    stop_request: dict[str, int | None],
    phase: dict[str, str],
) -> int:
    phase["value"] = "imports_after_app_start"
    import torch
    from isaaclab.utils.seed import configure_seed
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from research.fixed_fruit_grasp_v1.run_utils import (
        configure_fixed_fruit_agent_cfg,
        initialize_actor_output_bias,
    )
    from .actor_seed import (
        initialize_actor_seed,
        set_fresh_exploration_std,
    )
    from research.small_fruit_pinch_v1.source.agents import SmallFruitPinchPPORunnerCfg
    from research.firm_grasp_v1.env import FirmGraspEnv, make_config

    episode_writer = _JsonlWriter(output / "episodes.jsonl")
    update_writer = _JsonlWriter(output / "updates.jsonl") if args.mode == "train" else None
    reward_terms_writer = _JsonlWriter(output / "pad_reward_terms.jsonl") if args.mode == "train" else None
    trace_writer = _JsonlWriter(output / "pad_trace.jsonl") if args.mode == "evaluate" else None
    env = None
    wrapped = None
    runner = None
    error: BaseException | None = None
    controlled_stop: str | None = None
    task_state: dict[str, Any] = {}
    episode_rows: list[dict[str, Any]] = []
    episode_denominator_rows: list[dict[str, Any]] = []
    episode_cursor = 0
    vector_steps = 0
    transitions_this_run = 0
    traces_written = 0
    write_trace_callback: Any = None
    reward_terms_state: dict[str, Any] = {"sum": {}, "steps": 0, "missing_steps": 0}
    tracker: dict[str, Any] = {
        "completed_updates": 0,
        "updates_this_run": 0,
        "update_attempts_this_run": 0,
        "interaction_transitions": 0,
        "partial_rollout_steps": 0,
        "steps_since_update": 0,
        "pad_terms_sum": {},
        "pad_terms_steps": 0,
        "pad_terms_missing_steps": 0,
    }
    initial_actor_sha256 = None
    initial_critic_sha256 = None
    final_actor_sha256 = None
    final_critic_sha256 = None
    checkpoint_rows: list[dict[str, Any]] = []
    started = time.monotonic()

    def record_episode(raw_row: Any, *, reason: str | None = None) -> None:
        nonlocal episode_rows, episode_denominator_rows
        record = _json_ready(raw_row)
        if not isinstance(record, Mapping):
            record = {"raw_value": record}
        denominator_slot = None
        in_denominator = args.mode == "train"
        if args.mode == "evaluate" and len(episode_denominator_rows) < args.episodes:
            denominator_slot = len(episode_denominator_rows) + 1
            in_denominator = True
        envelope = {
            "record_type": "episode",
            "mode": args.mode,
            "policy": args.policy,
            "episode_index": len(episode_rows) + 1,
            "denominator_slot": denominator_slot,
            "in_requested_denominator": in_denominator,
            "outcome": _success_label(record),
            "record_status": reason,
            "record": dict(record),
        }
        episode_writer.write(envelope)
        print("EPISODE " + json.dumps(envelope, ensure_ascii=False, allow_nan=False), flush=True)
        episode_rows.append(envelope)
        if in_denominator and args.mode == "evaluate":
            episode_denominator_rows.append(envelope)

    try:
        phase["value"] = "seed_and_environment_config"
        configure_seed(args.seed)
        cfg = make_config(**_environment_config_kwargs(args))
        env = FirmGraspEnv(cfg=cfg)
        identity, runtime_identity, raw_task_identity = _identity_from_env(env)
        if identity.get("task_version") != TASK_VERSION:
            raise ValueError(
                f"环境任务版本 {identity.get('task_version')!r} 与入口声明 {TASK_VERSION!r} 不一致"
            )
        identity_sha256 = _sha256_json(identity)
        source_files, critical_source_sha256 = _critical_source_manifest(raw_task_identity)

        phase["value"] = "rsl_wrapper_reset_and_observation_discovery"
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        if not isinstance(getattr(env, "completed_episodes", None), list):
            raise TypeError("FirmGraspEnv.completed_episodes 必须为 append-only list")
        initial_episode_count = len(env.completed_episodes)
        episode_cursor = initial_episode_count
        initial_observations = wrapped.get_observations()
        observation_shapes = _tensor_dimensions(initial_observations)

        phase["value"] = "rsl_runner_construction"
        agent_cfg = SmallFruitPinchPPORunnerCfg()
        agent_cfg.logger = "tensorboard"
        agent_cfg.run_name = TASK_ID
        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, importlib.metadata.version("rsl-rl-lib")
        )
        agent_cfg = configure_fixed_fruit_agent_cfg(agent_cfg)
        # F3 每次更新固定采集 256 个策略步；fresh std 随后覆盖到所有动作维度。
        agent_cfg.num_steps_per_env = ROLLOUT_STEPS
        agent_cfg.actor.distribution_cfg.init_std = args.exploration_std
        agent_cfg.algorithm.learning_rate = args.learning_rate
        if int(agent_cfg.num_steps_per_env) != ROLLOUT_STEPS:
            raise ValueError(f"预期每环境 {ROLLOUT_STEPS} 个 rollout 步，实际为 {agent_cfg.num_steps_per_env}")
        agent_config = _json_ready(agent_cfg.to_dict())
        runner = OnPolicyRunner(wrapped, agent_config, log_dir=None, device=agent_cfg.device)
        runner.logger.writer = None

        observation_dimensions = {
            "actor_dim": int(runner.alg.actor.obs_dim),
            "critic_dim": int(runner.alg.critic.obs_dim),
            "action_dim": int(wrapped.num_actions),
        }
        if observation_dimensions != {
            "actor_dim": EXPECTED_ACTOR_DIM,
            "critic_dim": EXPECTED_CRITIC_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
        }:
            raise ValueError(f"新任务策略维度不匹配：{observation_dimensions}")
        expected_identity_dims = identity.get("observations", {})
        if not isinstance(expected_identity_dims, Mapping) or (
            int(expected_identity_dims.get("policy_dim", -1)) != observation_dimensions["actor_dim"]
            or int(expected_identity_dims.get("critic_dim", -1)) != observation_dimensions["critic_dim"]
        ):
            raise ValueError("task_identity 观测维度与 RslRlVecEnvWrapper / runner 实际维度不一致")

        initialization: dict[str, Any] = {
            "kind": "fresh_rsl_rl_initialization",
            "action_bias": None,
            "new_task_identity": identity_sha256,
            "fresh_critic": True,
            "fresh_optimizer": True,
        }
        if args.actor_seed_checkpoint is not None:
            if args.legacy_thumb_baseline:
                from .transfer_clock import transfer_actor

                initialization = transfer_actor(runner, args.actor_seed_checkpoint)
                initialization.update(
                    classification="frozen_legacy_thumb_actor_only_seed_new_identity",
                    new_task_identity=identity_sha256,
                    strict_full_checkpoint_resume=False,
                )
            else:
                expected_seed_sha256 = (
                    DEFAULT_F2_ACTOR_SEED_SHA256
                    if args.actor_seed_checkpoint.expanduser().resolve()
                    == DEFAULT_F2_ACTOR_SEED.resolve()
                    else None
                )
                initialization = initialize_actor_seed(
                    runner,
                    args.actor_seed_checkpoint,
                    target_identity=identity,
                    target_dimensions=observation_dimensions,
                    expected_sha256=expected_seed_sha256,
                )
                initialization["new_task_identity"] = identity_sha256
        if args.initial_action_bias is not None:
            initialization = {
                **initialization,
                "kind": "zero_output_weight_declared_action_bias_prior",
                **initialize_actor_output_bias(torch, runner.alg.get_policy(), args.initial_action_bias),
            }

        if args.actor_mean_offset is not None:
            actor = runner.alg.get_policy()
            with torch.no_grad():
                actor.mlp[-1].bias.add_(torch.tensor(args.actor_mean_offset,
                    device=actor.mlp[-1].bias.device, dtype=actor.mlp[-1].bias.dtype))
            initialization['declared_initial_mean_offset'] = args.actor_mean_offset
            initialization['offset_gain_claim'] = 'initialization_change_not_PPO_learning'

        # actor seed 先严格复制 mean 与 matching normalizer，再仅修改 fresh
        # Gaussian std；strict checkpoint 分支不调用此初始化，saved std 保持权威。
        if args.policy == "fresh":
            initialization["exploration_std"] = set_fresh_exploration_std(
                runner.alg.get_policy(), args.exploration_std
            )

        if args.actor_warmstart is not None:
            raise ValueError(ACTOR_WARMSTART_DISABLED_MESSAGE)

        if args.checkpoint is not None:
            phase["value"] = "checkpoint_metadata_validation"
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise ValueError("checkpoint 顶层必须为 mapping")
            infos = payload.get("infos")
            task_info = infos.get(TASK_ID) if isinstance(infos, Mapping) else None
            completed_before = validate_checkpoint_info(
                task_info,
                task_identity=identity,
                identity_sha256=identity_sha256,
                critical_source_sha256=critical_source_sha256,
                observation_dimensions=observation_dimensions,
            )
            if args.mode == "train":
                if args.initial_action_bias is not None:
                    raise ValueError("续训不能叠加 fresh-only 初始动作偏置")
                runner.load(str(args.checkpoint), map_location=args.device)
                runner.current_learning_iteration = completed_before
                initialization = dict(task_info.get("policy_initialization", {}))
            else:
                runner.load(str(args.checkpoint), map_location=args.device)
                initialization = dict(task_info.get("policy_initialization", {}))
            tracker["completed_updates"] = completed_before
            previous_transitions = task_info.get("interaction_transitions", 0)
            if isinstance(previous_transitions, bool) or not isinstance(previous_transitions, int) or previous_transitions < 0:
                raise ValueError("checkpoint interaction_transitions 必须是非负整数")
            tracker["interaction_transitions"] = previous_transitions
            checkpoint_sha256 = _sha256_file(args.checkpoint)
        else:
            completed_before = 0
            checkpoint_sha256 = None

        # v2.3 受约束 PPO：只约束 actor，不改变 critic 或物理执行链。
        actor_constraint: dict[str, Any] = {
            "enabled": bool(args.freeze_actor_normalizer or args.freeze_actor_log_std
                            or args.actor_mean_shift_max is not None),
            "freeze_actor_normalizer": bool(args.freeze_actor_normalizer),
            "freeze_actor_log_std": bool(args.freeze_actor_log_std),
            "actor_mean_shift_max": args.actor_mean_shift_max,
            "prior_actor_frozen": False,
            "projection_count": 0,
            "min_mean_abs_shift": None,
            "max_mean_abs_shift": None,
        }
        prior_actor = None
        if args.mode == "train" and actor_constraint["enabled"]:
            constrained_policy = runner.alg.get_policy()
            if args.freeze_actor_normalizer:
                def _skip_actor_normalization(*_args: Any, **_kwargs: Any) -> None:
                    return None

                constrained_policy.update_normalization = _skip_actor_normalization
            if args.freeze_actor_log_std:
                log_std = getattr(getattr(constrained_policy, "distribution", None),
                                  "log_std_param", None)
                if not isinstance(log_std, torch.nn.Parameter):
                    raise TypeError("冻结 log_std 需要 distribution.log_std_param Parameter")
                log_std.requires_grad_(False)
            if args.actor_mean_shift_max is not None:
                prior_actor = copy.deepcopy(constrained_policy)
                prior_actor.eval()
                for parameter in prior_actor.parameters():
                    parameter.requires_grad_(False)
                actor_constraint["prior_actor_frozen"] = True
            initialization = dict(initialization)
            initialization["actor_constraint"] = {
                key: value for key, value in actor_constraint.items()
                if key not in ("projection_count", "min_mean_abs_shift", "max_mean_abs_shift")
            }

        if args.mode == "evaluate" and args.policy == "checkpoint":
            inference_policy = runner.get_inference_policy(device=args.device)
        elif args.mode == "evaluate" and args.policy == "fresh":
            inference_policy = runner.get_inference_policy(device=args.device)
        else:
            inference_policy = None

        open_action = None
        open_policy_info = None
        if args.mode == "evaluate" and args.policy == "open":
            baseline_setter = getattr(env, "set_open_hand_baseline", None)
            if callable(baseline_setter):
                result = baseline_setter(True)
                if result is False or getattr(env, "open_hand_baseline_enabled", True) is False:
                    raise RuntimeError("环境未启用真实 neutral 目标 open-hand baseline")
                open_action = torch.zeros(
                    (args.num_envs, observation_dimensions["action_dim"]),
                    dtype=torch.float32,
                    device=args.device,
                )
                open_policy_info = {
                    "provider": "set_open_hand_baseline",
                    "payload_meaning": "环境旁路忽略零 payload，保持 reset 的 neutral 目标",
                }
            else:
                provider = getattr(env, "open_hand_action", None)
                if not callable(provider):
                    provider = getattr(env, "default_joint_pos_action", None)
                if not callable(provider):
                    raise AttributeError("open 对照需要环境 open-hand baseline 或显式动作提供器")
                open_action = torch.as_tensor(provider(), dtype=torch.float32, device=args.device)
                if open_action.ndim == 1:
                    open_action = open_action.unsqueeze(0).expand(args.num_envs, -1).contiguous()
                if tuple(open_action.shape) != (args.num_envs, observation_dimensions["action_dim"]):
                    raise ValueError(f"open-hand 动作形状错误：{tuple(open_action.shape)}")
                if not bool(torch.isfinite(open_action).all()):
                    raise ValueError("open-hand 动作含非有限值")
                open_policy_info = {"provider": getattr(provider, "__name__", "environment_action")}

        run_metadata = {
            "task_id": TASK_ID,
            "task_version": TASK_VERSION,
            "mode": args.mode,
            "policy": args.policy,
            "checkpoint_path": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "actor_seed_checkpoint_path": (
                str(args.actor_seed_checkpoint.expanduser().resolve())
                if args.actor_seed_checkpoint else None
            ),
            "actor_warmstart_path": str(args.actor_warmstart.resolve()) if args.actor_warmstart else None,
            "warmstart_action_offset": args.warmstart_action_offset,
            "checkpoint_sha256": checkpoint_sha256,
            "task_identity": identity,
            "identity_sha256": identity_sha256,
            "critical_source_sha256": critical_source_sha256,
            "critical_source_files": source_files,
            "observation_dimensions": observation_dimensions,
            "observation_shapes": observation_shapes,
            "runtime_identity": runtime_identity,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": args.device,
            "rollout_steps_per_env": ROLLOUT_STEPS,
            "support_ramp_policy_steps": args.support_ramp_policy_steps,
            "f3_f4_condition": {
                "goal": args.goal,
                "hold_seconds": args.hold_seconds,
                "disturbance_mode": args.disturbance_mode,
                "disturbance_axis": args.disturbance_axis,
                "disturbance_sign": args.disturbance_sign,
                "force_ratio": args.force_ratio,
                "torque_ratio": args.torque_ratio,
                "legacy_thumb_baseline": args.legacy_thumb_baseline,
            },
            "exploration_std_requested": args.exploration_std,
            "requested_pose_mm": {
                "center_local_mm": args.center_local_mm,
                "wrist_tilt_deg": args.wrist_tilt_deg,
                "wrist_tilt_axis_world_xyz": args.wrist_tilt_axis_world_xyz,
                "reset_fruit_reference_hand_mm": args.reset_fruit_reference_hand_mm,
                "base_lift_m": args.base_lift_m,
                "reset_reference_offset_hand_mm": args.reset_reference_offset_hand_mm,
                "pregrasp_distance_mm": args.pregrasp_distance_mm,
            },
            "agent_config": agent_config,
            "agent_config_sha256": _sha256_json(agent_config),
            "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
            "initialization": initialization,
            "actor_constraint": {
                key: value for key, value in actor_constraint.items()
                if key not in ("projection_count", "min_mean_abs_shift", "max_mean_abs_shift")
            },
            "open_policy": open_policy_info,
            "wrapper_reset": "RslRlVecEnvWrapper 构造时执行 env.reset；后续读取 get_observations TensorDict",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(sys.argv),
        }
        _write_json(output / "run_metadata.json", run_metadata)

        initial_actor_trainable_sha256 = _trainable_sha256(torch, runner.alg.get_policy())
        initial_critic_trainable_sha256 = _trainable_sha256(torch, runner.alg.critic)
        initial_actor_sha256 = _state_sha256(torch, runner.alg.get_policy())
        initial_critic_sha256 = _state_sha256(torch, runner.alg.critic)
        if args.mode == "train":
            info = _checkpoint_info(
                identity=identity,
                identity_sha256=identity_sha256,
                critical_source_sha256=critical_source_sha256,
                source_files=source_files,
                observation_dimensions=observation_dimensions,
                completed_updates=tracker["completed_updates"],
                interaction_transitions=tracker["interaction_transitions"],
                observation_shapes=observation_shapes,
                agent_config_sha256=run_metadata["agent_config_sha256"],
                initialization=initialization,
            )
            checkpoint_rows.append({"kind": "initial", **_save_checkpoint(runner, output, "initial.pt", info)})

        def write_trace(policy_step: int) -> None:
            nonlocal traces_written
            if trace_writer is None:
                return
            drain = getattr(env, "drain_pad_records", None)
            if not callable(drain):
                raise AttributeError("FirmGraspEnv 缺少 drain_pad_records()")
            rows = drain()
            if rows is None:
                rows = []
            if not isinstance(rows, (list, tuple)):
                raise TypeError("drain_pad_records() 必须返回逐物理子步 list")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError("pad trace 每条记录必须为 mapping")
                traces_written += 1
                trace_writer.write(
                    {
                        "trace_index": traces_written,
                        "evaluation_policy_step": policy_step,
                        "record": row,
                    }
                )

        write_trace_callback = write_trace

        if args.mode == "evaluate":
            phase["value"] = "evaluation_setup"
            if not hasattr(env, "pad_trace_enabled") or not callable(getattr(env, "drain_pad_records", None)):
                raise AttributeError("评估需要 pad_trace_enabled 和 drain_pad_records()")
            env.pad_trace_enabled = True
            if not bool(env.pad_trace_enabled):
                raise RuntimeError("环境没有启用逐物理子步 pad trace")
            effective_max_transitions = args.max_env_steps
            if effective_max_transitions is None:
                max_episode_length = int(getattr(wrapped, "max_episode_length", 0))
                if max_episode_length <= 0:
                    raise ValueError("环境未提供有效 max_episode_length，必须显式给出 --max-env-steps")
                effective_max_transitions = max_episode_length * (args.episodes + 1) * args.num_envs
            obs = wrapped.get_observations().to(args.device)
            phase["value"] = "evaluation_loop"
            while (
                len(episode_denominator_rows) < args.episodes
                and transitions_this_run < effective_max_transitions
            ):
                if stop_request.get("signal") is not None:
                    controlled_stop = f"termination_signal_{stop_request['signal']}"
                    break
                if args.policy in {"checkpoint", "fresh"}:
                    with torch.inference_mode():
                        action = inference_policy(obs, stochastic_output=False)
                        obs, _reward, dones, extras = wrapped.step(action.to(wrapped.device))
                        obs = obs.to(args.device)
                else:
                    with torch.inference_mode():
                        obs, _reward, dones, extras = wrapped.step(open_action.to(wrapped.device))
                        obs = obs.to(args.device)
                vector_steps += 1
                transitions_this_run += args.num_envs
                done_count = _count_dones(dones)
                episode_delta, next_cursor = _read_episode_delta(env, episode_cursor)
                episode_cursor = next_cursor
                if len(episode_delta) < done_count:
                    for missing_index in range(done_count - len(episode_delta)):
                        episode_delta.append(
                            {
                                "env_id": None,
                                "success": None,
                                "reason": "UNKNOWN_COMPLETION_RECORD_MISSING",
                                "missing_done_row_index": missing_index,
                            }
                        )
                for row in episode_delta:
                    record_episode(row, reason=None if row.get("success") is not None else "environment_result_incomplete")
                write_trace_callback(vector_steps)
            write_trace_callback(vector_steps)
            if len(episode_denominator_rows) < args.episodes:
                missing = args.episodes - len(episode_denominator_rows)
                for index in range(missing):
                    record_episode(
                        {
                            "env_id": None,
                            "success": None,
                            "reason": "UNKNOWN_NO_COMPLETED_EPISODE_BEFORE_LIMIT",
                            "missing_denominator_slot": len(episode_denominator_rows) + 1,
                        },
                        reason="evaluation_limit_or_signal",
                    )

            outcome_counts = {label: sum(row["outcome"] == label for row in episode_denominator_rows) for label in ("SUCCESS", "FAILURE", "UNKNOWN")}
            eval_summary = {
                "requested_denominator": args.episodes,
                "denominator_rows": len(episode_denominator_rows),
                "completed_episode_records": sum(row.get("record_status") != "evaluation_limit_or_signal" for row in episode_denominator_rows),
                "outcomes": outcome_counts,
                "vectorized_policy_steps": vector_steps,
                "interaction_transitions": transitions_this_run,
                "max_interaction_transitions": effective_max_transitions,
                "pad_physical_substep_trace_rows": traces_written,
            }
            if outcome_counts["UNKNOWN"] or len(episode_denominator_rows) != args.episodes:
                raise RuntimeError("评估分母包含 UNKNOWN/缺失回合；已保留所有分母行，结果标记为失败")
            summary = {
                "status": "EVALUATION_COMPLETE",
                "learning_claim": "NO_NEW_WEIGHT_SUCCESS_CLAIM",
                "task_id": TASK_ID,
                "policy": args.policy,
                "task_identity_sha256": identity_sha256,
                "critical_source_sha256": critical_source_sha256,
                "observation_dimensions": observation_dimensions,
                "evaluation": eval_summary,
                "episode_jsonl": str(output / "episodes.jsonl"),
                "pad_trace_jsonl": str(output / "pad_trace.jsonl"),
                "run_metadata": str(output / "run_metadata.json"),
                "elapsed_seconds": time.monotonic() - started,
            }
            _write_json(output / "summary.json", summary)
            return 0

        phase["value"] = "training_setup"
        if args.max_seconds is not None:
            deadline = started + args.max_seconds
        else:
            deadline = None
        original_step = wrapped.step

        def tracked_step(actions: Any) -> Any:
            nonlocal episode_cursor, vector_steps, transitions_this_run
            if stop_request.get("signal") is not None:
                raise _StopRun(f"termination_signal_{stop_request['signal']}")
            if deadline is not None and time.monotonic() >= deadline:
                raise _StopRun("max_seconds")
            result = original_step(actions)
            _next_obs, _reward, dones, extras = result
            vector_steps += 1
            transitions = args.num_envs
            transitions_this_run += transitions
            tracker["interaction_transitions"] += transitions
            tracker["steps_since_update"] += 1
            step_terms = extras.get("pad_reward_terms") if isinstance(extras, Mapping) else None
            if isinstance(step_terms, Mapping):
                tracker["pad_terms_steps"] += 1
                for key, value in step_terms.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                        tracker["pad_terms_sum"][str(key)] = tracker["pad_terms_sum"].get(str(key), 0.0) + float(value)
            else:
                tracker["pad_terms_missing_steps"] += 1

            nonlocal_episode_delta, next_cursor = _read_episode_delta(env, episode_cursor)
            episode_cursor = next_cursor
            done_count = _count_dones(dones)
            if len(nonlocal_episode_delta) < done_count:
                for missing_index in range(done_count - len(nonlocal_episode_delta)):
                    nonlocal_episode_delta.append(
                        {
                            "env_id": None,
                            "success": None,
                            "reason": "UNKNOWN_COMPLETION_RECORD_MISSING",
                            "missing_done_row_index": missing_index,
                        }
                    )
            for row in nonlocal_episode_delta:
                record_episode(row, reason=None if row.get("success") is not None else "environment_result_incomplete")
            return result

        wrapped.step = tracked_step
        original_update = runner.alg.update

        def tracked_update() -> Mapping[str, Any]:
            tracker["update_attempts_this_run"] += 1
            step_count = int(tracker["steps_since_update"])
            transitions = step_count * args.num_envs
            pad_steps = int(tracker["pad_terms_steps"])
            pad_means = {
                key: value / pad_steps for key, value in tracker["pad_terms_sum"].items()
            } if pad_steps else {}
            try:
                metrics = original_update()
            except BaseException as exc:
                update_writer.write(
                    {
                        "update_status": "FAILED",
                        "update_attempt": tracker["update_attempts_this_run"],
                        "completed_updates_before": tracker["completed_updates"],
                        "rollout_policy_steps": step_count,
                        "interaction_transitions": transitions,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                )
                tracker["steps_since_update"] = 0
                tracker["pad_terms_sum"] = {}
                tracker["pad_terms_steps"] = 0
                tracker["pad_terms_missing_steps"] = 0
                raise

            anchor_row = None
            if prior_actor is not None:
                with torch.no_grad():
                    storage_observations = runner.alg.storage.observations
                    probe = storage_observations.reshape(-1)
                    current_mean = runner.alg.get_policy()(probe, stochastic_output=False)
                    prior_mean = prior_actor(probe, stochastic_output=False)
                    mean_abs_shift = float((current_mean - prior_mean).abs().mean())
                limit = float(args.actor_mean_shift_max)
                alpha = 1.0
                projected = mean_abs_shift > limit
                if projected:
                    alpha = limit / max(mean_abs_shift, 1e-12)
                    with torch.no_grad():
                        for parameter, anchor in zip(
                                runner.alg.get_policy().parameters(),
                                prior_actor.parameters()):
                            parameter.lerp_(anchor, 1.0 - alpha)
                    actor_constraint["projection_count"] += 1
                actor_constraint["min_mean_abs_shift"] = (
                    mean_abs_shift if actor_constraint["min_mean_abs_shift"] is None
                    else min(actor_constraint["min_mean_abs_shift"], mean_abs_shift))
                actor_constraint["max_mean_abs_shift"] = (
                    mean_abs_shift if actor_constraint["max_mean_abs_shift"] is None
                    else max(actor_constraint["max_mean_abs_shift"], mean_abs_shift))
                anchor_row = {
                    "mean_abs_shift": mean_abs_shift,
                    "limit": limit,
                    "alpha": alpha,
                    "projected_to_prior": projected,
                }

            tracker["completed_updates"] += 1
            tracker["updates_this_run"] += 1
            tracker["partial_rollout_steps"] = 0
            flat_metrics, nonfinite_metrics = _flatten_metrics(metrics)
            rollout_valid = step_count == ROLLOUT_STEPS
            row = {
                "update_status": "COMPLETED",
                "completed_updates": tracker["completed_updates"],
                "updates_this_run": tracker["updates_this_run"],
                "runner_current_learning_iteration_before_runner_increment": int(runner.current_learning_iteration),
                "runner_iteration_semantics": "RSL-RL 当前值是 0-based iteration label，不是 update count；计数以本 wrapper 完成调用累计",
                "rollout_policy_steps": step_count,
                "rollout_contract_valid": rollout_valid,
                "interaction_transitions_this_update": transitions,
                "interaction_transitions_total": tracker["interaction_transitions"],
                "expected_transitions_per_update": args.num_envs * ROLLOUT_STEPS,
                "ppo_metrics": flat_metrics,
                "nonfinite_metric_names": nonfinite_metrics,
                "ppo_metrics_finite": not nonfinite_metrics and bool(flat_metrics),
                "pad_reward_terms_mean": pad_means,
                "pad_reward_terms_steps": pad_steps,
                "pad_reward_terms_missing_steps": tracker["pad_terms_missing_steps"],
                "actor_constraint": anchor_row,
            }
            update_writer.write(row)
            reward_terms_writer.write(
                {
                    "completed_updates": tracker["completed_updates"],
                    "steps_observed": pad_steps,
                    "missing_steps": tracker["pad_terms_missing_steps"],
                    "mean_by_term": pad_means,
                }
            )
            print("UPDATE " + json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            tracker["steps_since_update"] = 0
            tracker["pad_terms_sum"] = {}
            tracker["pad_terms_steps"] = 0
            tracker["pad_terms_missing_steps"] = 0
            if not rollout_valid:
                raise RuntimeError(f"PPO 更新采集到 {step_count} 个策略步，预期 {ROLLOUT_STEPS}")
            if nonfinite_metrics or not flat_metrics:
                raise RuntimeError("PPO update 返回空指标或非有限指标")
            if tracker["completed_updates"] == 1 or tracker["completed_updates"] % CHECKPOINT_INTERVAL == 0:
                info = _checkpoint_info(
                    identity=identity,
                    identity_sha256=identity_sha256,
                    critical_source_sha256=critical_source_sha256,
                    source_files=source_files,
                    observation_dimensions=observation_dimensions,
                    completed_updates=tracker["completed_updates"],
                    interaction_transitions=tracker["interaction_transitions"],
                    observation_shapes=observation_shapes,
                    agent_config_sha256=run_metadata["agent_config_sha256"],
                    initialization=initialization,
                )
                filename = f"update_{tracker['completed_updates']:06d}.pt"
                checkpoint_rows.append({"kind": "periodic", **_save_checkpoint(runner, output, filename, info)})
            return metrics

        runner.alg.update = tracked_update
        phase["value"] = "training_loop"
        while tracker["updates_this_run"] < args.iterations:
            chunk = min(CHECKPOINT_INTERVAL, args.iterations - tracker["updates_this_run"])
            runner.current_learning_iteration = tracker["completed_updates"]
            updates_before = tracker["updates_this_run"]
            try:
                runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=False)
            except _StopRun as stop:
                controlled_stop = stop.reason
                break
            except BaseException as exc:
                error = exc
                break
            if tracker["updates_this_run"] - updates_before != chunk:
                error = RuntimeError("runner.learn 返回的 update wrapper 计数与请求次数不一致")
                break

        if tracker["steps_since_update"]:
            tracker["partial_rollout_steps"] = tracker["steps_since_update"]
        final_actor_trainable_sha256 = _trainable_sha256(torch, runner.alg.get_policy())
        final_critic_trainable_sha256 = _trainable_sha256(torch, runner.alg.critic)
        final_actor_sha256 = _state_sha256(torch, runner.alg.get_policy())
        final_critic_sha256 = _state_sha256(torch, runner.alg.critic)
        phase["value"] = "final_checkpoint"
        final_info = _checkpoint_info(
            identity=identity,
            identity_sha256=identity_sha256,
            critical_source_sha256=critical_source_sha256,
            source_files=source_files,
            observation_dimensions=observation_dimensions,
            completed_updates=tracker["completed_updates"],
            interaction_transitions=tracker["interaction_transitions"],
            observation_shapes=observation_shapes,
            agent_config_sha256=run_metadata["agent_config_sha256"],
            initialization=initialization,
        )
        checkpoint_rows.append({"kind": "final", **_save_checkpoint(runner, output, "final.pt", final_info)})
        if error is not None or controlled_stop and controlled_stop.startswith("termination_signal_"):
            failure = error or RuntimeError(controlled_stop)
            _write_failure(
                output,
                failure,
                phase["value"],
                {
                    "completed_updates": tracker["completed_updates"],
                    "updates_this_run": tracker["updates_this_run"],
                    "interaction_transitions_this_run": transitions_this_run,
                    "final_checkpoint": str(output / "final.pt"),
                },
            )
            return 1

        actor_changed = initial_actor_trainable_sha256 != final_actor_trainable_sha256
        critic_changed = initial_critic_trainable_sha256 != final_critic_trainable_sha256
        summary = {
            "status": "TRAINING_COMPLETE" if controlled_stop is None else "TRAINING_STOPPED_AT_BOUNDARY",
            "learning_result": (
                "WEIGHTS_UPDATED_NO_GRASP_SUCCESS_CLAIM"
                if tracker["updates_this_run"] > 0 and (actor_changed or critic_changed)
                else "NO_PARAMETER_CHANGE_NO_GRASP_SUCCESS_CLAIM"
            ),
            "success_claim": False,
            "task_id": TASK_ID,
            "task_identity_sha256": identity_sha256,
            "critical_source_sha256": critical_source_sha256,
            "observation_dimensions": observation_dimensions,
            "updates_requested_this_run": args.iterations,
            "updates_completed_this_run": tracker["updates_this_run"],
            "updates_completed_total": tracker["completed_updates"],
            "update_attempts_this_run": tracker["update_attempts_this_run"],
            "interaction_transitions_this_run": transitions_this_run,
            "interaction_transitions_total": tracker["interaction_transitions"],
            "transitions_per_full_update": args.num_envs * ROLLOUT_STEPS,
            "partial_rollout_policy_steps_at_stop": tracker["partial_rollout_steps"],
            "stop_reason": controlled_stop or "iterations_completed",
            "policy_initialization": initialization,
            "actor_constraint": actor_constraint,
            "f3_f4_condition": run_metadata["f3_f4_condition"],
            "parameter_hashes": {
                "hash_scope": "trainable parameters for changed flags; full state hashes also retained",
                "actor_trainable_before": initial_actor_trainable_sha256,
                "actor_trainable_after": final_actor_trainable_sha256,
                "critic_trainable_before": initial_critic_trainable_sha256,
                "critic_trainable_after": final_critic_trainable_sha256,
                "actor_before": initial_actor_sha256,
                "actor_after": final_actor_sha256,
                "actor_changed": actor_changed,
                "critic_before": initial_critic_sha256,
                "critic_after": final_critic_sha256,
                "critic_changed": critic_changed,
            },
            "episode_records": len(episode_rows),
            "checkpoints": checkpoint_rows,
            "initial_checkpoint": str(output / "initial.pt"),
            "final_checkpoint": str(output / "final.pt"),
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(output / "summary.json", summary)
        return 0
    finally:
        if trace_writer is not None:
            try:
                if callable(write_trace_callback):
                    write_trace_callback(vector_steps)
            finally:
                trace_writer.close()
        if reward_terms_writer is not None:
            reward_terms_writer.close()
        if update_writer is not None:
            update_writer.close()
        episode_writer.close()
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


def _read_episode_delta(env: Any, cursor: int) -> tuple[list[dict[str, Any]], int]:
    rows = getattr(env, "completed_episodes", None)
    if not isinstance(rows, list):
        raise TypeError("completed_episodes 必须是 append-only list")
    if len(rows) < cursor:
        raise RuntimeError("completed_episodes 被清空或截断")
    result: list[dict[str, Any]] = []
    for row in rows[cursor:]:
        if not isinstance(row, Mapping):
            raise TypeError("completed_episodes 每行必须是 mapping")
        result.append(dict(row))
    return result, len(rows)


def _flatten_metrics(metrics: Any) -> tuple[dict[str, Any], list[str]]:
    flat: dict[str, Any] = {}
    nonfinite: list[str] = []

    def visit(value: Any, name: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{name}.{key}" if name else str(key))
            return
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                visit(float(tensor.item()), name)
            else:
                visit(tensor.tolist(), name)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{name}[{index}]")
            return
        if isinstance(value, bool):
            flat[name] = value
            return
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                flat[name] = None
                nonfinite.append(name)
            else:
                flat[name] = number
            return
        if value is None:
            flat[name] = None
            return
        flat[name] = str(value)

    if isinstance(metrics, Mapping):
        visit(metrics, "")
    return flat, nonfinite


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        output = _prepare_output(args.output)
    except Exception as exc:
        parser.error(str(exc))
    phase = {"value": "app_launcher_start"}
    app = None
    exit_code = 1
    stop_request: dict[str, int | None] = {"signal": None}
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(args).app

        def request_stop(signum: int, _frame: Any) -> None:
            stop_request["signal"] = int(signum)

        signal.signal(signal.SIGTERM, request_stop)
        exit_code = _run(args, output, stop_request, phase)
    except BaseException as exc:
        _write_failure(output, exc, phase["value"], {"command": list(sys.argv)})
        exit_code = 1
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as exc:
                _write_failure(output, exc, "app_close", {"summary_exists": (output / "summary.json").is_file()})
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
