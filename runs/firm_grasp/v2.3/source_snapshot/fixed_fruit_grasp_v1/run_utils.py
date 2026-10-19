"""固定果实 PPO 入口共用的轻量记录与校验工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONTRACT_ID = "fixed_fruit_grasp_v1"
STAGES = ("contact", "hold", "lift")
# v1.6.0 把 PPO 的两个优化旋钮改得更保守：固定学习率 3.0e-4 -> 1.0e-4，
# 策略初始方差 0.20 -> 0.08。它们是唯一允许在已审查迁移里变化的 agent 字段；
# 其余 PPO 超参（含 entropy_coef=0.003）、奖励权重、真实物理、动作过滤与触发
# 时序都必须逐字节保持，见 FIXED_FRUIT_ACTOR_TRANSFER_TRANSITIONS。
FIXED_FRUIT_PPO_LEARNING_RATE = 1.0e-4
FIXED_FRUIT_INITIAL_STD = 0.08
FIXED_FRUIT_ACTION_DIM = 14
MAX_EVALUATION_EPISODES = 20
# ------------------------------------------------- 显式 actor-only 迁移声明
# 允许的版本迁移对与白名单字段只在这里单点声明，校验函数只引用它们，避免
# “顺手再放宽一点”的隐式扩散：除下列路径外任何未知/新增字段变化一律拒绝。
FIXED_FRUIT_ACTOR_TRANSFER_TRANSITIONS = frozenset(
    {
        ("fixed_fruit_grasp_v1.3.0", "fixed_fruit_grasp_v1.4.0"),
        ("fixed_fruit_grasp_v1.4.0", "fixed_fruit_grasp_v1.5.0"),
        ("fixed_fruit_grasp_v1.5.0", "fixed_fruit_grasp_v1.6.0"),
    }
)
CONSERVATIVE_PPO_TRANSITION = ("fixed_fruit_grasp_v1.5.0", "fixed_fruit_grasp_v1.6.0")
CONSERVATIVE_PPO_SOURCE_LEARNING_RATE = 3.0e-4
CONSERVATIVE_PPO_SOURCE_INITIAL_STD = 0.20
AGENT_CONFIG_LEARNING_RATE_PATH = ("algorithm", "learning_rate")
AGENT_CONFIG_INITIAL_STD_PATH = ("actor", "distribution_cfg", "init_std")
CONSERVATIVE_PPO_AGENT_CONFIG_PATHS = frozenset(
    {AGENT_CONFIG_LEARNING_RATE_PATH, AGENT_CONFIG_INITIAL_STD_PATH}
)
CONSERVATIVE_PPO_ARCHITECTURE_PATHS = frozenset({AGENT_CONFIG_INITIAL_STD_PATH})
# RSL-RL 5.0.1 的实际探索方差参数：actor（MLPModel）内的 GaussianDistribution
# 在 std_type="log" 下持有 log_std_param，actor_state_dict 里的键名就是
# ``distribution.log_std_param``（本机 CPU 实物已确认：14 维 float32，与 4 个
# obs_normalizer 缓冲 + 8 个 mlp 均值权重合计 13 项）。
ACTOR_LOG_STD_STATE_KEY = "distribution.log_std_param"
ACTOR_EXPLORATION_STD_RESET_MARKER = "_fixed_fruit_exploration_std_reset_applied"
EPISODE_FIELDS = (
    "env_id",
    "episode_id",
    "success",
    "contact_fraction",
    "required_contact_fraction",
    "max_unsupported_hold_s",
    "max_height_gain_m",
    "reason",
)


def parse_initial_action_bias_json(value: str | None, *, resume: bool = False) -> list[float] | None:
    """解析 fresh-run 的 14D 动作先验；resume 时拒绝覆盖检查点策略。"""

    if value is None:
        return None
    if resume:
        raise ValueError("--initial-action-bias-json can only be used for a fresh run")
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("--initial-action-bias-json must be a JSON array") from exc
    if not isinstance(parsed, list) or len(parsed) != FIXED_FRUIT_ACTION_DIM:
        raise ValueError(
            f"--initial-action-bias-json must contain exactly {FIXED_FRUIT_ACTION_DIM} values"
        )
    values: list[float] = []
    for index, item in enumerate(parsed):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"initial action bias value {index} must be a finite number")
        try:
            number = float(item)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"initial action bias value {index} must be finite") from exc
        if not math.isfinite(number):
            raise ValueError(f"initial action bias value {index} must be finite")
        if not -1.0 <= number <= 1.0:
            raise ValueError(f"initial action bias value {index} must be in [-1, 1]")
        values.append(number)
    return values


def _tensor_sha256(value: Any, torch_module: Any) -> str:
    """按连续 CPU tensor 的原始字节计算 SHA-256。"""

    raw = value.detach().cpu().contiguous().view(torch_module.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def initialize_actor_output_bias(torch_module: Any, policy: Any, bias: Sequence[float]) -> dict[str, Any]:
    """把唯一 actor MLP 末层置为常量输出，并返回可审计的初始化记录。

    支持 RSL-RL ActorCritic 的 ``actor`` 子模块和 get_policy() 直接返回
    ``MLPModel`` 的结构。遇到含糊结构、缺失 bias 或错误输出维度时拒绝修改。
    """

    values = parse_initial_action_bias_json(json.dumps(list(bias), allow_nan=True))
    if values is None:  # pragma: no cover - json.dumps(list(...)) 总是提供字符串
        raise ValueError("initial action bias is required")

    named_modules = list(policy.named_modules())
    actor_roots = [
        (name, module)
        for name, module in named_modules
        if name and any(part == "actor" or part.startswith("actor_") for part in name.split("."))
        and name.split(".")[-1].startswith("actor")
    ]
    if not actor_roots and type(policy).__name__ == "MLPModel":
        actor_roots = [("", policy)]
    if len(actor_roots) != 1:
        raise ValueError(
            "could not identify a unique actor MLP root in policy; refusing action-bias initialization"
        )

    root_name, actor_root = actor_roots[0]
    linear_layers = [
        (name, module)
        for name, module in actor_root.named_modules()
        if isinstance(module, torch_module.nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("actor MLP has no nn.Linear output layer")
    relative_name, output_layer = linear_layers[-1]
    if int(output_layer.out_features) != FIXED_FRUIT_ACTION_DIM:
        raise ValueError(
            "actor MLP output layer must have 14 outputs; "
            f"found {output_layer.out_features} at {root_name}.{relative_name}"
        )
    if output_layer.bias is None:
        raise ValueError("actor MLP output layer has no bias parameter")
    if output_layer.weight.ndim != 2 or tuple(output_layer.weight.shape)[0] != FIXED_FRUIT_ACTION_DIM:
        raise ValueError("actor MLP output weight does not have 14 output rows")

    layer_name = ".".join(part for part in (root_name, relative_name) if part)
    weight_sha256_before = _tensor_sha256(output_layer.weight, torch_module)
    bias_sha256_before = _tensor_sha256(output_layer.bias, torch_module)
    bias_tensor = torch_module.as_tensor(
        values, device=output_layer.bias.device, dtype=output_layer.bias.dtype
    )
    with torch_module.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(bias_tensor)
    return {
        "kind": "zero_output_weight_declared_action_bias_prior",
        "classification": "declared_initialization_prior_not_learned_success",
        "bias": values,
        "changed": True,
        "actor_output_layer": layer_name,
        "weight_sha256_before": weight_sha256_before,
        "weight_sha256_after": _tensor_sha256(output_layer.weight, torch_module),
        "bias_sha256_before": bias_sha256_before,
        "bias_sha256_after": _tensor_sha256(output_layer.bias, torch_module),
    }


def configure_fixed_fruit_agent_cfg(agent_cfg: Any) -> Any:
    """为训练和独立评估应用同一固定 PPO schedule 与策略初始方差。"""

    algorithm = getattr(agent_cfg, "algorithm", None)
    actor = getattr(agent_cfg, "actor", None)
    distribution = getattr(actor, "distribution_cfg", None)
    if algorithm is None or distribution is None:
        raise TypeError("agent config must expose algorithm and actor.distribution_cfg")
    algorithm.learning_rate = FIXED_FRUIT_PPO_LEARNING_RATE
    algorithm.schedule = "fixed"
    distribution.init_std = FIXED_FRUIT_INITIAL_STD
    return agent_cfg


def fixed_fruit_agent_settings(agent_cfg: Any, num_actions: int) -> dict[str, Any]:
    """返回写入训练和评估证据的关键实际策略配置。"""

    algorithm = getattr(agent_cfg, "algorithm", None)
    actor = getattr(agent_cfg, "actor", None)
    distribution = getattr(actor, "distribution_cfg", None)
    if algorithm is None or distribution is None:
        raise TypeError("agent config must expose algorithm and actor.distribution_cfg")
    settings = {
        "num_actions": int(num_actions),
        "learning_rate": float(algorithm.learning_rate),
        "schedule": str(algorithm.schedule),
        "initial_std": float(distribution.init_std),
    }
    if settings["num_actions"] != 14:
        raise ValueError(f"fixed-fruit policy requires 14 actions; got {settings['num_actions']}")
    if settings["learning_rate"] != FIXED_FRUIT_PPO_LEARNING_RATE:
        raise ValueError("fixed-fruit PPO learning rate is not the required fixed value")
    if settings["schedule"] != "fixed":
        raise ValueError("fixed-fruit PPO schedule must be fixed")
    if settings["initial_std"] != FIXED_FRUIT_INITIAL_STD:
        raise ValueError("fixed-fruit policy initial std does not match the declared value")
    return settings


def _state_tensor_hashes(torch_module: Any, state: Mapping[str, Any]) -> dict[str, str]:
    """逐 tensor 计算原始字节 SHA-256；非 tensor 项退化为 JSON 值哈希。"""

    hashes: dict[str, str] = {}
    for name, value in state.items():
        if hasattr(value, "detach"):
            hashes[str(name)] = _tensor_sha256(value.reshape(-1), torch_module)
        else:
            hashes[str(name)] = sha256_json(json_safe(value))
    return hashes


def _resolve_attribute_path(policy: Any, name: str) -> tuple[Any, Any]:
    """按 ``a.b.c`` 路径解析对象属性，返回 (宿主模块, 属性值)。"""

    module: Any = policy
    parts = str(name).split(".")
    for part in parts[:-1]:
        if not hasattr(module, part):
            raise ValueError(f"actor exploration std parameter path is incomplete: {name}")
        module = getattr(module, part)
    if not hasattr(module, parts[-1]):
        raise ValueError(f"actor exploration std parameter path is incomplete: {name}")
    return module, getattr(module, parts[-1])


def reset_actor_exploration_std(
    torch_module: Any,
    policy: Any,
    *,
    target_std: float = FIXED_FRUIT_INITIAL_STD,
    transferred_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把已迁移 actor 的探索方差一次性显式重置为 ``log(target_std)``。

    只用于已审查的 v1.5.0 -> v1.6.0 迁移：被迁移 actor 的均值权重与 obs
    归一化缓冲必须逐字节保留，只允许唯一 ``distribution.log_std_param`` 被
    改写。唯一性、``std_type`` 与“只应用一次”都是 fail-closed 的；传入
    ``transferred_state`` 时还会逐键核验迁移字节确实未被带动，并确认
    log_std 自身确实发生了变化（否则只是静默无效重置）。

    该重置是显式探索重初始化，不是新语义下的学习成功证据，也不改变
    critic/optimizer/iteration 的重启语义。
    """

    target = float(target_std)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("actor exploration std target must be a positive finite number")
    if bool(getattr(policy, ACTOR_EXPLORATION_STD_RESET_MARKER, False)):
        raise ValueError("actor exploration std reset can be applied only once per policy")
    matches = [
        str(name)
        for name, _ in policy.named_parameters()
        if str(name) == ACTOR_LOG_STD_STATE_KEY
        or str(name).endswith("." + ACTOR_LOG_STD_STATE_KEY)
    ]
    if len(matches) != 1:
        raise ValueError(
            "actor exploration std reset requires exactly one "
            f"{ACTOR_LOG_STD_STATE_KEY} parameter; found {sorted(matches)}"
        )
    name = matches[0]
    owner, parameter = _resolve_attribute_path(policy, name)
    if getattr(owner, "std_type", None) != "log":
        raise ValueError("actor exploration std reset requires std_type 'log'")
    if not hasattr(parameter, "numel") or int(parameter.numel()) != FIXED_FRUIT_ACTION_DIM:
        raise ValueError("actor exploration std parameter must hold one log std per action")

    before_state = _state_tensor_hashes(torch_module, policy.state_dict())
    transferred: dict[str, str] | None = None
    if transferred_state is not None:
        transferred = _state_tensor_hashes(torch_module, dict(transferred_state))
        missing = sorted(set(transferred) - set(before_state))
        if missing:
            raise ValueError(
                "transferred actor state has keys absent from the policy: " + ", ".join(missing)
            )
    log_std_before = [float(value) for value in parameter.detach().cpu().reshape(-1).tolist()]
    with torch_module.no_grad():
        parameter.copy_(torch_module.full_like(parameter, float(math.log(target))))
    after_state = _state_tensor_hashes(torch_module, policy.state_dict())
    preserved_before = {key: value for key, value in before_state.items() if key != name}
    preserved_after = {key: value for key, value in after_state.items() if key != name}
    if not preserved_before or preserved_before != preserved_after:
        raise ValueError("actor exploration std reset changed parameters other than log_std")
    if transferred is not None:
        moved = sorted(
            key
            for key, digest in transferred.items()
            if key != name and preserved_after.get(key) != digest
        )
        if moved:
            raise ValueError(
                "transferred actor parameters were not preserved byte-for-byte: " + ", ".join(moved)
            )
        if transferred.get(name) == after_state[name]:
            raise ValueError("actor exploration std reset did not change the migrated log_std")
    log_std_after = [float(value) for value in parameter.detach().cpu().reshape(-1).tolist()]
    setattr(policy, ACTOR_EXPLORATION_STD_RESET_MARKER, True)
    return {
        "kind": "explicit_exploration_std_reinitialization",
        "classification": "declared_exploration_reinitialization_not_learned_success",
        "applied": True,
        "applied_once": True,
        "parameter_name": name,
        "std_type": str(getattr(owner, "std_type")),
        "target_std": target,
        "target_log_std": float(math.log(target)),
        "log_std_before": log_std_before,
        "log_std_after": log_std_after,
        "parameter_sha256_before": before_state[name],
        "parameter_sha256_after": after_state[name],
        "preserved_parameter_keys": sorted(preserved_after),
        "preserved_parameter_sha256": preserved_after,
        "preserved_parameters_byte_identical": bool(preserved_before == preserved_after),
        "mean_weights_and_obs_normalizer_preserved": bool(preserved_before == preserved_after),
        "transferred_state_keys_checked": None if transferred is None else len(transferred),
    }


_CONTRACT_RUNTIME_KEYS = {"stage", "curriculum_stage", "stage_name", "num_envs", "device", "seed"}
_STAGE_DEPENDENT_IDENTITY_PATHS = {
    ("physics", "support", "release_step"),
    ("physics", "support", "release_time_s"),
}
_PATH_IDENTITY_KEYS = {
    "asset_path",
    "asset_config_path",
    "usd_path",
    "urdf_path",
    "scene_file",
    "config_path",
    "source_path",
}
_HASHABLE_CONFIG_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".toml",
    ".urdf",
    ".usd",
    ".xml",
    ".yaml",
    ".yml",
}
_TRAINING_SEMANTIC_FILES = (
    "research/fixed_fruit_grasp_v1/train.py",
    "research/fixed_fruit_grasp_v1/run_utils.py",
    "research/fixed_fruit_grasp_v1/env.py",
    "research/fixed_fruit_grasp_v1/contract.py",
    "research/fixed_fruit_grasp_v1/physical_contacts.py",
    "research/fixed_fruit_grasp_v1/physical_hold.py",
    "research/fixed_fruit_grasp_v1/collision_cooking.py",
    "research/small_fruit_pinch_v1/source/agents.py",
)
_EVALUATION_SEMANTIC_FILES = (
    "research/fixed_fruit_grasp_v1/evaluate.py",
    "research/fixed_fruit_grasp_v1/run_utils.py",
    "research/fixed_fruit_grasp_v1/env.py",
    "research/fixed_fruit_grasp_v1/contract.py",
    "research/fixed_fruit_grasp_v1/physical_contacts.py",
    "research/fixed_fruit_grasp_v1/physical_hold.py",
    "research/fixed_fruit_grasp_v1/collision_cooking.py",
    "research/small_fruit_pinch_v1/source/agents.py",
)
_SOURCE_ASSET_COMPATIBILITY_KEYS = (
    "training_semantics_sha256",
    "evaluation_semantics_sha256",
    "asset_config_sha256",
    "asset_config_missing_reason",
    "asset_files_manifest_sha256",
    "referenced_asset_files",
    "missing_referenced_paths",
)


def json_safe(value: Any) -> Any:
    """把常见 Python、NumPy 和 Torch 值转换为严格 JSON 值。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if hasattr(value, "detach"):
        try:
            return json_safe(value.detach().cpu().tolist())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, Sequence):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return str(value)


def canonical_json(value: Any) -> str:
    """按稳定顺序序列化 JSON 数据，并拒绝 NaN 与无穷值。"""

    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _included_files(root: Path, suffixes: set[str] | None) -> list[Path]:
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in ignored for part in path.relative_to(root).parts):
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def sha256_path(path: str | Path, suffixes: set[str] | None = None) -> str:
    """计算单文件或目录内确定性文件清单的 SHA-256。"""

    path = Path(path).expanduser().resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    manifest = [
        {
            "relative_path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
        }
        for item in _included_files(path, suffixes)
    ]
    return sha256_json(manifest)


def normalize_contract_identity(identity: Any) -> Any:
    """移除运行规模和课程专属字段，仍严格保留任务与资产身份。"""

    def normalize(value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, Mapping):
            normalized = {}
            for key, item in value.items():
                key_text = str(key)
                lowered_path = (*path, key_text.lower())
                if key_text.lower() in _CONTRACT_RUNTIME_KEYS:
                    continue
                if lowered_path in _STAGE_DEPENDENT_IDENTITY_PATHS:
                    continue
                normalized[key_text] = normalize(item, lowered_path)
            return normalized
        if isinstance(value, (list, tuple)):
            return [normalize(item, path) for item in value]
        return json_safe(value)

    return normalize(identity, ())


def contract_sha256(identity: Any) -> str:
    return sha256_json(normalize_contract_identity(identity))


def comparable_agent_config(config: Any) -> Any:
    """保留完整 agent 配置，同时允许 checkpoint 在设备间迁移。"""

    if not isinstance(config, Mapping):
        return json_safe(config)
    comparable = dict(config)
    comparable.pop("device", None)
    return json_safe(comparable)


def agent_config_sha256(config: Any) -> str:
    return sha256_json(comparable_agent_config(config))


def resolve_checkpoint_pose(
    identity: Mapping[str, Any],
    requested_pregrasp_distance_m: float | None,
    requested_fruit_center_local_m: Sequence[float] | None,
) -> tuple[float, tuple[float, float, float]]:
    """从冻结任务身份恢复预抓取位姿，并校验显式覆盖值是否一致。"""

    initial_pose = identity.get("initial_pose")
    if not isinstance(initial_pose, Mapping):
        raise ValueError("checkpoint task identity has no initial_pose mapping")
    try:
        frozen_distance = float(initial_pose["pregrasp_distance_m"])
        frozen_center_values = initial_pose["closed_grasp_fruit_center_m"]
        frozen_center = tuple(float(value) for value in frozen_center_values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint task identity has incomplete initial-pose values") from exc
    if len(frozen_center) != 3 or not math.isfinite(frozen_distance):
        raise ValueError("checkpoint task identity has invalid initial-pose values")
    if not all(math.isfinite(value) for value in frozen_center):
        raise ValueError("checkpoint task identity has non-finite fruit-center values")

    if requested_pregrasp_distance_m is not None:
        requested_distance = float(requested_pregrasp_distance_m)
        if not math.isfinite(requested_distance) or not math.isclose(
            requested_distance, frozen_distance, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "explicit pregrasp distance does not match the checkpoint task identity"
            )
    if requested_fruit_center_local_m is not None:
        requested_center = tuple(float(value) for value in requested_fruit_center_local_m)
        if len(requested_center) != 3 or not all(math.isfinite(value) for value in requested_center):
            raise ValueError("explicit fruit center must contain three finite values")
        if any(
            not math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-9)
            for value, frozen in zip(requested_center, frozen_center)
        ):
            raise ValueError("explicit fruit center does not match the checkpoint task identity")
    return frozen_distance, frozen_center


def _walk_identity_paths(value: Any, key: str = ""):
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _walk_identity_paths(child, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_identity_paths(child, key)
    elif key.lower() in _PATH_IDENTITY_KEYS and isinstance(value, (str, os.PathLike)):
        yield Path(value).expanduser()


def hash_run_inputs(package_dir: str | Path, identity: Any) -> dict[str, Any]:
    """计算入口源码、task_identity 引用的资产配置及任务身份哈希。"""

    package_dir = Path(package_dir).resolve()
    source_sha = sha256_path(
        package_dir,
        suffixes={".py", ".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"},
    )
    repository_root = package_dir.parent.parent
    semantic_file_hashes: dict[str, str] = {}
    for relative_path in sorted(set(_TRAINING_SEMANTIC_FILES + _EVALUATION_SEMANTIC_FILES)):
        source_path = repository_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"semantic source required for checkpoint compatibility is missing: {source_path}"
            )
        semantic_file_hashes[relative_path] = sha256_file(source_path)
    training_files = {
        relative_path: semantic_file_hashes[relative_path]
        for relative_path in _TRAINING_SEMANTIC_FILES
    }
    evaluation_files = {
        relative_path: semantic_file_hashes[relative_path]
        for relative_path in _EVALUATION_SEMANTIC_FILES
    }
    asset_files: dict[str, str] = {}
    missing_paths: list[str] = []
    for candidate in sorted({item for item in _walk_identity_paths(identity)}, key=str):
        resolved = candidate.resolve()
        if resolved.is_file():
            asset_files[str(resolved)] = sha256_file(resolved)
        elif resolved.is_dir():
            try:
                asset_files[str(resolved)] = sha256_path(
                    resolved, suffixes=_HASHABLE_CONFIG_SUFFIXES
                )
            except FileNotFoundError:
                missing_paths.append(str(resolved))
        else:
            missing_paths.append(str(resolved))

    identity_json = json_safe(identity)
    asset_config = None
    if isinstance(identity_json, Mapping):
        asset_config = identity_json.get("asset_config")
        if asset_config is None:
            asset_config = identity_json.get("asset_configuration")
    result = {
        "package_source_sha256": source_sha,
        "training_semantics_source_files": training_files,
        "training_semantics_sha256": sha256_json(training_files),
        "evaluation_semantics_source_files": evaluation_files,
        "evaluation_semantics_sha256": sha256_json(evaluation_files),
        "task_identity_sha256": sha256_json(identity_json),
        "asset_config_sha256": sha256_json(asset_config) if asset_config is not None else None,
        "asset_config_missing_reason": (
            None if asset_config is not None else "task_identity has no asset_config field"
        ),
        "referenced_asset_files": asset_files,
        "missing_referenced_paths": missing_paths,
    }
    result["asset_files_manifest_sha256"] = sha256_json(asset_files)
    return result


def source_asset_compatibility_hashes(hashes: Mapping[str, Any]) -> dict[str, Any]:
    """选取语义源码与资产哈希，忽略 smoke、测试等目录总哈希噪声。"""

    missing = [key for key in _SOURCE_ASSET_COMPATIBILITY_KEYS if key not in hashes]
    if missing:
        raise ValueError(
            "source/asset hashes are incomplete for compatibility checking: "
            + ", ".join(missing)
        )
    return {key: json_safe(hashes[key]) for key in _SOURCE_ASSET_COMPATIBILITY_KEYS}


def atomic_write_json(path: str | Path, value: Any) -> None:
    """原子写入并同步 JSON，供可能快速退出的仿真进程安全留证。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: str | Path, value: Any) -> None:
    """追加一条完整 JSONL 记录并立即刷新缓冲区。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()


def fsync_file(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: str | Path) -> None:
    try:
        descriptor = os.open(Path(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_evidence(paths: Sequence[str | Path]) -> None:
    """同步已完成的证据文件及其父目录。"""

    parents = set()
    for path in paths:
        candidate = Path(path)
        if candidate.exists() and candidate.is_file():
            fsync_file(candidate)
            parents.add(candidate.parent)
    for parent in parents:
        fsync_directory(parent)


def checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    infos = payload.get("infos")
    if not isinstance(infos, Mapping):
        raise ValueError("checkpoint has no RSL-RL infos metadata")
    metadata = infos.get(CONTRACT_ID)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"checkpoint is missing {CONTRACT_ID} metadata")
    return dict(metadata)


def validate_checkpoint_compatibility(
    metadata: Mapping[str, Any],
    identity: Any,
    stage: str,
    source_and_asset_hashes: Mapping[str, Any] | None = None,
) -> str:
    """核对任务、源码和资产后返回完整续训或仅迁移 actor 的阶段切换模式。"""

    if metadata.get("contract_id") != CONTRACT_ID:
        raise ValueError("checkpoint belongs to a different task contract")
    if metadata.get("contract_sha256") != contract_sha256(identity):
        raise ValueError("checkpoint task identity does not match the current environment")
    if source_and_asset_hashes is not None:
        stored_hashes = metadata.get("source_and_asset_hashes")
        if not isinstance(stored_hashes, Mapping):
            raise ValueError("checkpoint has no source/asset hashes for compatibility checking")
        stored_compatibility = source_asset_compatibility_hashes(stored_hashes)
        current_compatibility = source_asset_compatibility_hashes(source_and_asset_hashes)
        changed = [
            key
            for key in _SOURCE_ASSET_COMPATIBILITY_KEYS
            if canonical_json(stored_compatibility[key])
            != canonical_json(current_compatibility[key])
        ]
        if changed:
            raise ValueError(
                "checkpoint source/asset compatibility mismatch: " + ", ".join(changed)
            )
    checkpoint_stage = metadata.get("stage")
    if checkpoint_stage not in STAGES:
        raise ValueError("checkpoint has no valid fixed-fruit training stage")
    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    return "full_resume" if checkpoint_stage == stage else "actor_only_stage_transition"



def _leaf_paths(source: Any, candidate: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """逐叶子比较两个 JSON 结构并返回改动路径；键集合不同即视为未知字段并拒绝。"""

    source = json_safe(source)
    candidate = json_safe(candidate)
    if isinstance(source, Mapping) and isinstance(candidate, Mapping):
        if set(source) != set(candidate):
            missing = sorted(set(source) - set(candidate))
            unexpected = sorted(set(candidate) - set(source))
            raise ValueError(
                "agent configuration changed unknown fields: "
                f"missing={missing} unexpected={unexpected}"
            )
        differences: set[tuple[str, ...]] = set()
        for key in sorted(source):
            differences |= _leaf_paths(source[key], candidate[key], (*path, str(key)))
        return differences
    if canonical_json(source) != canonical_json(candidate):
        return {path}
    return set()


def _path_value(mapping: Any, path: Sequence[str]) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def validate_conservative_ppo_transfer(
    metadata: Mapping[str, Any],
    architecture_sha256: str,
    config_sha256: str,
    actual_agent_config: Mapping[str, Any] | None,
    actual_architecture: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """仅允许已声明的 3.0e-4/0.20 -> 1.0e-4/0.08 两个 agent 字段变化。

    源 checkpoint 内的 ``agent_config``/``agent_architecture`` 必须与其自身
    SHA 自洽（不允许靠改哈希绕过配置检查），目标实际映射必须与调用方给出的
    SHA 绑定，且逐叶子差异必须恰好等于白名单；架构除
    ``actor.distribution_cfg.init_std`` 外必须完全一致。
    """

    source_config = metadata.get("agent_config")
    source_architecture = metadata.get("agent_architecture")
    if not isinstance(source_config, Mapping) or not isinstance(source_architecture, Mapping):
        raise ValueError("conservative PPO transfer source has no agent config/architecture")
    if not isinstance(actual_agent_config, Mapping) or not isinstance(actual_architecture, Mapping):
        raise ValueError("conservative PPO transfer requires the actual agent config and architecture")
    if metadata.get("agent_config_sha256") != sha256_json(source_config):
        raise ValueError("conservative PPO transfer source agent config digest is inconsistent")
    if metadata.get("agent_config_comparison_sha256") != agent_config_sha256(source_config):
        raise ValueError(
            "conservative PPO transfer source agent config comparison digest is inconsistent"
        )
    if metadata.get("agent_architecture_sha256") != sha256_json(source_architecture):
        raise ValueError("conservative PPO transfer source architecture digest is inconsistent")
    if agent_config_sha256(actual_agent_config) != config_sha256:
        raise ValueError("conservative PPO transfer actual agent config does not match the given digest")
    if sha256_json(actual_architecture) != architecture_sha256:
        raise ValueError("conservative PPO transfer actual architecture does not match the given digest")

    source_comparable = comparable_agent_config(source_config)
    destination_comparable = comparable_agent_config(actual_agent_config)
    config_paths = _leaf_paths(source_comparable, destination_comparable)
    if config_paths != set(CONSERVATIVE_PPO_AGENT_CONFIG_PATHS):
        raise ValueError(
            "conservative PPO transfer changed agent configuration outside the two declared fields: "
            + ", ".join(".".join(path) for path in sorted(config_paths))
        )
    architecture_paths = _leaf_paths(source_architecture, actual_architecture)
    if architecture_paths != set(CONSERVATIVE_PPO_ARCHITECTURE_PATHS):
        raise ValueError(
            "conservative PPO transfer changed architecture outside "
            "actor.distribution_cfg.init_std: "
            + ", ".join(".".join(path) for path in sorted(architecture_paths))
        )
    for label, mapping, learning_rate, initial_std in (
        (
            "source",
            source_comparable,
            CONSERVATIVE_PPO_SOURCE_LEARNING_RATE,
            CONSERVATIVE_PPO_SOURCE_INITIAL_STD,
        ),
        (
            "destination",
            destination_comparable,
            FIXED_FRUIT_PPO_LEARNING_RATE,
            FIXED_FRUIT_INITIAL_STD,
        ),
    ):
        actual_rate = _path_value(mapping, AGENT_CONFIG_LEARNING_RATE_PATH)
        actual_std = _path_value(mapping, AGENT_CONFIG_INITIAL_STD_PATH)
        if isinstance(actual_rate, bool) or not isinstance(actual_rate, (int, float)):
            raise ValueError(f"conservative PPO transfer {label} learning rate is not a number")
        if isinstance(actual_std, bool) or not isinstance(actual_std, (int, float)):
            raise ValueError(f"conservative PPO transfer {label} initial std is not a number")
        if float(actual_rate) != float(learning_rate):
            raise ValueError(
                f"conservative PPO transfer {label} learning rate is not the declared value"
            )
        if float(actual_std) != float(initial_std):
            raise ValueError(f"conservative PPO transfer {label} initial std is not the declared value")
    return {
        "changed_agent_config_paths": sorted(".".join(path) for path in config_paths),
        "changed_agent_architecture_paths": sorted(
            ".".join(path) for path in architecture_paths
        ),
        "source_learning_rate": float(CONSERVATIVE_PPO_SOURCE_LEARNING_RATE),
        "destination_learning_rate": float(FIXED_FRUIT_PPO_LEARNING_RATE),
        "source_initial_std": float(CONSERVATIVE_PPO_SOURCE_INITIAL_STD),
        "destination_initial_std": float(FIXED_FRUIT_INITIAL_STD),
        "source_agent_config_sha256": str(metadata.get("agent_config_sha256")),
        "source_agent_architecture_sha256": str(metadata.get("agent_architecture_sha256")),
        "destination_agent_config_comparison_sha256": str(config_sha256),
        "destination_agent_architecture_sha256": str(architecture_sha256),
    }


def validate_explicit_actor_transfer(
    metadata: Mapping[str, Any], identity: Mapping[str, Any], stage: str,
    hashes: Mapping[str, Any], architecture_sha256: str, config_sha256: str,
    actual_checkpoint_sha256: str, expected_checkpoint_sha256: str,
    actual_agent_config: Mapping[str, Any] | None = None,
    actual_architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """仅允许哈希绑定的已声明迁移；v1.5→v1.6 只改两个更保守的 PPO 字段。

    ``actual_agent_config``/``actual_architecture`` 只在新分支需要（当前仅
    v1.5.0→v1.6.0），用于逐字段核验“除白名单外禁止任何变化”；老迁移对仍
    只按既有 SHA 严格校验，正常 resume/evaluate 路径完全不走这里。
    """
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("actor transfer checkpoint SHA256 does not match the explicit pin")
    if metadata.get("contract_id") != CONTRACT_ID:
        raise ValueError("actor transfer belongs to a different contract")
    old = metadata.get("task_identity")
    if not isinstance(old, Mapping):
        raise ValueError("actor transfer has no source task identity")
    if metadata.get("task_identity_sha256") != sha256_json(old) or metadata.get("contract_sha256") != contract_sha256(old):
        raise ValueError("actor transfer source identity digest is inconsistent")
    transition = (old.get("task_version"), identity.get("task_version"))
    if transition not in FIXED_FRUIT_ACTOR_TRANSFER_TRANSITIONS:
        raise ValueError("actor transfer task-version transition has not been reviewed")
    reward_transfer = transition == ("fixed_fruit_grasp_v1.4.0", "fixed_fruit_grasp_v1.5.0")
    conservative_transfer = transition == CONSERVATIVE_PPO_TRANSITION
    if metadata.get("stage") != stage or stage != "contact":
        raise ValueError("actor transfer requires the same contact stage")
    conservative_validation: dict[str, Any] = {}
    if conservative_transfer:
        conservative_validation = validate_conservative_ppo_transfer(
            metadata,
            architecture_sha256,
            config_sha256,
            actual_agent_config,
            actual_architecture,
        )
    else:
        if metadata.get("agent_architecture_sha256") != architecture_sha256:
            raise ValueError("actor transfer architecture mismatch")
        if metadata.get("agent_config_comparison_sha256") != config_sha256:
            raise ValueError("actor transfer agent configuration mismatch")

    def portable(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {}
            for key, child in value.items():
                translated = portable(key)
                if translated in result:
                    raise ValueError("actor transfer has colliding normalized asset paths")
                result[translated] = portable(child)
            return result
        if isinstance(value, (list, tuple)):
            return [portable(x) for x in value]
        if isinstance(value, str) and value.startswith("/"):
            for root_marker in ("/source-v3/", "/source-v4/", "/source-v5/", "/source-v6/"):
                if root_marker in value:
                    return "repo/" + value.split(root_marker, 1)[1]
        return value

    def same(label: str, left: Any, right: Any) -> None:
        if left is None or right is None or canonical_json(portable(left)) != canonical_json(portable(right)):
            raise ValueError("actor transfer changed protected identity: " + label)

    if reward_transfer:
        # 奖励迁移比v3→v4更严格：整个物理身份只能做路径归一化，
        # 唯一准许变化的是奖励子树与显式版本；新字段也不能漏检。
        excluded = {"task_version", "reward"}
        same("reward_transfer_full_physical_identity",
             {k: v for k, v in old.items() if k not in excluded},
             {k: v for k, v in identity.items() if k not in excluded})
        for path in ("physical_hold.py", "physical_contacts.py", "collision_cooking.py"):
            relative = "research/fixed_fruit_grasp_v1/" + path
            same("unchanged_physical_source." + path,
                 metadata.get("source_and_asset_hashes", {}).get("training_semantics_source_files", {}).get(relative),
                 hashes.get("training_semantics_source_files", {}).get(relative))
    elif conservative_transfer:
        # 保守 PPO 迁移：整个任务身份除显式 task_version 外必须逐字节一致，
        # 奖励子树的权重与 profile 字符串也必须完全相同（不像 v1.4→v1.5
        # 那样允许奖励变化）；新增/缺失字段一律由 same() 的 None 检查拒绝。
        excluded = {"task_version"}
        same("conservative_ppo_full_task_identity",
             {k: v for k, v in old.items() if k not in excluded},
             {k: v for k, v in identity.items() if k not in excluded})
        for path in ("physical_hold.py", "physical_contacts.py", "collision_cooking.py"):
            relative = "research/fixed_fruit_grasp_v1/" + path
            same("unchanged_physical_source." + path,
                 metadata.get("source_and_asset_hashes", {}).get("training_semantics_source_files", {}).get(relative),
                 hashes.get("training_semantics_source_files", {}).get(relative))

    for key in ("actuation", "assets", "fruit", "initial_pose", "randomization", "run", "safety_guard"):
        same(key, old.get(key), identity.get(key))
    old_asset, new_asset = old.get("asset_config"), identity.get("asset_config")
    if not isinstance(old_asset, Mapping) or not isinstance(new_asset, Mapping):
        raise ValueError("actor transfer missing asset_config")
    for key in ("usd_path", "asset_config_path", "asset_files", "fruit", "actuation"):
        same("asset_config." + key, old_asset.get(key), new_asset.get(key))
    # asset_config同时包含派生的规则描述；允许120Hz规则变化，但不允许它
    # 与各自task_identity对应字段矛盾，也不跳过下面的物理参数比较。
    for key in ("physics", "contact", "criteria"):
        same("source_asset_config_consistency." + key, old_asset.get(key), old.get(key))
        same("destination_asset_config_consistency." + key, new_asset.get(key), identity.get(key))
    for key in ("sim_dt_s", "policy_dt_s", "decimation", "hand_collision_cooking", "support", "observation_dims"):
        same("physics." + key, old.get("physics", {}).get(key), identity.get("physics", {}).get(key))
    for key in ("hold_max_drift_m", "hold_max_relative_speed_m_s", "drop_height_tolerance_m", "drop_max_drift_m", "instability_max_fruit_speed_m_s", "instability_max_joint_velocity_rad_s", "joint_limit_margin_rad", "penetration_limit_m", "thumb_required"):
        same("criteria." + key, old.get("criteria", {}).get(key), identity.get("criteria", {}).get(key))
    for key in ("force_threshold_n", "hard_high_n", "min_contact_fingers", "real_self_collisions_enabled", "palm_fruit_collision_filter_enabled", "palm_only_counts_as_success", "declared_self_collision_pairs", "effective_self_collision_pairs", "self_collision_penetration_limit_m", "skipped_self_collision_pairs"):
        same("contact." + key, old.get("contact", {}).get(key), identity.get("contact", {}).get(key))
    for key in ("stage", "required_unsupported_hold_s", "support_release_step", "min_height_gain_m", "requires_lift_program"):
        same("stage." + key, old.get("stage", {}).get(key), identity.get("stage", {}).get(key))
    previous_hashes = metadata.get("source_and_asset_hashes")
    if not isinstance(previous_hashes, Mapping):
        raise ValueError("actor transfer missing source/asset hashes")
    if previous_hashes.get("missing_referenced_paths") or hashes.get("missing_referenced_paths"):
        raise ValueError("actor transfer has missing referenced assets")
    same("referenced_asset_files", previous_hashes.get("referenced_asset_files"), hashes.get("referenced_asset_files"))
    progress = metadata.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("actor transfer has no actual update provenance")
    updates = progress.get("total_updates_completed")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("actor transfer source update count must be a nonnegative integer")
    return {
        "mode": "actor_only_task_semantics_transfer", "source_task_version": old["task_version"],
        "destination_task_version": identity["task_version"], "source_checkpoint_sha256": actual_checkpoint_sha256,
        "source_total_updates_completed": updates, "new_semantics_updates_before": 0,
        "critic_optimizer_and_iteration_reset": True, "actor_normalizer_transferred": True,
        "is_new_semantics_success_evidence": False,
        "reward_profile_transfer_only": reward_transfer,
        "physical_criterion_unchanged": bool(reward_transfer or conservative_transfer),
        "exploration_std_reset_required": bool(conservative_transfer),
        "agent_config_validation": conservative_validation or None,
    }


def normalize_episode_record(row: Mapping[str, Any]) -> dict[str, Any]:
    result = json_safe(dict(row))
    raw_missing = result.get("missing_reason")
    missing = dict(raw_missing) if isinstance(raw_missing, Mapping) else {}
    for field in EPISODE_FIELDS:
        if field not in result or result[field] is None:
            result[field] = None
            missing.setdefault(field, "environment did not provide this episode field")
    if missing:
        result["missing_reason"] = missing
    return result


def completed_episode_delta(env: Any, cursor: int) -> tuple[list[dict[str, Any]], int]:
    rows = getattr(env, "completed_episodes", None)
    if not isinstance(rows, (list, tuple)):
        raise TypeError("environment.completed_episodes must be an append-only list")
    if len(rows) < cursor:
        raise RuntimeError("environment.completed_episodes was cleared or truncated")
    result = []
    for row in rows[cursor:]:
        if not isinstance(row, Mapping):
            raise TypeError("completed episode row must be a mapping")
        result.append(normalize_episode_record(row))
    return result, len(rows)


def score_evaluation_batch(
    rows: Sequence[Mapping[str, Any]], scored_count: int, target_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按批次到达顺序并以 env_id 排序，给目标内回合计分并保留超额行。"""

    if target_count <= 0 or target_count > MAX_EVALUATION_EPISODES:
        raise ValueError(
            f"evaluation target must be positive and at most {MAX_EVALUATION_EPISODES}"
        )
    if scored_count < 0 or scored_count > target_count:
        raise ValueError("scored episode count must be within the evaluation target")

    indexed_rows = list(enumerate(rows))

    def order_key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, int, int]:
        arrival_index, row = item
        env_id = row.get("env_id")
        if isinstance(env_id, bool):
            return (1, 0, arrival_index)
        try:
            numeric_env_id = int(env_id)
            if float(env_id) != numeric_env_id:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return (1, 0, arrival_index)
        return (0, numeric_env_id, arrival_index)

    ordered_rows = sorted(indexed_rows, key=order_key)
    remaining = target_count - scored_count
    scored: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    for _, raw_row in ordered_rows:
        row = normalize_episode_record(raw_row)
        if len(scored) < remaining:
            row["scored"] = True
            row["score_index"] = scored_count + len(scored) + 1
            scored.append(row)
        else:
            row["scored"] = False
            row["score_index"] = None
            row["unscored_reason"] = "evaluation_target_reached_within_vectorized_batch"
            unscored.append(row)
    return scored, unscored


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _immediate_failure_reason(row: Mapping[str, Any]) -> str | None:
    if row.get("success") is not False:
        return None
    if row.get("immediate_failure") is True:
        return str(row.get("reason") or "unknown")
    phase = str(row.get("failure_phase", "")).lower()
    if phase in {"immediate", "initial", "reset"}:
        return str(row.get("reason") or phase)
    reason = str(row.get("reason") or "")
    if reason.lower().startswith(("immediate_", "instant_", "reset_")):
        return reason
    return None


class EpisodeAccumulator:
    """以有界内存汇总只追加的完整回合记录。"""

    def __init__(self) -> None:
        self.count = 0
        self.success_count = 0
        self.success_observed = 0
        self.contact_sum = 0.0
        self.contact_count = 0
        self.contact_zero_count = 0
        self.required_contact_sum = 0.0
        self.required_contact_count = 0
        self.required_contact_zero_count = 0
        self.hold_count = 0
        self.height_count = 0
        self.max_hold_value: float | None = None
        self.max_height_value: float | None = None
        self.reason_counts: Counter[str] = Counter()
        self.failure_reason_counts: Counter[str] = Counter()
        self.failure_count = 0
        self.immediate_failure_count = 0
        self.immediate_failure_reason_counts: Counter[str] = Counter()

    def add(self, row: Mapping[str, Any]) -> None:
        row = normalize_episode_record(row)
        self.count += 1

        success = row.get("success")
        if isinstance(success, bool):
            self.success_observed += 1
            self.success_count += int(success)
            if not success:
                self.failure_count += 1
                reason = row.get("reason")
                if reason is not None:
                    self.failure_reason_counts[str(reason)] += 1
                immediate_reason = _immediate_failure_reason(row)
                if immediate_reason is not None:
                    self.immediate_failure_count += 1
                    self.immediate_failure_reason_counts[immediate_reason] += 1

        contact = _finite_number(row.get("contact_fraction"))
        if contact is not None:
            self.contact_sum += contact
            self.contact_count += 1
            self.contact_zero_count += int(contact <= 0.0)

        required_contact = _finite_number(row.get("required_contact_fraction"))
        if required_contact is not None:
            self.required_contact_sum += required_contact
            self.required_contact_count += 1
            self.required_contact_zero_count += int(required_contact <= 0.0)

        hold = _finite_number(row.get("max_unsupported_hold_s"))
        if hold is not None:
            self.hold_count += 1
            self.max_hold_value = hold if self.max_hold_value is None else max(self.max_hold_value, hold)
        height = _finite_number(row.get("max_height_gain_m"))
        if height is not None:
            self.height_count += 1
            self.max_height_value = (
                height if self.max_height_value is None else max(self.max_height_value, height)
            )

        reason = row.get("reason")
        if reason is not None:
            self.reason_counts[str(reason)] += 1

    def summary(self) -> dict[str, Any]:
        if self.contact_count:
            contact_mean: float | None = self.contact_sum / self.contact_count
            contact_zero: bool | None = (
                self.contact_zero_count == self.contact_count
                if self.contact_count == self.count
                else None
            )
            contact_missing = (
                None
                if self.contact_count == self.count
                else "some completed episodes did not provide contact_fraction"
            )
        else:
            contact_mean = None
            contact_zero = None
            contact_missing = "no completed episode provided contact_fraction"

        if self.required_contact_count:
            required_contact_mean: float | None = (
                self.required_contact_sum / self.required_contact_count
            )
            required_contact_zero: bool | None = (
                self.required_contact_zero_count == self.required_contact_count
                if self.required_contact_count == self.count
                else None
            )
            required_contact_missing = (
                None
                if self.required_contact_count == self.count
                else "some completed episodes did not provide required_contact_fraction"
            )
        else:
            required_contact_mean = None
            required_contact_zero = None
            required_contact_missing = "no completed episode provided required_contact_fraction"

        if self.success_observed == self.count and self.count:
            success_rate: float | None = self.success_count / self.count
            success_missing = None
        elif self.success_observed:
            success_rate = None
            success_missing = "some completed episodes did not provide a boolean success value"
        else:
            success_rate = None
            success_missing = "no completed episode provided a boolean success value"

        if self.success_observed < self.count:
            same_immediate_failure: bool | None = None
        elif self.failure_count == 0:
            same_immediate_failure: bool | None = False
        elif self.immediate_failure_count < self.failure_count:
            same_immediate_failure = None
        else:
            same_immediate_failure = len(self.immediate_failure_reason_counts) == 1

        return {
            "episodes_completed": self.count,
            "success_count": self.success_count if self.success_observed else None,
            "success_observed": self.success_observed,
            "success_missing_count": self.count - self.success_observed,
            "success_rate": success_rate,
            "success_missing_reason": success_missing,
            "contact_fraction_mean": contact_mean,
            "contact_fraction_zero": contact_zero,
            "contact_fraction_observed": self.contact_count,
            "contact_fraction_missing_count": self.count - self.contact_count,
            "contact_fraction_missing_reason": contact_missing,
            "required_contact_fraction_mean": required_contact_mean,
            "required_contact_fraction_zero": required_contact_zero,
            "required_contact_fraction_observed": self.required_contact_count,
            "required_contact_fraction_missing_count": self.count - self.required_contact_count,
            "required_contact_fraction_missing_reason": required_contact_missing,
            "max_unsupported_hold_s": self.max_hold_value,
            "max_unsupported_hold_observed": self.hold_count,
            "max_unsupported_hold_missing_count": self.count - self.hold_count,
            "max_unsupported_hold_missing_reason": (
                None
                if self.hold_count == self.count
                else "some completed episodes did not provide max_unsupported_hold_s"
            ),
            "max_height_gain_m": self.max_height_value,
            "max_height_gain_observed": self.height_count,
            "max_height_gain_missing_count": self.count - self.height_count,
            "max_height_gain_missing_reason": (
                None
                if self.height_count == self.count
                else "some completed episodes did not provide max_height_gain_m"
            ),
            "failure_count": self.failure_count,
            "failure_reason_counts": dict(sorted(self.failure_reason_counts.items())),
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "same_immediate_failure": same_immediate_failure,
            "immediate_failure_reason_counts": dict(
                sorted(self.immediate_failure_reason_counts.items())
            ),
            "same_immediate_failure_missing_reason": (
                None
                if same_immediate_failure is not None
                else "episode rows lack a complete success value or explicit immediate-failure marker"
            ),
        }


def summarize_episode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accumulator = EpisodeAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.summary()


def summarize_inspection_window(
    rows: Sequence[Mapping[str, Any]], minimum_major_failure_fraction: float = 0.5
) -> dict[str, Any]:
    """汇总一个非累计止损窗口，缺失接触数据不会被当作零接触。"""

    if not 0.0 < minimum_major_failure_fraction <= 1.0:
        raise ValueError("minimum_major_failure_fraction must be in (0, 1]")
    summary = summarize_episode_rows(rows)
    episode_count = len(rows)
    if summary["contact_fraction_observed"] != episode_count or episode_count == 0:
        summary["contact_fraction_zero"] = None
        summary["contact_fraction_zero_missing_reason"] = (
            "every completed episode in the inspection window must provide contact_fraction"
        )
    else:
        summary["contact_fraction_zero_missing_reason"] = None

    if summary["required_contact_fraction_observed"] != episode_count or episode_count == 0:
        summary["required_contact_fraction_zero"] = None
        summary["required_contact_fraction_zero_missing_reason"] = (
            "every completed episode in the inspection window must provide required_contact_fraction"
        )
    else:
        summary["required_contact_fraction_zero_missing_reason"] = None

    immediate_counts = summary["immediate_failure_reason_counts"]
    if immediate_counts:
        dominant_reason, dominant_count = min(
            immediate_counts.items(), key=lambda item: (-item[1], item[0])
        )
    else:
        dominant_reason, dominant_count = None, 0
    major_threshold_count = math.ceil(episode_count * minimum_major_failure_fraction)
    major_reason = (
        dominant_reason
        if dominant_reason is not None and dominant_count >= major_threshold_count
        else None
    )
    summary.update(
        {
            "dominant_immediate_failure_reason": dominant_reason,
            "dominant_immediate_failure_count": dominant_count,
            "dominant_immediate_failure_fraction": (
                dominant_count / episode_count if episode_count else None
            ),
            "major_immediate_failure_reason": major_reason,
            "major_immediate_failure_threshold_fraction": minimum_major_failure_fraction,
        }
    )
    return summary


class InspectionWindowStopState:
    """对连续的非重叠检查窗口计数，不把缺失接触值视为零。"""

    def __init__(self) -> None:
        self.consecutive_zero_contact_windows = 0
        self.zero_contact_metric: str | None = None
        self._previous_major_failure: str | None = None

    def observe(self, summary: Mapping[str, Any]) -> str | None:
        required_zero = summary.get("required_contact_fraction_zero")
        if isinstance(required_zero, bool):
            metric = "required_contact_fraction"
            zero_contact = required_zero
        else:
            generic_zero = summary.get("contact_fraction_zero")
            metric = "contact_fraction" if isinstance(generic_zero, bool) else None
            zero_contact = generic_zero if isinstance(generic_zero, bool) else None

        if zero_contact is True:
            self.consecutive_zero_contact_windows = (
                self.consecutive_zero_contact_windows + 1
                if self.zero_contact_metric == metric
                else 1
            )
            self.zero_contact_metric = metric
        else:
            self.consecutive_zero_contact_windows = 0
            self.zero_contact_metric = None

        stop_reason = None
        if self.consecutive_zero_contact_windows >= 2:
            if self.zero_contact_metric == "required_contact_fraction":
                stop_reason = "two_consecutive_zero_required_contact_windows"
            else:
                stop_reason = "two_consecutive_zero_contact_windows"

        current_major_failure = summary.get("major_immediate_failure_reason")
        if (
            stop_reason is None
            and current_major_failure is not None
            and current_major_failure == self._previous_major_failure
        ):
            stop_reason = "two_inspection_windows_same_failure"
        self._previous_major_failure = (
            str(current_major_failure) if current_major_failure is not None else None
        )
        return stop_reason


def open_hand_action_provider(env: Any) -> tuple[str, Any]:
    """优先启用环境 neutral 目标旁路，否则要求环境提供显式开手动作。"""

    baseline_setter = getattr(env, "set_open_hand_baseline", None)
    if callable(baseline_setter):
        result = baseline_setter(True)
        if result is False:
            raise RuntimeError("environment rejected set_open_hand_baseline(True)")
        if result is None or result is True:
            baseline_enabled = getattr(env, "open_hand_baseline_enabled", None)
            if baseline_enabled is not None and not bool(baseline_enabled):
                raise RuntimeError("environment did not enable the neutral open-hand baseline")
            return "set_open_hand_baseline", None
        return "set_open_hand_baseline", result

    for method_name in ("open_hand_action", "default_joint_pos_action"):
        method = getattr(env, method_name, None)
        if callable(method):
            action = method()
            if action is None:
                raise ValueError(f"environment.{method_name}() returned no action")
            return method_name, action
    raise AttributeError(
        "open policy requires env.set_open_hand_baseline(True), env.open_hand_action(), "
        "or env.default_joint_pos_action(); no implicit action fallback is allowed"
    )


def validate_open_action_shape(
    action: Any, num_envs: int, num_actions: int, torch_module: Any, device: str
) -> Any:
    """把显式开手动作整理为有限值向量批次，不自行推造动作。"""

    tensor = torch_module.as_tensor(action, dtype=torch_module.float32, device=device)
    if tensor.ndim == 1:
        if tensor.shape[0] != num_actions:
            raise ValueError(f"open-hand action must have {num_actions} values")
        tensor = tensor.unsqueeze(0).expand(num_envs, -1).contiguous()
    elif tensor.ndim == 2:
        if tensor.shape[1] != num_actions or tensor.shape[0] not in (1, num_envs):
            raise ValueError(
                f"open-hand action shape must be ({num_actions},), (1, {num_actions}), "
                f"or ({num_envs}, {num_actions})"
            )
        if tensor.shape[0] == 1 and num_envs > 1:
            tensor = tensor.expand(num_envs, -1).contiguous()
    else:
        raise ValueError("open-hand action must be a vector or a batch of vectors")
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise ValueError("open-hand action contains a non-finite value")
    return tensor
