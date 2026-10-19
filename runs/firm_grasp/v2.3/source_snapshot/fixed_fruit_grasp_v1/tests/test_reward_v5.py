"""CPU-only 回归：fixed_fruit_grasp_v1.5.0 的 reward-only 权重补丁。

可移植性与参照来源（重要）
--------------------------
* 常规奖励断言（逐项加权和、sticky 硬故障、drop/success、identity 权重、物理
  成功不变量）只依赖本工作树 + torch，不读任何外部盘。
* 只有「与 v1.4.0 基线逐行/AST 比对」的 reference-only 审查测试需要参照源，
  解析顺序为：
    1. git 对象 FIXED_FRUIT_GRASP_V4_REF（默认冻结提交
       51799b24e5356bd0ae62bccde08476cc54092fba）里的同名文件；
    2. 只读参照目录 FIXED_FRUIT_GRASP_V4_REFERENCE（兼容旧 source-v4 路径）。
  两者都拿不到时，reference-only 测试显式 pytest.skip 并记录缺源原因；不会
  自动 PASS，也不把旧源码打包进本工程。
* 在本机（source-v5 git 工作树）git 对象可用，因此全部 reference-only 检查
  都必须真正执行、零 skip；一旦出现 skip 即表示参照条件退化，需在报告说明。
* 边界：不启动 Kit/PhysX/GPU，不做训练或评估；参照源全程只读。
"""

from __future__ import annotations

import ast
import difflib
import functools
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.fixed_fruit_grasp_v1 import contract as C
from research.fixed_fruit_grasp_v1.env import (
    FixedFruitGraspEnv,
    _APPROACH_GAP_DECAY_M,
    _APPROACH_SHAPING_GAMMA,
    _HARD_FAULT_REWARD_NAMES,
    _STABILITY_DRIFT_DECAY_M,
)

# ---------------------------------------------------------------- 参照源解析
REFERENCE_ENV_VAR = "FIXED_FRUIT_GRASP_V4_REF"
REFERENCE_PATH_ENV_VAR = "FIXED_FRUIT_GRASP_V4_REFERENCE"
FROZEN_BASELINE_COMMIT = "51799b24e5356bd0ae62bccde08476cc54092fba"
LEGACY_REFERENCE_PATH = Path(
    "/tmp/linkbot-ppo-grasp-learning-20261013/source-v4/research/"
    "fixed_fruit_grasp_v1"
)
REFERENCE_SKIP_REASON = "reference-only v1.4.0 baseline audit needs a reference source"

# v1.5.0 目标奖励配置（带符号贡献系数：惩罚项为负）。
V150_TERM_WEIGHTS = {
    "approach": 1.0,
    "contact": 0.25,
    "opposition": 0.25,
    "band": 0.35,
    "closure": 0.0,
    "unsupported_hold": 6.0,
    "stability": 0.50,
    "height": 0.0,
    "palm_only_penalty": -0.10,
    "overload_penalty": -0.30,
    "forbidden_penalty": -0.0,
    "smoothness_penalty": -0.12,
    "hard_fault_penalty": -8.00,
    "drop_penalty": -6.00,
    "success_bonus": 30.0,
}

# v1.4.0 参照权重：只用于证明差异恰好是这七项。
V140_TERM_WEIGHTS = {
    **V150_TERM_WEIGHTS,
    "contact": 1.0,
    "opposition": 1.0,
    "unsupported_hold": 2.0,
    "stability": 0.20,
    "hard_fault_penalty": -2.00,
    "drop_penalty": -1.50,
    "success_bonus": 10.0,
}
V150_CHANGED_TERMS = {
    name
    for name, value in V150_TERM_WEIGHTS.items()
    if value != V140_TERM_WEIGHTS[name]
}

# 奖励分量名 -> 带符号罚项名（identity 里惩罚项写成 -float(cfg.<field>)）。
PENALTY_TERMS = frozenset(
    {
        "palm_only_penalty",
        "overload_penalty",
        "forbidden_penalty",
        "smoothness_penalty",
        "hard_fault_penalty",
        "drop_penalty",
    }
)

# 奖励分量名 -> FixedFruitGraspCfg 字段名（快照/identity 权重与配置字段的对应）。
TERM_CFG_FIELDS = {
    "approach": "approach_reward_weight",
    "contact": "contact_reward_weight",
    "opposition": "opposition_reward_weight",
    "band": "band_reward_weight",
    "closure": "closure_reward_weight",
    "unsupported_hold": "unsupported_hold_reward_weight",
    "stability": "stability_reward_weight",
    "height": "height_reward_weight",
    "palm_only_penalty": "palm_only_penalty_weight",
    "overload_penalty": "overload_penalty_weight",
    "forbidden_penalty": "forbidden_penalty_weight",
    "smoothness_penalty": "smoothness_penalty_weight",
    "hard_fault_penalty": "fault_penalty",
    "drop_penalty": "drop_penalty",
    "success_bonus": "success_bonus",
}

# env.py/contract.py 是本补丁的合法改动面；run_utils.py/train.py 归主线程。
_PACKAGE_EXCLUDED_FROM_BYTE_COMPARE = {
    "env.py",
    "contract.py",
    "run_utils.py",
    "train.py",
}
_BASELINE_REQUIRED_MODULES = (
    "physical_hold.py",
    "physical_contacts.py",
    "collision_cooking.py",
    "__init__.py",
)
# 只允许与 source-v4 不同的两个 env 方法。
_ALLOWED_ENV_METHOD_DIFFERENCES = {"_get_rewards", "_build_identity"}
_GUARDED_ENV_METHODS = (
    "_setup_scene",
    "_install_physical_hold_substep_hook",
    "_observe_physical_hold_substep",
    "_refresh_task_state",
    "_pre_physics_step",
    "_get_dones",
    "_reset_idx",
    "_finalize_episodes",
    "task_identity",
)
# v1.6.0 仍然只把合同版本字面量这一行相对 v1.4.0 参照源改掉（奖励权重子树
# 保持 v1.5.0 目标值），因此这里的期望值随当前版本更新，断言本身不删除。
_CONTRACT_CHANGED_LINES = (
    '-TASK_VERSION: Final = "fixed_fruit_grasp_v1.4.0"',
    '+TASK_VERSION: Final = "fixed_fruit_grasp_v1.6.0"',
)


class ReferenceUnavailable(RuntimeError):
    """v1.4.0 参照源既不在 git 对象里，也不在只读参照目录里。"""


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_reference(name: str, ref: str) -> tuple[str | None, str]:
    repo_root = _package_root().parents[1]
    relative = (_package_root() / name).relative_to(repo_root).as_posix()
    label = "git@" + ref + ":" + relative
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root), "show", ref + ":" + relative],
            capture_output=True,
            text=True,
        )
    except OSError as error:  # git 不可执行
        return None, label + " -> " + str(error)
    if done.returncode == 0:
        return done.stdout, label
    detail = done.stderr.strip().splitlines()
    return None, label + " -> " + (detail[-1] if detail else "git show failed")


@functools.lru_cache(maxsize=None)
def _reference_text(name: str) -> tuple[str, str]:
    """返回 (参照文本, 来源标签)；两种来源都不可用时抛 ReferenceUnavailable。"""

    attempts: list[str] = []
    ref = os.environ.get(REFERENCE_ENV_VAR, FROZEN_BASELINE_COMMIT)
    text, label = _git_reference(name, ref)
    if text is not None:
        return text, label
    attempts.append(label)

    root = Path(os.environ.get(REFERENCE_PATH_ENV_VAR, str(LEGACY_REFERENCE_PATH)))
    path = root / name
    if path.is_file():
        return _source_text(path), "path:" + str(path)
    attempts.append("path:" + str(path) + " -> missing")
    raise ReferenceUnavailable(" | ".join(attempts))


def _require_reference(name: str) -> tuple[str, str]:
    """reference-only 审查测试的入口：缺源时显式 skip（绝不自动 PASS）。"""

    try:
        text, label = _reference_text(name)
    except ReferenceUnavailable as error:
        pytest.skip(REFERENCE_SKIP_REASON + " :: " + str(error))
    assert text, name
    return text, label


def _parse_text(text: str) -> ast.Module:
    return ast.parse(text)


def _node_dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _expr_dump(text: str) -> str:
    return _node_dump(ast.parse(text.strip(), mode="eval").body)


def _class_node(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError("未找到类 " + class_name)


def _target_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.AnnAssign):
        targets: list[ast.expr] = [node.target]
    elif isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        return []
    return [target.id for target in targets if isinstance(target, ast.Name)]


def _constant_value(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


def _numeric_class_defaults(text: str, class_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for node in _class_node(_parse_text(text), class_name).body:
        if not isinstance(node, (ast.AnnAssign, ast.Assign)) or node.value is None:
            continue
        constant = _constant_value(node.value)
        if constant is None:
            continue
        for name in _target_names(node):
            out[name] = constant
    return out


def _method_dumps(text: str, class_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in _class_node(_parse_text(text), class_name).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = _node_dump(node)
    return out


def _top_level_names(text: str) -> set[str]:
    names: set[str] = set()
    for node in _parse_text(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        else:
            names.update(_target_names(node))
    return names


def _top_level_assignment_expressions(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in _parse_text(text).body:
        if not isinstance(node, (ast.AnnAssign, ast.Assign)) or node.value is None:
            continue
        dump = ast.unparse(node.value)
        for name in _target_names(node):
            out[name] = dump
    return out


def _literal_assignment_value(text: str, name: str):
    for node in _parse_text(text).body:
        if not isinstance(node, (ast.AnnAssign, ast.Assign)) or node.value is None:
            continue
        if name not in _target_names(node):
            continue
        return ast.literal_eval(node.value)
    raise AssertionError("未找到模块级赋值 " + name)


def _changed_lines(reference: str, current: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(
            reference.splitlines(), current.splitlines(), lineterm="", n=0
        )
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def _dict_entry(node: ast.Dict, key: str) -> ast.expr | None:
    for dict_key, dict_value in zip(node.keys, node.values):
        if isinstance(dict_key, ast.Constant) and dict_key.value == key:
            return dict_value
    return None


def _identity_reward_subtree(path: Path) -> ast.Dict:
    build = None
    for node in _class_node(_parse_text(_source_text(path)), "FixedFruitGraspEnv").body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_identity":
            build = node
    assert build is not None, "未找到 FixedFruitGraspEnv._build_identity"
    for node in ast.walk(build):
        if not isinstance(node, ast.Dict):
            continue
        entry = _dict_entry(node, "reward")
        if isinstance(entry, ast.Dict):
            return entry
    raise AssertionError("_build_identity 中没有 reward 子树")


def _env_method_source(method_name: str) -> str:
    text = _source_text(_package_root() / "env.py")
    for node in _class_node(_parse_text(text), "FixedFruitGraspEnv").body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.unparse(node)
    raise AssertionError("未找到方法 FixedFruitGraspEnv." + method_name)


def _reward_env(
    *,
    num_envs: int = 4,
    sticky_fault_env_ids: tuple[int, ...] = (),
    drop_env_ids: tuple[int, ...] = (),
    success_env_ids: tuple[int, ...] = (),
):
    """构造只用于 _get_rewards 的 CPU 替身 env，返回 (env, 输入快照)。"""

    env = FixedFruitGraspEnv.__new__(FixedFruitGraspEnv)
    env._is_closed = True  # 替身不跑 __init__，避免 DirectRLEnv 析构噪声
    env._state_ready = True
    env.extras = {}

    cfg = SimpleNamespace(
        required_unsupported_hold_steps=20,
        contact_force_threshold_n=C.CONTACT_FORCE_N,
    )
    for term, field in TERM_CFG_FIELDS.items():
        setattr(cfg, field, abs(V150_TERM_WEIGHTS[term]))
    env.cfg = cfg

    index = torch.arange(num_envs, dtype=torch.float32)
    force = 0.10 + 0.02 * index.unsqueeze(1) + torch.zeros(
        (1, len(C.FINGER_NAMES))
    )
    raw = force.clone()
    if num_envs > 1:
        raw[-1] = raw[-1] + 16.0  # 只触发策略采样 overload 项，远低于子步硬门
    released = torch.ones(num_envs, dtype=torch.bool)
    if num_envs > 1:
        released[-1] = False  # 最后一个环境未撤支撑
    hold_age = torch.full((num_envs,), 10, dtype=torch.long)
    hold_age[~released] = 0

    env._stem_detached = released
    env._hold_ok = released.clone()
    env._hold_age = hold_age
    env._hold_window_drift_m = torch.full((num_envs,), 0.005)
    env._last_physical_force_n = force
    env._raw_finger_force_n = raw
    env._soft_low = torch.tensor(C.FINGER_SOFT_LOW_N)
    env._soft_high = torch.tensor(C.FINGER_SOFT_HIGH_N)
    env._hard_high = torch.tensor(C.FINGER_HARD_HIGH_N)
    env._contact_mask = torch.ones(
        (num_envs, len(C.FINGER_NAMES)), dtype=torch.bool
    )
    env._required_contact = torch.arange(num_envs) % 2 == 0
    env._last_palm_force_n = torch.zeros(num_envs)
    env._forbidden_contact_force_n = torch.zeros((num_envs, 3))
    env._forbidden_proximal_force_n = torch.zeros((num_envs, 6))
    env._executed_actions = torch.full((num_envs, C.ACTION_DIM), 0.05)
    env._filter = SimpleNamespace(
        last_stats={
            "action_delta": torch.full((num_envs, C.ACTION_DIM), 0.01),
            "jerk": torch.full((num_envs, C.ACTION_DIM), 0.02),
        }
    )
    faults = {
        name: torch.zeros(num_envs, dtype=torch.bool)
        for name in (*_HARD_FAULT_REWARD_NAMES, "stem_fracture")
    }
    for env_id in sticky_fault_env_ids:
        faults["instability"][env_id] = True
    env._faults = faults
    env._height_gain_m = torch.zeros(num_envs)
    env._drift_m = torch.zeros(num_envs)
    env._rel_speed_m_s = torch.zeros(num_envs)
    env._self_collision = torch.zeros(num_envs, dtype=torch.bool)
    env._open_hand_baseline = False
    env._drop_now = torch.zeros(num_envs, dtype=torch.bool)
    env._drop_now[list(drop_env_ids)] = True
    env._terminal_success = torch.zeros(num_envs, dtype=torch.bool)
    env._terminal_success[list(success_env_ids)] = True
    env._approach_potential_prev = torch.zeros(num_envs)
    env._approach_potential_ready = torch.zeros(num_envs, dtype=torch.bool)

    # 5 指 AABB 到果实球心的距离：列序与 _hand_collision_body_ids 一致（无
    # base_link），间距固定，势值可在测试里解析复算。
    digit_gaps = (0.004, 0.004, 0.012, 0.020, 0.030)
    offsets = torch.zeros((len(C.FINGER_NAMES), 3))
    for position, gap in enumerate(digit_gaps):
        offsets[position, 0] = C.FRUIT_RADIUS_M + gap
    root = torch.zeros((num_envs, 3))
    env._hand_collision_closest_points_for = (
        lambda positions, env_ids: positions.unsqueeze(1) + offsets.unsqueeze(0)
    )
    env._tomato = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root))
    )
    env._fruit_axes_m = torch.full((num_envs, 3), C.FRUIT_RADIUS_M)
    env._robot = SimpleNamespace(
        body_names=["base_link", *(digit + "_distal" for digit in C.FINGER_NAMES)]
    )
    env._hand_collision_body_ids = list(range(1, len(C.FINGER_NAMES) + 1))
    env._all_env_ids = torch.arange(num_envs)

    inputs = {
        "released": released.float(),
        "hold_age": hold_age,
        "drift": env._hold_window_drift_m,
        "hold_ok": released.clone(),
        "contact_fraction": env._contact_mask.float().mean(dim=1),
        "required": env._required_contact.float(),
        "in_band": (
            (force >= env._soft_low) & (force <= env._soft_high)
        ).float().mean(dim=1),
        "drive": env._executed_actions.abs().mean(dim=1),
        "action_delta": torch.full((num_envs,), 0.01),
        "jerk": torch.full((num_envs,), 0.02),
        "overload": torch.relu(raw - env._hard_high).sum(dim=1),
        "palm_only": torch.zeros(num_envs),
        "forbidden": torch.zeros(num_envs),
        "hard_fault": torch.stack(
            [faults[name] for name in _HARD_FAULT_REWARD_NAMES], dim=0
        ).any(dim=0),
        "drop": env._drop_now.float(),
        "success": env._terminal_success.float(),
        "digit_gaps": torch.tensor(digit_gaps, dtype=torch.float32)
        .unsqueeze(0)
        .expand(num_envs, -1)
        .contiguous(),
        "required_hold_steps": 20,
    }
    return env, inputs


def _expected_reward(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """用 v1.5.0 目标权重独立复算 _get_rewards 的逐 env 结果。"""

    potential = torch.exp(
        -inputs["digit_gaps"].clamp_min(0.0) / _APPROACH_GAP_DECAY_M
    ).topk(2, dim=1).values.mean(dim=1)
    # 第二次策略步调用的势值未变：delta = gamma * Phi - Phi。
    approach = (_APPROACH_SHAPING_GAMMA - 1.0) * potential
    stability = (
        torch.exp(-inputs["drift"].clamp_min(0.0) / _STABILITY_DRIFT_DECAY_M)
        * inputs["hold_ok"].float()
        * inputs["released"]
    )
    unweighted = {
        "approach": approach,
        "contact": inputs["contact_fraction"],
        "opposition": inputs["required"],
        "band": inputs["in_band"],
        "closure": inputs["drive"] * (1.0 - inputs["required"]),
        "unsupported_hold": (
            inputs["hold_age"].float()
            / max(1, int(inputs["required_hold_steps"]))
        ).clamp(0.0, 1.0)
        * inputs["released"],
        "stability": stability,
        "height": torch.zeros_like(inputs["drive"]),
        "palm_only_penalty": inputs["palm_only"],
        "overload_penalty": inputs["overload"],
        "forbidden_penalty": inputs["forbidden"],
        "smoothness_penalty": inputs["action_delta"] + 0.05 * inputs["jerk"],
        "hard_fault_penalty": inputs["hard_fault"].float(),
        "drop_penalty": inputs["drop"],
        "success_bonus": inputs["success"],
    }
    total = None
    for name in V150_TERM_WEIGHTS:  # 与源码同一相加顺序
        term = V150_TERM_WEIGHTS[name] * unweighted[name]
        total = term if total is None else total + term
    assert total is not None
    return total


def test_reference_source_is_pinned_to_the_frozen_baseline() -> None:
    """标记本机参考审查条件；缺源时 skip 并记录原因，绝不自动 PASS。"""

    text, label = _require_reference("contract.py")
    assert label.startswith(("git@", "path:")), label
    if label.startswith("git@"):
        ref = os.environ.get(REFERENCE_ENV_VAR, FROZEN_BASELINE_COMMIT)
        assert label.startswith("git@" + ref), label
        assert len(ref) == 40, ref
    assert _literal_assignment_value(text, "TASK_VERSION") == "fixed_fruit_grasp_v1.4.0"


def test_reward_terms_sum_to_the_actual_reward() -> None:
    env, inputs = _reward_env(num_envs=4)
    env._get_rewards()  # 第一次调用只缓存接近势，不发 shaping
    reward = env._get_rewards()
    assert torch.allclose(reward, _expected_reward(inputs), atol=1e-6, rtol=1e-5)

    snapshot = env.extras["fixed_grasp_reward_terms"]
    assert snapshot["scope"] == "last_policy_step_batch_snapshot_not_episode_sum"
    assert snapshot["num_envs"] == 4
    assert snapshot["total_mean"] == pytest.approx(float(reward.mean().item()), abs=1e-9)
    assert snapshot["support_release_fraction"] == pytest.approx(
        float(inputs["released"].mean().item()), abs=1e-12
    )
    assert set(snapshot) == {
        "scope",
        "num_envs",
        "support_release_fraction",
        "total_mean",
        "terms",
    }

    terms = snapshot["terms"]
    assert set(terms) == set(V150_TERM_WEIGHTS)
    for name, term in terms.items():
        assert term["weight"] == pytest.approx(V150_TERM_WEIGHTS[name], abs=0.0)
        assert term["weighted_mean"] == pytest.approx(
            term["weight"] * term["unweighted_mean"], abs=1e-6
        ), name
    weighted_sum = sum(term["weighted_mean"] for term in terms.values())
    assert weighted_sum == pytest.approx(snapshot["total_mean"], abs=1e-6)


def test_single_env_snapshot_is_elementwise_equal_to_reward() -> None:
    env, inputs = _reward_env(num_envs=1)
    env._get_rewards()
    reward = env._get_rewards()
    assert torch.allclose(reward, _expected_reward(inputs), atol=1e-6, rtol=1e-5)
    terms = env.extras["fixed_grasp_reward_terms"]["terms"]
    # num_envs=1 时均值就是该元素本身：逐项之和必须逐元素等于 reward。
    assert sum(term["weighted_mean"] for term in terms.values()) == pytest.approx(
        float(reward[0].item()), abs=1e-6
    )


def test_drop_and_success_bonus_use_the_v150_values() -> None:
    env, inputs = _reward_env(num_envs=4, drop_env_ids=(1,), success_env_ids=(2,))
    env._get_rewards()
    reward = env._get_rewards()
    assert torch.allclose(reward, _expected_reward(inputs), atol=1e-6, rtol=1e-5)

    terms = env.extras["fixed_grasp_reward_terms"]["terms"]
    assert terms["drop_penalty"]["weight"] == pytest.approx(-6.0)
    assert terms["success_bonus"]["weight"] == pytest.approx(30.0)
    assert terms["drop_penalty"]["unweighted_mean"] == pytest.approx(0.25)
    assert terms["success_bonus"]["unweighted_mean"] == pytest.approx(0.25)
    assert terms["drop_penalty"]["weighted_mean"] == pytest.approx(-1.5)
    assert terms["success_bonus"]["weighted_mean"] == pytest.approx(7.5)


def test_sticky_substep_hard_fault_is_covered_by_the_fault_penalty() -> None:
    clean, _ = _reward_env(num_envs=4)
    faulty, _ = _reward_env(num_envs=4, sticky_fault_env_ids=(1, 3))
    clean._get_rewards()
    clean_reward = clean._get_rewards()
    faulty._get_rewards()
    faulty_reward = faulty._get_rewards()

    delta = (faulty_reward - clean_reward).tolist()
    assert delta == pytest.approx([0.0, -8.0, 0.0, -8.0], abs=1e-5)

    term = faulty.extras["fixed_grasp_reward_terms"]["terms"]["hard_fault_penalty"]
    assert term["weight"] == pytest.approx(-8.0)
    assert term["unweighted_mean"] == pytest.approx(0.5)
    assert term["weighted_mean"] == pytest.approx(-4.0)

    # 罚项覆盖的正是七类粘滞子步硬故障：名字集合必须逐字一致。
    assert set(_HARD_FAULT_REWARD_NAMES) == set(
        C.physical_hold_semantics()["hard_fault_names"]
    )
    compact = " ".join(_source_text(_package_root() / "env.py").split())
    assert (
        "for name in _HARD_FAULT_REWARD_NAMES: "
        "self._faults[name] |= guard.hard_faults[name]" in compact
    ), "粘滞子步硬故障并入 _faults 的接缝必须保留"


def test_identity_reward_subtree_lists_every_numeric_weight() -> None:
    reward = _identity_reward_subtree(_package_root() / "env.py")
    weights = _dict_entry(reward, "weights")
    assert isinstance(weights, ast.Dict)
    listed = {
        key.value: _node_dump(value)
        for key, value in zip(weights.keys, weights.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert set(listed) == set(V150_TERM_WEIGHTS)
    for term in V150_TERM_WEIGHTS:
        field = TERM_CFG_FIELDS[term]
        expression = (
            "-float(cfg." + field + ")"
            if term in PENALTY_TERMS
            else "float(cfg." + field + ")"
        )
        assert listed[term] == _expr_dump(expression), term

    key_entry = _dict_entry(reward, "per_policy_step_snapshot_key")
    scope_entry = _dict_entry(reward, "per_policy_step_snapshot_scope")
    assert isinstance(key_entry, ast.Constant)
    assert key_entry.value == "fixed_grasp_reward_terms"
    assert isinstance(scope_entry, ast.Constant)
    assert scope_entry.value == "last_policy_step_batch_snapshot_not_episode_sum"
    assert len(listed) == len(V150_TERM_WEIGHTS)


def test_task_version_and_cfg_weights_are_the_only_cfg_changes() -> None:
    reference, label = _require_reference("env.py")
    current = _source_text(_package_root() / "env.py")
    assert label

    assert C.TASK_VERSION == "fixed_fruit_grasp_v1.6.0"
    v4_defaults = _numeric_class_defaults(reference, "FixedFruitGraspCfg")
    v5_defaults = _numeric_class_defaults(current, "FixedFruitGraspCfg")
    field_to_term = {field: term for term, field in TERM_CFG_FIELDS.items()}

    changed = {
        field_to_term[field]
        for field in v5_defaults
        if field in v4_defaults
        and field in field_to_term
        and v5_defaults[field] != v4_defaults[field]
    }
    assert changed == V150_CHANGED_TERMS

    for term, expected in V150_TERM_WEIGHTS.items():
        field = TERM_CFG_FIELDS[term]
        assert v5_defaults[field] == pytest.approx(abs(expected))
        assert v4_defaults[field] == pytest.approx(abs(V140_TERM_WEIGHTS[term]))


def test_env_methods_identical_to_v4_except_reward_and_identity() -> None:
    reference, _ = _require_reference("env.py")
    current = _source_text(_package_root() / "env.py")
    v4_methods = _method_dumps(reference, "FixedFruitGraspEnv")
    v5_methods = _method_dumps(current, "FixedFruitGraspEnv")
    assert v4_methods and v5_methods

    for name in _GUARDED_ENV_METHODS:
        assert name in v4_methods, name
        assert name in v5_methods, name

    assert set(v5_methods) - set(v4_methods) == set()
    assert set(v4_methods) - set(v5_methods) == set()
    changed = {
        name for name in v4_methods if v4_methods[name] != v5_methods[name]
    }
    assert changed == _ALLOWED_ENV_METHOD_DIFFERENCES
    assert _top_level_names(current) == _top_level_names(reference)


def test_package_modules_byte_identical_to_v4_outside_the_reward_surface() -> None:
    compared: list[str] = []
    for path in sorted(_package_root().glob("*.py")):
        if path.name in _PACKAGE_EXCLUDED_FROM_BYTE_COMPARE:
            continue
        reference, label = _require_reference(path.name)
        assert reference == _source_text(path), path.name + " (" + label + ")"
        compared.append(path.name)
    for required in _BASELINE_REQUIRED_MODULES:
        assert required in compared, required


def test_contract_source_identical_to_v4_except_task_version() -> None:
    reference, label = _require_reference("contract.py")
    current = _source_text(_package_root() / "contract.py")
    assert label

    assert len(reference.splitlines()) == len(current.splitlines())
    assert _changed_lines(reference, current) == list(_CONTRACT_CHANGED_LINES)

    ref_assignments = _top_level_assignment_expressions(reference)
    cur_assignments = _top_level_assignment_expressions(current)
    assert set(cur_assignments) == set(ref_assignments)
    changed = {
        name
        for name in cur_assignments
        if cur_assignments[name] != ref_assignments[name]
    }
    assert changed == {"TASK_VERSION"}
    assert _literal_assignment_value(reference, "TASK_VERSION") == (
        "fixed_fruit_grasp_v1.4.0"
    )
    assert _literal_assignment_value(current, "TASK_VERSION") == (
        "fixed_fruit_grasp_v1.6.0"
    )


def test_physical_success_invariants_are_unchanged() -> None:
    assert C.POLICY_RATE_HZ == 20
    assert C.DECIMATION == 6
    assert C.SIM_DT_S == pytest.approx(1.0 / 120.0)
    assert C.FINGER_HARD_HIGH_N == (15.0,) * 5
    assert C.MIN_CONTACT_FINGERS == 2
    assert C.HOLD_MAX_DRIFT_M == pytest.approx(0.010)
    assert C.HOLD_MAX_RELATIVE_SPEED_M_S == pytest.approx(0.05)
    assert C.PENETRATION_LIMIT_M == pytest.approx(0.005)
    assert C.SELF_COLLISION_PENETRATION_LIMIT_M == pytest.approx(0.010)
    assert C.INSTABILITY_MAX_JOINT_VELOCITY_RAD_S == pytest.approx(4.0)
    assert C.ACTUATOR_EFFORT_LIMIT_NM == pytest.approx(0.20)
    assert C.ACTUATOR_VELOCITY_LIMIT_RAD_S == pytest.approx(1.20)
    assert C.ACTUATOR_STIFFNESS == pytest.approx(20.0)
    assert C.ACTUATOR_DAMPING == pytest.approx(1.0)
    assert C.ACTUATOR_ARMATURE_KG_M2 == pytest.approx(0.01)
    assert C.FRUIT_DIAMETER_M == pytest.approx(0.067)
    assert C.FRUIT_DENSITY_KG_M3 == pytest.approx(900.0)

    contact_stage = C.STAGE_SPECS["contact"]
    assert contact_stage.support_release_step == 60
    assert contact_stage.required_unsupported_hold_steps == 20
    assert contact_stage.required_unsupported_hold_substeps == 120
    assert contact_stage.min_height_gain_m == 0.0
    assert contact_stage.counts_as_goal is False

    semantics = C.physical_hold_semantics()
    assert semantics["guard_frequency"] == "per_physics_substep"
    assert semantics["hard_fault_persistence"] == "sticky_until_env_reset"
    assert semantics["missing_data_policy"] == "fail_closed"
    assert semantics["decimation"] == C.DECIMATION
    assert set(semantics["hard_fault_names"]) == set(_HARD_FAULT_REWARD_NAMES)


def test_main_thread_train_logs_the_snapshot_without_episode_accumulation() -> None:
    # train.py / run_utils.py 归主线程；这里只钉住两端共享的键名契约（只读）。
    text = _source_text(_package_root() / "train.py")
    assert 'env.extras.get("fixed_grasp_reward_terms")' in text
    assert "reward_component_last_policy_step" in text

    get_rewards = _env_method_source("_get_rewards")
    assert "fixed_grasp_reward_terms" in get_rewards
    assert "last_policy_step_batch_snapshot_not_episode_sum" in get_rewards
