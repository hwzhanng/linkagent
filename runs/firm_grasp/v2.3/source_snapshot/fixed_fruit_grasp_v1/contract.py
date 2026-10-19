"""固定单果抓握 v1 的纯 Python 合同（不导入 torch / Isaac / USD）。

本模块是任务身份、阶段判据、固定支撑释放时序、终止原因与逐回合记录
schema 的唯一来源，可在没有启动 Kit 的进程里直接导入和测试。张量实现
在 :mod:`research.fixed_fruit_grasp_v1.env`。

设计边界（与 ``.omo/plans/ppo-grasp-first-result-20261013.md`` 一致）：

* 固定 67 mm 等效直径单果，固定初始位置/朝向，无尺寸/位置/质量/摩擦随机化。
* 初始无接触；果实由原始声明的果梗弹簧支撑，在固定策略步彻底解除。
* 提举需要一根明确声明的受驱动升降关节；本版本未实现，因此 lift 阶段
  fail-closed 拒绝，交付物是 holding-only 任务（``unsupported_hold_s``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final


# --------------------------------------------------------------- 任务身份
TASK_ID: Final = "LinkAgent-FixedFruitGrasp-v1"
TASK_VERSION: Final = "fixed_fruit_grasp_v1.6.0"
RECORD_SCHEMA_VERSION: Final = "fixed_fruit_grasp_episode_record.v1"
POLICY_RATE_HZ: Final = 20
POLICY_DT_S: Final = 1.0 / POLICY_RATE_HZ
ACTION_DIM: Final = 14
ACTOR_OBSERVATION_DIM: Final = 114
CRITIC_STATE_DIM: Final = 175
SIM_DT_S: Final = 1.0 / 120.0
DECIMATION: Final = 6
EPISODE_STEPS: Final = 360

# --------------------------------------------- 物理子步保持守护语义（v1.4.0）
# v3 的 20 Hz 策略采样门控无法排除六个物理子步之间的瞬时失效（独立审查
# HOLD-P1-01；现场证据已确认 29 次速度越界与 2 次真实过载被漏掉）。v1.4.0
# 把保持计数与七类硬故障判定下移到真实 1/120 s 物理子步；下列常量是唯一
# 语义声明来源，env 与逐回合记录都必须引用它们，而不是各自写字符串。
PHYSICAL_HOLD_GUARD_FREQUENCY: Final = "per_physics_substep"
PHYSICAL_HOLD_HARD_FAULT_PERSISTENCE: Final = "sticky_until_env_reset"
PHYSICAL_HOLD_MISSING_DATA_POLICY: Final = "fail_closed"
PHYSICAL_HOLD_STREAK_SEMANTICS: Final = (
    "continuous_physical_substeps_zeroed_on_any_violation"
)
PHYSICAL_HOLD_ANCHOR_SEMANTICS: Final = (
    "hand_frame_fruit_position_at_physical_streak_start"
)
PHYSICAL_HOLD_SUCCESS_SEMANTICS: Final = (
    "success_requires_physical_streak_ge_required_policy_steps_times_decimation"
)


def physical_hold_semantics() -> dict[str, Any]:
    """稳定语义声明（写入任务身份与逐回合记录，便于外部核验）。"""

    return {
        "guard_frequency": PHYSICAL_HOLD_GUARD_FREQUENCY,
        "hold_streak_unit": "physics_substeps",
        "hold_streak_semantics": PHYSICAL_HOLD_STREAK_SEMANTICS,
        "hard_fault_persistence": PHYSICAL_HOLD_HARD_FAULT_PERSISTENCE,
        "hard_fault_names": [
            "overload",
            "nonfinite",
            "penetration",
            "joint_limit",
            "self_collision",
            "instability",
            "joint_effort",
        ],
        "anchor_reference": PHYSICAL_HOLD_ANCHOR_SEMANTICS,
        "success_semantics": PHYSICAL_HOLD_SUCCESS_SEMANTICS,
        "missing_data_policy": PHYSICAL_HOLD_MISSING_DATA_POLICY,
        "sim_dt_s": SIM_DT_S,
        "policy_dt_s": POLICY_DT_S,
        "decimation": DECIMATION,
    }


# --------------------------------------------------------- 固定果实与实际物性
# 等效直径 67 mm 的球形果实：半轴退化为单点，prestartup event 的采样结果
# 与 seed 无关。
FRUIT_DIAMETER_M: Final = 0.067
FRUIT_RADIUS_M: Final = FRUIT_DIAMETER_M / 2.0
FRUIT_AXES_M: Final = (FRUIT_RADIUS_M, FRUIT_RADIUS_M, FRUIT_RADIUS_M)
# 声明固定密度，取自既有声明域 650--1050 kg/m3 内部，不扩大该域。实际质量
# 由父环境按 4/3*pi*a*b*c*rho 计算并写回 PhysX，不由 cfg.tomato.mass 声明。
FRUIT_DENSITY_KG_M3: Final = 900.0
FRUIT_MASS_KG: Final = (
    4.0 / 3.0 * math.pi * FRUIT_RADIUS_M**3 * FRUIT_DENSITY_KG_M3
)
FRUIT_MASS_TOLERANCE_KG: Final = 5.0e-4
FRUIT_STATIC_FRICTION: Final = 0.70
FRUIT_DYNAMIC_FRICTION: Final = 0.58
FRUIT_RESTITUTION: Final = 0.01
FRUIT_FRICTION_COMBINE_MODE: Final = "multiply"
FRUIT_SOLVER_STATIC_FRICTION: Final = 1.0
FRUIT_SOLVER_DYNAMIC_FRICTION: Final = 1.0

# 驱动来源：沿用 canonical 孤立左手 USD 的显式 PD 参数（有依据值），只做
# wave2 记录的两处最小修复：限位 brake 之后再次按逐轴 effort 上限裁剪，
# 以及 active/follower armature 0.0001 -> 0.01。effort 上限不扩大。
ACTUATOR_EFFORT_LIMIT_NM: Final = 0.20
ACTUATOR_VELOCITY_LIMIT_RAD_S: Final = 1.20
ACTUATOR_STIFFNESS: Final = 20.0
ACTUATOR_DAMPING: Final = 1.0
ACTUATOR_FRICTION: Final = 0.01
ACTUATOR_ARMATURE_KG_M2: Final = 0.01
FOLLOWER_STIFFNESS: Final = 1.0
FOLLOWER_DAMPING: Final = 0.50

# 保留 canonical 软件 mimic 关系（multiplier 不变）。PPV1 派生资产
# (``assets/physical_prior_v1/left_hand_ppv1.usda``) 要求 offset=0 + Newton
# mimic，本任务不继承该未证实的运行时后端，因此显式记录而非静默切换。
MIMIC_RELATIONS: Final = {
    "thumb_joint4": ("thumb_joint3", 1.11, -0.04545),
    "index_joint3": ("index_joint2", 0.94, -0.04545),
    "middle_joint3": ("middle_joint2", 0.94, -0.04545),
    "ring_joint3": ("ring_joint2", 0.94, -0.04545),
    "little_joint3": ("little_joint2", 0.94, -0.04545),
}

# 固定支撑时序：支撑来自原始声明的果梗弹簧，不再由“已满足成功”触发解除。
# 断力取一个实际不可达的固定值，使解除完全由固定策略步决定（更可比）；
# 早于释放时刻的自然断梗仍在逐回合记录里单列。
STEM_DECLARED_BREAK_FORCE_N: Final = 60.0
STEM_MAXIMUM_FORCE_N: Final = 100.0
STEM_MAXIMUM_TORQUE_NM: Final = 0.30
STEM_LINEAR_NATURAL_FREQUENCY_RAD_S: Final = 20.0
STEM_LINEAR_DAMPING_RATIO: Final = 1.0
STEM_ANGULAR_NATURAL_FREQUENCY_RAD_S: Final = 15.0
STEM_ANGULAR_DAMPING_RATIO: Final = 1.0


# ------------------------------------------------------------------- 判据
CONTACT_FORCE_N: Final = 0.08
MIN_CONTACT_FINGERS: Final = 2
FINGER_NAMES: Final = ("thumb", "index", "middle", "ring", "little")
FINGER_SOFT_LOW_N: Final = (0.20, 0.15, 0.12, 0.08, 0.05)
FINGER_SOFT_HIGH_N: Final = (1.80, 1.50, 1.30, 1.20, 1.00)
FINGER_HARD_HIGH_N: Final = (15.00, 15.00, 15.00, 15.00, 15.00)
HOLD_MAX_DRIFT_M: Final = 0.010
HOLD_MAX_RELATIVE_SPEED_M_S: Final = 0.05
DROP_HEIGHT_TOLERANCE_M: Final = 0.080
DROP_MAX_DRIFT_M: Final = 0.120
PENETRATION_LIMIT_M: Final = 0.005
COLLIDER_OVERLAP_LIMIT_M: Final = 0.012
JOINT_LIMIT_MARGIN_RAD: Final = 0.010
SELF_COLLISION_PENETRATION_LIMIT_M: Final = 0.010
INSTABILITY_MAX_JOINT_VELOCITY_RAD_S: Final = 4.0
INSTABILITY_MAX_FRUIT_SPEED_M_S: Final = 5.0
MIN_HEIGHT_GAIN_M: Final = 0.030
REQUIRED_HOLD_S: Final = 3.0

# 既有拇指/食指代理只覆盖一对刚体，不足以支撑新任务的穿透检查；这里显式
# 声明一组候选对，运行时按实际碰撞体存在性过滤并记录生效子集。
SELF_COLLISION_PAIRS: Final = (
    ("thumb_link4", "index_link3"),
    ("thumb_link4", "middle_link3"),
    ("index_link3", "middle_link3"),
    ("index_link3", "ring_link3"),
    ("middle_link3", "ring_link3"),
)


class FixedFruitGraspError(RuntimeError):
    """本任务配置或运行前提不成立。"""


class UnknownStageError(ValueError):
    """stage 不在声明集合内。"""


class LiftStageUnavailable(FixedFruitGraspError):
    """lift 阶段需要一根明确声明的受驱动升降关节，本版本未实现。"""


class OpenHandActionUnavailable(FixedFruitGraspError):
    """14D 流形无法精确复现 reset 的真实 neutral 关节目标。"""


# --------------------------------------------------------------- 阶段规格
@dataclass(frozen=True)
class StageSpec:
    """一个课程阶段的固定判据。阶段之间的成功不得混算。"""

    stage: str
    success_criterion: str
    support_release_step: int
    required_unsupported_hold_s: float
    min_height_gain_m: float
    requires_lift_program: bool
    counts_as_goal: bool

    @property
    def required_unsupported_hold_steps(self) -> int:
        return int(round(self.required_unsupported_hold_s / POLICY_DT_S))

    @property
    def required_unsupported_hold_substeps(self) -> int:
        """真实物理子步阈值：策略步阈值 × decimation（1 s = 20 × 6 = 120）。"""

        return int(self.required_unsupported_hold_steps) * int(DECIMATION)

    @property
    def support_release_time_s(self) -> float:
        return self.support_release_step * POLICY_DT_S

    @property
    def success_requires_height_gain(self) -> bool:
        """该阶段的成功判据是否包含高度增量要求。

        holding-only 阶段（contact/hold）的成功只要求“真实多指接触 + 固定时刻
        撤支撑后连续保持”，不含任何高度增量；30 mm 是需要声明式受驱动升降关节
        的目标，只在 lift 阶段进入判据。
        """

        return self.min_height_gain_m > 0.0

    @property
    def goal_status(self) -> str:
        """该阶段相对于 30 mm+3 s 总目标的状态声明。"""

        return (
            GOAL_STATUS_LIFT_REQUIRED
            if self.counts_as_goal
            else GOAL_STATUS_LIFT_NOT_IMPLEMENTED
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "success_criterion": self.success_criterion,
            "support_release_step": int(self.support_release_step),
            "support_release_time_s": float(self.support_release_time_s),
            "required_unsupported_hold_s": float(self.required_unsupported_hold_s),
            "required_unsupported_hold_steps": int(self.required_unsupported_hold_steps),
            "required_unsupported_hold_substeps": int(
                self.required_unsupported_hold_substeps
            ),
            "min_height_gain_m": float(self.min_height_gain_m),
            "requires_lift_program": bool(self.requires_lift_program),
            "counts_as_goal": bool(self.counts_as_goal),
            "success_requires_height_gain": bool(self.success_requires_height_gain),
            "goal_status": self.goal_status,
        }


STAGE_SPECS: Final = {
    # 接触建立课程：较晚解除支撑，独立成功只证明“真实多指接触 + 无支撑保持
    # 1 s”。它不含任何高度要求（min_height_gain_m=0.0），也不计入 30 mm+3 s
    # 目标（counts_as_goal=False）。
    "contact": StageSpec(
        stage="contact",
        success_criterion="contact_grasp_hold_1s_without_support",
        support_release_step=60,
        required_unsupported_hold_s=1.0,
        min_height_gain_m=0.0,
        requires_lift_program=False,
        counts_as_goal=False,
    ),
    # 保持阶段 = 本轮交付的独立任务：固定时刻解除支撑，要求真实多指接触下
    # 连续 3 s 无外部支撑保持。同样不含高度要求，也不计入 30 mm+3 s 目标：
    # holding-only 是独立任务名（指标 unsupported_hold_s），不是提举成功。
    "hold": StageSpec(
        stage="hold",
        success_criterion="unsupported_hold_3s_with_real_finger_contact",
        support_release_step=40,
        required_unsupported_hold_s=REQUIRED_HOLD_S,
        min_height_gain_m=0.0,
        requires_lift_program=False,
        counts_as_goal=False,
    ),
    # 提举阶段：需要声明的受驱动升降关节，本版本未实现。
    "lift": StageSpec(
        stage="lift",
        success_criterion="declared_low_speed_lift_35mm_then_unsupported_hold_3s",
        support_release_step=40,
        required_unsupported_hold_s=REQUIRED_HOLD_S,
        min_height_gain_m=MIN_HEIGHT_GAIN_M,
        requires_lift_program=True,
        counts_as_goal=True,
    ),
}
STAGES: Final = tuple(STAGE_SPECS)
DEFAULT_STAGE: Final = "hold"

LIFT_STAGE_STATUS: Final = "NOT_IMPLEMENTED_DECLARED_ONLY"
GOAL_STATUS_LIFT_NOT_IMPLEMENTED: Final = "GOAL_NOT_REACHED_LIFT_PROGRAM_NOT_IMPLEMENTED"
GOAL_STATUS_LIFT_REQUIRED: Final = "GOAL_REQUIRES_DECLARED_LIFT_PROGRAM"


def stage_spec(stage: str) -> StageSpec:
    """返回阶段的固定判据；未知阶段 fail-closed。"""

    key = str(stage).strip().lower()
    if key not in STAGE_SPECS:
        raise UnknownStageError(
            f"未知 stage {stage!r}；允许 {STAGES}"
        )
    return STAGE_SPECS[key]


def lift_stage_available() -> bool:
    """提举阶段当前是否可用（需要声明的受驱动升降关节）。"""

    return False


# ------------------------------------------------------- 声明式提举轨迹（未接线）
@dataclass(frozen=True)
class LiftProgramSpec:
    """固定低速提举程序的声明式定义。

    该程序只能由一个明确声明的受驱动关节执行；禁止逐帧传送手根或果实。
    本版本没有该关节，因此这里只提供可复核的参数与轨迹，供主线程下一步
    接线或在 USD 上新增关节时复用。
    """

    axis: str = "world_z"
    travel_m: float = 0.035
    speed_m_s: float = 0.010
    min_height_gain_m: float = MIN_HEIGHT_GAIN_M
    hold_s: float = REQUIRED_HOLD_S

    @property
    def duration_steps(self) -> int:
        return int(math.ceil(self.travel_m / (self.speed_m_s * POLICY_DT_S)))

    @property
    def duration_s(self) -> float:
        return self.travel_m / self.speed_m_s

    def height_at_step(self, step: int) -> float:
        """第 ``step`` 个策略步的目标高度增量（线性、限速、不超程）。"""

        return min(self.travel_m, self.speed_m_s * max(int(step), 0) * POLICY_DT_S)

    def to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "travel_m": float(self.travel_m),
            "speed_m_s": float(self.speed_m_s),
            "duration_s": float(self.duration_s),
            "duration_steps": int(self.duration_steps),
            "min_height_gain_m": float(self.min_height_gain_m),
            "hold_s": float(self.hold_s),
            "executed": False,
        }


LIFT_PROGRAM: Final = LiftProgramSpec()


# --------------------------------------------------------------- 终止原因
REASON_CODES: Final = {
    0: "running",
    1: "success",
    2: "timeout",
    3: "drop",
    4: "overload",
    5: "nonfinite",
    6: "penetration",
    7: "joint_limit",
    8: "self_collision",
    9: "stem_fracture",
    10: "instability",
    11: "joint_effort",
}
REASON_CODES_BY_NAME: Final = {name: code for code, name in REASON_CODES.items()}
SUCCESS_REASON: Final = "success"
TIMEOUT_REASON: Final = "timeout"


def reason_name(code: int) -> str:
    """把终止原因码翻译为稳定字符串（未知码 fail-closed）。"""

    value = int(code)
    if value not in REASON_CODES:
        raise ValueError(f"未知终止原因码 {code!r}；允许 {sorted(REASON_CODES)}")
    return REASON_CODES[value]


# ------------------------------------------------------- 逐回合记录 schema
EPISODE_RECORD_FIELDS: Final = (
    "schema_version",
    "task_id",
    "task_version",
    "stage",
    "success_criterion",
    "env_id",
    "episode_id",
    "success",
    "goal_success",
    "goal_status",
    "reason",
    "reason_code",
    "terminated",
    "truncated",
    "steps",
    "sim_time_s",
    "contact_fraction",
    "finger_contact_fraction",
    "required_contact_fraction",
    "unsupported_contact_fraction",
    "any_contact_fraction",
    "mean_contact_fingers",
    "max_contact_fingers",
    "palm_contact_fraction",
    "palm_only_fraction",
    "forbidden_contact_steps",
    "peak_finger_force_n",
    "mean_finger_force_n",
    "support_release_step",
    "support_release_time_s",
    "support_released",
    "stem_fracture_before_release",
    "max_unsupported_hold_s",
    "final_unsupported_hold_s",
    "hold_steps_required",
    "max_physical_hold_streak_substeps",
    "final_physical_hold_streak_substeps",
    "physical_hold_streak_required_substeps",
    "max_unsupported_hold_s_from_physical_substeps",
    "physical_hold_streak_semantics",
    "physical_substep_hard_fault_flags",
    "max_height_gain_m",
    "final_height_gain_m",
    "height_gain_sufficient",
    "min_height_gain_m",
    "goal_min_height_gain_m",
    "stage_success_requires_height_gain",
    "stage_requires_lift_program",
    "stage_is_goal",
    "max_center_drift_m",
    "max_relative_speed_m_s",
    "max_joint_limit_excursion_rad",
    "max_applied_effort_nm",
    "min_self_collision_clearance_m",
    "min_collider_distance_m",
    "max_penetration_depth_m",
    "instability_step",
    "first_fault_step",
    "immediate_failure",
    "failure_phase",
    "stem_fracture",
    "overload",
    "nonfinite",
    "penetration",
    "joint_limit",
    "self_collision",
    "instability",
    "joint_effort",
    "action_effect_ok",
    "open_hand_baseline",
    "open_hand_fraction",
)


def json_safe(value: Any) -> Any:
    """把标量/序列规范化成 JSON-safe 值（不接受张量或对象）。"""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    raise TypeError(f"不是 JSON-safe 的标量/序列：{type(value).__name__}")


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按逐回合记录汇总完整分母与终止原因（供评估脚本直接使用）。"""

    count = len(rows)
    reasons = {name: 0 for name in REASON_CODES.values() if name != "running"}
    stages: dict[str, int] = {}
    success = 0
    goal_success = 0
    hold_seconds: list[float] = []
    contact_fraction: list[float] = []
    height_gain: list[float] = []
    for row in rows:
        reason = str(row.get("reason", ""))
        if reason in reasons:
            reasons[reason] += 1
        stage = str(row.get("stage", ""))
        stages[stage] = stages.get(stage, 0) + 1
        success += int(bool(row.get("success", False)))
        goal_success += int(bool(row.get("goal_success", False)))
        hold_seconds.append(float(row.get("max_unsupported_hold_s", 0.0)))
        contact_fraction.append(float(row.get("contact_fraction", 0.0)))
        height_gain.append(float(row.get("max_height_gain_m", 0.0)))

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "records": count,
        "success_count": success,
        "success_rate": float(success / count) if count else 0.0,
        "goal_success_count": goal_success,
        "goal_status": GOAL_STATUS_LIFT_NOT_IMPLEMENTED,
        "reason_counts": reasons,
        "stage_counts": stages,
        "mean_max_unsupported_hold_s": _mean(hold_seconds),
        "best_max_unsupported_hold_s": max(hold_seconds) if hold_seconds else 0.0,
        "mean_contact_fraction": _mean(contact_fraction),
        "best_max_height_gain_m": max(height_gain) if height_gain else 0.0,
    }


__all__ = [
    "ACTION_DIM",
    "ACTOR_OBSERVATION_DIM",
    "ACTUATOR_ARMATURE_KG_M2",
    "ACTUATOR_EFFORT_LIMIT_NM",
    "ACTUATOR_VELOCITY_LIMIT_RAD_S",
    "COLLIDER_OVERLAP_LIMIT_M",
    "CONTACT_FORCE_N",
    "CRITIC_STATE_DIM",
    "DECIMATION",
    "DEFAULT_STAGE",
    "DROP_HEIGHT_TOLERANCE_M",
    "DROP_MAX_DRIFT_M",
    "EPISODE_RECORD_FIELDS",
    "EPISODE_STEPS",
    "FINGER_HARD_HIGH_N",
    "FINGER_NAMES",
    "FINGER_SOFT_HIGH_N",
    "FINGER_SOFT_LOW_N",
    "FRUIT_AXES_M",
    "FRUIT_DENSITY_KG_M3",
    "FRUIT_DIAMETER_M",
    "FRUIT_DYNAMIC_FRICTION",
    "FRUIT_FRICTION_COMBINE_MODE",
    "FRUIT_MASS_KG",
    "FRUIT_MASS_TOLERANCE_KG",
    "FRUIT_RADIUS_M",
    "FRUIT_RESTITUTION",
    "FRUIT_STATIC_FRICTION",
    "FixedFruitGraspError",
    "GOAL_STATUS_LIFT_NOT_IMPLEMENTED",
    "GOAL_STATUS_LIFT_REQUIRED",
    "HOLD_MAX_DRIFT_M",
    "HOLD_MAX_RELATIVE_SPEED_M_S",
    "INSTABILITY_MAX_FRUIT_SPEED_M_S",
    "INSTABILITY_MAX_JOINT_VELOCITY_RAD_S",
    "JOINT_LIMIT_MARGIN_RAD",
    "LIFT_PROGRAM",
    "LIFT_STAGE_STATUS",
    "LiftProgramSpec",
    "LiftStageUnavailable",
    "MIN_CONTACT_FINGERS",
    "MIN_HEIGHT_GAIN_M",
    "MIMIC_RELATIONS",
    "OpenHandActionUnavailable",
    "PENETRATION_LIMIT_M",
    "PHYSICAL_HOLD_ANCHOR_SEMANTICS",
    "PHYSICAL_HOLD_GUARD_FREQUENCY",
    "PHYSICAL_HOLD_HARD_FAULT_PERSISTENCE",
    "PHYSICAL_HOLD_MISSING_DATA_POLICY",
    "PHYSICAL_HOLD_STREAK_SEMANTICS",
    "PHYSICAL_HOLD_SUCCESS_SEMANTICS",
    "POLICY_DT_S",
    "POLICY_RATE_HZ",
    "REASON_CODES",
    "REASON_CODES_BY_NAME",
    "RECORD_SCHEMA_VERSION",
    "REQUIRED_HOLD_S",
    "SELF_COLLISION_PAIRS",
    "SELF_COLLISION_PENETRATION_LIMIT_M",
    "SIM_DT_S",
    "STAGES",
    "STAGE_SPECS",
    "STEM_DECLARED_BREAK_FORCE_N",
    "STEM_MAXIMUM_FORCE_N",
    "STEM_MAXIMUM_TORQUE_NM",
    "SUCCESS_REASON",
    "StageSpec",
    "TASK_ID",
    "TASK_VERSION",
    "TIMEOUT_REASON",
    "UnknownStageError",
    "json_safe",
    "lift_stage_available",
    "physical_hold_semantics",
    "reason_name",
    "stage_spec",
    "summarize_records",
]
