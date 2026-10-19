#!/usr/bin/env python3
"""直接在已启动的 Isaac App 内运行小果捏取 PPO。

当前 Isaac Lab 统一训练入口会在 SimulationApp 前导入 USD 模块并在本机崩溃；
本入口保留相同的 RSL-RL 算法和配置，但严格先启动 App，再导入任务与 Runner。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-manifest", type=Path, default=None)
parser.add_argument("--support-mode", choices=("audit_release", "natural_detachment"), default="audit_release")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log-dir", type=Path, required=True)
parser.add_argument("--resume", type=Path, default=None, help="加载同一 14D 合同检查点并继续 PPO。")
parser.add_argument(
    "--actor-init",
    type=Path,
    default=None,
    help="仅继承 Actor 的 MLP 与观测归一化；Critic、优化器和探索方差均重新初始化。",
)
parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
parser.add_argument("--run-name", default="native_train")
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--entropy-coef", type=float, default=None)
parser.add_argument("--fruit-axis-min-mm", type=float, default=None)
parser.add_argument("--fruit-axis-max-mm", type=float, default=None)
parser.add_argument(
    "--diameter-bins-mm",
    default=None,
    help="按等效直径分桶，格式如 62:64,64:66；与 fruit-axis-* 互斥。",
)
parser.add_argument(
    "--diameter-bin-weights",
    default=None,
    help="分桶环境配额权重，格式如 0.4,0.2,0.2,0.2。",
)
parser.add_argument(
    "--geometry-cohort-id",
    default="default",
    help="与 seed 共同确定 prestartup 独立几何 cohort 的稳定标识。",
)
parser.add_argument("--offset-x-mm", type=float, default=None)
parser.add_argument("--offset-y-mm", type=float, default=None)
parser.add_argument("--offset-z-mm", type=float, default=None)
parser.add_argument(
    "--center-local-mm",
    nargs=3,
    type=float,
    default=None,
    metavar=("X", "Y", "Z"),
    help="覆盖左手抓取基坐标中的固定果心（mm）；用于几何候选的物理验证。",
)
parser.add_argument(
    "--palm-fruit-collision-filter",
    action="store_true",
    help="诊断模式：仅过滤 base_link↔Tomato 碰撞并忽略其粗糙 AABB；不用于最终实机/发布合同。",
)
parser.add_argument(
    "--geometric-contact-proxy",
    action="store_true",
    help="仅离线实验：用指尖到果面的距离生成代理接触力；日志保留原始 PhysX 力。",
)
parser.add_argument("--geometric-contact-proxy-band-mm", type=float, default=None)
parser.add_argument(
    "--guided-residual",
    action="store_true",
    help="启用结构化 compliant prior；PPO 只学习其上的小幅残差。",
)
parser.add_argument(
    "--geometry-prior",
    action="store_true",
    help="guided-residual 中启用经物理扫描验证的三指几何弱先验。",
)
parser.add_argument(
    "--geometry-prior-scale",
    type=float,
    default=None,
    help="覆盖几何弱先验幅度；默认保留环境合同值。",
)
parser.add_argument("--geometry-prior-size-adaptive", action="store_true", help="按当前果实半轴将几何先验从基线连续缩放到上限")
parser.add_argument("--geometry-prior-max-scale", type=float, default=None)
parser.add_argument(
    "--geometry-prior-values",
    default=None,
    help="覆盖 14D 几何先验向量；用于按指消融，格式为 14 个逗号分隔值。",
)
parser.add_argument(
    "--guide-action-values",
    default=None,
    help="guided-residual 的 14D 可行工作区弱先验；不改变成功判据，格式为 14 个逗号分隔值。",
)
parser.add_argument(
    "--geometry-prior-after-contact",
    action="store_true",
    help="每根手指接触后才施加对应几何弱先验。",
)
parser.add_argument(
    "--guide-feedback",
    action="store_true",
    help="启用仅基于上一帧真实指尖力的低层回撤反馈。",
)
parser.add_argument(
    "--adaptive-closure",
    action="store_true",
    help="启用按上一帧真实接触力渐进闭合的低层反射。",
)
parser.add_argument(
    "--force-balance",
    action="store_true",
    help="guided-residual 中启用带死区/限幅的三指压力平衡闭环。",
)
parser.add_argument(
    "--force-servo",
    action="store_true",
    help="guided-residual 中启用每指受力误差双通道。",
)
parser.add_argument("--force-servo-seek-gain", type=float, default=None)
parser.add_argument("--force-servo-band-seek-gain", type=float, default=None)
parser.add_argument("--force-servo-target-gain", type=float, default=None)
parser.add_argument("--force-servo-retract-gain", type=float, default=None)
parser.add_argument("--force-servo-alpha", type=float, default=None)
parser.add_argument("--force-servo-max-offset", type=float, default=None)
parser.add_argument("--force-servo-warning-fraction", type=float, default=None)
parser.add_argument("--force-servo-rate-gain", type=float, default=None)
parser.add_argument("--force-servo-rate-deadband-n", type=float, default=None)
parser.add_argument("--force-servo-rate-scale-n", type=float, default=None)
parser.add_argument("--direct-force-projection", action="store_true", help="启用 canonical 16D 目标层逐指受力投影")
parser.add_argument("--direct-force-warning-fraction", type=float, default=None)
parser.add_argument("--direct-force-high-gain", type=float, default=None)
parser.add_argument(
    "--direct-force-high-gain-by-digit",
    default=None,
    help="可选的拇指/食指/中指逐指高力回撤增益，格式为 3 个非负浮点数。",
)
parser.add_argument("--direct-force-low-gain", type=float, default=None)
parser.add_argument("--direct-force-alpha", type=float, default=None)
parser.add_argument("--direct-force-max-open", type=float, default=None)
parser.add_argument("--direct-force-root-fraction", type=float, default=None)
parser.add_argument("--direct-force-rate-gain", type=float, default=None)
parser.add_argument("--direct-force-rate-deadband-n", type=float, default=None)
parser.add_argument("--direct-force-rate-scale-n", type=float, default=None)
parser.add_argument("--force-servo-seek-direction", default=None, help="3 个逗号分隔的方向符号")
parser.add_argument("--force-servo-retract-direction", default=None, help="3 个逗号分隔的方向符号")
parser.add_argument("--phase-controller", action="store_true", help="启用逐指 SEARCH/CLOSE/HOLD/RETRACT 相位控制器")
parser.add_argument("--phase-search-step", type=float, default=None)
parser.add_argument("--phase-close-step", type=float, default=None)
parser.add_argument("--phase-retract-step", type=float, default=None)
parser.add_argument("--phase-offset-alpha", type=float, default=None)
parser.add_argument("--phase-max-offset", type=float, default=None)
parser.add_argument("--phase-root-fraction", type=float, default=None)
parser.add_argument("--phase-low-delay-steps", type=int, default=None)
parser.add_argument("--phase-low-ramp-steps", type=int, default=None)
parser.add_argument("--phase-hysteresis-n", type=float, default=None)
parser.add_argument("--phase-geometry", action="store_true", help="按指尖到椭球表面的有符号间隙进行寻触")
parser.add_argument("--phase-geometry-step", type=float, default=None)
parser.add_argument("--phase-geometry-gap-scale-m", type=float, default=None)
parser.add_argument("--phase-target-controller", action="store_true", help="启用逐指压力设定点有界积分控制")
parser.add_argument("--phase-target-force-n", default=None, help="3 个逗号分隔的目标力（N）")
parser.add_argument("--phase-target-step", type=float, default=None)
parser.add_argument("--phase-target-alpha", type=float, default=None)
parser.add_argument("--phase-target-deadband-n", type=float, default=None)
parser.add_argument("--phase-target-error-scale-n", default=None, help="3 个逗号分隔的误差归一化尺度（N）")
parser.add_argument("--phase-target-raw-mix", type=float, default=None)
parser.add_argument(
    "--force-gate-start-fraction",
    type=float,
    default=None,
    help="从软压力带的该比例开始收缩 residual 权限；省略保持历史门控。",
)
parser.add_argument(
    "--approach-speed",
    type=float,
    default=None,
    help="guided-residual temporal filter 的接近速度上限。",
)
parser.add_argument(
    "--projection-guard-fraction",
    type=float,
    default=None,
    help="在 soft_high 的该比例处启动目标回撤投影。",
)
parser.add_argument(
    "--residual-scale",
    type=float,
    default=None,
    help="guided-residual 的动作残差缩放，默认使用环境安全默认值。",
)
parser.add_argument(
    "--distal-residual-scale",
    type=float,
    default=None,
    help="guided-residual 新增远端维度的独立残差缩放；省略则沿用 residual-scale。",
)
parser.add_argument(
    "--init-std",
    type=float,
    default=None,
    help="覆盖 Actor 初始高斯标准差，适合结构化残差训练。",
)
parser.add_argument(
    "--p1-mean-init",
    action="store_true",
    help="仅用 P1 可达性动作初始化 Actor 均值；不是执行脚本，也不冻结策略。",
)
parser.add_argument(
    "--zero-residual-init",
    action="store_true",
    help="将 Actor 输出层初始化为零；仅适用于 guided-residual，使策略从 compliant prior 的零残差起步。",
)
parser.add_argument(
    "--precontact-residual-fraction",
    type=float,
    default=None,
    help="guided-residual 接触前保留的 PPO 残差比例（0–1）；用于学习随机果位的安全微调。",
)
parser.add_argument(
    "--residual-require-all-contact",
    action="store_true",
    help="仅在拇指/食指/中指均已接触后开放 PPO 残差。",
)
parser.add_argument(
    "--residual-trust-region",
    action="store_true",
    help="启用按三指实时压力限制 PPO 残差的局部 trust-region。",
)
parser.add_argument("--residual-trust-warning-fraction", type=float, default=None)
parser.add_argument("--residual-trust-in-band-scale", type=float, default=None)
parser.add_argument(
    "--freeze-required-distal",
    action="store_true",
    help="诊断消融：冻结三根必需手指远端 canonical 关节（不用于发布训练）。",
)
parser.add_argument("--hold-speed", type=float, default=None, help="保持段动作域速度上限。")
parser.add_argument("--hold-acceleration", type=float, default=None, help="保持段动作域加速度上限。")
parser.add_argument("--natural-frequency", type=float, default=None, help="临界阻尼滤波器固有频率（Hz）。")
parser.add_argument("--hand-damping", type=float, default=None, help="覆盖两个手部位置驱动组的阻尼（仅仿真）。")
parser.add_argument("--hand-stiffness", type=float, default=None, help="覆盖两个手部位置驱动组的刚度（仅仿真）。")
parser.add_argument(
    "--hand-armature",
    type=float,
    default=None,
    help="覆盖两个手部位置驱动组的关节等效惯量（仅仿真；用于抑制接触瞬态）。",
)
parser.add_argument(
    "--hand-viscous-friction",
    type=float,
    default=None,
    help="覆盖两个手部位置驱动组的粘性关节摩擦系数（仅仿真；用于阻尼）。",
)
parser.add_argument("--geometry-approach", action="store_true", help="接触前按实际指尖—椭球间隙连续寻触（仅仿真）。")
parser.add_argument("--geometry-approach-target-gap-mm", type=float, default=None)
parser.add_argument("--geometry-approach-scale-mm", type=float, default=None)
parser.add_argument("--geometry-approach-gain", type=float, default=None)
parser.add_argument("--geometry-approach-max-offset", type=float, default=None)
parser.add_argument("--geometry-approach-alpha", type=float, default=None)
parser.add_argument("--external-forces-every-iteration", action="store_true", help="PhysX 每次位置迭代施加外力（仅仿真）。")
parser.add_argument("--enhanced-determinism", action="store_true", help="启用 PhysX 增强确定性（仅仿真）。")
parser.add_argument("--robot-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--fruit-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--fruit-contact-offset", type=float, default=None)
parser.add_argument("--fruit-rest-offset", type=float, default=None)
parser.add_argument("--solver-position-iterations", type=int, default=None, help="机器人 PhysX 位置求解迭代次数（仅仿真）。")
parser.add_argument("--solver-velocity-iterations", type=int, default=None, help="机器人 PhysX 速度求解迭代次数（仅仿真）。")
parser.add_argument("--solver-type", type=int, choices=(0, 1), default=None, help="PhysX 求解器：0=PGS，1=TGS（仅仿真）。")
parser.add_argument("--hand-velocity-limit", type=float, default=None, help="覆盖手部位置驱动速度上限（rad/s，仅仿真）。")
parser.add_argument(
    "--explicit-hand-pd",
    action="store_true",
    help="将两组 G20 手部执行器切换为显式 IdealPD 力矩控制（仅仿真；默认关闭）。",
)
parser.add_argument(
    "--hand-effort-limit",
    type=float,
    default=None,
    help="覆盖左手两个执行器的仿真努力上限；仅用于动力学敏感性候选。",
)
parser.add_argument("--hold-joint-target-speed", type=float, default=None, help="保持段 canonical 关节目标角速度上限（rad/s）。")
parser.add_argument("--hold-joint-target-open-speed", type=float, default=None, help="保持段沿预抓取方向回撤的 canonical 关节目标角速度上限（rad/s）。")
parser.add_argument("--hold-hand-velocity-limit", type=float, default=None, help="仅安全保持段的 PhysX 手部关节速度上限（rad/s）；压力接近软上限时自动恢复原上限。")
parser.add_argument("--hold-hand-velocity-limit-by-joint", default=None, help="安全保持段按 canonical 16 个主动轴设置 PhysX 速度上限，逗号分隔（rad/s）。")
parser.add_argument("--contact-hand-velocity-limit", type=float, default=None, help="仅已接触的必需手指逐指 PhysX 速度上限（rad/s，仅仿真）。")
parser.add_argument("--contact-hand-stiffness", type=float, default=None, help="仅已接触的必需手指逐指 PhysX 刚度降档（仅仿真）。")
parser.add_argument("--contact-hand-damping", type=float, default=None, help="仅已接触的必需手指逐指 PhysX 阻尼降档（仅仿真）。")
parser.add_argument("--contact-hand-target-speed", type=float, default=None, help="仅已接触的必需手指逐指 canonical 目标斜率上限（rad/s，仅仿真）。")
parser.add_argument("--hold-hand-stiffness", type=float, default=None, help="仅安全保持段的 PhysX 手部关节刚度；压力接近软上限时自动恢复原刚度。")
parser.add_argument("--hold-hand-stiffness-by-joint", default=None, help="仅安全保持段按 canonical 16 个主动轴设置刚度，逗号分隔。")
parser.add_argument("--hold-hand-damping", type=float, default=None, help="仅安全保持段的 PhysX 手部关节阻尼；压力接近软上限时自动恢复原阻尼。")
parser.add_argument("--hold-hand-damping-by-joint", default=None, help="仅安全保持段按 canonical 16 个主动轴设置阻尼，逗号分隔。")
parser.add_argument("--hold-joint-velocity-target", type=float, default=None, help="保持段位置驱动的有界关节速度参考（rad/s，仅仿真）。")
parser.add_argument(
    "--hold-zero-velocity-target",
    action="store_true",
    help="保持段将手部关节速度参考设为零，仅保留位置目标产生的接触力（仅仿真；可回退）。",
)
parser.add_argument("--hold-hand-effort-limit", type=float, default=None, help="仅安全保持段的 PhysX 手部驱动努力上限；过压时恢复默认上限（仿真）。")
parser.add_argument("--hold-joint-target-error-limit", type=float, default=None, help="安全保持段目标相对实际关节位置的误差窗（rad，仅仿真）。")
parser.add_argument("--hold-joint-target-error-limit-by-joint", default=None, help="安全保持段按 canonical 16 个主动轴设置目标误差窗，逗号分隔（rad，仅仿真）。")
parser.add_argument("--hold-joint-target-anchor-fraction", type=float, default=None, help="安全保持段将目标锚定到当前实测关节位置的比例（0–1，仅仿真）。")
parser.add_argument("--hold-joint-target-anchor-fraction-by-joint", default=None, help="安全保持段按 canonical 16 个主动轴设置目标锚定比例，逗号分隔。")
parser.add_argument("--hold-target-filter-alpha", type=float, default=None, help="安全保持段 canonical 目标低通系数（0<alpha≤1；仅仿真）。")
parser.add_argument("--hold-velocity-damping-effort", type=float, default=None, help="安全保持段按实测速度施加反向努力的增益（N·m/(rad/s)，仅仿真）。")
parser.add_argument("--hold-effort-control", action="store_true", help="保持段使用当前位置锚定+真实力反馈有界努力通道（仅仿真）。")
parser.add_argument("--hold-effort-stiffness", type=float, default=None, help="保持努力通道的隐式位置刚度（可为 0）。")
parser.add_argument("--hold-effort-damping", type=float, default=None, help="保持努力通道的隐式速度阻尼（可为 0）。")
parser.add_argument("--hold-effort-max", type=float, default=None, help="保持努力通道每关节最大努力。")
parser.add_argument("--hold-effort-in-band", type=float, default=None, help="压力带内每关节保持努力。")
parser.add_argument("--hold-effort-low-fraction", type=float, default=None, help="低压补进努力相对最大努力的比例。")
parser.add_argument("--hold-effort-high-fraction", type=float, default=None, help="高压回撤努力相对最大努力的比例。")
parser.add_argument("--hold-force-impedance", action="store_true", help="保持段使用当前关节位置+实时力误差的小角度阻抗目标（仅仿真）。")
parser.add_argument(
    "--hold-current-impedance",
    action="store_true",
    help="保持段以当前实测关节为锚点，用逐指小努力和真实力反馈维持接触（仅仿真）。",
)
parser.add_argument("--hold-impedance-effort-by-digit", default=None, help="拇指/食指/中指保持努力（3 个 N·m 值）。")
parser.add_argument("--hold-impedance-force-target", default=None, help="保持努力的逐指目标力（3 个 N 值）。")
parser.add_argument("--hold-impedance-force-gain", type=float, default=None)
parser.add_argument("--hold-impedance-start-fraction", type=float, default=None)
parser.add_argument("--hold-impedance-ramp-substeps", type=int, default=None)
parser.add_argument("--hold-impedance-damping", type=float, default=None)
parser.add_argument(
    "--position-target-interpolation",
    type=int,
    default=None,
    help="每个策略步内将手部位置目标分摊到多少个 PhysX 子步（默认 1，不改变历史合同）。",
)
parser.add_argument(
    "--manual-decimation-control",
    action="store_true",
    help="仅在 PhysX 后端显式启用逐物理子步目标更新；必须与位置目标插值配合使用。",
)
parser.add_argument("--joint-smoothness-weight", type=float, default=None, help="保持段实际关节速度/加速度/jerk 的奖励惩罚权重。")
parser.add_argument("--proximity-speed", action="store_true", help="接触前按指尖到椭球表面的间隙逐指降速。")
parser.add_argument("--proximity-speed-start-mm", type=float, default=None, help="接触前降速起始间隙（mm）。")
parser.add_argument("--proximity-speed-floor", type=float, default=None, help="接触前逐指速度比例下限。")
parser.add_argument("--proximity-speed-global", action="store_true", help="诊断：同时按最慢必需指缩放全局动作。")
parser.add_argument("--geometry-contact-projection", action="store_true", help="按指尖几何穿透量做逐指接触回撤保护。")
parser.add_argument("--geometry-contact-margin-mm", type=float, default=None, help="几何接触投影归一化穿透余量（mm）。")
parser.add_argument("--geometry-contact-gain", type=float, default=None, help="几何接触投影增益。")
parser.add_argument("--geometry-contact-max-open", type=float, default=None, help="几何接触投影最大回撤比例。")
parser.add_argument("--geometry-contact-root-fraction", type=float, default=None, help="几何接触投影作用于根部的比例。")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import importlib.metadata as metadata
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.seed import configure_seed
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from research.small_fruit_pinch_v1.source.agents import SmallFruitPinchPPORunnerCfg
from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)


P1_MEAN_ACTION = (
    0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345,
    -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0,
)


def _initialize_actor_mean_from_p1(runner: OnPolicyRunner) -> None:
    """用已验证可达动作做初始均值，不改变 PPO 的可学习性或探索方差。"""

    policy = runner.alg.get_policy()
    output = next(
        module for module in reversed(policy.mlp) if isinstance(module, torch.nn.Linear)
    )
    target = torch.tensor(P1_MEAN_ACTION, device=output.bias.device, dtype=output.bias.dtype)
    if tuple(output.bias.shape) != tuple(target.shape):
        raise RuntimeError("P1 初始动作维度与 Actor 输出不一致")
    with torch.no_grad():
        output.bias.copy_(target)


def _initialize_actor_residual_zero(runner: OnPolicyRunner) -> None:
    """让结构化残差策略从零均值起步而不改变观测归一化。"""

    policy = runner.alg.get_policy()
    output = next(
        module for module in reversed(policy.mlp) if isinstance(module, torch.nn.Linear)
    )
    with torch.no_grad():
        output.weight.zero_()
        output.bias.zero_()


def _initialize_actor_from_checkpoint(runner: OnPolicyRunner, checkpoint: Path) -> None:
    """继承已学到的动作表征，而不把旧奖励下的价值/动量带入新阶段。"""

    payload = torch.load(checkpoint, map_location=runner.device, weights_only=False)
    state = payload.get("actor_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("检查点不包含 actor_state_dict")
    # log_std_param 是探索状态，必须随新奖励从配置的保守初值重新开始。
    transferable = {
        key: value for key, value in state.items()
        if key.startswith("mlp.") or key.startswith("obs_normalizer.")
    }
    missing, unexpected = runner.alg.get_policy().load_state_dict(transferable, strict=False)
    allowed_missing = {"distribution.log_std_param"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"Actor 初始化合同不匹配: missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    if args.num_envs <= 0 or args.iterations <= 0:
        raise ValueError("num-envs 与 iterations 必须为正数")
    for checkpoint in (args.resume, args.actor_init):
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.resume is not None and args.actor_init is not None:
        raise ValueError("--resume 与 --actor-init 不可同时使用")
    if args.zero_residual_init and not args.guided_residual:
        raise ValueError("--zero-residual-init 只能与 guided-residual 一起使用")
    if args.zero_residual_init and args.p1_mean_init:
        raise ValueError("--zero-residual-init 与 --p1-mean-init 不可同时使用")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    configure_seed(args.seed, True)
    env_cfg = NativeSmallFruitPinchCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.verification_release_support = args.support_mode == "audit_release"
    if args.hand_effort_limit is not None:
        if args.hand_effort_limit <= 0.0:
            raise ValueError("hand-effort-limit 必须为正数")
        for actuator_name in ("g20_independent", "g20_followers"):
            source = env_cfg.robot.actuators[actuator_name]
            source.effort_limit = args.hand_effort_limit
            source.effort_limit_sim = args.hand_effort_limit
    if args.explicit_hand_pd:
        # 只替换手部两组执行器，手臂仍保留隐式重力补偿驱动。显式 PD
        # 将位置/速度误差在每个物理子步转换为有界力矩，便于保持段
        # 真正施加阻尼；默认关闭以保持历史隐式执行器合同。
        from isaaclab.actuators import IdealPDActuatorCfg

        for actuator_name in ("g20_independent", "g20_followers"):
            source = env_cfg.robot.actuators[actuator_name]
            env_cfg.robot.actuators[actuator_name] = IdealPDActuatorCfg(
                joint_names_expr=list(source.joint_names_expr),
                effort_limit=source.effort_limit,
                velocity_limit=source.velocity_limit,
                effort_limit_sim=source.effort_limit_sim,
                velocity_limit_sim=source.velocity_limit_sim,
                stiffness=source.stiffness,
                damping=source.damping,
                armature=source.armature,
                friction=source.friction,
                dynamic_friction=source.dynamic_friction,
                viscous_friction=source.viscous_friction,
            )
    if args.external_forces_every_iteration:
        env_cfg.sim.physics.enable_external_forces_every_iteration = True
    if args.enhanced_determinism:
        env_cfg.sim.physics.enable_enhanced_determinism = True
    if args.robot_max_depenetration_velocity is not None:
        if args.robot_max_depenetration_velocity <= 0.0:
            raise ValueError("robot-max-depenetration-velocity 必须为正数")
        env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = args.robot_max_depenetration_velocity
    if args.fruit_max_depenetration_velocity is not None:
        if args.fruit_max_depenetration_velocity <= 0.0:
            raise ValueError("fruit-max-depenetration-velocity 必须为正数")
        env_cfg.tomato.spawn.rigid_props.max_depenetration_velocity = args.fruit_max_depenetration_velocity
    if args.fruit_contact_offset is not None:
        if args.fruit_contact_offset <= 0.0:
            raise ValueError("fruit-contact-offset 必须为正数")
        env_cfg.tomato.spawn.collision_props.contact_offset = args.fruit_contact_offset
    if args.fruit_rest_offset is not None:
        if args.fruit_rest_offset < 0.0:
            raise ValueError("fruit-rest-offset 不能为负数")
        env_cfg.tomato.spawn.collision_props.rest_offset = args.fruit_rest_offset
    if args.solver_position_iterations is not None:
        if args.solver_position_iterations < 1:
            raise ValueError("solver-position-iterations 必须为正整数")
        env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = args.solver_position_iterations
    if args.solver_velocity_iterations is not None:
        if args.solver_velocity_iterations < 1:
            raise ValueError("solver-velocity-iterations 必须为正整数")
        env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = args.solver_velocity_iterations
    if args.solver_type is not None:
        env_cfg.sim.physics.solver_type = args.solver_type
    if args.hand_damping is not None:
        if args.hand_damping <= 0.0:
            raise ValueError("hand-damping 必须为正数")
        for actuator_name in ("g20_independent", "g20_followers"):
            env_cfg.robot.actuators[actuator_name].damping = args.hand_damping
    if args.hand_stiffness is not None:
        if args.hand_stiffness <= 0.0:
            raise ValueError("hand-stiffness 必须为正数")
        for actuator_name in ("g20_independent", "g20_followers"):
            env_cfg.robot.actuators[actuator_name].stiffness = args.hand_stiffness
    if args.hand_armature is not None:
        if args.hand_armature < 0.0:
            raise ValueError("hand-armature 不能为负数")
        for actuator_name in ("g20_independent", "g20_followers"):
            env_cfg.robot.actuators[actuator_name].armature = args.hand_armature
    if args.hand_viscous_friction is not None:
        if args.hand_viscous_friction < 0.0:
            raise ValueError("hand-viscous-friction 不能为负数")
        for actuator_name in ("g20_independent", "g20_followers"):
            env_cfg.robot.actuators[actuator_name].viscous_friction = args.hand_viscous_friction
    if args.hand_velocity_limit is not None:
        if args.hand_velocity_limit <= 0.0:
            raise ValueError("hand-velocity-limit 必须为正数")
        for actuator_name in ("g20_independent", "g20_followers"):
            env_cfg.robot.actuators[actuator_name].velocity_limit_sim = args.hand_velocity_limit
    if (args.fruit_axis_min_mm is None) != (args.fruit_axis_max_mm is None):
        raise ValueError("fruit-axis-min-mm 与 fruit-axis-max-mm 必须同时提供")
    if args.diameter_bins_mm is not None and args.fruit_axis_min_mm is not None:
        raise ValueError("diameter-bins-mm 与 fruit-axis-min/max-mm 互斥")
    if args.diameter_bin_weights is not None and args.diameter_bins_mm is None:
        raise ValueError("diameter-bin-weights 必须与 diameter-bins-mm 一起提供")
    if args.fruit_axis_min_mm is not None:
        if not (0.0 < args.fruit_axis_min_mm <= args.fruit_axis_max_mm):
            raise ValueError("果径半轴范围必须为正且 min <= max")
        # 参数表示椭球半轴长度（mm）；环境内部同样以半轴米为单位。
        axis_min_m = args.fruit_axis_min_mm / 1000.0
        axis_max_m = args.fruit_axis_max_mm / 1000.0
        env_cfg.events.fruit_geometry.params["axis_ranges_m"] = {
            "x": (axis_min_m, axis_max_m),
            "y": (axis_min_m, axis_max_m),
            "z": (axis_min_m, axis_max_m),
        }
    if args.diameter_bins_mm is not None:
        from simulation.tomato_pick.isaac.geometry_sampling import (
            parse_bin_weights,
            parse_diameter_bins_mm,
        )

        bins_mm = parse_diameter_bins_mm(args.diameter_bins_mm)
        weights = parse_bin_weights(args.diameter_bin_weights, len(bins_mm))
        env_cfg.events.fruit_geometry.params.update(
            {
                "diameter_bins_m": tuple(
                    (low / 1000.0, high / 1000.0) for low, high in bins_mm
                ),
                "diameter_bin_weights": weights,
                "geometry_seed": int(args.seed),
                "geometry_cohort_id": str(args.geometry_cohort_id),
            }
        )
    offset_args = (args.offset_x_mm, args.offset_y_mm, args.offset_z_mm)
    if any(value is not None for value in offset_args):
        if not all(value is not None for value in offset_args):
            raise ValueError("offset-x/y/z-mm 必须同时提供")
        env_cfg.pinch_working_offset_m = tuple(value / 1000.0 for value in offset_args)
    if args.center_local_mm is not None:
        env_cfg.closed_grasp_fruit_center_m = tuple(
            value / 1000.0 for value in args.center_local_mm
        )
    env_cfg.palm_fruit_collision_filter_enabled = bool(
        args.palm_fruit_collision_filter
    )
    env_cfg.geometric_contact_proxy_enabled = bool(args.geometric_contact_proxy)
    if args.geometric_contact_proxy_band_mm is not None:
        if args.geometric_contact_proxy_band_mm <= 0.0:
            raise ValueError("geometric-contact-proxy-band-mm 必须为正数")
        env_cfg.geometric_contact_proxy_band_m = args.geometric_contact_proxy_band_mm / 1000.0
    agent_cfg = SmallFruitPinchPPORunnerCfg()
    agent_cfg.max_iterations = args.iterations
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            raise ValueError("learning-rate 必须为正数")
        agent_cfg.algorithm.learning_rate = args.learning_rate
    if args.entropy_coef is not None:
        if args.entropy_coef < 0.0:
            raise ValueError("entropy-coef 不能为负数")
        agent_cfg.algorithm.entropy_coef = args.entropy_coef
    if args.init_std is not None:
        if args.init_std <= 0.0:
            raise ValueError("init-std 必须为正数")
        agent_cfg.actor.distribution_cfg.init_std = args.init_std
    if args.residual_scale is not None and not args.guided_residual:
        raise ValueError("residual-scale 只能与 guided-residual 一起使用")
    if args.precontact_residual_fraction is not None:
        if not args.guided_residual:
            raise ValueError("precontact-residual-fraction 只能与 guided-residual 一起使用")
        if not 0.0 <= args.precontact_residual_fraction <= 1.0:
            raise ValueError("precontact-residual-fraction 必须在 [0, 1] 内")
    agent_cfg.logger = args.logger
    agent_cfg.run_name = args.run_name
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, metadata.version("rsl-rl-lib")
    )
    env_cls = NativeSmallFruitPinchEnv
    if args.guided_residual:
        from research.small_fruit_pinch_v1.source.guided_residual_env import (
            GuidedResidualSmallFruitPinchEnv,
        )

        env_cls = GuidedResidualSmallFruitPinchEnv
        if args.guide_action_values is not None:
            try:
                values = tuple(float(value.strip()) for value in args.guide_action_values.split(","))
            except ValueError as exc:
                raise ValueError("guide-action-values 必须是 14 个浮点数") from exc
            if len(values) != 14 or any(abs(value) > 1.0 for value in values):
                raise ValueError("guide-action-values 必须包含 14 个 [-1, 1] 值")
            env_cls.guide_action_values = values
        env_cls.geometry_prior_enabled = args.geometry_prior
        env_cls.geometry_prior_size_adaptive = args.geometry_prior_size_adaptive
        if args.geometry_prior_max_scale is not None:
            if not 0.0 <= args.geometry_prior_max_scale <= 1.0:
                raise ValueError("geometry-prior-max-scale 必须在 [0, 1] 内")
            env_cls.geometry_prior_max_scale = args.geometry_prior_max_scale
        env_cls.geometry_prior_after_contact = args.geometry_prior_after_contact
        if args.geometry_prior_scale is not None:
            if not 0.0 <= args.geometry_prior_scale <= 1.0:
                raise ValueError("geometry-prior-scale 必须在 [0, 1] 内")
            env_cls.geometry_prior_scale = args.geometry_prior_scale
        if args.geometry_prior_values is not None:
            try:
                values = tuple(float(value.strip()) for value in args.geometry_prior_values.split(","))
            except ValueError as exc:
                raise ValueError("geometry-prior-values 必须是 14 个浮点数") from exc
            if len(values) != 14 or any(abs(value) > 1.0 for value in values):
                raise ValueError("geometry-prior-values 必须包含 14 个 [-1, 1] 值")
            env_cls.geometry_prior_residual = values
        env_cls.force_balance_enabled = args.force_balance
        env_cls.force_servo_enabled = args.force_servo
        env_cls.direct_force_projection_enabled = args.direct_force_projection
        env_cls.phase_controller_enabled = args.phase_controller
        env_cls.phase_target_controller_enabled = args.phase_target_controller
        env_cls.phase_geometry_enabled = args.phase_geometry
        for name, value in (
            ("phase_search_step", args.phase_search_step),
            ("phase_close_step", args.phase_close_step),
            ("phase_retract_step", args.phase_retract_step),
            ("phase_offset_alpha", args.phase_offset_alpha),
            ("phase_max_offset", args.phase_max_offset),
            ("phase_root_fraction", args.phase_root_fraction),
            ("phase_low_delay_steps", args.phase_low_delay_steps),
            ("phase_low_ramp_steps", args.phase_low_ramp_steps),
            ("phase_hysteresis_n", args.phase_hysteresis_n),
            ("phase_geometry_step", args.phase_geometry_step),
            ("phase_geometry_gap_scale_m", args.phase_geometry_gap_scale_m),
            ("phase_target_step", args.phase_target_step),
            ("phase_target_alpha", args.phase_target_alpha),
            ("phase_target_deadband_n", args.phase_target_deadband_n),
            ("phase_target_raw_mix", args.phase_target_raw_mix),
        ):
            if value is not None:
                if name in {"phase_low_delay_steps", "phase_low_ramp_steps"}:
                    if value < 0:
                        raise ValueError(f"{name} 不能为负数")
                elif value < 0.0:
                    raise ValueError(f"{name} 不能为负数")
                setattr(env_cls, name, value)
        for attr, raw in (
            ("phase_target_force_n", args.phase_target_force_n),
            ("phase_target_error_scale_n", args.phase_target_error_scale_n),
        ):
            if raw is not None:
                try:
                    values = tuple(float(value.strip()) for value in raw.split(","))
                except ValueError as exc:
                    raise ValueError(f"{attr} 必须是 3 个浮点数") from exc
                if len(values) != 3 or any(value < 0.0 for value in values):
                    raise ValueError(f"{attr} 必须包含 3 个非负浮点数")
                setattr(env_cls, attr, values)
        for name, value in (
            ("force_servo_seek_gain", args.force_servo_seek_gain),
            ("force_servo_band_seek_gain", args.force_servo_band_seek_gain),
            ("force_servo_target_gain", args.force_servo_target_gain),
            ("force_servo_retract_gain", args.force_servo_retract_gain),
            ("force_servo_alpha", args.force_servo_alpha),
            ("force_servo_max_offset", args.force_servo_max_offset),
            ("force_servo_warning_fraction", args.force_servo_warning_fraction),
            ("force_servo_rate_gain", args.force_servo_rate_gain),
            ("force_servo_rate_deadband_n", args.force_servo_rate_deadband_n),
            ("force_servo_rate_scale_n", args.force_servo_rate_scale_n),
            ("direct_force_projection_warning_fraction", args.direct_force_warning_fraction),
            ("direct_force_projection_high_gain", args.direct_force_high_gain),
            ("direct_force_projection_low_gain", args.direct_force_low_gain),
            ("direct_force_projection_alpha", args.direct_force_alpha),
            ("direct_force_projection_max_open", args.direct_force_max_open),
            ("direct_force_projection_root_fraction", args.direct_force_root_fraction),
            ("direct_force_projection_rate_gain", args.direct_force_rate_gain),
            ("direct_force_projection_rate_deadband_n", args.direct_force_rate_deadband_n),
            ("direct_force_projection_rate_scale_n", args.direct_force_rate_scale_n),
        ):
            if value is not None:
                if value < 0.0:
                    raise ValueError(f"{name} 不能为负数")
                setattr(env_cls, name, value)
        if args.direct_force_high_gain_by_digit is not None:
            try:
                values = tuple(
                    float(value.strip())
                    for value in args.direct_force_high_gain_by_digit.split(",")
                )
            except ValueError as exc:
                raise ValueError(
                    "direct-force-high-gain-by-digit 必须是 3 个浮点数"
                ) from exc
            if len(values) != 3 or any(value < 0.0 for value in values):
                raise ValueError(
                    "direct-force-high-gain-by-digit 必须包含 3 个非负浮点数"
                )
            env_cls.direct_force_projection_high_gain_by_digit = values
        for attr, raw in (
            ("force_servo_seek_direction", args.force_servo_seek_direction),
            ("force_servo_retract_direction", args.force_servo_retract_direction),
        ):
            if raw is not None:
                try:
                    direction = tuple(float(value.strip()) for value in raw.split(","))
                except ValueError as exc:
                    raise ValueError(f"{attr} 必须是 3 个浮点数") from exc
                if len(direction) != 3 or any(abs(value) > 1.0 for value in direction):
                    raise ValueError(f"{attr} 必须包含 3 个 [-1, 1] 值")
                setattr(env_cls, attr, direction)
        if args.projection_guard_fraction is not None:
            if not 0.0 < args.projection_guard_fraction <= 1.0:
                raise ValueError("projection-guard-fraction 必须在 (0, 1] 内")
            env_cls.projection_guard_fraction = args.projection_guard_fraction
        env_cls.guide_feedback_enabled = args.guide_feedback
        env_cls.adaptive_closure_enabled = args.adaptive_closure
        env_cls.residual_require_all_contact = args.residual_require_all_contact
        env_cls.residual_trust_region_enabled = args.residual_trust_region
        env_cls.freeze_required_distal = args.freeze_required_distal
        if args.residual_trust_warning_fraction is not None:
            if not 0.0 < args.residual_trust_warning_fraction <= 1.0:
                raise ValueError("residual-trust-warning-fraction 必须在 (0, 1] 内")
            env_cls.residual_trust_warning_fraction = args.residual_trust_warning_fraction
        if args.residual_trust_in_band_scale is not None:
            if not 0.0 <= args.residual_trust_in_band_scale <= 1.0:
                raise ValueError("residual-trust-in-band-scale 必须在 [0, 1] 内")
            env_cls.residual_trust_in_band_scale = args.residual_trust_in_band_scale
        if args.approach_speed is not None:
            if args.approach_speed <= 0.0:
                raise ValueError("approach-speed 必须为正数")
            env_cls.approach_speed = args.approach_speed
        for name, value in (
            ("hold_speed", args.hold_speed),
            ("hold_acceleration", args.hold_acceleration),
            ("natural_frequency_hz", args.natural_frequency),
        ):
            if value is not None:
                if value <= 0.0:
                    raise ValueError(f"{name} 必须为正数")
                setattr(env_cls, name, value)
        if args.hold_joint_target_speed is not None:
            if args.hold_joint_target_speed <= 0.0:
                raise ValueError("hold-joint-target-speed 必须为正数")
            env_cls.hold_joint_target_speed_rad_s = args.hold_joint_target_speed
        if args.hold_joint_target_open_speed is not None:
            if args.hold_joint_target_open_speed <= 0.0:
                raise ValueError("hold-joint-target-open-speed 必须为正数")
            env_cls.hold_joint_target_open_speed_rad_s = args.hold_joint_target_open_speed
        if args.hold_hand_velocity_limit is not None:
            if args.hold_hand_velocity_limit <= 0.0:
                raise ValueError("hold-hand-velocity-limit 必须为正数")
            env_cls.hold_hand_velocity_limit_rad_s = args.hold_hand_velocity_limit
        if args.contact_hand_velocity_limit is not None:
            if args.contact_hand_velocity_limit <= 0.0:
                raise ValueError("contact-hand-velocity-limit 必须为正数")
            env_cls.contact_hand_velocity_limit_rad_s = args.contact_hand_velocity_limit
        if args.contact_hand_stiffness is not None:
            if args.contact_hand_stiffness <= 0.0:
                raise ValueError("contact-hand-stiffness 必须为正数")
            env_cls.contact_hand_stiffness = args.contact_hand_stiffness
        if args.contact_hand_damping is not None:
            if args.contact_hand_damping <= 0.0:
                raise ValueError("contact-hand-damping 必须为正数")
            env_cls.contact_hand_damping = args.contact_hand_damping
        if args.contact_hand_target_speed is not None:
            if args.contact_hand_target_speed <= 0.0:
                raise ValueError("contact-hand-target-speed 必须为正数")
            env_cls.contact_hand_target_speed_rad_s = args.contact_hand_target_speed
        if args.hold_hand_stiffness is not None:
            if args.hold_hand_stiffness <= 0.0:
                raise ValueError("hold-hand-stiffness 必须为正数")
            env_cls.hold_hand_stiffness = args.hold_hand_stiffness
        if args.hold_hand_stiffness_by_joint is not None:
            try:
                values = tuple(float(value.strip()) for value in args.hold_hand_stiffness_by_joint.split(","))
            except ValueError as exc:
                raise ValueError("hold-hand-stiffness-by-joint 必须是 16 个浮点数") from exc
            if len(values) != 16 or any(value <= 0.0 for value in values):
                raise ValueError("hold-hand-stiffness-by-joint 必须包含 16 个正数")
            env_cls.hold_hand_stiffness_by_joint = values
        if args.hold_hand_damping is not None:
            if args.hold_hand_damping <= 0.0:
                raise ValueError("hold-hand-damping 必须为正数")
            env_cls.hold_hand_damping = args.hold_hand_damping
        if args.hold_hand_damping_by_joint is not None:
            try:
                values = tuple(float(value.strip()) for value in args.hold_hand_damping_by_joint.split(","))
            except ValueError as exc:
                raise ValueError("hold-hand-damping-by-joint 必须是 16 个浮点数") from exc
            if len(values) != 16 or any(value <= 0.0 for value in values):
                raise ValueError("hold-hand-damping-by-joint 必须包含 16 个正数")
            env_cls.hold_hand_damping_by_joint = values
        if args.hold_joint_velocity_target is not None:
            if args.hold_joint_velocity_target <= 0.0:
                raise ValueError("hold-joint-velocity-target 必须为正数")
            env_cls.hold_joint_velocity_target_rad_s = args.hold_joint_velocity_target
        env_cls.hold_zero_velocity_target = args.hold_zero_velocity_target
        if args.hold_hand_effort_limit is not None:
            if args.hold_hand_effort_limit <= 0.0:
                raise ValueError("hold-hand-effort-limit 必须为正数")
            env_cls.hold_hand_effort_limit = args.hold_hand_effort_limit
        if args.hold_joint_target_error_limit is not None:
            if args.hold_joint_target_error_limit <= 0.0:
                raise ValueError("hold-joint-target-error-limit 必须为正数")
            env_cls.hold_joint_target_error_limit_rad = args.hold_joint_target_error_limit
        if args.hold_joint_target_error_limit_by_joint is not None:
            try:
                values = tuple(
                    float(value.strip())
                    for value in args.hold_joint_target_error_limit_by_joint.split(",")
                )
            except ValueError as exc:
                raise ValueError(
                    "hold-joint-target-error-limit-by-joint 必须是 16 个浮点数"
                ) from exc
            if len(values) != 16 or any(value <= 0.0 for value in values):
                raise ValueError(
                    "hold-joint-target-error-limit-by-joint 必须包含 16 个正数"
                )
            env_cls.hold_joint_target_error_limit_by_joint = values
        if args.hold_joint_target_anchor_fraction is not None:
            if not 0.0 <= args.hold_joint_target_anchor_fraction <= 1.0:
                raise ValueError("hold-joint-target-anchor-fraction 必须在 [0, 1] 内")
            env_cls.hold_joint_target_anchor_fraction = args.hold_joint_target_anchor_fraction
        if args.hold_joint_target_anchor_fraction_by_joint is not None:
            try:
                values = tuple(
                    float(value.strip())
                    for value in args.hold_joint_target_anchor_fraction_by_joint.split(",")
                )
            except ValueError as exc:
                raise ValueError(
                    "hold-joint-target-anchor-fraction-by-joint 必须是 16 个浮点数"
                ) from exc
            if len(values) != 16 or any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(
                    "hold-joint-target-anchor-fraction-by-joint 必须包含 16 个 [0,1] 值"
                )
            env_cls.hold_joint_target_anchor_fraction_by_joint = values
        if args.hold_target_filter_alpha is not None:
            if not 0.0 < args.hold_target_filter_alpha <= 1.0:
                raise ValueError("hold-target-filter-alpha 必须在 (0, 1] 内")
            env_cls.hold_target_filter_alpha = args.hold_target_filter_alpha
        if args.hold_hand_velocity_limit_by_joint is not None:
            try:
                values = tuple(
                    float(value.strip())
                    for value in args.hold_hand_velocity_limit_by_joint.split(",")
                )
            except ValueError as exc:
                raise ValueError("hold-hand-velocity-limit-by-joint 必须是 16 个浮点数") from exc
            if len(values) != 16 or any(value <= 0.0 for value in values):
                raise ValueError("hold-hand-velocity-limit-by-joint 必须包含 16 个正数")
            env_cls.hold_hand_velocity_limit_by_joint = values
        if args.hold_velocity_damping_effort is not None:
            if args.hold_velocity_damping_effort < 0.0:
                raise ValueError("hold-velocity-damping-effort 不能为负数")
            env_cls.hold_velocity_damping_effort = args.hold_velocity_damping_effort
        env_cls.hold_effort_control_enabled = args.hold_effort_control
        env_cls.hold_force_impedance_enabled = args.hold_force_impedance
        env_cls.hold_current_impedance_enabled = args.hold_current_impedance
        for attr, raw in (
            ("hold_impedance_effort_by_digit", args.hold_impedance_effort_by_digit),
            ("hold_impedance_force_target_n", args.hold_impedance_force_target),
        ):
            if raw is not None:
                try:
                    values = tuple(float(value.strip()) for value in raw.split(","))
                except ValueError as exc:
                    raise ValueError(f"{attr} 必须是 3 个浮点数") from exc
                if len(values) != 3 or any(value <= 0.0 for value in values):
                    raise ValueError(f"{attr} 必须包含 3 个正数")
                setattr(env_cls, attr, values)
        for attr, value in (
            ("hold_impedance_force_gain", args.hold_impedance_force_gain),
            ("hold_impedance_start_fraction", args.hold_impedance_start_fraction),
            ("hold_impedance_ramp_substeps", args.hold_impedance_ramp_substeps),
            ("hold_impedance_damping", args.hold_impedance_damping),
        ):
            if value is not None:
                if attr == "hold_impedance_ramp_substeps":
                    if value < 1:
                        raise ValueError("hold_impedance_ramp_substeps 必须为正整数")
                elif attr == "hold_impedance_start_fraction":
                    if not 0.0 < value <= 1.0:
                        raise ValueError("hold_impedance_start_fraction 必须在 (0, 1] 内")
                elif value < 0.0:
                    raise ValueError(f"{attr} 不能为负数")
                setattr(env_cls, attr, value)
        for attr, value in (
            ("hold_effort_stiffness", args.hold_effort_stiffness),
            ("hold_effort_damping", args.hold_effort_damping),
            ("hold_effort_max", args.hold_effort_max),
            ("hold_effort_in_band", args.hold_effort_in_band),
            ("hold_effort_low_fraction", args.hold_effort_low_fraction),
            ("hold_effort_high_fraction", args.hold_effort_high_fraction),
        ):
            if value is not None:
                if attr in {"hold_effort_stiffness", "hold_effort_damping", "hold_effort_in_band"}:
                    if value < 0.0:
                        raise ValueError(f"{attr} 不能为负数")
                elif value <= 0.0:
                    raise ValueError(f"{attr} 必须为正数")
                setattr(env_cls, attr, value)
        if args.position_target_interpolation is not None:
            if args.position_target_interpolation < 1:
                raise ValueError("position-target-interpolation 必须为正整数")
            env_cls.position_target_interpolation_steps = args.position_target_interpolation
        env_cls.manual_decimation_control = args.manual_decimation_control
        if args.manual_decimation_control and args.position_target_interpolation is None:
            env_cls.position_target_interpolation_steps = int(env_cfg.decimation)
        if args.joint_smoothness_weight is not None:
            if args.joint_smoothness_weight < 0.0:
                raise ValueError("joint-smoothness-weight 不能为负数")
            env_cls.joint_smoothness_reward_weight = args.joint_smoothness_weight
        env_cls.proximity_speed_enabled = args.proximity_speed
        if args.proximity_speed_start_mm is not None:
            if args.proximity_speed_start_mm <= 0.0:
                raise ValueError("proximity-speed-start-mm 必须为正数")
            env_cls.proximity_speed_start_m = args.proximity_speed_start_mm / 1000.0
        if args.proximity_speed_floor is not None:
            if not 0.0 < args.proximity_speed_floor <= 1.0:
                raise ValueError("proximity-speed-floor 必须在 (0, 1] 内")
            env_cls.proximity_speed_floor = args.proximity_speed_floor
        env_cls.proximity_speed_global = args.proximity_speed_global
        env_cls.geometry_contact_projection_enabled = args.geometry_contact_projection
        if args.geometry_contact_margin_mm is not None:
            if args.geometry_contact_margin_mm <= 0.0:
                raise ValueError("geometry-contact-margin-mm 必须为正数")
            env_cls.geometry_contact_projection_margin_m = args.geometry_contact_margin_mm / 1000.0
        if args.geometry_contact_gain is not None:
            if args.geometry_contact_gain < 0.0:
                raise ValueError("geometry-contact-gain 不能为负数")
            env_cls.geometry_contact_projection_gain = args.geometry_contact_gain
        if args.geometry_contact_max_open is not None:
            if args.geometry_contact_max_open < 0.0:
                raise ValueError("geometry-contact-max-open 不能为负数")
            env_cls.geometry_contact_projection_max_open = args.geometry_contact_max_open
        if args.geometry_contact_root_fraction is not None:
            if not 0.0 <= args.geometry_contact_root_fraction <= 1.0:
                raise ValueError("geometry-contact-root-fraction 必须在 [0, 1] 内")
            env_cls.geometry_contact_projection_root_fraction = args.geometry_contact_root_fraction
        env_cls.geometry_approach_enabled = args.geometry_approach
        if args.geometry_approach_target_gap_mm is not None:
            if args.geometry_approach_target_gap_mm < 0.0:
                raise ValueError("geometry-approach-target-gap-mm 不能为负数")
            env_cls.geometry_approach_target_gap_m = args.geometry_approach_target_gap_mm / 1000.0
        if args.geometry_approach_scale_mm is not None:
            if args.geometry_approach_scale_mm <= 0.0:
                raise ValueError("geometry-approach-scale-mm 必须为正数")
            env_cls.geometry_approach_scale_m = args.geometry_approach_scale_mm / 1000.0
        for name, value in (
            ("geometry_approach_gain", args.geometry_approach_gain),
            ("geometry_approach_max_offset", args.geometry_approach_max_offset),
            ("geometry_approach_alpha", args.geometry_approach_alpha),
        ):
            if value is not None:
                if value < 0.0:
                    raise ValueError(f"{name} 不能为负数")
                setattr(env_cls, name, value)
        if args.force_gate_start_fraction is not None:
            if not 0.0 < args.force_gate_start_fraction <= 1.0:
                raise ValueError("force-gate-start-fraction 必须在 (0, 1] 内")
            env_cls.force_gate_start_fraction = args.force_gate_start_fraction
        if args.residual_scale is not None:
            if not 0.0 <= args.residual_scale <= 1.0:
                raise ValueError("residual-scale 必须在 [0, 1] 内")
            env_cls.residual_scale = args.residual_scale
        if args.distal_residual_scale is not None:
            if not 0.0 <= args.distal_residual_scale <= 1.0:
                raise ValueError("distal-residual-scale 必须在 [0, 1] 内")
            env_cls.distal_residual_scale = args.distal_residual_scale
        if args.precontact_residual_fraction is not None:
            env_cls.precontact_residual_fraction = args.precontact_residual_fraction
    env = env_cls(cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        from research.small_fruit_pinch_v1.source.native_run_manifest import freeze_native_run

        freeze_native_run(env, agent_cfg, args)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=str(args.log_dir), device=agent_cfg.device
        )
        if args.resume is not None:
            runner.load(str(args.resume), map_location=args.device)
            print(f"native_train resumed_from={args.resume}", flush=True)
        elif args.actor_init is not None:
            _initialize_actor_from_checkpoint(runner, args.actor_init)
            print(f"native_train actor_initialized_from={args.actor_init}", flush=True)
        elif args.p1_mean_init:
            _initialize_actor_mean_from_p1(runner)
            print("native_train actor_mean_initialized_from_p1=true", flush=True)
        elif args.zero_residual_init:
            _initialize_actor_residual_zero(runner)
            print("native_train actor_residual_initialized_zero=true", flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        runner.save(str(args.log_dir / "model_final.pt"))
        print(
            f"native_train_complete iterations={args.iterations} num_envs={args.num_envs} log_dir={args.log_dir}",
            flush=True,
        )
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
