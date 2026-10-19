"""小果捏取 v1 的当前 Isaac Lab 原生左手 PhysX 环境。

这不是历史 89% 任务的复训：手臂保持在确定的左臂初始位姿，策略只输出
历史可迁移的 10D 手部流形动作。水果、五个指尖接触、茎弹簧和真实 PhysX
仍由当前项目的原生 ``TomatoPickEnv`` 提供。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from .contract import ACTOR_OBSERVATION_DIM, ACTION_DIM, CRITIC_STATE_DIM, EPISODE_STEPS
from .pinch_action_manifold import pinch_action_to_joint_targets
from .temporal import CriticalDampedActionFilter
from .verifier import TimStrictVerifier
from .verification_geometry import contact_topology


LEFT_HAND_USD = (
    Path(__file__).resolve().parents[1]
    / "assets/left_hand_urdf/left_hand_l20_physx.usd"
)
LEFT_HAND_BODY_NAMES = (
    "base_link",
    *(f"thumb_link{index}" for index in range(6)),
    *(
        f"{finger}_link{index}"
        for finger in ("index", "middle", "ring", "little")
        for index in range(5)
    ),
)
LEFT_FINGERTIP_FRAME_BODY_NAMES = {
    "thumb": "thumb_link5",
    "index": "index_link4",
    "middle": "middle_link4",
    "ring": "ring_link4",
    "little": "little_link4",
}
LEFT_FINGERTIP_CONTACT_BODY_NAMES = {
    "thumb": "thumb_link4",
    "index": "index_link3",
    "middle": "middle_link3",
    "ring": "ring_link3",
    "little": "little_link3",
}
LEFT_FINGERTIP_PATHS = {
    "thumb": "/World/envs/env_.*/Robot/thumb_link4",
    "index": "/World/envs/env_.*/Robot/index_link3",
    "middle": "/World/envs/env_.*/Robot/middle_link3",
    "ring": "/World/envs/env_.*/Robot/ring_link3",
    "little": "/World/envs/env_.*/Robot/little_link3",
}
def _left(value: str) -> str:
    """把当前原生任务的右手命名空间映射到同构的左手命名空间。"""

    return re.sub(
        r"/R([1-8]_Link)", lambda match: f"/L{match.group(1)}", value.replace("right_hand", "left_hand")
    )


def configure_native_module_for_left() -> tuple[object, object]:
    """在本进程内切换原生任务到独立左手命名表。

    旧版本只把双臂 USD 的 ``right_hand_*`` 字符串改成
    ``left_hand_*``，并没有从场景中移除机械臂。当前配置使用 prefixless
    手部 USD，因此所有查找、mimic 和传感器路径均必须与该资产完全一致。
    """

    from simulation.tomato_pick.isaac import env as native
    from simulation.tomato_pick.isaac import env_cfg as cfg_module

    native.ACTION_DIM = ACTION_DIM
    native.RIGHT_HAND_CONTROL_JOINT_NAMES = (
        "thumb_joint2", "index_joint1", "middle_joint1", "ring_joint1", "little_joint1",
        "thumb_joint0", "index_joint0", "middle_joint0", "ring_joint0", "little_joint0",
        "thumb_joint1", "thumb_joint3", "index_joint2", "middle_joint2", "ring_joint2", "little_joint2",
    )
    native.RIGHT_HAND_MIMIC_RAD = {
        "thumb_joint4": ("thumb_joint3", 1.11, -0.04545),
        "index_joint3": ("index_joint2", 0.94, -0.04545),
        "middle_joint3": ("middle_joint2", 0.94, -0.04545),
        "ring_joint3": ("ring_joint2", 0.94, -0.04545),
        "little_joint3": ("little_joint2", 0.94, -0.04545),
    }
    native.MIMIC_FOLLOWER_JOINT_NAMES = tuple(native.RIGHT_HAND_MIMIC_RAD)
    native.RIGHT_FINGERTIP_FRAME_BODY_NAMES = dict(LEFT_FINGERTIP_FRAME_BODY_NAMES)
    native.RIGHT_FINGERTIP_CONTACT_BODY_NAMES = dict(LEFT_FINGERTIP_CONTACT_BODY_NAMES)
    native.RIGHT_HAND_BODY_NAMES = LEFT_HAND_BODY_NAMES
    native.RIGHT_HAND_BODY_ROOT = "/World/envs/env_.*/Robot/base_link"
    cfg_module.RIGHT_HAND_BODY_ROOT = native.RIGHT_HAND_BODY_ROOT
    cfg_module.RIGHT_FINGERTIP_PATHS = dict(LEFT_FINGERTIP_PATHS)
    cfg_module.RIGHT_FINGERTIP_FRAME_BODY_NAMES = dict(LEFT_FINGERTIP_FRAME_BODY_NAMES)
    cfg_module.RIGHT_FINGERTIP_CONTACT_BODY_NAMES = dict(LEFT_FINGERTIP_CONTACT_BODY_NAMES)
    return native, cfg_module


native, native_cfg_module = configure_native_module_for_left()


class LeftHandDirectionalPDActuator(IdealPDActuator):
    """显式 PD 驱动，并在 canonical 坐标中执行软件关节边界保护。

    孤立 USD 的关节轴向已通过正负方向探针逐一核对，当前资产不需要
    额外取反。方向表仍保留在执行器层，便于未来替换资产时显式记录而
    不把符号变更泄漏到 14D 策略语义或真实硬件接口。
    """

    # The isolated USD reports the same generalized-coordinate sign as the
    # canonical FK writer for every active revolute axis.  Earlier probes
    # falsely marked thumb_joint0/2 as mirrored because their milligram-scale
    # links did not overcome static friction at the first torque setting;
    # flipping those efforts made a full grasp target run into the upper/lower
    # limits.  Keep the map explicit (rather than silently assuming a future
    # asset has the same convention), but empty for this validated asset.
    _INVERTED_JOINT_NAMES = frozenset()
    _JOINT_LIMITS = {
        **{f"{finger}_joint0": (-0.26, 0.26) for finger in ("index", "middle", "ring", "little")},
        "thumb_joint0": (-1.05, 0.49),
        **{f"{finger}_joint1": (0.0, 1.66) for finger in ("index", "middle", "ring", "little")},
        "thumb_joint1": (0.0, 1.45),
        **{f"{finger}_joint2": (0.0, 1.36) for finger in ("index", "middle", "ring", "little")},
        "thumb_joint2": (0.0, 0.84),
        **{f"{finger}_joint3": (0.0, 1.48) for finger in ("index", "middle", "ring", "little")},
        "thumb_joint3": (0.0, 1.32),
        "thumb_joint4": (0.0, 1.34),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        signs = torch.ones(len(self.joint_names), dtype=torch.float32, device=self._device)
        for index, name in enumerate(self.joint_names):
            if name in self._INVERTED_JOINT_NAMES:
                signs[index] = -1.0
        self._joint_direction_sign = signs
        limits = [self._JOINT_LIMITS.get(name, (-float("inf"), float("inf"))) for name in self.joint_names]
        self._joint_lower_limit = torch.tensor(
            [item[0] for item in limits], dtype=torch.float32, device=self._device
        )
        self._joint_upper_limit = torch.tensor(
            [item[1] for item in limits], dtype=torch.float32, device=self._device
        )

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        result = super().compute(control_action, joint_pos, joint_vel)
        sign = self._joint_direction_sign.unsqueeze(0)
        # ``sign`` converts a canonical-coordinate effort to the physical
        # actuator effort.  The values returned by ``super().compute`` are
        # still in canonical target coordinates, so apply the software limit
        # guard *before* multiplying by ``sign``.  (Multiplying first would
        # incorrectly classify every positive command on a mirrored joint as
        # an outward lower-limit command and pin that joint at zero.)
        canonical_computed = self.computed_effort.clone()
        canonical_effort = self.applied_effort.clone()
        lower = self._joint_lower_limit.unsqueeze(0)
        upper = self._joint_upper_limit.unsqueeze(0)
        margin = 0.015
        near_lower_outward = (joint_pos <= lower + margin) & (canonical_effort < 0.0)
        near_upper_outward = (joint_pos >= upper - margin) & (canonical_effort > 0.0)
        # A tiny numerical excursion below a zero lower limit is common just
        # after reset.  Do not brake an *inward* command there: using only the
        # measured velocity would keep that joint pinned forever whenever the
        # reset noise happens to be negative.  The guard is for an effort that
        # is both outside the limit and still pointing farther outward.
        below_and_outward = (joint_pos < lower) & (canonical_effort < 0.0)
        above_and_outward = (joint_pos > upper) & (canonical_effort > 0.0)
        brake = 0.30 * joint_vel.abs()
        canonical_effort = torch.where(
            near_lower_outward | near_upper_outward,
            torch.zeros_like(canonical_effort),
            canonical_effort,
        )
        canonical_effort = torch.where(
            below_and_outward,
            brake,
            canonical_effort,
        )
        canonical_effort = torch.where(
            above_and_outward,
            -brake,
            canonical_effort,
        )
        self.computed_effort = canonical_computed * sign
        self.applied_effort = canonical_effort * sign
        if result.joint_efforts is not None:
            result.joint_efforts = self.applied_effort
        return result


@configclass
class NativeSmallFruitPinchCfg(native_cfg_module.TomatoPickEnvCfg):
    """固定基座左手、14D 策略、114D/175D 合同的训练配置。

    ``robot`` is intentionally rebuilt from the isolated left-hand USD.  No
    arm actuator, arm joint target, right-hand link, or bimanual articulation
    is present in this scene.
    """

    hand_only_scene = True
    # 当前左手 USD 的 PhysX articulation 已关闭自碰撞；AABB 代理仅作日志，
    # 不应在训练中把相邻碰撞壳的保守重叠变成终止条件。
    self_collision_proxy_terminates = False
    ee_body_name = "base_link"
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LEFT_HAND_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # The isolated hand is fixed-base and its intended palm-up
                # training frame should not drift under the imported URDF's
                # arbitrary world orientation.  Fruit gravity remains active;
                # the hand joints are driven explicitly by the low-torque PD
                # model below so the policy learns contact forces, not a
                # gravity-induced thumb bias.
                disable_gravity=True,
                max_depenetration_velocity=0.75,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "g20_independent": IdealPDActuatorCfg(
                joint_names_expr=[
                    r"thumb_joint[0-3]",
                    r"(index|middle|ring|little)_joint[0-2]",
                ],
                # 该孤立 USD 的末端连杆质量很小，过低的远端力矩会让根部
                # 已经到位而指尖永远停在果实外侧。统一的仿真上限先确保
                # 14D 的根部/指尖都能实际跟踪；安全由速度上限、动作滤波
                # 和下方的 canonical 边界保护共同承担，后续再按实测接触
                # 力做逐指降额。
                effort_limit_sim=0.20,
                effort_limit=0.20,
                velocity_limit_sim=1.20,
                velocity_limit=1.20,
                armature=0.0001,
                stiffness=20.0,
                damping=1.0,
                friction=0.01,
                class_type=LeftHandDirectionalPDActuator,
            ),
            "g20_followers": IdealPDActuatorCfg(
                joint_names_expr=[
                    r"thumb_joint4",
                    r"(index|middle|ring|little)_joint3",
                ],
                effort_limit_sim=0.20,
                effort_limit=0.20,
                velocity_limit_sim=1.20,
                velocity_limit=1.20,
                armature=0.0001,
                stiffness=1.0,
                damping=0.50,
                friction=0.01,
                class_type=LeftHandDirectionalPDActuator,
            ),
        },
    )

    tomato = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tomato",
        spawn=sim_utils.MeshSphereCfg(
            radius=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.50,
                linear_damping=0.08,
                angular_damping=0.08,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0002,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.70,
                dynamic_friction=0.58,
                restitution=0.01,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.06, 0.03), roughness=0.72
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.025, -0.036, 0.184)),
    )

    episode_length_s = EPISODE_STEPS / 20.0
    verification_release_support = True
    action_space = ACTION_DIM
    observation_space = ACTOR_OBSERVATION_DIM
    state_space = CRITIC_STATE_DIM
    # 训练阶段不让策略寻找水果；只在闭合工作空间内做尺寸和横向扰动。
    # 孤立 USD 的 base_link 坐标与历史双臂资产不同：五指沿 +Z 展开，
    # 拇指从 Y<0 侧横向进入。因此历史任务的 (0.007,-0.066,0.160)
    # 会把水果放到手掌下方约 18 cm，所有动作都会“无接触掉落”。这里
    # 用左手 USD 的实测 FK 工作区表达闭合中心：转换到父任务抓取基后，
    # 对应世界坐标约 (0.088,-0.017,0.165) m（64--70 mm 球径）。它是初始工作区中心，
    # 不是策略终点；后续只对该中心做小范围随机化。
    # The parent reset expresses this point in the grasp basis, not in world
    # coordinates.  The isolated-left-hand FK scan found a collision-free
    # three-finger sphere centre at about (88, -17, 165) mm in the base frame
    # for the 64--70 mm band.  Converting that point through the reset grasp
    # basis gives this local reference (approximately -90, -4, -165) mm.  It
    # is deliberately a *working centre*, not a terminal joint pose: PPO still
    # has to establish thumb opposition and bring the three fingertips onto
    # the surface.  The previous (-90, -30, -150) mm centre was safe at the
    # open pose but left the middle fingertip 10--15 mm outside the sphere,
    # so a dynamic probe could only produce thumb/index contact.
    # 注意：该值是抓取基（grasp basis）坐标，不是环境/世界坐标。
    # 当前孤立左手的几何候选约为世界坐标 (58,-23,177) mm；按 reset
    # 中的抓取基换算后约为 (-61,-13,-177) mm。此前误把世界坐标
    # (148,-60.7,53.5) mm 直接写入此处，使果实实际落在掌后/地面下，
    # 造成所有教师与 PPO 零接触。
    closed_grasp_fruit_center_m = (-0.061, -0.013, -0.177)
    # 从掌前闭合中心沿抓取轴预置在外侧，避免 reset 时果实已经穿入
    # 指节而被首接触/近端门立即终止；这与历史 PPO 的连续接近合同一致。
    pregrasp_distance_range_m = (0.055, 0.075)
    pregrasp_lateral_offset_range_m = (0.005, 0.020)
    pregrasp_axial_noise_m = 0.005
    pregrasp_arm_joint_noise_deg = 0.0
    minimum_initial_clearance_m = 0.002
    # 几何审计模式：当前孤立 USD 的 base_link AABB 大于真实掌面局部形状，
    # 对 64--70 mm 球会把一个本应在掌前的对向候选误判为初始穿透。开启
    # 后只过滤 ``base_link`` 与水果的碰撞并从 reset clearance 中排除该
    # 粗糙包围盒；指节碰撞、真实三指接触、过载和自碰撞仍保持物理有效。
    # 默认关闭，正式发布前必须用精确掌面碰撞体替换此诊断开关。
    palm_fruit_collision_filter_enabled = False
    # P1 联合扫描得到的无穿透三指共同可达位移；它相对父环境已经通过
    # 全碰撞体 clearance 修正后的初始位姿表达，不是硬编码最终关节姿态。
    # 孤立手的中心已经直接写入 closed_grasp_fruit_center_m，不再叠加
    # 历史双臂任务的 18/30/24 mm 偏移，避免中心再次被移到掌下。
    pinch_working_offset_m = (0.0, 0.0, 0.0)
    action_delay_steps = (0, 0)
    fingertip_contacts = {
        name: ContactSensorCfg(
            prim_path=prim_path,
            update_period=1.0 / 120.0,
            history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
            track_pose=True,
            track_contact_points=True,
            track_friction_forces=True,
            # 旧值 32 在三指同时压住球面时会被 PhysX 截断，导致接触点
            # 质心和“首接触区域”审计失真。128 是诊断容量，不改变求解器。
            max_contact_data_count_per_prim=128,
        )
        for name, prim_path in LEFT_FINGERTIP_PATHS.items()
    }
    auxiliary_contacts = {
        "palm": ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/base_link",
            update_period=1.0 / 120.0,
            history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
            track_pose=True,
            track_contact_points=False,
            track_friction_forces=False,
            max_contact_data_count_per_prim=128,
        ),
        **{
            f"forbidden_{body}": ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{body}",
                update_period=1.0 / 120.0,
                history_length=0,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Tomato"],
                track_pose=False,
                track_contact_points=False,
                track_friction_forces=False,
                max_contact_data_count_per_prim=128,
            )
            for body in (
                "thumb_link2", "thumb_link3",
                "index_link1", "index_link2",
                "middle_link1", "middle_link2",
            )
        },
    }
    # B0 的物理域按计划固定为果径 64–70 mm（椭球半轴 32–35 mm）。父任务
    # 默认是 50–70 mm；如果不在这里覆写，训练日志会把小于 B0 的样本误算
    # 成标称域，导致阶段门和回归比较失真。后续 B1–B3 通过显式配置分支
    # 扩大该范围，不修改历史检查点的含义。
    events = deepcopy(native_cfg_module.TomatoPickEventCfg())
    events.fruit_geometry.params["axis_ranges_m"] = {
        "x": (0.032, 0.035),
        "y": (0.032, 0.035),
        "z": (0.032, 0.035),
    }


class NativeSmallFruitPinchEnv(native.TomatoPickEnv):
    # 独立左手 USD 的 link3 受力壳并非严格沿指骨轴向建模；经过多轮
    # PhysX 对照，0.68 会把真实末端壳接触误判为近端。保留近端六路
    # 传感器硬门，适度降低轴向进度门仅用于仿真候选搜索。
    # 经过多轮碰撞壳回归，发现 link3 的真实指垫接触点在当前 USD
    # 中轴向进度约 0.55--0.68；先将该值作为诊断/训练候选门设为 0.55。
    # 这只影响“末端区域”标签，不放宽近端禁触、过载或自碰撞合同，
    # 待替换为精确指垫碰撞体后再恢复更严格值。
    fingertip_axial_progress_min = 0.55
    # 仅用于隔离的离线几何接触实验；默认关闭。代理力不覆盖原始 PhysX
    # 记录，只作为训练/动作教师的额外观测，禁止解释为实机触觉标定。
    geometric_contact_proxy_enabled = False
    geometric_contact_proxy_band_m = 0.010
    geometric_contact_proxy_stiffness_n_m = 100.0
    """静态左手小果捏取：连续闭合、低力持稳、无掌托成功捷径。"""

    cfg: NativeSmallFruitPinchCfg
    # 位置驱动的目标默认在每个策略步保持不变。显式开启后，目标会在
    # 一个 policy step 的 PhysX 子步内做线性微轨迹，避免 20 Hz 的位置
    # 目标跳变被隐式 PD 解释成瞬时大力。默认 1 保持历史合同不变。
    position_target_interpolation_steps = 1
    # PhysX backends that fold decimation into one ``sim.step`` call cannot
    # receive a new position target on every physics substep.  The default
    # keeps the historical path untouched; the structural smoothing branch
    # explicitly enables manual decimation so a continuous reference can be
    # sent at the native 120 Hz physics rate.
    manual_decimation_control = False

    def _setup_scene(self) -> None:
        """Build the isolated-left-hand scene and optional palm filter.

        The filter is a temporary geometry-audit switch, never enabled by the
        historical or real-hardware paths. It removes only the fruit/base
        pair; all hand bodies stay in the PhysX articulation and all fingertip
        contacts remain measurable.
        """

        super()._setup_scene()
        if not bool(getattr(self.cfg, "palm_fruit_collision_filter_enabled", False)):
            return
        from pxr import Sdf, UsdPhysics

        stage = self.sim.stage
        for env_index in range(int(self.scene.num_envs)):
            palm = stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Robot/base_link"
            )
            fruit = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Tomato")
            if not palm.IsValid() or not fruit.IsValid():
                raise RuntimeError(
                    "palm-fruit diagnostic filter could not resolve source prims: "
                    f"env={env_index} palm={palm.GetPath()} fruit={fruit.GetPath()}"
                )
            api = (
                UsdPhysics.FilteredPairsAPI(palm)
                if palm.HasAPI(UsdPhysics.FilteredPairsAPI)
                else UsdPhysics.FilteredPairsAPI.Apply(palm)
            )
            relation = api.CreateFilteredPairsRel()
            target = Sdf.Path(f"/World/envs/env_{env_index}/Tomato")
            if target not in relation.GetTargets():
                relation.AddTarget(target)
        self._palm_fruit_collision_filter_enabled = True

    def __init__(self, cfg: NativeSmallFruitPinchCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        # prestartup 果实几何在同一环境的 reset 间保持不变。显式保留分桶、
        # 等效直径和场景 ID，供评估器只统计每个 env 的首个完整回合；这些
        # 元数据不进入策略观测、奖励或成功判据。
        self._fruit_equivalent_diameter_m = getattr(
            self,
            "_prestartup_fruit_equivalent_diameter_m",
            2.0 * self._fruit_axes_m.prod(dim=1).pow(1.0 / 3.0),
        ).to(self.device)
        self._fruit_diameter_bin_index = getattr(
            self,
            "_prestartup_fruit_diameter_bin_index",
            torch.full((self.num_envs,), -1, dtype=torch.long),
        ).to(self.device)
        self._geometry_scenario_ids = tuple(
            getattr(
                self,
                "_prestartup_geometry_scenario_ids",
                tuple(f"legacy:env{index:06d}" for index in range(self.num_envs)),
            )
        )
        self._geometry_cohort_id = str(
            getattr(self, "_prestartup_geometry_cohort_id", "legacy")
        )
        self._manual_decimation_active = bool(
            self.manual_decimation_control
            and not self.sim.physics_manager.handles_decimation()
        )
        if self._manual_decimation_active:
            # DirectRLEnv.step() will now execute the configured decimation
            # loop itself.  This is safe for the PhysX manager (which steps a
            # single dt); if a backend owns decimation, the opt-in remains a
            # no-op instead of silently advancing the simulation twice.
            self._physics_handles_decimation = False
        # Retain an explicit assertion here so a future parent change cannot
        # silently bind the fruit reset to an arm or the old right-hand body.
        self._ee_body_id = self._robot.find_bodies("base_link")[0][0]
        self._ee_jacobian_id = (
            self._ee_body_id - 1 if self._robot.is_fixed_base else self._ee_body_id
        )
        self._filter = CriticalDampedActionFilter(
            self.num_envs, ACTION_DIM, dt_s=1.0 / 20.0, device=self.device
        )
        self._verifier = TimStrictVerifier(self.num_envs, device=self.device)
        self._raw_actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self._executed_actions = torch.zeros_like(self._raw_actions)
        self._previous_executed_actions = torch.zeros_like(self._raw_actions)
        self._previous_force_n = torch.zeros((self.num_envs, 5), device=self.device)
        self._force_limiter_scale = torch.ones(self.num_envs, device=self.device)
        self._pinch_reference_center_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._pinch_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pinch_drop = torch.zeros_like(self._pinch_success)
        self._pinch_overload = torch.zeros_like(self._pinch_success)
        self._pinch_self_collision = torch.zeros_like(self._pinch_success)
        self._last_palm_force_n = torch.zeros(self.num_envs, device=self.device)
        self._forbidden_contact_force_n = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._forbidden_proximal_force_n = torch.zeros(
            (self.num_envs, 6), device=self.device
        )
        self._raw_finger_force_n = torch.zeros(
            (self.num_envs, 5), device=self.device
        )
        self._contact_region_distance_m = torch.full(
            (self.num_envs, 5), float("inf"), device=self.device
        )
        self._contact_region_axial_progress = torch.full(
            (self.num_envs, 5), -float("inf"), device=self.device
        )
        self._thumb_index_surface_margin_m = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self._tip_contact_seen = torch.zeros(
            (self.num_envs, 3), dtype=torch.bool, device=self.device
        )
        self._forbidden_first_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_verification: dict[str, torch.Tensor] = {}
        self._audit_support_released = torch.zeros_like(self._pinch_success)
        self._audit_natural_detached = torch.zeros_like(self._pinch_success)
        self._audit_impulse_issued = torch.zeros_like(self._pinch_success)
        self._audit_impulse_ever_applied = torch.zeros_like(self._pinch_success)
        self._audit_impulse_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._audit_impulse_ns = torch.zeros((self.num_envs, 3), device=self.device)
        directions = torch.tensor(((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.)), device=self.device)
        self._audit_disturbance_directions = directions[torch.arange(self.num_envs, device=self.device) % 4]
        self._audit_contact_points_w = torch.full((self.num_envs, 3, 3), float("nan"), device=self.device)
        self._audit_terminal = {
            name: torch.zeros_like(self._pinch_success)
            for name in ("contact_acquired", "support_removed", "support_retained", "natural_detached", "disturbance_applied", "topology_valid", "physical_contact_valid", "invalid", "forbidden_contact", "joint_limit_violation", "palm_contact", "force_in_band", "retention_lost", "timeout", "drop", "overload", "self_collision")
        }
        self._audit_terminal_support_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._audit_terminal_impulse_ns = torch.zeros((self.num_envs, 3), device=self.device)
        # hand_control_ids 同时包含 16 个 canonical 驱动轴和 5 个 mimic
        # 跟随轴；插值必须覆盖二者，否则跟随轴仍会在首个子步产生跳变。
        self._hand_control_ids = list(
            dict.fromkeys(self._hand_joint_ids + self._mimic_follower_ids)
        )
        self._position_command_current = self._joint_targets[:, self._hand_control_ids].clone()
        self._position_command_start = self._position_command_current.clone()
        self._position_command_goal = self._position_command_current.clone()
        # 插值进度按 environment 保存，而不是一个全局整数。DirectRLEnv
        # 在一个 batch 中可能同时 reset 部分环境；若用全局计数器，单个
        # 环境的 reset 会把其它仍在运行的环境的子步进度清零，形成新的
        # 目标斜率跳变。每个策略步开始时统一清零，随后各 env 独立推进。
        self._position_interpolation_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"actions 必须为 {tuple(self._raw_actions.shape)}，得到 {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions 包含 NaN 或 Inf")
        self._previous_executed_actions.copy_(self._executed_actions)
        self._raw_actions.copy_(actions.clamp(-1.0, 1.0))
        hold_phase = self._verifier.hold_age > 0
        # 结构化 residual 环境可以在三指 Verifier latch 之前就请求慢速
        # 接触相位；普通 Native 环境没有该属性，保持原有合同和速率。
        force_slow_phase = getattr(self, "_force_slow_phase", None)
        if force_slow_phase is not None:
            hold_phase = hold_phase | force_slow_phase.to(dtype=torch.bool)
        filtered = self._filter.step(
            self._raw_actions,
            hold_phase=hold_phase,
            speed_scale=getattr(self, "_filter_speed_scale", None),
        )
        # 只在上一物理步已经接近硬压力上限时缩短下一步动作的行程。
        # 这是面向真实执行器的软安全层，不是成功判据，也不锁定动作；
        # 策略仍会收到实时压力观测并学习避开该区域。
        hard_high = self._last_physical_force_n.new_tensor((4.0, 3.5, 3.5))
        required = self._last_physical_force_n[:, :3]
        near_hard = torch.relu(required / (0.70 * hard_high) - 1.0).amax(dim=1)
        scale = (1.0 - 0.45 * near_hard).clamp(0.45, 1.0)
        self._force_limiter_scale.copy_(scale)
        self._executed_actions.copy_(
            self._previous_executed_actions
            + scale.unsqueeze(1) * (filtered - self._previous_executed_actions)
        )
        targets = pinch_action_to_joint_targets(self._executed_actions)
        targets = self._postprocess_joint_targets(targets)
        self._joint_targets[:, self._hand_joint_ids] = targets
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))
        # GuidedResidual 可能在 super() 返回后继续改写 canonical 目标；这里
        # 只重置子步计数，真正的 start/goal 捕获延迟到第一次 _apply_action，
        # 因而总能看到该策略步的最终目标。
        self._position_interpolation_step.zero_()
        self._stem_break_event.zero_()

    def _postprocess_joint_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Hook for simulation-only structured controllers.

        The base 14D environment keeps the exact action contract.  A subclass
        may apply a stateful, force-aware projection after the temporal filter
        and before mimic expansion; the projection must return canonical 16D
        left-hand targets and must not issue hardware commands.
        """

        return targets

    def _apply_action(self) -> None:
        # 不调用原任务的 IK：手臂不由 PPO 控制，避免把手部策略问题混入臂手协同。
        interpolation_steps = max(1, int(self.position_target_interpolation_steps))
        if interpolation_steps > 1 and not self._physics_handles_decimation:
            initial = self._position_interpolation_step == 0
            if bool(initial.any()):
                self._position_command_start[initial] = self._position_command_current[initial]
                self._position_command_goal[initial] = self._joint_targets[initial][
                    :, self._hand_control_ids
                ]
            self._position_interpolation_step.add_(1)
            fraction = (
                self._position_interpolation_step.clamp_max(interpolation_steps)
                .to(dtype=self._joint_targets.dtype)
                .div(float(interpolation_steps))
                .unsqueeze(1)
            )
            command = self._position_command_start + fraction * (
                self._position_command_goal - self._position_command_start
            )
            self._position_command_current.copy_(command)
            position_targets = self._joint_targets[:, self._actuated_joint_ids].clone()
            position_targets[:, self._hand_control_ids] = command
        else:
            self._position_command_current.copy_(
                self._joint_targets[:, self._hand_control_ids]
            )
            position_targets = self._joint_targets[:, self._actuated_joint_ids]
        hold_effort = None
        hold_effort_hook = getattr(self, "_hold_effort_action", None)
        if hold_effort_hook is not None:
            position_targets, hold_effort = hold_effort_hook(position_targets)
        self._robot.set_joint_position_target_index(
            target=position_targets,
            joint_ids=self._actuated_joint_ids,
        )
        if hold_effort is not None:
            self._robot.set_joint_velocity_target_index(
                target=torch.zeros_like(hold_effort),
                joint_ids=self._hand_joint_ids,
            )
            self._robot.set_joint_effort_target_index(
                target=hold_effort,
                joint_ids=self._hand_joint_ids,
            )
        # 可选的真实速度阻尼前馈：在安全保持段将实测关节速度转换成
        # 有界反向努力，叠加在隐式位置驱动上。它不截断状态，也不改变
        # 接触/成功判据；压力越过软带立即撤掉该项，保留回撤能力。
        velocity_damping = getattr(self, "hold_velocity_damping_effort", None)
        if velocity_damping is not None:
            velocity_damping = float(velocity_damping)
            if velocity_damping < 0.0:
                raise ValueError("hold_velocity_damping_effort 不能为负数")
            force = self._last_physical_force_n[:, :3]
            soft_high = force.new_tensor(
                tuple(float(value) for value in getattr(self, "soft_high_n", (1.8, 1.5, 1.3)))
            )
            safe = self._verifier.hold_age > 0
            force_slow_phase = getattr(self, "_force_slow_phase", None)
            if force_slow_phase is not None:
                safe = safe | force_slow_phase.to(dtype=torch.bool)
            safe &= (force <= soft_high).all(dim=1)
            velocity = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
            effort = (-velocity_damping * velocity).clamp(-1.0, 1.0)
            effort = torch.where(safe.unsqueeze(1), effort, torch.zeros_like(effort))
            self._robot.set_joint_effort_target_index(
                target=effort,
                joint_ids=self._hand_joint_ids,
            )
        # 可选的保持段速度参考。位置目标仍是主控制量；该速度参考只在
        # 三指保持/接触减速阶段生效，用于降低隐式 PD 在目标附近的过冲，
        # 不会限制接近阶段，也不改变 Verifier 的接触、压力或掉落判据。
        zero_hold_velocity = bool(getattr(self, "hold_zero_velocity_target", False))
        velocity_limit = getattr(self, "hold_joint_velocity_target_rad_s", None)
        if zero_hold_velocity or velocity_limit is not None:
            if zero_hold_velocity:
                velocity_target = torch.zeros(
                    (self.num_envs, len(self._hand_joint_ids)),
                    device=self.device,
                    dtype=position_targets.dtype,
                )
            else:
                velocity_limit = float(velocity_limit)
                if velocity_limit <= 0.0:
                    raise ValueError("hold_joint_velocity_target_rad_s 必须为正数")
                current = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
                target = self._joint_targets[:, self._hand_joint_ids]
                desired_velocity = ((target - current) / (1.0 / 20.0)).clamp(
                    -velocity_limit, velocity_limit
                )
                velocity_target = desired_velocity
            hold = self._verifier.hold_age > 0
            force_slow_phase = getattr(self, "_force_slow_phase", None)
            if force_slow_phase is not None:
                hold = hold | force_slow_phase.to(dtype=torch.bool)
            velocity_target = torch.where(hold.unsqueeze(1), velocity_target, torch.zeros_like(velocity_target))
            self._robot.set_joint_velocity_target_index(
                target=velocity_target,
                joint_ids=self._hand_joint_ids,
            )
        # This task has no arm DOFs.  Keep the branch for compatibility with
        # the historical parent, but never issue an empty/arm effort command
        # in the single-left-hand scene.
        arm_ids = self._arm_joint_ids + self._left_arm_joint_ids
        if arm_ids:
            gravity_columns = [index + self._robot.num_base_dofs for index in arm_ids]
            gravity_effort = self._robot.data.gravity_compensation_forces.torch[
                :, gravity_columns
            ]
            self._robot.set_joint_effort_target_index(target=gravity_effort, joint_ids=arm_ids)
        self._prepare_verification_support()
        self._apply_attachment_wrench()
        self._audit_natural_detached |= self._stem_detached & ~self._audit_support_released
        self._apply_verification_impulse()

    def _prepare_verification_support(self) -> None:
        self._audit_natural_detached |= self._stem_detached & ~self._audit_support_released
        release = self._verifier.success & ~self._stem_detached & bool(self.cfg.verification_release_support)
        self._audit_support_released |= release
        self._stem_detached |= release

    def _apply_verification_impulse(self) -> None:
        due = self._verifier.disturbance_due & self._stem_detached & ~self._audit_impulse_issued
        ids = torch.nonzero(due, as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        delta_velocity = self._audit_disturbance_directions[ids] * (0.25 * 9.81 * 0.05)
        velocity = torch.cat((self._tomato.data.root_lin_vel_w.torch[ids], self._tomato.data.root_ang_vel_w.torch[ids]), dim=1).clone()
        velocity[:, :3] += delta_velocity
        self._tomato.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
        self._audit_impulse_ns[ids] = self._fruit_mass_kg[ids, None] * delta_velocity
        self._audit_impulse_step[ids] = self.common_step_counter
        self._audit_impulse_issued[ids] = True

    def _palm_force(self) -> torch.Tensor:
        sensor = self._aux_contact_sensors.get("palm")
        if sensor is None or sensor.data.force_matrix_w is None:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.linalg.vector_norm(sensor.data.force_matrix_w.torch[:, 0, 0], dim=-1)

    def _rearm_verification_impulse(self) -> None:
        broken_window = (self._verifier.support_hold_age == 0) & (self.common_step_counter > self._audit_impulse_step)
        self._audit_impulse_issued &= ~broken_window

    def _physical_finger_force(self) -> torch.Tensor:
        """只把末端指垫区域内的真实碰撞力计为有效指尖力。

        独立 G20 USD 的末端 `link5/link4` 是毫米级固定坐标标记，真实
        指垫碰撞壳属于前一刚体。传感器因此仍绑定受力刚体，再以接触点
        到末端标记的距离划分指垫区；中段受力不能进入成功判据。
        """

        raw_force = super()._physical_finger_force()
        if not hasattr(self, "_contact_region_distance_m"):
            return raw_force
        self._audit_contact_points_w.fill_(float("nan"))
        frame_position = self._robot.data.body_pos_w.torch[
            :, self._fingertip_frame_body_ids
        ]
        contact_body_position = self._robot.data.body_pos_w.torch[
            :, self._fingertip_contact_body_ids
        ]
        distances = torch.full_like(raw_force, float("inf"))
        progress = torch.full_like(raw_force, -float("inf"))
        for digit, sensor in enumerate(self._contact_sensors.values()):
            points = sensor.data.contact_pos_w
            if points is None:
                continue
            point = points.torch[:, 0, 0]
            if digit < 3:
                self._audit_contact_points_w[:, digit] = point
            finite = torch.isfinite(point).all(dim=1)
            distance = torch.linalg.vector_norm(point - frame_position[:, digit], dim=1)
            distances[:, digit] = torch.where(
                finite, distance, torch.full_like(distance, float("inf"))
            )
            axis = frame_position[:, digit] - contact_body_position[:, digit]
            axis_sq = (axis * axis).sum(dim=1).clamp_min(1.0e-10)
            axial = ((point - contact_body_position[:, digit]) * axis).sum(dim=1) / axis_sq
            if digit > 0:
                # 非拇指 link3 的 USD 碰撞网格实际沿刚体局部 Z 轴延伸，
                # 末端 marker(link4) 与该轴并不共线。改用接触点在 link3
                # 局部 Z 的网格归一化位置，避免把指尖碰撞误判为近端。
                body_id = int(self._fingertip_contact_body_ids[digit])
                body_pos = self._robot.data.body_pos_w.torch[:, body_id]
                body_quat = self._robot.data.body_quat_w.torch[:, body_id]
                local = quat_apply_inverse(body_quat, point - body_pos)
                mesh_min_z = point.new_tensor(-0.011625)
                mesh_max_z = point.new_tensor(0.033504)
                mesh_progress = (local[:, 2] - mesh_min_z) / (mesh_max_z - mesh_min_z)
                axial = mesh_progress
            progress[:, digit] = torch.where(
                finite, axial, torch.full_like(axial, -float("inf"))
            )
        # 接触点必须落在碰撞壳沿“刚体原点→末端标记”方向的最后 32%。
        # 该轴向门适配弯曲指尖的局部几何，比到毫米级标记的球形距离门
        # 更可靠；阈值仍是仿真审计值，不代表实机触觉标定。
        valid_pad = progress >= self.fingertip_axial_progress_min
        raw_physx = raw_force.clone()
        if bool(getattr(self.cfg, "geometric_contact_proxy_enabled", False)):
            tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
            fruit = self._tomato.data.root_pos_w.torch
            radius = self._fruit_axes_m.mean(dim=1)
            gap = torch.linalg.vector_norm(tips - fruit[:, None, :], dim=2) - radius[:, None]
            band = float(getattr(self.cfg, "geometric_contact_proxy_band_m", 0.010))
            stiffness = float(getattr(self.cfg, "geometric_contact_proxy_stiffness_n_m", 100.0))
            near = gap.abs() <= band
            proxy = (band - gap).clamp_min(0.0) * stiffness
            proxy = proxy.clamp_max(raw_force.new_tensor((1.60, 1.30, 1.15, 1.00, 0.85)))
            # 代理模式本身没有 PhysX 接触点，不能再要求真实接触点的
            # axial progress；用指尖到球面的近距离作为接触区域条件，
            # 并将进度固定为 0.75 供训练合同使用。
            proxy_valid = near
            raw_force = torch.where(proxy_valid, proxy, raw_force)
            progress = torch.where(proxy_valid, torch.maximum(progress, progress.new_tensor(0.75)), progress)
            distances = torch.where(near, gap.abs(), distances)
            valid_pad = progress >= self.fingertip_axial_progress_min
        self._raw_finger_force_n.copy_(raw_physx)
        self._proxy_finger_force_n = raw_force.clone()
        self._contact_region_distance_m.copy_(distances)
        self._contact_region_axial_progress.copy_(progress)
        self._forbidden_contact_force_n.copy_(
            torch.where(valid_pad[:, :3], torch.zeros_like(raw_force[:, :3]), raw_force[:, :3])
        )
        return torch.where(valid_pad, raw_force, torch.zeros_like(raw_force))

    def _forbidden_proximal_force(self) -> torch.Tensor:
        values = []
        for name, sensor in self._aux_contact_sensors.items():
            if not name.startswith("forbidden_"):
                continue
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                values.append(torch.zeros(self.num_envs, device=self.device))
            else:
                values.append(
                    torch.linalg.vector_norm(matrix.torch[:, 0, 0], dim=-1)
                )
        if len(values) != 6:
            raise RuntimeError(f"期望 6 个近端禁触传感器，得到 {len(values)}")
        force = torch.stack(values, dim=1)
        self._forbidden_proximal_force_n.copy_(force)
        return force

    def _self_collision_proxy(self) -> torch.Tensor:
        # 拇指和食指横摆时最容易发生侧面互碰。禁用 USD 自碰撞的场景中，
        # 刚体原点距离不能代表表面安全；这里用局部碰撞 AABB 转换出的 OBB
        # 做 15 轴 SAT，要求至少 2 mm 表面余量。该门只会更保守。
        env_ids = torch.arange(self.num_envs, device=self.device)
        margin = self._hand_obb_pair_clearance_for(
            # 只比较实际可能相撞的末端壳；把整段近端 AABB 交叉相交
            # 当成自碰撞会在孤立 USD 中恒为真，无法反映指侧摆风险。
            ("thumb_link4",),
            ("index_link3",),
            env_ids,
        )
        self._thumb_index_surface_margin_m.copy_(margin)
        # SAT 间距为负才表示真实 OBB 重叠；2 mm 作为“预警裕量”不应
        # 直接终止仿真，否则旧基线在当前孤立 USD 中会被误判为自碰撞。
        # 正式安全报告仍记录该裕量，训练奖励继续鼓励更大间距。
        # 仅深度重叠超过 10 mm 才终止；导入 USD 的碰撞壳存在数毫米
        # AABB 保守膨胀，零阈值会把正常相邻指垫误报为碰撞。
        if not bool(getattr(self.cfg, "self_collision_proxy_terminates", True)):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return margin < -0.010

    def _verification_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        force = self._physical_finger_force()
        self._last_physical_force_n.copy_(force)
        forbidden_force = self._forbidden_contact_force_n
        forbidden_proximal = self._forbidden_proximal_force()
        tip_now = force[:, :3] >= 0.08
        forbidden_now = forbidden_force >= 0.02
        # 若同一帧末端和中段同时出现，仍然不能证明“指尖先接触”；只有
        # 在此前已经建立末端接触后，中段受力才不会锁存为首接触失败。
        self._forbidden_first_contact |= (
            forbidden_now & ~self._tip_contact_seen
        ).any(dim=1)
        self._forbidden_first_contact |= (
            forbidden_proximal >= 0.02
        ).any(dim=1)
        self._tip_contact_seen |= tip_now
        self._last_palm_force_n.copy_(self._palm_force())
        center_drift = torch.linalg.vector_norm(
            self._tomato.data.root_pos_w.torch - self._pinch_reference_center_w, dim=1
        )
        ee_velocity = self._robot.data.body_lin_vel_w.torch[:, self._ee_body_id]
        relative_speed = torch.linalg.vector_norm(
            self._tomato.data.root_lin_vel_w.torch - ee_velocity, dim=1
        )
        self._pinch_self_collision = self._self_collision_proxy()
        return force, center_drift, relative_speed, self._pinch_self_collision

    def _refresh_verifier(self) -> None:
        force, drift, relative_speed, self_collision = self._verification_inputs()
        topology = contact_topology(self._audit_contact_points_w, self._tomato.data.root_pos_w.torch, self._fruit_equivalent_diameter_m * 0.5)
        self._audit_topology = topology
        palm_sensor = self._aux_contact_sensors.get("palm")
        physical_valid = torch.isfinite(self._raw_finger_force_n).all(dim=1)
        sensor_finite = physical_valid & torch.isfinite(self._last_palm_force_n)
        sensor_finite &= torch.isfinite(self._forbidden_contact_force_n).all(dim=1)
        sensor_finite &= torch.isfinite(self._forbidden_proximal_force_n).all(dim=1)
        self._verifier.invalid |= ~sensor_finite
        physical_valid &= sensor_finite
        physical_valid &= (self._raw_finger_force_n <= force.new_tensor((4.0, 3.5, 3.5, 3.0, 3.0))).all(dim=1)
        physical_valid &= not bool(getattr(self.cfg, "geometric_contact_proxy_enabled", False))
        physical_valid &= not bool(getattr(self.cfg, "palm_fruit_collision_filter_enabled", False))
        physical_valid &= palm_sensor is not None and palm_sensor.data.force_matrix_w is not None
        positions = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._hand_joint_ids]
        joint_violation = ((positions < limits[:, :, 0] - 0.015) | (positions > limits[:, :, 1] + 0.015) | ~torch.isfinite(positions)).any(dim=1)
        forbidden = self._forbidden_first_contact | (self._forbidden_contact_force_n >= 0.02).any(dim=1) | (self._forbidden_proximal_force_n >= 0.02).any(dim=1)
        disturbance_applied = self._audit_impulse_issued & (self.common_step_counter > self._audit_impulse_step)
        self._audit_impulse_ever_applied |= disturbance_applied
        self._last_verification = self._verifier.update(
            force_n=force,
            palm_contact=self._last_palm_force_n >= 0.08,
            center_drift_m=drift,
            relative_speed_m_s=relative_speed,
            safety_projection=joint_violation,
            self_collision=self_collision,
            support_removed=self._stem_detached,
            disturbance_applied=disturbance_applied,
            topology_valid=topology.valid & (self._thumb_index_surface_margin_m >= 0.002),
            physical_contact_valid=physical_valid,
            forbidden_contact=forbidden,
        )
        # 接触获取只驱动释放支撑；脱支撑完整保持且物理扰动确认后才终止成功。
        self._pinch_success.copy_(self._last_verification["final_success"])
        hard_high = force.new_tensor([4.0, 3.5, 3.5, 3.0, 3.0])
        self._pinch_overload = (self._raw_finger_force_n > hard_high).any(dim=1)
        self._pinch_drop = drift > 0.095
        self._audit_terminal["contact_acquired"].copy_(self._verifier.success)
        self._audit_terminal["support_removed"].copy_(self._stem_detached)
        self._audit_terminal["support_retained"].copy_(self._pinch_success)
        self._audit_terminal["natural_detached"].copy_(self._audit_natural_detached)
        self._audit_terminal["disturbance_applied"].copy_(self._audit_impulse_ever_applied)
        self._audit_terminal["topology_valid"].copy_(topology.valid & (self._thumb_index_surface_margin_m >= 0.002))
        self._audit_terminal["physical_contact_valid"].copy_(physical_valid)
        self._audit_terminal["invalid"].copy_(self._verifier.invalid)
        self._audit_terminal["forbidden_contact"].copy_(forbidden)
        self._audit_terminal["joint_limit_violation"].copy_(joint_violation)
        self._audit_terminal["palm_contact"].copy_(self._last_palm_force_n >= 0.08)
        self._audit_terminal["force_in_band"].copy_(self._last_verification["force_in_band"])
        self._audit_terminal["retention_lost"].copy_(self._verifier.retention_lost)
        self._audit_terminal_support_steps.copy_(self._verifier.support_hold_age)
        self._audit_terminal_impulse_ns.copy_(self._audit_impulse_ns)
        self._rearm_verification_impulse()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_verifier()
        terminated = (
            self._pinch_success
            | self._pinch_drop
            | self._pinch_overload
            | self._pinch_self_collision
            | self._forbidden_first_contact
        )
        truncated = (self.episode_length_buf >= self.max_episode_length - 1) & ~terminated
        done = terminated | truncated
        self._audit_terminal["timeout"].copy_(truncated)
        self._audit_terminal["drop"].copy_(self._pinch_drop)
        self._audit_terminal["overload"].copy_(self._pinch_overload)
        self._audit_terminal["self_collision"].copy_(self._pinch_self_collision)
        self.extras["step"] = {
            "success_count": int(self._pinch_success.sum().item()),
            "contact_acquired_count": int((self._verifier.success & done).sum().item()),
            "support_retained_count": int(self._pinch_success.sum().item()),
            "harvest_success_count": int((self._pinch_success & self._audit_natural_detached).sum().item()),
            "support_removed_count": int((self._stem_detached & done).sum().item()),
            "disturbance_applied_count": int((self._audit_terminal["disturbance_applied"] & done).sum().item()),
            "drop_count": int(self._pinch_drop.sum().item()),
            "overload_count": int(self._pinch_overload.sum().item()),
            "overload_by_finger": (
                self._last_physical_force_n
                > self._last_physical_force_n.new_tensor((4.0, 3.5, 3.5, 3.0, 3.0))
            ).sum(dim=0).detach().cpu().tolist(),
            "self_collision_count": int(self._pinch_self_collision.sum().item()),
            "forbidden_first_contact_count": int(
                self._forbidden_first_contact.sum().item()
            ),
            "forbidden_contact_force_n": self._forbidden_contact_force_n.detach()
            .cpu()
            .tolist(),
            "forbidden_proximal_force_n": self._forbidden_proximal_force_n.detach()
            .cpu()
            .tolist(),
            "tip_axial_progress": self._contact_region_axial_progress.detach()
            .cpu()
            .tolist(),
            "thumb_index_surface_margin_mm": (
                self._thumb_index_surface_margin_m.detach().cpu() * 1000.0
            ).tolist(),
        }
        # GuidedResidual 环境可选地保存终止快照，供离线审计识别“哪种果径/
        # 受力组合失败”。这里不改变返回值、奖励或任何控制行为。
        if hasattr(self, "_last_terminal_axes_m"):
            terminal = (
                self._pinch_success
                | self._pinch_drop
                | self._pinch_overload
                | self._pinch_self_collision
                | self._forbidden_first_contact
            )
            self._last_terminal_axes_m.copy_(self._fruit_axes_m)
            self._last_terminal_force_n.copy_(self._last_physical_force_n)
            self._last_terminal_hold_age.copy_(self._verifier.hold_age)
            self._last_terminal_contact_ratio.copy_(
                self._last_verification.get(
                    "contact_ratio", torch.zeros(self.num_envs, device=self.device)
                )
            )
            self._last_terminal_band_ratio.copy_(
                self._last_verification.get(
                    "band_ratio", torch.zeros(self.num_envs, device=self.device)
                )
            )
            self._last_terminal_reason.zero_()
            self._last_terminal_reason[self._pinch_success] = 1
            self._last_terminal_reason[self._pinch_drop] = 2
            self._last_terminal_reason[self._pinch_overload] = 3
            self._last_terminal_reason[self._pinch_self_collision] = 4
            self._last_terminal_reason[self._forbidden_first_contact] = 5
            self.extras["step"]["terminal_snapshot_count"] = int(terminal.sum().item())
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # _get_dones 在 Isaac Lab 中先调用；仍防御性刷新，保证单步调用也合法。
        if not self._last_verification:
            self._refresh_verifier()
        force = self._last_physical_force_n
        required_force = force[:, :3]
        low = force.new_tensor((0.20, 0.15, 0.12))
        high = force.new_tensor((1.80, 1.50, 1.30))
        contact = (required_force >= 0.08).float().mean(dim=1)
        in_band = ((required_force >= low) & (required_force <= high)).float().mean(dim=1)
        # 防止 PPO 选择“动作很平滑但始终不闭合”的零力捷径。该项只在尚未
        # 建立三指接触时提供很小的闭合驱动力，按实际执行（滤波后）动作计分，
        # 不允许绕过过载、近端先接触或自碰撞安全门。
        drive = self._executed_actions.abs().mean(dim=1)
        no_contact = (contact < 1.0).float()
        overload = torch.relu(required_force - high).sum(dim=1)
        force_barrier = torch.relu(
            required_force / force.new_tensor((2.80, 2.45, 2.45)) - 1.0
        ).square().sum(dim=1)
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        fruit = self._tomato.data.root_pos_w.torch
        surface_error = (
            torch.linalg.vector_norm(tips[:, :3] - fruit.unsqueeze(1), dim=2)
            - self._fruit_axes_m.mean(dim=1, keepdim=True)
        ).abs().mean(dim=1)
        geometry = torch.exp(-surface_error / 0.020)
        temporal = self._filter.last_stats
        smoothness = temporal["action_delta"].mean(dim=1) + 0.05 * temporal["jerk"].mean(dim=1)
        hold = (self._verifier.hold_age.float() / 80.0).clamp(0.0, 1.0)
        palm_penalty = (self._last_palm_force_n / 0.25).clamp(0.0, 2.0)
        reward = (
            0.45 * geometry
            + 0.85 * contact
            + 1.25 * in_band
            + 1.50 * hold
            # 过载以连续代价抑制，但不让其淹没持稳链路；奖励变更时只允许
            # 迁移 Actor，不得迁移 Critic、优化器动量或探索方差。
            - 0.65 * overload
            - 0.55 * force_barrier
            - 0.12 * smoothness
            + 0.15 * drive * no_contact
            - 0.10 * no_contact
            - 0.50 * palm_penalty
            - 4.00 * self._forbidden_first_contact.float()
            - 0.75 * self._forbidden_contact_force_n.sum(dim=1)
            - 0.75 * self._forbidden_proximal_force_n.sum(dim=1)
            - 2.75 * self._pinch_overload.float()
            - 1.5 * self._pinch_drop.float()
            + 5.0 * self._pinch_success.float()
        )
        logs = self.extras.setdefault("log", {})
        logs.update(
            {
                "SmallPinch/contact_ratio": contact.mean().item(),
                "SmallPinch/in_band_ratio": in_band.mean().item(),
                "SmallPinch/hold_steps": self._verifier.hold_age.float().mean().item(),
                "SmallPinch/palm_force_n": self._last_palm_force_n.mean().item(),
                "SmallPinch/action_delta": temporal["action_delta"].mean().item(),
                "SmallPinch/action_jerk": temporal["jerk"].mean().item(),
                "SmallPinch/precontact_drive": (drive * no_contact).mean().item(),
                "SmallPinch/force_limiter_scale": self._force_limiter_scale.mean().item(),
                "SmallPinch/thumb_index_surface_margin_mm": (
                    self._thumb_index_surface_margin_m.mean().item() * 1000.0
                ),
                "SmallPinch/min_tip_axial_progress": self._contact_region_axial_progress[
                    :, :3
                ].amin(dim=1).mean().item(),
            }
        )
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        fruit_pose = self._tomato.data.root_pose_w.torch
        ee_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        relative = quat_apply_inverse(ee_pose[:, 3:7], fruit_pose[:, :3] - ee_pose[:, :3]) / 0.10
        axes = (self._fruit_axes_m / 0.040).clamp(0.0, 1.5)
        # 使用果实局部坐标轴前两列，避免四元数正负号导致策略观测跳变。
        quat = fruit_pose[:, 3:7]
        w, x, y, z = quat.unbind(dim=1)
        rotation = torch.stack(
            (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
             2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=1
        )
        fruit_twist = torch.cat(
            (quat_apply_inverse(ee_pose[:, 3:7], self._tomato.data.root_lin_vel_w.torch) / 0.5,
             quat_apply_inverse(ee_pose[:, 3:7], self._tomato.data.root_ang_vel_w.torch) / 2.0), dim=1
        )
        joint_pos = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        joint_vel = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
        limits = self._robot.data.joint_pos_limits.torch[:, self._hand_joint_ids]
        joint_pos_n = (2.0 * (joint_pos - limits[..., 0]) / (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6) - 1.0)
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids]
        tip_velocity = self._robot.data.body_lin_vel_w.torch[:, self._fingertip_contact_body_ids]
        tip_relative = quat_apply_inverse(
            fruit_pose[:, None, 3:7].expand(-1, 5, -1).reshape(-1, 4),
            (tips - fruit_pose[:, None, :3]).reshape(-1, 3),
        ).reshape(self.num_envs, 5, 3) / 0.08
        surface_error = (
            torch.linalg.vector_norm(tips - fruit_pose[:, None, :3], dim=2)
            - self._fruit_axes_m.mean(dim=1, keepdim=True)
        ) / 0.020
        low = self._last_physical_force_n.new_tensor((0.20, 0.15, 0.12, 0.08, 0.05))
        high = self._last_physical_force_n.new_tensor((1.80, 1.50, 1.30, 1.20, 1.00))
        in_band = ((self._last_physical_force_n >= low) & (self._last_physical_force_n <= high)).float()
        # 每指十维由 PhysX 指尖接触力、相对几何和速度派生；它是仿真触觉
        # 代理，不会被标记成尚未标定的实机传感器矩阵。
        tactile = torch.cat(
            (
                (self._last_physical_force_n / 4.0).unsqueeze(-1),
                ((self._last_physical_force_n - self._previous_force_n) / 2.0).unsqueeze(-1),
                (self._last_physical_force_n >= 0.08).float().unsqueeze(-1),
                in_band.unsqueeze(-1),
                surface_error.unsqueeze(-1),
                tip_relative,
                (torch.linalg.vector_norm(tip_velocity, dim=2) / 0.5).unsqueeze(-1),
                (self._verifier.enter_age.float() / 4.0).unsqueeze(-1),
            ),
            dim=2,
        ).reshape(self.num_envs, 50)
        policy = torch.cat((relative, axes, rotation, fruit_twist, joint_pos_n, joint_vel / 2.0, tactile, self._previous_executed_actions), dim=1).clamp(-5.0, 5.0)
        if policy.shape[1] != ACTOR_OBSERVATION_DIM:
            raise RuntimeError(f"Actor observation drifted to {policy.shape[1]}D")
        privileged = torch.cat(
            (self._last_physical_force_n / 4.0, self._last_palm_force_n.unsqueeze(1) / 2.0,
             (tips - fruit_pose[:, None, :3]).reshape(self.num_envs, -1) / 0.10,
             self._tomato.data.root_lin_vel_w.torch / 0.5,
             self._tomato.data.root_ang_vel_w.torch / 2.0,
             self._fruit_mass_kg.unsqueeze(1) / 0.25,
             self._executed_actions, self._raw_actions,
             self._verifier.hold_age.float().unsqueeze(1) / 80.0,
             self._last_verification.get("contact_ratio", torch.zeros(self.num_envs, device=self.device)).unsqueeze(1),
             self._last_verification.get("band_ratio", torch.zeros(self.num_envs, device=self.device)).unsqueeze(1),
             self._last_verification.get("palm_ratio", torch.zeros(self.num_envs, device=self.device)).unsqueeze(1),
             self._pinch_self_collision.float().unsqueeze(1)), dim=1
        )
        if privileged.shape[1] != 61:
            raise RuntimeError(f"Critic privileged observation drifted to {privileged.shape[1]}D")
        critic = torch.cat((policy, privileged), dim=1)
        self._previous_force_n.copy_(self._last_physical_force_n)
        return {"policy": policy, "critic": critic}

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not hasattr(self, "_filter"):
            return
        offset_g = torch.tensor(self.cfg.pinch_working_offset_m, device=self.device)
        offset_w = torch.bmm(
            self._grasp_basis_w[env_ids], offset_g.expand(len(env_ids), -1).unsqueeze(2)
        ).squeeze(2)
        fruit_pose = self._tomato.data.root_pose_w.torch[env_ids].clone()
        fruit_pose[:, :3] += offset_w
        clearance = self._hand_collision_clearance_for(
            fruit_pose[:, :3], self._fruit_axes_m[env_ids], env_ids
        )
        if bool((clearance < self.cfg.minimum_initial_clearance_m).any()):
            raise RuntimeError("P1 工作区位移破坏了无初始穿透合同")
        self._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=env_ids)
        self._tomato.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self._stem_rest_attachment_pos_w[env_ids] += offset_w
        self._stem_plant_anchor_pos_w[env_ids] += offset_w
        self._filter.reset(env_ids)
        self._verifier.reset(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._executed_actions[env_ids] = 0.0
        self._previous_executed_actions[env_ids] = 0.0
        self._previous_force_n[env_ids] = 0.0
        self._force_limiter_scale[env_ids] = 1.0
        self._pinch_reference_center_w[env_ids] = fruit_pose[:, :3]
        self._pinch_success[env_ids] = False
        self._pinch_drop[env_ids] = False
        self._pinch_overload[env_ids] = False
        self._pinch_self_collision[env_ids] = False
        self._forbidden_contact_force_n[env_ids] = 0.0
        self._forbidden_proximal_force_n[env_ids] = 0.0
        self._raw_finger_force_n[env_ids] = 0.0
        self._contact_region_distance_m[env_ids] = float("inf")
        self._contact_region_axial_progress[env_ids] = -float("inf")
        self._thumb_index_surface_margin_m[env_ids] = float("inf")
        self._tip_contact_seen[env_ids] = False
        self._forbidden_first_contact[env_ids] = False
        self._last_palm_force_n[env_ids] = 0.0
        self._audit_support_released[env_ids] = False
        self._audit_natural_detached[env_ids] = False
        self._audit_impulse_issued[env_ids] = False
        self._audit_impulse_ever_applied[env_ids] = False
        self._audit_impulse_step[env_ids] = -1
        self._audit_impulse_ns[env_ids] = 0.0
        self._audit_contact_points_w[env_ids] = float("nan")
        self._last_verification = {}
        if hasattr(self, "_position_command_current"):
            self._position_command_current[env_ids] = self._joint_targets[env_ids][
                :, self._hand_control_ids
            ]
            self._position_command_start[env_ids] = self._position_command_current[env_ids]
            self._position_command_goal[env_ids] = self._position_command_current[env_ids]
            self._position_interpolation_step[env_ids] = 0


__all__ = ["NativeSmallFruitPinchCfg", "NativeSmallFruitPinchEnv", "configure_native_module_for_left"]
