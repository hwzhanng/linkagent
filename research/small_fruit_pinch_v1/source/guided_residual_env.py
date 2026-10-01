"""结构化的 compliant-prior + PPO residual 小果捏取环境。

第三次克制调参失败后，根因审计表明：在当前接触工作区内，纯 PPO 的高斯
探索会在每一个物理步同时改变 14 个绝对关节目标，先破坏已经存在的三指
接触，再从稀疏的 80+40 步判据中学习恢复，样本效率很低。本模块不把成功
姿态写成终点脚本，而是加入一个可解释、带压力反馈的低层 compliant prior：

* P1 动作只作为初始手型，不直接判定成功；
* 每个必需手指根据上一物理步的真实 PhysX 接触力调节自己的近端和远端
  残差，低于目标力缓慢闭合，接近软上限则回撤；
* PPO 只输出小幅 residual，仍可使用位置、速度和触觉观测学习协调、
  纠偏和扰动恢复；
* 二阶临界阻尼、安全投影、TIM strict verifier 与历史分支完全复用。

该结构是仿真训练专用的“策略 + 低层力反馈”组合，不改变历史 89% 权重，
也不允许把任何输出发送到实机。
"""

from __future__ import annotations

import torch

from .native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from .pinch_action_manifold import pinch_action_to_joint_targets
from .temporal import CriticalDampedActionFilter, TemporalLimits


P1_GUIDE_ACTION = (
    0.9562,
    -0.5214,
    0.0,
    0.0,
    0.4017,
    -0.7592,
    0.2345,
    -0.9126,
    0.0,
    0.0,
    -0.3269,
    0.0349,
    0.0,
    0.0,
)

# 由孤立左手 FK/PhysX 扫描得到的 B0 工作区弱先验。它只定义一个可行的
# 初始闭合方向；PPO residual、触觉门控和后续扰动仍然可以改变目标，因而
# 不能把它当作成功终姿或回放脚本。保留 P1 常量用于历史回归和消融。
WORKSPACE_GUIDE_ACTION = (
    0.8239,
    -0.8590,
    0.0811,
    0.6406,
    0.4097,
    0.2718,
    -0.7128,
    0.5314,
    -0.9198,
    0.1658,
    -0.7514,
    -0.8352,
    -0.5911,
    0.7236,
)


class GuidedResidualSmallFruitPinchEnv(NativeSmallFruitPinchEnv):
    """在可学习残差外叠加三指压力闭环先验。"""

    # 残差缩放在动作域内，避免重新引入旧实验的 0.20 高斯噪声破坏接触。
    residual_scale = 0.0
    # 可选的 B0 几何弱先验。默认仍为历史 P1，避免无提示地改变旧分支；
    # 本轮 64--70 mm 物理正确性训练显式切换为 WORKSPACE_GUIDE_ACTION。
    guide_action_values = P1_GUIDE_ACTION
    # 新增的食指/中指远端残差使用独立权限。默认 None 表示沿用
    # residual_scale；结构化候选可让根部保持低幅、指尖在接触后承担更
    # 多的几何补偿，但仍经过同一接触/压力门控，避免退化成固定远端姿态。
    distal_residual_scale = None
    guide_feedback_enabled = False
    # 早期版本在 0.08 N 的首次接触瞬间锁存绝对关节目标。探针证明这会
    # 把目标固定在冲击瞬间并制造过载；当前结构只使用力感知的小步投影，
    # 不锁死策略的连续调节能力。保留开关仅用于回归旧分支，不作为新训练
    # 的默认行为。
    digit_latch_enabled = False
    precontact_residual_fraction = 0.0
    # 训练消融开关：默认允许逐指接触后开放该指残差；在策略容易在
    # 三指尚未齐备时破坏先验的候选中，可要求三指全部接触后再开放，
    # 保留连续调节但避免“先接触的一指带偏其余手指”。
    residual_require_all_contact = False
    # 结构重构候选：PPO 残差先经过一个力感知 trust-region，再叠加到
    # compliant prior。三指未同时接触时残差保持为零；接触后每指只允许
    # 朝“加压/回撤”的安全方向在带外微调，并在力带内显著收缩。它是
    # 可关闭的仿真消融开关，不改变 Verifier 的成功/过载门。
    residual_trust_region_enabled = False
    residual_trust_warning_fraction = 0.60
    residual_trust_in_band_scale = 0.10
    residual_trust_zero_global = True
    # 小果三指捏取的必需集合是拇指/食指/中指。无名指和小指只保留
    # 诊断传感器，不应因全局协同闭合而撞上果实并触发非目标过载。
    isolate_nonrequired_digits = True
    # 先验默认沿用 P1 的可行闭合；只在接近软上沿时回撤，不在低力侧
    # 继续追一个固定力。这样不会把已验证的可达动作推向硬上限。
    soft_high_n = (1.80, 1.50, 1.30)
    # 低通压力误差和目标动作，抑制触觉临界附近的反向抽动。
    force_error_alpha = 0.18
    guide_action_alpha = 0.22
    proportional_gain = 0.50
    derivative_gain = 0.03
    # 方向探针在同一 P1 接触状态下确认：三根手指的正向 residual
    # 都会降低对应指尖力，负向会增加接触力。因此低层回撤统一使用
    # 正号；不能把几何先验的“寻触方向”(拇指正、食中指负)误用为
    # 压力回撤方向。
    force_retract_sign = (1.0, 1.0, 1.0)
    warning_fraction = 0.72
    max_guide_offset = 0.08
    # P1 的可行力带允许在软上沿附近有 PhysX 瞬态波动；保护投影只在
    # 略高于软上沿时介入，避免把已验证的低力接触过早打开。
    guard_margin_n = (0.20, 0.20, 0.20, 0.12, 0.10)
    # 可选的目标投影提前线。默认 None 保持历史 soft_high+margin；新安全
    # 分支可在 soft_high 的比例处沿预抓取目标回撤，专门抑制一个策略步
    # 内的力冲量。它不改变硬判据，只改变发送给仿真的关节目标。
    projection_guard_fraction = None
    retract_fraction = 0.08
    # 接触后的残差在接近硬上限时按指单独收缩，而不是把整只手冻结。
    # 最低保留 5% 权限用于恢复丢失接触，避免安全层成为固定姿态脚本。
    force_gate_floor = 0.05
    force_gate_enabled = True
    # None preserves the original gate (soft-high + guard margin).  A value
    # below one starts shrinking learned residuals earlier, before a fast PD
    # step can jump across the hard force boundary; it is an explicit safety
    # ablation for post-branch6 training, not a success-rule relaxation.
    force_gate_start_fraction = None
    # 低压力 shaping 只作为连续学习信号，不改变 Verifier 的硬门。
    force_center_targets_n = (0.85, 0.65, 0.45)
    force_center_width_n = (0.90, 0.75, 0.65)
    force_center_reward = 0.30
    # 物理扫描发现 P1 在果径上沿偏大端会让拇指/食指同时进入带宽上沿。
    # 该弱先验不是成功姿态：它只在接触前把三根必需手指的局部闭合量
    # 作小幅、可学习的偏置，PPO residual 仍可在完整观测下改变它。默认
    # 关闭以保持 branch1--5 的评估合同不变，新的结构分支显式开启。
    geometry_prior_enabled = False
    geometry_prior_scale = 0.30
    geometry_prior_size_adaptive = False
    geometry_prior_max_scale = 0.30
    geometry_prior_axis_min_m = 0.032
    geometry_prior_axis_max_m = 0.035
    # 可选的接触门控先验：拇指先沿 P1 快速建立接触，之后才施加局部
    # 开放偏置；食指/中指各自接触后分别施加自己的偏置。它仍由实时
    # PhysX 接触力驱动，不锁存终点，也不读取成功标签。
    geometry_prior_after_contact = False
    geometry_prior_residual = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # 可选的低层受力平衡器只使用上一物理步的真实指尖力，不读取 episode
    # 成功标签或固定终姿。它在目标压力附近带死区并限幅，防止力传感器
    # 临界噪声把动作变成反向抽搐；默认同样关闭，便于独立消融。
    force_balance_enabled = False
    force_balance_target_n = (0.85, 0.65, 0.45)
    force_balance_width_n = (0.90, 0.75, 0.65)
    force_balance_gain = 0.10
    force_balance_alpha = 0.12
    force_balance_deadband_n = 0.05
    force_balance_max_offset = 0.22
    # 结构性双通道候选：PPO 仍输出 14D residual，低层另有一个每指的
    # 受力误差通道。它只根据 EMA 指尖力在接触/目标带外给出有界动作偏置，
    # 不读取成功标签，不锁存绝对位姿；默认关闭，先用固定动作探针验证。
    force_servo_enabled = False
    force_servo_target_n = (0.85, 0.65, 0.45)
    # 接触已建立但压力仍明显低于每指工作中心时的微小闭合增益。
    # 默认关闭以保持 branch1--6 的 level-only 合同；启用时仍受同一
    # 方向、EMA、动作限幅和力门控约束，不会写入绝对终姿态。
    force_servo_target_gain = 0.0
    force_servo_contact_n = 0.08
    force_servo_seek_gain = 0.20
    # 接触已经建立但仍低于每指 soft_low 时的微小补压；与 contact_gap
    # 分开，专门提高“80 步内持续在带内”的比例，而不是把目标力硬推高。
    force_servo_band_seek_gain = 0.0
    force_servo_retract_gain = 0.20
    force_servo_alpha = 0.16
    force_servo_max_offset = 0.35
    # 受力变化率通道：只对上一帧 EMA 的正向突增做很小的提前回撤，
    # 用来处理“力已经跨过软线但电流/位置反馈尚未稳定”的单步冲量。
    # 默认关闭，保留原有 level-only servo 合同；正式启用必须经过静态
    # 消融与独立盲评，不能把瞬态保护误当成成功率提升。
    force_servo_rate_gain = 0.0
    force_servo_rate_deadband_n = 0.05
    force_servo_rate_scale_n = 0.25
    # 不在目标力刚超过中心值时立即回撤；只有接近该指软上沿才启动
    # 减压通道，避免“安全控制器”把正常带内接触误判为过载并丢指。
    force_servo_warning_fraction = 0.78
    # 方向探针得到的三指共同局部梯度：负向寻触/加压，正向回撤/减压。
    # 两个方向显式分离，避免高力时继续沿寻触方向推进。
    force_servo_seek_direction = (-1.0, -1.0, -1.0)
    force_servo_retract_direction = (1.0, 1.0, 1.0)
    # 逐指 16D 目标层受力投影。与上面的 action-domain servo 不同，该层
    # 在 14D 映射、PPO residual 和动作滤波之后再工作，因此不会被全局
    # 协同量的非线性耦合抵消。它只在真实上一帧力接近/超过软线时沿
    # ``pregrasp -> current`` 的反方向平滑回撤；低于软下限且已经接触时
    # 仅保留很小的补进量。默认关闭，必须通过 zero-PPO 消融后才能训练。
    direct_force_projection_enabled = False
    direct_force_projection_warning_fraction = 0.60
    direct_force_projection_high_gain = 0.30
    # 可选的逐指高力回撤增益；None 时沿用上面的标量。中指的软上限
    # 更低且在静态探针中出现更大的冲量，因此后续消融可以只加大中指
    # 的保护，而不牺牲拇指的可达性。
    direct_force_projection_high_gain_by_digit = None
    direct_force_projection_low_gain = 0.06
    direct_force_projection_alpha = 0.30
    direct_force_projection_max_open = 0.28
    direct_force_projection_root_fraction = 0.45
    direct_force_projection_contact_n = 0.08
    # 目标层的单步提前量：用当前原始力相对上一帧的上升趋势预测
    # PhysX 冲量，在下一次位置目标发送前提前进入回撤。默认关闭，
    # 只用于候选消融，避免把触觉噪声误放大为动作。
    direct_force_projection_rate_gain = 0.0
    direct_force_projection_rate_deadband_n = 0.05
    direct_force_projection_rate_scale_n = 0.25
    # 逐指相位控制器：这是在三类单变量调整未通过后的结构重构候选。
    # 它不写入终点位姿，而是把每根手指的局部偏置作为有界积分量维护：
    # SEARCH/CLOSE 只在实际压力不足时小步寻触，HOLD 保持已建立的偏置，
    # RETRACT 接近软上限时沿实测减压方向回撤。默认关闭，只有显式候选
    # 才能改变历史 guided-residual 的控制合同。
    phase_controller_enabled = False
    phase_search_step = 0.0030
    phase_close_step = 0.0015
    phase_retract_step = 0.0030
    phase_offset_alpha = 0.30
    phase_max_offset = 0.18
    phase_root_fraction = 0.55
    phase_low_delay_steps = 3
    phase_low_ramp_steps = 5
    phase_hysteresis_n = 0.03
    # 接触后的逐指速率降档；未接触指仍保持正常 approach_speed。
    phase_contact_speed_scale = 0.42
    phase_high_speed_scale = 0.24
    # 可选的每指几何寻触项：用指尖到当前椭球表面的有符号间隙，
    # 只在压力带下方生成极小的闭合目标；进入压力带后不再推进。
    phase_geometry_enabled = False
    phase_geometry_step = 0.0015
    phase_geometry_gap_scale_m = 0.010
    # 逐指压力设定点模式：把动作偏置视作对每指目标力的有界积分响应，
    # 而不是在 soft_low/soft_high 两条线之间反复切换。目标默认位于各自
    # 工作带中部，仍由真实力反馈驱动，PPO residual 可在外层改变目标。
    phase_target_controller_enabled = False
    phase_target_force_n = (0.70, 0.55, 0.30)
    phase_target_step = 0.006
    phase_target_alpha = 0.50
    phase_target_deadband_n = 0.04
    phase_target_error_scale_n = (0.70, 0.55, 0.35)
    phase_target_raw_mix = 0.35
    # 真正的每指压力闭环：未建立接触时沿该指可解释动作维度小步闭合，
    # 建立接触后停止先验推进，超过软上限时由下面的 guard 投影回撤。该
    # 闭环只提供局部低层反射，不读取成功位姿或 episode 标签；PPO 仍可
    # 在接触后学习残差。
    # 物理探针表明 P1 已经能在中心域形成三指接触，直接用固定步长继续
    # 推进反而会把已接触手指推向过载。因此正式结构分支默认关闭这层
    # “无接触即闭合”启发式；接触后的连续调整由 PPO residual、每指受力
    # 门控和恢复权限完成。保留开关供单独诊断，不把它混入发布训练。
    adaptive_closure_enabled = False
    # 拇指沿已由物理探针确认的独立屈曲维度 dim 5 做极小负向闭合；
    # 不能把 opposition dim 4 当作“更紧”，该方向会把拇指带离果实。
    # 食指、中指仍分别用近端和新增远端维度渐进闭合。
    adaptive_closure_steps = (0.0, 0.007, 0.007)
    adaptive_contact_threshold_n = 0.08
    # 结构分支可单独做小幅速率消融；默认值保持 branch1--6 的历史合同。
    # 训练候选若启用更低 approach_speed，必须在相同 320 步时域内复核到达
    # 率，不能用“减速后没撞”冒充成功。
    approach_speed = 0.45
    hold_speed = 0.15
    approach_acceleration = 4.0
    hold_acceleration = 1.5
    natural_frequency_hz = 5.0
    # 可选的 canonical 关节目标跟踪限速。动作域滤波器限制的是 14D
    # policy 信号，手部位置驱动仍可能因较大的关节跨度在首次接触后
    # 保持较高实际速度；启用此项可在三指齐备时直接限制目标每步的
    # 角度变化，避免把物理跟踪振荡误归因于 PPO。None 保持历史合同。
    hold_joint_target_speed_rad_s = None
    # 保持段允许沿预抓取方向更快回撤。若与闭合方向共用同一限速，
    # 过载后的安全回撤会被阻塞，形成“限速反而过载”的闭环；None 表示
    # 与 hold_joint_target_speed_rad_s 相同，保持旧合同。
    hold_joint_target_open_speed_rad_s = None
    # 仅在三指已接触且压力仍在软带内时限制 PhysX 位置驱动的实际速度。
    # 一旦压力越过软带，恢复原始上限以保证安全回撤不被物理限速阻塞。
    hold_hand_velocity_limit_rad_s = None
    # 可选的逐指接触后 PhysX 速度上限。与全局保持限速不同，它只
    # 作用于已经达到接触阈值的拇指/食指/中指驱动轴，未接触手指仍可
    # 正常寻触；高压力时仍恢复默认上限以保证回撤通道畅通。
    contact_hand_velocity_limit_rad_s = None
    # 可选的 16 个 canonical 轴保持段速度上限。用于只约束远端必需轴，
    # 避免全手限速造成根部力不足；None 时沿用全局/逐指合同。
    hold_hand_velocity_limit_by_joint = None
    # 接触后的逐指增益降档。首次接触阶段仍保留原始几何目标，但将该
    # 手指的隐式 PD 刚度/阻尼切换到较温和的值，避免 20 Hz 目标更新在
    # 触觉尚未稳定时造成瞬时力冲量。压力越过软上限立即恢复默认增益，
    # 保留可靠的回撤能力；None 表示不改变历史执行器合同。
    contact_hand_stiffness = None
    contact_hand_damping = None
    # 接触后逐指 canonical 目标的斜率上限。与 PhysX 速度上限不同，
    # 这里限制的是位置命令变化，不会把已经发生的接触速度硬截断，
    # 因而不会因执行器追不上目标而制造新的过载；过压时立即放开。
    contact_hand_target_speed_rad_s = None
    # 保持段可选的临界阻尼增益；只在安全压力带内启用，压力上升时
    # 恢复训练时的原始增益，避免削弱过载回撤响应。
    hold_hand_stiffness = None
    hold_hand_damping = None
    # 可选的 16 个 canonical 主动轴保持段阻尼；用于仅对接触冲击最明显
    # 的远端/拇指根部增加阻尼，避免把所有手指统一降刚度而丢失夹持力。
    # None 时沿用 hold_hand_damping 的全局值或训练默认值。
    hold_hand_damping_by_joint = None
    # 可选的 16 个 canonical 主动轴保持段刚度；与逐关节阻尼配对，
    # 只在三指安全压力带内降低远端/根部位置驱动的追踪冲量。过压或
    # 接触丢失时恢复训练默认刚度，确保仍有完整回撤与重新寻触能力。
    hold_hand_stiffness_by_joint = None
    # 仅保持段给隐式位置驱动写入一个有界速度参考；None 保持历史
    # position-only 合同。该参考不替代位置目标，不作用于接近阶段。
    hold_joint_velocity_target_rad_s = None
    # 可选的保持段“目标—实测位置”误差上限。接触后位置驱动若继续
    # 追逐一个远离当前关节的残差目标，会在低频策略更新之间积累 PD
    # 努力并把接触冲量放大；此限幅只在三指安全保持带内启用，目标仍
    # 可由下一步的触觉误差逐渐移动，过压时恢复完整回撤权限。
    hold_joint_target_error_limit_rad = None
    # 可选的 canonical 16D 逐关节误差窗。标量误差窗会把根部夹持和
    # 远端指尖一并限制，容易在三指已经接触时削弱根部支撑；逐关节窗
    # 允许只收紧实测目标误差最大的远端轴，同时保留根部的力恢复能力。
    # 顺序与 _hand_joint_ids / pinch_action_manifold 完全一致；None 时
    # 继续使用上面的标量合同。该窗只在安全保持段生效，过压立即恢复
    # 原始目标，避免安全层阻塞减压。
    hold_joint_target_error_limit_by_joint = None
    # 可选的保持段 canonical 目标低通。位置驱动在 PhysX 后端按策略步
    # 接收目标时，即使目标斜率本身不大，也可能因隐式 PD 跟踪误差在接触
    # 上沿产生高频实际速度。该滤波只作用于三指安全保持段，并在压力
    # 超过软带时立即旁路，确保减压路径不被阻塞；None 保持历史合同。
    hold_target_filter_alpha = None
    # 可选的保持目标锚定比例。安全保持段将期望目标向当前实测关节
    # 位置收拢：0 表示完全跟随当前位置、1 表示不锚定。该项只用于
    # 诊断位置驱动的跟踪滞后；它不改变力判据，过压或接触丢失时立即
    # 恢复原始目标，保留回撤/恢复能力。
    hold_joint_target_anchor_fraction = None
    # 可选的逐 canonical 轴保持目标锚定比例；用于让根部保留夹持
    # 力，而让食指/中指远端在保持段更快收拢到实测位置，减少远端
    # 隐式 PD 饱和。顺序与 _hand_joint_ids 一致，None 时使用标量项。
    hold_joint_target_anchor_fraction_by_joint = None
    # 保持段可选的实测速度阻尼努力前馈（N·m per rad/s）。与改变
    # PhysX 阻尼不同，该项可在运行时按压力安全门开启/撤销；默认关闭。
    hold_velocity_damping_effort = None
    # 结构性保持候选：三指共同接触后将位置驱动锚在当前实测关节，
    # 只用有界的努力前馈维持压力。该通道默认关闭，避免改变既有
    # branch1--6 的位置驱动合同；启用时仍由真实力和目标方向决定，
    # 不写入成功姿态，也不发送任何实机命令。
    hold_effort_control_enabled = False
    hold_effort_stiffness = 0.0
    hold_effort_damping = 0.35
    hold_effort_max = 0.45
    hold_effort_in_band = 0.16
    hold_effort_low_fraction = 0.78
    hold_effort_high_fraction = 0.55
    # 与直接 effort 模式不同，这个候选保留隐式位置驱动的标称刚度，
    # 但把保持目标写成“当前 q + 力误差对应的小角度偏置”。这样不会
    # 追逐远端绝对目标，也避免在 PhysX 后端切换到零刚度时产生冲击。
    hold_force_impedance_enabled = False
    # 当前姿态锚定的结构性保持候选。与旧的 hold_force_impedance 不同，
    # 该模式不把远端 PPO 目标作为隐式 PD 的位置误差，而是每个物理子步
    # 以实测 q 为中心生成很小的、带真实接触力反馈的等效努力。默认关闭，
    # 只有显式探针/训练合同才启用。
    hold_current_impedance_enabled = False
    hold_impedance_effort_by_digit = (0.12, 0.10, 0.08)
    hold_impedance_force_target_n = (0.70, 0.50, 0.35)
    hold_impedance_force_gain = 0.50
    hold_impedance_start_fraction = 0.65
    hold_impedance_ramp_substeps = 12
    # 可选的保持段驱动努力上限。位置目标控制在接触后仍会产生较大
    # 瞬态力，单纯限速会让 PhysX 额外刹车；这里在三指安全保持段对
    # 所有手部驱动轴施加较低的电机努力上限，压力接近软上沿或进入
    # 回撤阶段立即恢复默认上限。None 保持历史合同。
    hold_hand_effort_limit = None
    # 训练专用的实际关节运动质量奖励。默认 0 保持既有结构；启用后
    # 只在三指保持段惩罚实际关节的速度/加速度/jerk，不改变 Verifier。
    joint_smoothness_reward_weight = 0.0
    joint_velocity_reward_scale = 0.20
    joint_acceleration_reward_scale = 3.0
    joint_jerk_reward_scale = 80.0
    # 评估消融：锁住三根必需手指的 canonical 远端轴到预抓取值，
    # 用来证明远端自由度对成功/恢复的真实贡献；默认关闭，不参与发布。
    freeze_required_distal = False
    # 结构重构候选：依据上一物理步的指尖到当前椭球表面的有符号间隙，
    # 在“即将接触”前逐指降低动作域速度。它只改变接近的动态，不把
    # 目标位置写成脚本；远离表面时速度仍为 approach_speed，进入表面
    # 附近后平滑降到 floor，专门抑制首次碰撞的 PhysX 冲量。
    proximity_speed_enabled = False
    proximity_speed_start_m = 0.012
    proximity_speed_floor = 0.30
    proximity_speed_global = False
    # 结构性接触保护：若指尖几何中心已经进入椭球表面，而力传感器尚未
    # 在本步反映出冲量，则沿预抓取方向做一次有界、逐指的退让。它是
    # 接触柔顺投影，不是碰撞成功捷径；仍保留真实 PhysX 接触和 Verifier
    # 的全部限制。默认关闭，需先通过零残差探针再用于训练。
    geometry_contact_projection_enabled = False
    geometry_contact_projection_margin_m = 0.006
    geometry_contact_projection_gain = 0.80
    geometry_contact_projection_max_open = 0.45
    geometry_contact_projection_root_fraction = 0.45
    # 接触前几何寻触器：根据当前指尖到椭球表面的有符号间隙，连续地
    # 给三根必需手指一个很小的根部偏置。它不是把动作写成 P1 终点：
    # 目标随每个 episode 的实际果实半轴和指尖状态变化，进入软压力带
    # 后自动退出并把后续自由度交给 PPO/压力闭环。方向来自独立的局部
    # 运动探针（拇指负、食指正、中指负），不读取成功标签。
    geometry_approach_enabled = False
    geometry_approach_target_gap_m = 0.003
    geometry_approach_scale_m = 0.015
    geometry_approach_gain = 0.20
    geometry_approach_max_offset = 0.12
    geometry_approach_alpha = 0.25
    geometry_approach_direction = (-1.0, 1.0, -1.0)

    def __init__(self, cfg: NativeSmallFruitPinchCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        # 保留已验证的接近速率；压力闭环只在单指接触后回撤对应残差，
        # 不用全局慢速去牺牲其余手指的到达率。
        self._filter = CriticalDampedActionFilter(
            self.num_envs,
            14,
            dt_s=1.0 / 20.0,
            device=self.device,
            limits=TemporalLimits(
                approach_speed=self.approach_speed,
                hold_speed=self.hold_speed,
                approach_acceleration=self.approach_acceleration,
                hold_acceleration=self.hold_acceleration,
                natural_frequency_hz=self.natural_frequency_hz,
                reversal_deadband=0.005,
            ),
        )
        self._guide_action = torch.tensor(
            P1_GUIDE_ACTION, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1).clone()
        self._base_guide_action = self._make_base_guide_action()
        self._guide_action.copy_(self._base_guide_action)
        self._guide_force_ema = torch.zeros((self.num_envs, 3), device=self.device)
        self._guide_force_previous = torch.zeros_like(self._guide_force_ema)
        self._guide_offset = torch.zeros_like(self._guide_force_ema)
        self._force_balance_offset = torch.zeros_like(self._guide_force_ema)
        self._force_servo_offset = torch.zeros_like(self._guide_force_ema)
        # 正值表示向预抓取目标回撤的比例；负值表示接触后极小的补进。
        self._direct_force_open = torch.zeros_like(self._guide_force_ema)
        self._phase_offset = torch.zeros_like(self._guide_force_ema)
        self._phase_state = torch.zeros(
            (self.num_envs, 3), dtype=torch.int8, device=self.device
        )
        self._phase_low_age = torch.zeros(
            (self.num_envs, 3), dtype=torch.long, device=self.device
        )
        self._geometry_approach_offset = torch.zeros_like(self._guide_force_ema)
        self._filter_speed_scale = torch.ones((self.num_envs, 14), device=self.device)
        self._force_slow_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_actuator_joint_ids = list(
            dict.fromkeys(self._hand_joint_ids + self._mimic_follower_ids)
        )
        self._default_hand_velocity_limits = self._robot.data.joint_vel_limits.torch[
            :, self._hold_actuator_joint_ids
        ].clone()
        self._default_hold_stiffness = self._robot.data.joint_stiffness.torch[
            :, self._hold_actuator_joint_ids
        ].clone()
        self._default_hold_damping = self._robot.data.joint_damping.torch[
            :, self._hold_actuator_joint_ids
        ].clone()
        self._default_hand_effort_limits = self._robot.data.joint_effort_limits.torch[
            :, self._hold_actuator_joint_ids
        ].clone()
        self._reward_previous_joint_velocity = self._robot.data.joint_vel.torch[
            :, self._hand_joint_ids
        ].clone()
        self._reward_previous_joint_acceleration = torch.zeros_like(
            self._reward_previous_joint_velocity
        )
        # 保持段目标限速必须以“上一帧已发送的 canonical 目标”为参考，
        # 不能以滞后的物理关节位置为参考。后者会在压力回撤时把目标
        # 误判为闭合方向，导致安全打开被限速、压力反而继续上升。
        self._previous_hold_targets = self._joint_targets[:, self._hand_joint_ids].clone()
        self._hold_target_filter_state = self._previous_hold_targets.clone()
        # 每个必需手指独立进入当前姿态锚定；不能等三指全部接触后才
        # 切换，否则先接触的手指会在其余手指寻触期间继续追逐远端目标。
        self._hold_impedance_active = torch.zeros(
            (self.num_envs, 3), dtype=torch.bool, device=self.device
        )
        self._hold_impedance_ramp = torch.zeros((self.num_envs, 3), device=self.device)
        self._hold_impedance_direction = torch.zeros(
            (self.num_envs, len(self._hand_joint_ids)), device=self.device
        )
        self._hold_impedance_effort = torch.zeros_like(self._hold_impedance_direction)
        # 每个指尖首次建立物理接触后锁存其当前关节目标，直到压力迟滞
        # 低于退出阈值；过高时沿预抓取目标小步回撤。这是低层柔顺投影，
        # 不把终姿态写成脚本，残差策略仍可改变未接触手指与持稳过程。
        self._digit_latched = torch.zeros((self.num_envs, 5), dtype=torch.bool, device=self.device)
        hand_target = self._joint_targets[:, self._hand_joint_ids]
        self._digit_hold_targets = hand_target.unsqueeze(1).expand(-1, 5, -1).clone()
        self._pregrasp_hand_targets = self._digit_hold_targets.clone()
        # 仅供离线审计的终止快照；不进入观测、奖励或控制链。DirectRLEnv
        # 会在 _get_dones 后立即 reset，因此需在这里复制本回合末的几何/力。
        self._last_terminal_axes_m = self._fruit_axes_m.clone()
        self._last_terminal_force_n = self._last_physical_force_n.clone()
        self._last_terminal_reason = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self._last_terminal_hold_age = self._verifier.hold_age.clone()
        self._last_terminal_contact_ratio = torch.zeros(self.num_envs, device=self.device)
        self._last_terminal_band_ratio = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "_guide_action"):
            # DirectRLEnv invokes reset during NativeSmallFruitPinchEnv.__init__.
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._base_guide_action[env_ids] = self._make_base_guide_action(env_ids)
        self._guide_action[env_ids] = self._base_guide_action[env_ids]
        self._guide_force_ema[env_ids] = 0.0
        self._guide_force_previous[env_ids] = 0.0
        self._guide_offset[env_ids] = 0.0
        self._force_balance_offset[env_ids] = 0.0
        self._force_servo_offset[env_ids] = 0.0
        self._direct_force_open[env_ids] = 0.0
        self._phase_offset[env_ids] = 0.0
        self._phase_state[env_ids] = 0
        self._phase_low_age[env_ids] = 0
        self._geometry_approach_offset[env_ids] = 0.0
        self._filter_speed_scale[env_ids] = 1.0
        self._force_slow_phase[env_ids] = False
        self._digit_latched[env_ids] = False
        self._digit_hold_targets[env_ids] = self._pregrasp_hand_targets[env_ids]
        if hasattr(self, "_reward_previous_joint_velocity"):
            self._reward_previous_joint_velocity[env_ids] = self._robot.data.joint_vel.torch[
                env_ids
            ][:, self._hand_joint_ids]
            self._reward_previous_joint_acceleration[env_ids] = 0.0
        if hasattr(self, "_previous_hold_targets"):
            self._previous_hold_targets[env_ids] = self._joint_targets[env_ids][
                :, self._hand_joint_ids
            ]
        if hasattr(self, "_hold_target_filter_state"):
            self._hold_target_filter_state[env_ids] = self._joint_targets[env_ids][
                :, self._hand_joint_ids
            ]
        if hasattr(self, "_hold_impedance_active"):
            self._hold_impedance_active[env_ids] = False
            self._hold_impedance_ramp[env_ids] = 0.0
            self._hold_impedance_direction[env_ids] = 0.0
            self._hold_impedance_effort[env_ids] = 0.0

    _DIGIT_JOINTS = ((0, 5, 10, 11), (1, 6, 12), (2, 7, 13), (3, 8, 14), (4, 9, 15))

    def _make_base_guide_action(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return P1 plus an optional geometry-only, bounded weak prior."""

        if env_ids is None:
            count = self.num_envs
        else:
            count = len(env_ids)
        values = tuple(float(value) for value in self.guide_action_values)
        if len(values) != 14 or any(abs(value) > 1.0 for value in values):
            raise ValueError("guide_action_values 必须包含 14 个 [-1, 1] 值")
        guide = self._guide_action.new_tensor(values).expand(count, -1).clone()
        if self.geometry_prior_enabled and not self.geometry_prior_after_contact:
            prior = self._guide_action.new_tensor(self.geometry_prior_residual)
            if self.geometry_prior_size_adaptive:
                if env_ids is None:
                    axes = self._fruit_axes_m
                else:
                    axes = self._fruit_axes_m[env_ids]
                mean_axis = axes.mean(dim=1)
                size_fraction = (
                    (mean_axis - float(self.geometry_prior_axis_min_m))
                    / max(
                        float(self.geometry_prior_axis_max_m)
                        - float(self.geometry_prior_axis_min_m),
                        1.0e-6,
                    )
                ).clamp(0.0, 1.0)
                scale = float(self.geometry_prior_scale) + size_fraction * (
                    float(self.geometry_prior_max_scale)
                    - float(self.geometry_prior_scale)
                )
                guide += scale.unsqueeze(1) * prior
            else:
                guide += self.geometry_prior_scale * prior
        return guide.clamp(-1.0, 1.0)

    def _postprocess_joint_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """按指尖接触建立迟滞锁存，并在软上限附近向预抓取位回撤。"""

        if targets.shape != (self.num_envs, 16):
            raise ValueError("structured target must have shape (num_envs, 16)")
        force = self._last_physical_force_n
        if not self.digit_latch_enabled:
            # 新结构不在首次接触时锁存目标。只有超过软保护线才沿预抓取
            # 方向轻微回撤，并把回撤限制在本次目标的一个小比例内；下一步
            # 仍可由触觉和 PPO 残差连续修正，避免“撞上即冻结”。
            if self.projection_guard_fraction is None:
                guard = force.new_tensor((*self.soft_high_n, 1.20, 1.00)) + force.new_tensor(self.guard_margin_n)
            else:
                fraction = float(self.projection_guard_fraction)
                if not 0.0 < fraction <= 1.0:
                    raise ValueError("projection_guard_fraction 必须在 (0, 1] 内")
                guard = force.new_tensor((*self.soft_high_n, 1.20, 1.00)) * fraction
            over = torch.relu(force - guard)
            retract = (self.retract_fraction * over / guard.clamp_min(1.0e-6)).clamp(0.0, self.retract_fraction)
            projected = targets.clone()
            for digit, joints in enumerate(self._DIGIT_JOINTS):
                joint_ids = list(joints)
                opened = targets[:, joint_ids] + retract[:, digit : digit + 1] * (
                    self._pregrasp_hand_targets[:, digit, joint_ids] - targets[:, joint_ids]
                )
                projected[:, joint_ids] = torch.where(
                    (over[:, digit : digit + 1] > 0.0), opened, projected[:, joint_ids]
                )
                if self.isolate_nonrequired_digits and digit >= 3:
                    projected[:, joint_ids] = self._pregrasp_hand_targets[:, digit, joint_ids]
            projected = self._apply_direct_force_projection(projected)
            return self._apply_required_distal_freeze(
                self._apply_geometry_contact_projection(projected)
            )

        enter = force >= 0.08
        exit_ = force < 0.04
        previous = self._digit_latched.clone()
        self._digit_latched |= enter
        self._digit_latched &= ~exit_
        newly_latched = self._digit_latched & ~previous
        for digit, joints in enumerate(self._DIGIT_JOINTS):
            joint_ids = list(joints)
            self._digit_hold_targets[:, digit, joint_ids] = torch.where(
                newly_latched[:, digit : digit + 1],
                targets[:, joint_ids],
                self._digit_hold_targets[:, digit, joint_ids],
            )
        # 压力超过软上限时，每个策略步最多向预抓取目标回撤 6%，避免
        # 用一次大反向动作制造抽搐；低于退出阈值后自动解除锁存再接近。
        high = force >= force.new_tensor((*self.soft_high_n, 1.20, 1.00))
        retract = self._digit_latched & high
        opened = self._digit_hold_targets + 0.06 * (
            self._pregrasp_hand_targets - self._digit_hold_targets
        )
        self._digit_hold_targets = torch.where(
            retract.unsqueeze(-1), opened, self._digit_hold_targets
        )
        projected = targets.clone()
        for digit, joints in enumerate(self._DIGIT_JOINTS):
            joint_ids = list(joints)
            projected[:, joint_ids] = torch.where(
                self._digit_latched[:, digit : digit + 1],
                self._digit_hold_targets[:, digit, joint_ids],
                projected[:, joint_ids],
            )
            if self.isolate_nonrequired_digits and digit >= 3:
                projected[:, joint_ids] = self._pregrasp_hand_targets[:, digit, joint_ids]
        projected = self._apply_direct_force_projection(projected)
        return self._apply_required_distal_freeze(
            self._apply_geometry_contact_projection(projected)
        )

    def _apply_required_distal_freeze(self, targets: torch.Tensor) -> torch.Tensor:
        """评估用远端冻结，保留其它根部/几何动作不变。"""

        if not self.freeze_required_distal:
            return targets
        frozen = targets.clone()
        for digit, joint in ((0, 11), (1, 12), (2, 13)):
            frozen[:, joint] = self._pregrasp_hand_targets[:, digit, joint]
        return frozen

    def _apply_direct_force_projection(self, targets: torch.Tensor) -> torch.Tensor:
        """在 canonical 16D 目标层做逐指、平滑、可回退的受力修正。

        只使用上一物理步的指尖力和接触迟滞，不读取成功标签或固定终姿。
        拇指、食指、中指分别作用于其独立的近端驱动轴和远端轴；无名指、
        小指保持预抓取目标。这样既能抑制某一个手指的瞬时过载，也能让
        新增远端自由度在低压接触后承担少量补进，而不是再次只靠根部。
        """

        if not self.direct_force_projection_enabled:
            self._direct_force_open.zero_()
            return targets
        force = self._last_physical_force_n[:, :3]
        raw = force
        ema = self._guide_force_ema
        # 保护侧取原始力与 EMA 的较大值，避免滤波滞后漏掉一个 PhysX 冲量；
        # 补进侧只用 EMA，防止触觉噪声把目标来回推拉。
        signal_high = torch.maximum(raw, ema)
        rate_gain = float(self.direct_force_projection_rate_gain)
        if rate_gain > 0.0:
            rising = torch.relu(
                raw - self._previous_force_n[:, :3]
                - float(self.direct_force_projection_rate_deadband_n)
            )
            rising = (
                rising / max(float(self.direct_force_projection_rate_scale_n), 1.0e-6)
            ).clamp(0.0, 1.0)
            predicted = raw + rate_gain * rising
            signal_high = torch.maximum(signal_high, predicted)
        soft_low = force.new_tensor((0.20, 0.15, 0.12))
        soft_high = force.new_tensor(self.soft_high_n[:3])
        hard_high = force.new_tensor((4.0, 3.5, 3.5))
        warning = soft_high * float(self.direct_force_projection_warning_fraction)
        high_gap = (
            (signal_high - warning) / (hard_high - warning).clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        # 只有已经建立接触的手指才允许低压补进；未接触阶段仍由 P1
        # 几何先验负责到达，避免本层把开手状态直接推到物体上。
        contact = raw >= float(self.direct_force_projection_contact_n)
        low_gap = ((soft_low - ema) / soft_low.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        high_gain = self.direct_force_projection_high_gain_by_digit
        if high_gain is None:
            high_gain_tensor = force.new_full(
                (3,), float(self.direct_force_projection_high_gain)
            )
        else:
            if len(high_gain) != 3 or any(float(value) < 0.0 for value in high_gain):
                raise ValueError("direct_force_projection_high_gain_by_digit 必须为 3 个非负值")
            high_gain_tensor = force.new_tensor(tuple(float(value) for value in high_gain))
        desired = (
            high_gain_tensor * high_gap
            - float(self.direct_force_projection_low_gain) * low_gap * contact
        ).clamp(
            -float(self.direct_force_projection_max_open),
            float(self.direct_force_projection_max_open),
        )
        alpha = float(self.direct_force_projection_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("direct_force_projection_alpha 必须在 (0, 1] 内")
        self._direct_force_open.mul_(1.0 - alpha).add_(alpha * desired)
        self._direct_force_open.clamp_(
            -float(self.direct_force_projection_max_open),
            float(self.direct_force_projection_max_open),
        )
        projected = targets.clone()
        root_fraction = float(self.direct_force_projection_root_fraction)
        if not 0.0 <= root_fraction <= 1.0:
            raise ValueError("direct_force_projection_root_fraction 必须在 [0, 1] 内")
        # canonical 顺序：thumb(0,11), index(1,12), middle(2,13)。
        # 根部和远端的权重互补，确保远端残差确实有几何作用。
        for digit, joints in enumerate(((0, 11), (1, 12), (2, 13))):
            weights = (root_fraction, 1.0 - root_fraction)
            for joint, weight in zip(joints, weights):
                open_target = targets[:, joint] + self._direct_force_open[:, digit] * float(weight) * (
                    self._pregrasp_hand_targets[:, digit, joint] - targets[:, joint]
                )
                # 接触且低于软下限时允许负 offset 补进；未接触时只允许
                # 正向保护回撤，避免和先验到达路径形成反向竞赛。
                allowed = contact[:, digit] | (self._direct_force_open[:, digit] > 0.0)
                projected[:, joint] = torch.where(allowed, open_target, projected[:, joint])
        return projected

    def _apply_residual_trust_region(
        self,
        residual: torch.Tensor,
        digit_contact: torch.Tensor,
        all_required: torch.Tensor,
    ) -> torch.Tensor:
        """将策略残差限制在实时指尖力定义的局部安全切空间。

        该层只作用于 PPO 输出，不改写 compliant prior 或成功判定：
        未三指齐备时让先验完成寻触；齐备后，低于工作带的残差只允许
        负向补进，接近 soft_high 的残差只允许正向回撤，带内只保留小
        幅连续调节。全局协同维度默认置零，避免一个手指的压力把其它
        手指的绝对目标一起推离可行域；逐指根部/远端维度仍可学习。
        """

        if not self.residual_trust_region_enabled:
            return residual
        # 不从 ``self.num_envs`` 取形状：单元测试/离线诊断可在未构造
        # Isaac 场景的情况下调用此纯张量门控函数，且实际环境仍会由
        # digit_contact/all_required 的形状检查保证批次一致。
        expected_envs = residual.shape[0]
        if residual.ndim != 2 or residual.shape[1] != 14:
            raise ValueError("residual trust-region 输入必须为 (num_envs, 14)")
        if digit_contact.shape != (expected_envs, 5):
            raise ValueError("digit_contact 必须为 (num_envs, 5)")
        if all_required.shape != (expected_envs,):
            raise ValueError("all_required 必须为 (num_envs,)")
        force = self._last_physical_force_n[:, :3]
        soft_low = force.new_tensor((0.20, 0.15, 0.12))
        soft_high = force.new_tensor(self.soft_high_n[:3])
        warning_fraction = float(self.residual_trust_warning_fraction)
        if not 0.0 < warning_fraction <= 1.0:
            raise ValueError("residual_trust_warning_fraction 必须在 (0, 1] 内")
        in_band_scale = float(self.residual_trust_in_band_scale)
        if not 0.0 <= in_band_scale <= 1.0:
            raise ValueError("residual_trust_in_band_scale 必须在 [0, 1] 内")
        output = residual.clone()
        # 结构性约束：先让三指建立共同接触，再开放局部残差。
        output = torch.where(all_required.unsqueeze(1), output, torch.zeros_like(output))
        high = force >= soft_high * warning_fraction
        low = (force < soft_low) & digit_contact[:, :3]
        middle = ~(high | low)

        def project(value: torch.Tensor, digit: int) -> torch.Tensor:
            projected = torch.where(high[:, digit], torch.relu(value), value)
            projected = torch.where(low[:, digit], -torch.relu(-projected), projected)
            projected = torch.where(
                middle[:, digit], projected * in_band_scale, projected
            )
            return projected

        # 局部根部维度：thumb/index/middle = 5/6/7。
        for digit, joint_dim in enumerate((5, 6, 7)):
            output[:, joint_dim] = project(output[:, joint_dim], digit)
        # 新增远端维度：index/middle = 10/11。其它远端已由
        # isolate_nonrequired_digits 保持关闭，不在这里重新开放。
        for digit, joint_dim in ((1, 10), (2, 11)):
            output[:, joint_dim] = project(output[:, joint_dim], digit)
        if self.residual_trust_zero_global:
            output[:, :5] = 0.0
        else:
            # 非默认诊断模式仍不允许高力时的负向全局动作；带内仅保留
            # 很小幅度，避免隐式耦合破坏逐指安全切向空间。
            global_high = high.any(dim=1)
            global_low = low.all(dim=1)
            output[:, :5] = torch.where(
                global_high.unsqueeze(1), torch.relu(output[:, :5]), output[:, :5]
            )
            output[:, :5] = torch.where(
                global_low.unsqueeze(1), -torch.relu(-output[:, :5]), output[:, :5]
            )
            output[:, :5] = torch.where(
                (~global_high & ~global_low).unsqueeze(1),
                output[:, :5] * in_band_scale,
                output[:, :5],
            )
        return output

    def _limit_hold_joint_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """在三指齐备后按关节角速度限制 canonical 目标移动。"""

        close_speed = self.hold_joint_target_speed_rad_s
        hold = self._force_slow_phase | (self._verifier.hold_age > 0)
        result = targets
        # 目标限速解决的是“命令跳变”，而不是物理跟踪滞后。再提供一层
        # 可选的误差窗：在三指安全保持段把目标约束在当前实际位置附近，
        # 让隐式 PD 不因上一帧的远目标产生大冲量。该层不参与过压回撤，
        # 也不锁定绝对姿态；力下降后策略仍可逐帧补进。
        error_limit = self.hold_joint_target_error_limit_rad
        error_limit_by_joint = self.hold_joint_target_error_limit_by_joint
        if error_limit is not None or error_limit_by_joint is not None:
            if error_limit is not None:
                error_limit = float(error_limit)
                if error_limit <= 0.0:
                    raise ValueError("hold_joint_target_error_limit_rad 必须为正数")
                error_limit_tensor = torch.full(
                    (len(self._hand_joint_ids),),
                    error_limit,
                    dtype=targets.dtype,
                    device=targets.device,
                )
            else:
                values = tuple(float(value) for value in error_limit_by_joint)
                if len(values) != len(self._hand_joint_ids) or any(value <= 0.0 for value in values):
                    raise ValueError("hold_joint_target_error_limit_by_joint 必须为 16 个正数")
                error_limit_tensor = targets.new_tensor(values)
            force = self._last_physical_force_n[:, :3]
            soft_high = force.new_tensor(self.soft_high_n)
            safe_hold = hold & (force <= soft_high).all(dim=1)
            current = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
            bounded_error = torch.maximum(
                torch.minimum(result - current, error_limit_tensor.unsqueeze(0)),
                -error_limit_tensor.unsqueeze(0),
            )
            bounded = current + bounded_error
            result = torch.where(safe_hold.unsqueeze(1), bounded, result)
        anchor_fraction = self.hold_joint_target_anchor_fraction
        anchor_fraction_by_joint = getattr(
            self, "hold_joint_target_anchor_fraction_by_joint", None
        )
        if anchor_fraction is not None or anchor_fraction_by_joint is not None:
            if anchor_fraction_by_joint is not None:
                values = tuple(float(value) for value in anchor_fraction_by_joint)
                if len(values) != len(self._hand_joint_ids) or any(
                    not 0.0 <= value <= 1.0 for value in values
                ):
                    raise ValueError(
                        "hold_joint_target_anchor_fraction_by_joint 必须为 16 个 [0,1] 值"
                    )
                anchor = targets.new_tensor(values).unsqueeze(0)
            else:
                anchor_fraction = float(anchor_fraction)
                if not 0.0 <= anchor_fraction <= 1.0:
                    raise ValueError("hold_joint_target_anchor_fraction 必须在 [0, 1] 内")
                anchor = torch.full(
                    (1, len(self._hand_joint_ids)),
                    anchor_fraction,
                    dtype=targets.dtype,
                    device=targets.device,
                )
            force = self._last_physical_force_n[:, :3]
            soft_high = force.new_tensor(self.soft_high_n)
            safe_hold = hold & (force <= soft_high).all(dim=1)
            current = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
            anchored = current + anchor * (result - current)
            result = torch.where(safe_hold.unsqueeze(1), anchored, result)
        # 误差投影可能在首次进入保持段时把远端目标从旧目标直接拉到
        # 当前关节附近。目标速率限幅必须在误差窗之后执行，否则这一跳会
        # 在 50 ms 内表现为十几 rad/s 的虚假命令，并把隐式 PD 推入饱和。
        contact_speed = getattr(self, "contact_hand_target_speed_rad_s", None)
        if contact_speed is not None:
            contact_speed = float(contact_speed)
            if contact_speed <= 0.0:
                raise ValueError("contact_hand_target_speed_rad_s 必须为正数")
            force = self._last_physical_force_n[:, :3]
            soft_high = force.new_tensor(self.soft_high_n)
            contact = (
                (force >= 0.08) | self._verifier.contact[:, :3]
            ) & (force <= soft_high)
            previous = self._previous_hold_targets
            delta = result - previous
            max_delta = torch.full_like(delta, contact_speed * (1.0 / 20.0))
            # 只对三根必需手指的 canonical 轴限斜率；无名指/小指以及
            # 未接触的手指继续使用原始目标，避免把寻触过程整体拖慢。
            for digit, joints in enumerate(((0, 5, 10, 11), (1, 6, 12), (2, 7, 13))):
                joint_ids = list(joints)
                limited = previous[:, joint_ids] + delta[:, joint_ids].clamp(
                    -max_delta[:, joint_ids], max_delta[:, joint_ids]
                )
                result[:, joint_ids] = torch.where(
                    contact[:, digit : digit + 1],
                    limited,
                    result[:, joint_ids],
                )
        if close_speed is not None:
            close_speed = float(close_speed)
            if close_speed <= 0.0:
                raise ValueError("hold_joint_target_speed_rad_s 必须为正数")
            open_speed = self.hold_joint_target_open_speed_rad_s
            if open_speed is None:
                open_speed = close_speed
            open_speed = float(open_speed)
            if open_speed <= 0.0:
                raise ValueError("hold_joint_target_open_speed_rad_s 必须为正数")
            if bool(hold.any()):
                # 使用上一帧目标而不是实际位置：位置驱动存在跟踪滞后，若用实际
                # 位置判定方向，回撤目标可能被错误地当作闭合，从而阻塞减压。
                previous = self._previous_hold_targets
                # 预抓取目标是安全回撤方向的参考，不是成功终姿态。只把目标
                # 朝该参考移动的分量视为 opening；其它方向仍使用慢速闭合上限。
                open_reference = self._pregrasp_hand_targets[:, 0, :]
                delta = result - previous
                opening = delta * (open_reference - previous) > 0.0
                speed = torch.where(
                    opening,
                    torch.full_like(delta, open_speed),
                    torch.full_like(delta, close_speed),
                )
                max_delta = speed * (1.0 / 20.0)
                limited = previous + delta.clamp(-max_delta, max_delta)
                result = torch.where(hold.unsqueeze(1), limited, result)
        filter_alpha = getattr(self, "hold_target_filter_alpha", None)
        if filter_alpha is not None:
            filter_alpha = float(filter_alpha)
            if not 0.0 < filter_alpha <= 1.0:
                raise ValueError("hold_target_filter_alpha 必须在 (0, 1] 内")
            force = self._last_physical_force_n[:, :3]
            soft_high = force.new_tensor(self.soft_high_n)
            safe_hold = hold & (force <= soft_high).all(dim=1)
            previous = self._hold_target_filter_state
            filtered = previous + filter_alpha * (result - previous)
            # 只在安全保持段低通；过压/接触丢失时立即采用原始目标，
            # 并同步状态，保证下一步不会从过期目标重新追逐。
            result = torch.where(safe_hold.unsqueeze(1), filtered, result)
            self._hold_target_filter_state.copy_(result)
        self._previous_hold_targets.copy_(result)
        return result

    def _update_hold_velocity_limits(self) -> None:
        """仅对安全保持段写入较低的 PhysX 关节速度上限。"""

        hold_limit = self.hold_hand_velocity_limit_rad_s
        contact_limit = self.contact_hand_velocity_limit_rad_s
        hold_limit_by_joint = getattr(self, "hold_hand_velocity_limit_by_joint", None)
        if hold_limit is None and contact_limit is None and hold_limit_by_joint is None:
            return
        if hold_limit is not None and float(hold_limit) <= 0.0:
            raise ValueError("hold_hand_velocity_limit_rad_s 必须为正数")
        if contact_limit is not None and float(contact_limit) <= 0.0:
            raise ValueError("contact_hand_velocity_limit_rad_s 必须为正数")
        force = self._last_physical_force_n[:, :3]
        soft_high = force.new_tensor(self.soft_high_n)
        limits = self._default_hand_velocity_limits.clone()
        if contact_limit is not None:
            # 每个 required digit 的驱动/跟随关节共享该指的接触门；
            # 采用原始力而不是 EMA，避免首次冲量在滤波延迟内失去限速。
            contact = force >= 0.08
            cap = torch.minimum(
                self._default_hand_velocity_limits,
                torch.full_like(
                    self._default_hand_velocity_limits, float(contact_limit)
                ),
            )
            for digit, finger_name in enumerate(("thumb", "index", "middle")):
                joint_mask = torch.tensor(
                    [finger_name in self._robot.joint_names[joint_id]
                     for joint_id in self._hold_actuator_joint_ids],
                    dtype=torch.bool,
                    device=self.device,
                )
                active = (
                    contact[:, digit]
                    & (force[:, digit] <= soft_high[digit])
                ).unsqueeze(1) & joint_mask.unsqueeze(0)
                limits = torch.where(active, cap, limits)
        if hold_limit_by_joint is not None:
            values = tuple(float(value) for value in hold_limit_by_joint)
            if len(values) != len(self._hand_joint_ids) or any(value <= 0.0 for value in values):
                raise ValueError("hold_hand_velocity_limit_by_joint 必须为 16 个正数")
            hand_name_to_value = {
                self._robot.joint_names[joint_id]: value
                for joint_id, value in zip(self._hand_joint_ids, values)
            }
            requested_values = []
            for offset, joint_id in enumerate(self._hold_actuator_joint_ids):
                name = self._robot.joint_names[joint_id]
                driver = name
                if name.endswith("_joint3"):
                    driver = name.removesuffix("_joint3") + "_joint2"
                elif name.endswith("_joint4"):
                    driver = name.removesuffix("_joint4") + "_joint3"
                requested_values.append(
                    hand_name_to_value.get(
                        name,
                        hand_name_to_value.get(
                            driver,
                            float(self._default_hand_velocity_limits[0, offset].item()),
                        ),
                    )
                )
            requested = torch.tensor(
                requested_values,
                device=self.device,
                dtype=limits.dtype,
            ).unsqueeze(0).expand_as(limits)
            safe_hold = self._force_slow_phase & (self._verifier.hold_age > 0)
            safe_hold &= (force <= soft_high).all(dim=1)
            limits = torch.where(
                safe_hold.unsqueeze(1), torch.minimum(limits, requested), limits
            )
        if hold_limit is not None:
            # 只有三指均接触且处于安全压力带内才对剩余手部轴整体限速。
            # 超过软上限立即恢复原上限，避免“刹车挡住退让”。
            safe_hold = self._force_slow_phase & (self._verifier.hold_age > 0)
            safe_hold &= (force <= soft_high).all(dim=1)
            capped = torch.minimum(
                self._default_hand_velocity_limits,
                torch.full_like(self._default_hand_velocity_limits, float(hold_limit)),
            )
            limits = torch.where(safe_hold.unsqueeze(1), capped, limits)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._robot.write_joint_velocity_limit_to_sim_index(
            limits=limits,
            joint_ids=self._hold_actuator_joint_ids,
            env_ids=env_ids,
        )

    def _update_hold_actuator_gains(self) -> None:
        """在安全保持段切换阻尼增益，过载时恢复原始跟踪响应。"""

        if (
            self.hold_hand_stiffness is None
            and self.hold_hand_damping is None
            and self.hold_hand_damping_by_joint is None
            and self.hold_hand_stiffness_by_joint is None
            and getattr(self, "contact_hand_stiffness", None) is None
            and getattr(self, "contact_hand_damping", None) is None
            and not bool(getattr(self, "hold_effort_control_enabled", False))
            and not bool(getattr(self, "hold_current_impedance_enabled", False))
        ):
            return
        force = self._last_physical_force_n[:, :3]
        soft_high = force.new_tensor(self.soft_high_n)
        safe_hold = self._force_slow_phase & (self._verifier.hold_age > 0)
        if bool(getattr(self, "hold_current_impedance_enabled", False)):
            # 当前姿态锚定从逐指真实接触开始，而不是等全局 hold latch；
            # 这只影响阻尼切换，不改变 Verifier 的严格保持计时。
            contact_phase = (
                (self._last_physical_force_n[:, :3] >= 0.08)
                | self._verifier.contact[:, :3]
            ).any(dim=1)
            safe_hold = safe_hold | contact_phase
        safe_hold &= (force <= soft_high).all(dim=1)
        stiffness = self._default_hold_stiffness
        damping = self._default_hold_damping
        if self.hold_hand_stiffness_by_joint is not None:
            values = tuple(float(value) for value in self.hold_hand_stiffness_by_joint)
            if len(values) != len(self._hand_joint_ids) or any(value <= 0.0 for value in values):
                raise ValueError("hold_hand_stiffness_by_joint 必须为 16 个正数")
            hand_name_to_value = {
                self._robot.joint_names[joint_id]: value
                for joint_id, value in zip(self._hand_joint_ids, values)
            }
            requested_values = []
            for offset, joint_id in enumerate(self._hold_actuator_joint_ids):
                name = self._robot.joint_names[joint_id]
                driver = name
                if name.endswith("_joint3"):
                    driver = name.removesuffix("_joint3") + "_joint2"
                elif name.endswith("_joint4"):
                    driver = name.removesuffix("_joint4") + "_joint3"
                requested_values.append(
                    hand_name_to_value.get(
                        name,
                        hand_name_to_value.get(
                            driver,
                            float(self._default_hold_stiffness[0, offset].item()),
                        ),
                    )
                )
            requested = torch.tensor(
                requested_values, device=self.device, dtype=stiffness.dtype
            ).unsqueeze(0).expand_as(stiffness)
            stiffness = torch.where(
                safe_hold.unsqueeze(1), requested, self._default_hold_stiffness
            )
        elif self.hold_hand_stiffness is not None:
            value = float(self.hold_hand_stiffness)
            if value <= 0.0:
                raise ValueError("hold_hand_stiffness 必须为正数")
            stiffness = torch.where(
                safe_hold.unsqueeze(1),
                torch.full_like(stiffness, value),
                self._default_hold_stiffness,
            )
        if self.hold_hand_damping is not None or self.hold_hand_damping_by_joint is not None:
            if self.hold_hand_damping_by_joint is not None:
                values = tuple(float(value) for value in self.hold_hand_damping_by_joint)
                if len(values) != len(self._hand_joint_ids) or any(value <= 0.0 for value in values):
                    raise ValueError("hold_hand_damping_by_joint 必须为 16 个正数")
                hand_name_to_value = {
                    self._robot.joint_names[joint_id]: value
                    for joint_id, value in zip(self._hand_joint_ids, values)
                }
                requested_values = []
                for offset, joint_id in enumerate(self._hold_actuator_joint_ids):
                    name = self._robot.joint_names[joint_id]
                    driver = name
                    if name.endswith("_joint3"):
                        driver = name.removesuffix("_joint3") + "_joint2"
                    elif name.endswith("_joint4"):
                        driver = name.removesuffix("_joint4") + "_joint3"
                    requested_values.append(
                        hand_name_to_value.get(
                            name,
                            hand_name_to_value.get(
                                driver,
                                float(self._default_hold_damping[0, offset].item()),
                            ),
                        )
                    )
                requested = torch.tensor(
                    requested_values, device=self.device, dtype=damping.dtype
                ).unsqueeze(0).expand_as(damping)
            else:
                value = float(self.hold_hand_damping)
                if value <= 0.0:
                    raise ValueError("hold_hand_damping 必须为正数")
                requested = torch.full_like(damping, value)
            damping = torch.where(
                safe_hold.unsqueeze(1),
                requested,
                self._default_hold_damping,
            )
        # 逐指接触后增益降档必须在全局保持增益之后应用，且只覆盖该指的
        # 驱动/跟随轴。这样先接触的指可以柔顺等待，未接触的指仍按原始
        # 目标寻触；过压时直接回到上面的默认/保持增益，避免阻塞退让。
        contact_stiffness = getattr(self, "contact_hand_stiffness", None)
        contact_damping = getattr(self, "contact_hand_damping", None)
        if contact_stiffness is not None or contact_damping is not None:
            if contact_stiffness is not None and float(contact_stiffness) <= 0.0:
                raise ValueError("contact_hand_stiffness 必须为正数")
            if contact_damping is not None and float(contact_damping) <= 0.0:
                raise ValueError("contact_hand_damping 必须为正数")
            contact = (
                (self._last_physical_force_n[:, :3] >= 0.08)
                | self._verifier.contact[:, :3]
            )
            contact &= self._last_physical_force_n[:, :3] <= soft_high
            for digit, finger_name in enumerate(("thumb", "index", "middle")):
                joint_mask = torch.tensor(
                    [finger_name in self._robot.joint_names[joint_id]
                     for joint_id in self._hold_actuator_joint_ids],
                    dtype=torch.bool,
                    device=self.device,
                )
                if contact_stiffness is not None:
                    stiffness = torch.where(
                        (contact[:, digit].unsqueeze(1) & joint_mask.unsqueeze(0)),
                        torch.full_like(stiffness, float(contact_stiffness)),
                        stiffness,
                    )
                if contact_damping is not None:
                    damping = torch.where(
                        (contact[:, digit].unsqueeze(1) & joint_mask.unsqueeze(0)),
                        torch.full_like(damping, float(contact_damping)),
                        damping,
                    )
        if bool(getattr(self, "hold_effort_control_enabled", False)):
            effort_phase = self._force_slow_phase & (self._verifier.hold_age > 0)
            effort_stiffness = float(self.hold_effort_stiffness)
            effort_damping = float(self.hold_effort_damping)
            if effort_stiffness < 0.0 or effort_damping < 0.0:
                raise ValueError("hold_effort_stiffness/damping 不能为负数")
            stiffness = torch.where(
                effort_phase.unsqueeze(1),
                torch.full_like(stiffness, effort_stiffness),
                stiffness,
            )
            damping = torch.where(
                effort_phase.unsqueeze(1),
                torch.full_like(damping, effort_damping),
                damping,
            )
        impedance_damping = getattr(self, "hold_impedance_damping", None)
        if (
            bool(getattr(self, "hold_current_impedance_enabled", False))
            and impedance_damping is not None
        ):
            impedance_damping = float(impedance_damping)
            if impedance_damping <= 0.0:
                raise ValueError("hold_impedance_damping 必须为正数")
            damping = torch.where(
                safe_hold.unsqueeze(1),
                torch.full_like(damping, impedance_damping),
                damping,
            )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._robot.write_joint_stiffness_to_sim_index(
            stiffness=stiffness,
            joint_ids=self._hold_actuator_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_damping_to_sim_index(
            damping=damping,
            joint_ids=self._hold_actuator_joint_ids,
            env_ids=env_ids,
        )

    def _hold_effort_action(
        self, position_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """在保持段把位置驱动改成当前位姿锚定加有界努力前馈。

        该函数只在 ``hold_effort_control_enabled`` 显式开启时执行。位置
        目标锚在当前实测关节，三根必需手指的努力方向由当前目标/预抓取
        目标确定，幅值由上一物理步真实指尖力决定；因此它不会把成功
        位姿或固定力写死成控制脚本。非保持 environment 返回原始位置
        目标和零努力，保持原位置控制合同。
        """

        if bool(getattr(self, "hold_current_impedance_enabled", False)):
            return self._hold_current_impedance_action(position_targets)
        effort_mode = bool(getattr(self, "hold_effort_control_enabled", False))
        impedance_mode = bool(getattr(self, "hold_force_impedance_enabled", False))
        if not effort_mode and not impedance_mode:
            return position_targets, None
        effort_phase = self._force_slow_phase & (self._verifier.hold_age > 0)
        if not bool(effort_phase.any()):
            return position_targets, None
        if float(self.hold_effort_max) <= 0.0:
            raise ValueError("hold_effort_max 必须为正数")
        if float(self.hold_effort_in_band) < 0.0:
            raise ValueError("hold_effort_in_band 不能为负数")
        if float(self.hold_effort_low_fraction) < 0.0 or float(self.hold_effort_high_fraction) < 0.0:
            raise ValueError("hold_effort_low/high_fraction 不能为负数")
        current_control = self._robot.data.joint_pos.torch[:, self._hand_control_ids]
        adjusted = position_targets.clone()
        adjusted[:, self._hand_control_ids] = current_control
        current_hand = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        desired_hand = self._joint_targets[:, self._hand_joint_ids]
        force = self._last_physical_force_n[:, :3]
        target_force = force.new_tensor(self.force_center_targets_n)
        soft_low = force.new_tensor((0.20, 0.15, 0.12))
        soft_high = force.new_tensor(self.soft_high_n)
        effort = torch.zeros(
            (self.num_envs, len(self._hand_joint_ids)),
            device=self.device,
            dtype=position_targets.dtype,
        )
        if impedance_mode:
            # canonical 驱动轴的当前隐式刚度用于把小努力换算成角度偏置；
            # 读取默认值而不是把 12.0 写死，兼容后续资产标定。
            canonical_stiffness = self._robot.data.joint_stiffness.torch[
                :, self._hand_joint_ids
            ].clamp_min(1.0e-3)
            root_fraction = 0.65
        else:
            canonical_stiffness = torch.ones_like(current_hand)
            root_fraction = 1.0
        for digit, joints in enumerate(((0, 11), (1, 12), (2, 13))):
            joint_ids = list(joints)
            close_delta = desired_hand[:, joint_ids] - current_hand[:, joint_ids]
            open_delta = self._pregrasp_hand_targets[:, digit, joint_ids] - current_hand[:, joint_ids]
            high = force[:, digit] >= soft_high[digit]
            direction = torch.where(
                high.unsqueeze(1), torch.sign(open_delta), torch.sign(close_delta)
            )
            force_error = (
                (target_force[digit] - force[:, digit]) / target_force[digit].clamp_min(1.0e-3)
            ).clamp(-1.0, 1.0)
            low = force[:, digit] < soft_low[digit]
            low_gain = float(self.hold_effort_max) * float(self.hold_effort_low_fraction)
            base = float(self.hold_effort_in_band)
            magnitude = torch.full_like(force[:, digit], base)
            magnitude = torch.where(
                low,
                base + (low_gain - base) * force_error.clamp_min(0.0),
                magnitude,
            )
            high_gap = (
                (force[:, digit] - soft_high[digit]) / soft_high[digit].clamp_min(1.0e-3)
            ).clamp(0.0, 1.0)
            high_gain = float(self.hold_effort_max) * float(self.hold_effort_high_fraction)
            magnitude = torch.where(
                high,
                high_gain * high_gap + 0.05 * float(self.hold_effort_max),
                magnitude,
            )
            contribution = direction * magnitude.unsqueeze(1)
            if impedance_mode:
                weights = current_hand.new_tensor((root_fraction, 1.0 - root_fraction))
                angle_offset = contribution * weights.unsqueeze(0) / canonical_stiffness[:, joint_ids]
                adjusted[:, self._hand_joint_ids[joint_ids[0]]] = torch.where(
                    effort_phase, current_hand[:, joint_ids[0]] + angle_offset[:, 0],
                    adjusted[:, self._hand_joint_ids[joint_ids[0]]],
                )
                adjusted[:, self._hand_joint_ids[joint_ids[1]]] = torch.where(
                    effort_phase, current_hand[:, joint_ids[1]] + angle_offset[:, 1],
                    adjusted[:, self._hand_joint_ids[joint_ids[1]]],
                )
            else:
                effort[:, joint_ids] = torch.where(
                    effort_phase.unsqueeze(1), contribution, effort[:, joint_ids]
                )
        if self.isolate_nonrequired_digits:
            for joints in ((3, 8, 14), (4, 9, 15)):
                effort[:, list(joints)] = 0.0
        effort.clamp_(-float(self.hold_effort_max), float(self.hold_effort_max))
        return adjusted, effort

    def _hold_current_impedance_action(
        self, position_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """以当前实测关节为中心的低努力保持控制。

        该控制器只在三指已经共同接触且 Verifier 进入保持段后生效。它
        从该时刻的目标方向提取局部闭合符号，随后把位置目标锚定在实测
        ``q`` 附近，并用每指独立的力反馈调节小角度偏置。这样保持力由
        可解释的局部努力产生，既不会复现远端绝对姿态，也不会因 PPO
        的微小目标噪声在远处积累 PD 努力。过压或接触丢失时由外层
        Verifier 退出本模式，恢复原始回撤路径。
        """

        # 逐指以原始上一帧力或 Verifier 的迟滞接触状态进入锚定；这使得
        # 已经接触的手指先减速，未接触手指仍保留正常寻触目标。
        phase = (
            (self._last_physical_force_n[:, :3] >= 0.08)
            | self._verifier.contact[:, :3]
        )
        if not bool(phase.any()):
            self._hold_impedance_active.zero_()
            self._hold_impedance_ramp.zero_()
            self._hold_impedance_direction.zero_()
            self._hold_impedance_effort.zero_()
            return position_targets, None

        current_control = self._robot.data.joint_pos.torch[:, self._hand_control_ids]
        current_hand = self._robot.data.joint_pos.torch[:, self._hand_joint_ids]
        desired_hand = self._joint_targets[:, self._hand_joint_ids]
        starting = phase & ~self._hold_impedance_active
        if bool(starting.any()):
            # 方向只在进入保持时捕获一次；后续的 PPO residual 不会把
            # “开合方向”翻转成高频反向目标。远端目标若恰好贴近当前
            # 位置，则退回到预抓取—实测位置的已验证局部方向。
            direction = self._hold_impedance_direction.clone()
            for digit, joints in enumerate(((0, 5, 10, 11), (1, 6, 12), (2, 7, 13))):
                joint_ids = list(joints)
                delta = desired_hand[:, joint_ids] - current_hand[:, joint_ids]
                fallback = self._pregrasp_hand_targets[:, digit, joint_ids] - current_hand[:, joint_ids]
                candidate = torch.sign(
                    torch.where(delta.abs() > 0.02, delta, fallback)
                )
                for column, joint_id in enumerate(joint_ids):
                    direction[starting[:, digit], joint_id] = candidate[starting[:, digit], column]
            self._hold_impedance_direction.copy_(direction)
            for digit, joints in enumerate(((0, 5, 10, 11), (1, 6, 12), (2, 7, 13))):
                joint_ids = list(joints)
                self._hold_impedance_effort[:, joint_ids] = torch.where(
                    starting[:, digit : digit + 1],
                    torch.zeros_like(self._hold_impedance_effort[:, joint_ids]),
                    self._hold_impedance_effort[:, joint_ids],
                )
            self._hold_impedance_ramp[starting] = float(
                self.hold_impedance_start_fraction
            )
            self._hold_impedance_active |= starting

        # 接触迟滞退出后释放该指的锚定；其它手指的保持状态不受影响。
        released = self._hold_impedance_active & ~phase
        if bool(released.any()):
            for digit, joints in enumerate(((0, 5, 10, 11), (1, 6, 12), (2, 7, 13))):
                self._hold_impedance_effort[:, list(joints)] = torch.where(
                    released[:, digit : digit + 1],
                    torch.zeros_like(self._hold_impedance_effort[:, list(joints)]),
                    self._hold_impedance_effort[:, list(joints)],
                )
            self._hold_impedance_active &= phase
            self._hold_impedance_ramp = torch.where(
                phase, self._hold_impedance_ramp, torch.zeros_like(self._hold_impedance_ramp)
            )

        effort_by_digit = tuple(float(value) for value in self.hold_impedance_effort_by_digit)
        target_force = current_hand.new_tensor(tuple(float(value) for value in self.hold_impedance_force_target_n))
        if len(effort_by_digit) != 3 or any(value <= 0.0 for value in effort_by_digit):
            raise ValueError("hold_impedance_effort_by_digit 必须为 3 个正数")
        if target_force.shape != (3,) or bool((target_force <= 0.0).any()):
            raise ValueError("hold_impedance_force_target_n 必须为 3 个正数")
        force_gain = float(self.hold_impedance_force_gain)
        if force_gain < 0.0:
            raise ValueError("hold_impedance_force_gain 不能为负数")
        ramp_substeps = int(self.hold_impedance_ramp_substeps)
        if ramp_substeps < 1:
            raise ValueError("hold_impedance_ramp_substeps 必须为正整数")

        force = self._guide_force_ema[:, :3]
        force_error = ((target_force.unsqueeze(0) - force) / target_force.unsqueeze(0)).clamp(-1.0, 1.0)
        factor = (1.0 + force_gain * force_error).clamp(0.35, 1.60)
        # 接近每指软上沿时渐进降努力；不等到硬过载才刹车。
        soft_high = force.new_tensor(self.soft_high_n[:3])
        high_fraction = ((force - 0.72 * soft_high) / (0.28 * soft_high).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        factor *= 1.0 - 0.70 * high_fraction
        desired_effort = torch.zeros_like(self._hold_impedance_effort)
        active = self._hold_impedance_active
        for digit, joints in enumerate(((0, 5, 10, 11), (1, 6, 12), (2, 7, 13))):
            # 根部和远端共享同一逐指力设定，但每个关节仍保留
            # 独立的闭合方向；这不会把远端自由度退化成锁死。
            desired_effort[:, list(joints)] = (
                self._hold_impedance_direction[:, list(joints)]
                * (float(effort_by_digit[digit]) * factor[:, digit]).unsqueeze(1)
            )
            desired_effort[:, list(joints)] *= active[:, digit].unsqueeze(1)
        alpha = min(1.0, 1.0 / float(ramp_substeps))
        self._hold_impedance_effort = torch.where(
            active.any(dim=1, keepdim=True),
            self._hold_impedance_effort + alpha * (desired_effort - self._hold_impedance_effort),
            torch.zeros_like(self._hold_impedance_effort),
        )

        # 从配置快照读取每个 canonical 轴的隐式刚度，避免把 12 N·m/rad
        # 写死；后续若替换资产，局部努力仍转换为相同的物理量。
        canonical_offsets = []
        for joint_id in self._hand_joint_ids:
            try:
                actuator_offset = self._hold_actuator_joint_ids.index(joint_id)
                stiffness = self._default_hold_stiffness[:, actuator_offset].clamp_min(1.0e-3)
            except ValueError:
                stiffness = current_hand.new_full((self.num_envs,), 12.0)
            canonical_offsets.append(stiffness)
        stiffness = torch.stack(canonical_offsets, dim=1)
        angle_offset = self._hold_impedance_effort / stiffness
        adjusted = position_targets.clone()
        for digit, joints in enumerate(((0, 5, 10, 11), (1, 6, 12), (2, 7, 13))):
            joint_ids = list(joints)
            articulated_ids = [self._hand_joint_ids[index] for index in joint_ids]
            active_digit = active[:, digit]
            if not bool(active_digit.any()):
                continue
            # 驱动轴及其 mimic follower 都锚在当前实测位置，避免 follower
            # 的远端目标继续产生隐式 PD 冲量；canonical 轴再叠加小努力偏置。
            finger_name = ("thumb", "index", "middle")[digit]
            prefix = f"left_hand_{finger_name}"
            control_ids = [
                joint_id
                for joint_id in self._hand_control_ids
                if self._robot.joint_names[joint_id].startswith(prefix)
            ]
            if control_ids:
                control_offsets = [self._hand_control_ids.index(joint_id) for joint_id in control_ids]
                adjusted[:, control_ids] = torch.where(
                    active_digit.unsqueeze(1),
                    current_control[:, control_offsets],
                    adjusted[:, control_ids],
                )
            adjusted[:, articulated_ids] = torch.where(
                active_digit.unsqueeze(1),
                current_hand[:, joint_ids] + angle_offset[:, joint_ids],
                adjusted[:, articulated_ids],
            )
        # 返回零努力张量仅用于让 Native 层显式写入 velocity_target=0；
        # 隐式位置驱动仍负责将上述小角度偏置转换成接触努力。
        zero_effort = torch.zeros_like(self._hold_impedance_effort)
        return adjusted, zero_effort

    def _update_hold_effort_limits(self) -> None:
        """在安全保持段限制手部驱动努力，过压时恢复回撤能力。"""

        limit = self.hold_hand_effort_limit
        if limit is None:
            return
        limit = float(limit)
        if limit <= 0.0:
            raise ValueError("hold_hand_effort_limit 必须为正数")
        force = self._last_physical_force_n[:, :3]
        soft_high = force.new_tensor(self.soft_high_n)
        safe_hold = self._force_slow_phase & (self._verifier.hold_age > 0)
        safe_hold &= (force <= soft_high).all(dim=1)
        capped = torch.minimum(
            self._default_hand_effort_limits,
            torch.full_like(self._default_hand_effort_limits, limit),
        )
        limits = torch.where(
            safe_hold.unsqueeze(1), capped, self._default_hand_effort_limits
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._robot.write_joint_effort_limit_to_sim_index(
            limits=limits,
            joint_ids=self._hold_actuator_joint_ids,
            env_ids=env_ids,
        )

    def _update_proximity_speed_scale(self) -> None:
        """按指尖—椭球表面间隙生成逐指接近速度比例。"""

        if not self.proximity_speed_enabled:
            self._filter_speed_scale.fill_(1.0)
            return
        if self.proximity_speed_start_m <= 0.0:
            raise ValueError("proximity_speed_start_m 必须为正数")
        floor = float(self.proximity_speed_floor)
        if not 0.0 < floor <= 1.0:
            raise ValueError("proximity_speed_floor 必须在 (0, 1] 内")
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids[:3]]
        fruit_center = self._tomato.data.root_pos_w.torch
        relative = tips - fruit_center.unsqueeze(1)
        radial = torch.linalg.vector_norm(relative, dim=2).clamp_min(1.0e-6)
        unit = relative / radial.unsqueeze(-1)
        axes = self._fruit_axes_m[:, None, :].clamp_min(1.0e-6)
        ellipsoid_radius = 1.0 / torch.sqrt(
            torch.sum(torch.square(unit / axes), dim=2).clamp_min(1.0e-8)
        )
        signed_gap = radial - ellipsoid_radius
        # gap >= start => full approach speed; at/inside surface => floor.
        fraction = (signed_gap / float(self.proximity_speed_start_m)).clamp(0.0, 1.0)
        scale = floor + (1.0 - floor) * fraction
        speed = torch.ones_like(self._filter_speed_scale)
        speed[:, 5:8] = scale
        speed[:, 10] = scale[:, 1]
        speed[:, 11] = scale[:, 2]
        if self.proximity_speed_global:
            speed[:, :5] = scale.amin(dim=1, keepdim=True)
        self._filter_speed_scale.copy_(speed)

    def _apply_geometry_contact_projection(self, targets: torch.Tensor) -> torch.Tensor:
        """按实际指尖—表面穿透量生成一次有界逐指回撤。"""

        if not self.geometry_contact_projection_enabled:
            return targets
        margin = float(self.geometry_contact_projection_margin_m)
        gain = float(self.geometry_contact_projection_gain)
        maximum = float(self.geometry_contact_projection_max_open)
        root_fraction = float(self.geometry_contact_projection_root_fraction)
        if margin <= 0.0 or gain < 0.0 or maximum < 0.0 or not 0.0 <= root_fraction <= 1.0:
            raise ValueError("几何接触投影参数非法")
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids[:3]]
        fruit_center = self._tomato.data.root_pos_w.torch
        relative = tips - fruit_center.unsqueeze(1)
        radial = torch.linalg.vector_norm(relative, dim=2).clamp_min(1.0e-6)
        unit = relative / radial.unsqueeze(-1)
        axes = self._fruit_axes_m[:, None, :].clamp_min(1.0e-6)
        ellipsoid_radius = 1.0 / torch.sqrt(
            torch.sum(torch.square(unit / axes), dim=2).clamp_min(1.0e-8)
        )
        penetration = torch.relu(ellipsoid_radius - radial)
        open_fraction = (gain * penetration / margin).clamp(0.0, maximum)
        projected = targets.clone()
        for digit, joints in enumerate(((0, 11), (1, 12), (2, 13))):
            weights = (root_fraction, 1.0 - root_fraction)
            for joint, weight in zip(joints, weights):
                open_target = projected[:, joint] + open_fraction[:, digit] * float(weight) * (
                    self._pregrasp_hand_targets[:, digit, joint] - projected[:, joint]
                )
                projected[:, joint] = torch.where(
                    penetration[:, digit] > 0.0, open_target, projected[:, joint]
                )
        return projected

    def _update_geometry_approach(self, force: torch.Tensor) -> torch.Tensor:
        """根据实际表面间隙给接触前三指生成有界连续偏置。"""

        if not self.geometry_approach_enabled:
            self._geometry_approach_offset.zero_()
            return torch.zeros_like(self._geometry_approach_offset)
        scale_m = float(self.geometry_approach_scale_m)
        gain = float(self.geometry_approach_gain)
        maximum = float(self.geometry_approach_max_offset)
        alpha = float(self.geometry_approach_alpha)
        target_gap = float(self.geometry_approach_target_gap_m)
        direction = force.new_tensor(tuple(float(v) for v in self.geometry_approach_direction))
        if scale_m <= 0.0 or gain < 0.0 or maximum < 0.0 or not 0.0 < alpha <= 1.0:
            raise ValueError("geometry_approach 参数非法")
        if direction.shape != (3,) or bool((direction.abs() > 1.0).any()):
            raise ValueError("geometry_approach_direction 必须为 3 个 [-1, 1] 值")
        tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids[:3]]
        fruit_center = self._tomato.data.root_pos_w.torch
        relative = tips - fruit_center.unsqueeze(1)
        radial = torch.linalg.vector_norm(relative, dim=2).clamp_min(1.0e-6)
        unit = relative / radial.unsqueeze(-1)
        axes = self._fruit_axes_m[:, None, :].clamp_min(1.0e-6)
        ellipsoid_radius = 1.0 / torch.sqrt(
            torch.sum(torch.square(unit / axes), dim=2).clamp_min(1.0e-8)
        )
        signed_gap = radial - ellipsoid_radius
        error = ((signed_gap - target_gap) / scale_m).clamp(-1.0, 1.0)
        # 只在该指尚未达到软带时寻触/纠偏；达到带内后不再用几何误差
        # 继续加压，避免几何模型误差转成持稳段抽搐。
        soft_low = force.new_tensor((0.20, 0.15, 0.12))
        active = force[:, :3] < soft_low
        desired = direction.unsqueeze(0) * gain * error
        desired = torch.where(active, desired, torch.zeros_like(desired))
        target = self._geometry_approach_offset + desired
        target = target.clamp(-maximum, maximum)
        self._geometry_approach_offset.mul_(1.0 - alpha).add_(alpha * target)
        return self._geometry_approach_offset

    def _update_compliant_prior(self) -> torch.Tensor:
        force = self._last_physical_force_n[:, :3]
        # 不能由单个先接触的手指提前把全手切到 HOLD：否则另两根手指
        # 还在接近时也会被整体限速，容易永远达不到三指共同接触。只有
        # 三根必需手指都已建立接触，才进入全手的低速持稳相位；在此之前
        # 仍由接近限速完成其余手指的闭合。
        self._force_slow_phase.copy_((force >= 0.08).all(dim=1))
        alpha = self.force_error_alpha
        self._guide_force_ema.mul_(1.0 - alpha).add_(alpha * force)
        delta_force = self._guide_force_ema - self._guide_force_previous
        self._guide_force_previous.copy_(self._guide_force_ema)
        high = force.new_tensor(self.soft_high_n)
        warning = torch.relu(self._guide_force_ema - self.warning_fraction * high)
        # 仅在压力接近软上限或快速上升时回撤，带内保持 P1 轨迹不动。
        retract_sign = warning.new_tensor(self.force_retract_sign)
        error = self.proportional_gain * warning * retract_sign
        error += self.derivative_gain * delta_force * retract_sign
        desired_offset = error.clamp(-self.max_guide_offset, self.max_guide_offset)
        # 第一轮结构验证先固定为 P1 物理可行动作，只验证慢速接触相位；
        # 压力反馈残差在 PPO 训练阶段通过 residual_scale 显式开启。
        if not self.guide_feedback_enabled:
            desired_offset.zero_()
        self._guide_offset.mul_(1.0 - self.guide_action_alpha).add_(
            self.guide_action_alpha * desired_offset
        )
        if self.force_balance_enabled:
            target = self._guide_force_ema.new_tensor(self.force_balance_target_n)
            width = self._guide_force_ema.new_tensor(self.force_balance_width_n).clamp_min(1.0e-6)
            error = target - self._guide_force_ema
            outside_deadband = torch.sign(error) * torch.relu(error.abs() - self.force_balance_deadband_n)
            desired_balance = self.force_balance_gain * (outside_deadband / width)
            desired_balance = desired_balance.clamp(-self.force_balance_max_offset, self.force_balance_max_offset)
            self._force_balance_offset.mul_(1.0 - self.force_balance_alpha).add_(
                self.force_balance_alpha * desired_balance
            )
        else:
            self._force_balance_offset.zero_()
        if self.force_servo_enabled:
            target = self._guide_force_ema.new_tensor(self.force_servo_target_n)
            contact_gap = (
                (self.force_servo_contact_n - self._guide_force_ema)
                / max(self.force_servo_contact_n, 1.0e-6)
            ).clamp(0.0, 1.0)
            soft_low = self._guide_force_ema.new_tensor((0.20, 0.15, 0.12))
            band_seek_gap = (
                (soft_low - self._guide_force_ema)
                / (soft_low - self.force_servo_contact_n).clamp_min(1.0e-6)
            ).clamp(0.0, 1.0)
            target_gap = (
                (target - self._guide_force_ema - self.force_servo_contact_n)
                / (target - self.force_servo_contact_n).clamp_min(1.0e-6)
            ).clamp(0.0, 1.0)
            soft_high = self._guide_force_ema.new_tensor(self.soft_high_n)
            warning_level = soft_high * float(self.force_servo_warning_fraction)
            high_gap = (
                (self._guide_force_ema - warning_level)
                / (soft_high - warning_level).clamp_min(1.0e-6)
            ).clamp(0.0, 1.0)
            seek_direction = self._guide_force_ema.new_tensor(self.force_servo_seek_direction)
            retract_direction = self._guide_force_ema.new_tensor(self.force_servo_retract_direction)
            if self.phase_controller_enabled:
                # 结构重构候选：逐指有限状态 + 有界偏置积分。与 level-only
                # servo 的关键区别是 HOLD 不把偏置衰减回零，因此低带接触
                # 不会因控制器“松手”而反复丢失；所有相位转换仍只依赖上一
                # 物理步的真实指尖力，并带迟滞避免触觉噪声抖动。
                contact_n = float(self.force_servo_contact_n)
                hysteresis = float(self.phase_hysteresis_n)
                force = self._guide_force_ema
                # 低带判断继续使用 EMA，避免触觉噪声触发闭合；高力保护
                # 同时看原始上一帧力，避免 0.18 的滤波滞后让一次 PhysX
                # 冲量在状态机识别前先跨过硬上限。
                high_signal = torch.maximum(force, self._last_physical_force_n[:, :3])
                low_contact = force >= contact_n
                low_band = force < soft_low
                high_band = high_signal >= warning_level
                self._phase_low_age.copy_(
                    torch.where(
                        low_contact & low_band,
                        self._phase_low_age + 1,
                        torch.zeros_like(self._phase_low_age),
                    )
                )
                state = self._phase_state
                # 0 SEARCH，1 CLOSE，2 HOLD，3 RETRACT。
                state.copy_(torch.where(force < contact_n - hysteresis, 0, state))
                state.copy_(torch.where(
                    low_contact & low_band
                    & (self._phase_low_age >= int(self.phase_low_delay_steps)),
                    torch.ones_like(state),
                    state,
                ))
                state.copy_(torch.where(
                    low_contact & (force >= soft_low),
                    torch.full_like(state, 2),
                    state,
                ))
                state.copy_(torch.where(high_band, torch.full_like(state, 3), state))
                state.copy_(torch.where(
                    (state == 3) & (force < warning_level - hysteresis)
                    & (force >= soft_low),
                    torch.full_like(state, 2),
                    state,
                ))
                search_gap = contact_gap
                close_gap = band_seek_gap
                close_ramp = (
                    (self._phase_low_age.to(dtype=force.dtype)
                     - float(self.phase_low_delay_steps))
                    / max(float(self.phase_low_ramp_steps), 1.0)
                ).clamp(0.0, 1.0)
                search_mag = self.phase_search_step * search_gap
                close_mag = self.phase_close_step * close_gap * close_ramp
                geometry_delta = torch.zeros_like(close_mag)
                if self.phase_geometry_enabled:
                    tips = self._robot.data.body_pos_w.torch[:, self._fingertip_contact_body_ids[:3]]
                    fruit_center = self._tomato.data.root_pos_w.torch
                    relative = tips - fruit_center.unsqueeze(1)
                    radial = torch.linalg.vector_norm(relative, dim=2).clamp_min(1.0e-6)
                    unit = relative / radial.unsqueeze(-1)
                    axes = self._fruit_axes_m[:, None, :].clamp_min(1.0e-6)
                    ellipsoid_radius = 1.0 / torch.sqrt(
                        torch.sum(torch.square(unit / axes), dim=2).clamp_min(1.0e-8)
                    )
                    # gap > 0 表示指尖仍在表面外，需沿负向闭合；gap < 0
                    # 表示穿入表面，沿正向打开。仅在压力低于 soft_low
                    # 时启用，避免几何误差在持稳段继续加压。
                    signed_gap = radial - ellipsoid_radius
                    gap_unit = (
                        signed_gap / max(float(self.phase_geometry_gap_scale_m), 1.0e-6)
                    ).clamp(-1.0, 1.0)
                    geometry_gate = (
                        (soft_low - force) / soft_low.clamp_min(1.0e-6)
                    ).clamp(0.0, 1.0)
                    geometry_delta = (
                        float(self.phase_geometry_step) * gap_unit * geometry_gate
                    )
                # 未接触时只寻触；接触后且低于 soft_low 才进入 CLOSE。
                seek_mag = torch.where(
                    ~low_contact,
                    search_mag,
                    torch.where(state == 1, close_mag, torch.zeros_like(close_mag))
                    + geometry_delta,
                )
                high_gap_phase = (
                    (high_signal - warning_level)
                    / (soft_high - warning_level).clamp_min(1.0e-6)
                ).clamp(0.0, 1.0)
                retract_mag = self.phase_retract_step * high_gap_phase * (state == 3).to(force.dtype)
                delta_offset = seek_mag * seek_direction + retract_mag * retract_direction
                target_offset = (
                    self._phase_offset + delta_offset
                ).clamp(-float(self.phase_max_offset), float(self.phase_max_offset))
                self._phase_offset.mul_(1.0 - float(self.phase_offset_alpha)).add_(
                    float(self.phase_offset_alpha) * target_offset
                )
                self._force_servo_offset.copy_(self._phase_offset)
                if self.phase_target_controller_enabled:
                    # 设定点闭环：信号由 EMA 与原始上一帧力混合，低通保留
                    # 噪声鲁棒性，原始分量让过压在滤波滞后前进入正向回撤。
                    raw_force = self._last_physical_force_n[:, :3]
                    target_force = self._guide_force_ema.new_tensor(self.phase_target_force_n)
                    signal = (
                        (1.0 - float(self.phase_target_raw_mix)) * self._guide_force_ema
                        + float(self.phase_target_raw_mix) * raw_force
                    )
                    error = target_force - signal
                    error = torch.sign(error) * torch.relu(
                        error.abs() - float(self.phase_target_deadband_n)
                    )
                    scale_n = self._guide_force_ema.new_tensor(
                        self.phase_target_error_scale_n
                    ).clamp_min(1.0e-6)
                    normalized_error = (error / scale_n).clamp(-1.0, 1.0)
                    # 正误差表示力不足，沿已校验的 seek_direction 闭合；
                    # 负误差表示压力偏高，沿 retract_direction 打开。
                    delta_target = (
                        float(self.phase_target_step)
                        * normalized_error
                        * seek_direction
                    )
                    hard_signal = torch.maximum(self._guide_force_ema, raw_force)
                    hard_gap_target = (
                        (hard_signal - warning_level)
                        / (soft_high - warning_level).clamp_min(1.0e-6)
                    ).clamp(0.0, 1.0)
                    delta_target += (
                        float(self.phase_target_step)
                        * hard_gap_target
                        * retract_direction
                    )
                    target_offset = (
                        self._phase_offset + delta_target
                    ).clamp(-float(self.phase_max_offset), float(self.phase_max_offset))
                    self._phase_offset.mul_(1.0 - float(self.phase_target_alpha)).add_(
                        float(self.phase_target_alpha) * target_offset
                    )
                    self._force_servo_offset.copy_(self._phase_offset)
            else:
                desired_servo = (
                    (
                        self.force_servo_seek_gain * contact_gap
                        + self.force_servo_band_seek_gain * band_seek_gap
                        + self.force_servo_target_gain * target_gap
                    )
                    * seek_direction
                    + self.force_servo_retract_gain * high_gap * retract_direction
                ).clamp(-self.force_servo_max_offset, self.force_servo_max_offset)
                if self.force_servo_rate_gain > 0.0:
                    # 只响应正向力冲量；力下降时由已有 EMA 偏置自然衰减，
                    # 避免把测量噪声转换为反向抽动。
                    rising = torch.relu(
                        delta_force[:, :3] - float(self.force_servo_rate_deadband_n)
                    )
                    rising = (
                        rising / max(float(self.force_servo_rate_scale_n), 1.0e-6)
                    ).clamp(0.0, 1.0)
                    desired_servo = (
                        desired_servo
                        + self.force_servo_rate_gain * rising * retract_direction
                    ).clamp(-self.force_servo_max_offset, self.force_servo_max_offset)
                self._force_servo_offset.mul_(1.0 - self.force_servo_alpha).add_(
                    self.force_servo_alpha * desired_servo
                )
        else:
            self._force_servo_offset.zero_()
            self._phase_offset.zero_()
            self._phase_state.zero_()
            self._phase_low_age.zero_()
        # Rebuild from the bounded base each policy step.  This avoids cumulative
        # drift while retaining continuous pressure feedback and PPO residuals.
        guide = self._base_guide_action.clone()
        if self.geometry_prior_enabled and self.geometry_prior_after_contact:
            prior = guide.new_tensor(self.geometry_prior_residual)
            contact_gate = (force >= 0.08).to(dtype=guide.dtype)
            guide[:, 5:8] += (
                self.geometry_prior_scale
                * prior[5:8].unsqueeze(0)
                * contact_gate
            )
        # 14D 顺序：[5 个协同, 5 个每指残差, 4 个远端残差]。
        # 近端/远端同时修正，避免只靠根部挤压；拇指没有新增远端维度，
        # 使用其原有独立残差。
        guide[:, 5:8] += self._guide_offset + self._force_balance_offset
        if self.phase_controller_enabled:
            phase_root = guide.new_tensor((1.0, self.phase_root_fraction, self.phase_root_fraction))
            guide[:, 5:8] += self._force_servo_offset * phase_root
        else:
            guide[:, 5:8] += self._force_servo_offset
        guide[:, 10] += 0.70 * (
            self._guide_offset[:, 1] + self._force_balance_offset[:, 1]
        )
        guide[:, 11] += 0.70 * (
            self._guide_offset[:, 2] + self._force_balance_offset[:, 2]
        )
        if self.phase_controller_enabled:
            distal_fraction = 1.0 - float(self.phase_root_fraction)
            guide[:, 10] += distal_fraction * self._force_servo_offset[:, 1]
            guide[:, 11] += distal_fraction * self._force_servo_offset[:, 2]
        else:
            guide[:, 10] += 0.70 * self._force_servo_offset[:, 1]
            guide[:, 11] += 0.70 * self._force_servo_offset[:, 2]
        if self.phase_controller_enabled:
            # 接触后的逐指降档只作用于该指的根部/远端维度；尚未接触
            # 的手指不被全局 HOLD 拖慢，保持并行寻触。软上限附近再降一档，
            # 让原始力信号有时间经过滤波器后再决定回撤，不改变硬判据。
            raw_force = self._last_physical_force_n[:, :3]
            speed = torch.ones_like(self._filter_speed_scale)
            contact_scale = torch.where(
                raw_force >= float(self.force_servo_contact_n),
                raw_force.new_full(raw_force.shape, float(self.phase_contact_speed_scale)),
                torch.ones_like(raw_force),
            )
            warning_level = raw_force.new_tensor(self.soft_high_n) * float(self.force_servo_warning_fraction)
            speed = speed.clone()
            speed[:, 5:8] = contact_scale
            speed[:, 10] = contact_scale[:, 1]
            speed[:, 11] = contact_scale[:, 2]
            high_scale = torch.where(
                raw_force >= warning_level,
                raw_force.new_full(raw_force.shape, float(self.phase_high_speed_scale)),
                torch.ones_like(raw_force),
            )
            speed[:, 5:8] = torch.minimum(speed[:, 5:8], high_scale)
            speed[:, 10] = torch.minimum(speed[:, 10], high_scale[:, 1])
            speed[:, 11] = torch.minimum(speed[:, 11], high_scale[:, 2])
            self._filter_speed_scale.copy_(speed)
        else:
            self._filter_speed_scale.fill_(1.0)
        if self.adaptive_closure_enabled:
            # 接触前的推进信号只由实际上一帧指尖力产生：力为零时最多少量
            # 闭合，接触达到 0.08 N 后立即归零；若接触后来丢失则重新渐进
            # 闭合。这样不会因把“目标力”写死而把 P1 的低力可行解推向硬
            # 上限，也不会在接触瞬间锁住绝对关节目标。
            threshold = float(self.adaptive_contact_threshold_n)
            need_contact = (
                (threshold - self._guide_force_ema) / max(threshold, 1.0e-6)
            ).clamp(0.0, 1.0)
            steps = self._guide_force_ema.new_tensor(self.adaptive_closure_steps)
            delta = steps * need_contact
            # 拇指 dim 5 的负向是已验证的接近方向；食指/中指同时利用
            # 近端和新增远端维度。对掌 dim 4 保持 P1，不在接触闭环中漂移。
            guide[:, 5] += delta[:, 0]
            guide[:, 6] += delta[:, 1]
            guide[:, 7] += delta[:, 2]
            guide[:, 10] += 0.65 * delta[:, 1]
            guide[:, 11] += 0.65 * delta[:, 2]
            guide[:, 4] = guide[:, 4].clamp(-1.0, 1.0)
            guide[:, 6:8] = guide[:, 6:8].clamp(-1.0, 1.0)
            guide[:, 10:12] = guide[:, 10:12].clamp(-1.0, 1.0)
        geometry_approach = self._update_geometry_approach(force)
        if self.geometry_approach_enabled:
            guide[:, 5:8] += geometry_approach
        return guide.clamp(-1.0, 1.0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 14):
            raise ValueError(f"actions 必须为 ({self.num_envs}, 14)")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions 包含 NaN 或 Inf")
        guide = self._update_compliant_prior()
        # 接触前先根据上一物理步的真实几何间隙降档；与压力投影互补，
        # 避免第一次接触的冲量在触觉回传之前已经越过硬上限。
        self._update_proximity_speed_scale()
        # residual policy 的零均值并非“固定姿态”：先验每步都由触觉更新，
        # PPO 残差仍会根据完整观测改变五个协同量和四个远端自由度。
        # 接触前的 PPO 残差只保留 20% 权限，避免高斯探索在尚未建立
        # 三指接触时破坏可行闭合；某指已接触后才开放该指的微调维度，
        # 三指齐备后再开放全局协同量。这样仍是连续策略，而不是分步
        # 轨迹回放或固定终姿脚本。
        residual = actions.clamp(-1.0, 1.0)
        root_scale_value = float(self.residual_scale)
        distal_scale_value = (
            root_scale_value
            if self.distal_residual_scale is None
            else float(self.distal_residual_scale)
        )
        if not 0.0 <= root_scale_value <= 1.0:
            raise ValueError("residual_scale 必须在 [0, 1] 内")
        if not 0.0 <= distal_scale_value <= 1.0:
            raise ValueError("distal_residual_scale 必须在 [0, 1] 内")
        scale = residual.new_full(
            residual.shape, root_scale_value * self.precontact_residual_fraction
        )
        # 远端维度从一开始就有自己的受限权限；接触后在下面按指开放，
        # 未接触时仍只保留 precontact_residual_fraction 比例。
        scale[:, 10:14] = residual.new_full(
            (self.num_envs, 4),
            distal_scale_value * self.precontact_residual_fraction,
        )
        # 不使用一次性锁存判断接触阶段；以连续的上一物理步力值作为门控，
        # 接触丢失时自动收回对应残差权限，避免未接触手指被策略噪声拉偏。
        digit_contact = self._last_physical_force_n >= 0.08
        all_required = digit_contact[:, :3].all(dim=1)
        scale[:, :5] = torch.where(
            all_required.unsqueeze(1),
            residual.new_full((self.num_envs, 5), root_scale_value),
            scale[:, :5],
        )
        for digit in range(5):
            digit_open = digit_contact[:, digit]
            if self.residual_require_all_contact:
                digit_open = digit_open & all_required
            if self.isolate_nonrequired_digits and digit >= 3:
                scale[:, 5 + digit] = 0.0
                if digit > 0:
                    scale[:, 10 + digit - 1] = 0.0
                continue
            scale[:, 5 + digit] = torch.where(
                digit_open,
                residual.new_full((self.num_envs,), root_scale_value),
                scale[:, 5 + digit],
            )
            if digit > 0:
                scale[:, 10 + digit - 1] = torch.where(
                    digit_open,
                    residual.new_full((self.num_envs,), distal_scale_value),
                    scale[:, 10 + digit - 1],
                )
        if self.force_gate_enabled:
            # 每个维度的安全门只依赖上一物理步的实际指尖力。超过
            # soft_high+margin 后逐步收缩至 hard 上限附近的 floor；
            # 三指协同量取必需指中的最小门控，防止任一指被继续推高。
            soft = self._last_physical_force_n.new_tensor(
                (*self.soft_high_n, 1.20, 1.00)
            )
            hard = self._last_physical_force_n.new_tensor((4.0, 3.5, 3.5, 3.0, 3.0))
            if self.force_gate_start_fraction is None:
                guard = soft + self._last_physical_force_n.new_tensor(self.guard_margin_n)
            else:
                fraction = float(self.force_gate_start_fraction)
                if not 0.0 < fraction <= 1.0:
                    raise ValueError("force_gate_start_fraction 必须在 (0, 1] 内")
                guard = soft * fraction
            headroom = ((hard - self._last_physical_force_n) / (hard - guard).clamp_min(1.0e-6)).clamp(0.0, 1.0)
            gate = torch.where(
                self._last_physical_force_n <= guard,
                torch.ones_like(headroom),
                self.force_gate_floor + (1.0 - self.force_gate_floor) * headroom,
            )
            scale[:, :5] *= gate[:, :3].amin(dim=1, keepdim=True)
            for digit in range(5):
                scale[:, 5 + digit] *= gate[:, digit]
                if digit > 0:
                    scale[:, 10 + digit - 1] *= gate[:, digit]
        residual = self._apply_residual_trust_region(
            residual, digit_contact=digit_contact, all_required=all_required
        )
        combined = (guide + scale * residual).clamp(-1.0, 1.0)
        self._guide_action.copy_(guide)
        super()._pre_physics_step(combined)
        # super() 已完成 canonical 映射和安全投影；再对即将发送给 PhysX
        # 的目标做可选保持段限速，避免下一步位置驱动产生大角速度。
        self._joint_targets[:, self._hand_joint_ids] = self._limit_hold_joint_targets(
            self._joint_targets[:, self._hand_joint_ids]
        )
        self._update_hold_velocity_limits()
        self._update_hold_actuator_gains()
        self._update_hold_effort_limits()
        self._update_mimic_targets(torch.arange(self.num_envs, device=self.device))

    def _get_rewards(self) -> torch.Tensor:
        """在原有合同奖励上提供带接触门控的低压力连续梯度。"""

        reward = super()._get_rewards()
        force = self._last_physical_force_n[:, :3]
        target = force.new_tensor(self.force_center_targets_n)
        width = force.new_tensor(self.force_center_width_n).clamp_min(1.0e-6)
        contact = (force >= 0.08).all(dim=1).float()
        center = torch.exp(-((force - target) / width).square()).mean(dim=1) * contact
        # 将实际关节运动纳入可选学习信号。使用无量纲平方量并裁剪，
        # 防止接触瞬态的单步高 jerk 淹没接触奖励；硬安全门仍由
        # Verifier 和过载事件独立决定，不能通过调该权重放宽。
        velocity = self._robot.data.joint_vel.torch[:, self._hand_joint_ids]
        acceleration = (velocity - self._reward_previous_joint_velocity) / (1.0 / 20.0)
        jerk = (acceleration - self._reward_previous_joint_acceleration) / (1.0 / 20.0)
        motion_cost = (
            (velocity.abs() / float(self.joint_velocity_reward_scale)).square().mean(dim=1)
            + 0.20
            * (acceleration.abs() / float(self.joint_acceleration_reward_scale)).square().mean(dim=1)
            + 0.05
            * (jerk.abs() / float(self.joint_jerk_reward_scale)).square().mean(dim=1)
        ).clamp_max(50.0)
        motion_cost *= (self._verifier.hold_age > 0).float()
        smooth_weight = float(self.joint_smoothness_reward_weight)
        if smooth_weight < 0.0:
            raise ValueError("joint_smoothness_reward_weight 不能为负数")
        self._reward_previous_joint_velocity.copy_(velocity)
        self._reward_previous_joint_acceleration.copy_(acceleration)
        reward = reward + self.force_center_reward * center - smooth_weight * motion_cost
        logs = self.extras.setdefault("log", {})
        logs["SmallPinch/force_center_score"] = center.mean().item()
        logs["SmallPinch/joint_motion_cost"] = motion_cost.mean().item()
        return reward


__all__ = [
    "GuidedResidualSmallFruitPinchEnv",
    "NativeSmallFruitPinchCfg",
    "P1_GUIDE_ACTION",
]
