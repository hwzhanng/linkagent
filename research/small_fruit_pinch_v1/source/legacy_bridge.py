"""只读加载历史左手 PhysX 资产接口。

历史工作区不是本任务的源码目录；本模块仅在运行时把其已验证的 USD 与
低层 PhysX 环境作为外部依赖加载。新任务的奖励、成功判据与训练产物均在
``research/small_fruit_pinch_v1`` 内独立维护。
"""

from __future__ import annotations

import importlib
from pathlib import Path


HISTORICAL_ROOT = Path("/home/miobarnacle/linkbot/ppo_tomato_grasp_stage01_work")


def load_legacy_physx() -> tuple[type, type]:
    if not HISTORICAL_ROOT.is_dir():
        raise FileNotFoundError(f"缺少只读历史 PhysX 工作区：{HISTORICAL_ROOT}")
    # 历史环境使用 Isaac Lab 2.x 的 ``isaaclab.sim.PhysxCfg`` 导出；当前
    # 受控 3.0 运行时将它移动到 ``isaaclab_physx.physics``。仅补齐这个
    # 导入别名，既不修改历史文件，也不改变 PhysX 参数本身。
    import isaaclab.sim as sim_utils
    if not hasattr(sim_utils, "PhysxCfg"):
        from isaaclab_physx.physics import PhysxCfg

        sim_utils.PhysxCfg = PhysxCfg
    # ``SimulationCfg(physx=...)`` 也是旧命名。使用一个窄兼容构造器将
    # 关键字映射到当前 ``physics``，其余参数仍完全交给当前运行时校验。
    if not getattr(sim_utils.SimulationCfg, "_small_fruit_legacy_compat", False):
        current_simulation_cfg = sim_utils.SimulationCfg

        def compatible_simulation_cfg(*args, **kwargs):
            if "physx" in kwargs:
                if "physics" in kwargs:
                    raise TypeError("cannot pass both physx and physics")
                kwargs["physics"] = kwargs.pop("physx")
            return current_simulation_cfg(*args, **kwargs)

        compatible_simulation_cfg._small_fruit_legacy_compat = True
        sim_utils.SimulationCfg = compatible_simulation_cfg
    import simulation

    historical_simulation = str(HISTORICAL_ROOT / "simulation")
    if historical_simulation not in simulation.__path__:
        simulation.__path__.append(historical_simulation)
    module = importlib.import_module("simulation.ppo_tomato_grasp.physx_tomato_env")
    return module.PhysxTomatoGraspEnv, module.PhysxTomatoGraspEnvCfg


def load_legacy_hand_manifold():
    """只读复用历史 10D→16 关节映射，不加载旧 Isaac 环境。"""

    if not HISTORICAL_ROOT.is_dir():
        raise FileNotFoundError(f"缺少只读历史 PhysX 工作区：{HISTORICAL_ROOT}")
    import simulation

    historical_simulation = str(HISTORICAL_ROOT / "simulation")
    if historical_simulation not in simulation.__path__:
        simulation.__path__.append(historical_simulation)
    return importlib.import_module("simulation.ppo_tomato_grasp.hand_manifold")
