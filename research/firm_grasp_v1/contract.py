"""firm_grasp_v1 的可序列化抓握判据。"""
from dataclasses import dataclass
from math import isfinite
from typing import Final


DIGITS: Final[tuple[str, ...]] = ("thumb", "index", "middle", "ring", "little")
COMBINATIONS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
    (0, 3, 4),
)


@dataclass(frozen=True)
class FirmGraspContract:
    """定义每步的 PAD 受力阈值和连续承载时长要求。

    required_substeps 表示上层状态 guard 必须连续检查的步数；本模块的
    指标函数只对单批观测逐行判定，不负责计时或跨步累积。
    """

    f_pad_min_n: float = 0.15
    min_load_share: float = 0.05
    f_nonpad_noise_max_n: float = 0.0
    required_substeps: int = 120
    schema_version: str = "firm_grasp_v1.7.0"
    task_version: str = "firm_grasp_v1.7.0"

    def __post_init__(self) -> None:
        """拒绝无效阈值，避免配置错误放宽抓握判据。"""
        for name in ("f_pad_min_n", "min_load_share", "f_nonpad_noise_max_n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是有限数值")
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        if self.min_load_share > 1:
            raise ValueError("min_load_share 必须位于 [0, 1]")
        if self.f_pad_min_n <= 0 or self.min_load_share <= 0:
            raise ValueError("PAD力与承载份额下限必须严格为正，禁止无接触通过")
        if type(self.required_substeps) is not int or self.required_substeps <= 0:
            raise ValueError("required_substeps 必须是正整数")
        for name in ("schema_version", "task_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} 必须是非空字符串")



def load_frozen_contract() -> FirmGraspContract:
    """训练和评估均读取F1冻结文件，不能悄悄回退到宽松默认值。"""
    import json
    from pathlib import Path
    payload = json.loads(Path(__file__).with_name("firm_grasp_contract_v1.json").read_text())
    contract = FirmGraspContract(**payload["contract"])
    if contract != FirmGraspContract():
        raise ValueError("F1冻结判据与当前实现版本不一致，必须显式版本化")
    return contract
