#!/usr/bin/env python3
"""从 direct-joint CEM 候选生成仅供人工回放的左手教师权重。"""
from pathlib import Path
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--control", nargs=10, type=float, required=True,
                    help="按 direct CEM 顺序：0,1,2,5,6,7,10,11,12,13")
parser.add_argument("--center-mm", nargs=3, type=float, required=True)
args = parser.parse_args()

payload = torch.load(args.source, map_location="cpu", weights_only=False)
meta = dict((payload.get("infos") or {}).get("small_fruit_teacher") or {})
direct_indices = (0, 1, 2, 5, 6, 7, 10, 11, 12, 13)
lo = torch.tensor((0.02, 0.02745098, 0.02745098, 0.02745098, 0.02745098, -1.0,
                   -0.147843137, -0.058117647, 0.029568627, 0.158039216,
                   0.0, 0.025294118, 0.021176471, 0.021176471, 0.021176471, 0.021176471))
hi = torch.tensor((0.8, 1.317647059, 1.37254902, 1.37254902, 1.317647059, 0.45,
                   -0.117254902, -0.025490196, 0.062196078, 0.174352941,
                   1.38, 1.25, 1.058823529, 1.058823529, 1.037647059, 1.058823529))
control = torch.tensor(args.control, dtype=torch.float32).clamp(-1.0, 1.0)
# 与 CEM/运行时一致：非搜索轴保持 14D 零动作的安全模板，而不是直接落到
# 关节下限（下限在该 USD 中并非张开姿态，会导致回放首步自碰撞）。
target = torch.tensor((0.66116005, 0.25710589, 0.26687843, 0.26687843,
                       0.25710589, -0.74189997, -0.13254902, -0.04180392,
                       0.04588235, 0.16619608, 1.13436007, 0.24329177,
                       0.20587766, 0.20587766, 0.20210785, 0.20587766), dtype=torch.float32)
target[list(direct_indices)] = lo[list(direct_indices)] + 0.5 * (control + 1.0) * (hi[list(direct_indices)] - lo[list(direct_indices)])
meta.update({
    "type": "corrected_direct_joint_teacher",
    "warning": "仅用于左手离线人工回放与物理验收；未通过连续三指稳定门，不得用于实机或宣称 PPO 成功率",
    "direct_joint_targets": [float(x) for x in target],
    "direct_joint_control": [float(x) for x in control],
    "canonical_joint_indices": list(direct_indices),
    "center_local_mm": [float(x) for x in args.center_mm],
})
payload["infos"] = dict(payload.get("infos") or {})
payload["infos"]["small_fruit_teacher"] = meta
payload["infos"]["manual_inspection_contract"] = {
    "type": "left_hand_only_corrected_direct_teacher",
    "center_local_mm": [float(x) for x in args.center_mm],
    "stages": ["distal_preposition", "thumb_opposition_and_distal_closure", "root_closure"],
    "warning": "人工验收权重，不是训练成功权重；回放必须启用安全停止并保持离线。",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, args.output)
print(args.output)
