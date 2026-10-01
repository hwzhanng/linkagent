#!/usr/bin/env python3
"""从已有 14D Actor 生成可人工验收的恒定弱教师权重（不宣称 PPO 成功）。"""
from pathlib import Path
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--action", nargs=14, type=float, required=True)
args = parser.parse_args()

payload = torch.load(args.source, map_location="cpu", weights_only=False)
actor = payload["actor_state_dict"]
action = torch.tensor(args.action, dtype=torch.float32).clamp(-0.95, 0.95)
actor["mlp.6.weight"].zero_()
actor["mlp.6.bias"].copy_(torch.atanh(action))
payload["actor_state_dict"] = actor
payload["infos"] = dict(payload.get("infos") or {})
payload["infos"]["small_fruit_teacher"] = {
    "type": "constant_14d_weak_teacher",
    "warning": "仅用于动作物理人工验收，不代表 PPO 成功率",
    "action": [float(x) for x in action],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, args.output)
print(args.output)
