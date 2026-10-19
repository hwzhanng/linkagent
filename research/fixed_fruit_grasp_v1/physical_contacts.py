"""读取已有 PhysX packed 接触距离，AABB 重叠不作为真实穿透深度。"""
from __future__ import annotations
import re
import torch
import warp as wp


def penetration_from_sensors(sensors, num_envs: int, device):
    """按具名刚体→env映射汇总当前策略步的真实接触距离；通道缺失直接失败。"""
    deepest = torch.zeros(num_envs, device=device)
    totals = torch.zeros(num_envs, dtype=torch.long, device=device)
    for sensor in sensors:
        # 仅新指肚任务安装逐物理步缓存；旧任务不安装时保持原始读取路径。
        cached_reader = getattr(sensor, '_thumb_pad_contact_read', None)
        raw = (cached_reader() if cached_reader is not None
               else sensor.contact_view.get_contact_data(1.0 / 120.0))
        if len(raw) != 6:
            raise RuntimeError('PhysX packed 接触布局不是6个数组')
        counts = wp.to_torch(raw[4]).reshape(-1).long()
        starts = wp.to_torch(raw[5]).reshape(-1).long()
        separations = wp.to_torch(raw[3]).reshape(-1)
        paths = sensor.body_physx_view.prim_paths
        if len(paths) != num_envs or counts.numel() != num_envs or starts.numel() != num_envs:
            raise RuntimeError('单刚体传感器的接触行数与环境数不一致')
        env_indices = []
        for path in paths:
            match = re.search(r'/env_(\d+)/', path)
            if match is None:
                raise RuntimeError('接触刚体路径缺少env索引')
            env_indices.append(int(match.group(1)))
        if sorted(env_indices) != list(range(num_envs)):
            raise RuntimeError('接触刚体路径没有一一覆盖环境')
        capacity = int(sensor.cfg.max_contact_data_count_per_prim)
        if bool(((counts < 0) | (counts >= capacity) | ((counts > 0) & ((starts < 0) | (starts + counts > separations.numel())))).any()):
            raise RuntimeError(f'接触缓冲越界或达到容量上限，不能判为无穿透: prim={sensor.cfg.prim_path if hasattr(sensor.cfg, "prim_path") else "unknown"}, capacity={capacity}, max_count={int(counts.max())}, separation_slots={separations.numel()}, starts={starts.tolist()}')
        # 空缓冲无需读取；unused 槽位不能纳入有限性或最小距离检查。
        columns = torch.arange(capacity, device=device).unsqueeze(0)
        used = columns < counts.unsqueeze(1)
        indices = (starts.unsqueeze(1) + columns).clamp(0, max(0, separations.numel() - 1))
        if separations.numel() == 0:
            if bool((counts > 0).any()):
                raise RuntimeError('接触数量非零但距离缓冲为空')
            depths = torch.zeros(num_envs, device=device)
        else:
            values = separations[indices]
            if not bool(torch.isfinite(values[used]).all()):
                raise RuntimeError('真实接触距离包含NaN/Inf')
            values = torch.where(used, values, torch.full_like(values, float('inf')))
            depths = torch.relu(-values.amin(dim=1))
        ids = torch.tensor(env_indices, device=device, dtype=torch.long)
        by_env = torch.zeros_like(deepest).scatter_(0, ids, depths)
        count_by_env = torch.zeros_like(totals).scatter_(0, ids, counts)
        deepest = torch.maximum(deepest, by_env)
        totals += count_by_env
    return deepest, totals
