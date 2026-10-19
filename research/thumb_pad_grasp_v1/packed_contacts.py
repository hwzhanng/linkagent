"""实际六数组布局的逐物理步缓存；仅已用槽位参与有效性与几何判断。"""
from __future__ import annotations
from dataclasses import dataclass
import re
import torch


@dataclass
class PackedPoints:
    env_ids: torch.Tensor
    forces_n: torch.Tensor
    points_w: torch.Tensor
    normals_inward_w: torch.Tensor
    separations_m: torch.Tensor
    counts_by_env: torch.Tensor
    invalid_by_env: torch.Tensor | None = None


def unpack_points(arrays, *, paths, num_envs, capacity, invalid_policy="raise"):
    if invalid_policy not in {"raise", "mark_unknown"}:
        raise ValueError("未知无效接触处理模式")
    """严格绑定一刚体、一果实过滤器；单位由真实预检查而非 mock 确定。"""
    if len(arrays) != 6:
        raise ValueError('接触布局必须为6数组')
    forces, points, normals, separations, counts, starts = arrays
    size = forces.shape[0]
    if forces.shape != (size,1) or points.shape != (size,3) or normals.shape != (size,3) or separations.shape != (size,1):
        raise ValueError('packed 接触点数组shape与实测布局不符')
    if counts.shape != (num_envs,1) or starts.shape != (num_envs,1) or len(paths) != num_envs:
        raise ValueError('接触过滤行数必须与环境数一一对应')
    env_rows = []
    body_names = set()
    for path in paths:
        match = re.fullmatch(r'/World/envs/env_(\d+)/Robot/([^/]+)',path)
        if match is None:
            raise ValueError('传感器刚体路径不满足已绑定的环境命名')
        env_rows.append(int(match.group(1)))
        body_names.add(match.group(2))
    if sorted(env_rows) != list(range(num_envs)) or len(body_names) != 1:
        raise ValueError('传感器路径没有覆盖相同具名刚体的全部环境')
    count = counts.reshape(-1).long()
    start = starts.reshape(-1).long()
    invalid = (count<0) | (count>=capacity) | ((count>0) & ((start<0) | (start+count>size)))
    if bool(invalid.any()):
        raise ValueError('接触缓冲溢出或越界，不能判定为无接触')
    indices, env_ids = [], []
    for row, env_id in enumerate(env_rows):
        n = int(count[row])
        if n:
            indices.append(torch.arange(int(start[row]),int(start[row])+n,device=forces.device))
            env_ids.append(torch.full((n,),env_id,dtype=torch.long,device=forces.device))
    index = torch.cat(indices) if indices else torch.empty(0,dtype=torch.long,device=forces.device)
    env_id = torch.cat(env_ids) if env_ids else torch.empty(0,dtype=torch.long,device=forces.device)
    used = [forces[index,0],points[index],normals[index],separations[index,0]]
    invalid_point = (~torch.isfinite(used[0])) | (used[0] < 0)
    invalid_point |= ~torch.isfinite(used[1]).all(dim=1)
    invalid_point |= ~torch.isfinite(used[2]).all(dim=1)
    invalid_point |= ~torch.isfinite(used[3])
    bad_normal = (torch.linalg.vector_norm(used[2],dim=1)-1).abs().gt(1e-3)
    invalid_point |= bad_normal
    invalid_by_env = torch.zeros(num_envs,dtype=torch.bool,device=forces.device)
    if bool(invalid_point.any()):
        invalid_by_env[env_id[invalid_point]] = True
        if invalid_policy == 'raise':
            if bool(bad_normal.any()):
                raise ValueError('有效接触法向不是单位向量')
            raise ValueError('有效接触槽位包含非有限值或负标量法向力: '+repr({'body':paths[0], 'forces':used[0].tolist(), 'points':used[1].tolist(), 'normals':used[2].tolist(), 'separations':used[3].tolist()}))
        # 标为UNKNOWN的环境不能累积成功；不对负力取绝对值，也不推断为无接触。
        keep = ~invalid_point
        env_id = env_id[keep]
        used = [value[keep] for value in used]
    by_env = torch.zeros(num_envs,dtype=torch.long,device=forces.device)
    by_env[torch.tensor(env_rows,device=forces.device)] = count
    return PackedPoints(env_id,*used,by_env,invalid_by_env)


class PhysicalStepContactCache:
    """每个真实物理计数/传感器只调用一次底层 getter。reset 的 dt=0 不判成功。"""
    def __init__(self, step_counter):
        self.step_counter = step_counter
        self.counter = None
        self.values = {}

    def read(self, sensor, dt_s=1/120):
        if abs(float(dt_s)-1/120)>1e-9:
            raise ValueError('接触力dt须为实测1/120秒')
        counter = int(self.step_counter())
        if self.counter != counter:
            self.values.clear()
            self.counter = counter
        key = id(sensor)
        if key not in self.values:
            self.values[key] = sensor.contact_view.get_contact_data(dt_s)
        return self.values[key]

    def invalidate(self):
        self.counter = None
        self.values.clear()
