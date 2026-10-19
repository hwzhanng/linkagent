"""复用原逐子步时钟和七故障，分别维护六种固定三指组合的真实连续窗。"""
import math
import torch


class CombinationHoldGuard:
    def __init__(self, num_envs, *, device, initial_counter, physics_dt_s=1/120,
                 decimation=6, max_speed=.05, max_drift=.01):
        self.last_counter = int(initial_counter)
        self.dt = float(physics_dt_s)
        self.decimation = int(decimation)
        self.interval_count = 0
        self.max_speed, self.max_drift = float(max_speed), float(max_drift)
        self.streak = torch.zeros((num_envs, 6), dtype=torch.long, device=device)
        self.peak = torch.zeros_like(self.streak)
        self.anchor = torch.zeros((num_envs, 6, 3), device=device)
        self.last_ok = torch.zeros_like(self.streak, dtype=torch.bool)
        self.drift = torch.full_like(self.streak, float('inf'), dtype=torch.float32)

    @torch.inference_mode()
    def reset(self, ids):
        self.streak[ids] = 0
        self.peak[ids] = 0
        self.anchor[ids] = 0
        self.last_ok[ids] = False
        self.drift[ids] = float('inf')

    @torch.inference_mode()
    def observe(self, *, state, qualified, hard_fault_any, support_zero, counter, dt):
        if not math.isfinite(float(dt)) or abs(float(dt)-self.dt) > 1e-9:
            raise RuntimeError('三指守卫物理dt不符')
        if int(counter) != self.last_counter + 1:
            raise RuntimeError('三指守卫物理子步漏采或重复')
        if qualified.shape != self.streak.shape or qualified.dtype != torch.bool:
            raise RuntimeError('三指组合形状或类型错误')
        self.last_counter = int(counter)
        self.interval_count += 1
        fresh = self.streak == 0
        position = state.fruit_in_hand_m[:, None, :].expand_as(self.anchor)
        self.anchor.copy_(torch.where(fresh[..., None], position, self.anchor))
        drift = torch.linalg.vector_norm(position-self.anchor, dim=-1)
        ok = (qualified & state.released[:, None] & support_zero[:, None]
              & (state.relative_speed_m_s[:, None] <= self.max_speed)
              & ~hard_fault_any[:, None] & (drift <= self.max_drift))
        self.streak = torch.where(ok, self.streak+1, torch.zeros_like(self.streak))
        self.peak = torch.maximum(self.peak, self.streak)
        self.last_ok = ok
        self.drift = torch.where(ok, drift, torch.full_like(drift, float('inf')))

    def complete_policy_interval(self):
        if self.interval_count != self.decimation:
            raise RuntimeError(f'三指守卫采样数错误：{self.interval_count} != {self.decimation}')
        self.interval_count = 0
