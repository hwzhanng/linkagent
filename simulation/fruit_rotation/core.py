"""Pure tensor contracts for the single-L20 fruit-rotation experiment.

All fruit surface pressure values are *contact-derived proxy pressures*, not
measured tissue stress or calibrated bruise thresholds.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

FINGERS = ("thumb", "index", "middle", "ring", "little")
ACTIVE_JOINTS = (
    "thumb_joint2", "index_joint1", "middle_joint1", "ring_joint1",
    "little_joint1", "thumb_joint0", "index_joint0", "middle_joint0",
    "ring_joint0", "little_joint0", "thumb_joint1", "thumb_joint3",
    "index_joint2", "middle_joint2", "ring_joint2", "little_joint2",
)
MIMICS = {
    "thumb_joint4": ("thumb_joint3", 1.11),
    "index_joint3": ("index_joint2", 0.94),
    "middle_joint3": ("middle_joint2", 0.94),
    "ring_joint3": ("ring_joint2", 0.94),
    "little_joint3": ("little_joint2", 0.94),
}
POLICY_PARTS = (
    ("tactile", 360),              # five 12x6 fingertip maps; unitless / 5 N proxy
    ("sparse_pressure", 24),      # six 2x2 physical-layout panels / P_REF
    ("sparse_pressure_delta", 24),
    ("sparse_valid", 24),         # per panel dropout and invalidity
    ("joint_pos", 16), ("joint_vel", 16),
    ("fruit_rel_pos", 3), ("fruit_rotation6d", 6),
    ("fruit_lin_vel", 3), ("fruit_ang_vel", 3),
    ("gravity_hand", 3), ("rotation_axis_hand", 3),
    ("desired_speed", 1), ("previous_action", 16),
)
PRIV_PARTS = (
    ("dense_pressure", 96),      # simulator oracle ONLY in critic and targets
    ("fruit_pose_w", 7), ("fruit_lin_vel_w", 3), ("fruit_ang_vel_w", 3),
    ("contact_forces", 6), ("fruit_mass", 1),
    ("fruit_axes", 3), ("unwrapped_angle", 1),
)
POLICY_DIM = sum(n for _, n in POLICY_PARTS)
CRITIC_DIM = POLICY_DIM + sum(n for _, n in PRIV_PARTS)
ACTION_DIM = len(ACTIVE_JOINTS)
PRESSURE_REF_PA = 5000.0  # scale, not a validated damage threshold
SAFETY_PROXY_THRESHOLD = 0.70  # = 3.5 kPa, uncalibrated engineering prior


def offsets(parts: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    start = 0
    mapping: dict[str, slice] = {}
    for name, count in parts:
        mapping[name] = slice(start, start + count)
        start += count
    return mapping


POLICY_SLICES = offsets(POLICY_PARTS)


def surface_sites(device: torch.device | str, dtype: torch.dtype = torch.float32):
    """96 approximately equal-area dense sites and 24 sites in six 2x2 boxes."""
    k = torch.arange(96, device=device, dtype=dtype)
    z = 1.0 - 2.0 * (k + 0.5) / 96.0
    angle = k * (math.pi * (3.0 - math.sqrt(5.0)))
    circle = (1.0 - z * z).sqrt()
    dense = torch.stack((circle * angle.cos(), circle * angle.sin(), z), -1)
    panels = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            i, j = (axis + 1) % 3, (axis + 2) % 3
            for du in (-0.12, 0.12):
                for dv in (-0.12, 0.12):
                    point = torch.zeros(3, device=device, dtype=dtype)
                    point[axis], point[i], point[j] = sign, du, dv
                    panels.append(F.normalize(point, dim=0))
    return dense, torch.stack(panels)


def quat_inv_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by inverse scalar-first (w,x,y,z) quaternion."""
    qv = quat[..., 1:]
    t = 2.0 * torch.cross(qv, vec, dim=-1)
    return vec - quat[..., :1] * t + torch.cross(qv, t, dim=-1)


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    qv = quat[..., 1:]
    t = 2.0 * torch.cross(qv, vec, dim=-1)
    return vec + quat[..., :1] * t + torch.cross(qv, t, dim=-1)


def pressure_from_contacts(
    force_n: torch.Tensor,        # [N,6] five tips + optional palm
    contact_pos_w: torch.Tensor,  # [N,6,3]
    fruit_pos_w: torch.Tensor,    # [N,3]
    fruit_quat_w: torch.Tensor,   # [N,4]
    fruit_axes_m: torch.Tensor,   # [N,3]
    sites: torch.Tensor,         # [K,3] in fruit unit coordinates
    sigma_m: float = 0.010,
) -> torch.Tensor:
    """Gaussian pressure proxy from resolved contact locations and normal loads.

    Assumes one effective contact point/normal force per measured rigid link.
    It does not capture uninstrumented contacts or real fruit deformation.
    """
    n, c = force_n.shape
    local = quat_inv_rotate(
        fruit_quat_w[:, None, :].expand(-1, c, -1),
        contact_pos_w - fruit_pos_w[:, None, :],
    )
    # Project measured contacts onto ellipsoid and reject missing far-away data.
    unit = local / fruit_axes_m[:, None, :].clamp_min(1e-3)
    distance = torch.linalg.vector_norm(unit, dim=-1).clamp_min(1e-7)
    projected = (unit / distance[..., None]) * fruit_axes_m[:, None, :]
    valid = (force_n > 0.025) & (distance > 0.65) & (distance < 1.5)
    locations = sites[None, None, :, :] * fruit_axes_m[:, None, None, :]
    sqdist = ((projected[:, :, None, :] - locations) ** 2).sum(-1)
    kernel = torch.exp(-sqdist / (2.0 * sigma_m ** 2))
    kernel = kernel * (force_n * valid).unsqueeze(-1)
    return kernel.sum(dim=1) / (2.0 * math.pi * sigma_m ** 2)


def fingertip_taxels(
    force_n: torch.Tensor,    # [N,5], force proxy, not calibrated G20 output
    local_xy_m: torch.Tensor, # [N,5,2] sensor-pad local position
) -> torch.Tensor:
    """Deterministic 5x12x6 tactile surrogate; does not model real taxel wiring."""
    device, dtype = force_n.device, force_n.dtype
    y = torch.linspace(-0.0072, 0.0072, 12, device=device, dtype=dtype)
    x = torch.linspace(-0.0048, 0.0048, 6, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    d2 = ((grid[None, None] - local_xy_m[:, :, None, None]) ** 2).sum(-1)
    weights = torch.exp(-d2 / (2.0 * 0.0025 ** 2))
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (weights * force_n[:, :, None, None] / 5.0 * 72.0).clamp(0.0, 4.0)


def pressure_risk(normalized_dense: torch.Tensor, threshold: float = SAFETY_PROXY_THRESHOLD):
    """Smooth, upper-tail-localized risk; no incentive to reduce safe pressure."""
    excess = F.relu(normalized_dense - threshold)
    k = max(1, normalized_dense.shape[-1] // 10)
    return excess.topk(k, dim=-1).values.square().mean(-1)


def tangent_project(stress_grad: torch.Tensor, task_grad: torch.Tensor,
                    damping: float = 0.03) -> torch.Tensor:
    """Orthogonally suppress local changes in predicted rotation."""
    scale = (stress_grad * task_grad).sum(-1, keepdim=True) / (
        task_grad.square().sum(-1, keepdim=True) + damping
    )
    return stress_grad - scale * task_grad


def rotation_reward(
    signed_omega: torch.Tensor,
    desired_speed: torch.Tensor,
    force_n: torch.Tensor,
    dense_pressure: torch.Tensor,
    translation_error: torch.Tensor,
    tilt_error: torch.Tensor,
    slip_speed: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    dt: float,
):
    """Dense objective, with stop/stall penalty and no positive pressure-drop reward."""
    direction = desired_speed.sign()
    progress = (direction * signed_omega * dt).clamp(-0.20, 0.20)
    thumb = force_n[:, 0] > 0.15
    others = (force_n[:, 1:5] > 0.15).sum(-1) >= 2
    supported = (thumb & others).float()
    follow = torch.exp(-((signed_omega - desired_speed) / 0.45).square())
    risk = pressure_risk(dense_pressure)
    # Detached metrics; PPO updates from advantage. Auxiliary gradient uses model.
    terms = {
        "rotation": 8.0 * progress * supported,
        "speed_tracking": 0.13 * follow * supported,
        "stable_support": 0.025 * supported,
        "unsecured_motion": -0.075 * (1.0 - supported),
        # Pressure risk is now optimized as a separate primal-dual PPO constraint.
        "centering": -0.04 * (translation_error / 0.05).square().clamp(max=9),
        "tilt": -0.06 * tilt_error.square(),
        "slip": -0.03 * slip_speed.clamp(0.0, 1.0),
        "action_smoothness": -0.006 * (actions - previous_actions).square().mean(-1),
        "time": torch.full_like(progress, -0.025),
    }
    reward = torch.stack(list(terms.values()), dim=-1).sum(-1)
    return reward, terms


assert POLICY_DIM == 502, POLICY_DIM
assert CRITIC_DIM == 622, CRITIC_DIM
assert ACTION_DIM == 16
