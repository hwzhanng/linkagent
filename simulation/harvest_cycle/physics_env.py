"""Renderer-free physical harvest cycle for PPO and post-pick acceptance."""

from __future__ import annotations

import torch

from simulation.tomato_pick.isaac.env import TomatoPickEnv

from .isaac_env_cfg import HarvestCycleIsaacEnvCfg


class HarvestCyclePhysicsIsaacEnv(TomatoPickEnv):
    """Keep the physical scene alive without allocating an RGB-D renderer."""

    cfg: HarvestCycleIsaacEnvCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._spawn_harvest_container()

    def _spawn_harvest_container(self) -> None:
        import isaaclab.sim as sim_utils

        cx, cy, bottom_z = self.cfg.container_center_world_m
        size_x, size_y, wall_height = self.cfg.container_inner_size_m
        thickness = self.cfg.container_wall_thickness_m
        material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.08, 0.24, 0.62), roughness=0.72
        )
        common = {
            "collision_props": sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0
            ),
            "visual_material": material,
        }
        parts = {
            "Bottom": (
                (size_x + 2 * thickness, size_y + 2 * thickness, thickness),
                (cx, cy, bottom_z),
            ),
            "WallPosX": (
                (thickness, size_y + 2 * thickness, wall_height),
                (cx + size_x / 2 + thickness / 2, cy, bottom_z + wall_height / 2),
            ),
            "WallNegX": (
                (thickness, size_y + 2 * thickness, wall_height),
                (cx - size_x / 2 - thickness / 2, cy, bottom_z + wall_height / 2),
            ),
            "WallPosY": (
                (size_x, thickness, wall_height),
                (cx, cy + size_y / 2 + thickness / 2, bottom_z + wall_height / 2),
            ),
            "WallNegY": (
                (size_x, thickness, wall_height),
                (cx, cy - size_y / 2 - thickness / 2, bottom_z + wall_height / 2),
            ),
        }
        for name, (size, translation) in parts.items():
            shape = sim_utils.CuboidCfg(size=size, **common)
            shape.func(
                f"/World/HarvestContainer/{name}",
                shape,
                translation=translation,
            )

    @property
    def container_placement_position_w(self) -> tuple[float, float, float]:
        cx, cy, bottom_z = self.cfg.container_center_world_m
        return (cx, cy, bottom_z + self.cfg.container_inner_size_m[2] * 0.72)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        super()._get_dones()
        keep_running = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return keep_running, keep_running


__all__ = ["HarvestCyclePhysicsIsaacEnv"]
