"""Physical harvest-cycle environment layered over the frozen Clean-v5 MDP."""

from __future__ import annotations

import torch

from isaaclab.sensors import Camera

from simulation.tomato_pick.isaac.env import TomatoPickEnv

from .isaac_env_cfg import HarvestCycleIsaacEnvCfg


class HarvestCycleIsaacEnv(TomatoPickEnv):
    """Keep one Isaac scene alive across pre-pick, PPO, and post-pick skills."""

    cfg: HarvestCycleIsaacEnvCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._spawn_harvest_container()
        # A single fixed top-center camera observes env_0. The arm may move to
        # reduce self-occlusion, but the camera pose is never wrist-coupled.
        self._harvest_camera = Camera(self.cfg.harvest_camera)
        self.scene.sensors["harvest_camera"] = self._harvest_camera

    def _spawn_harvest_container(self) -> None:
        """Spawn a fixed open bin used by transport/place acceptance."""

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
            "Bottom": ((size_x + 2 * thickness, size_y + 2 * thickness, thickness), (cx, cy, bottom_z)),
            "WallPosX": ((thickness, size_y + 2 * thickness, wall_height), (cx + size_x / 2 + thickness / 2, cy, bottom_z + wall_height / 2)),
            "WallNegX": ((thickness, size_y + 2 * thickness, wall_height), (cx - size_x / 2 - thickness / 2, cy, bottom_z + wall_height / 2)),
            "WallPosY": ((size_x, thickness, wall_height), (cx, cy + size_y / 2 + thickness / 2, bottom_z + wall_height / 2)),
            "WallNegY": ((size_x, thickness, wall_height), (cx, cy - size_y / 2 - thickness / 2, bottom_z + wall_height / 2)),
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
        # Preserve inherited physical outcome calculations and diagnostics,
        # but let the outer harvest-cycle verifier decide when to reset.
        super()._get_dones()
        keep_running = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return keep_running, keep_running

    def aim_harvest_camera_from_top_mount(self) -> None:
        """Apply the fixed top-center, forward/downward camera installation."""

        eye = torch.tensor(
            self.cfg.harvest_camera_mount_world_m,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        target = torch.tensor(
            self.cfg.harvest_camera_look_at_world_m,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._harvest_camera.set_world_poses_from_view(eye, target)

    def capture_harvest_rgbd(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return RGB, optical-axis depth and intrinsics from the current scene."""

        self.aim_harvest_camera_from_top_mount()
        self.sim.render()
        self._harvest_camera.update(self.sim.get_physics_dt(), force_recompute=True)
        output = self._harvest_camera.data.output
        return (
            output["rgb"].clone(),
            output["distance_to_image_plane"].clone(),
            self._harvest_camera.data.intrinsic_matrices.torch.clone(),
        )


__all__ = ["HarvestCycleIsaacEnv"]
