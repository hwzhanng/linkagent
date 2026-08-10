"""Camera-enabled, non-resetting configuration for atomic-skill acceptance."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from simulation.tomato_pick.isaac.env_cfg import TomatoPickEnvCfg


@configclass
class HarvestCycleIsaacEnvCfg(TomatoPickEnvCfg):
    """One physical tomato scene that remains alive after PPO termination.

    The inherited robot, fruit, stem, contact sensors, reset distribution and
    66D/83D/11D policy contract stay unchanged. This overlay only adds RGB-D
    and enough workspace/time for pre- and post-pick atomic skills.
    """

    episode_length_s = 120.0
    scene = TomatoPickEnvCfg().scene.replace(
        num_envs=1,
        env_spacing=1.8,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    action_delay_steps = (0, 0)
    target_position_noise_m = 0.0
    fruit_axis_relative_noise = 0.0
    tactile_force_gain = (1.0, 1.0)
    tactile_force_bias_n = (0.0, 0.0)

    container_center_world_m = (0.25, -0.25, 1.69)
    container_inner_size_m = (0.18, 0.18, 0.12)
    container_wall_thickness_m = 0.012

    # Fixed camera centered above the bimanual shoulder line, looking toward
    # the forward/downward harvest volume.  It does not move with either arm.
    harvest_camera_mount_world_m = (0.02, -0.05, 2.38)
    harvest_camera_look_at_world_m = (0.43, -0.08, 1.76)

    harvest_camera = CameraCfg(
        prim_path="/World/HarvestCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="world",
        ),
        update_latest_camera_pose=True,
    )


__all__ = ["HarvestCycleIsaacEnvCfg"]
