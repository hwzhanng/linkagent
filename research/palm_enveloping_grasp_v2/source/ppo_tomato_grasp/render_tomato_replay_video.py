"""Render a recorded tomato grasp replay to an MP4/GIF without opening a GUI."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.ppo_tomato_grasp.render_nearest_replay import (
    ACTIVE_ORDER,
    _forward_kinematics,
    _parse_urdf,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    data = np.load(args.replay)
    joint_frames = np.asarray(data["joint_pos_rad"])
    tomato_frames = np.asarray(data["tomato_pos_m"])
    digit_force = np.asarray(data["digit_force_n"])
    palm_force = np.asarray(data["palm_force_n"])
    contact_names = ("thumb", "index", "middle", "ring", "little")
    root_link, joints = _parse_urdf(args.urdf)
    link_edges = [(joint.parent, joint.child) for joint in joints]
    frame_ids = np.arange(0, len(joint_frames), max(1, args.stride))

    fig = plt.figure(figsize=(11, 7), dpi=args.dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.25, 0.75))
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    pressure_ax = fig.add_subplot(grid[0, 1])
    pressure_ax.set_title("contact force")
    pressure_ax.set_xlabel("policy step (20 Hz)")
    pressure_ax.set_ylabel("N")
    pressure_ax.set_xlim(0, max(1, len(joint_frames) - 1))
    pressure_ax.set_ylim(0, max(2.0, float(digit_force.max()) * 1.25, float(palm_force.max()) * 1.25))
    pressure_lines = [pressure_ax.plot([], [], label=name)[0] for name in contact_names]
    palm_line = pressure_ax.plot([], [], "k--", label="palm")[0]
    pressure_ax.legend(loc="upper left", fontsize=8, ncol=2)
    frames: list[np.ndarray] = []
    steps = np.arange(len(joint_frames))
    for frame_id in frame_ids:
        poses = _forward_kinematics(root_link, joints, joint_frames[frame_id])
        ax.clear()
        for parent, child in link_edges:
            if parent in poses and child in poses:
                p0, p1 = poses[parent][:3, 3], poses[child][:3, 3]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#475569", linewidth=2.3)
        points = np.array([pose[:3, 3] for pose in poses.values()])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=13, color="#1e293b")
        tomato = tomato_frames[frame_id]
        ax.scatter([tomato[0]], [tomato[1]], [tomato[2]], s=320, color="#dc2626", edgecolors="white", linewidths=1.4)
        force_text = ", ".join(f"{value:.2f}" for value in digit_force[frame_id])
        ax.text2D(
            0.02,
            0.96,
            f"frame {frame_id}/{len(joint_frames)-1}  |  digit N: {force_text}\n"
            f"palm: {palm_force[frame_id]:.2f} N",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )
        ax.set_title("Final PhysX tomato grasp replay")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_xlim(-0.16, 0.16)
        ax.set_ylim(-0.16, 0.16)
        ax.set_zlim(-0.02, 0.38)
        ax.view_init(elev=21, azim=-67)
        ax.set_box_aspect((1, 1, 1.35))
        pressure_ax.clear()
        pressure_ax.set_title("contact force")
        pressure_ax.set_xlabel("policy step (20 Hz)")
        pressure_ax.set_ylabel("N")
        pressure_ax.set_xlim(0, max(1, len(joint_frames) - 1))
        pressure_ax.set_ylim(0, max(2.0, float(digit_force.max()) * 1.25, float(palm_force.max()) * 1.25))
        for index, name in enumerate(contact_names):
            pressure_ax.plot(steps[: frame_id + 1], digit_force[: frame_id + 1, index], label=name)
        pressure_ax.plot(steps[: frame_id + 1], palm_force[: frame_id + 1], "k--", label="palm")
        pressure_ax.axvline(frame_id, color="#94a3b8", linewidth=0.8)
        pressure_ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        imageio.mimsave(args.output, frames, duration=1.0 / max(args.fps, 1.0), loop=0)
    else:
        imageio.mimsave(args.output, frames, fps=max(args.fps, 1.0), codec="libx264", quality=7)
    plt.close(fig)
    print(f"[video] wrote {args.output.resolve()} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
