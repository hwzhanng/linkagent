"""Render a saved nearest-replay trajectory as a lightweight URDF skeleton movie."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    mimic: tuple[str, float, float] | None


ACTIVE_ORDER = (
    "thumb_joint2", "index_joint1", "middle_joint1", "ring_joint1", "little_joint1",
    "thumb_joint0", "index_joint0", "middle_joint0", "ring_joint0", "little_joint0",
    "thumb_joint1", "thumb_joint3", "index_joint2", "middle_joint2", "ring_joint2", "little_joint2",
)


def _vec(text: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.fromstring(text or default, sep=" ", dtype=float)


def _rpy(rpy: np.ndarray) -> np.ndarray:
    # URDF uses fixed-axis roll-pitch-yaw, equivalent to Rz(yaw)Ry(pitch)Rx(roll).
    return Rotation.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()


def _transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def _parse_urdf(path: Path) -> tuple[str, list[Joint]]:
    root = ET.parse(path).getroot()
    links = [x.attrib["name"] for x in root.findall("link")]
    child_links = set()
    joints: list[Joint] = []
    for item in root.findall("joint"):
        parent = item.find("parent").attrib["link"]
        child = item.find("child").attrib["link"]
        child_links.add(child)
        origin = item.find("origin")
        axis = item.find("axis")
        mimic = item.find("mimic")
        mimic_data = None
        if mimic is not None:
            mimic_data = (
                mimic.attrib["joint"],
                float(mimic.attrib.get("multiplier", "1")),
                float(mimic.attrib.get("offset", "0")),
            )
        joints.append(Joint(
            name=item.attrib["name"],
            parent=parent,
            child=child,
            kind=item.attrib.get("type", "fixed"),
            xyz=_vec(origin.attrib.get("xyz") if origin is not None else None),
            rpy=_vec(origin.attrib.get("rpy") if origin is not None else None),
            axis=_vec(axis.attrib.get("xyz") if axis is not None else None, "0 0 0"),
            mimic=mimic_data,
        ))
    root_link = next(link for link in links if link not in child_links)
    return root_link, joints


def _forward_kinematics(root_link: str, joints: list[Joint], active: np.ndarray) -> dict[str, np.ndarray]:
    by_parent: dict[str, list[Joint]] = {}
    q = {name: float(value) for name, value in zip(ACTIVE_ORDER, active)}
    for joint in joints:
        by_parent.setdefault(joint.parent, []).append(joint)

    poses: dict[str, np.ndarray] = {root_link: np.eye(4)}

    def visit(parent: str) -> None:
        for joint in by_parent.get(parent, []):
            angle = q.get(joint.name, 0.0)
            if joint.mimic is not None:
                master, multiplier, offset = joint.mimic
                angle = multiplier * q.get(master, 0.0) + offset
            motion = np.eye(4)
            if joint.kind in ("revolute", "continuous"):
                axis = joint.axis / max(np.linalg.norm(joint.axis), 1e-12)
                motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            poses[joint.child] = poses[parent] @ _transform(_rpy(joint.rpy), joint.xyz) @ motion
            visit(joint.child)

    visit(root_link)
    return poses


def _render(args: argparse.Namespace) -> None:
    data = np.load(args.trajectory)
    joint_frames = data["joint_pos"]
    sphere_frames = data["sphere_pos"]
    contact = data["contact_forces"]
    urdf = Path(args.urdf)
    root_link, joints = _parse_urdf(urdf)
    link_edges = [(joint.parent, joint.child) for joint in joints]
    frame_ids = np.arange(0, len(joint_frames), max(1, args.stride))

    fig = plt.figure(figsize=(7, 7), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    frames: list[np.ndarray] = []
    for frame_id in frame_ids:
        poses = _forward_kinematics(root_link, joints, joint_frames[frame_id])
        ax.clear()
        for parent, child in link_edges:
            if parent in poses and child in poses:
                p0, p1 = poses[parent][:3, 3], poses[child][:3, 3]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#5b6472", linewidth=2.2)
        pts = np.array([pose[:3, 3] for pose in poses.values()])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=11, color="#334155", alpha=0.9)
        sphere = sphere_frames[frame_id]
        ax.scatter([sphere[0]], [sphere[1]], [sphere[2]], s=260, color="#e63946", edgecolors="white", linewidths=1.2)
        forces = contact[frame_id]
        ax.text2D(0.03, 0.95, f"reconstructed nearest replay  |  frame {frame_id}/{len(joint_frames)-1}\n"
                  f"contact N: " + ", ".join(f"{x:.2f}" for x in forces), transform=ax.transAxes, fontsize=9)
        ax.set_title("PhysX sphere grasp — URDF kinematic view")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(-0.16, 0.10); ax.set_ylim(-0.16, 0.16); ax.set_zlim(-0.02, 0.38)
        ax.view_init(elev=19, azim=-73)
        ax.set_box_aspect((1, 1, 1.4))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(rgba)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=1.0 / args.fps, loop=0)
    if args.png_frame is not None:
        imageio.imwrite(args.png_frame, frames[-1])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-frame", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=110)
    _render(parser.parse_args())


if __name__ == "__main__":
    main()

