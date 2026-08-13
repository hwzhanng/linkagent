"""RGB-D, point-cloud, health, and placement evidence from the Isaac scene."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from isaaclab.sensors.camera.utils import create_pointcloud_from_depth

from proto import (
    CameraHealthSnapshot,
    CameraIntrinsics,
    ClearanceScene,
    DroppedPayloadLocalizationRequest,
    DroppedPayloadObservationEvidence,
    EmergencyStopHealthSnapshot,
    HarvestContainer,
    HarvestPayload,
    PlacementReceipt,
    PlacementVerificationEvidence,
    Pose,
    PregraspClearanceRequest,
    TargetReobserveEvidence,
    TargetReobserveRequest,
    TomatoTarget,
    VisionWorkerHealthSnapshot,
)
from script.tomato_vision_spool import (
    RgbdSpoolFrame,
    SpoolPublication,
    TomatoVisionSpoolWriter,
)

from .robotics import IsaacHarvestRoboticsBundle


@dataclass(frozen=True, slots=True)
class IsaacRgbdSample:
    rgb: torch.Tensor
    depth_m: torch.Tensor
    intrinsics: torch.Tensor
    camera_position_w: torch.Tensor
    camera_quaternion_ros_xyzw: torch.Tensor
    sample_monotonic_ns: int


def capture_isaac_rgbd(bundle: IsaacHarvestRoboticsBundle) -> IsaacRgbdSample:
    rgb, depth, intrinsics = bundle.env.capture_harvest_rgbd()
    camera = bundle.env._harvest_camera.data
    return IsaacRgbdSample(
        rgb=rgb[0],
        depth_m=depth[0, ..., 0],
        intrinsics=intrinsics[0],
        camera_position_w=camera.pos_w.torch[0].clone(),
        camera_quaternion_ros_xyzw=camera.quat_w_ros.torch[0].clone(),
        sample_monotonic_ns=bundle.clock.monotonic_ns(),
    )


def publish_isaac_rgbd(
    bundle: IsaacHarvestRoboticsBundle,
    output_root: str | Path,
    *,
    capture_id: str,
) -> SpoolPublication:
    """Publish one rendered frame using the same spool contract as a depth camera."""

    from PIL import Image

    sample = capture_isaac_rgbd(bundle)
    rgb_array = sample.rgb.detach().cpu().numpy().astype("uint8")
    depth = sample.depth_m.detach().cpu()
    valid = torch.isfinite(depth) & (depth > 0.0)
    depth_mm = torch.zeros_like(depth, dtype=torch.int32)
    depth_mm[valid] = torch.clamp(
        torch.round(depth[valid] * 1000.0), 1, 65535
    ).to(torch.int32)
    rgb_bytes = BytesIO()
    depth_bytes = BytesIO()
    Image.fromarray(rgb_array, mode="RGB").save(rgb_bytes, format="PNG")
    Image.fromarray(depth_mm.numpy().astype("uint16"), mode="I;16").save(
        depth_bytes, format="PNG"
    )
    k = sample.intrinsics
    qx, qy, qz, qw = (
        float(value) for value in sample.camera_quaternion_ros_xyzw
    )
    frame = RgbdSpoolFrame(
        capture_id=capture_id,
        rgb_png=rgb_bytes.getvalue(),
        aligned_depth_png=depth_bytes.getvalue(),
        depth_scale_m_per_unit=0.001,
        intrinsics=CameraIntrinsics(
            width_px=int(sample.rgb.shape[1]),
            height_px=int(sample.rgb.shape[0]),
            fx_px=float(k[0, 0]),
            fy_px=float(k[1, 1]),
            cx_px=float(k[0, 2]),
            cy_px=float(k[1, 2]),
            calibration_id="isaac_top_rgbd_v1",
        ),
        camera_pose_in_world=Pose(
            tuple(float(value) for value in sample.camera_position_w),
            (qw, qx, qy, qz),
            "world",
        ),
        rgb_sample_monotonic_ns=sample.sample_monotonic_ns,
        depth_sample_monotonic_ns=sample.sample_monotonic_ns,
        device_id="isaac-top-rgbd-01",
        evidence_sources=(
            "isaac:rtx-rgb",
            "isaac:aligned-optical-depth",
            "isaac:fixed-top-camera-pose",
        ),
    )
    return TomatoVisionSpoolWriter(output_root).publish(frame)


class IsaacClearanceScenePort:
    def __init__(
        self,
        *,
        bundle: IsaacHarvestRoboticsBundle,
        pixel_stride: int = 8,
    ) -> None:
        self._bundle = bundle
        self._stride = int(pixel_stride)
        if self._stride < 1:
            raise ValueError("pixel_stride must be >= 1")
        self._closed = False
        self._sequence = 0

    def capture(self, request: PregraspClearanceRequest) -> ClearanceScene:
        if self._closed:
            raise RuntimeError("Isaac clearance port is closed")
        sample = capture_isaac_rgbd(self._bundle)
        depth = sample.depth_m[:: self._stride, :: self._stride]
        # Adjust intrinsics after regular image decimation.
        intrinsics = sample.intrinsics.clone()
        intrinsics[0, :] /= self._stride
        intrinsics[1, :] /= self._stride
        points = create_pointcloud_from_depth(
            intrinsics,
            depth,
            keep_invalid=False,
            position=sample.camera_position_w,
            orientation=sample.camera_quaternion_ros_xyzw,
            device=self._bundle.env.device,
        )
        # A top-mounted depth camera sees both arms. Remove samples belonging
        # to the live robot before treating the remaining cloud as obstacles.
        # This is the simulation equivalent of the robot-model self filter
        # required by the real depth-camera pipeline.
        robot_centers = self._bundle.env._robot.data.body_pos_w.torch[0]
        if points.numel() and robot_centers.numel():
            nearest_robot = torch.cdist(
                points.unsqueeze(0), robot_centers.unsqueeze(0)
            )[0].amin(dim=1)
            points = points[nearest_robot > 0.055]
        finite_depth = torch.isfinite(depth) & (depth > 0.0)
        self._sequence += 1
        return ClearanceScene(
            scene_id=f"isaac-rgbd-clearance-{self._sequence}",
            sample_monotonic_ns=sample.sample_monotonic_ns,
            points_world_m=tuple(
                tuple(float(value) for value in point)
                for point in points.detach().cpu().tolist()
            ),
            coverage_ratio=float(finite_depth.float().mean().item()),
            calibration_id="isaac_top_rgbd_v1",
            simulated=True,
            evidence_sources=(
                "isaac:rtx-depth",
                "isaac:camera-intrinsics",
                "isaac:camera-world-pose",
                "isaac:robot-body-self-filter",
            ),
        )

    def close(self) -> None:
        self._closed = True


class IsaacTargetObservationPort:
    """Call a real detector on a newly rendered frame and return its 3D target."""

    def __init__(
        self,
        *,
        observe_target: Callable[[str, int], TomatoTarget],
    ) -> None:
        self._observe_target = observe_target
        self._closed = False

    def reobserve(self, request: TargetReobserveRequest) -> TargetReobserveEvidence:
        if self._closed:
            raise RuntimeError("Isaac target observation port is closed")
        target = self._observe_target(request.target_id, request.observation.sequence)
        return TargetReobserveEvidence(
            target_id=request.target_id,
            observed=target.target_id == request.target_id,
            occlusion_ratio=target.occlusion,
            depth_available=True,
            evidence_sources=(
                "isaac:rtx-rgb",
                "isaac:aligned-depth",
                "ark:multimodal-box",
                "rgbd:metric-fit",
            ),
            target=target,
            details={"simulation_backend": "isaac_lab_cuda_physx"},
        )

    def close(self) -> None:
        self._closed = True


class IsaacPlacementVerifierPort:
    """Verify release from independent fruit pose and open-hand state."""

    def __init__(self, *, bundle: IsaacHarvestRoboticsBundle) -> None:
        self._bundle = bundle
        self._closed = False

    def verify(
        self,
        receipt: PlacementReceipt,
        payload: HarvestPayload,
        container: HarvestContainer,
    ) -> PlacementVerificationEvidence:
        if self._closed:
            raise RuntimeError("Isaac placement verifier is closed")
        self._bundle.advance(0.35)
        fruit = self._bundle.env._tomato.data.root_pos_w.torch[0]
        cx, cy, bottom_z = self._bundle.env.cfg.container_center_world_m
        size_x, size_y, wall_height = self._bundle.env.cfg.container_inner_size_m
        placed = (
            abs(float(fruit[0]) - cx) <= size_x / 2
            and abs(float(fruit[1]) - cy) <= size_y / 2
            and bottom_z - 0.02 <= float(fruit[2]) <= bottom_z + wall_height + 0.08
        )
        hand = self._bundle.hand.get_state(payload.arm_side)
        hand_empty = hand.synergy.closure <= 0.05 and not hand.fault_codes
        return PlacementVerificationEvidence(
            placement_id=receipt.placement_id,
            container_id=container.container_id,
            placed_in_container=placed,
            release_confirmed=receipt.release_confirmed,
            hand_empty=hand_empty,
            dropped=not placed,
            evidence_sources=(
                "isaac:fruit-rigid-body-pose",
                "isaac:container-collision-volume",
                "isaac:g20-joint-state",
            ),
        )

    def close(self) -> None:
        self._closed = True


class IsaacDroppedPayloadObservationPort:
    """Simulation-truth observer for physical drop/recovery acceptance.

    This adapter is intentionally labeled as simulated. Hardware deployment
    must replace it with fixed-top RGB-D plus detector/depth evidence.
    """

    def __init__(self, *, bundle: IsaacHarvestRoboticsBundle) -> None:
        self._bundle = bundle
        self._closed = False

    def observe(
        self, request: DroppedPayloadLocalizationRequest
    ) -> DroppedPayloadObservationEvidence:
        if self._closed:
            raise RuntimeError("Isaac dropped-payload observation port is closed")
        state = self._bundle.payload_state()
        if not bool(state["dropped"]):
            raise RuntimeError("Isaac payload is not physically in a dropped state")
        fruit = self._bundle.env._tomato.data.root_pos_w.torch[0]
        position = tuple(float(value) for value in fruit)
        cx, cy, bottom_z = self._bundle.env.cfg.container_center_world_m
        size_x, size_y, wall_height = self._bundle.env.cfg.container_inner_size_m
        inside_container = (
            abs(position[0] - cx) <= size_x / 2
            and abs(position[1] - cy) <= size_y / 2
            and bottom_z - 0.02 <= position[2] <= bottom_z + wall_height + 0.08
        )
        return DroppedPayloadObservationEvidence(
            payload_id=request.payload.payload_id,
            observed=True,
            center_pose=Pose(position, (1.0, 0.0, 0.0, 0.0), "world"),
            confidence=1.0,
            inside_container=inside_container,
            sample_monotonic_ns=self._bundle.clock.monotonic_ns(),
            simulated=True,
            evidence_sources=(
                "isaac:fruit-rigid-body-pose",
                "isaac:container-collision-volume",
            ),
        )

    def close(self) -> None:
        self._closed = True


class IsaacCameraHealthPort:
    def __init__(self, bundle: IsaacHarvestRoboticsBundle) -> None:
        self._bundle = bundle

    def snapshot(self) -> CameraHealthSnapshot:
        sample = capture_isaac_rgbd(self._bundle)
        return CameraHealthSnapshot(
            backend_name="isaac_rtx_top_rgbd",
            sdk_available=True,
            device_connected=True,
            rgb_live=sample.rgb.numel() > 0,
            depth_live=bool(torch.isfinite(sample.depth_m).any()),
            simulated=True,
            sample_monotonic_ns=sample.sample_monotonic_ns,
            calibration_id="isaac_top_rgbd_v1",
            device_id="isaac-top-rgbd-01",
            evidence_sources=("isaac:rtx-rgb", "isaac:aligned-depth"),
        )

    def close(self) -> None:
        return None


class IsaacVisionWorkerHealthPort:
    def __init__(
        self,
        bundle: IsaacHarvestRoboticsBundle,
        *,
        ready: Callable[[], bool],
        backend_id: str,
    ) -> None:
        self._bundle = bundle
        self._ready = ready
        self._backend_id = backend_id

    def snapshot(self) -> VisionWorkerHealthSnapshot:
        return VisionWorkerHealthSnapshot(
            worker_id="isaac-acceptance-vision",
            backend_id=self._backend_id,
            ready=bool(self._ready()),
            simulated=True,
            sample_monotonic_ns=self._bundle.clock.monotonic_ns(),
            evidence_sources=("ark:configured", "isaac:render-source"),
        )

    def close(self) -> None:
        return None


class IsaacEmergencyStopHealthPort:
    def __init__(self, bundle: IsaacHarvestRoboticsBundle) -> None:
        self._bundle = bundle

    def snapshot(self) -> EmergencyStopHealthSnapshot:
        return EmergencyStopHealthSnapshot(
            backend_name="isaac_controller_stop",
            available=True,
            healthy=True,
            simulated=True,
            sample_monotonic_ns=self._bundle.clock.monotonic_ns(),
            evidence_sources=("isaac:controller-enabled",),
        )

    def close(self) -> None:
        return None


__all__ = [
    "IsaacCameraHealthPort",
    "IsaacClearanceScenePort",
    "IsaacEmergencyStopHealthPort",
    "IsaacDroppedPayloadObservationPort",
    "IsaacPlacementVerifierPort",
    "IsaacRgbdSample",
    "IsaacTargetObservationPort",
    "IsaacVisionWorkerHealthPort",
    "capture_isaac_rgbd",
    "publish_isaac_rgbd",
]
