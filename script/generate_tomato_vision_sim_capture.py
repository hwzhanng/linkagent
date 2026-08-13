"""Generate one deterministic synthetic tomato RGB-D spool capture.

The artifact is only a software-in-the-loop perception fixture.  It is not a
camera calibration, a photorealistic simulator, or evidence for real hardware.
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
from pathlib import Path
import struct
import time
from typing import Sequence
import zlib

from proto import CameraIntrinsics, Pose
from script.tomato_vision_spool import (
    RgbdSpoolFrame,
    SpoolPublication,
    TomatoVisionSpoolWriter,
)


WIDTH = 640
HEIGHT = 480
FRUIT_CENTER_PX = (395, 240)
FRUIT_RADIUS_PX = 29
FRUIT_FRONT_DEPTH_M = 0.760
FRUIT_EDGE_DEPTH_M = 0.820


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum)
    )


def _png(width: int, height: int, *, colour_type: int, rows: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8 if colour_type == 2 else 16, colour_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(rows, level=9))
        + _chunk(b"IEND", b"")
    )


def _synthetic_images() -> tuple[bytes, bytes]:
    center_x, center_y = FRUIT_CENTER_PX
    rgb_rows = bytearray()
    depth_rows = bytearray()
    for y in range(HEIGHT):
        rgb_rows.append(0)
        depth_rows.append(0)
        for x in range(WIDTH):
            # A dark greenhouse-like background with sparse leaf shapes keeps
            # the fixture visually meaningful without hiding the fruit.
            green_texture = (x * 7 + y * 11) % 9
            rgb = (15, 34 + green_texture, 22)
            depth_units = 0
            leaf_left = ((x - 315) / 58.0) ** 2 + ((y - 226) / 15.0) ** 2 <= 1.0
            leaf_right = ((x - 480) / 54.0) ** 2 + ((y - 285) / 14.0) ** 2 <= 1.0
            if leaf_left or leaf_right:
                rgb = (25, 82 + green_texture, 35)

            dx_px = x - center_x
            dy_px = y - center_y
            radius_sq = dx_px * dx_px + dy_px * dy_px
            if radius_sq <= FRUIT_RADIUS_PX * FRUIT_RADIUS_PX:
                radial = math.sqrt(radius_sq) / FRUIT_RADIUS_PX
                highlight = max(
                    0.0,
                    1.0
                    - math.sqrt((x - (center_x - 9)) ** 2 + (y - (center_y - 10)) ** 2)
                    / 17.0,
                )
                rgb = (
                    min(255, int(205 - 70 * radial + 45 * highlight)),
                    min(95, int(28 + 22 * highlight)),
                    min(70, int(22 + 18 * highlight)),
                )
                normalized_radius_sq = radius_sq / (
                    FRUIT_RADIUS_PX * FRUIT_RADIUS_PX
                )
                depth_units = int(
                    round(
                        (
                            FRUIT_FRONT_DEPTH_M
                            + (FRUIT_EDGE_DEPTH_M - FRUIT_FRONT_DEPTH_M)
                            * normalized_radius_sq
                        )
                        * 1000.0
                    )
                )

            # The visible stalk meets the fruit inside the fruit box.  Its
            # background section has depth too, but the attachment remains on
            # the calibrated fruit surface.
            if abs(x - center_x) <= 3 and 190 <= y <= 214:
                rgb = (32, 128, 46)
                if depth_units == 0:
                    depth_units = 805

            rgb_rows.extend(rgb)
            depth_rows.extend(struct.pack(">H", depth_units))
    return (
        _png(WIDTH, HEIGHT, colour_type=2, rows=bytes(rgb_rows)),
        _png(WIDTH, HEIGHT, colour_type=0, rows=bytes(depth_rows)),
    )


def generate_capture(
    output_root: str | Path,
    *,
    capture_id: str = "ark-synthetic-tomato-001",
    sample_monotonic_ns: int | None = None,
) -> SpoolPublication:
    root = Path(output_root).expanduser().resolve()
    sample_ns = time.monotonic_ns() if sample_monotonic_ns is None else sample_monotonic_ns
    rgb_png, depth_png = _synthetic_images()
    frame = RgbdSpoolFrame(
        capture_id=capture_id,
        rgb_png=rgb_png,
        aligned_depth_png=depth_png,
        depth_scale_m_per_unit=0.001,
        intrinsics=CameraIntrinsics(
            width_px=WIDTH,
            height_px=HEIGHT,
            fx_px=600.0,
            fy_px=600.0,
            cx_px=320.0,
            cy_px=240.0,
            calibration_id="synthetic-orbbec-aligned-rgbd-v1",
        ),
        camera_pose_in_world=Pose(
            position_m=(0.35, -0.25, -0.08),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            frame_id="world",
        ),
        rgb_sample_monotonic_ns=sample_ns,
        depth_sample_monotonic_ns=sample_ns + 1_000_000,
        device_id="synthetic-rgbd-camera-001",
        evidence_sources=(
            "simulation:procedural-rgb",
            "simulation:analytic-aligned-depth",
        ),
    )
    return TomatoVisionSpoolWriter(root).publish(frame)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic tomato RGB-D capture")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/vision_spool/ark-synthetic"),
    )
    parser.add_argument("--capture-id", default="ark-synthetic-tomato-001")
    parser.add_argument("--sample-monotonic-ns", type=int, default=None)
    args = parser.parse_args(argv)
    publication = generate_capture(
        args.output_root,
        capture_id=args.capture_id,
        sample_monotonic_ns=args.sample_monotonic_ns,
    )
    print(
        json.dumps(
            {
                "capture_bundle": publication.pointer.capture_bundle,
                "sample_monotonic_ns": publication.pointer.sample_monotonic_ns,
                "rgb_sha256": publication.rgb_sha256,
                "aligned_depth_sha256": publication.aligned_depth_sha256,
                "simulation_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_capture", "main"]
