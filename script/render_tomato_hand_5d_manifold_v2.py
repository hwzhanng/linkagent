#!/usr/bin/env python3
"""Render a reproducible 20-frame contact sheet of the calibrated G20 v2 path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont


KEYFRAMES = (
    (0, (-1.00, -0.25, 0.00, 1.00, -1.00)),
    (5, (-0.40, -0.10, 0.04, 0.55, -1.00)),
    (10, (0.18, 0.05, 0.10, 0.00, -0.90)),
    (14, (0.68, 0.18, 0.12, -0.12, -0.65)),
    (19, (0.88, 0.22, 0.12, -0.18, 0.92)),
)


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def trajectory() -> list[tuple[float, ...]]:
    result = []
    for frame in range(20):
        for (start_frame, start), (end_frame, end) in zip(
            KEYFRAMES[:-1], KEYFRAMES[1:], strict=True
        ):
            if start_frame <= frame <= end_frame:
                phase = smoothstep((frame - start_frame) / (end_frame - start_frame))
                result.append(
                    tuple(a + phase * (b - a) for a, b in zip(start, end, strict=True))
                )
                break
    if len(result) != 20 or any(not math.isfinite(x) for row in result for x in row):
        raise AssertionError("invalid 20-frame trajectory")
    return result


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/hand_5d_viewer/manifold_v2_20frame"),
    )
    args = parser.parse_args()
    chrome = shutil.which("google-chrome")
    if not chrome:
        raise RuntimeError("google-chrome is required")
    with urlopen(f"{args.server}/api/config.json", timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("hand viewer server is unavailable")

    output = args.output_dir.resolve()
    raw = output / "raw"
    labeled = output / "labeled"
    raw.mkdir(parents=True, exist_ok=True)
    labeled.mkdir(parents=True, exist_ok=True)
    records = []
    for index, values in enumerate(trajectory(), start=1):
        value_text = ",".join(f"{value:.6f}" for value in values)
        query = urlencode(
            {
                "mapping": "v2",
                "capture": "1",
                "camera": "side_right",
                "values": value_text,
                "v": "12",
            }
        )
        source = raw / f"frame_{index:02d}.png"
        destination = labeled / f"frame_{index:02d}.png"
        for attempt in range(1, 4):
            run(
                [
                    chrome,
                    "--headless=new",
                    "--no-sandbox",
                    "--hide-scrollbars",
                    "--enable-unsafe-swiftshader",
                    "--use-angle=swiftshader",
                    "--window-size=600,700",
                    "--virtual-time-budget=20000",
                    f"--screenshot={source}",
                    f"{args.server}/?{query}",
                ]
            )
            # A premature capture contains only the translucent loading mask
            # and is consistently below 100 kB at this fixed resolution.
            if source.stat().st_size >= 100_000:
                break
        else:
            raise RuntimeError(
                f"frame {index} remained on the URDF loading overlay after {attempt} attempts"
            )
        stage = (
            "OPEN" if index <= 5 else
            "PREGRASP" if index <= 10 else
            "WRAP" if index <= 15 else
            "CLOSE"
        )
        label = f"{index:02d}/20 {stage}  q=[{', '.join(f'{x:+.2f}' for x in values)}]"
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
        )
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = max(8, (image.width - text_width) // 2)
        y = image.height - text_height - 14
        draw.rounded_rectangle(
            (x - 8, y - 6, x + text_width + 8, y + text_height + 6),
            radius=6,
            fill=(7, 17, 14, 220),
        )
        draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
        image.save(destination)
        records.append({"frame": index, "stage": stage.lower(), "synergy": values})

    frames = [labeled / f"frame_{index:02d}.png" for index in range(1, 21)]
    contact_sheet = output / "tomato_hand_v2_grasp_process_20frame.png"
    tile_width, tile_height, gap = 300, 350, 8
    sheet = Image.new(
        "RGB",
        (5 * tile_width + 6 * gap, 4 * tile_height + 5 * gap),
        color=(7, 16, 13),
    )
    full_frames = []
    for frame_index, path in enumerate(frames):
        frame = Image.open(path).convert("RGB")
        full_frames.append(frame.copy())
        tile = frame.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        row, column = divmod(frame_index, 5)
        sheet.paste(tile, (gap + column * (tile_width + gap), gap + row * (tile_height + gap)))
    sheet.save(contact_sheet)
    animation = output / "tomato_hand_v2_grasp_process.gif"
    full_frames[0].save(
        animation,
        save_all=True,
        append_images=full_frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    manifest = {
        "schema_version": 1,
        "mapping_id": "tomato_g20_user_calibrated_5d_v2_1_collision_safe",
        "frame_count": 20,
        "render_source": "formal right-hand URDF meshes",
        "hardware_connected": False,
        "contact_sheet": str(contact_sheet),
        "animation": str(animation),
        "trajectory": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
