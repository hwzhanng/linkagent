#!/usr/bin/env python3
"""Build a traceable Isaac USD variant with compensated L20 hand masses."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pxr import Usd


DERIVED_ROOT = Path(__file__).resolve().parents[3] / "urdf/02_project_derived"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = DERIVED_ROOT / "isaac_assets/linkbot_bimanual_sdk_description"
DEFAULT_OUTPUT = DERIVED_ROOT / "isaac_assets/linkbot_bimanual_sdk_training_mass1kg"
SOURCE_USD_NAME = "linkbot_bimanual_sdk_description.usda"
OUTPUT_USD_NAME = "linkbot_bimanual_sdk_training_mass1kg.usda"
PHYSICS_LAYERS = (
    Path("payloads/Physics/physics.usda"),
    Path("payloads/Physics/physx.usda"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer repository-relative provenance for canonical in-repo assets."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def find_mass_prim(stage: Usd.Stage, name: str) -> Usd.Prim:
    matches = [
        prim
        for prim in stage.TraverseAll()
        if prim.GetName() == name
        and prim.GetAttribute("physics:mass").IsValid()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one mass prim named {name}, found {len(matches)}"
        )
    return matches[0]


def hand_mass(stage: Usd.Stage, side: str) -> float:
    prefix = f"{side}_hand_"
    return sum(
        float(prim.GetAttribute("physics:mass").Get())
        for prim in stage.TraverseAll()
        if prim.GetName().startswith(prefix)
        and prim.GetAttribute("physics:mass").IsValid()
    )


def calculate_corrections(
    physics_stage: Usd.Stage, target_hand_mass_kg: float
) -> dict[str, dict[str, object]]:
    corrections: dict[str, dict[str, object]] = {}
    for side in ("left", "right"):
        total_before = hand_mass(physics_stage, side)
        if total_before <= 0.0:
            continue
        base = find_mass_prim(physics_stage, f"{side}_hand_base_link")
        base_mass_before = float(base.GetAttribute("physics:mass").Get())
        missing_mass = target_hand_mass_kg - total_before
        if missing_mass <= 0.0:
            raise RuntimeError(
                f"{side} hand already weighs {total_before:.9f} kg, "
                f"not below target {target_hand_mass_kg:.9f} kg"
            )
        base_mass_after = base_mass_before + missing_mass
        inertia_scale = base_mass_after / base_mass_before
        corrections[side] = {
            "total_mass_before_kg": total_before,
            "target_total_mass_kg": target_hand_mass_kg,
            "missing_mass_added_to_palm_kg": missing_mass,
            "base_mass_before_kg": base_mass_before,
            "base_mass_after_kg": base_mass_after,
            "base_inertia_scale": inertia_scale,
        }
    return corrections


def apply_corrections(
    stage: Usd.Stage, corrections: dict[str, dict[str, object]]
) -> None:
    for side, values in corrections.items():
        base = find_mass_prim(stage, f"{side}_hand_base_link")
        mass_attribute = base.GetAttribute("physics:mass")
        inertia_attribute = base.GetAttribute("physics:diagonalInertia")
        if not inertia_attribute.IsValid():
            raise RuntimeError(
                f"Missing diagonal inertia on {side}_hand_base_link"
            )
        diagonal_before = inertia_attribute.Get()
        scale = float(values["base_inertia_scale"])
        mass_attribute.Set(float(values["base_mass_after_kg"]))
        inertia_attribute.Set(
            type(diagonal_before)(
                float(diagonal_before[0]) * scale,
                float(diagonal_before[1]) * scale,
                float(diagonal_before[2]) * scale,
            )
        )
    stage.GetRootLayer().Save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-hand-mass-kg", type=float, default=1.0)
    parser.add_argument("--source-usd-name", default=SOURCE_USD_NAME)
    parser.add_argument("--output-usd-name", default=OUTPUT_USD_NAME)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_usd = source / args.source_usd_name
    if not source_usd.is_file():
        raise FileNotFoundError(source_usd)
    if output.exists():
        raise FileExistsError(
            f"Output already exists; archive it before rebuilding: {output}"
        )

    shutil.copytree(source, output)
    copied_source_usd = output / args.source_usd_name
    output_usd = output / args.output_usd_name
    copied_source_usd.rename(output_usd)

    physics_stage = Usd.Stage.Open(str(output / PHYSICS_LAYERS[0]))
    if physics_stage is None:
        raise RuntimeError("Unable to open copied physics layer")
    corrections = calculate_corrections(
        physics_stage, args.target_hand_mass_kg
    )
    for relative_layer in PHYSICS_LAYERS:
        stage = Usd.Stage.Open(str(output / relative_layer))
        if stage is None:
            raise RuntimeError(f"Unable to open {relative_layer}")
        apply_corrections(stage, corrections)

    verification_stage = Usd.Stage.Open(str(output / PHYSICS_LAYERS[0]))
    if verification_stage is None:
        raise RuntimeError("Unable to reopen corrected physics layer")
    totals_after = {
        side: hand_mass(verification_stage, side)
        for side in corrections
    }
    for side, total in totals_after.items():
        if abs(total - args.target_hand_mass_kg) > 1e-6:
            raise RuntimeError(
                f"{side} corrected mass is {total}, expected "
                f"{args.target_hand_mass_kg}"
            )

    original_manifest = output / "linkbot_asset_manifest.json"
    if original_manifest.exists():
        original_manifest.rename(output / "source_linkbot_asset_manifest.json")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_role": "training_mass_baseline_not_measured_inertia",
        "source_asset": portable_path(source_usd),
        "source_asset_sha256": sha256(source_usd),
        "generated_asset": portable_path(output_usd),
        "target_hand_mass_kg": args.target_hand_mass_kg,
        "method": (
            "Keep all vendor finger-link masses unchanged. Add the published "
            "whole-hand mass gap to hand_base_link and scale that palm link's "
            "principal inertia in direct proportion to its mass. Preserve the "
            "vendor palm center of mass and principal axes."
        ),
        "rationale": (
            "The missing motors, electronics, housing and harness mass is "
            "expected to be concentrated in the palm/wrist rather than the "
            "fingertips."
        ),
        "corrections": corrections,
        "verified_total_mass_after_kg": totals_after,
        "limitations": [
            "The 1 kg whole-hand value is published, not measured on this unit.",
            "Palm center of mass and inertia distribution remain estimates.",
            "Joint damping, friction, torque and drive gains remain provisional.",
        ],
    }
    (output / "mass_correction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# LinkBot 双臂双手 Isaac 训练质量基线\n\n"
        "本目录由官方/项目基准 USD 派生。左右 L20 整手总质量均补偿为 "
        f"`{args.target_hand_mass_kg:.3f} kg`。\n\n"
        "补偿只作用于左右 `hand_base_link`：官方指节质量保持不变，"
        "缺失质量加入掌部，掌部主惯量按质量同比缩放，质心和主轴保持不变。"
        "这是缺少实物质心/转动惯量测量时的训练起始估算，不是最终 Sim2Real "
        "辨识结果。\n\n"
        f"入口资产：`{args.output_usd_name}`\n\n"
        "详细数值和限制见 `mass_correction_manifest.json`。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"TRAINING_USD={output_usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
