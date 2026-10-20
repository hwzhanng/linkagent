"""Freeze completed stage1 training/diagnostics, including interrupted lineage."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/simulation/rotation_v8/stage1"
OUT = ROOT / "research/roto2_review/2026-10-20/v8-stage1-300m"
UPLOAD = ROOT / "simulation/.runtime/review_upload/v8-stage1-300m"
TAG = "roto2-review-20261020-v8-stage1-300m"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    import numpy as np
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    p = argparse.ArgumentParser()
    p.add_argument("--final-run", type=Path, required=True)
    a = p.parse_args()
    run = a.final_run.resolve()
    result = json.loads((run / "training_result.json").read_text())
    assert result["training_completed"] and result["finite_parameters"]
    assert result["cumulative_training_transitions"] >= 299_000_000
    pipeline = json.loads((BASE / "final_validation/VALIDATION_PIPELINE.json").read_text())
    assert pipeline["status"] == "completed_pending_visual_review"
    packaged_at = datetime.now(timezone.utc)
    # Include diagnostics explicitly run after the automatic video pipeline.
    cutoff = packaged_at.astimezone(
        ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
    gate = json.loads((BASE / "final_validation/QUANTITATIVE_GATE.json").read_text())
    assert gate["visual_review_pending"] is False, "Actual visual review must precede delivery"
    spec = importlib.util.spec_from_file_location("previous_packager", ROOT / "script/package_roto2_review.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    OUT.mkdir(parents=True, exist_ok=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    # Exact known stage1 cohort. Other project jobs and their data are excluded.
    roots = [BASE]
    log_root = ROOT / "logs/roto2_pressure_fruit"
    run_dirs = []
    for seed in (49, 5101, 6101, 6102):
        for directory in sorted((log_root / f"plain_seed{seed}").glob("*")):
            if directory.is_dir() and "2026-10-20_20-53-18" <= directory.name <= cutoff:
                run_dirs.append(directory)
                roots.append(directory)
    assert run in run_dirs
    files = sorted({f for root in roots for f in root.rglob("*")
                    if f.is_file() and "__pycache__" not in f.parts})
    assert not any(f.name.endswith(".tmp") for f in files), "Unfinished checkpoint copies remain"
    sources = sorted(set(
        list((ROOT / "simulation/roto2_pressure").glob("*.py"))
        + [ROOT / f for f in [
            "simulation/roto2_pressure/agent.yaml",
            "simulation/roto2_pressure/reset_grasps_v8_validated256.json",
            "script/train_roto2_pressure.py", "script/validate_roto2_stage1_run.py",
            "script/snapshot_roto2_milestones.py", "script/plot_roto2_policy_diagnostic.py",
            "script/package_roto2_stage1_review.py",
            "script/probe_roto2_shifted_grasp_ik.py",
            "simulation/.runtime/setup/search_grasp.py"]]))
    scan = old.security_check(files + sources)
    archive_name = TAG + "-evidence.tar.gz"
    manifest, schemas, checkpoints, scalar_rows, training_rows = [], [], [], [], []
    unique_checkpoint_hashes = {}
    for f in files:
        digest = sha(f)
        rel = str(f.relative_to(ROOT))
        binary_only = f.suffix in (".npz", ".pt") or f.name.startswith("events.out")
        mirror = None if binary_only else Path("evidence") / rel
        if mirror:
            target = OUT / mirror
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        manifest.append(dict(source_path=rel, bytes=f.stat().st_size, sha256=digest,
                             review_path=str(mirror) if mirror else None,
                             archive=archive_name, archive_member=rel))
        if f.suffix == ".npz":
            with np.load(f, allow_pickle=False) as x:
                schemas.append(dict(source=rel, arrays={
                    k: {"shape": list(x[k].shape), "dtype": str(x[k].dtype)} for k in x.files}))
        if f.suffix == ".pt":
            if digest not in unique_checkpoint_hashes:
                x = torch.load(f, map_location="cpu", weights_only=True)
                assert all(k in x for k in ("encoder", "policy", "value")), rel

                def tensors(v):
                    if isinstance(v, torch.Tensor):
                        yield v
                    elif isinstance(v, dict):
                        for item in v.values():
                            yield from tensors(item)
                    elif isinstance(v, (tuple, list)):
                        for item in v:
                            yield from tensors(item)

                values = list(tensors(x))
                assert all(torch.isfinite(v).all().item() for v in values), rel
                unique_checkpoint_hashes[digest] = dict(
                    modules=list(x), tensor_count=len(values), all_tensors_finite=True)
            checkpoints.append(dict(source=rel, sha256=digest, **unique_checkpoint_hashes[digest]))
    for directory in run_dirs:
        cfg_path = directory / "run_config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        finish = directory / "training_result.json"
        progress = directory / "progress.json"
        metadata = json.loads((finish if finish.exists() else progress).read_text()) if finish.exists() or progress.exists() else {}
        training_rows.append(dict(run=str(directory.relative_to(ROOT)),
                                  seed=cfg["seed"], resumed_from=cfg["resumed_from"],
                                  status="completed_segment" if metadata.get("training_completed") else
                                  "partial_saved_training" if progress.exists() else "evaluation_or_diagnostic",
                                  **metadata))
        if list(directory.glob("events.out*")):
            e = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
            for tag in e.Tags()["scalars"]:
                scalar_rows.extend(dict(run=str(directory.relative_to(ROOT)), tag=tag,
                                        step=x.step, wall_time=x.wall_time, value=x.value)
                                   for x in e.Scalars(tag))
    write(OUT / "training_segments.json", training_rows)
    write(OUT / "checkpoints.json", checkpoints)
    write(OUT / "npz_schema.json", schemas)
    with (OUT / "tensorboard_scalars.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["run", "tag", "step", "wall_time", "value"], lineterminator="\n")
        w.writeheader()
        w.writerows(scalar_rows)
    source_manifest = []
    for source in sources:
        target = OUT / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_manifest.append(dict(path=str(source.relative_to(ROOT)), sha256=sha(source)))
    write(OUT / "source_manifest.json", {
        "meaning": "Handoff source; each training/diagnostic run retains its own launch snapshot.",
        "files": source_manifest})
    expected = {r["source_path"]: r["sha256"] for r in manifest}
    expected.update({"handoff_source/" + str(s.relative_to(ROOT)): sha(s) for s in sources})
    archive = UPLOAD / archive_name
    with tarfile.open(archive, "w:gz", compresslevel=6) as t:
        for f in files:
            t.add(f, arcname=str(f.relative_to(ROOT)), recursive=False)
        for f in sources:
            t.add(f, arcname="handoff_source/" + str(f.relative_to(ROOT)), recursive=False)
    with tarfile.open(archive) as t:
        members = [m for m in t.getmembers() if m.isfile()]
        assert {m.name for m in members} == set(expected)
        for m in members:
            assert hashlib.sha256(t.extractfile(m).read()).hexdigest() == expected[m.name], m.name
    digest = sha(archive)
    (UPLOAD / "SHA256SUMS").write_text(f"{digest}  {archive_name}\n")
    write(OUT / "manifest.json", manifest)
    write(OUT / "release_assets.json", [{
        "name": archive_name, "bytes": archive.stat().st_size, "sha256": digest,
        "release_tag": TAG, "verified_members": len(expected)}])
    write(OUT / "VALIDATION.json", {
        "evidence_files": len(files), "run_directories": len(run_dirs),
        "npz_files": len(schemas), "checkpoint_files": len(checkpoints),
        "unique_checkpoints_checked_finite": len(unique_checkpoint_hashes),
        "tensorboard_scalar_rows": len(scalar_rows), "credential_scan": scan,
        "training_completed": True, "stage1_task_accepted": gate["stage1_accepted"],
        "cumulative_samples": result["cumulative_training_transitions"],
        "cumulative_ppo_updates": result["cumulative_ppo_updates"],
        "packaged_at_utc": packaged_at.isoformat(),
        "full_PPO_rollout_dataset_saved": False,
        "note": "On-policy rollout buffers are reused. This archive contains saved evidence, not an invented300M-row offline dataset.",
    })
    print(json.dumps({"files": len(files), "archive": str(archive),
                      "bytes": archive.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
