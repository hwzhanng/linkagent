"""Freeze completed physical calibration evidence; exclude changing PPO logs."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/simulation/rotation_v8/stage0"
OUT = ROOT / "research/roto2_review/2026-10-20/v8-stage0-passed"
UPLOAD = ROOT / "simulation/.runtime/review_upload/v8-stage0-passed"
TAG = "roto2-review-20261020-v8-stage0-passed"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    old = module("previous_packager", ROOT / "script/package_roto2_review.py")
    n1 = module("n1", ROOT / "script/summarize_roto2_n1.py")
    files = sorted(p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    sources = sorted(set(
        list((ROOT / "simulation/roto2_pressure").glob("*.py"))
        + [ROOT / p for p in [
            "simulation/roto2_pressure/reset_grasps_v8_validated256.json",
            "script/train_roto2_pressure.py",
            "script/search_roto2_palm_grasps.py",
            "script/expand_roto2_grasp_neighborhood.py",
            "script/freeze_roto2_reset_library.py",
            "script/summarize_roto2_n1.py",
            "script/package_roto2_stage0_milestone.py"]]))
    scan = old.security_check(files + sources)
    archive_name = TAG + "-evidence.tar.gz"
    manifest, schemas, n1_runs, hold_runs, searches = [], [], [], [], []
    for p in files:
        rel = str(p.relative_to(ROOT))
        mirror = Path("evidence") / rel if p.suffix != ".npz" else None
        if mirror:
            target = OUT / mirror
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        manifest.append(dict(source_path=rel, bytes=p.stat().st_size, sha256=sha(p),
                             review_path=str(mirror) if mirror else None,
                             archive=archive_name, archive_member=rel))
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as a:
                schemas.append(dict(source=rel, arrays={
                    k: dict(shape=list(a[k].shape), dtype=str(a[k].dtype)) for k in a.files}))
        if p.suffix == ".json" and p.parent == BASE:
            d = json.loads(p.read_text())
            if isinstance(d, dict) and d.get("results") and "trials" in d["results"][0]:
                s = n1.summarize(p)
                s["source"] = rel
                n1_runs.append(s)
            elif isinstance(d, dict) and d.get("no_automatic_resets"):
                hold_runs.append(dict(source=rel, seed=d["seed"], duration_seconds=d["duration_seconds"],
                                      count=len(d["results"]),
                                      stable_no_failure=sum(r["stable_after_settling"] and not r["physical_failure"]
                                                            for r in d["results"]),
                                      four_pad_duty_80pct_and_stable=d["passed_count"],
                                      max_displacement_m=max(r["max_displacement_m"] for r in d["results"]),
                                      max_pressure_pa=max(r["peak_pressure_pa"] for r in d["results"])))
            elif isinstance(d, dict) and "search_completed" in d:
                searches.append(dict(source=rel, **d))
    source_manifest = []
    for p in sources:
        dest = OUT / "source_snapshot" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        source_manifest.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
    for name, rows in [("n1_runs", n1_runs), ("long_holds", hold_runs), ("searches", searches)]:
        write(OUT / "tables" / (name + ".json"), rows)
        flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
        with (OUT / "tables" / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, sorted({k for r in flat for k in r}), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flat)
    checks = {}
    for path, names in {
        "script/train_roto2_pressure.py": ["build_roto2_training"],
        "simulation/roto2_pressure/env.py": ["_pre_physics_step", "_apply_action", "_get_dones", "_get_observations"],
    }.items():
        def definitions(text):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}
        before = definitions(subprocess.check_output(["git", "show", f"71c1568:{path}"], cwd=ROOT).decode())
        after = definitions((ROOT / path).read_text())
        for name in names:
            checks[name] = before[name] == after[name]
    assert all(checks.values()), checks
    expected = {str(p.relative_to(ROOT)): sha(p) for p in files}
    expected.update({"handoff_source/" + str(p.relative_to(ROOT)): sha(p) for p in sources})
    archive = UPLOAD / archive_name
    with tarfile.open(archive, "w:gz", compresslevel=6) as t:
        for p in files:
            t.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        for p in sources:
            t.add(p, arcname="handoff_source/" + str(p.relative_to(ROOT)), recursive=False)
    with tarfile.open(archive) as t:
        members = [m for m in t.getmembers() if m.isfile()]
        assert {m.name for m in members} == set(expected)
        for m in members:
            assert hashlib.sha256(t.extractfile(m).read()).hexdigest() == expected[m.name], m.name
    digest = sha(archive)
    (UPLOAD / "SHA256SUMS").write_text(f"{digest}  {archive_name}\n")
    write(OUT / "manifest.json", manifest)
    write(OUT / "source_manifest.json", {
        "meaning": "Source at handoff; historical probes use their recorded source snapshots where available.",
        "code_commit_before_packaging": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": source_manifest})
    write(OUT / "npz_schema.json", schemas)
    write(OUT / "release_assets.json", [{
        "name": archive_name, "bytes": archive.stat().st_size, "sha256": digest,
        "verified_members": len(expected), "release_tag": TAG}])
    write(OUT / "VALIDATION.json", {
        "scope": "Complete stage0 evidence through seed2118, including historical failed probes; no changing stage1 PPO logs.",
        "evidence_files": len(files), "npz_files": len(schemas),
        "branch_mirrors": sum(r["review_path"] is not None for r in manifest),
        "n1_runs": len(n1_runs), "long_hold_runs": len(hold_runs),
        "core_ast_unchanged": checks, "credential_scan": scan,
        "archive_members_sha256_verified": len(expected),
        "stage0_library_passed": True, "sustained_rotation_passed": False})
    print(json.dumps(dict(files=len(files), archive=str(archive), bytes=archive.stat().st_size,
                          sha256=digest, n1_runs=len(n1_runs), long_holds=len(hold_runs)), indent=2))


if __name__ == "__main__":
    main()
