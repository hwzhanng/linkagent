"""Append an immutable v8 evidence supplement without rewriting the v1-v7 archive."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/roto2_review/2026-10-20/v8"
UPLOAD = ROOT / "simulation/.runtime/review_upload/v8"
BASE = ROOT / "artifacts/simulation/rotation_v8"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def table(name, rows):
    write(OUT / f"tables/{name}.json", rows)
    if rows:
        with (OUT / f"tables/{name}.csv").open("w", newline="") as f:
            fields = sorted({k for row in rows for k in row})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    old = module("old_package", ROOT / "script/package_roto2_review.py")
    n1 = module("n1_summary", ROOT / "script/summarize_roto2_n1.py")
    previous = json.loads((OUT.parent / "manifest.json").read_text())
    previous_map = {x["source_path"]: x for x in previous}
    selected, excluded = old.selected_files()
    added, changed, missing = [], [], []
    for f in selected:
        rel = str(f.relative_to(ROOT))
        if rel not in previous_map:
            added.append(f)
        elif sha(f) != previous_map[rel]["sha256"]:
            changed.append(rel)
    for rel in previous_map:
        if not (ROOT / rel).is_file():
            missing.append(rel)
    assert not changed and not missing, (changed, missing)
    assert all(f.is_relative_to(BASE) for f in added), "Unexpected evidence outside v8"
    scan = old.security_check(added)
    files = sorted(p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    manifest = []
    archive_name = "roto2-20261020-v8-complete-evidence.tar.gz"
    for p in files:
        rel = p.relative_to(ROOT)
        # Full trajectories remain in the complete Release archive; readable
        # JSON, stdout logs, pictures and video are also browsable on the branch.
        mirror = None if p.suffix == ".npz" else Path("evidence") / rel
        if mirror:
            target = OUT / mirror
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        manifest.append(dict(source_path=str(rel), bytes=p.stat().st_size, sha256=sha(p),
                             review_path=str(mirror) if mirror else None,
                             archive=archive_name, archive_member=str(rel)))
    sources = sorted(set(
        list((ROOT / "simulation/roto2_pressure").glob("*.py"))
        + [ROOT / p for p in [
            "script/train_roto2_pressure.py", "script/search_roto2_palm_grasps.py",
            "script/summarize_roto2_n1.py", "script/package_roto2_v8_review.py"]]))
    old.security_check(sources)
    source_manifest = []
    for p in sources:
        target = OUT / "source_snapshot" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        source_manifest.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
    write(OUT / "source_manifest.json", {
        "meaning": "Current handoff source, not a claim that every earlier probe used this exact source",
        "files": source_manifest})
    diff = subprocess.check_output(["git", "diff", "b4b556d", "--", "script", "simulation"], cwd=ROOT)
    (OUT / "tracked_changes_from_b4b556d.patch").write_bytes(diff)
    searches, validations, trials = [], [], []
    for p in sorted((BASE / "stage0").glob("*status.json")):
        d = json.loads(p.read_text())
        searches.append({"source": str(p.relative_to(ROOT)), **d})
    for p in sorted((BASE / "stage0").glob("*.json")):
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "validation_completed" in d:
            s = n1.summarize(p)
            validations.append({k: v for k, v in s.items() if k != "per_condition"})
            for row in d["results"]:
                for trial in row["trials"]:
                    trials.append({"source": str(p.relative_to(ROOT)), "seed": d.get("seed"),
                                   "gate_version": d.get("gate_version", "legacy"),
                                   "candidate_id": row["candidate_id"],
                                   "source_grasp_id": row.get("source_grasp_id", row["candidate_id"]),
                                   "baseline_ok": row["baseline_ok"],
                                   "n_minus_one_passed": row["n_minus_one_passed"], **trial})
    table("searches", searches)
    table("n1_validations", validations)
    table("n1_trials", trials)
    schemas = []
    for p in files:
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as a:
                fields = {k: {"shape": list(a[k].shape), "dtype": str(a[k].dtype)} for k in a.files}
                schemas.append({"source": str(p.relative_to(ROOT)), "arrays": fields})
    write(OUT / "npz_schema.json", schemas)
    checks = {}
    for path, names in {
        "script/train_roto2_pressure.py": ["build_roto2_training"],
        "simulation/roto2_pressure/env.py": ["_pre_physics_step", "_apply_action", "_get_dones", "_get_observations"],
    }.items():
        def definitions(text):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        before = definitions(subprocess.check_output(["git", "show", f"b4b556d:{path}"], cwd=ROOT).decode())
        after = definitions((ROOT / path).read_text())
        for name in names:
            checks[name] = before[name] == after[name]
    assert all(checks.values()), checks
    tests = []
    import sys
    for name in ["test_redundant_baseline", "test_continuous_reward", "test_reward_geometry", "test_rotation_curriculum"]:
        result = subprocess.run([sys.executable, "-m", f"simulation.roto2_pressure.{name}"],
                                cwd=ROOT, capture_output=True, text=True)
        tests.append(dict(module=name, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, tests[-1]
    archive = UPLOAD / archive_name
    with tarfile.open(archive, "w:gz", compresslevel=6) as t:
        for p in files:
            t.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        for p in sources:
            t.add(p, arcname="handoff_source/" + str(p.relative_to(ROOT)), recursive=False)
    # Validate every archive member against its exact source bytes.
    expected = {str(p.relative_to(ROOT)): sha(p) for p in files}
    expected.update({"handoff_source/" + str(p.relative_to(ROOT)): sha(p) for p in sources})
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
        "verified_members": len(expected),
        "release_tag": "roto2-review-20261020-v8"}])
    write(OUT / "VALIDATION.json", {
        "utc": datetime.now(timezone.utc).isoformat(),
        "previous_evidence_files_verified_unchanged": len(previous),
        "new_evidence_files": len(files),
        "evidence_bytes": sum(p.stat().st_size for p in files),
        "branch_mirrored_files": sum(r["review_path"] is not None for r in manifest),
        "npz_files_read_and_archived": len(schemas),
        "search_runs": len(searches), "search_physical_rollouts": sum(r["evaluated_proposals"] for r in searches),
        "n1_validation_runs": len(validations),
        "n1_configuration_tests": sum(r["candidates"] for r in validations),
        "n1_condition_rows": len(trials),
        "n1_accepted_configurations": sum(r["accepted"] for r in validations),
        "core_ast_unchanged": checks, "tests": tests,
        "credential_scan": scan,
        "all_archive_members_hash_verified": True,
        "data_boundary": "No new PPO training. Probe trials are not independent policies or PPO transitions.",
        "excluded": excluded})
    print(json.dumps({"archive": str(archive), "sha256": digest, "files": len(files),
                      "physical_search_rollouts": sum(r["evaluated_proposals"] for r in searches),
                      "n1_configs": sum(r["candidates"] for r in validations)}, indent=2))


if __name__ == "__main__":
    main()
