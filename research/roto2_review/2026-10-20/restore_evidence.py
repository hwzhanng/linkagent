"""Verify saved evidence; optionally restore original paths into a checkout.

Python standard library only. Never overwrites a different existing file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

BASE = Path(__file__).resolve().parent


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def check(path, row):
    with path.open("rb") as stream:
        assert path.stat().st_size == row["bytes"] and digest(stream) == row["sha256"], path


def destination(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe path: {relative}")
    target = root / p
    # Disallow symlinks, including dangling links and existing parent links.
    if any(item.is_symlink() for item in [target, *target.parents]):
        raise ValueError(f"Symlink destination: {target}")
    return target


def restore(stream, target, row):
    if target.exists():
        check(target, row)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".review-restore-part")
    with temporary.open("xb") as out:
        shutil.copyfileobj(stream, out)
    check(temporary, row)
    temporary.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--restore-to", type=Path)
    args = parser.parse_args()
    rows = json.loads((BASE / "manifest.json").read_text())
    root = args.restore_to.absolute() if args.restore_to else None
    if root and not args.archive:
        parser.error("--restore-to requires --archive so model assets are not silently omitted")
    verified = 0
    for row in rows:
        if row["storage"] != "git_branch":
            continue
        source = destination(BASE, row["review_path"])
        check(source, row)
        if root:
            with source.open("rb") as stream:
                restore(stream, destination(root, row["source_path"]), row)
        verified += 1
    if args.archive:
        spec = json.loads((BASE / "release_assets.json").read_text())[0]
        check(args.archive, spec)
        expected = {r["archive_member"]: r for r in rows if r["storage"] == "github_release_archive"}
        seen = set()
        with tarfile.open(args.archive, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                row = expected[member.name]
                with archive.extractfile(member) as stream:
                    if root:
                        restore(stream, destination(root, member.name), row)
                    else:
                        assert member.size == row["bytes"] and digest(stream) == row["sha256"], member.name
                seen.add(member.name)
                verified += 1
        assert seen == set(expected), "Missing archive members"
    print(json.dumps({"verified_files": verified, "archive_verified": bool(args.archive),
                      "restored_to": str(root) if root else None}))


if __name__ == "__main__":
    main()
