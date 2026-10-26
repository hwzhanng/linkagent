"""Download, restore and SHA-256 verify the complete research release."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'research/roto2_handoff/2026-10-26'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['download', 'restore', 'verify'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--archive-dir', type=Path, default=ROOT / 'archive/handoff-20261026')
    args = p.parse_args()
    root = args.root.resolve(); folder = args.archive_dir.resolve()
    package = json.loads((DOC / 'PACKAGE.json').read_text())
    manifest = json.loads((DOC / 'DATA_MANIFEST.json').read_text())
    if args.action == 'download':
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(['gh', 'release', 'download', package['release_tag'], '--repo', package['repository'],
                        '--pattern', 'roto2-all-evidence.tar.zst.part*', '--dir', str(folder), '--skip-existing'], check=True)
    if args.action in ('download', 'restore'):
        for part in package['parts']:
            f = folder / part['name']
            assert f.stat().st_size == part['bytes'] and sha(f) == part['sha256'], f'Bad part: {f}'
        print('All release parts match SHA-256.', flush=True)
        if args.action == 'download':
            return
    if args.action == 'restore':
        # Never replace differing local research data, even during a retry.
        for row in manifest['files']:
            f = root / row['path']
            if f.exists():
                assert f.is_file() and sha(f) == row['sha256'], f'Existing local data differs: {f}'
        root.mkdir(parents=True, exist_ok=True)
        archive = folder / 'roto2-all-evidence.tar.zst'
        if not archive.exists():
            temp = archive.with_suffix('.assembling')
            with temp.open('wb') as dst:
                for part in package['parts']:
                    with (folder / part['name']).open('rb') as src:
                        for chunk in iter(lambda: src.read(8 << 20), b''):
                            dst.write(chunk)
            temp.replace(archive)
        assert sha(archive) == package['archive_sha256'], 'Assembled archive checksum mismatch'
        proc = subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
                tar.extractall(root, filter='data')
            assert proc.wait() == 0
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
        aliases = json.loads((DOC / 'DIRECTORY_ALIASES.json').read_text())
        for row in aliases:
            link = root / row['path']; target = root / row['target']
            if not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
    failures = []
    for row in manifest['files']:
        f = root / row['path']
        if not f.is_file() or f.stat().st_size != row['bytes'] or sha(f) != row['sha256']:
            failures.append(row['path'])
    result = dict(files=len(manifest['files']), failures=failures, verified=not failures,
                  root=str(root), package=package['release_tag'])
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)
    (root / 'HANDOFF_DATA_VERIFIED.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
