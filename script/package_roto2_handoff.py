"""Inventory every saved research artifact and build a split, hashed release."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'research/roto2_handoff/2026-10-26'
ROOTS = ['artifacts', 'logs', 'simulation/.runtime/setup',
         'simulation/.runtime/literature', 'simulation/.runtime/review_upload',
         'handoff_dependencies',
         'botSDK/urdf/02_project_derived/isaac_assets/single_l20_right_rotation']
EXCLUDED = {'__pycache__', '.git', 'uv-cache', 'wheels'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = []
    aliases = []
    for name in ROOTS:
        for base, dirs, names in os.walk(ROOT / name):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
            for directory in dirs:
                link = Path(base) / directory
                if link.is_symlink():
                    assert link.resolve().is_relative_to(ROOT), link
                    aliases.append(dict(path=link.relative_to(ROOT).as_posix(),
                                        target=link.resolve().relative_to(ROOT).as_posix()))
            files.extend(Path(base) / n for n in sorted(names) if not n.endswith(('.pyc', '.pyo')))
    for p in (ROOT / 'simulation/.runtime').iterdir():
        if p.is_file():
            files.append(p)
    files = sorted(set(files))
    rows = []
    secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for i, f in enumerate(files):
        rel = f.relative_to(ROOT).as_posix()
        if f.name == '.env' or f.name.startswith('.env.'):
            raise RuntimeError('Credential-like file requires review: ' + rel)
        link = os.readlink(f) if f.is_symlink() else None
        if link:
            assert f.resolve().is_relative_to(ROOT) and f.is_file(), (rel, link)
        if f.suffix.lower() in ('.txt', '.log', '.json', '.jsonl', '.yaml', '.yml', '.sh', '.py', '.md'):
            with f.open('rb') as stream:
                previous = b''
                for block in iter(lambda: stream.read(1 << 20), b''):
                    assert not secret.search(previous + block), 'Potential credential: ' + rel
                    previous = block[-256:]
        row = dict(path=rel, type='file', bytes=f.stat().st_size, sha256=digest(f))
        if link:
            row['original_symlink_materialized_for_portability'] = link
        rows.append(row)
        if i % 2000 == 0:
            print(f'inventoried {i}/{len(files)}', flush=True)
    inventory = dict(roots=ROOTS, excluded_rebuildable=['Python/Isaac runtime installation', 'downloaded wheels',
        'uv cache', 'Git internals', 'Python bytecode'], files=rows,
        file_count=len(rows), bytes=sum(r.get('bytes', 0) for r in rows))
    (DOC / 'DATA_MANIFEST.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    (DOC / 'DIRECTORY_ALIASES.json').write_text(json.dumps(aliases, indent=2) + '\n')
    weights = [r for r in rows if Path(r['path']).suffix in ('.pt', '.pth', '.ckpt', '.onnx')]
    videos = [r for r in rows if Path(r['path']).suffix.lower() in ('.mp4', '.webm', '.mov', '.avi')]
    (DOC / 'ALL_WEIGHTS.json').write_text(json.dumps(weights, indent=2) + '\n')
    (DOC / 'ALL_VIDEOS.json').write_text(json.dumps(videos, indent=2) + '\n')
    reports = [r['path'] for r in rows if Path(r['path']).suffix.lower() in ('.md', '.pdf', '.html')]
    (DOC / 'ALL_REPORTS.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    listing = output / 'archive-files.nul'
    listing.write_bytes(b''.join(os.fsencode(r['path']) + b'\0' for r in rows))
    archive = output / 'roto2-all-evidence.tar.zst'
    assert not archive.exists(), 'Preserve existing archive'
    env = {**os.environ, 'ZSTD_NBTHREADS': '4'}
    with archive.open('wb') as target:
        tar = subprocess.Popen(['tar', '--dereference', '--null', '--no-recursion', '-cf', '-', '-T', str(listing)], cwd=ROOT, stdout=subprocess.PIPE)
        compressor = subprocess.Popen(['zstd', '-T4', '-3', '-q'], stdin=tar.stdout, stdout=target, env=env)
        tar.stdout.close()
        assert compressor.wait() == 0 and tar.wait() == 0
    parts = []
    with archive.open('rb') as src:
        index = 0
        while True:
            block = src.read(8 << 20)
            if not block:
                break
            name = f'roto2-all-evidence.tar.zst.part{index:03d}'
            part = output / name
            with part.open('wb') as dst:
                remaining = 1_000_000_000
                while block:
                    take = min(len(block), remaining)
                    dst.write(block[:take]); remaining -= take
                    if take < len(block):
                        src.seek(-(len(block) - take), 1)
                    if remaining == 0:
                        break
                    block = src.read(min(8 << 20, remaining))
            parts.append(dict(name=name, bytes=part.stat().st_size, sha256=digest(part)))
            index += 1
    package = dict(repository='hwzhanng/linkagent', branch='research/roto2-object-pressure-track',
        release_tag='roto2-handoff-20261026-all', archive_sha256=digest(archive),
        archive_bytes=archive.stat().st_size, parts=parts,
        inventory_file_count=len(rows), inventory_bytes=inventory['bytes'],
        weight_files=len(weights), video_files=len(videos), report_files=len(reports),
        counts_by_root=dict(Counter(r['path'].split('/')[0] for r in rows)))
    (DOC / 'PACKAGE.json').write_text(json.dumps(package, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in parts))
    print(json.dumps(package, indent=2), flush=True)


if __name__ == '__main__':
    main()
