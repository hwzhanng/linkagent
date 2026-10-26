"""Rebuild the captured Python 3.12 environment without copying a venv."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'research/roto2_handoff/2026-10-26/environment'
SOURCES = [
    ('IsaacLab', 'isaac-sim/IsaacLab', 'ffff603eafc6b74264a5261cc0183d6a65390d78', 'simulation/.runtime/IsaacLab-3.0.0-beta2.patch1'),
    ('multimodal_rl', 'elle-miller/multimodal_rl', '4ac55e854b4c732534d443715a0322d4220b461c', '.external_roto2/multimodal_rl'),
    ('roto', 'elle-miller/roto', '0fa344e6e224f141cb6ab3ed4c2fbbbd86cf1cc2', '.external_roto2/roto'),
]


def run(cmd):
    print(' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print('With --execute: create pinned source checkouts and the isolated env_isaaclab, install exact non-editable packages with --no-deps, then editable sources with --no-deps. System drivers and NVIDIA license acceptance are not automated.')
        return
    for program in ['git', 'uv', 'nvidia-smi', 'ffmpeg', 'zstd']:
        assert shutil.which(program), 'Missing system prerequisite: ' + program
    run(['nvidia-smi', '-L'])
    for name, repo, sha, relative in SOURCES:
        path = ROOT / relative; bundle = ROOT / 'handoff_dependencies' / (name + '.bundle')
        if not (path / '.git').exists():
            assert not path.exists(), 'Non-Git source directory already exists: ' + str(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            run(['git', 'clone', '--no-checkout', bundle if bundle.exists() else f'https://github.com/{repo}.git', path])
            run(['git', '-C', path, 'remote', 'set-url', 'origin', f'https://github.com/{repo}.git'])
        assert not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip(), 'Preserve dirty source checkout: ' + str(path)
        if subprocess.run(['git', '-C', str(path), 'cat-file', '-e', sha + '^{commit}'], stderr=subprocess.DEVNULL).returncode:
            run(['git', '-C', path, 'fetch', '--depth=1', 'origin', sha])
        run(['git', '-C', path, 'checkout', '--detach', sha])
    isaac = ROOT / SOURCES[0][3]
    py = isaac / 'env_isaaclab/bin/python'
    if not py.exists():
        run(['uv', 'venv', '--python', '3.12', '--seed', isaac / 'env_isaaclab'])
    run([py, '-c', 'import sys; assert sys.version_info[:2] == (3,12),sys.version'])
    run(['uv', 'pip', 'install', '--python', py, '--no-deps', '--prerelease=allow',
         '--index-strategy', 'unsafe-best-match', '--index', 'https://pypi.nvidia.com',
         '--index', 'https://download.pytorch.org/whl/cu128', '-r', DOC / 'requirements-exact.txt'])
    snapshot = json.loads((DOC / 'installed.json').read_text())
    for entry in snapshot['editable']:
        relative = entry['source_url'].removeprefix('file:///home/descfly/linkagent-ppo/')
        assert not relative.startswith(('file:', '/'))
        run(['uv', 'pip', 'install', '--python', py, '--no-deps', '-e', ROOT / relative])
    run(['git', '-C', isaac, 'lfs', 'pull'])
    # Exact version verification; known metadata conflicts are recorded separately.
    run([py, '-c', "import importlib.metadata as m,json; from pathlib import Path; d=json.loads(Path('research/roto2_handoff/2026-10-26/environment/installed.json').read_text()); bad=[(r['name'],r['version'],m.version(r['name'])) for r in d['distributions'] if m.version(r['name'])!=r['version']]; assert not bad,bad; print('Exact installed-version verification passed')"])
    # Preserve old forensic wrapper in the evidence archive, create a portable current one.
    wrapper = ROOT / 'simulation/.runtime/run_roto2.sh'
    wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"\nexec bash "$ROOT/script/run_roto2.sh" "$@"\n')
    wrapper.chmod(0o755)
    print('Dependencies installed. Next: bash script/run_roto2.sh smoke. GPU/PhysX smoke remains required on this machine.')


if __name__ == '__main__':
    main()
