"""单 GPU 锁、只读安装环境、外置缓存的接触预检查启动器。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='probe_contacts_01')
    parser.add_argument('--fixture-source-surface', choices=('none', 'thumb_pad', 'thumb_side'), default='none')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--fixture-center-distance-m', type=float, default=.0334)
    parser.add_argument('--fixture-pad-target-m',nargs=3,type=float,default=None)
    parser.add_argument('--fixture-digit', choices=('thumb','index','middle','ring','little'), default='thumb')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--module', choices=('probe_contacts','smoke_new_task','run','record_video'), default='probe_contacts')
    parser.add_argument('--task', choices=('base','thumb_pad'), default='base')
    parser.add_argument('entry_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name.replace('_','').isalnum():
        parser.error('name 仅允许字母、数字、下划线')
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).resolve().parents[2]
    cache = root / 'cache'
    (cache/'var-tmp').mkdir(parents=True,exist_ok=True)
    for name in ('xdg', 'cuda', 'torch', 'hf', 'gl', 'omni-user', 'kit', 'pycache'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    short = Path('/tmp/linkbot-ppo-thumb-pad-v1-20261016')
    if short.is_symlink():
        assert short.resolve() == root
    elif short.exists():
        raise RuntimeError('短路径已被占用')
    else:
        short.symlink_to(root, target_is_directory=True)
    lock = open('/tmp/linkagent-phase-b-cuda0.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = root / 'evidence' / args.name
    if output.exists():
        raise FileExistsError(output)
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                'TMPDIR': str(root / 'tmp'), 'XDG_CACHE_HOME': str(cache / 'xdg'),
                'CUDA_CACHE_PATH': str(cache / 'cuda'), 'TORCH_HOME': str(cache / 'torch'),
                'HF_HOME': str(cache / 'hf'), '__GL_SHADER_DISK_CACHE_PATH': str(cache / 'gl'),
                'OMNI_USER_CONFIG_PATH': str(cache / 'omni-user'),
                'PYTHONPYCACHEPREFIX': str(cache / 'pycache')})
    python = '/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python'
    kit = Path(python).parent.parent / 'lib/python3.12/site-packages/isaacsim/kit'
    # portable 选项未改写本机这两个路径；仅在私有挂载空间把写入导向外盘。
    derived = cache / 'DerivedDataCache'
    derived.mkdir(exist_ok=True)
    user_config = cache / 'kit-user.config.json'
    installed_config = kit / 'data/Kit/IsaacLab/3.0/user.config.json'
    if not user_config.exists():
        shutil.copyfile(installed_config, user_config)
    entry_args = args.entry_args[1:] if args.entry_args[:1]==['--'] else args.entry_args
    probe_args = ['--steps',str(args.steps),'--fixture-source-surface',args.fixture_source_surface,
                  '--fixture-center-distance-m',str(args.fixture_center_distance_m),'--fixture-digit',args.fixture_digit,
                  '--task',args.task] if args.module=='probe_contacts' else entry_args
    if args.module=='probe_contacts' and args.fixture_pad_target_m is not None:
        probe_args.extend(['--fixture-pad-target-m',*[str(x) for x in args.fixture_pad_target_m]])
    command = ['bwrap', '--die-with-parent', '--ro-bind', '/', '/', '--dev-bind', '/dev', '/dev',
               '--proc', '/proc', '--bind', str(root), str(root), '--bind', '/tmp', '/tmp',
               '--bind',str(cache/'var-tmp'),'/var/tmp',
               '--bind', str(derived), str(kit / 'cache/DerivedDataCache'),
               '--bind', str(user_config), str(installed_config),
               '--chdir', str(source), python, '-m', 'research.thumb_pad_grasp_v1.'+args.module,
               '--headless', '--device', 'cuda:0', '--num-envs', str(args.num_envs),*probe_args,
               '--output', str(output), '--kit_args',
               f'--portable --portable-root={short}/cache/kit --/log/file={short}/evidence/{args.name}-kit.log']
    started = time.time()
    receipt = {'command': command, 'cwd': str(source), 'started_unix': started,
               'status': 'RUNNING', 'purpose': 'calibration_not_training' if args.module!='run' else 'new_task_PPO_or_evaluation'}
    path = root / 'evidence' / (args.name + '-process.json')
    if path.exists():
        raise FileExistsError(path)
    snapshot = root / 'evidence' / (args.name + '-inputs')
    snapshot.mkdir(exist_ok=False)
    bindings = {}
    inputs = list((source / 'research/thumb_pad_grasp_v1').rglob('*.py'))
    inputs += list((source / 'research/thumb_pad_grasp_v1/assets').glob('*.json'))
    inputs += list((source / 'research/thumb_pad_grasp_v1/assets').glob('*.npz'))
    inputs += list((source / 'research/fixed_fruit_grasp_v1').glob('*.py'))
    for item in inputs:
        relative = item.relative_to(source)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target)
        bindings[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt['input_sha256'] = bindings
    (snapshot / 'MANIFEST.json').write_text(json.dumps(bindings,indent=2))
    path.write_text(json.dumps(receipt, indent=2))
    with (root / 'evidence' / (args.name + '.log')).open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt['pid'] = process.pid
        path.write_text(json.dumps(receipt, indent=2))
        code = process.wait()
    process_code = code
    failures = [str(p) for p in output.glob('failure*')]
    expected = 'metadata.json' if args.module=='probe_contacts' else 'result.json' if args.module=='smoke_new_task' else 'summary.json'
    # Kit close 可能吞掉 Python 异常并返回0；必须核查产物及显式失败文件。
    if code == 0 and (failures or not (output/expected).is_file()):
        code = 2
    receipt.update(process_exit_code=process_code, failure_artifacts=failures,
                   expected_completion_artifact=str(output/expected),
                   completion_artifact_exists=(output/expected).is_file())
    receipt.update(exit_code=code, elapsed_seconds=time.time()-started,
                   status='EXITED_OK' if code == 0 else 'FAILED')
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
