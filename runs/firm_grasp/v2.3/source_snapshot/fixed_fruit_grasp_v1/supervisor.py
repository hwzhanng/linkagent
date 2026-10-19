"""单次物理进程的资源边界和原始退出记录；不自动重试或启动主训。"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--budget-seconds', type=int, default=43200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name.replace('_', '').replace('-', '').isalnum():
        parser.error('name 只能包含字母、数字、下划线和短横线')
    if not 0 < args.timeout_seconds <= args.budget_seconds <= 43200:
        parser.error('单次时限和累计运行预算必须为正且最多12小时')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('缺少实际命令')
    root = args.run_root.resolve()
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise RuntimeError('运行目录不存在或不可写')
    mount = subprocess.run(['findmnt', '-n', '-o', 'TARGET', '-T', str(root)], capture_output=True, text=True, check=True).stdout.strip()
    if mount == '/' or shutil.disk_usage(root).free < 5 * 1024**3:
        raise RuntimeError('运行目录必须位于已挂载外置盘且至少余5GiB')
    evidence = root / 'evidence'
    evidence.mkdir(exist_ok=True)
    receipt = evidence / (args.name + '-process.json')
    log_path = evidence / (args.name + '.log')
    if receipt.exists() or log_path.exists():
        raise FileExistsError('运行身份已存在，请保留旧失败记录并使用新身份')
    with open('/tmp/linkagent-phase-b-cuda0.lock', 'a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = [json.loads(p.read_text()) for p in evidence.glob('*-process.json')]
        if any(p.get('status') == 'RUNNING' or 'elapsed_seconds' not in p for p in prior):
            raise RuntimeError('存在未终结的进程记录，先核对并结算后才能继续运行')
        used = sum(float(p['elapsed_seconds']) for p in prior)
        allowance = args.budget_seconds - used
        if allowance <= 0:
            raise RuntimeError('本轮累计运行预算已用尽')
        timeout = min(args.timeout_seconds, allowance)
        alias = Path('/tmp/linkbot-ppo-grasp-learning-20261013')
        if alias.is_symlink():
            if alias.resolve() != root:
                raise RuntimeError('缓存路径别名指向其他目录')
        elif alias.exists():
            raise RuntimeError('缓存路径别名被占用')
        else:
            alias.symlink_to(root, target_is_directory=True)
        env = os.environ.copy()
        for key, relative in {
            'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache/xdg', 'CUDA_CACHE_PATH': 'cache/cuda',
            'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/hf', 'MPLCONFIGDIR': 'cache/matplotlib',
        }.items():
            directory = alias / relative
            directory.mkdir(parents=True, exist_ok=True)
            env[key] = str(directory)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        env['WANDB_MODE'] = 'disabled'
        # AppLauncher 使用空白 split，故以无空格别名传入 Kit 本地缓存路径。
        if '--kit_args' not in command:
            command += ['--kit_args', f'--portable --portable-root={alias}/cache/kit --/log/file={alias}/evidence/{args.name}-kit.log']
        started = time.monotonic()
        record = {'name': args.name, 'command': command, 'cwd': os.getcwd(), 'start_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'status': 'RUNNING', 'timeout_seconds': timeout, 'prior_consumed_seconds': used,
                  'budget_seconds': args.budget_seconds, 'log': str(log_path), 'cache_root': str(root / 'cache')}
        atomic_json(receipt, record)
        stop_signals = []
        def request_stop(signum, _frame):
            stop_signals.append(signum)
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous_handlers:
            signal.signal(sig, request_stop)
        with log_path.open('x') as log:
            try:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(gpu_lock.fileno(),))
            except BaseException as error:
                record.update(status='SPAWN_FAILED', error=repr(error), exit_code=None,
                              elapsed_seconds=time.monotonic() - started,
                              end_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                atomic_json(receipt, record)
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
                print(json.dumps(record, ensure_ascii=False))
                return 1
            record['pid'] = process.pid
            atomic_json(receipt, record)
            try:
                while process.poll() is None:
                    if stop_signals:
                        record['status'] = 'INTERRUPTED'
                        record['received_signal'] = stop_signals[0]
                        break
                    if time.monotonic() - started >= timeout:
                        record['status'] = 'TIMEOUT'
                        break
                    if shutil.disk_usage('/').free < 256 * 1024**2 or shutil.disk_usage(root).free < 2 * 1024**3:
                        record['status'] = 'DISK_STOP'
                        break
                    time.sleep(1)
            except BaseException as error:
                record['status'] = 'INTERRUPTED'
                record['error'] = repr(error)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                record['exit_code'] = process.returncode
                if record['status'] == 'RUNNING':
                    record['status'] = 'EXITED_OK' if process.returncode == 0 else 'FAILED'
                record['elapsed_seconds'] = time.monotonic() - started
                record['end_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
                atomic_json(receipt, record)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        print(json.dumps(record, ensure_ascii=False))
        return 0 if record['status'] == 'EXITED_OK' else 1


if __name__ == '__main__':
    sys.exit(main())
