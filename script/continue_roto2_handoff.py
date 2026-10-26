"""Prepare or execute a relocated baseline evaluation/continuation command."""
import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = '/home/descfly/linkagent-ppo'
MATURE = 'logs/roto2_pressure_fruit/plain_seed7401/2026-10-21_16-16-13/checkpoints/final_agent.pt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['evaluate', 'record', 'train'])
    p.add_argument('--condition', choices=['nominal24', 'nominal48', 'heldout_gentle48'], default='nominal48')
    p.add_argument('--checkpoint', type=Path, default=ROOT / MATURE)
    p.add_argument('--million-steps', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--video-env-index', type=int, default=45)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve(); assert checkpoint.is_file(), checkpoint
    if args.mode == 'train':
        if not args.million_steps or args.million_steps <= 0:
            p.error('Train requires an explicit positive --million-steps budget; this does not approve a research design.')
        assert checkpoint == ROOT / MATURE, 'This continuation recipe is for the mature native model only.'
        plan = json.loads((ROOT / 'artifacts/simulation/rotation_v16/value_gradient/PLAN.json').read_text())
        cmd = [c.replace(OLD_ROOT, str(ROOT)) for c in plan['command']]
        i = cmd.index('--value-gradient-diagnostic'); del cmd[i:i+2]
        cmd[cmd.index('--resume-checkpoint')+1] = str(checkpoint)
        cmd[cmd.index('--million-steps')+1] = str(args.million_steps)
        cmd[cmd.index('--seed')+1] = str(args.seed if args.seed is not None else 9901)
    else:
        plan = json.loads((ROOT / 'artifacts/simulation/rotation_v17/ppo_pair/PLAN.json').read_text())
        condition = next(c for c in plan['evaluations'] if c['label'] == args.condition)
        cmd = [c.replace(OLD_ROOT, str(ROOT)) for c in condition['command']]
        cmd[cmd.index('--evaluate-checkpoint')+1] = str(checkpoint)
        output = ROOT / 'artifacts/simulation/handoff_new' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cmd[cmd.index('--evaluation-output')+1] = str(output / 'evaluation.json')
        if args.seed is not None:
            cmd[cmd.index('--seed')+1] = str(args.seed)
        if args.mode == 'record':
            cmd[2] = 'record'; i = cmd.index('--evaluation-output'); del cmd[i:i+2]
            cmd += ['--record-video', str(output / 'simulation.mp4'), '--video-env-index', str(args.video_env_index), '--video-max-steps', '960']
    cmd[1] = 'script/run_roto2.sh'
    if args.num_envs:
        cmd[cmd.index('--num-envs')+1] = str(args.num_envs)
    print(shlex.join(cmd), flush=True)
    if args.execute:
        if args.mode != 'train':
            output.mkdir(parents=True, exist_ok=False)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
