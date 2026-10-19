"""复核真实评估轨迹并提取候选成功窗；不替代原运行时物理守卫。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


FAULTS = {'overload', 'nonfinite', 'penetration', 'joint_limit',
          'self_collision', 'instability', 'joint_effort'}
DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(vector):
    if not all(math.isfinite(v) for v in vector):
        raise ValueError('非有限物理向量')
    return math.sqrt(sum(v * v for v in vector))


def audit(trace, approved, output):
    output.mkdir(exist_ok=False, parents=True)
    regions = json.loads(approved.read_text())
    if not regions['approved']:
        raise ValueError('指肚区域未获确认')
    surfaces = {r['body_path'].split('/')[-1]: set(r['selected_triangle_ids'])
                for r in regions['digits']}
    stats = {'schema': 'offline_pad_window_audit.v1', 'trace': str(trace.resolve()),
             'trace_sha256': digest(trace), 'approved_sha256': digest(approved),
             'auditor_sha256': digest(Path(__file__)),
             'checked_thresholds': {'pad_min_n': .08, 'other_min_n': .08,
                 'thumb_nonpad_max_n': 0, 'source_distance_max_m': .0005,
                 'world_speed_max_m_s': .05, 'world_drift_max_m': .01,
                 'required_substeps': 120, 'physics_hz': 120},
             'frames': 0, 'candidate_windows': [], 'per_env': [],
             'scope': '独立核对逐点载荷归约、已批准面编号、连续计数、零支撑、速度和七故障；分类标签/法向容差仍依赖已绑定运行时分类器。世界坐标漂移作为额外检查；手坐标漂移依赖原守卫。候选窗口不自动等于最终验收。'}
    histories = None
    previous_counter = None
    for line_number, line in enumerate(trace.open(), 1):
        wrapped = json.loads(line)
        row = wrapped.get('record', wrapped)
        counter = row['sim_step']
        if previous_counter is not None and counter != previous_counter + 1:
            raise ValueError(f'物理计数缺步/重复：行{line_number}')
        previous_counter = counter
        if abs(row['dt_s'] - 1 / 120) > 1e-9:
            raise ValueError(f'非120Hz子步：行{line_number}')
        if set(row['hard_faults']) != FAULTS:
            raise ValueError('七硬故障字段不完整')
        n = len(row['released'])
        if histories is None:
            histories = [[] for _ in range(n)]
            stats['per_env'] = [{'max_reported_streak': 0, 'max_checked_window': 0,
                                 'post_release_frames': 0, 'nonzero_support_frames': 0}
                                for _ in range(n)]
        if len(histories) != n:
            raise ValueError('环境数改变')
        pad = [[0.] * 5 for _ in range(n)]
        nonpad = [[0.] * 5 for _ in range(n)]
        unknown = [[False] * 5 for _ in range(n)]
        for channel in row['invalid_contact_channels']:
            for env in channel['env_ids']:
                unknown[env] = [True] * 5
        for contact in row['contacts']:
            body = contact['body']
            digit = DIGITS.index(body.split('_')[0])
            count = len(contact['env_ids'])
            if any(len(contact[k]) != count for k in ['forces_n', 'labels']):
                raise ValueError('逐点字段长度不一致')
            for j, env in enumerate(contact['env_ids']):
                force, label = contact['forces_n'][j], contact['labels'][j]
                if not math.isfinite(force) or force <= 0:
                    raise ValueError('真实承载点载荷非正或非有限')
                if label == 1:
                    if body not in surfaces or contact['nearest_triangle_ids'][j] not in surfaces[body]:
                        raise ValueError('PAD点不属于批准面')
                    if contact['source_distance_m'][j] > .0005 + 1e-9:
                        raise ValueError('PAD点超过冻结位置容差')
                    pad[env][digit] += force
                elif label == 2:
                    nonpad[env][digit] += force
                elif label == 3:
                    unknown[env][digit] = True
                else:
                    raise ValueError('未知分类标签')
        truth = row['truth']
        for env in range(n):
            for key, values in [('pad_force_n', pad), ('nonpad_force_n', nonpad)]:
                if any(abs(a-b) > 2e-5 for a, b in zip(values[env], truth[key][env])):
                    raise ValueError(f'载荷归约不一致：行{line_number}/环境{env}')
            if unknown[env] != truth['unknown'][env]:
                raise ValueError('UNKNOWN归约不一致')
            required = (pad[env][0] >= .08 and nonpad[env][0] == 0
                        and not any(unknown[env])
                        and any(a+b >= .08 for a, b in zip(pad[env][1:], nonpad[env][1:])))
            if required != truth['required_contact'][env]:
                raise ValueError('合格接触归约不一致')
            s = stats['per_env'][env]
            streak = row['physical_streak'][env]
            s['max_reported_streak'] = max(streak, s['max_reported_streak'])
            support_zero = all(v == 0 for v in row['support_force_w'][env] + row['support_torque_w'][env])
            if row['released'][env]:
                s['post_release_frames'] += 1
                s['nonzero_support_frames'] += not support_zero
            valid = (row['released'][env] and support_zero and required
                     and not any(row['hard_faults'][k][env] for k in FAULTS)
                     and norm(row['fruit_linear_velocity_w'][env]) <= .05)
            history = histories[env]
            if not valid or streak == 0:
                history.clear()
            else:
                if history and streak != history[-1][2] + 1:
                    history.clear()
                history.append((line_number, row['fruit_pose_w'][env][:3], streak))
                s['max_checked_window'] = max(s['max_checked_window'], len(history))
                if len(history) == 120:
                    anchor = history[0][1]
                    drift = max(norm([x-y for x, y in zip(pos, anchor)]) for _, pos, _ in history)
                    if drift <= .01:
                        stats['candidate_windows'].append({'env': env, 'first_line': history[0][0],
                            'last_line': line_number, 'duration_s': 1., 'max_world_drift_m': drift})
        stats['frames'] += 1
    for index, window in enumerate(stats['candidate_windows']):
        path = output / f'candidate_{index:03d}_raw.jsonl'
        with path.open('x') as target, trace.open() as source:
            for line_number, line in enumerate(source, 1):
                if window['first_line'] <= line_number <= window['last_line']:
                    target.write(line)
                if line_number >= window['last_line']:
                    break
        window.update(raw_path=str(path.resolve()), raw_sha256=digest(path))
    stats['status'] = 'CANDIDATE_WINDOWS_REQUIRE_FINAL_REVIEW' if stats['candidate_windows'] else 'NO_1S_WINDOW'
    (output / 'audit.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--approved', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.trace, args.approved, args.output), ensure_ascii=False, indent=2))
