"""用实际接触数据验证 packed 字段及球面对应；不推定指肚成功。"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.directory / 'contacts.jsonl').read_text().splitlines()]
    force_error, normal_length_error, direction_cos, distance_models = [], [], [], []
    bodies = Counter()
    loaded_points = 0
    for row in rows:
        force = np.asarray(row['raw_0']).reshape(-1)
        normal = np.asarray(row['raw_2'])
        points = np.asarray(row['points_w'])
        sep = np.asarray(row['separations_m']).reshape(-1)
        center = np.asarray(row['fruit_pose_w'][:3])
        aggregate = np.asarray(row['filtered_force_w'])
        force_error.append(np.linalg.norm((force[:, None] * normal).sum(0) - aggregate))
        normal_length_error.extend(abs(np.linalg.norm(normal, axis=1) - 1).tolist())
        mask = force > .001
        loaded_points += int(mask.sum())
        bodies[row['body']] += int(mask.sum())
        direction = center - points
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        direction_cos.extend(np.sum(normal[mask] * direction[mask], axis=1).tolist())
        radial_gap = np.linalg.norm(points - center, axis=1) - .0335
        # 三种位置定义候选只输出误差，不自动冻结运行时容差。
        distance_models.extend(np.stack([radial_gap, radial_gap-sep/2, radial_gap-sep], axis=1)[mask].tolist())
    def stats(values):
        value = np.asarray(values)
        if value.size == 0:
            return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
        return {'min': float(value.min()), 'median': float(np.median(value)),
                'p95': float(np.quantile(value,.95)), 'max': float(value.max())}
    summary = {
        'schema': 'actual_packed_contact_probe_analysis_v1',
        'contact_rows': len(rows), 'loaded_points_above_0_001N': loaded_points,
        'loaded_points_by_body': dict(bodies),
        'sum_raw0_times_raw2_vs_filtered_force_error_N': stats(force_error),
        'raw2_unit_length_error': stats(normal_length_error),
        'raw2_dot_point_to_fruit_center_loaded': stats(direction_cos),
        'surface_model_abs_error_m': {name: stats(np.abs(np.asarray(distance_models)[:,i]))
            for i,name in enumerate(('point_on_fruit', 'point_is_midpoint', 'point_on_hand'))} if distance_models else {},
        'scope': 'one_env_constant_action_probe; not known_pad_positive; tolerances not frozen',
        'pad_success': 'UNKNOWN_NOT_EVALUATED',
    }
    (args.directory / 'PACKED_LAYOUT_ANALYSIS.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
