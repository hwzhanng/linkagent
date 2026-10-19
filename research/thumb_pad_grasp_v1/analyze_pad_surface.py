"""逐个承载点检查源 mesh 对应误差；容差扫描是诊断而非冻结判据。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .pad_geometry import PadSurface, classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--digit', default='thumb')
    args = parser.parse_args()
    config = json.loads((Path(__file__).parent / 'assets/pad_regions_v1.approved.json').read_text())
    digit = next(d for d in config['digits'] if d['digit'] == args.digit)
    geometry = np.load(args.directory / 'source_geometry.npz')
    surface = PadSurface(geometry[args.digit + '_vertices_body_m'],
                         geometry[args.digit + '_faces'], digit['selected_triangle_ids'])
    samples = []
    for line in (args.directory / 'contacts.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['body'] != digit['body_path'].rsplit('/',1)[-1]:
            continue
        for i, value in enumerate(row['raw_0']):
            if value[0] > .001:
                samples.append((row, i))
    points = np.array([r['points_body_m'][i] for r,i in samples]).reshape(-1,3)
    # 已由 sum(force * normal) 对齐聚合力及果心方向实测：raw normal 朝向手内。
    outward = -np.array([r['normals_body_raw'][i] for r,i in samples]).reshape(-1,3)
    result = classify(surface, points, outward, position_tolerance_m=1., normal_tolerance_rad=np.pi/6)
    distances = result['distance_m']
    summary = {'schema':'source_surface_correspondence_probe_v1', 'digit':args.digit,
               'loaded_point_count':len(samples),
               'distance_to_source_mesh_m': {k:float(v) for k,v in zip(
                   ('min','median','p95','max'), [distances.min(),np.median(distances),np.quantile(distances,.95),distances.max()])} if len(distances) else {},
               'nearest_source_face_is_approved_pad_count':int(surface._pad_mask[result['nearest_triangle_id']].sum()),
               'unbounded_position_30deg_diagnostic_labels':dict(Counter(result['state'])),
               'runtime_tolerances_frozen':False, 'pad_success':'NOT_EVALUATED',
               'note':'源表面最近面诊断；位置阈值1米仅收集误差，不可用于奖励或成功判定'}
    np.savez_compressed(args.directory / (args.digit+'_surface_diagnostic.npz'),
                        points_body_m=points, outward_normals=outward, **result)
    target = args.directory / (args.digit+'_SURFACE_ANALYSIS.json')
    target.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
