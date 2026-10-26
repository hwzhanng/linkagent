"""Propose recovery resets from real first-episode states; heldout source seeds stay separate."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v9/sprint'


def main():
    anchors=json.loads((ROOT/'simulation/roto2_pressure/reset_grasps_v8_validated256.json').read_text())
    qref=np.asarray([r['joint_positions'] for r in anchors]);posref=np.asarray([r['fruit_position_hand'] for r in anchors])
    mount=Rotation.from_quat([0,-2**-.5,0,2**-.5]);rows=[];counts={}
    for eval_seed,split in [(9101,'train'),(9102,'heldout')]:
        pools={time:[] for time in (2,6,10,14,18)}
        for seed in (8301,8302,8303):
            path=ROOT/f'artifacts/simulation/rotation_v8/stage3/results/plain_seed{seed}/seed{eval_seed}/action_trace_seed{eval_seed}.npz'
            with np.load(path) as a:
                dist=np.linalg.norm(a['grasp_reference_initial'][:,None]-qref[None],axis=-1)
                inds=dist.argmin(1);assert dist.min(1).max()<1e-5
                bases=a['center_reference_initial']-mount.apply(posref[inds])
                assert np.abs(bases[:,2]-.9).max()<1e-5
                for seconds in pools:
                    t=int(seconds/.05)-1
                    for ei in range(64):
                        if not (a['active'][t+1,ei] and not a['done'][t,ei] and not a['done'][t+1,ei]):continue
                        if a['peak_pressure_pa'][t,ei]>12000 or a['pad'][t,ei].sum()<1:continue
                        pose=a['pose'][t,ei];local=mount.inv().apply(pose[:3]-bases[ei])
                        q=a['joint'][t+1,ei];target=a['target'][t+1,ei]
                        assert np.isfinite(q).all() and np.isfinite(pose).all()
                        pools[seconds].append(dict(id=f'phase_{split}_p{seed}_e{eval_seed}_n{ei}_t{t}',
                            joint_positions=q.tolist(),joint_targets=target.tolist(),
                            fruit_position_hand=local.tolist(),fruit_quaternion_xyzw=pose[3:].tolist(),
                            hand_posture='palm_up',proposal_only=True,physical_validation=False,
                            reset_role='recovery_curriculum',not_n_minus_one_anchor=True,split=split,
                            source_trace=str(path),source_trace_step=t,source_env=ei,source_seconds=seconds,
                            source_pad_count=int(a['pad'][t,ei].sum()),
                            time_alignment='post-action pose[t] paired with pre-action joint/target[t+1]; nonterminal',
                            source_initial_grasp_id=anchors[int(inds[ei])]['id']))
        rng=np.random.default_rng(9390+eval_seed)
        for seconds,pool in pools.items():
            rng.shuffle(pool);take=pool[:80];rows.extend(take);counts[f'{split}_{seconds}s']=len(take)
    assert rows and len({r['id'] for r in rows})==len(rows)
    (OUT/'phase_proposals.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'PHASE_PROPOSAL_METADATA.json').write_text(json.dumps(dict(counts=counts,total=len(rows),
       physical_validation=False,old_evaluation_seeds_reclassified_as_curriculum_sources=[9101,9102],
       future_evaluation_seeds=[9501,9502,9601],initial_velocities_reset_to_zero=True,
       note='Recovery states supplement independently verified four-pad anchors; not mislabeled as N-1 grasps.'),indent=2)+'\n')
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
