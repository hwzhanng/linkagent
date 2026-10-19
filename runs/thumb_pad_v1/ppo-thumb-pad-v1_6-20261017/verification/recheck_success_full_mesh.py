"""用未加速完整源网格分类器复核指定评估的成功窗；学习归因另由参数更新及对照证明。"""
import sys,pathlib,json,hashlib,importlib.util,math,numpy as np
from scipy.spatial.transform import Rotation
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--audit-dir",type=pathlib.Path,required=True)
parser.add_argument("--run-dir",type=pathlib.Path,required=True)
parser.add_argument("--output",type=pathlib.Path,required=True)
parser.add_argument("--window-index",type=int,default=0)
parser.add_argument("--source-root",type=pathlib.Path,required=True)
parser.add_argument("--original-classifier",type=pathlib.Path,default=pathlib.Path(__file__).parent/"original_pad_geometry.py")
parser.add_argument("--raw-window",type=pathlib.Path)
args=parser.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
s=args.source_root;p=args.original_classifier;sp=importlib.util.spec_from_file_location('original_full_mesh',p);m=importlib.util.module_from_spec(sp);sys.modules['original_full_mesh']=m;sp.loader.exec_module(m)
audit_dir=args.audit_dir;audit=json.loads((audit_dir/'audit.json').read_text());window=audit['candidate_windows'][args.window_index];eid=window['env']
raw=args.raw_window or pathlib.Path(window['raw_path']);assert hashlib.sha256(raw.read_bytes()).hexdigest()==window['raw_sha256'];records=[json.loads(l) for l in raw.open()];records=[x.get('record',x) for x in records];assert len(records)==120 and [x['sim_step'] for x in records]==list(range(records[0]['sim_step'],records[0]['sim_step']+120));assert [x['physical_streak'][eid] for x in records]==list(range(1,121))
regions=json.loads((s/'research/thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json').read_text());g=np.load(s/'research/thumb_pad_grasp_v1/assets/source_geometry_v1.npz');surfaces={};edges={}
for d in regions['digits']:
 body=d['body_path'].split('/')[-1];digit=d['digit'];surfaces[body]=m.PadSurface(g[digit+'_vertices_body_m'],g[digit+'_faces'],d['selected_triangle_ids']);edges[body]=np.array(d['boundary_edges_body_local_m'])
batches={}
for t in records:
 assert t['released'][eid] and all(v==0 for v in t['support_force_w'][eid]+t['support_torque_w'][eid])
 assert t['truth']['required_contact'][eid] and not any(t['truth']['unknown'][eid]) and not any(v[eid] for v in t['hard_faults'].values())
 for c in t['contacts']:
  for j,env in enumerate(c['env_ids']):
   if env!=eid:continue
   b=batches.setdefault(c['body'],[]);b.append({k:c[k][j] for k in ['points_body_m','outward_normals_body','points_w','raw_normals_w','body_pose_w','labels','forces_n']})
reports=[]
def classify(body,points,normals):
 result=m.classify(surfaces[body],points,normals,position_tolerance_m=.0005,normal_tolerance_rad=np.deg2rad(25 if body.startswith('thumb_') else 45));e=edges[body];delta=e[:,1]-e[:,0];amount=np.sum((result['closest_point_body_m'][:,None]-e[:,0])*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=1),1e-24);nearest=e[:,0]+np.clip(amount,0,1)[...,None]*delta;distance=np.linalg.norm(result['closest_point_body_m'][:,None]-nearest,axis=2).min(1);result['state'][distance<.0005]='UNKNOWN';return result,distance
for body,rows in batches.items():
 if body not in surfaces:
  assert all(x['labels']==2 for x in rows);reports.append({'body':body,'points':len(rows),'all_original_nondistal_NON_PAD':True});continue
 points=np.array([x['points_body_m'] for x in rows]);normals=np.array([x['outward_normals_body'] for x in rows]);expected=np.array([{1:'PAD',2:'NON_PAD',3:'UNKNOWN'}[x['labels']] for x in rows]);pose=np.array([x['body_pose_w'] for x in rows]);transform=Rotation.from_quat(pose[:,3:]).inv();recomputed_points=transform.apply(np.array([x['points_w'] for x in rows])-pose[:,:3]);recomputed_normals=transform.apply(-np.array([x['raw_normals_w'] for x in rows]));a,bd=classify(body,points,normals);b,_=classify(body,recomputed_points,recomputed_normals);assert np.array_equal(a['state'],expected) and np.array_equal(b['state'],expected),(body,'classification mismatch')
 reports.append({'body':body,'points':len(rows),'original_full_mesh_labels_match':True,'world_pose_reconstructed_labels_match':True,'label_counts':{label:int(np.sum(expected==label)) for label in set(expected.tolist())},'max_local_point_reconstruction_error_m':float(np.max(np.linalg.norm(recomputed_points-points,axis=1))),'max_source_distance_m':float(np.max(a['distance_m'])),'min_approved_boundary_distance_m':float(np.min(bd))})
episodes=[json.loads(line) for line in (args.run_dir/'episodes.jsonl').open()]
episodes=[x.get('record',x) for x in episodes]
episode=next(x for x in episodes if x['env_id']==eid and x['success'] and x['max_physical_hold_streak_substeps']>=120)
metadata=json.loads((args.run_dir/'run_metadata.json').read_text())
out={'status':'EVALUATION_PHYSICAL_SUCCESS_WINDOW_RECLASSIFIED','env':eid,'raw_window':str(raw),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'original_full_classifier_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frames':120,'duration_s':sum(x['dt_s'] for x in records),'zero_support_all_frames':True,'seven_faults_all_false':True,'required_contact_all_frames':True,'body_reports':reports,'completed_episode':episode,'learning_attribution':'NOT_ESTABLISHED_BY_GEOMETRY_REVIEW', 'run_metadata_sha256':hashlib.sha256((args.run_dir/'run_metadata.json').read_bytes()).hexdigest(), 'checkpoint_sha256':metadata.get('checkpoint_sha256'),'scope':'Original complete triangle classifier independently rerun on recorded local points and reconstructed world/body transforms; raw frame audit retains original runtime guard provenance for hand-frame drift and all hard fault inputs.'};args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='completed_episode'},indent=2))
