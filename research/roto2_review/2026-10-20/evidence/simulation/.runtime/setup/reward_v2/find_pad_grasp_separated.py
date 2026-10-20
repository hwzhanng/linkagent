"""Candidate generator for physical validation, never a policy controller."""
exec(open('simulation/.runtime/setup/reward_v2/check_pad_reachability.py').read().split('results=[]')[0])
results=[]
for support in [(0,1,2)]:
 for trial in range(2):
  def residual(x):
   tf=g.fk(x[:16]);c=np.array([tf[name][:3,3]+tf[name][:3,:3]@centers[i] for i,name in enumerate(g.tips)]);n=np.array([tf[name][:3,:3]@normals[i] for i,name in enumerate(g.tips)])
   errors=(c[list(support)]+n[list(support)]*.032-x[16:])*100
   original=g.residual(x);collision=original[5:5+len(g.names)]
   opposition=max(0.,np.dot(n[0],n[list(support[1:])].mean(0))+.65)*4
   separation=[max(0.,.018-np.linalg.norm(c[a]-c[b]))*200 for a,b in itertools.combinations(support,2)]
   return np.r_[errors.flatten(),collision*.7,opposition,separation,.003*(x[:16]-q0),.1*(x[16:]-fruit)]
  old=json.loads((ROOT/'simulation/.runtime/setup/reward_v2/closure_probe_candidates.json').read_text())[trial]
  start=np.r_[old['joint_positions'],old['fruit_position_hand']]
  if trial:
   start[16:]=[.055,0,.17]
   for finger in support[1:]: start[g.ACTIVE_JOINTS.index(g.tips[finger].split('_')[0]+'_joint2')]=.20+.25*(trial-1)
  fit=least_squares(residual,np.clip(start,np.r_[lo,[.015,-.04,.12]]+1e-6,np.r_[hi,[.11,.04,.22]]-1e-6),bounds=(np.r_[lo,[.015,-.04,.12]],np.r_[hi,[.11,.04,.22]]),max_nfev=200,diff_step=1e-4)
  tf=g.fk(fit.x[:16]);c=np.array([tf[name][:3,3]+tf[name][:3,:3]@centers[i] for i,name in enumerate(g.tips)]);n=np.array([tf[name][:3,:3]@normals[i] for i,name in enumerate(g.tips)]);errors=np.linalg.norm(c+n*.032-fit.x[16:],axis=-1)
  r={'id':f'pad_candidate_{support}_{trial}','support_fingers':support,'pad_target_error_mm':(errors*1000).tolist(),'cost':float(np.linalg.norm(residual(fit.x))),'joint_positions':fit.x[:16].tolist(),'fruit_position_hand':fit.x[16:].tolist(),'physical_validation':False};results.append(r);print(json.dumps(r),flush=True)
results.sort(key=lambda x:x['cost']);(ROOT/'simulation/.runtime/setup/reward_v2/pad_grasp_separated_candidates.json').write_text(json.dumps(results,indent=2)+'\n')
