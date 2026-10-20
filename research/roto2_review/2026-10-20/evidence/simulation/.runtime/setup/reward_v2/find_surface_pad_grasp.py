"""Read-only candidate search against actual sensor-cap front triangles."""
exec(open('simulation/.runtime/setup/reward_v2/check_pad_reachability.py').read().split('results=[]')[0])
import trimesh
cap_tri=[];cap_norm=[]
for i,name in enumerate(g.tips):
 m=g.meshes[name];part=sorted(m.split(only_watertight=False),key=lambda x:-x.area)[3]
 c,u,v,n=[x.numpy()[i] for x in pad_frames('cpu')]
 rel=part.triangles_center-c
 keep=(part.face_normals@n>.15)&(abs(rel@u)<.0047)&(abs(rel@v)<.0071)
 cap_tri.append(part.triangles[keep]);cap_norm.append(part.face_normals[keep])
print('sensor triangles',[len(x) for x in cap_tri],flush=True)
results=[]
for support in [(0,1,2),(0,2,3)]:
 for trial in range(2):
  def measure(q,fruit):
   tf=g.fk(q);pts=[];directions=[];distances=[]
   for i in support:
    t=tf[g.tips[i]];p=t[:3,:3].T@(fruit-t[:3,3]);near=trimesh.triangles.closest_point(cap_tri[i],np.broadcast_to(p,(len(cap_tri[i]),3)))
    delta=p-near;dist=np.linalg.norm(delta,axis=-1);j=dist.argmin();sign=1 if delta[j]@cap_norm[i][j]>=0 else -1
    distances.append(dist[j]*sign);pts.append(t[:3,3]+t[:3,:3]@near[j]);directions.append(t[:3,:3]@delta[j]/max(dist[j],1e-8))
   return np.array(distances),np.array(pts),np.array(directions)
  def residual(x):
   distances,points,directions=measure(x[:16],x[16:]);original=g.residual(x);collision=original[5:5+len(g.names)]
   opposition=max(0.,directions[0]@directions[1:].mean(0)+.5)*2
   separation=[max(0.,.016-np.linalg.norm(points[a]-points[b]))*150 for a,b in itertools.combinations(range(3),2)]
   return np.r_[(distances-.0315)*150,collision*.8,opposition,separation,.005*(x[:16]-q0)]
  start=np.r_[q0,[.065,0,.19]]
  for i in range(1,5):
   finger=g.tips[i].split('_')[0]
   start[g.ACTIVE_JOINTS.index(finger+'_joint0')]=0
   start[g.ACTIVE_JOINTS.index(finger+'_joint1')]=.5 if i in support else .2
   start[g.ACTIVE_JOINTS.index(finger+'_joint2')]=.4 if i in support else .2
  if trial:start[16:]=[.06,.01,.17];start[5]=.45;start[10]=-.6
  l=np.r_[lo,[.015,-.04,.12]];h=np.r_[hi,[.12,.04,.235]]
  for finger in ['index','middle','ring','little']:
   i=g.ACTIVE_JOINTS.index(finger+'_joint0');l[i]=-.1;h[i]=.1
  fit=least_squares(residual,np.clip(start,l+1e-6,h-1e-6),bounds=(l,h),max_nfev=160,diff_step=1e-4)
  d,p,n=measure(fit.x[:16],fit.x[16:]);r={'id':f'cap_surface_{support}_{trial}','cost':float(np.linalg.norm(residual(fit.x))),'support_fingers':support,'surface_gap_mm':((d-.032)*1000).tolist(),'opposition':float(n[0]@n[1:].mean(0)),'joint_positions':fit.x[:16].tolist(),'fruit_position_hand':fit.x[16:].tolist(),'physical_validation':False};results.append(r);print(json.dumps(r),flush=True)
results.sort(key=lambda x:x['cost']);(ROOT/'simulation/.runtime/setup/reward_v2/surface_pad_candidates.json').write_text(json.dumps(results,indent=2)+'\n')
