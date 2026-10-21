import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from simulation.roto2_pressure.core import ACTIVE_JOINTS, MIMICS
from module.skills.tomato_pick.hand_synergy import g20_synergy_to_joint_targets
folder = ROOT / 'botSDK/urdf/01_vendor_original/02_extracted_packages/L20_8_r_s1'
urdf = ET.parse(folder / 'L20_8_r_s1.urdf').getroot()

def origin(node):
    out = np.eye(4)
    if node is not None:
        out[:3,:3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
        out[:3,3] = np.fromstring(node.get('xyz','0 0 0'),sep=' ')
    return out

joints = []
limits = {}
for j in urdf.findall('joint'):
    name = j.get('name')
    axis = np.fromstring(j.find('axis').get('xyz'),sep=' ') if j.find('axis') is not None else np.array([1.,0,0])
    joints.append((name,j.find('parent').get('link'),j.find('child').get('link'),origin(j.find('origin')),axis))
    if j.find('limit') is not None:
        limits[name] = [float(j.find('limit').get(k)) for k in ('lower','upper')]
meshes = {}
planes = {}
triangles = {}
for link in urdf.findall('link'):
    c = link.find('collision')
    if c is None: continue
    mesh = trimesh.load(folder / c.find('geometry/mesh').get('filename'), force='mesh')
    mesh.apply_transform(origin(c.find('origin')))
    meshes[link.get('name')] = mesh
    hull = mesh.convex_hull
    triangles[link.get('name')] = hull.triangles
    planes[link.get('name')] = (hull.face_normals, -(hull.face_normals * hull.triangles_center).sum(-1))

def fk(q):
    q = dict(zip(ACTIVE_JOINTS,q))
    q.update({f:q[d]*r for f,(d,r) in MIMICS.items()})
    transforms = {'base_link':np.eye(4)}
    for name,parent,child,frame,axis in joints:
        rot = np.eye(4)
        rot[:3,:3] = Rotation.from_rotvec(axis*q.get(name,0)).as_matrix()
        transforms[child] = transforms[parent] @ frame @ rot
    return transforms

tips = ['thumb_link4','index_link3','middle_link3','ring_link3','little_link3']
names = list(planes)
lower = np.array([limits[n][0]+.002 for n in ACTIVE_JOINTS]+[.01,-.04,.11])
upper = np.array([limits[n][1]-.002 for n in ACTIVE_JOINTS]+[.12,.04,.24])
radius = .032

def residual(x):
    transforms = fk(x[:16])
    distances = []
    directions = {}
    for name,(normals,offsets) in planes.items():
        tf = transforms[name]
        p = tf[:3,:3].T@(x[16:]-tf[:3,3])
        nearest = trimesh.triangles.closest_point(triangles[name], np.broadcast_to(p, (len(triangles[name]),3)))
        norms = np.linalg.norm(nearest-p,axis=-1)
        idx = norms.argmin()
        distance = norms[idx] * (1 if np.max(normals@p+offsets) >= 0 else -1)
        distances.append(distance)
        directions[name] = tf[:3,:3] @ (nearest[idx]-p) / max(norms[idx],1e-9)
    distances = np.array(distances)
    contact = [(distances[names.index(n)]-(radius-.001))*100 for n in tips]
    collision = np.minimum(distances-radius+.0015,0)*200
    # Stay away from extreme joint limits; the physical hold test is decisive.
    opposing = max(0, np.dot(directions[tips[0]], np.mean([directions[n] for n in tips[1:]],axis=0))+.7)
    return np.r_[contact,collision, opposing*.2, .005*(x[:16]-.5)]

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    results=[]
    previous=json.loads((ROOT/'simulation/.runtime/setup/grasp_candidates.json').read_text())
    for index in range(16):
        q=np.array(g20_synergy_to_joint_targets((rng.uniform(.1,.7),rng.uniform(-.2,.3),0,0,rng.uniform(0,.7))))
        x=np.r_[q,[.065,0,.17]]
        if index>4: x[:16] += rng.normal(0,.12,16)
        if index < 8: x=np.r_[previous[index]['joint_positions'],previous[index]['fruit_position_hand']]
        result=least_squares(residual,np.clip(x,lower+1e-6,upper-1e-6),bounds=(lower,upper),max_nfev=160,diff_step=1e-4)
        results.append({'cost':float(np.linalg.norm(residual(result.x))), 'joint_positions':result.x[:16].tolist(), 'fruit_position_hand':result.x[16:].tolist()})
        print(index,results[-1]['cost'],results[-1]['fruit_position_hand'],flush=True)
    results.sort(key=lambda r:r['cost'])
    (ROOT/'simulation/.runtime/setup/grasp_candidates_exact.json').write_text(json.dumps(results,indent=2)+'\n')
