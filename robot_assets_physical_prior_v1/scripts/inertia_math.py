#!/usr/bin/env python3
"""Pure URDF inertia tensor and principal-axis conversion helpers."""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,xml.etree.ElementTree as ET
import numpy as np
from pxr import Gf,Sdf,Usd,UsdPhysics

URDF='botSDK/urdf/02_project_derived/ros2_packages/linkbot_bimanual_sdk_description/urdf/linkbot_bimanual_sdk_description.urdf'
ASSET='botSDK/urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg'
NAMES=('linkbot_bimanual_sdk_training_mass1kg.usda','linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda')

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def xyz(s):return np.array([float(x) for x in s.split()],dtype=float)
def rotation_rpy(rpy):
 r,p,y=rpy;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
 return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
def quat_matrix(q):
 # Gf rotates column basis vectors through Quat.Transform; no row/column ambiguity.
 return np.column_stack([np.array(q.Transform(Gf.Vec3f(*v))) for v in np.eye(3)])
def tensor(diag,q):
 R=quat_matrix(q);return R@np.diag(diag)@R.T

def urdf_bodies(path):
 out={}
 for l in ET.parse(path).getroot().findall('link'):
  i=l.find('inertial')
  if i is None:continue
  mass=float(i.find('mass').get('value'));a=i.find('inertia').attrib
  I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],[float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]])
  o=i.find('origin');com=xyz(o.get('xyz','0 0 0')) if o is not None else np.zeros(3);R=rotation_rpy(xyz(o.get('rpy','0 0 0'))) if o is not None else np.eye(3);I=R@I@R.T
  d,V=np.linalg.eigh(I)
  if mass<=0 or d[0]<=0 or d[2]>d[0]+d[1]+max(1e-14,d[2]*1e-8):raise ValueError(f'Invalid physical inertia: {l.get("name")}')
  if np.linalg.det(V)<0:V[:,0]*=-1
  q=Gf.Quatf(Gf.Matrix3d(*[float(x) for x in V.T.reshape(-1)]).ExtractRotation().GetQuat())
  if np.linalg.norm(tensor(d,q)-I)/np.linalg.norm(I)>2e-6:raise ValueError('Principal-axis round trip failed')
  out[l.get('name')]={'mass':mass,'com':com,'I':I,'diag':d,'q':q}
 return out
