"""Matched finite obstacle geometry for procedural X550 scene training."""
import copy
import numpy as np
from nextgen.env import VectorFlightEnv
from .catalog import FAMILIES, make_scene

FIELDS=('obstacle_n','obstacle_e','obstacle_z','obstacle_r','obstacle_horizontal',
        'obstacle_length','obstacle_yaw','goal_north','goal_east','scene_shape',
        'half_n','half_e','half_z','lesson_ids')


class World(VectorFlightEnv):
    def __init__(self,n=1024,seed=550,horizon=400,training=False,suite='legacy'):
        if suite not in ('legacy','realistic'): raise ValueError('unknown suite')
        self.training=bool(training); self.suite=suite
        self.failure_layouts=[]; self.lesson_level=0; self.scene_initial_rejections=0
        self.lesson_ids=np.zeros(n,np.int32)
        self.scene_shape=np.zeros((n,8),np.int32)
        self.half_n=np.zeros((n,8),np.float32)
        self.half_e=np.zeros((n,8),np.float32)
        self.half_z=np.zeros((n,8),np.float32)
        super().__init__(n,seed,horizon,autoreset=False,profiles=('x550',))

    def apply_scene(self,i,scene):
        self.scene_shape[i]=0; self.obstacle_n[i]=50.; self.obstacle_e[i]=50.
        self.obstacle_horizontal[i]=False; self.obstacle_yaw[i]=0.
        self.goal_north[i]=scene['goal_north']; self.goal_east[i]=scene['goal_east']
        self.lesson_ids[i]=1+FAMILIES.index(scene['family'])
        for j,o in enumerate(scene['objects']):
            self.obstacle_n[i,j]=o['north']; self.obstacle_e[i,j]=o['east']
            self.obstacle_z[i,j]=o['altitude']; self.obstacle_yaw[i,j]=o['yaw']
            self.half_n[i,j],self.half_e[i,j],self.half_z[i,j]=np.asarray(o['size'])/2
            self.scene_shape[i,j]=1 if o['shape']=='box' else 2
            self.obstacle_r[i,j]=np.hypot(*np.asarray(o['size'][:2])/2) if o['shape']=='box' else o['size'][0]/2
            self.obstacle_horizontal[i,j]=o['shape']=='box'
            self.obstacle_length[i,j]=max(o['size'])

    def reset(self,mask=None):
        ids=np.arange(self.n) if mask is None else np.flatnonzero(mask)
        self.scene_shape[ids]=0; self.lesson_ids[ids]=0
        super().reset(mask)
        if self._reset_depth: return self.observe()
        if not self.training and self.suite=='legacy': return self.observe()
        draws=self.rng.random(len(ids)) if self.training else np.zeros(len(ids))
        for i,u in zip(ids,draws):
            if u<.60:
                family=None if self.training else FAMILIES[int(i)%len(FAMILIES)]
                self.apply_scene(i,make_scene(self.rng,family))
            elif u<.70 and self.failure_layouts:
                layout=self.failure_layouts[int(self.rng.integers(len(self.failure_layouts)))]
                for k,v in layout.items(): getattr(self,k)[i]=copy.deepcopy(v)
        invalid=np.zeros(self.n,bool)
        invalid[ids]=self._geometry()[0][ids]<.10
        if invalid.any():
            if not self.training: raise RuntimeError('invalid evaluation scene initial pose')
            # A replayed layout may overlap a newly randomized starting pose.
            # Reject that INITIAL draw, not a flown episode or its collision.
            self.scene_initial_rejections+=int(invalid.sum())
            self.scene_shape[invalid]=0; self.lesson_ids[invalid]=0
            super().reset(invalid)
        return self.observe()
    def _clearance_matrix(self):
        original=super()._clearance_matrix()
        dn=self.north[:,None]-self.obstacle_n; de=self.east[:,None]-self.obstacle_e
        c=np.cos(self.obstacle_yaw); s=np.sin(self.obstacle_yaw)
        qn=np.abs(c*dn+s*de)-self.half_n-self.footprint[:,None]
        qe=np.abs(-s*dn+c*de)-self.half_e-self.footprint[:,None]
        qz=np.abs(self.altitude[:,None]-self.obstacle_z)-self.half_z-self.vertical_radius[:,None]
        q=np.stack((qn,qe,qz),axis=-1)
        box=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)
        radial=np.hypot(dn,de)-self.obstacle_r-self.footprint[:,None]
        cyl=np.hypot(np.maximum(radial,0),np.maximum(qz,0))+np.minimum(np.maximum(radial,qz),0)
        # Conservative rotor envelope: expanded boxes/cylinders include corners.
        return np.where(self.scene_shape==1,box,np.where(self.scene_shape==2,cyl,original))

    def remember_failures(self,collision):
        if not self.training: return
        for i in np.flatnonzero(collision)[:8]:
            self.failure_layouts.append({k:copy.deepcopy(getattr(self,k)[i]) for k in FIELDS})
        self.failure_layouts=self.failure_layouts[-256:]
