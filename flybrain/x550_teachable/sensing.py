"""Sparse pinhole stereo-depth surrogate and the matching numeric-depth adapter.
No RGB rendering, actual camera calibration, or hardware I/O is claimed.
"""
import numpy as np
from .schema import OBSERVATION_SIZE

MAX_DEPTH=8.0
MIN_DEPTH=.30
GRID=(3,9)

def make_rays():
    # Nine samples per sector, ordered row, column, sample-row, sample-column.
    v=[]
    for row in range(3):
        for col in range(9):
            for sy in range(3):
                for sx in range(3):
                    right=(2*(col+(sx+.5)/3)/9-1)*np.tan(np.deg2rad(40))
                    up=(1-2*(row+(sy+.5)/3)/3)*np.tan(np.deg2rad(27.5))
                    v.append((1.,right,up))
    return np.asarray(v,np.float32)

RAYS=make_rays()

def reduce_samples(depth,valid):
    depth=np.asarray(depth,np.float32).reshape(-1,27,9)
    valid=np.asarray(valid,bool).reshape(depth.shape)
    usable=valid & np.isfinite(depth) & (depth>=MIN_DEPTH) & (depth<=MAX_DEPTH)
    near=np.min(np.where(usable,depth,np.inf),axis=-1)
    near=np.where(np.isfinite(near),near,0.)
    return near.astype(np.float32),usable.mean(axis=-1).astype(np.float32)

def from_depth_frame(depth, intrinsics, unit_scale=.001):
    """Conservatively pool complete image sectors in optical-Z coordinates."""
    d=np.asarray(depth); k=np.asarray(intrinsics,float)
    if not np.isfinite(unit_scale): raise ValueError("finite depth units required")
    if d.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1],unit_scale)<=0:
        raise ValueError('finite pinhole calibration and positive depth units required')
    x_edges=np.linspace(-np.tan(np.deg2rad(40)),np.tan(np.deg2rad(40)),10)
    y_edges=np.linspace(np.tan(np.deg2rad(27.5)),-np.tan(np.deg2rad(27.5)),4)
    result=[]; coverage=[]
    for row in range(3):
        for col in range(9):
            u=np.sort(k[0,0]*x_edges[col:col+2]+k[0,2])
            v=np.sort(-k[1,1]*y_edges[row:row+2]+k[1,2])
            u0,u1=max(0,int(np.floor(u[0]))),min(d.shape[1],int(np.ceil(u[1])))
            v0,v1=max(0,int(np.floor(v[0]))),min(d.shape[0],int(np.ceil(v[1])))
            region=d[v0:v1,u0:u1].astype(np.float32)*unit_scale
            usable=np.isfinite(region)&(region>=MIN_DEPTH)&(region<=MAX_DEPTH)
            coverage.append(float(usable.mean()) if region.size else 0.)
            # Fifth percentile retains narrow connected foreground while rejecting
            # isolated one-pixel speckle in an otherwise clear sector.
            result.append(float(np.percentile(region[usable],5)) if usable.any() else 0.)
    return np.asarray(result,np.float32)[None,:],np.asarray(coverage,np.float32)[None,:]

def pack_observation(state, depth, valid, age, quality):
    s=np.asarray(state,np.float32); d=np.asarray(depth,np.float32); v=np.asarray(valid,np.float32)
    if s.shape[-1]!=20 or d.shape!=s.shape[:-1]+(27,) or v.shape!=d.shape:
        raise ValueError('expected 20 vehicle/goal features and 27 depth sectors')
    a=np.broadcast_to(age,s.shape[:-1]); q=np.broadcast_to(quality,s.shape[:-1])
    if not all(np.isfinite(x).all() for x in (s,d,v,a,q)) or np.any(a<0):
        raise ValueError('invalid observation; do not substitute missing estimates silently')
    result=np.concatenate((s,np.clip(d/MAX_DEPTH,0,1),np.clip(v,0,1),
                           np.clip(a/.5,0,1)[...,None],np.clip(q,0,1)[...,None]),axis=-1).astype(np.float32)
    if result.shape[-1] != OBSERVATION_SIZE: raise AssertionError('observation schema mismatch')
    return result

def _slab(origin,direction,half):
    parallel=np.abs(direction)<1e-8; div=np.where(parallel,1.,direction)
    a=(-half-origin)/div; b=(half-origin)/div
    lo=np.where(parallel,-np.inf,np.minimum(a,b)); hi=np.where(parallel,np.inf,np.maximum(a,b))
    outside=parallel & (np.abs(origin)>half)
    return lo,hi,outside

def ray_depth(world):
    """Nearest intersection across every obstacle and wall; no rotor inflation."""
    n=world.n; c=np.cos(world.yaw)[:,None]; s=np.sin(world.yaw)[:,None]
    dx=c-s*RAYS[None,:,1]; dy=s+c*RAYS[None,:,1]
    dz=np.broadcast_to(RAYS[None,:,2],dx.shape)
    best=np.full(dx.shape,np.inf,np.float32)
    for j in range(world.max_obstacles):
        shape=world.scene_shape[:,j]; bar=world.obstacle_horizontal[:,j]
        angle=np.where(shape==0,-world.obstacle_yaw[:,j],world.obstacle_yaw[:,j])
        co=np.cos(angle)[:,None]; si=np.sin(angle)[:,None]
        ox=(world.north-world.obstacle_n[:,j])[:,None]; oy=(world.east-world.obstacle_e[:,j])[:,None]
        oz=(world.altitude-world.obstacle_z[:,j])[:,None]
        # Legacy bars use conservative boxes; scene boxes use exact dimensions.
        hn=np.where(shape==1,world.half_n[:,j],world.obstacle_r[:,j])[:,None]
        he=np.where(shape==1,world.half_e[:,j],world.obstacle_length[:,j]/2+world.obstacle_r[:,j])[:,None]
        hz=np.where(shape==1,world.half_z[:,j],world.obstacle_r[:,j])[:,None]
        slabs=[_slab(co*ox+si*oy,co*dx+si*dy,hn),
               _slab(-si*ox+co*oy,-si*dx+co*dy,he),_slab(oz,dz,hz)]
        enter=np.maximum.reduce([x[0] for x in slabs]); leave=np.minimum.reduce([x[1] for x in slabs])
        hit=(leave>=np.maximum(enter,0)) & ~np.logical_or.reduce([x[2] for x in slabs])
        box=np.where(hit,np.where(enter>0,enter,leave),np.inf)
        radius=world.obstacle_r[:,j,None]; a=dx*dx+dy*dy
        b=ox*dx+oy*dy; cc=ox*ox+oy*oy-radius*radius
        disc=b*b-a*cc; root=np.sqrt(np.maximum(disc,0))
        near=(-b-root)/a; far=(-b+root)/a
        half=np.where(shape==2,world.half_z[:,j],1000.)[:,None]
        zl,zh,zout=_slab(oz,dz,half)
        ent=np.maximum(near,zl); ext=np.minimum(far,zh)
        ok=(disc>=0)&(ext>=np.maximum(ent,0))&~zout
        cylinder=np.where(ok,np.where(ent>0,ent,ext),np.inf)
        distance=np.where(((shape==1)|((shape==0)&bar))[:,None],box,cylinder)
        best=np.minimum(best,distance)
    for position,direction,low,high in ((world.north,dx,-2.875,14.875),
            (world.east,dy,-4.875,4.875),(world.altitude,dz,1.2,5.)):
        denom=np.where(np.abs(direction)>1e-8,direction,np.nan)
        for limit in (low,high):
            t=(limit-position[:,None])/denom
            best=np.minimum(best,np.where(t>0,t,np.inf))
    # Infinity is a no-return, not a claim of free space beyond sensor range.
    valid=np.isfinite(best)&(best>=MIN_DEPTH)&(best<=MAX_DEPTH)
    return np.where(valid,best,0.).astype(np.float32),valid
