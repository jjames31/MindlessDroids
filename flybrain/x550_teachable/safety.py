"""Sensor-consistent independent navigation shield for the recurrent stack."""
import numpy as np
from nextgen.config import MAX_CLIMB_MPS, MAX_SPEED_MPS, MAX_YAW_RPS
from .schema import AGE_INDEX, DEPTH_SLICE, QUALITY_INDEX, VALIDITY_SLICE, validate_observation


def shield_action(action, observation):
    a = np.asarray(action, np.float32)
    o = validate_observation(observation)
    if a.shape != o.shape[:-1]+(3,) or not np.isfinite(a).all():
        raise ValueError('shield requires finite three-axis actions')
    a = np.clip(a, -1, 1)
    depth = o[..., DEPTH_SLICE]*8.
    valid = o[..., VALIDITY_SLICE] > 0
    # Central three columns in every row are the forward collision corridor.
    corridor = np.asarray([r*9+c for r in range(3) for c in range(3, 6)])
    corridor_coverage=o[...,VALIDITY_SLICE][...,corridor]
    coverage_ok=np.all(corridor_coverage>=.5,axis=-1)
    front = np.min(np.where(valid[..., corridor], depth[..., corridor], np.inf), axis=-1)
    unhealthy = (o[..., AGE_INDEX] > .6) | (o[..., QUALITY_INDEX] < .5)
    no_return = ~np.isfinite(front)|~coverage_ok
    speed = (a[..., 0]+1)*.5*MAX_SPEED_MPS
    yaw = a[..., 1]*MAX_YAW_RPS
    climb = a[..., 2]*MAX_CLIMB_MPS
    footprint=np.maximum(o[...,12]/1.5,.1)
    forward_velocity=np.maximum(o[...,0],0.)
    stopping_distance=forward_velocity*(.7+1.0)+footprint+.35
    clearance=front-footprint
    speed=np.where(clearance<=stopping_distance,0.,speed)
    speed = np.where(front < 2.0, np.minimum(speed, np.maximum(0., (front-.5)/2.0)), speed)
    speed = np.where(front < .8, 0., speed)
    yaw = np.where((front < .8) & (np.abs(yaw) < .35),
                   np.where(a[..., 1] >= 0, .35, -.35), yaw)
    altitude = o[..., 6]*4.
    climb = np.where(altitude < 1.45, np.maximum(0, climb), climb)
    climb = np.where(altitude > 4.75, np.minimum(0, climb), climb)
    inhibited=unhealthy|no_return
    speed = np.where(inhibited, 0., speed)
    yaw=np.where(inhibited,0.,yaw); climb=np.where(inhibited,0.,climb)
    return np.stack((2*speed/MAX_SPEED_MPS-1, yaw/MAX_YAW_RPS,
                     climb/MAX_CLIMB_MPS), axis=-1).astype(np.float32)
