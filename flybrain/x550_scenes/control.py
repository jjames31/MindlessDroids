"""Match the existing Gazebo adapter's deterministic vertical post-processing."""
import numpy as np


def flight_action(action,altitude,vertical_radius):
    executed=np.asarray(action,np.float32).copy()
    lower=1.2+vertical_radius+.18; upper=5.-vertical_radius-.18
    climb=executed[:,2].copy()
    climb=np.where(climb<0,climb*np.clip((altitude-lower)/.65,0,1),
                   climb*np.clip((upper-altitude)/.65,0,1))
    climb=np.clip(climb,-1.,1.)  # BASELINE altitude_hold_gain=0, vertical_speed_limit=1
    climb=np.where(altitude<=lower,np.maximum(0,climb),climb)
    climb=np.where(altitude>=upper,np.minimum(0,climb),climb)
    executed[:,2]=climb
    return executed
