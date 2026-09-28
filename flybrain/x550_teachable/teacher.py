"""Privileged training-only teacher; never exported or used as a safety claim."""
import numpy as np
from nextgen.config import MAX_SPEED_MPS


class BrakingTeacher:
    """Augment the preserved lane planner with attainable stopping limits.

    Geometry includes the simulated airframe footprint. Delay, response time and
    current forward velocity bound the speed command before corrective labels are
    returned on learner-visited states.
    """
    def __init__(self, margin=.25, minimum_deceleration=.35):
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('finite nonnegative teacher margin required')
        if not np.isfinite(minimum_deceleration) or minimum_deceleration <= 0:
            raise ValueError('positive teacher deceleration required')
        self.margin = float(margin)
        self.minimum_deceleration = float(minimum_deceleration)

    def labels(self, env, learner_action):
        world = env.world
        action = world.teacher_action().astype(np.float32, copy=True)
        dn = world.obstacle_n - world.north[:, None]
        de = world.obstacle_e - world.east[:, None]
        bearing = np.arctan2(de, dn) - world.yaw[:, None]
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        forward = dn*np.cos(world.yaw[:, None]) + de*np.sin(world.yaw[:, None])
        ahead = (np.abs(bearing) < 1.05) & (forward > 0)
        clearance = np.min(np.where(ahead, world._clearance_matrix(), np.inf), axis=1)
        wall = np.minimum.reduce((world.north+2.875, 14.875-world.north,
                                  world.east+4.875, 4.875-world.east)) - world.footprint
        available = np.maximum(np.minimum(clearance, wall) - self.margin, 0.)
        delay = (world.delay.astype(np.float32)+1.)*.1 + world.response_tau
        deceleration = np.maximum(self.minimum_deceleration,
                                  MAX_SPEED_MPS/np.maximum(world.response_tau, .05))
        speed_cap = deceleration * (np.sqrt(delay*delay + 2*available/deceleration)-delay)
        speed_cap = np.clip(speed_cap, 0., MAX_SPEED_MPS)
        queued = world.delay_buffer[:, np.arange(world.n), 0]
        queued_speed = np.max((queued+1)*.5*MAX_SPEED_MPS, axis=0)
        moving_speed = np.maximum(world.forward, queued_speed)
        stopping = moving_speed*delay + moving_speed*moving_speed/(2*deceleration)
        normalized_cap = 2*speed_cap/MAX_SPEED_MPS - 1
        action[:, 0] = np.minimum(action[:, 0], normalized_cap)
        braking = stopping > available
        action[braking, 0] = -1.
        action = np.clip(action, -1, 1).astype(np.float32)
        observation = env.cache
        healthy = (observation[:, -2] <= .6) & (observation[:, -1] >= .5)
        valid=observation[:,47:74]>0
        depth=np.where(valid,observation[:,20:47]*8.,np.inf)
        observable_near=np.min(depth,axis=1)<2.
        disagreement = np.linalg.norm(action-np.asarray(learner_action),axis=1) > .05
        mask = healthy & disagreement & np.isfinite(available) & (braking|observable_near)
        reason = np.where(mask&braking,2,np.where(mask&observable_near,1,0)).astype(np.int8)
        return action, mask, reason

    def __call__(self, env):
        action, _, _ = self.labels(env, np.zeros((env.n,3),np.float32))
        return action


