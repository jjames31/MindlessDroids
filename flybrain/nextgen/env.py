"""Fast vectorized surrogate used for teacher imitation and PPO rollouts.

Gazebo remains the high-fidelity evaluator. This environment deliberately keeps
rendering out of the training loop so thousands of policies can be sampled cheaply.
"""
import json
import math
from pathlib import Path

import numpy as np

from .config import (ACTION_SIZE, MAX_CLIMB_MPS, MAX_SPEED_MPS, MAX_YAW_RPS,
                     PRIVILEGED_SIZE, PROFILE_IDS, PROFILE_PATH, normalize_profiles)
from .features import actor_observation


class VectorFlightEnv:
    def __init__(self, num_envs=512, seed=550, horizon=400, hard_cases=None,
                 autoreset=True, difficulty=1.0, profiles=None):
        self.profile_ids = normalize_profiles(profiles)
        self.spawn_rejections = 0
        self._reset_depth = 0
        self.autoreset = bool(autoreset)
        self.difficulty = float(difficulty)
        if not 0 <= self.difficulty <= 1: raise ValueError("difficulty must be in [0,1]")
        self.n = int(num_envs)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.profile_defs = json.loads(PROFILE_PATH.read_text())["profiles"]
        self.hard_cases = Path(hard_cases) if hard_cases else None
        self.hard_templates = self._load_hard_templates()
        self.max_obstacles = 8
        self.delay_buffer = np.zeros((6, self.n, ACTION_SIZE), np.float32)
        self.reset()

    def _load_hard_templates(self):
        if not self.hard_cases or not self.hard_cases.exists(): return []
        templates=[]
        try: cases=json.loads(self.hard_cases.read_text()).get("cases",[])
        except (OSError,ValueError): return []
        for case in cases:
            try:
                data=case.get("course")
                if data is None:
                    result=str(case.get("result") or "")
                    if not result: continue
                    course=Path(result).with_name("course.json")
                    if not course.exists(): continue
                    data=json.loads(course.read_text())
                templates.append({"goal_north":data["goal_north"],"goal_east":data["goal_east"],
                                  "obstacles":data.get("obstacles",[])})
            except (OSError,ValueError,KeyError): pass
        return templates

    def _sample_range(self, pair, size):
        return self.rng.uniform(float(pair[0]), float(pair[1]), size).astype(np.float32)

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.n, bool)
            self.step_count = np.zeros(self.n, np.int32)
            self.north = np.zeros(self.n, np.float32)
            self.east = np.zeros(self.n, np.float32)
            self.altitude = np.full(self.n, 3., np.float32)
            self.yaw = np.zeros(self.n, np.float32)
            self.forward = np.zeros(self.n, np.float32)
            self.lateral = np.zeros(self.n, np.float32)
            self.vertical = np.zeros(self.n, np.float32)
            self.previous_action = np.zeros((self.n, ACTION_SIZE), np.float32)
            self.previous_action[:,0] = -1.
            selected = np.asarray([PROFILE_IDS.index(p) for p in self.profile_ids])
            self.profile_index = selected[np.arange(self.n) % len(selected)]
            self.obstacle_n = np.zeros((self.n, self.max_obstacles), np.float32)
            self.obstacle_e = np.zeros_like(self.obstacle_n)
            self.obstacle_z = np.zeros_like(self.obstacle_n)
            self.obstacle_r = np.zeros_like(self.obstacle_n)
            self.obstacle_horizontal = np.zeros_like(self.obstacle_n, bool)
            self.obstacle_length = np.zeros_like(self.obstacle_n)
            self.obstacle_yaw = np.zeros_like(self.obstacle_n)
            self.delay = np.zeros(self.n, np.int32)
            self.response_tau = np.zeros(self.n, np.float32)
            self.wind = np.zeros(self.n, np.float32)
            self.sensor_noise = np.zeros(self.n, np.float32)
            self.mass = np.zeros(self.n, np.float32)
            self.footprint = np.zeros(self.n, np.float32)
            self.vertical_radius = np.zeros(self.n, np.float32)
            self.propulsion = np.zeros(self.n, np.float32)
            self.max_pitch = np.zeros(self.n, np.float32)
            self.camera_fov = np.zeros(self.n, np.float32)
            self.goal_north = np.zeros(self.n, np.float32)
            self.goal_east = np.zeros(self.n, np.float32)
        ids = np.flatnonzero(mask)
        if not len(ids):
            return self.observe()
        # Each vector slot owns one profile, keeping every simultaneous batch balanced.
        self.north[ids] = 0.; self.east[ids] = self.rng.uniform(-.35, .35, len(ids))
        self.altitude[ids] = self.rng.uniform(2.6, 3.4, len(ids))
        self.yaw[ids] = self.rng.uniform(-.15, .15, len(ids))
        self.forward[ids] = 0.; self.lateral[ids] = 0.; self.vertical[ids] = 0.
        self.previous_action[ids] = (-1.,0.,0.); self.step_count[ids] = 0
        self.goal_north[ids] = self.rng.uniform(7., 12., len(ids))
        self.goal_east[ids] = self.rng.uniform(-2.5, 2.5, len(ids))
        self.delay[ids] = self.rng.integers(0, 5, len(ids))
        self.response_tau[ids] = self.rng.uniform(.12, .65, len(ids))
        self.wind[ids] = self.rng.uniform(-.25, .25, len(ids))
        self.sensor_noise[ids] = self.rng.uniform(0., .035, len(ids))
        for profile_i, profile_id in enumerate(PROFILE_IDS):
            chosen = ids[self.profile_index[ids] == profile_i]
            if not len(chosen): continue
            p = self.profile_defs[profile_id]
            self.mass[chosen] = self._sample_range(p["mass_kg"], len(chosen))
            self.footprint[chosen] = self._sample_range(p["rotor_footprint_radius_m"], len(chosen))
            self.vertical_radius[chosen] = self._sample_range(p["vertical_clearance_radius_m"], len(chosen))
            self.propulsion[chosen] = self._sample_range(p["propulsion_efficiency"], len(chosen))
            self.max_pitch[chosen] = self._sample_range(p["maximum_pitch_deg"], len(chosen))
            self.camera_fov[chosen] = self._sample_range(p["camera_horizontal_fov_rad"], len(chosen))
        self.obstacle_n[ids] = 50.; self.obstacle_e[ids] = 50.; self.obstacle_z[ids] = 3.
        self.obstacle_r[ids] = .4; self.obstacle_horizontal[ids] = False
        self.obstacle_yaw[ids] = 0.
        self.obstacle_length[ids] = 5.
        counts = self.rng.integers(2, self.max_obstacles+1, len(ids))
        for row, count in zip(ids, counts):
            ns = np.linspace(2., self.goal_north[row]-1., count)
            self.obstacle_n[row, :count] = ns + self.rng.uniform(-.35, .35, count)
            self.obstacle_e[row, :count] = self.rng.uniform(-2.8, 2.8, count)
            self.obstacle_r[row, :count] = self.rng.uniform(.3, .75, count)
            horizontal = self.rng.random(count) < .28
            self.obstacle_horizontal[row, :count] = horizontal
            self.obstacle_z[row, :count] = np.where(horizontal,
                self.rng.uniform(1.8, 4.2, count), 3.)
            self.obstacle_length[row, :count] = np.where(horizontal,
                self.rng.uniform(1.2, 3.4, count), 5.)
            self.obstacle_yaw[row, :count] = self.rng.uniform(-math.pi,math.pi,count)
        # Replay exact layouts from recent Gazebo collisions in 30% of resets.
        if self.hard_templates:
            replay=ids[self.rng.random(len(ids))<.30]
            for row in replay:
                template=self.hard_templates[int(self.rng.integers(0,len(self.hard_templates)))]
                self.goal_north[row]=template["goal_north"]; self.goal_east[row]=template["goal_east"]
                self.obstacle_n[row]=50.; self.obstacle_e[row]=50.
                for j,o in enumerate(template["obstacles"][:self.max_obstacles]):
                    self.obstacle_n[row,j]=o["north"]; self.obstacle_e[row,j]=o["east"]
                    self.obstacle_z[row,j]=o.get("altitude",3.); self.obstacle_r[row,j]=o.get("radius",.5)
                    self.obstacle_horizontal[row,j]=o.get("orientation")=="horizontal"
                    self.obstacle_length[row,j]=o.get("length",5.)
                    self.obstacle_yaw[row,j]=o.get("yaw",0.)
        # Easier episodes are curriculum anchors, never certification episodes.
        if self.difficulty < 1.:
            d = self.difficulty
            self.goal_north[ids] *= .6 + .4*d
            self.wind[ids] *= d
            self.sensor_noise[ids] *= d
            self.delay[ids] = np.rint(self.delay[ids]*d).astype(np.int32)
            self.response_tau[ids] = .2 + d*(self.response_tau[ids]-.2)
            keep = max(0, int(np.ceil(self.max_obstacles*d)))
            self.obstacle_n[ids, keep:] = 50.
            self.obstacle_e[ids, keep:] = 50.
            if d < .5: self.obstacle_horizontal[ids] = False
        # A valid trial must begin outside every obstacle with a small clearance
        # margin. Reject initial scene draws, never completed flight outcomes.
        # Resampling the entire slot also preserves exact hard-course geometry.
        invalid_spawn = mask & (self._geometry()[0] < .10)
        if np.any(invalid_spawn):
            if self._reset_depth >= 16:
                raise RuntimeError("could not generate a collision-free initial pose")
            self.spawn_rejections += int(invalid_spawn.sum())
            self._reset_depth += 1
            try:
                self.reset(invalid_spawn)
            finally:
                self._reset_depth -= 1
        self.delay_buffer[:, ids] = 0.
        self.delay_buffer[:, ids, 0] = -1.
        return self.observe()

    def _clearance_matrix(self):
        dn = self.obstacle_n-self.north[:, None]
        de = self.obstacle_e-self.east[:, None]
        along=de*np.cos(self.obstacle_yaw)+dn*np.sin(self.obstacle_yaw)
        along=np.clip(along,-self.obstacle_length/2,self.obstacle_length/2)
        closest_de=de-along*np.cos(self.obstacle_yaw)
        closest_dn=dn-along*np.sin(self.obstacle_yaw)
        horizontal_distance = np.sqrt(closest_dn*closest_dn+closest_de*closest_de)
        vertical_distance = np.abs(self.obstacle_z-self.altitude[:, None])
        vertical_clear = self.obstacle_r+self.vertical_radius[:, None]
        planar_clear = self.obstacle_r+self.footprint[:, None]
        horizontal_bar_clearance = np.sqrt(
            (horizontal_distance/np.maximum(planar_clear, .01))**2+
            (vertical_distance/np.maximum(vertical_clear, .01))**2)-1.
        return np.where(self.obstacle_horizontal,
            horizontal_bar_clearance*np.minimum(planar_clear, vertical_clear),
            np.sqrt(dn*dn+de*de)-planar_clear)

    def _geometry(self):
        dn = self.obstacle_n-self.north[:, None]
        de = self.obstacle_e-self.east[:, None]
        clearance=self._clearance_matrix()
        nearest = np.argmin(clearance, axis=1)
        rows = np.arange(self.n)
        return clearance[rows, nearest], dn[rows, nearest], de[rows, nearest], nearest

    def observe(self):
        clearance, dn, de, nearest = self._geometry()
        rows = np.arange(self.n)
        bearing = np.arctan2(de, dn)-self.yaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        # Match MAVLink ground velocity expressed in the vehicle heading frame.
        forward_ground = self.forward + self.wind*np.sin(self.yaw)
        lateral_ground = self.wind*np.cos(self.yaw)
        closing = np.maximum(0., forward_ground*np.cos(bearing) +
                             lateral_ground*np.sin(bearing))
        looming = np.clip(closing/np.maximum(clearance+.45, .08)*.35, 0., 1.)
        left = looming*(bearing < 0); right = looming*(bearing >= 0)
        radial = looming
        vertical_bias = np.where(self.obstacle_horizontal[rows, nearest],
            np.clip((self.obstacle_z[rows, nearest]-self.altitude)/1.2, -1., 1.), 0.)
        noise = self.rng.normal(0, self.sensor_noise, (5, self.n)).astype(np.float32)
        # Calibrated against retained Gazebo telemetry: raw EMD channels are near
        # 1e-2, radial confidence is usually below .55, and BANC rates include a
        # small asymmetric baseline. Keeping these ranges aligned is essential for
        # transferring a surrogate-trained actor into the rendered simulator.
        visual_left=np.maximum(0,.014*left+.05*noise[0])
        visual_right=np.maximum(0,.014*right+.05*noise[1])
        visual_wide=.10*noise[2]
        radial_sensor=np.clip(.55*radial+noise[3],0,.65)
        vertical_sensor=np.clip(vertical_bias+3*noise[4],-1,1)
        banc_looming=.25*looming
        banc_left=np.clip(.06+.06*left-.02*right,-.12,.14)
        banc_right=np.clip(.06*right-.02*left,-.08,.12)
        banc_power=np.clip(.065-.025*looming,0,.11)
        goal_distance = np.hypot(self.goal_north-self.north, self.goal_east-self.east)
        goal_heading=np.arctan2(self.goal_east-self.east,self.goal_north-self.north)-self.yaw
        goal_heading=np.arctan2(np.sin(goal_heading),np.cos(goal_heading))
        # Vectorized equivalent of features.actor_observation; this is the hottest
        # training path and avoids thousands of Python calls per simulator step.
        obs=np.stack([
            visual_left*8,visual_right*8,visual_wide*8,radial_sensor*2,vertical_sensor*2,
            banc_left,banc_right,banc_power,banc_looming*2,
            (.25*left)*2,(.25*right)*2,vertical_bias*banc_looming*2,
            forward_ground,lateral_ground,self.vertical,np.sin(goal_heading),np.cos(goal_heading),
            goal_distance*.08,self.altitude*.25,np.sin(self.yaw),np.cos(self.yaw),
            np.zeros(self.n),np.zeros(self.n),self.mass*.25,self.footprint*1.5,
            self.vertical_radius*3,self.propulsion,self.max_pitch*.04,self.camera_fov*.5,
            self.previous_action[:,0]+1,self.previous_action[:,1]*1.2,self.previous_action[:,2],
        ],axis=1).astype(np.float32)
        obs=np.clip(obs,-5,5)
        privileged = np.concatenate([obs, np.stack([
            clearance, dn*.1, de*.1, goal_distance*.1,
            self.response_tau, self.delay/5., self.wind, self.step_count/self.horizon,
        ], axis=1).astype(np.float32)], axis=1)
        assert privileged.shape[1] == PRIVILEGED_SIZE
        return obs, privileged

    def teacher_action(self):
        clearance, dn_near, de_near, nearest = self._geometry()
        goal_dn=self.goal_north-self.north; goal_de=self.goal_east-self.east
        dn=self.obstacle_n-self.north[:,None]; de=self.obstacle_e-self.east[:,None]
        planar=np.maximum(np.hypot(dn,de),.05)
        planar_clear=planar-self.obstacle_r-self.footprint[:,None]
        # Privileged lane planner: aim beyond the next pillar through a clearance
        # lane selected from both sides. This avoids the oscillation of a purely
        # reactive potential field and gives the student consistent demonstrations.
        upcoming=(~self.obstacle_horizontal)&(dn>-.1)&(dn<4.5)
        next_index=np.argmin(np.where(upcoming,dn,99.),axis=1); rows=np.arange(self.n)
        has_next=np.any(upcoming,axis=1)
        radius=self.obstacle_r[rows,next_index]+self.footprint+.65
        center_e=self.obstacle_e[rows,next_index]
        option_left=np.clip(center_e-radius,-4.0,4.0)
        option_right=np.clip(center_e+radius,-4.0,4.0)
        bypass=np.where(np.abs(option_left-self.goal_east)<np.abs(option_right-self.goal_east),
                        option_left,option_right)
        blocks_lane=has_next & (np.abs(self.east-center_e)<radius+.25)
        target_n=np.where(blocks_lane,self.obstacle_n[rows,next_index]+.8,self.goal_north)
        target_e=np.where(blocks_lane,bypass,self.goal_east)
        desired_heading=np.arctan2(target_e-self.east,target_n-self.north)
        heading_error=np.arctan2(np.sin(desired_heading-self.yaw),np.cos(desired_heading-self.yaw))
        yaw=np.clip(1.5*heading_error,-1,1)
        bearings=np.arctan2(de,dn)-self.yaw[:,None]
        bearings=np.arctan2(np.sin(bearings),np.cos(bearings))
        forward=dn*np.cos(self.yaw[:,None])+de*np.sin(self.yaw[:,None])
        ahead=(np.abs(bearings)<1.05)&(forward>0)
        front_clear=np.min(np.where(ahead,self._clearance_matrix(),99.),axis=1)
        speed=np.clip((front_clear-.2)/1.8,.03,.9)*(1-.62*np.abs(yaw))
        speed=np.where(front_clear<1.1,np.minimum(speed,.10),speed)
        threat=np.clip((2.6-front_clear)/2.6,0,1)
        rows=np.arange(self.n); horizontal=self.obstacle_horizontal[rows,nearest]
        bar_z=self.obstacle_z[rows,nearest]
        climb=np.where(horizontal & (clearance<2.2),np.where(bar_z<3.,.85,-.85),0.)
        altitude_center=np.clip((3.-self.altitude)*.4,-.35,.35)
        climb=np.clip(climb+altitude_center,-1,1)
        return np.stack([2*speed-1,yaw,climb],axis=1).astype(np.float32)

    def step(self, action, dt=.1):
        action=np.clip(np.asarray(action,np.float32),-1,1)
        self.delay_buffer[1:] = self.delay_buffer[:-1]
        self.delay_buffer[0] = action
        delayed=self.delay_buffer[self.delay, np.arange(self.n)].copy()
        speed=(delayed[:,0]+1)*.5*MAX_SPEED_MPS
        yaw_rate=delayed[:,1]*MAX_YAW_RPS
        climb=delayed[:,2]*MAX_CLIMB_MPS
        alpha=1-np.exp(-dt/np.maximum(self.response_tau,.03))
        self.forward += alpha*(speed-self.forward)
        self.vertical += alpha*(climb-self.vertical)
        self.yaw += yaw_rate*dt
        old_distance=np.hypot(self.goal_north-self.north,self.goal_east-self.east)
        self.north += self.forward*np.cos(self.yaw)*dt
        self.east += (self.forward*np.sin(self.yaw)+self.wind)*dt
        self.altitude += self.vertical*dt
        self.step_count += 1
        clearance, _, _, _ = self._geometry()
        wall=np.minimum.reduce([self.north+2.875,14.875-self.north,
                                self.east+4.875,4.875-self.east]) - self.footprint
        vertical_wall=np.minimum(self.altitude-1.2-self.vertical_radius,
                                 5.-self.altitude-self.vertical_radius)
        clearance=np.minimum(clearance,np.minimum(wall,vertical_wall))
        distance=np.hypot(self.goal_north-self.north,self.goal_east-self.east)
        collision=clearance <= 0
        completed=(distance <= .75) & ~collision
        timeout=self.step_count >= self.horizon
        done=collision|completed|timeout
        progress=(old_distance-distance)*8.
        near=np.clip((.8-clearance)/.8,0,1)
        smooth=np.sum((action-self.previous_action)**2,axis=1)
        reward=progress-.8*near-.03*smooth-.01
        reward += completed*25.-collision*25.
        self.previous_action=action.copy()
        final_obs, final_privileged = self.observe()
        info=dict(collision=collision.copy(),completed=completed.copy(),
                  terminated=(collision|completed).copy(),
                  truncated=(timeout & ~(collision|completed)).copy(),
                  profile_index=self.profile_index.copy(),clearance=clearance.copy(),
                  distance=distance.copy(), progress_m=(old_distance-distance).copy(),
                  action_change=smooth.copy(),
                  final_observation=final_obs, final_privileged=final_privileged)
        if self.autoreset and np.any(done):
            self.reset(done)
            obs,privileged=self.observe()
        else:
            obs,privileged=final_obs,final_privileged
        return obs,privileged,reward.astype(np.float32),done,info
