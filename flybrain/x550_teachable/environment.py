"""Recurrent observations and a measured curriculum on preserved X550 physics."""
import numpy as np
from x550_scenes.env import World as BaseWorld
from .safety import shield_action
from .sensing import ray_depth,reduce_samples,pack_observation

class CurriculumWorld(BaseWorld):
    def __init__(self,n,seed,horizon,training=False,suite='legacy',level=0,
                 visible_lesson_backdrop=False):
        self.visible_lesson_backdrop=bool(visible_lesson_backdrop)
        self.level=int(level)
        self.curriculum_slots=np.zeros(n,bool)
        self.level_at_episode_start=np.full(n,self.level,np.int16)
        super().__init__(n,seed,horizon,training,suite)

    def reset(self,mask=None):
        obs,priv=super().reset(mask)
        ids=np.arange(self.n) if mask is None else np.flatnonzero(mask)
        self.curriculum_slots[ids]=False
        self.level_at_episode_start[ids]=self.level
        if self.training and not self._reset_depth and self.level<3:
            ids=ids[self.rng.random(len(ids))<.55]
            self.curriculum_slots[ids]=True
            for i in ids:
                self.scene_shape[i]=0; self.obstacle_n[i]=50.; self.obstacle_e[i]=50.
                self.obstacle_horizontal[i]=False; self.goal_north[i]=7.; self.goal_east[i]=0.
                for j in range(self.level):
                    self.obstacle_n[i,j]=3.+2*j; self.obstacle_e[i,j]=(-1)**j*self.rng.uniform(.35,.8)
                    self.obstacle_r[i,j]=self.rng.uniform(.25,.4)
                if getattr(self,'visible_lesson_backdrop',False):
                    # A sensed, collidable wall beyond the goal gives the
                    # fail-closed depth shield evidence without fabricating a
                    # depth return. Evaluation scenes never enable this.
                    self.goal_north[i]=6.
                    self.scene_shape[i,7]=1
                    self.obstacle_n[i,7]=7.8; self.obstacle_e[i,7]=0.
                    self.obstacle_z[i,7]=3.1; self.obstacle_yaw[i,7]=0.
                    self.half_n[i,7]=.125; self.half_e[i,7]=4.875; self.half_z[i,7]=1.9
                    self.obstacle_r[i,7]=np.hypot(.125,4.875)
                    self.obstacle_horizontal[i,7]=True; self.obstacle_length[i,7]=9.75
                self.wind[i]*=.35+.2*self.level
            obs,priv=self.observe()
        return obs,priv

class Env:
    def __init__(self,n=128,seed=550,horizon=400,training=False,suite='legacy',level=0,
                 visible_lesson_backdrop=False):
        self.world=CurriculumWorld(n,seed,horizon,training,suite,level,
                                   visible_lesson_backdrop)
        self.n=n; self.training=training
        self.sensor_rng=np.random.default_rng(seed+170003)
        self.starts=np.ones(n,bool)
        self.legacy,self.privileged=self.world.observe()
        self.cache=self._sense(self.legacy)

    def _sense(self,legacy):
        samples,valid=ray_depth(self.world)
        state=legacy[:,12:32].copy()
        quality=np.ones(self.n,np.float32); age=np.zeros(self.n,np.float32)
        if self.training:
            # Sensor errors are experimental, not calibrated OAK/VIO measurements.
            samples*=self.sensor_rng.normal(1,.015,samples.shape)
            valid&=self.sensor_rng.random(valid.shape)>.015
            state[:,:3]+=self.sensor_rng.normal(0,.025,(self.n,3))
            lost=self.sensor_rng.random(self.n)<.003
            valid[lost]=False; quality[lost]=0.; age[lost]=.4
        depth,confidence=reduce_samples(samples,valid)
        self.shield_obs=legacy.copy(); self.shield_obs[:,12:32]=state
        return pack_observation(state,depth,confidence,age,quality)

    def observe(self):
        return self.cache.copy(),np.concatenate((self.cache,self.privileged[:,-8:]),axis=-1)

    def safe_action(self,proposed):
        safe=shield_action(proposed,self.cache)
        stale=(self.cache[:,-2]>.6)|(self.cache[:,-1]<.5)
        safe[stale]=(-1.,0.,0.)
        return safe

    def step(self,action):
        episode_level=self.world.level_at_episode_start.copy()
        legacy,priv,reward,done,info=self.world.step(action)
        self.legacy=legacy; self.privileged=priv
        final=self._sense(legacy)
        info['final_features']=final.copy()
        info['final_privileged']=np.concatenate((final,priv[:,-8:]),axis=-1)
        info['curriculum_slot']=self.world.curriculum_slots.copy()
        info['scene_family']=self.world.lesson_ids.copy()
        info['level_at_episode_start']=episode_level
        rows=np.arange(self.n)
        info['applied_action']=self.world.delay_buffer[self.world.delay,rows].copy()
        self.world.remember_failures(info['collision'])
        self.cache=final
        if self.training and done.any():
            self.legacy,self.privileged=self.world.reset(done)
            final_shield=self.shield_obs.copy()
            fresh=self._sense(self.legacy)
            self.shield_obs[~done]=final_shield[~done]
            self.cache[done]=fresh[done]
        self.starts=done.copy()
        obs,priv=self.observe()
        return obs,priv,reward,done,info

class Curriculum:
    """Consume disjoint competence windows; duplicate evidence is idempotent."""
    def __init__(self,level=0,min_episodes=128,required_windows=2):
        self.level=int(level); self.streak=0
        self.min_episodes=int(min_episodes); self.required_windows=int(required_windows)
        self.seen_window_ids=set(); self.decisions=[]
    def consider(self,window_id,level,episodes,collisions,completions,timeouts=0):
        values=(episodes,collisions,completions,timeouts)
        if any(type(v) not in (int,np.int32,np.int64) for v in values) \
                or episodes<0 or any(v<0 or v>episodes for v in values[1:]) \
                or collisions+completions+timeouts!=episodes:
            raise ValueError('invalid curriculum evidence accounting')
        if window_id in self.seen_window_ids:
            return False
        self.seen_window_ids.add(window_id)
        eligible=(level==self.level and episodes>=self.min_episodes)
        good=(eligible and collisions/episodes<=.05 and completions/episodes>=.80)
        self.streak=self.streak+1 if good else 0
        advanced=self.streak>=self.required_windows and self.level<3
        self.decisions.append(dict(window_id=window_id,level=level,episodes=episodes,
            collisions=collisions,completions=completions,timeouts=timeouts,
            eligible=eligible,successful=good,advanced=advanced))
        if advanced:
            self.level+=1; self.streak=0
        return advanced
