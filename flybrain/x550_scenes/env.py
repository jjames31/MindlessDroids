"""X550-only courses with training-only lessons/replay and finite sensor history."""
import copy
import numpy as np
from nextgen.env import VectorFlightEnv
from .model import FRAMES, INPUT_SIZE
from .control import flight_action

LAYOUT_FIELDS=('obstacle_n','obstacle_e','obstacle_z','obstacle_r','obstacle_horizontal',
               'obstacle_length','obstacle_yaw','goal_north','goal_east')


from .scene_world import World as SceneWorld

class World(SceneWorld):
    def step(self,action,dt=.1):
        proposed=np.asarray(action,np.float32).copy()
        prior=self.previous_action.copy()
        executed=flight_action(proposed,self.altitude,self.vertical_radius)
        obs,priv,reward,done,info=super().step(executed,dt)
        # Runtime supplies previous policy output, before its altitude-hold adapter.
        self.previous_action=proposed
        obs[:,29:32]=np.column_stack((proposed[:,0]+1,proposed[:,1]*1.2,proposed[:,2]))
        priv[:,:32]=obs
        proposed_change=np.sum((proposed-prior)**2,axis=1)
        reward+=.03*(info['action_change']-proposed_change)
        info['action_change']=proposed_change
        info['final_observation']=obs.copy(); info['final_privileged']=priv.copy()
        info['executed_action']=executed
        return obs,priv,reward,done,info



class HistoryEnv:
    def __init__(self,n=1024,seed=550,horizon=400,training=False,suite="legacy"):
        self.world=World(n,seed,horizon,training,suite=suite)
        self.n=n; self.profile_ids=('x550',); self.training=training
        obs,_=self.world.observe()
        self.history=np.repeat(obs[:,None,:],FRAMES,axis=1)
        self.priv=self._privileged()

    def _privileged(self,priv=None):
        if priv is None: _,priv=self.world.observe()
        return np.concatenate((self.history.reshape(self.n,INPUT_SIZE),priv[:,-8:]),axis=1)

    def observe(self):
        return self.history.reshape(self.n,INPUT_SIZE).copy(),self.priv.copy()

    def step(self,action):
        obs,priv,reward,done,info=self.world.step(action)
        self.history[:,1:]=self.history[:,:-1].copy()
        self.history[:,0]=obs
        final_obs=self.history.reshape(self.n,INPUT_SIZE).copy()
        final_priv=self._privileged(priv)
        info['final_observation']=final_obs; info['final_privileged']=final_priv
        info['lesson_id']=self.world.lesson_ids.copy()
        self.world.remember_failures(info['collision'])
        if self.training and done.any():
            new_obs,new_priv=self.world.reset(done)
            self.history[done]=np.repeat(new_obs[done,None,:],FRAMES,axis=1)
            self.priv=self._privileged(new_priv)
        else:
            self.priv=final_priv
        return self.history.reshape(self.n,INPUT_SIZE).copy(),self.priv.copy(),reward,done,info
