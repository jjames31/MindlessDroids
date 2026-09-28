"""Regressions for reset, invalid-observation and launch evidence edge cases."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from x550_teachable.environment import Env
from x550_teachable.evaluate import evaluate
from x550_teachable.learning import Config, Engine, unroll
from x550_teachable.model import HIDDEN_SIZE, Model, OBS_SIZE
from x550_teachable.train import session_deadline


class ResetEdgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(921)
        torch.set_num_threads(1)

    def test_invalid_episode_start_clears_previous_memory(self):
        actor=Model().actor
        obs=torch.randn(3,2,OBS_SIZE)
        starts=torch.tensor([[False,False],[True,False],[False,False]])
        valid=torch.tensor([[True,True],[False,True],[True,True]])
        logits,memories,_=unroll(actor,obs,torch.randn(2,HIDDEN_SIZE),starts,valid)
        self.assertEqual(float(memories[1,0].detach().abs().max()),0.)
        expected,_=actor(obs[2,:1],torch.zeros(1,HIDDEN_SIZE),torch.zeros(1,dtype=torch.bool))
        torch.testing.assert_close(logits[2,:1],expected,atol=1e-6,rtol=0)

    def test_wall_budget_is_relative_to_each_launch(self):
        self.assertEqual(session_deadline(2.,started=1000.),8200.)
        self.assertEqual(session_deadline(.5,started=9000.),10800.)

    def test_collector_invalid_episode_start_clears_previous_memory(self):
        engine=Engine(Model(),Env(2,922,horizon=30),Config(rollout=1,sequence=1,
            batch_sequences=1,epochs=1))
        engine.hidden=torch.randn_like(engine.hidden)
        engine.starts[0]=True;engine.obs[0,-1]=0
        engine.collect()
        self.assertEqual(float(engine.hidden[0].abs().max()),0.)

    def test_all_invalid_batch_skips_optimizer(self):
        engine=Engine(Model(),Env(2,923,horizon=2,training=True),
            Config(rollout=2,sequence=2,batch_sequences=2,epochs=1))
        data,histories,_=engine.collect();data['policy_valid'][:]=False
        before=copy.deepcopy(engine.model.state_dict());result=engine.optimize(data,histories)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['skipped_invalid_batches'],1)
        for name,value in before.items():
            torch.testing.assert_close(engine.model.state_dict()[name],value,atol=0,rtol=0)

    def test_scenario_identity_ignores_finish_order(self):
        reverse=[False]
        class FixtureEnv:
            def __init__(self,n,*args,**kwargs):
                self.world=SimpleNamespace(horizon=2);self.t=0
            def observe(self): return np.zeros((2,OBS_SIZE),np.float32),None
            def safe_action(self,action): return action
            def step(self,action):
                self.t+=1
                done=np.ones(2,bool) if self.t==2 else np.array([not reverse[0],reverse[0]])
                info=dict(clearance=np.ones(2),progress_m=np.zeros(2),
                    collision=np.zeros(2,bool),completed=done,truncated=np.zeros(2,bool),
                    scene_family=np.zeros(2,int))
                return self.observe()[0],None,None,done,info
        with patch('x550_teachable.evaluate.Env',FixtureEnv):
            first=evaluate(Model(),2,2,925,horizon=2);reverse[0]=True
            second=evaluate(Model(),2,2,925,horizon=2)
        self.assertEqual(first['identity']['scenario_set_sha256'],
                         second['identity']['scenario_set_sha256'])

    def test_backdrop_never_changes_evaluation_scenes(self):
        for suite in ('legacy','realistic'):
            first=Env(8,927,training=False,suite=suite)
            second=Env(8,927,training=False,suite=suite,visible_lesson_backdrop=True)
            np.testing.assert_array_equal(first.observe()[0],second.observe()[0])
            np.testing.assert_array_equal(first.world._clearance_matrix(),
                                          second.world._clearance_matrix())

    def test_lesson_backdrop_is_sensed_and_collidable(self):
        env=Env(32,929,training=True,level=0,visible_lesson_backdrop=True)
        slots=np.flatnonzero(env.world.curriculum_slots)
        self.assertGreater(len(slots),0)
        from x550_teachable.sensing import ray_depth,reduce_samples
        samples,valid=ray_depth(env.world);_,coverage=reduce_samples(samples,valid)
        self.assertTrue((coverage[slots,13]>.5).all())
        env.world.north[slots]=env.world.obstacle_n[slots,7]
        self.assertTrue((env.world._clearance_matrix()[slots,7]<0).all())
