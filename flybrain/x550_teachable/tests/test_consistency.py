"""Offline consistency checks for the isolated recurrent PPO engine."""
import copy
import unittest
import numpy as np
import torch

from x550_teachable.environment import Env
from x550_teachable.learning import (Config, Engine, advantages, auxiliary_targets,
                                     burn_in, sequence_batch, unroll)
from x550_teachable.model import HIDDEN_SIZE, Model, OBS_SIZE
from x550_teachable.teacher import BrakingTeacher
from x550_teachable.safety import shield_action as sensor_shield
from x550_teachable.evaluate import improves


class RecurrentConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5501)
        np.random.seed(5501)

    def config(self):
        return Config(rollout=4, sequence=2, batch_sequences=2, epochs=1)

    def test_truncation_bootstraps_but_termination_does_not(self):
        reward = np.zeros((1, 2), np.float32)
        value = np.zeros_like(reward)
        next_value = np.full_like(reward, 3.)
        terminated = np.array([[False, True]])
        done = np.ones((1, 2), bool)
        adv, _ = advantages(reward, value, next_value, terminated, done, gamma=.5)
        np.testing.assert_allclose(adv, [[1.5, 0.]])

    def test_padding_does_not_update_hidden_state(self):
        actor = Model().actor
        obs = torch.randn(3, 2, OBS_SIZE)
        hidden = torch.zeros(2, HIDDEN_SIZE)
        starts = torch.zeros(3, 2, dtype=torch.bool)
        valid = torch.tensor([[True, True], [True, False], [True, False]])
        _, memories, final = unroll(actor, obs, hidden, starts, valid)
        torch.testing.assert_close(final[1], memories[0, 1])

    def test_sequence_batch_preserves_order_and_padding(self):
        data = {'x': np.arange(12).reshape(6, 2)}
        batch = sequence_batch(data, [(1, 4, 0), (4, 6, 1)], 'cpu')
        np.testing.assert_array_equal(batch['x'][:, 0], [2, 4, 6])
        np.testing.assert_array_equal(batch['x'][:, 1], [9, 11, 0])
        np.testing.assert_array_equal(batch['valid'][:, 1], [True, True, False])

    def test_update_is_finite_and_uses_auxiliary_masks(self):
        engine = Engine(Model(), Env(4, 78, horizon=8, training=True), self.config())
        metrics = engine.update()
        self.assertGreater(metrics['optimizer_steps'], 0)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()
                            if isinstance(v, (float, np.floating))))
        self.assertLess(metrics['behavior_logp_max_error'], 2e-6)

    def test_motion_target_uses_velocity_not_airframe_descriptors(self):
        current=np.zeros((1,76),np.float32); final=current.copy()
        current[:,-1]=final[:,-1]=1
        current[:,47:74]=final[:,47:74]=1; current[:,20:47]=final[:,20:47]=.5
        final[0,:3]=(.6,-.2,.1); final[0,12:15]=(9,8,7)
        target,mask=auxiliary_targets(current,final,{'collision':np.array([False])})
        np.testing.assert_allclose(target[0,:3],[.6,-.2,.1],atol=1e-7)
        self.assertTrue(mask[0,:3].all())
        changed=final.copy(); changed[0,12:15]=(-5,-4,-3)
        target2,_=auxiliary_targets(current,changed,{'collision':np.array([False])})
        np.testing.assert_array_equal(target2[0,:3],target[0,:3])

    def test_collision_completion_and_reset_do_not_mix_targets(self):
        env=Env(2,781,horizon=3,training=True); world=env.world
        world.goal_north[0]=world.north[0]; world.goal_east[0]=world.east[0]
        world.obstacle_n[1,0]=world.north[1]; world.obstacle_e[1,0]=world.east[1]
        world.obstacle_z[1,0]=world.altitude[1]; world.obstacle_r[1,0]=1
        world.obstacle_horizontal[1,0]=False
        before,_=env.observe()
        returned,_,_,done,info=env.step(np.tile([-1,0,0],(2,1)))
        self.assertTrue(done.all()); self.assertTrue(info['completed'][0])
        self.assertTrue(info['collision'][1])
        target,_=auxiliary_targets(before,info['final_features'],info)
        np.testing.assert_array_equal(target[:,:3],info['final_features'][:,:3])
        self.assertGreater(np.max(np.abs(returned-info['final_features'])),0)

    def test_updated_actor_memory_is_reconstructed(self):
        engine=Engine(Model(),Env(4,780,horizon=30,training=True),self.config())
        engine.update()
        expected=burn_in(engine.model.actor,engine.histories,engine.device)
        torch.testing.assert_close(engine.hidden,expected,atol=0,rtol=0)

    def test_unhealthy_training_frame_does_not_advance_memory(self):
        engine=Engine(Model(),Env(2,782,horizon=30),Config(rollout=1,sequence=1,
            batch_sequences=1,epochs=1))
        engine.hidden=torch.randn_like(engine.hidden);before=engine.hidden.clone()
        engine.starts[:]=False
        engine.obs[0,-1]=0
        data,_,_=engine.collect()
        torch.testing.assert_close(engine.hidden[0],before[0],atol=0,rtol=0)
        self.assertFalse(data['policy_valid'][0,0])

    def test_behavior_reconstruction_skips_invalid_midrollout_frames(self):
        engine=Engine(Model(),Env(2,7821,horizon=30),Config(rollout=3,sequence=1,
            batch_sequences=1,epochs=1))
        original=engine.env.step;calls={'n':0}
        def step(action):
            result=original(action);calls['n']+=1
            if calls['n']==1: result[0][0,-1]=0
            return result
        engine.env.step=step
        data,histories,_=engine.collect()
        self.assertFalse(data['policy_valid'][1,0])
        self.assertLess(engine.behavior_logp_error(data,histories),2e-6)

    def test_behavior_mismatch_aborts_before_optimization(self):
        engine=Engine(Model(),Env(2,783,horizon=30),Config(rollout=1,sequence=1,
            batch_sequences=1,epochs=1))
        before=copy.deepcopy(engine.model.state_dict())
        engine.behavior_logp_error=lambda data,histories:1.
        with self.assertRaises(FloatingPointError):engine.update()
        for key,value in before.items():torch.testing.assert_close(engine.model.state_dict()[key],value)

    def test_promotion_requires_complete_matched_valid_evidence(self):
        def reports(seed=1,count=4):
            rows=[]
            for i in range(count):
                records=[{'scenario_id':f'{i}:{j}'} for j in range(10)]
                rows.append({'identity':{'panel_id':f'panel{i}','seed':seed+i,'suite':'legacy',
                    'episodes':10,'horizon':400,'observation_schema':'x550-recurrent-depth-v2',
                    'shield_version':'v','scenario_set_sha256':'hash'+str(i)},
                    'metrics':{'episodes':10,'collisions':1,'completions':8,'timeouts':1},
                    'episodes':records})
            return rows
        self.assertTrue(improves(reports(),None))
        with self.assertRaises(ValueError):improves(reports(count=3),reports())
        candidate=reports(seed=9)
        with self.assertRaises(ValueError):improves(candidate,reports())
        bad=reports();bad[0]['metrics']['collisions']=-1
        with self.assertRaises(ValueError):improves(bad,None)

    def test_gru_learns_delayed_memory_task(self):
        torch.manual_seed(900)
        actor=Model().actor
        optimizer=torch.optim.Adam(actor.parameters(),lr=.01)
        observations=torch.zeros(5,32,OBS_SIZE)
        cue=torch.where(torch.arange(32)%2==0,1.,-1.)
        observations[0,:,0]=cue
        starts=torch.zeros(5,32,dtype=torch.bool); starts[0]=True
        def loss():
            logits,_,_=unroll(actor,observations,torch.zeros(32,HIDDEN_SIZE),starts)
            return (torch.tanh(logits[-1,:,0])-cue).square().mean()
        initial=float(loss().detach())
        for _ in range(80):
            value=loss(); optimizer.zero_grad(); value.backward(); optimizer.step()
        self.assertLess(float(loss().detach()),initial*.1)

    def test_checkpoint_restores_exact_next_rollout(self):
        first = Engine(Model(), Env(4, 79, horizon=8, training=True), self.config(), seed=80)
        first.update()
        state = first.state_dict()
        expected, _, _ = first.collect()
        second = Engine(Model(), Env(4, 999, horizon=8, training=True), self.config(), seed=1)
        second.load_state_dict(copy.deepcopy(state))
        actual, _, _ = second.collect()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_checkpoint_restores_exact_next_optimization(self):
        first = Engine(Model(), Env(4, 790, horizon=8, training=True), self.config(), seed=791)
        first.update()
        state = first.state_dict()
        first.update()
        expected = copy.deepcopy(first.model.state_dict())
        second = Engine(Model(), Env(4, 999, horizon=8, training=True), self.config(), seed=1)
        second.load_state_dict(copy.deepcopy(state))
        second.update()
        for key, value in expected.items():
            torch.testing.assert_close(second.model.state_dict()[key], value, atol=0, rtol=0)

    def test_checkpoint_rejects_config_mismatch(self):
        first = Engine(Model(), Env(2, 81), self.config())
        state = first.state_dict()
        other = Engine(Model(), Env(2, 82), Config(rollout=5, sequence=2,
                                                  batch_sequences=2, epochs=1))
        with self.assertRaises(ValueError):
            other.load_state_dict(state)

    def test_training_teacher_is_bounded_and_brakes_for_clearance(self):
        env = Env(2, 83)
        world = env.world
        world.north[:] = 0; world.east[:] = 0; world.yaw[:] = 0
        world.obstacle_n[:] = 50; world.obstacle_e[:] = 50
        world.obstacle_horizontal[:] = False; world.obstacle_r[:] = .3
        world.obstacle_n[0, 0] = 1.; world.obstacle_e[0, 0] = 0
        action = BrakingTeacher()(env)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((np.abs(action) <= 1).all())
        self.assertLess(action[0, 0], action[1, 0])

    def test_teacher_accounts_for_current_speed_and_queue(self):
        env=Env(1,831); world=env.world
        world.north[:]=0; world.east[:]=0; world.yaw[:]=0
        world.obstacle_n[:]=50; world.obstacle_e[:]=50
        world.obstacle_horizontal[:]=False; world.obstacle_r[:]=.3
        world.obstacle_n[0,0]=1.4; world.obstacle_e[0,0]=0
        world.delay_buffer.fill(-1); world.forward[:]=0
        slow=BrakingTeacher().labels(env,np.zeros((1,3),np.float32))[0]
        world.forward[:]=.9
        fast=BrakingTeacher().labels(env,np.zeros((1,3),np.float32))[0]
        self.assertLess(fast[0,0],slow[0,0])

    def test_sensor_shield_is_invariant_to_unprovided_geometry(self):
        observation=np.full((2,76),.5,np.float32)
        observation[:,:20]=0; observation[:,6]=.75
        observation[:,-2]=0; observation[:,-1]=1
        action=np.array([[1,.1,0],[1,.1,0]],np.float32)
        np.testing.assert_array_equal(sensor_shield(action,observation)[0],
                                      sensor_shield(action,observation)[1])

    def test_curriculum_rejects_duplicate_insufficient_and_mixed_level_windows(self):
        from x550_teachable.environment import Curriculum
        c=Curriculum()
        self.assertFalse(c.consider('small',0,127,0,127))
        self.assertFalse(c.consider('small',0,999,0,999))
        self.assertFalse(c.consider('wrong',1,128,0,128))
        self.assertFalse(c.consider('good1',0,128,0,128))
        self.assertTrue(c.consider('good2',0,128,0,128))
        with self.assertRaises(ValueError):c.consider('bad',1,128,-5,999,-1)


if __name__ == '__main__':
    unittest.main()
