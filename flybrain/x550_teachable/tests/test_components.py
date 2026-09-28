"""Offline regression checks for completed redesign components only."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from x550_teachable.model import Model,export_actor,OBS_SIZE,HIDDEN_SIZE
from x550_teachable.runtime import Policy
from x550_teachable.sensing import from_depth_frame,pack_observation,ray_depth,reduce_samples
from x550_teachable.schema import validate_observation
from x550_teachable.safety import shield_action as depth_shield
from x550_teachable.environment import Env,Curriculum

class ComponentTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(55)
    def test_actor_budget_and_recurrent_reset(self):
        m=Model(); x=torch.randn(4,OBS_SIZE); h=torch.randn(4,HIDDEN_SIZE)
        a,_=m.actor(x,h,torch.ones(4,dtype=torch.bool))
        b,_=m.actor(x,torch.zeros_like(h),torch.zeros(4,dtype=torch.bool))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertLess(sum(p.numel() for p in m.actor.parameters()),250000)
    def test_numpy_temporal_parity(self):
        m=Model(); h=torch.zeros(4,HIDDEN_SIZE)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.npz'; export_actor(m,p); actor=Policy(p)
            for t in range(12):
                x=torch.rand(4,OBS_SIZE); x[:,-2]=0; x[:,-1]=1
                reset=np.array([t%3==0,False,False,t%4==0])
                raw,h=m.actor(x,torch.from_numpy(reset))
                actual=actor.normalized_action(x.numpy(),reset)
                np.testing.assert_allclose(actual,torch.tanh(raw).detach().numpy(),atol=2e-6)

    def test_depth_units_and_invalid_returns(self):
        image=np.full((400,640),2000,np.uint16)
        k=np.array([[400,0,320],[0,400,200],[0,0,1]],float)
        depth,valid=from_depth_frame(image,k)
        np.testing.assert_allclose(depth,2.)
        self.assertTrue((valid>0).all())
        depth,valid=from_depth_frame(np.zeros_like(image),k)
        self.assertFalse(depth.any()); self.assertFalse(valid.any())
    def test_dense_pooling_detects_thin_vertical_obstacle(self):
        image=np.full((400,640),4000,np.uint16); image[:,330:340]=1000
        k=np.array([[400,0,320],[0,400,200],[0,0,1]],float)
        depth,valid=from_depth_frame(image,k)
        self.assertLess(depth.min(),1.1); self.assertTrue((valid>0).all())
    def test_schema_and_shield_reject_sparse_or_bad_health(self):
        x=np.zeros((1,76),np.float32); x[:,-1]=1
        x[:,20:47]=.5; x[:,47:74]=0; x[:,47+13]=1/9
        a=depth_shield(np.array([[1,.6,.5]],np.float32),x)
        np.testing.assert_allclose(a,[[-1,0,0]])
        for index,value in ((74,-1),(75,2)):
            bad=x.copy();bad[0,index]=value
            with self.assertRaises(ValueError):validate_observation(bad)
    def test_stopping_bound_inhibits_delayed_fast_approach(self):
        x=np.zeros((1,76),np.float32);x[:,-1]=1;x[:,0]=1
        x[:,12]=.48*1.5;x[:,20:47]=3/8;x[:,47:74]=1
        self.assertEqual(depth_shield(np.array([[1,0,0]],np.float32),x)[0,0],-1)
    def test_delayed_response_fixture_brakes_without_contact(self):
        e=Env(1,444,horizon=100,training=False);w=e.world
        w.north[:]=0;w.east[:]=0;w.altitude[:]=3;w.yaw[:]=0;w.forward[:]=1
        w.vertical[:]=0;w.wind[:]=0;w.response_tau[:]=.65;w.delay[:]=4
        w.footprint[:]=.48;w.vertical_radius[:]=.22
        w.obstacle_n[:]=50;w.obstacle_e[:]=50;w.obstacle_z[:]=3;w.obstacle_r[:]=.3
        w.obstacle_horizontal[:]=False;w.obstacle_n[0,0]=3;w.obstacle_e[0,0]=0
        w.delay_buffer.fill(0);w.delay_buffer[:,:,0]=1
        legacy,priv=w.observe();e.legacy=legacy;e.privileged=priv;e.cache=e._sense(legacy)
        for _ in range(100):
            _,_,_,_,info=e.step(e.safe_action(np.array([[1,0,0]],np.float32)))
            self.assertFalse(info['collision'][0])
        self.assertGreater(info['clearance'][0],1.)
    def test_invalid_observation_rejected(self):
        with self.assertRaises(ValueError): pack_observation(np.full((1,20),np.nan),np.zeros((1,27)),np.zeros((1,27)),0,1)
    def test_second_obstacle_changes_visual_input(self):
        e=Env(1,50); w=e.world
        w.north[:]=0; w.east[:]=0; w.altitude[:]=3; w.yaw[:]=0
        w.obstacle_n[:]=50; w.obstacle_e[:]=50; w.scene_shape[:]=0
        w.obstacle_horizontal[:]=False; w.obstacle_r[:]=.2
        w.obstacle_n[0,0]=1.5; w.obstacle_e[0,0]=0
        a,_=ray_depth(w)
        w.scene_shape[0,1]=1; w.obstacle_n[0,1]=3; w.obstacle_e[0,1]=1.2
        w.obstacle_z[0,1]=3; w.half_n[0,1]=.25; w.half_e[0,1]=.5; w.half_z[0,1]=1
        b,_=ray_depth(w)
        self.assertGreater(np.count_nonzero(abs(a-b)>.01),0)
    def test_curriculum_requires_two_competent_checks(self):
        c=Curriculum(); self.assertFalse(c.consider('a',0,128,0,128))
        self.assertFalse(c.consider('a',0,128,0,128))
        self.assertTrue(c.consider('b',0,128,0,128)); self.assertEqual(c.level,1)
        self.assertFalse(c.consider('c',1,128,40,88)); self.assertEqual(c.level,1)

    def test_environment_contract(self):
        e=Env(8,52,horizon=2,training=True); x,p=e.observe()
        self.assertEqual(x.shape,(8,76)); self.assertEqual(p.shape,(8,84))
        for _ in range(5):
            x,p,r,done,info=e.step(e.safe_action(np.tile([-1,0,0],(8,1))))
            self.assertTrue(np.isfinite(x).all()); self.assertTrue(np.isfinite(p).all())
            self.assertEqual(set(e.world.profile_index),{2})
            self.assertEqual(info['final_privileged'].shape,(8,84))
    def test_curriculum_changes_actual_obstacles(self):
        a=Env(64,53,training=True,level=0)
        b=Env(64,53,training=True,level=2)
        self.assertTrue(a.world.curriculum_slots.any())
        self.assertTrue((a.world.obstacle_n[a.world.curriculum_slots]>40).all())
        self.assertTrue((b.world.obstacle_n[b.world.curriculum_slots,:2]<10).all())
    def test_auxiliary_gradients_are_finite(self):
        m=Model(); x=torch.randn(4,76); h=torch.zeros(4,64)
        _,h=m.actor(x,torch.zeros(4,64),torch.zeros(4,dtype=torch.bool))
        prediction=m.auxiliaries(h,torch.zeros(4,3))
        self.assertEqual(prediction.shape,(4,6)); prediction.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.actor.parameters() if p.grad is not None))
    def test_quality_guard_does_not_depend_on_network_output(self):
        e=Env(4,54); e.cache[0,-1]=0; e.cache[1,-2]=1
        a=e.safe_action(np.ones((4,3),np.float32))
        np.testing.assert_array_equal(a[:2],np.tile([-1,0,0],(2,1)))

if __name__=='__main__': unittest.main()
