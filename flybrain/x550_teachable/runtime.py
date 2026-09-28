"""NumPy-only recurrent policy. No device access or aircraft command publishing."""
import json
import numpy as np
from .schema import OBSERVATION_SIZE, SCHEMA, validate_observation

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))

class Policy:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.metadata=json.loads(str(z['metadata']))
            if self.metadata.get('schema')!=SCHEMA:
                raise ValueError('incompatible recurrent policy schema')
            self.w={k:z[k].copy() for k in z.files if k!='metadata'}
        expected={'encoder.0.weight':(96,76),'encoder.0.bias':(96,),
            'encoder.1.weight':(96,),'encoder.1.bias':(96,),
            'memory.weight_ih':(192,96),'memory.weight_hh':(192,64),
            'memory.bias_ih':(192,),'memory.bias_hh':(192,),
            'norm.weight':(64,),'norm.bias':(64,),
            'head.0.weight':(32,64),'head.0.bias':(32,),
            'head.2.weight':(3,32),'head.2.bias':(3,)}
        if set(self.w)!=set(expected) or any(self.w[k].shape!=v for k,v in expected.items()):
            raise ValueError('invalid recurrent weights')
        if not all(np.isfinite(v).all() for v in self.w.values()): raise ValueError('non-finite weights')
        self.reset()

    def reset(self): self.hidden=None; self.last_timestamp=None
    def linear(self,x,name): return x@self.w[name+'.weight'].T+self.w[name+'.bias']
    def norm(self,x,name):
        return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*self.w[name+'.weight']+self.w[name+'.bias']

    def normalized_action(self,observation,reset=None,timestamp=None):
        x=np.asarray(observation,np.float32)
        x=validate_observation(x)
        if (x[..., -2]>.6).any() or (x[..., -1]<.5).any():
            raise ValueError('navigation inhibited: unhealthy observation')
        shape=x.shape[:-1]+(64,)
        if self.hidden is None or self.hidden.shape!=shape: self.hidden=np.zeros(shape,np.float32)
        if reset is not None: self.hidden*=~np.asarray(reset,bool)[...,None]
        if timestamp is not None:
            if not np.isfinite(timestamp): raise ValueError('invalid timestamp')
            if self.last_timestamp is not None:
                gap=timestamp-self.last_timestamp
                if gap<=0: raise ValueError('duplicate or out-of-order observation')
                if gap>.5: self.hidden.fill(0.)
            self.last_timestamp=timestamp
        y=self.norm(self.linear(x,'encoder.0'),'encoder.1'); y=y*sigmoid(y)
        gi=y@self.w['memory.weight_ih'].T+self.w['memory.bias_ih']
        gh=self.hidden@self.w['memory.weight_hh'].T+self.w['memory.bias_hh']
        ir,iz,inn=np.split(gi,3,axis=-1); hr,hz,hn=np.split(gh,3,axis=-1)
        r=sigmoid(ir+hr); z=sigmoid(iz+hz); new=np.tanh(inn+r*hn)
        self.hidden=((1-z)*new+z*self.hidden).astype(np.float32)
        y=self.linear(self.norm(self.hidden,'norm'),'head.0'); y=y*sigmoid(y)
        return np.tanh(self.linear(y,'head.2')).astype(np.float32)
