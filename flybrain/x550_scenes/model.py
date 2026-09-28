"""Frozen legacy controller plus a trainable finite-history correction."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from nextgen.model import ActorCritic as LegacyModel, MLP
from nextgen.features import FEATURE_NAMES

FRAMES = 8
FRAME_SIZE = 32
INPUT_SIZE = FRAMES*FRAME_SIZE
PRIV_SIZE = INPUT_SIZE+8
RESIDUAL_LIMIT = .75
SCHEMA = 'x550-history-residual-v1'


class HistoryActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = MLP(32,3)
        self.base.requires_grad_(False)
        self.residual = MLP(INPUT_SIZE,3,(128,64))
        nn.init.zeros_(self.residual.net[-1].weight)
        nn.init.zeros_(self.residual.net[-1].bias)

    def forward(self,x):
        return self.base(x[..., :32])+RESIDUAL_LIMIT*torch.tanh(self.residual(x))


class Model(LegacyModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.actor=HistoryActor()
        self.critic=MLP(PRIV_SIZE,1)
        self.log_std=nn.Parameter(torch.tensor([-1.8,-1.5,-2.0]))

    def initialize(self, parent_state):
        self.actor.base.load_state_dict({k[len('actor.'):]:v for k,v in parent_state.items()
                                         if k.startswith('actor.')})
        with torch.no_grad():
            self.critic.net[0].weight.zero_()
            self.critic.net[0].weight[:,:32].copy_(parent_state['critic.net.0.weight'][:,:32])
            self.critic.net[0].weight[:,INPUT_SIZE:].copy_(parent_state['critic.net.0.weight'][:,32:])
            for key,v in self.critic.state_dict().items():
                if key!='net.0.weight': v.copy_(parent_state['critic.'+key])


def export_policy(model,path):
    data={}
    for prefix,net in [('base',model.actor.base),('residual',model.actor.residual)]:
        for i,layer in enumerate(x for x in net.net if isinstance(x,nn.Linear)):
            data[f'{prefix}_w{i}']=layer.weight.detach().cpu().numpy().astype(np.float32)
            data[f'{prefix}_b{i}']=layer.bias.detach().cpu().numpy().astype(np.float32)
    data['metadata']=np.asarray(json.dumps(dict(schema=SCHEMA,frames=FRAMES,
        frame_size=32,feature_names=FEATURE_NAMES,history_order='newest_first',
        residual_limit=RESIDUAL_LIMIT,training_profiles=['x550'],hardware_authorized=False)))
    path=Path(path); temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**data); temporary.replace(path)
