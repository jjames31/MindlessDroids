"""Compact recurrent X550 actor; predictions are training-only, not safety logic."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .schema import (ACTION_NAMES, ACTION_STAGES, AUXILIARY_NAMES, OBSERVATION_NAMES,
                     OBSERVATION_SIZE, SCHEMA)

OBS_SIZE = OBSERVATION_SIZE
PRIV_SIZE = OBS_SIZE + 8
HIDDEN_SIZE = 64


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OBS_SIZE, 96), nn.LayerNorm(96), nn.SiLU())
        self.memory = nn.GRUCell(96, HIDDEN_SIZE)
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.head = nn.Sequential(nn.Linear(HIDDEN_SIZE, 32), nn.SiLU(), nn.Linear(32, 3))
        nn.init.orthogonal_(self.head[-1].weight, gain=.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs, hidden, reset):
        hidden = hidden * (~reset.bool()).unsqueeze(-1)
        hidden = self.memory(self.encoder(obs), hidden)
        return self.head(self.norm(hidden)), hidden


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = nn.Sequential(nn.Linear(PRIV_SIZE,128), nn.LayerNorm(128),
                                    nn.SiLU(), nn.Linear(128,64), nn.SiLU(), nn.Linear(64,1))
        self.prediction = nn.Sequential(nn.Linear(HIDDEN_SIZE+3,64), nn.SiLU(), nn.Linear(64,6))
        self.log_std = nn.Parameter(torch.tensor([-1.8,-1.4,-1.8]))

    def distribution(self, logits):
        return torch.distributions.Normal(logits, self.log_std.exp().expand_as(logits))

    def auxiliaries(self, hidden, action):
        return self.prediction(torch.cat((hidden,action),dim=-1))


def export_actor(model, path):
    data = {k:v.detach().cpu().numpy().astype(np.float32) for k,v in model.actor.state_dict().items()}
    data['metadata'] = np.asarray(json.dumps(dict(schema=SCHEMA,observation_size=OBS_SIZE,
        hidden_size=HIDDEN_SIZE,grid=[3,9],depth_units='metres',history='recurrent',
        observation_names=OBSERVATION_NAMES, action_names=ACTION_NAMES,
        action_stages=ACTION_STAGES, auxiliary_names=AUXILIARY_NAMES,
        velocity_target='next observed heading-frame velocity in metres/second',
        auxiliary_action_condition='delay_applied normalized command',
        training_profiles=['x550'],hardware_authorized=False,normalization_eps=1e-5)))
    path=Path(path); tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**data); tmp.replace(path)
