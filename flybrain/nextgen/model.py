"""Compact asymmetric actor-critic and deployment export."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ACTION_SIZE, OBSERVATION_SIZE, PRIVILEGED_SIZE
from .features import FEATURE_NAMES


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden=(128,128)):
        super().__init__()
        layers=[]
        for before,after in zip((input_size,)+tuple(hidden),tuple(hidden)):
            layers.extend((nn.Linear(before,after),nn.Tanh()))
        layers.append(nn.Linear(hidden[-1],output_size))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=MLP(OBSERVATION_SIZE,ACTION_SIZE)
        self.critic=MLP(PRIVILEGED_SIZE,1)
        self.log_std=nn.Parameter(torch.full((ACTION_SIZE,),-1.2))

    def distribution(self,obs):
        return torch.distributions.Normal(self.actor(obs),self.log_std.exp())

    def act(self,obs,privileged,deterministic=False):
        dist=self.distribution(obs)
        raw=dist.mean if deterministic else dist.rsample()
        action=torch.tanh(raw)
        # Change of variables for tanh-squashed Gaussian.
        logp=(dist.log_prob(raw)-torch.log(1-action.square()+1e-6)).sum(-1)
        return action,logp,self.critic(privileged).squeeze(-1)

    def evaluate(self,obs,privileged,raw):
        dist=self.distribution(obs); action=torch.tanh(raw)
        logp=(dist.log_prob(raw)-torch.log(1-action.square()+1e-6)).sum(-1)
        return logp,dist.entropy().sum(-1),self.critic(privileged).squeeze(-1)


def export_actor(model, path, training_profiles=None):
    """Export plain NumPy weights so deployment does not require PyTorch."""
    path=Path(path); payload={}
    linears=[layer for layer in model.actor.net if isinstance(layer,nn.Linear)]
    for i,layer in enumerate(linears):
        payload[f"w{i}"]=layer.weight.detach().cpu().numpy().astype(np.float32)
        payload[f"b{i}"]=layer.bias.detach().cpu().numpy().astype(np.float32)
    payload["metadata"]=np.asarray(json.dumps({"schema":2,"activation":"tanh",
        "input_size":OBSERVATION_SIZE,"output_size":ACTION_SIZE,
        "feature_names":FEATURE_NAMES,
        "training_profiles":list(training_profiles) if training_profiles is not None else None,
        "hardware_authorized":False}))
    np.savez_compressed(path,**payload)
