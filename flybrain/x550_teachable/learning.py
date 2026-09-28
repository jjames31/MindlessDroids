"""Sequential recurrent PPO for offline X550 simulation experiments.

Training data retain time order. Each optimization sequence reconstructs its
initial memory from the episode prefix, then backpropagates through its chunk.
This module neither starts a campaign nor connects to an aircraft.
"""
from dataclasses import dataclass, asdict
import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import HIDDEN_SIZE, OBS_SIZE, PRIV_SIZE
from .environment import Curriculum
from .schema import (AGE_INDEX, DEPTH_SLICE, QUALITY_INDEX, VALIDITY_SLICE,
                     VELOCITY_SLICE)


def unroll(actor, observations, hidden, starts, valid=None):
    """Process [time,batch,features], preserving memory and episode boundaries."""
    logits, memories = [], []
    for t in range(len(observations)):
        # A reset is state-changing even when the first frame is unhealthy and
        # therefore excluded from policy learning.
        hidden = hidden.masked_fill(starts[t].bool().unsqueeze(-1), 0.)
        output, candidate = actor(observations[t], hidden, starts[t])
        hidden = candidate if valid is None else torch.where(
            valid[t, :, None], candidate, hidden)
        logits.append(output)
        memories.append(hidden)
    return torch.stack(logits), torch.stack(memories), hidden

@dataclass(frozen=True)
class Config:
    rollout: int = 32
    sequence: int = 16
    batch_sequences: int = 4
    epochs: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = .995
    gae_lambda: float = .95
    clip: float = .2
    target_kl: float = .015
    entropy_coef: float = .001
    auxiliary_coef: float = .1
    imitation_coef: float = .1
    imitation_decay_updates: int = 500
    extra_collision_cost: float = 50.
    max_grad_norm: float = .5
    behavior_logp_tolerance: float = 2e-6

    def validate(self):
        counts = (self.rollout, self.sequence, self.batch_sequences, self.epochs,
                  self.imitation_decay_updates)
        if any(type(v) is not int or v < 1 for v in counts):
            raise ValueError('positive integer trajectory budgets required')
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('finite nonnegative settings required')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid discount settings')
        if min(self.actor_lr, self.critic_lr, self.target_kl, self.max_grad_norm) <= 0:
            raise ValueError('positive optimizer settings required')


def advantages(reward, value, next_value, terminated, done, gamma=.995, lam=.95):
    """Bootstrap timeouts from final state, never from an autoreset state."""
    arrays = (reward, value, next_value, terminated, done)
    if any(x.shape != reward.shape for x in arrays) or reward.ndim != 2:
        raise ValueError('GAE expects equal [time,env] arrays')
    result = np.empty_like(reward, dtype=np.float32)
    tail = np.zeros(reward.shape[1], np.float32)
    for t in range(len(reward) - 1, -1, -1):
        delta = reward[t] + gamma * next_value[t] * (~terminated[t]) - value[t]
        tail = delta + gamma * lam * (~done[t]) * tail
        result[t] = tail
    return result, result + value


def log_probability(model, logits, raw):
    correction = 2 * (np.log(2.) - raw - F.softplus(-2 * raw))
    return (model.distribution(logits).log_prob(raw) - correction).sum(-1)


def masked_mean(value, valid):
    if value.shape != valid.shape or not bool(valid.any()):
        raise ValueError('nonempty, shape-matched loss mask required')
    return value[valid].mean()


def auxiliary_targets(current, final, info):
    """One-step labels from the true final observation, before any autoreset."""
    current_valid=current[:,VALIDITY_SLICE]>0; final_valid=final[:,VALIDITY_SLICE]>0
    current_depth=np.where(current_valid,current[:,DEPTH_SLICE]*8.,np.inf)
    final_depth=np.where(final_valid,final[:,DEPTH_SLICE]*8.,np.inf)
    current_near=np.min(current_depth,axis=1); final_near=np.min(final_depth,axis=1)
    depth_ok=np.isfinite(current_near)&np.isfinite(final_near)
    motion_ok=((current[:,AGE_INDEX]<=.6)&(current[:,QUALITY_INDEX]>=.5)
               &(final[:,AGE_INDEX]<=.6)&(final[:,QUALITY_INDEX]>=.5))
    depth_delta=np.zeros(len(current),np.float32)
    depth_delta[depth_ok]=np.clip((final_near[depth_ok]-current_near[depth_ok])/4.,-1,1)
    targets=np.column_stack((final[:,VELOCITY_SLICE],depth_delta,
        np.asarray(info['collision'],np.float32),(final_near<.5).astype(np.float32))).astype(np.float32)
    masks=np.column_stack((np.repeat(motion_ok[:,None],3,axis=1),depth_ok,
                           np.ones(len(current),bool),depth_ok))
    return targets,masks


def optimizer_for(model, config):
    """AdamW without weight decay on biases, normalization or log-variance."""
    groups = {}
    for name, parameter in model.named_parameters():
        critic = name.startswith('critic.')
        decay = config.weight_decay if parameter.ndim > 1 else 0.
        key = (critic, decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([
        dict(params=parameters, lr=config.critic_lr if critic else config.actor_lr,
             weight_decay=decay) for (critic, decay), parameters in groups.items()], eps=1e-5)


def burn_in(actor, prefixes, device):
    """Reconstruct current-parameter memory from full episode prefixes."""
    hidden = torch.zeros(len(prefixes), HIDDEN_SIZE, device=device)
    with torch.no_grad():
        for t in range(max((len(p) for p in prefixes), default=0)):
            valid = torch.tensor([t < len(p) for p in prefixes], device=device)
            obs = np.stack([p[t] if t < len(p) else np.zeros(OBS_SIZE) for p in prefixes])
            _, candidate = actor(torch.as_tensor(obs, dtype=torch.float32, device=device),
                                 hidden, torch.zeros_like(valid))
            hidden = torch.where(valid[:, None], candidate, hidden)
    return hidden


def sequence_prefix(data, histories, start, slot):
    resets = np.flatnonzero(data['starts'][:start + 1, slot])
    if len(resets):
        begin=resets[-1]
        return list(data['obs'][begin:start,slot][data['policy_valid'][begin:start,slot]])
    return list(histories[slot]) + list(data['obs'][:start,slot][data['policy_valid'][:start,slot]])


def sequence_batch(data, sequences, device):
    length = max(end - start for start, end, slot in sequences)
    valid = np.zeros((length, len(sequences)), bool)
    result = {}
    for key, array in data.items():
        shape = (length, len(sequences)) + array.shape[2:]
        padded = np.zeros(shape, dtype=array.dtype)
        for column, (start, end, slot) in enumerate(sequences):
            padded[:end - start, column] = array[start:end, slot]
            valid[:end - start, column] = True
        result[key] = torch.as_tensor(padded, device=device)
    result['valid'] = torch.as_tensor(valid, device=device)
    return result


class Engine:
    """Synchronous bounded updates; launching/ownership belongs to the runner."""
    def __init__(self, model, env, config=None, teacher=None, seed=550):
        self.config = config or Config()
        self.config.validate()
        self.model, self.env, self.teacher = model, env, teacher
        self.device = next(model.parameters()).device
        self.optimizer = optimizer_for(model, self.config)
        self.rng = np.random.default_rng(seed)
        self.obs, self.priv = env.observe()
        self.hidden = torch.zeros(env.n, HIDDEN_SIZE, device=self.device)
        self.starts = env.starts.copy()
        self.histories = [[] for _ in range(env.n)]
        self.curriculum = Curriculum(env.world.level)
        self.lesson_counts = [0, 0, 0, 0]
        self.curriculum_window_id = 0
        self.updates = 0

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def collect(self):
        keys = ('obs', 'priv', 'starts', 'policy_valid', 'raw', 'proposed', 'shielded', 'adapted',
                'applied', 'logp', 'value', 'reward', 'next_value', 'terminated',
                'done', 'targets', 'target_mask', 'teacher', 'teacher_mask',
                'teacher_reason')
        store = {key: [] for key in keys}
        histories = copy.deepcopy(self.histories)
        counts = dict(episodes=0, collisions=0, completions=0, timeouts=0)
        shield_count = 0
        for _ in range(self.config.rollout):
            with torch.no_grad():
                healthy=((self.obs[:,AGE_INDEX]<=.6)&(self.obs[:,QUALITY_INDEX]>=.5))
                reset=torch.as_tensor(self.starts,device=self.device)
                base_hidden=self.hidden.masked_fill(reset[:,None],0.)
                logits, candidate = self.model.actor(self.tensor(self.obs), base_hidden,reset)
                memory=torch.where(torch.as_tensor(healthy,device=self.device)[:,None],
                                   candidate,base_hidden)
                raw = self.model.distribution(logits).sample()
                logp = log_probability(self.model, logits, raw)
                value = self.model.critic(self.tensor(self.priv)).squeeze(-1)
            proposed = torch.tanh(raw).cpu().numpy()
            action = self.env.safe_action(proposed)
            if self.teacher is not None:
                target_action, teacher_mask, teacher_reason = self.teacher.labels(self.env, proposed)
            else:
                target_action=np.zeros_like(action); teacher_mask=np.zeros(self.env.n,bool)
                teacher_reason=np.zeros(self.env.n,np.int8)
            shield_count += int(np.any(abs(proposed - action) > 1e-5, axis=1).sum())
            nxt, npriv, reward, done, info = self.env.step(action)
            reward = reward - self.config.extra_collision_cost * info['collision']
            reward = reward - .05 * np.sum((proposed - action) ** 2, axis=1)
            with torch.no_grad():
                next_value = self.model.critic(self.tensor(info['final_privileged'])).squeeze(-1)
            final_features = info['final_features']
            targets,target_mask=auxiliary_targets(self.obs,final_features,info)
            row = dict(obs=self.obs, priv=self.priv, starts=self.starts,policy_valid=healthy,
                raw=raw.cpu().numpy(), proposed=proposed, shielded=action,
                adapted=info['executed_action'], applied=info['applied_action'],
                logp=logp.cpu().numpy(), value=value.cpu().numpy(),
                reward=reward.astype(np.float32), next_value=next_value.cpu().numpy(),
                terminated=info['terminated'], done=done,
                targets=targets, target_mask=target_mask,
                teacher=target_action, teacher_mask=teacher_mask,
                teacher_reason=teacher_reason)
            for key, value in row.items():
                store[key].append(np.array(value, copy=True))
            for slot in range(self.env.n):
                if healthy[slot]: self.histories[slot].append(self.obs[slot].copy())
                if done[slot]:
                    self.histories[slot] = []
            counts['episodes'] += int(done.sum())
            for key, source in (('collisions', 'collision'), ('completions', 'completed'),
                                ('timeouts', 'truncated')):
                counts[key] += int(info[source][done].sum())
            lessons = (done & info['curriculum_slot']
                       & (info['level_at_episode_start']==self.curriculum.level))
            self.lesson_counts[0] += int(lessons.sum())
            self.lesson_counts[1] += int(info['collision'][lessons].sum())
            self.lesson_counts[2] += int(info['completed'][lessons].sum())
            self.lesson_counts[3] += int(info['truncated'][lessons].sum())
            self.obs, self.priv, self.starts = nxt, npriv, done.copy()
            self.hidden = memory.detach() * self.tensor(~done)[:, None]
        data = {key: np.asarray(value) for key, value in store.items()}
        if any(not np.isfinite(value).all() for value in data.values()):
            raise FloatingPointError('non-finite rollout; no optimizer update allowed')
        adv, ret = advantages(data['reward'], data['value'], data['next_value'],
            data['terminated'], data['done'], self.config.gamma, self.config.gae_lambda)
        data['adv'] = ((adv - adv.mean()) / (adv.std() + 1e-6)).astype(np.float32)
        data['ret'] = ret.astype(np.float32)
        counts['shield_rate'] = shield_count / (self.config.rollout * self.env.n)
        corridor = np.array([r*9+c for r in range(3) for c in range(3,6)])
        coverage = data['obs'][...,VALIDITY_SLICE][...,corridor]
        counts['coverage_block_rate'] = float(np.any(coverage < .5, axis=-1).mean())
        counts['forward_inhibition_rate'] = float((data['shielded'][...,0] <= -.999).mean())
        counts['unhealthy_frame_rate'] = float((~data['policy_valid']).mean())
        counts['teacher_label_rate'] = float(data['teacher_mask'].mean())
        counts['teacher_braking_rate'] = float((data['teacher_reason']==2).mean())
        counts['motion_label_rate']=float(data['target_mask'][...,:3].mean())
        counts['depth_change_label_rate']=float(data['target_mask'][...,3].mean())
        counts['collision_label_prevalence']=float(data['targets'][...,4].mean())
        near_mask=data['target_mask'][...,5]
        counts['near_label_prevalence']=float(data['targets'][...,5][near_mask].mean()) if near_mask.any() else 0.
        return data, histories, counts

    def behavior_logp_error(self, data, histories):
        """Reconstruct collection likelihoods before any optimizer mutation."""
        initial=burn_in(self.model.actor,histories,self.device)
        with torch.no_grad():
            consume=torch.as_tensor(data['policy_valid'],device=self.device)
            logits,_,_=unroll(self.model.actor,self.tensor(data['obs']),initial,
                torch.as_tensor(data['starts'],device=self.device),consume)
            rebuilt=log_probability(self.model,logits,self.tensor(data['raw']))
        errors = torch.abs(rebuilt-self.tensor(data['logp'])).cpu().numpy()
        t, slot = np.unravel_index(np.argmax(errors), errors.shape)
        self.behavior_diagnostic = dict(max_error=float(errors[t, slot]),
            timestep=int(t),slot=int(slot),policy_valid=bool(data['policy_valid'][t,slot]),
            episode_start=bool(data['starts'][t,slot]),
            tolerance=float(self.config.behavior_logp_tolerance))
        return float(errors[t,slot])

    def refresh_hidden(self):
        """Rebuild continuing rollout memory with the newly updated actor."""
        self.hidden=burn_in(self.model.actor,self.histories,self.device)

    def optimize(self, data, histories):
        config = self.config
        sequences = [(start, min(start + config.sequence, len(data['obs'])), slot)
            for slot in range(self.env.n) for start in range(0, len(data['obs']), config.sequence)]
        actor_parameters = list(self.model.actor.parameters()) + [self.model.log_std]
        actor_parameters += list(self.model.prediction.parameters())
        measurements, stopped = [], False
        skipped_invalid_batches = 0
        for epoch in range(config.epochs):
            order = self.rng.permutation(len(sequences))
            for offset in range(0, len(order), config.batch_sequences):
                selected = [sequences[i] for i in order[offset:offset + config.batch_sequences]]
                batch = sequence_batch(data, selected, self.device)
                if not bool((batch['valid']&batch['policy_valid']).any()):
                    skipped_invalid_batches += 1
                    continue
                prefixes = [sequence_prefix(data, histories, start, slot)
                            for start, end, slot in selected]
                initial = burn_in(self.model.actor, prefixes, self.device)
                consume=batch['valid']&batch['policy_valid']
                logits, memories, _ = unroll(self.model.actor, batch['obs'], initial,
                    batch['starts'], consume)
                logp = log_probability(self.model, logits, batch['raw'])
                logratio = logp - batch['logp']
                ratio = logratio.exp()
                valid = consume
                kl = masked_mean((ratio - 1) - logratio, valid)
                if not torch.isfinite(kl):
                    raise FloatingPointError('non-finite recurrent PPO KL')
                if float(kl.detach()) > config.target_kl:
                    stopped = True
                    break
                advantage = batch['adv']
                policy = -masked_mean(torch.minimum(ratio * advantage,
                    ratio.clamp(1 - config.clip, 1 + config.clip) * advantage), valid)
                value = self.model.critic(batch['priv']).squeeze(-1)
                value_loss = .5 * masked_mean((value - batch['ret']).square(), valid)
                predictions = self.model.auxiliaries(memories, batch['applied'])
                zero=predictions.sum()*0.
                motion_mask=valid[...,None]&batch['target_mask'][...,:3]
                depth_mask=valid&batch['target_mask'][...,3]
                collision_mask=valid&batch['target_mask'][...,4]
                near_mask=valid&batch['target_mask'][...,5]
                motion_loss=(F.smooth_l1_loss(predictions[...,:3][motion_mask],
                    batch['targets'][...,:3][motion_mask]) if bool(motion_mask.any()) else zero)
                depth_loss=(F.smooth_l1_loss(predictions[...,3][depth_mask],
                    batch['targets'][...,3][depth_mask]) if bool(depth_mask.any()) else zero)
                collision_loss=(F.binary_cross_entropy_with_logits(predictions[...,4][collision_mask],
                    batch['targets'][...,4][collision_mask]) if bool(collision_mask.any()) else zero)
                near_loss=(F.binary_cross_entropy_with_logits(predictions[...,5][near_mask],
                    batch['targets'][...,5][near_mask]) if bool(near_mask.any()) else zero)
                auxiliary=motion_loss+depth_loss+collision_loss+near_loss
                teacher_mask = valid & batch['teacher_mask']
                imitation = logits.sum() * 0.
                if bool(teacher_mask.any()):
                    imitation = F.smooth_l1_loss(
                        torch.tanh(logits)[teacher_mask], batch['teacher'][teacher_mask])
                entropy = masked_mean(self.model.distribution(logits).entropy().sum(-1), valid)
                imitation_weight=config.imitation_coef*max(0.,1-self.updates/config.imitation_decay_updates)
                loss = (policy + value_loss + config.auxiliary_coef * auxiliary
                        + imitation_weight * imitation
                        - config.entropy_coef * entropy)
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite recurrent PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = [p for p in self.model.parameters() if p.grad is not None]
                if any(not torch.isfinite(p.grad).all() for p in parameters):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError('non-finite recurrent PPO gradient')
                actor_norm = nn.utils.clip_grad_norm_(actor_parameters, config.max_grad_norm)
                critic_norm = nn.utils.clip_grad_norm_(self.model.critic.parameters(),
                                                        config.max_grad_norm)
                self.optimizer.step()
                measurements.append(dict(loss=float(loss.detach()),
                    policy_loss=float(policy.detach()), value_loss=float(value_loss.detach()),
                    auxiliary_loss=float(auxiliary.detach()), imitation_loss=float(imitation.detach()),
                    imitation_weight=float(imitation_weight),
                    motion_loss=float(motion_loss.detach()),depth_change_loss=float(depth_loss.detach()),
                    collision_loss=float(collision_loss.detach()),near_obstacle_loss=float(near_loss.detach()),
                    entropy=float(entropy.detach()), kl=float(kl.detach()),
                    actor_grad_norm=float(actor_norm), critic_grad_norm=float(critic_norm)))
            if stopped:
                break
        if not measurements:
            return dict(optimizer_steps=0,early_stop=stopped,
                        skipped_invalid_batches=skipped_invalid_batches,
                        kl=float(kl.detach()) if stopped else 0.)
        result = {key: float(np.mean([row[key] for row in measurements]))
                  for key in measurements[0]}
        result.update(optimizer_steps=len(measurements),early_stop=stopped,
                      skipped_invalid_batches=skipped_invalid_batches)
        return result

    def update(self):
        started=time.perf_counter(); data, histories, rollout = self.collect()
        collected=time.perf_counter()
        behavior_error=self.behavior_logp_error(data,histories)
        if not np.isfinite(behavior_error) or behavior_error>self.config.behavior_logp_tolerance:
            raise FloatingPointError('behavior-policy reconstruction mismatch before optimization: '
                +repr(getattr(self,'behavior_diagnostic',{'max_error':float(behavior_error)})))
        optimization = self.optimize(data, histories)
        optimized=time.perf_counter()
        self.refresh_hidden()
        refreshed=time.perf_counter()
        episodes, collisions, completions, timeouts = self.lesson_counts
        advanced=False; consumed=False
        if episodes>=self.curriculum.min_episodes:
            consumed=True
            advanced=self.curriculum.consider(self.curriculum_window_id,
                self.curriculum.level,episodes,collisions,completions,timeouts)
            self.curriculum_window_id+=1
            self.lesson_counts=[0,0,0,0]
            if advanced: self.env.world.level=self.curriculum.level
        self.updates += 1
        return dict(update=self.updates, curriculum_level=self.curriculum.level,
                    curriculum_advanced=advanced,curriculum_window_consumed=consumed,
                    behavior_logp_max_error=behavior_error,
                    environment_transitions=self.config.rollout*self.env.n,
                    collection_seconds=collected-started,
                    optimization_seconds=optimized-collected,
                    hidden_refresh_seconds=refreshed-optimized,
                    **rollout, **optimization)

    def state_dict(self):
        """Complete trusted-local checkpoint state for exact continuation."""
        return dict(format='x550-recurrent-engine-1', config=asdict(self.config),
            model=copy.deepcopy(self.model.state_dict()),
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            environment=copy.deepcopy(self.env.__dict__), obs=self.obs.copy(),
            priv=self.priv.copy(), hidden=self.hidden.detach().cpu().clone(),
            starts=self.starts.copy(), histories=copy.deepcopy(self.histories),
            curriculum=copy.deepcopy(self.curriculum.__dict__),
            lesson_counts=list(self.lesson_counts),curriculum_window_id=self.curriculum_window_id,
            teacher=copy.deepcopy(getattr(self.teacher,'__dict__',None)),updates=self.updates,
            engine_rng=copy.deepcopy(self.rng.bit_generator.state),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def load_state_dict(self, state):
        if state.get('format') != 'x550-recurrent-engine-1':
            raise ValueError('incompatible recurrent checkpoint')
        if state.get('config') != asdict(self.config):
            raise ValueError('checkpoint configuration mismatch')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.env.__dict__ = copy.deepcopy(state['environment'])
        self.obs = np.array(state['obs'], copy=True)
        self.priv = np.array(state['priv'], copy=True)
        hidden = torch.as_tensor(state['hidden'], device=self.device)
        if hidden.shape != (self.env.n, HIDDEN_SIZE) or not torch.isfinite(hidden).all():
            raise ValueError('invalid checkpoint hidden state')
        self.hidden = hidden.clone()
        self.starts = np.array(state['starts'], dtype=bool, copy=True)
        self.histories = copy.deepcopy(state['histories'])
        self.curriculum.__dict__ = copy.deepcopy(state['curriculum'])
        self.lesson_counts = list(state['lesson_counts'])
        self.curriculum_window_id=int(state['curriculum_window_id'])
        if self.teacher is not None and state.get('teacher') is not None:
            self.teacher.__dict__=copy.deepcopy(state['teacher'])
        self.updates = int(state['updates'])
        self.rng.bit_generator.state = copy.deepcopy(state['engine_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if self.device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
