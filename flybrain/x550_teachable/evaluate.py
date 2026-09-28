"""Frozen recurrent-policy development evaluation; no training or hardware I/O."""
import hashlib
import json
import numpy as np
import torch
from .environment import Env
from .model import HIDDEN_SIZE
from .schema import SCHEMA


def policy_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate(model,episodes=512,envs=128,seed=3305500000,suite='legacy',device=None,
             panel_id=None,horizon=400):
    if episodes<1 or envs<1: raise ValueError('positive evaluation budgets required')
    device=device or next(model.parameters()).device
    totals=dict(episodes=0,collisions=0,completions=0,timeouts=0)
    clearances=[]; families={}; records=[]
    model.eval()
    while totals['episodes']<episodes:
        n=min(envs,episodes-totals['episodes'])
        offset=totals['episodes']; env=Env(n,seed+offset,horizon=horizon,training=False,suite=suite)
        obs,_=env.observe(); hidden=torch.zeros(n,HIDDEN_SIZE,device=device)
        starts=np.ones(n,bool); finished=np.zeros(n,bool)
        minimum=np.full(n,np.inf); interventions=np.zeros(n,np.int32); durations=np.zeros(n,np.int32)
        progress=np.zeros(n,np.float32)
        for step in range(env.world.horizon):
            with torch.no_grad():
                logits,hidden=model.actor(torch.as_tensor(obs,dtype=torch.float32,device=device),
                    hidden,torch.as_tensor(starts,device=device))
                proposed=torch.tanh(logits).cpu().numpy()
            safe=env.safe_action(proposed); interventions+=np.any(abs(safe-proposed)>1e-5,axis=1)&~finished
            obs,_,_,done,info=env.step(safe); starts=np.zeros(n,bool)
            minimum=np.minimum(minimum,np.asarray(info['clearance']));durations[~finished]=step+1
            progress[~finished]+=np.asarray(info['progress_m'])[~finished]
            newly=done&~finished
            if newly.any():
                totals['collisions']+=int(info['collision'][newly].sum())
                totals['completions']+=int(info['completed'][newly].sum())
                totals['timeouts']+=int(info['truncated'][newly].sum())
                clearances.extend(np.asarray(info['clearance'])[newly].tolist())
                for family in np.asarray(info['scene_family'])[newly]:
                    key=str(int(family)); families[key]=families.get(key,0)+1
                for slot in np.flatnonzero(newly):
                    records.append(dict(scenario_id=f'{suite}:{seed+offset}:{slot}',
                        suite=suite,scene_family=int(info['scene_family'][slot]),
                        collision=bool(info['collision'][slot]),completed=bool(info['completed'][slot]),
                        timeout=bool(info['truncated'][slot]),duration_steps=int(durations[slot]),
                        progress_m=float(progress[slot]),minimum_clearance_m=float(minimum[slot]),
                        shield_interventions=int(interventions[slot])))
                finished|=newly
            if finished.all(): break
        if not finished.all():
            raise RuntimeError('evaluation failed to retain every episode outcome')
        totals['episodes']+=n
    totals['collision_free_rate']=1-totals['collisions']/episodes
    totals['completion_rate']=totals['completions']/episodes
    totals['mean_terminal_clearance']=float(np.mean(clearances))
    identity=dict(panel_id=panel_id or f'{suite}-{seed}',seed=seed,suite=suite,
        episodes=episodes,horizon=horizon,observation_schema=SCHEMA,
        shield_version='x550-depth-coverage-stopping-v2')
    identity['scenario_set_sha256']=hashlib.sha256(json.dumps(
        sorted(r['scenario_id'] for r in records),separators=(',',':')).encode()).hexdigest()
    return dict(identity=identity,policy_sha256=policy_hash(model),metrics=totals,
                scene_family_counts=families,episodes=records,hardware_authorized=False)


def improves(candidate,incumbent):
    def validate(reports):
        if not isinstance(reports,list) or len(reports)!=4: raise ValueError('exactly four panels required')
        ids=[]
        for report in reports:
            identity=report.get('identity',{}); metrics=report.get('metrics',{})
            ids.append(identity.get('panel_id'))
            required=('episodes','collisions','completions','timeouts')
            if any(type(metrics.get(k)) not in (int,np.int32,np.int64) for k in required):
                raise ValueError('integer outcome counts required')
            n,c,s,t=(metrics[k] for k in required)
            if n<1 or any(v<0 or v>n for v in (c,s,t)) or c+s+t!=n:
                raise ValueError('invalid evaluation accounting')
            if len(report.get('episodes',[]))!=n or not identity.get('scenario_set_sha256'):
                raise ValueError('complete episode evidence required')
        if len(set(ids))!=4 or any(x is None for x in ids): raise ValueError('unique panel identities required')
    validate(candidate)
    if incumbent is None: return True
    validate(incumbent)
    candidate=sorted(candidate,key=lambda x:x['identity']['panel_id'])
    incumbent=sorted(incumbent,key=lambda x:x['identity']['panel_id'])
    identity_keys=('panel_id','seed','suite','episodes','horizon','observation_schema',
                   'shield_version','scenario_set_sha256')
    for new,old in zip(candidate,incumbent):
        if any(new['identity'].get(k)!=old['identity'].get(k) for k in identity_keys):
            raise ValueError('candidate and incumbent evidence are not matched')
        a,b=new['metrics'],old['metrics']
        if a['collisions']>b['collisions'] or a['completions']<b['completions']:
            return False
    return (sum(x['metrics']['collisions'] for x in candidate)
            <sum(x['metrics']['collisions'] for x in incumbent)
            or sum(x['metrics']['completions'] for x in candidate)
            >sum(x['metrics']['completions'] for x in incumbent))
