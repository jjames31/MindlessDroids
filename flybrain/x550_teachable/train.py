"""Bounded offline recurrent X550 experiment runner; never publishes commands."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch

from nextgen.config import PROJECT
from .environment import Env
from .evaluate import evaluate, improves
from .learning import Config, Engine
from .model import Model, export_actor
from .teacher import BrakingTeacher
from .util import conflicts, fingerprint, resource_health, stop_requested


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); os.replace(temporary, path)


def session_deadline(wall_hours, started=None):
    """Return a wall-clock deadline for this process, not the run's birth time."""
    started=time.time() if started is None else float(started)
    if not np.isfinite(started) or not np.isfinite(wall_hours) or wall_hours<=0:
        raise ValueError('finite positive session timing required')
    return started+float(wall_hours)*3600.


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--rollout', type=int, default=32)
    parser.add_argument('--sequence', type=int, default=16)
    parser.add_argument('--batch-sequences', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260927550)
    parser.add_argument('--wall-hours', type=float, default=2.)
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-episodes', type=int, default=512)
    parser.add_argument('--dev-seed', type=int, default=3355000000)
    parser.add_argument('--no-auxiliary', action='store_true')
    parser.add_argument('--visible-lesson-backdrop', action='store_true',
                        help='Use a sensed finite wall in introductory training lessons only')
    parser.add_argument('--teacher', action='store_true',
                        help='Opt in to audited privileged corrective labels')
    args = parser.parse_args(argv)
    budgets = (args.updates, args.envs, args.horizon, args.rollout, args.sequence,
               args.batch_sequences, args.epochs, args.checkpoint_every,
               args.eval_every,args.eval_episodes)
    if any(type(v) is not int or v < 1 for v in budgets):
        parser.error('positive integer budgets required')
    if args.updates > 1000000:
        parser.error('supervised recurrent campaigns are limited to 1000000 total updates')
    if not 0 < args.wall_hours <= 2:
        parser.error('wall-hours must be in (0,2]')
    return args


def main(argv=None):
    args = parse(argv)
    lock = open('/tmp/fba-x550-recurrent.lock', 'w')
    shared_lock = open('/tmp/fba-nextgen-training.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('another recurrent runner owns the learner lock') from exc
    active = conflicts()
    if active:
        raise RuntimeError('conflicting simulator/trainer is active: '+json.dumps(active))
    output = Path(args.output).resolve()
    if stop_requested(output):
        raise RuntimeError('explicit stop request is present')
    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        if checkpoint_path.parent != output:
            raise ValueError('resume checkpoint must belong to the output directory')
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved.get('source_sha256') != fingerprint():
            raise ValueError('checkpoint source mismatch; no forced migration')
        if args.updates <= saved['engine']['updates']:
            raise ValueError('new update target must exceed saved progress')
        if saved['runner_config'] != {k:v for k,v in vars(args).items()
                                      if k not in ('updates','resume')}:
            raise ValueError('runner configuration mismatch')
    else:
        saved = None
        if output.exists():
            raise FileExistsError('new experiment output must not already exist')
        output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    streams=np.random.SeedSequence(args.seed).generate_state(5,dtype=np.uint32)
    python_seed, numpy_seed, model_seed, world_seed, engine_seed=map(int,streams)
    random.seed(python_seed); np.random.seed(numpy_seed); torch.manual_seed(model_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(model_seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.set_num_threads(2)
    config = Config(rollout=args.rollout, sequence=args.sequence,
                    batch_sequences=args.batch_sequences, epochs=args.epochs,
                    auxiliary_coef=0. if args.no_auxiliary else .1,
                    imitation_coef=.1 if args.teacher else 0.)
    engine = Engine(Model().to(device), Env(args.envs,world_seed,args.horizon,training=True,
                    visible_lesson_backdrop=args.visible_lesson_backdrop),
                    config, BrakingTeacher() if args.teacher else None, engine_seed)
    if saved:
        engine.load_state_dict(saved['engine'])
        created = saved['created']
        selection=copy.deepcopy(saved['selection'])
        run_metadata=json.loads((output/'run.json').read_text())
    else:
        created = time.time()
        selection=dict(best_reports=None,best_state=None,best_update=None,evaluated_updates=[])
        run_metadata=dict(
            format='x550-recurrent-run-1', created=created,
            source_sha256=fingerprint(), runner_config={k:v for k,v in vars(args).items()
                if k not in ('updates','resume')}, hardware_authorized=False,
            seed_streams=dict(python=python_seed,numpy=numpy_seed,model=model_seed,
                              world=world_seed,minibatch=engine_seed),
            versions=dict(python=os.sys.version,numpy=np.__version__,torch=torch.__version__,
                          cuda=torch.version.cuda),
            determinism=dict(cudnn_deterministic=True,cudnn_benchmark=False,
                             cuda_bitwise_guaranteed=False),
            initialization='fresh random recurrent policy; no incompatible MLP load')
        export_actor(engine.model,output/'initial-policy.npz')
        atomic_checkpoint(output/'initial-model.pt',dict(model=engine.model.state_dict(),update=0,
            source_sha256=fingerprint(),hardware_authorized=False))
    session_started=time.time()
    run_metadata.setdefault('sessions',[]).append(dict(started=session_started,
        start_update=engine.updates,requested_updates=args.updates,
        wall_hours=args.wall_hours,resumed=bool(saved)))
    atomic_json(output/'run.json',run_metadata)
    deadline=session_deadline(args.wall_hours,session_started)
    health=resource_health(output,device.type=='cuda')
    atomic_json(output/'status.json',dict(status='starting',update=engine.updates,
        session_started=session_started,hardware_authorized=False))
    stopping = {'requested': False}
    def request_stop(signum, frame): stopping['requested'] = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    def save(engine_state=None):
        atomic_checkpoint(output/'training-checkpoint.pt', dict(
            format='x550-recurrent-run-1', created=created,
            source_sha256=fingerprint(), runner_config={k:v for k,v in vars(args).items()
                if k not in ('updates','resume')}, engine=engine.state_dict() if engine_state is None else engine_state,
            selection=selection))
    def assess(number):
        reports=[evaluate(engine.model,args.eval_episodes,min(128,args.eval_episodes),
            args.dev_seed+1000*i,'legacy' if i<2 else 'realistic',device,
            panel_id=f'panel{i}') for i in range(4)]
        for i,report in enumerate(reports): atomic_json(output/f'dev-{number:06d}-panel{i}.json',report)
        if improves(reports,selection['best_reports']):
            selection.update(best_reports=reports,best_state=copy.deepcopy(engine.model.state_dict()),
                             best_update=number)
            export_actor(engine.model,output/'best-policy.npz')
            atomic_checkpoint(output/'best-model.pt',dict(model=selection['best_state'],update=number,
                reports=reports,source_sha256=fingerprint(),hardware_authorized=False))
        selection['evaluated_updates'].append(number)
        return reports
    reason = 'update_budget_complete'
    last_good=engine.state_dict()
    try:
        if not saved:
            assess(0); save()
        last_good=engine.state_dict()
        while engine.updates < args.updates:
            if stopping['requested'] or stop_requested(output):
                reason = 'user_stop'; break
            if time.time() >= deadline:
                reason = 'wall_budget_complete'; break
            if fingerprint() != json.loads((output/'run.json').read_text())['source_sha256']:
                raise RuntimeError('source changed during recurrent experiment')
            health=resource_health(output,device.type=='cuda')
            metrics = engine.update(); metrics.update(time=time.time(),health=health)
            last_good=engine.state_dict()
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            atomic_json(output/'status.json',dict(status='training',session_started=session_started,
                **metrics,hardware_authorized=False))
            if engine.updates % args.checkpoint_every == 0:
                save()
            if engine.updates%args.eval_every==0:
                assess(engine.updates); save()
    except BaseException as exc:
        try: save(last_good)
        finally:
            failure=dict(status='failed',error=repr(exc),update=last_good['updates'],
                attempted_update=last_good['updates']+1,time=time.time(),health=health,
                behavior_diagnostic=getattr(engine,'behavior_diagnostic',None),
                hardware_authorized=False)
            atomic_json(output/'failure.json',failure)
            atomic_json(output/'status.json',failure)
        raise
    if engine.updates not in selection['evaluated_updates']:
        assess(engine.updates)
    save(); export_actor(engine.model, output/'candidate-final.npz')
    atomic_checkpoint(output/'final-model.pt',dict(model=engine.model.state_dict(),
        update=engine.updates,source_sha256=fingerprint(),hardware_authorized=False))
    atomic_json(output/'status.json',dict(status=reason,update=engine.updates,
        session_started=session_started,hardware_authorized=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
