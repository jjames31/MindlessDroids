"""Local ownership, provenance and atomic persistence utilities."""
import hashlib
import json
import os
from pathlib import Path
import torch
from nextgen.config import PROJECT


def fingerprint():
    groups=[('scenes',Path(__file__).parent),('nextgen',PROJECT/'nextgen')]
    parts=[(prefix+'/'+p.name,p.read_bytes()) for prefix,folder in groups for p in sorted(folder.glob('*.py'))]
    for name in ('drone_profiles.json','banc_controller_graph.json','fly_brain_nextgen.py','obstacle_course.py','curriculum.py'):
        parts.append((name,(PROJECT/name).read_bytes()))
    return hashlib.sha256(b''.join(n.encode()+b for n,b in parts)).hexdigest()


def atomic_json(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)); os.replace(tmp,path)


def save(path,data):
    path=Path(path); tmp=path.with_suffix('.tmp')
    torch.save(data,tmp); os.replace(tmp,path)


def conflicts():
    rows=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name)==os.getpid(): continue
        try: args=path.read_bytes().decode(errors='replace').strip('\0').split('\0')
        except OSError: continue
        if not args: continue
        executable=Path(args[0]).name
        if any(a in ('nextgen.train','nextgen.train_safe','nextgen.gazebo_evaluate','x550_scenes.train','x550_scenes.evaluate','x550_scenes.flight','x550_adapt.train',
                     'x550_adapt.gazebo','x550_adapt.flight','x550_focus.train','x550_focus.gazebo',
                     'x550_focus.flight','x550_focus.evaluate','x550_adapt.evaluate') or Path(a).name in ('train_fly_brain.py',
                     'fly_brain.py','fly_brain_nextgen.py','supervise_nextgen.py') for a in args[1:]) or executable=='arducopter' or                      (executable=='gz' and 'sim' in args):
            rows.append({'pid':int(path.parent.name),'command':' '.join(args)})
    return rows
