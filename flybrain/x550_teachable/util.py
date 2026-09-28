"""Provenance, atomic checkpoints and strict single-learner ownership."""
import hashlib,json,os
from pathlib import Path
from nextgen.config import PROJECT
from x550_scenes.util import atomic_json,save
from x550_scenes.health import resource_health

def fingerprint():
    items=[]
    for prefix,folder in [('recurrent',Path(__file__).parent),('scenes',PROJECT/'x550_scenes'),('nextgen',PROJECT/'nextgen')]:
        items += [(prefix+'/'+p.name,p.read_bytes()) for p in sorted(folder.glob('*.py'))]
    items += [(p,(PROJECT/p).read_bytes()) for p in ('drone_profiles.json','banc_controller_graph.json')]
    return hashlib.sha256(b''.join(k.encode()+v for k,v in items)).hexdigest()

def conflicts():
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name)==os.getpid(): continue
        try: a=p.read_bytes().decode().strip('\0').split('\0')
        except (OSError,UnicodeError): continue
        if not a: continue
        tokens={'x550_teachable.train','x550_teachable.evaluate','x550_scenes.train',
                'x550_adapt.train','x550_focus.train','nextgen.train_safe','nextgen.train'}
        files={'x550_continuous.py','train_fly_brain.py','fly_brain.py','fly_brain_nextgen.py'}
        if any(x in tokens or Path(x).name in files for x in a[1:]) or Path(a[0]).name=='arducopter' or (Path(a[0]).name=='gz' and 'sim' in a):
            result.append(dict(pid=int(p.parent.name),command=' '.join(a)))
    return result

def stop_requested(out):
    return any(p.exists() for p in (PROJECT/'X550-IMPROVEMENT.stop',PROJECT/'X550-HARDWARE-READY.flag',PROJECT/'nextgen/training.stop',Path(out)/'training.stop'))
