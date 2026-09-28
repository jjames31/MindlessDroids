"""Review-first GitHub updates for an existing Linux/WSL Flybrain workspace.

Only this administrative helper is executed. Downloaded project code is parsed,
not imported or run. No training, hardware, dependency installation or restart.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

REPOSITORY = 'https://github.com/jjames31/MindlessDroids.git'
BRANCH = 'main'
LOCAL_ONLY = ('x550_teachable/runtime.py', 'banc_controller_graph.json')
INITIAL_PATHS = tuple('''drone_profiles.json
nextgen/__init__.py
nextgen/config.py
nextgen/env.py
nextgen/features.py
nextgen/model.py
x550_scenes/__init__.py
x550_scenes/catalog.py
x550_scenes/control.py
x550_scenes/env.py
x550_scenes/health.py
x550_scenes/model.py
x550_scenes/scene_world.py
x550_scenes/util.py
x550_teachable/__init__.py
x550_teachable/environment.py
x550_teachable/evaluate.py
x550_teachable/learning.py
x550_teachable/model.py
x550_teachable/safety.py
x550_teachable/schema.py
x550_teachable/sensing.py
x550_teachable/teacher.py
x550_teachable/tests/test_components.py
x550_teachable/tests/test_consistency.py
x550_teachable/tests/test_reset_edge.py
x550_teachable/train.py
x550_teachable/util.py'''.splitlines())


def digest(data: bytes) -> str:
    # Ignore CRLF and EOF-newline differences only, not indentation or content.
    return hashlib.sha256(data.replace(b'\r\n', b'\n').rstrip(b'\n')).hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def target(root: Path, name: str) -> Path:
    if name in LOCAL_ONLY or not re.fullmatch(
        r'(?:drone_profiles\.json|(?:nextgen|x550_scenes|x550_teachable)/'
        r'(?:tests/)?[A-Za-z_][A-Za-z_0-9]*\.py)', name
    ):
        raise ValueError('unapproved project path: ' + name)
    path = root / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink in project path: ' + name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('path escapes project: ' + name)
    return path


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.flybrain-sync-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


@contextmanager
def locks(paths):
    with ExitStack() as stack:
        for path in paths:
            handle = stack.enter_context(Path(path).open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('busy lock; no changes applied: ' + str(path)) from exc
        yield


def git(checkout: Path, *args: str) -> bytes:
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    command = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'core.autocrlf=false',
               '-c', 'submodule.recurse=false', '-C', str(checkout), *args]
    result = subprocess.run(command, capture_output=True, env=env, timeout=120)
    if result.returncode:
        raise RuntimeError('git failed: ' + result.stderr.decode(errors='replace').strip())
    return result.stdout


def pull(checkout: Path) -> str:
    if not (checkout / '.git').is_dir() or checkout.is_symlink():
        raise RuntimeError('missing dedicated checkout; see REMOTE_UPDATE.md')
    if git(checkout, 'remote', 'get-url', 'origin').decode().strip() != REPOSITORY:
        raise RuntimeError('unexpected origin; refusing to pull')
    if git(checkout, 'branch', '--show-current').decode().strip() != BRANCH:
        raise RuntimeError('checkout must be on main')
    if git(checkout, 'status', '--porcelain', '--untracked-files=all').strip():
        raise RuntimeError('dedicated checkout has local edits; preserve and review them')
    git(checkout, 'pull', '--ff-only', '--no-recurse-submodules', 'origin', BRANCH)
    head = git(checkout, 'rev-parse', 'HEAD').decode().strip()
    if head != git(checkout, 'rev-parse', 'refs/remotes/origin/main').decode().strip():
        raise RuntimeError('local commits detected; refusing an unpublished update')
    return head


def incoming_files(checkout: Path, commit: str) -> tuple[dict, bytes]:
    result = {}
    entries = git(checkout, 'ls-tree', '-rz', commit, '--', 'flybrain/').split(b'\0')
    for entry in filter(None, entries):
        metadata, raw_name = entry.split(b'\t', 1)
        name = raw_name.decode('utf-8').removeprefix('flybrain/')
        if not (name.endswith('.py') or name == 'drone_profiles.json'):
            continue
        mode, kind, object_id = metadata.decode().split()
        if mode not in ('100644', '100755') or kind != 'blob':
            raise ValueError('non-regular source file: ' + name)
        target(Path('/'), name)
        content = git(checkout, 'cat-file', 'blob', object_id)
        if len(content) > 2_000_000:
            raise ValueError('oversized source file: ' + name)
        if name.endswith('.py'):
            compile(content, name, 'exec')
        else:
            json.loads(content)
        result[name] = content
    requirements = git(checkout, 'show', commit + ':flybrain/requirements.txt')
    return result, requirements


def training_command(arguments: list[str]) -> bool:
    for arg in arguments:
        name = Path(arg).name
        if arg.startswith(('nextgen.', 'x550_')) and not name.endswith('.json'):
            return True
        if name in ('train_fly_brain.py', 'fly_brain.py', 'fly_brain_nextgen.py',
                    'arducopter', 'supervise_nextgen.py', 'run-supervised.sh'):
            return True
        if name.startswith('x550') and name.endswith('.py'):
            return True
    return any(Path(a).name == 'gz' for a in arguments) and 'sim' in arguments


def active_jobs() -> list[dict]:
    rows = []
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        if int(file.parent.name) == os.getpid():
            continue
        try:
            args = file.read_bytes().decode().strip('\0').split('\0')
        except FileNotFoundError:
            continue
        if training_command(args):
            rows.append(dict(pid=int(file.parent.name), command=' '.join(args), system='WSL/Linux'))
    if os.environ.get('WSL_DISTRO_NAME'):
        command = "$p=@(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -match '^(python|pythonw|arducopter|gz)([0-9.]*)?\\.exe$' -and $_.CommandLine -match '(fba|x550|fly_brain)' }); ConvertTo-Json -InputObject $p -Compress"
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                                 "$ErrorActionPreference='Stop'; " + command],
                                capture_output=True, timeout=20, check=True)
        for p in json.loads(result.stdout.decode('utf-8-sig').strip() or '[]'):
            rows.append(dict(pid=p['ProcessId'], command=p['CommandLine'], system='Windows'))
    return rows


def plan(root: Path, incoming: dict, baseline: dict) -> dict:
    changes, conflicts, original = [], [], {}
    for name in sorted(set(incoming) | set(baseline)):
        if name not in baseline:
            conflicts.append(name + ': new file needs review/enrollment')
            continue
        if name not in incoming:
            conflicts.append(name + ': removed upstream; local file preserved')
            continue
        path = target(root, name)
        if not path.is_file():
            conflicts.append(name + ': local file missing; no automatic restoration')
            continue
        raw = path.read_bytes()
        original[name] = hashlib.sha256(raw).hexdigest()
        local, remote = digest(raw), digest(incoming[name])
        if local == remote:
            continue
        if local != baseline[name]:
            conflicts.append(name + ': local changes would be overwritten')
        else:
            changes.append(name)
    return dict(changes=changes, conflicts=conflicts, original_sha256=original)


def apply_transaction(root: Path, state_path: Path, incoming: dict, proposal: dict,
                      next_state: dict, backup: Path) -> None:
    if proposal['conflicts']:
        raise RuntimeError('unresolved local/upstream conflicts')
    for name, expected in proposal['original_sha256'].items():
        if hashlib.sha256(target(root, name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('project changed after preview: ' + name)
    backup.mkdir(parents=True, exist_ok=False)
    old_state = state_path.read_bytes()
    originals = {name: target(root, name).read_bytes() for name in proposal['changes']}
    atomic_write(backup / 'state-before.json', old_state)
    for name, data in originals.items():
        atomic_write(backup / 'files' / name, data)
    journal = dict(status='pending', commit=next_state['commit'], files=proposal['changes'])
    write_json(backup / 'transaction.json', journal)
    applied = []
    try:
        for name, data in originals.items():
            if target(root, name).read_bytes() != data:
                raise RuntimeError('concurrent local edit: ' + name)
            atomic_write(target(root, name), incoming[name])
            applied.append(name)
        write_json(state_path, next_state)
        journal['status'] = 'applied'
        write_json(backup / 'transaction.json', journal)
    except BaseException:
        for name in reversed(applied):
            atomic_write(target(root, name), originals[name])
        atomic_write(state_path, old_state)
        journal['status'] = 'rolled_back'
        write_json(backup / 'transaction.json', journal)
        raise


def run(args) -> dict:
    root = Path(args.project).resolve(strict=True)
    admin = root / '.maintenance' / 'github-sync'
    admin.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (admin, admin.parent)):
        raise RuntimeError('administrative directory must not be a symlink')
    state_path = admin / 'state.json'
    with locks([admin / 'sync.lock']):
        commit = pull(admin / 'checkout')
        incoming, requirements = incoming_files(admin / 'checkout', commit)
        jobs = active_jobs()
        report = dict(time_utc=stamp(), repository=REPOSITORY, commit=commit,
                      active_jobs=jobs, applied=False, project=str(root))
        report['missing_local_dependencies'] = [n for n in LOCAL_ONLY if not (root / n).is_file()]
        if args.initialize:
            if state_path.exists():
                raise RuntimeError('baseline already exists; do not reset it to hide local edits')
            if set(incoming) != set(INITIAL_PATHS):
                raise RuntimeError('initial repository file set changed; review enrollment first')
            baseline = {n: digest(data) for n, data in incoming.items()}
            proposal = plan(root, incoming, baseline)
            if proposal['changes'] or proposal['conflicts']:
                raise RuntimeError('initial source mismatch: ' + json.dumps(proposal))
            state = dict(version=1, repository=REPOSITORY, project=str(root), commit=commit,
                         files=baseline, requirements_sha256=digest(requirements))
            write_json(state_path, state)
            report.update(status='baseline_initialized', enrolled_files=len(baseline))
        else:
            state = json.loads(state_path.read_text())
            if state.get('version') != 1 or state.get('repository') != REPOSITORY or state.get('project') != str(root):
                raise RuntimeError('invalid baseline identity')
            proposal = plan(root, incoming, state['files'])
            if digest(requirements) != state['requirements_sha256']:
                proposal['conflicts'].append('requirements.txt changed: review dependencies before applying')
            for journal in (admin / 'backups').glob('*/transaction.json'):
                if json.loads(journal.read_text()).get('status') not in ('applied', 'rolled_back'):
                    proposal['conflicts'].append('unfinished transaction: ' + str(journal))
            report.update(proposal, baseline_commit=state['commit'], status='preview_only')
            if args.apply:
                if args.commit != commit:
                    raise RuntimeError('review the current preview and provide its full --commit SHA')
                if jobs or proposal['conflicts'] or report['missing_local_dependencies']:
                    report['status'] = 'apply_blocked'
                else:
                    with locks(['/tmp/fba-nextgen-training.lock', '/tmp/fba-x550-recurrent.lock']):
                        if active_jobs():
                            raise RuntimeError('training started after preview; nothing applied')
                        next_state = dict(state, commit=commit,
                                          files={n: digest(data) for n, data in incoming.items()})
                        backup = admin / 'backups' / stamp()
                        apply_transaction(root, state_path, incoming, proposal, next_state, backup)
                        report.update(status='applied', applied=True, backup=str(backup))
            report['checkpoint_notice'] = 'Source changes may invalidate exact resume. No checkpoint checks, stop markers or trainers are modified.'
        write_json(admin / 'latest-report.json', report)
        write_json(admin / 'reports' / (report['time_utc'] + '.json'), report)
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='/mnt/c/fba')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--initialize', action='store_true')
    mode.add_argument('--check', action='store_true', help='default: pull and preview only')
    mode.add_argument('--apply', action='store_true', help='apply a reviewed commit only when idle')
    parser.add_argument('--commit', help='full commit SHA from a reviewed preview')
    args = parser.parse_args(argv)
    try:
        report = run(args)
        print(json.dumps(report, indent=2))
        return 3 if report['status'] == 'apply_blocked' else 0
    except Exception as exc:
        print(json.dumps(dict(status='error', error=str(exc)), indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
