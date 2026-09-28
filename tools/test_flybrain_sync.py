"""Updater-only tests: temporary dummy files, no Flybrain imports or training."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import flybrain_sync as sync


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.admin = self.root / '.maintenance' / 'github-sync'
        self.admin.mkdir(parents=True)
        self.original = {'nextgen/config.py': b'VALUE = 1\n',
                         'nextgen/features.py': b'FEATURES = ()\n'}
        self.incoming = dict(self.original, **{'nextgen/config.py': b'VALUE = 2\n'})
        for name, data in self.original.items():
            sync.atomic_write(self.root / name, data)
        self.baseline = {n: sync.digest(data) for n, data in self.original.items()}
        self.state_path = self.admin / 'state.json'
        self.state = dict(version=1, project=str(self.root), repository=sync.REPOSITORY,
                          files=self.baseline, commit='a'*40, requirements_sha256=sync.digest(b''))
        sync.write_json(self.state_path, self.state)

    def proposal(self):
        return sync.plan(self.root, self.incoming, self.baseline)

    def test_only_newline_differences_are_ignored(self):
        self.assertEqual(sync.digest(b'x=1\r\n\r\n'), sync.digest(b'x=1\n'))
        self.assertNotEqual(sync.digest(b'x=1 '), sync.digest(b'x=1'))

    def test_clean_update_is_planned_without_writes(self):
        result = self.proposal()
        self.assertEqual(result['changes'], ['nextgen/config.py'])
        self.assertEqual(result['conflicts'], [])
        self.assertEqual((self.root/'nextgen/config.py').read_bytes(), b'VALUE = 1\n')

    def test_local_changes_are_not_overwritten(self):
        (self.root/'nextgen/config.py').write_bytes(b'LOCAL = True\n')
        result = self.proposal()
        self.assertTrue(result['conflicts'])
        self.assertEqual(result['changes'], [])

    def test_converged_local_and_remote_is_noop(self):
        (self.root/'nextgen/config.py').write_bytes(self.incoming['nextgen/config.py'])
        self.assertEqual(self.proposal()['conflicts'], [])
        self.assertEqual(self.proposal()['changes'], [])

    def test_new_and_removed_files_need_review(self):
        self.incoming.pop('nextgen/features.py')
        self.incoming['nextgen/new_module.py'] = b'x=1\n'
        self.assertEqual(len(self.proposal()['conflicts']), 2)
        self.assertTrue((self.root/'nextgen/features.py').exists())

    def test_missing_local_file_is_not_silently_restored(self):
        (self.root/'nextgen/config.py').unlink()
        self.assertTrue(self.proposal()['conflicts'])

    def test_unsafe_and_local_only_paths_rejected(self):
        for name in ('../outside.py', '/tmp/file.py', 'nextgen/runs/a.py',
                     'x550_teachable/runtime.py', 'banc_controller_graph.json',
                     'nextgen/training.stop', '.env', 'nextgen/../config.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                sync.target(self.root, name)

    def test_symlink_destination_rejected(self):
        path = self.root/'nextgen/config.py'
        path.unlink()
        path.symlink_to(self.root/'outside.py')
        with self.assertRaises(ValueError):
            sync.target(self.root, 'nextgen/config.py')

    def test_backup_and_apply_preserve_unmanaged_files(self):
        marker = self.root/'training.stop'
        marker.write_bytes(b'keep')
        backup = self.admin/'backups'/'test'
        sync.apply_transaction(self.root, self.state_path, self.incoming,
                               self.proposal(), dict(self.state, commit='b'*40), backup)
        self.assertEqual((self.root/'nextgen/config.py').read_bytes(), b'VALUE = 2\n')
        self.assertEqual((backup/'files/nextgen/config.py').read_bytes(), b'VALUE = 1\n')
        self.assertEqual(marker.read_bytes(), b'keep')
        self.assertEqual(json.loads((backup/'transaction.json').read_text())['status'], 'applied')

    def test_concurrent_edit_after_preview_is_rejected(self):
        proposal = self.proposal()
        (self.root/'nextgen/config.py').write_bytes(b'new local edit\n')
        with self.assertRaises(RuntimeError):
            sync.apply_transaction(self.root, self.state_path, self.incoming,
                                   proposal, self.state, self.admin/'backups'/'test')
        self.assertEqual((self.root/'nextgen/config.py').read_bytes(), b'new local edit\n')

    def test_failed_write_rolls_back_successful_prior_writes(self):
        self.incoming['nextgen/features.py'] = b'FEATURES = (1,)\n'
        proposal = self.proposal()
        original_state = self.state_path.read_bytes()
        real_write = sync.atomic_write
        failed = [False]
        def fail_once(path, data):
            if path == self.root/'nextgen/features.py' and not failed[0]:
                failed[0] = True
                raise OSError('injected write failure in temporary fixture')
            real_write(path, data)
        with patch.object(sync, 'atomic_write', fail_once), self.assertRaises(OSError):
            sync.apply_transaction(self.root, self.state_path, self.incoming,
                                   proposal, self.state, self.admin/'backups'/'test')
        for name, data in self.original.items():
            self.assertEqual((self.root/name).read_bytes(), data)
        self.assertEqual(self.state_path.read_bytes(), original_state)

    def test_busy_lock_is_respected(self):
        lock = self.admin/'test.lock'
        with sync.locks([lock]):
            with self.assertRaises(RuntimeError):
                with sync.locks([lock]):
                    self.fail('second lock must not be acquired')

    def test_training_processes_are_recognized(self):
        for args in (['python', '-m', 'x550_teachable.train'],
                     ['python', '/mnt/c/fba/x550_continuous.py'],
                     ['python', '-m', 'nextgen.train_safe'], ['gz', 'sim', 'world.sdf']):
            self.assertTrue(sync.training_command(args))
        self.assertFalse(sync.training_command(['python3', 'flybrain_sync.py', '--check']))

    def test_apply_is_blocked_by_active_job_without_writes(self):
        args = SimpleNamespace(project=str(self.root), initialize=False,
                               apply=True, commit='b'*40)
        with patch.object(sync, 'pull', return_value='b'*40), \
             patch.object(sync, 'incoming_files', return_value=(self.incoming, b'')), \
             patch.object(sync, 'active_jobs', return_value=[dict(pid=123, command='test trainer')]):
            report = sync.run(args)
        self.assertEqual(report['status'], 'apply_blocked')
        self.assertFalse(report['applied'])
        self.assertEqual((self.root/'nextgen/config.py').read_bytes(), b'VALUE = 1\n')

    def test_wrong_reviewed_commit_is_rejected(self):
        args = SimpleNamespace(project=str(self.root), initialize=False,
                               apply=True, commit='c'*40)
        with patch.object(sync, 'pull', return_value='b'*40), \
             patch.object(sync, 'incoming_files', return_value=(self.incoming, b'')), \
             patch.object(sync, 'active_jobs', return_value=[]), self.assertRaises(RuntimeError):
            sync.run(args)

    def test_preview_does_not_apply_even_when_idle(self):
        args = SimpleNamespace(project=str(self.root), initialize=False,
                               apply=False, commit=None)
        with patch.object(sync, 'pull', return_value='b'*40), \
             patch.object(sync, 'incoming_files', return_value=(self.incoming, b'')), \
             patch.object(sync, 'active_jobs', return_value=[]):
            report = sync.run(args)
        self.assertEqual(report['status'], 'preview_only')
        self.assertEqual((self.root/'nextgen/config.py').read_bytes(), b'VALUE = 1\n')


if __name__ == '__main__':
    unittest.main()
