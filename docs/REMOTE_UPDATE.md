# Updating the existing Flybrain workspace from GitHub

This workflow pulls `jjames31/MindlessDroids`, branch `main`, into a separate
checkout, then previews the mapping from `flybrain/` to the existing workspace.
It does not replace the workspace with the incomplete repository snapshot.

## Ask Remote Desktop Commander

> Update my Flybrain code from MindlessDroids. Read
> `C:\fba\GITHUB-UPDATE.md` and `C:\fba\REMOTE-DESKTOP-COMMANDER.txt`, run
> `C:\fba\Pull-Flybrain.cmd --check`, and review the commit and changed files.
> If training is idle and there are no conflicts, apply that exact reviewed
> commit. If training is running, download and report only. Preserve local-only
> files, checkpoints, logs and stop markers. Do not stop or restart training.

A request to pull is not permission to stop an active experiment, overwrite
uncommitted local changes, remove files, disable a safety check, or resume an
incompatible checkpoint. Changed navigation/safety behavior still needs review
and validation; a successful file transfer is not flight qualification.

## Commands on the PC

```bat
C:\fba\Pull-Flybrain.cmd --check
C:\fba\Pull-Flybrain.cmd --apply --commit FULL_COMMIT_SHA_FROM_REVIEWED_PREVIEW
```

No arguments means preview only. `--check` performs `git pull --ff-only` in the
separate checkout and prints the current commit, proposed changes, conflicts,
missing local dependencies, and live process information. It does not update
working-project code. If the upstream commit advances between review and apply,
rerun the preview and review the new commit rather than substituting a new SHA
without inspection.

`--apply` checks active jobs again, obtains the existing learner locks, verifies
that local files still match the preview, and backs up the affected files before
replacing them. It has no force option. Incoming Python is syntax-checked, but
not imported or executed. Requirements changes and new or deleted source files
require a separate review and explicit enrollment; they are not silently applied.

Inspect the exact source diff in the dedicated checkout before applying:

```bash
git -C /mnt/c/fba/.maintenance/github-sync/checkout diff BASELINE_COMMIT REVIEWED_COMMIT -- flybrain
```

Use the two full commit IDs printed by the preview. Safety, provenance, or
checkpoint checks must not be weakened simply to make an update pass.

## Locations

| Purpose | PC location |
| --- | --- |
| User entry point | `C:\fba\Pull-Flybrain.cmd` |
| Installed, reviewed helper | `C:\fba\.maintenance\github-sync\bin\flybrain_sync.py` |
| Separate Git checkout | `C:\fba\.maintenance\github-sync\checkout` |
| Managed-file baseline | `C:\fba\.maintenance\github-sync\state.json` |
| Most recent successful report write | `C:\fba\.maintenance\github-sync\latest-report.json` |
| Historical reports | `C:\fba\.maintenance\github-sync\reports` |
| Per-update backups and transaction journals | `C:\fba\.maintenance\github-sync\backups` |

Check the command's exit code and printed output. A fatal error may occur before
`latest-report.json` is refreshed; do not mistake an old report for this run.
Exit code 0 means the requested preview/initialization/apply completed; code 3
means an apply was blocked; code 1 means an error. Read `status` and `applied`
in the report rather than treating a successful preview as an applied update.

## Files preserved and limitations

The initial managed set is the 28 published source/configuration files listed
in `tools/flybrain_sync.py`. Files elsewhere in the local workspace are never
removed to mirror the repository. The local `x550_teachable/runtime.py` and
`banc_controller_graph.json` remain outside this sync set and are not overwritten,
uploaded, or replaced by placeholders. The repository remains incomplete as a
standalone release. No prior rejected runtime upload is retried by this workflow.

Local edits cause a conflict instead of being overwritten. Resolve the specific
diff and record the decision before adjusting enrollment. Do not delete the
baseline and initialize again to hide local changes. The updater never pushes
local work to GitHub or installs packages.

No checkpoint, optimizer state, log, run directory, model weight, stop marker,
training launcher, or source-fingerprint guard is modified by applying source
updates. Source changes can invalidate exact checkpoint resume. Keep the old
source and checkpoint together; plan a separately authorized compatible restart
rather than rewriting stored fingerprints or clearing stop requests.

Backups contain previous file bytes and `state-before.json`. Caught write errors
attempt rollback. An unfinished transaction after a crash blocks subsequent
updates until reviewed recovery is completed. Recovery must respect the same
idle-process and local-edit checks. The update is not a multi-file filesystem
transaction, and arbitrary external programs that ignore the learner locks are
not controlled by this helper.

Changes to the helper under `tools/` are downloaded but are NOT automatically
installed or executed. Review and test a helper revision separately, then replace
the installed copy explicitly. This prevents a routine project-code pull from
silently replacing its own update controls.

## Initial installation / recovery reference

The configured PC uses WSL distribution `Ubuntu-22.04`, user `jaco`, project
`/mnt/c/fba`, and the existing system `python3` and Git. The helper uses only the
Python standard library. Do not install dependencies for it.

Clone into a NEW dedicated checkout only. Never clone over the live workspace,
run `git reset --hard`, or use a destructive directory mirror against `C:\fba`.
After reviewing the helper, copy it to the installed `bin` location and copy the
CMD entry point to the project root. Then run this once:

```bat
C:\fba\Pull-Flybrain.cmd --initialize
```

Initialization writes only administrative metadata after confirming that all 28
published files match their local counterparts, ignoring only line-ending and
trailing-newline differences. It refuses to replace an existing baseline.

Updater-only regression tests use temporary dummy files, not navigation code:

```bash
cd /mnt/c/fba/.maintenance/github-sync/checkout
python3 -B -m unittest discover -s tools -p test_flybrain_sync.py -v
```

Git reference: https://git-scm.com/docs/git-pull
