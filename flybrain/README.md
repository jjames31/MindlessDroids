# Flybrain

Source-only snapshot of the X550 recurrent navigation research project, staged
September 28, 2026. This directory contains the recurrent model, learner,
simulation environment, sensing, independent safety checks, evaluation, runner,
and regression tests, plus their imported supporting modules.
## Layout

| Path | Purpose |
| --- | --- |
| `x550_teachable/` | Recurrent model, sequential PPO, curriculum, sensing, safety shield, evaluation, runner, and tests |
| `x550_scenes/` | Imported procedural scene and simulation support |
| `nextgen/` | Imported base dynamics, feature schema, configuration, and model support |
| `drone_profiles.json` | Required simulation envelopes; historical profile identifiers are preserved |
| `requirements.txt` | NumPy and PyTorch versions observed in the source environment |

Only the imported supporting modules are included, not the older training
campaigns. The recurrent environment targets X550. Run module commands from this
`flybrain` directory so the existing imports and relative configuration paths
resolve correctly.

## Environment

The source environment used Python 3.10, NumPy 1.26.4, and PyTorch 2.8.0+cu128.
The dependency file pins the base PyTorch release, not a specific CUDA wheel.
The runner requires Linux or WSL because it uses `fcntl` and `/proc`.

```bash
cd flybrain
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Installation alone does not resolve the external asset requirement below. This
package has not been validated as a standalone release.

## Verification status

Current repository inspection confirms that `x550_teachable/runtime.py` is
present and `flybrain/banc_controller_graph.json` is still absent. A lightweight
test-discovery attempt was run from `flybrain` with:

```bash
python3 -m unittest discover x550_teachable/tests
```

In the current shell environment, discovery fails before running tests because
PyTorch is not installed (`ModuleNotFoundError: No module named 'torch'`). This
does not validate runtime behavior, training, checkpoints, or flight readiness.

## External asset and checkpoint compatibility

`banc_controller_graph.json` is omitted from this code-only snapshot. The
unchanged recurrent provenance function reads it, although the recurrent
simulation environment does not directly execute the graph. Training therefore
requires the original trusted graph at `flybrain/banc_controller_graph.json`.
Do not substitute a placeholder or disable the provenance check.

This reduced source set has a different fingerprint from the full local
workspace. Existing full-workspace checkpoints must not be force-resumed by
bypassing the source/configuration checks. Resume checkpoints are trusted-local
artifacts; do not load untrusted checkpoint files.

The legacy `x550_scenes.util.fingerprint()` also references omitted legacy files.
It is not a standalone entry point in this snapshot; the recurrent runner uses
its own fingerprint helper.

## Exclusions and scope

No model weights, run directories, logs, caches, virtual environments, raw
datasets, private settings, or local maintenance records are included. The
`.gitignore` keeps those artifacts out of ordinary future commits.

Sensor and dynamics models are simulation approximations. No live camera
pipeline, VIO, Gazebo bridge, Raspberry Pi benchmark, or aircraft command
publisher is included. Independent safety, stop, and single-learner checks are
preserved. Corrective teacher imitation remains opt-in. Do not run a competing
learner beside the original project or treat simulation tests as flight approval.
