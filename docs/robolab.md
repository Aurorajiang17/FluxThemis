# RoboLab Integration

RoboLab and Isaac Sim run in FluxThemis simulator workers. Themis selects tasks,
advances native environments and reports results; FluxVLA model workers receive
observations over ZMQ and return environment-unit actions. Simulator assets
belong on the simulator side; weights, tokenizer and statistics belong on the
model side. Each side uses its own prepared Python environment.

- [Environment and model preparation](#environment-and-model-preparation)
- [Evaluation data and quotas](#evaluation-data-and-quotas)
- [Launch and network](#launch-and-network)
- [Progress and results](#progress-and-results)
- [Compatibility and troubleshooting](#compatibility-and-troubleshooting)

## Environment and Model Preparation

Cloud model and simulator tasks use the Python environment from their own image.
Simulation requires an NVIDIA RTX/RT-capable GPU and working Vulkan drivers;
compute-only GPUs such as A100/A800 can run the model. Shared code can be
installed into a task's environment independently of the model environment.

### Install

Prepare Git, Git LFS, [uv](https://docs.astral.sh/uv/getting-started/installation/),
ffmpeg and the host's graphics drivers, then run:

```bash
cd /root/projects/FluxThemis
bash scripts/install_robolab.sh --dry-run
bash scripts/install_robolab.sh
```

The [installer](../scripts/install_robolab.sh) creates a sibling RoboLab checkout
at revision `9cce00cadd5dc575fd2b8ea3836f5bdcde574229`, uses its official locked
`uv sync --extra isaac50` workflow, and installs RoboLab and FluxThemis into the
same simulator environment. The default is `RoboLab/.venv`: Python 3.11,
IsaacLab 2.2.0 and Isaac Sim 5.0.0. Git LFS downloads only `assets/**`.
Package consistency and static task/asset checks run after installation.
The installer does not start a simulator or model.

Existing simulator images can supply their own compatible Python:

```bash
bash scripts/install_robolab.sh \
  --robolab-root /path/to/RoboLab --python /path/to/simulator-env/bin/python
```

Use `--env-path` to place a uv environment elsewhere, or `--python` to add the
packages to an already prepared environment without running uv sync. These
options are mutually exclusive. `--skip-assets` skips both downloading and
static asset verification; prepare and verify those assets before evaluation.
Existing checkouts must match `ROBOLAB_REF` and have no tracked edits. The
installer preserves their revision and refuses conflicting sources or Python
versions. `ROBOLAB_ROOT`, `ROBOLAB_REPO` and `ROBOLAB_REF` select source paths
and revisions; other revisions require separate compatibility validation.

Activate the default simulator environment or select it explicitly:

```bash
export ROBOLAB_ROOT=/root/projects/RoboLab
export FLUXTHEMIS_PYTHON="$ROBOLAB_ROOT/.venv/bin/python"
source "$ROBOLAB_ROOT/.venv/bin/activate"
```

Use this Python directly after installing Themis. A later exact `uv sync` can
remove packages added outside RoboLab's lockfile; rerun the installer to restore
the combined environment. Both environments need compatible ZMQ protocols;
Themis installs its shared Python dependencies from `requirements.txt`.

### Required Configuration

- The simulator uses the [RoboLab configs](../configs/robolab). The DROID
  adapter accepts seven absolute arm joint targets and one gripper target,
  with `over_shoulder_left_camera`, `wrist_cam` and `states` observations.
- The selected FluxVLA checkout must provide the RoboLab dataset/recipes,
  evaluation reporting and the matching `process_noise_stream_v1` protocol.
  Process evaluation also requires the companion runtime controls. An older FluxVLA main branch may not provide
  these capabilities; record both repositories' selected revisions.
- For pi0.5, use the matching FluxVLA-compatible checkpoint, statistics and
  tokenizer with the DROID inference recipe:

  | Variable | Model-side asset |
  | --- | --- |
  | `FLUXVLA_ROBOLAB_CHECKPOINT` | `model.safetensors` |
  | `FLUXVLA_ROBOLAB_STATS` | `dataset_statistics.json` with the `robolab_droid` key |
  | `FLUXVLA_ROBOLAB_TOKENIZER` | Matching `paligemma_tokenizer.model` |

  [RoboLab's pi0-family guide](https://github.com/NVlabs/RoboLab/blob/9cce00cadd5dc575fd2b8ea3836f5bdcde574229/policies/pi0_family/README.md)
  identifies the source checkpoint. Prepare the converted weights and statistics
  according to the companion FluxVLA recipe; a JAX checkpoint or OpenPI
  `norm_stats.json` is not a drop-in replacement for these files.
- Model loading, precision, normalization, tokenizer and action horizon belong
  to the model recipe. Other policies, including trained checkpoints, supply
  their own matching recipe and convert outputs to the DROID action space.
  Themis does not impose a model-internal dimension or prediction horizon.

## Evaluation Data and Quotas

### Assets and Storage

Native tasks, scenes, objects, robots and backgrounds come from the selected
[RoboLab checkout](https://github.com/NVlabs/RoboLab/tree/9cce00cadd5dc575fd2b8ea3836f5bdcde574229).
The installer materializes its LFS assets; training demonstrations and model
checkpoints are not part of that download. Allocate storage separately for the
Isaac environment, model assets, runtime caches and HDF5 results.

Static preflight rejects missing required files and Git LFS pointers. It does
not establish CUDA/Vulkan compatibility or resolve every binary/transitive
USD/MDL reference; inspect its `unchecked_assets` and validate rendering on
the target host. The [native task list](https://github.com/NVlabs/RoboLab/blob/9cce00cadd5dc575fd2b8ea3836f5bdcde574229/robolab/tasks/README.md)
describes the pinned 120-task benchmark.

### Episode Quotas

The [DROID config](../configs/robolab/droid.py) uses ten native environments
per worker and ten episodes per task: 1,200 episodes for the pinned benchmark.
The [process template](../configs/robolab/process.yaml) selects all installed
tasks, one logical stream, one GPU and **startup seed 0**. The model seeds once
after loading and advances its default device stream across lanes, episodes
and tasks. Native environment creation uses `simulator_seed=0`.

Canonical report seeds identify episodes independently of dispatch order.
Double reset, same-task reuse, native step limits and success rules are
preserved. `execute_horizon=None` consumes the model's returned action chunk.

## Launch and Network

### Default Process Workflow

The process entry manages a model service and simulator for each logical stream
on one host, using their separate Python environments. It calls the existing
public FluxVLA/Themis CLIs. Set the model asset variables above and these paths:

```bash
export FLUXVLA_ROOT=/root/projects/FluxVLA
export FLUXVLA_PYTHON=/path/to/model-env/bin/python
export ROBOLAB_ROOT=/root/projects/RoboLab
export FLUXTHEMIS_PYTHON="$ROBOLAB_ROOT/.venv/bin/python"
export ROBOLAB_EVAL_OUTPUT=/path/to/results
export ROBOLAB_EVAL_CACHE=/path/to/cache
cd /root/projects/FluxThemis
```

Accept NVIDIA's simulator license before setting `OMNI_KIT_ACCEPT_EULA=YES`.
When running as root, set `allow_root: true` in your process YAML. Check its
GPU/Vulkan mapping, CPU allocation and paths for the target host first.

Inspect one-task settings, then execute its ten native episodes:

```bash
"$FLUXTHEMIS_PYTHON" scripts/run_robolab_process.py \
  --tasks BananaInBowlTask --dry-run
"$FLUXTHEMIS_PYTHON" scripts/run_robolab_process.py --tasks BananaInBowlTask
```

Dry-run writes the plan, expanded configs and commands without starting child
processes, contacting services, initializing CUDA or allocating simulator
caches. Removing `--tasks` selects the full benchmark:

```bash
"$FLUXTHEMIS_PYTHON" scripts/run_robolab_process.py
```

Task discovery is deterministic. `--num-streams` partitions the ordered task
list into persistent model streams; physical GPUs can execute several complete
streams sequentially. A two-GPU example, after verifying graphics ordinals:

```bash
"$FLUXTHEMIS_PYTHON" scripts/run_robolab_process.py \
  --num-streams 2 --gpu-ids 0,1 --render-devices 0 1 --dry-run
```

The graphics ordinals are examples, not automatic discovery. Copy
`process.yaml` to configure paths, CPU/port ranges, timeouts and `allow_root`.
Relative paths resolve against that YAML. Set `simulator_config` to an absolute
path when moving the template. `model_env` and `simulator_env` carry declared
asset or binary dependency settings, without replacing Python imports or GPU
routing. The default assigns two model CPUs and six simulator CPUs per GPU.

An optional `streams` JSON manifest fixes canonical tasks and dispatch order:

```json
{
  "version": 1,
  "canonical_tasks": ["BananaInBowlTask", "MustardInRightBinTask"],
  "episode_batch_size": 10,
  "report_base_seed": 0,
  "streams": [
    {"id": 0, "tasks": ["MustardInRightBinTask"]},
    {"id": 1, "tasks": ["BananaInBowlTask"]}
  ]
}
```

Each task must appear exactly once; an explicit manifest cannot be combined
with `--tasks` or `--num-streams`. `--process-seeds 0 1` selects two independent
repetitions. Even logical streams use the supplied seed order, odd streams the
reverse order; each stream exits before the next repetition starts on its GPU.
Changing stream grouping or request order changes noise allocation.

### Network

The process entry creates independent loopback endpoints on one host. Each
stream has one persistent model service and one simulator worker, using their
own Python environments. Keep the generated model frontend/backend, simulator
coordinator and rendezvous ports available and distinct.

## Progress and Results

Themis writes plans/config snapshots, events, progress, `result.json`, simulator
logs and resource reports. FluxVLA writes the authoritative journal, per-task
results and aggregate summaries. Keep both output roots. The process launcher
checks complete task/lane coverage, canonical identities, matching journals,
finite Scores and the model's continuous request sequence; `summary.json`
contains results for each selected startup seed.

Native HDF5 determines Score and trajectory metrics. Batch episodes are
identified by `native_hdf5_path` and `native_demo_key`. Score and success are
independent; EE speed uses simulation time rather than inference wall time.
Undefined native metrics retain null values and reasons. Missing or unfinished
recordings are errors, and missing episodes remain unscored.

If a stream encounters a technical error, its processes stop and partial
outputs are retained; other healthy streams continue. Missing episodes remain
unscored, and an incomplete evaluation exits unsuccessfully. Automatic retries
and resume are unsupported. Normal task failures remain valid results.

## Compatibility and Troubleshooting

- **Simulator versions:** the installer selects Python 3.11, IsaacLab 2.2.0
  and Isaac Sim 5.0.0. Other Isaac versions may change physics and rendering;
  they are not silently substituted by this installer.
- **CUDA and Vulkan:** match `nvidia-smi --query-gpu=index,uuid --format=csv`
  against `vulkaninfo --summary`. An isolated worker uses CUDA ordinal 0,
  but its graphics ordinal can differ. `render_device_map` maps assigned GPU
  tokens/UUIDs to verified Vulkan ordinals; missing entries are errors.
- **Worker caches and threads:** `kit_cache_root` creates isolated worker
  caches and must be writable, whitespace-free and separate from HDF5 output.
  `cpu_thread_limit` controls supported simulator thread pools; CPU affinity
  and resource admission remain deployment responsibilities.
- **Kit arguments:** place space-separated `kit_args` in an inherited Python
  config, for example `--allow-root --/renderer/multiGpu/enabled=false` when
  running Kit as root. MMEngine's `DictAction` removes spaces, so these strings
  should not be passed through `--cfg-options`. The process YAML exposes
  `allow_root` for this setting.
- **Native lifecycle:** workers reuse a task's environment, close it on task
  changes, and perform `reset_eval_state()` followed by two resets. Finished
  batch lanes freeze while others continue. Native results determine success;
  reaching an outer guard without native results is an error.
- **CPU versus runtime checks:** core tests and static preflight do not prove
  rendering or full evaluation on another host. After installation, run a small
  selected task on that host and inspect native recordings and both reports.

Development checks use the existing repository commands:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q test
pre-commit run --all-files
```
