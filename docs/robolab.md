# RoboLab DROID evaluation

FluxThemis runs RoboLab environments and executes environment-unit actions.
FluxVLA loads the policy, preprocesses observations and produces actions.
Use separate prepared Python environments for the model and simulator.

## Requirements and installation

The adapter targets RoboLab DROID joint-position tasks with two cameras.
Its runtime interface is pinned to RoboLab
`9cce00cadd5dc575fd2b8ea3836f5bdcde574229`, Python 3.11, IsaacLab 2.2 and
Isaac Sim 5.0. Follow RoboLab's installation and Git LFS asset instructions
before installing Themis. A CUDA-capable GPU and working Vulkan/RTX driver
are required for simulation; CPU tests do not exercise the renderer.

In the prepared **simulator** environment:

```bash
cd /path/to/FluxThemis
python -m ensurepip --upgrade  # Only if this environment has no pip.
python -m pip install -e .
python -m pip check
```

Themis declares its shared dependencies in `requirements.txt`, including
`mmengine==0.10.7` and `pyzmq>=25,<27`. The RoboLab environment also supplies
Torch, OpenCV, h5py and SciPy. Check dependency resolution before changing an
existing simulator environment; a dedicated environment protects other jobs.
Do not add the model environment's entire site-packages to the simulator path.

The model environment requires a companion FluxVLA version providing
`RoboLabEvalDataset`, the DROID pi0.5 inference recipe,
`episode_noise_stream_v1` serving and RoboLab evaluation reporting.
Verify those capabilities in the selected FluxVLA checkout; an older main
branch may not contain them.

For pi0.5, prepare three matching local assets:

| Environment variable | Required asset |
| --- | --- |
| `FLUXVLA_ROBOLAB_CHECKPOINT` | FluxVLA-compatible `model.safetensors` |
| `FLUXVLA_ROBOLAB_STATS` | FluxVLA `dataset_statistics.json` with `robolab_droid` state/action statistics |
| `FLUXVLA_ROBOLAB_TOKENIZER` | Matching `paligemma_tokenizer.model` |

The statistics asset is not the original OpenPI `norm_stats.json`. Model
conversion, tokenizer, normalization, precision and action horizon belong
to the selected FluxVLA recipe. Themis neither downloads nor converts weights.
Other model families need their own compatible recipes.

[RoboLab's pinned pi0-family instructions](https://github.com/NVlabs/RoboLab/blob/9cce00cadd5dc575fd2b8ea3836f5bdcde574229/policies/pi0_family/README.md)
identify `gs://openpi-assets-simeval/pi05_droid_jointpos` as the source checkpoint.
The matching tokenizer is `gs://big_vision/paligemma_tokenizer.model`.
Obtain a converted `model.safetensors` with a documented conversion/precision
recipe from the companion FluxVLA release. The source JAX checkpoint is not
directly accepted by this service. This guide begins with those prepared assets;
publishing Themis alone does not make the companion code or converted model
available in the two repositories' main branches.

Convert the matching checkpoint's `assets/droid/norm_stats.json` to the small
FluxVLA statistics file without changing the original:

```bash
python - /path/to/norm_stats.json /path/to/dataset_statistics.json <<'PY'
import json
import sys
from pathlib import Path
source = json.loads(Path(sys.argv[1]).read_text())["norm_stats"]
result = {"robolab_droid": {
    target: {key: source[original][key][:8]
             for key in ("mean", "std", "q01", "q99")}
    for target, original in (("state", "state"), ("action", "actions"))
}}
with Path(sys.argv[2]).open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
PY
```

This mapping is specific to the supplied DROID joint-position recipe. Keep its
checkpoint, statistics and tokenizer together; another policy needs its own
asset and preprocessing contract.

## Configuration

| File | Default scope |
| --- | --- |
| `configs/robolab/droid.py` | All benchmark tasks, one environment, one episode per task |
| `configs/robolab/native_batch.py` | All benchmark tasks, ten native environments and ten episodes per task |
| `configs/robolab/remote.yaml` | One BananaInBowlTask episode on one simulator worker |

Copy `remote.yaml` into a deployment directory. Set `simulator.robolab_root`,
`simulator.python`, `simulator.config`, GPU allocation, `model.host`/`port`
and `run.output_root`. Relative paths resolve against that YAML, not the
current working directory; `~` and defined environment variables are expanded.
The selected simulator interpreter must contain Themis and RoboLab dependencies.

Use an inherited Python file for deployment-only settings, for example:

```python
_base_ = '/path/to/FluxThemis/configs/robolab/native_batch.py'
themis = dict(evaluation=dict(environment=dict(
    # Replace these tokens and graphics ordinals after checking your host.
    render_device_map={'2': 1, '5': 0},
    kit_cache_root='/path/to/writable-kit-cache',
    cpu_thread_limit=6,
    rendering_mode='balanced',
    # Include --allow-root only when running Kit as root in a container.
    kit_args='--allow-root --/renderer/multiGpu/enabled=false',
)))
```

Point `simulator.config` to that file and use the same assigned GPU tokens,
such as `--gpu-ids 2,5`. The example numbers are not an automatic mapping.

Compare device UUIDs before filling in the map:

```bash
nvidia-smi --query-gpu=index,uuid --format=csv
vulkaninfo --summary
```

Match NVIDIA UUIDs (without the `GPU-` prefix) to Vulkan `deviceUUID` values,
then use the corresponding Vulkan `GPU` ordinal as the map value.

- `device_id` is a CUDA ordinal visible to the worker; an isolated GPU is CUDA 0.
- `render_device_id` optionally selects one Vulkan graphics ordinal.
- `render_device_map` instead maps an assigned GPU index string or GPU UUID
  to its verified Vulkan ordinal. Missing entries are errors; do not set both
  rendering options. CUDA and Vulkan numbering need not agree.
- `kit_cache_root` creates a distinct worker directory for Kit portable data,
  logs, XDG, OptiX and shader caches before Kit imports. Use a writable path
  without whitespace, separate from HDF5 output. Manual portable-root/log
  arguments conflict with this option. Directories remain after shutdown.
- `cpu_thread_limit` sets supported SimulationApp thread limits. CPU affinity
  and GPU/resource allocation remain deployment responsibilities.
- Put space-separated `kit_args` in the Python file. MMEngine's command-line
  `DictAction` removes spaces, so do not pass such strings via `--cfg-options`.

Check actual GPU UUIDs, renderer assignment, cache directories and resource
use with a small run before increasing worker counts. Defaults preserve the
installed IsaacLab device behavior; they do not discover graphics mappings.

## Resolve and preflight

From the simulator environment, resolve deployment examples without starting
a model, discovering GPUs or launching Isaac:

```bash
python scripts/run_evaluation.py --config /path/to/deployment.yaml --dry-run
python scripts/run_evaluation.py --config /path/to/deployment.yaml --smoke --dry-run
python scripts/run_evaluation.py --config /path/to/deployment.yaml --full --dry-run
```

Dry-run writes the resolved plan and `config_snapshot.py` under the output
root. Check them before real execution. `--smoke` chooses BananaInBowlTask
and one native batch per simulator worker; `--full` selects all tasks while
keeping the Python configuration's quotas. For native batches, per-task quotas
must be multiples of the batch size.

Run static environment/asset preflight against the emitted snapshot:

```bash
PYTHONPATH="$PWD/scripts:$PWD" python -m preflight.environment \
  /path/to/dry-run/config_snapshot.py
```

Preflight checks package metadata, native task registrations, scene paths,
text USD dependencies and unresolved Git LFS pointers without starting Isaac.
It cannot prove binary/transitive asset compatibility or working rendering;
inspect `unchecked_assets` and validate those during real execution.

## Start the model service

In the prepared FluxVLA environment:

```bash
cd /path/to/FluxVLA
export FLUXVLA_ROBOLAB_CHECKPOINT=/path/to/model.safetensors
export FLUXVLA_ROBOLAB_STATS=/path/to/dataset_statistics.json
export FLUXVLA_ROBOLAB_TOKENIZER=/path/to/paligemma_tokenizer.model
CUDA_VISIBLE_DEVICES=0 python scripts/distributed_zmq_server.py \
  --config configs/pi05/pi05_paligemma_robolab_droid_jointpos_inference.py \
  --frontend-bind tcp://127.0.0.1:15555 \
  --backend-bind tcp://127.0.0.1:15556 \
  --backend-endpoint tcp://127.0.0.1:15556 \
  --cfg-options \
  themis.server.evaluation_reporting.result_output_dir=/path/to/model-results
```

For multiple model workers, use Torch's launcher with the same service options:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 --master_port=15557 \
  scripts/distributed_zmq_server.py \
  --config configs/pi05/pi05_paligemma_robolab_droid_jointpos_inference.py \
  --frontend-bind tcp://127.0.0.1:15555 \
  --backend-bind tcp://127.0.0.1:15556 \
  --backend-endpoint tcp://127.0.0.1:15556 \
  --cfg-options \
  themis.server.evaluation_reporting.result_output_dir=/path/to/model-results
```

Model and simulator GPU allocations are independent. Choose worker counts
after checking startup memory and CPU limits; loading several checkpoints at
once can have a larger memory peak than inference. Distributed model ranks
must join initialization together; waiting for the first rank to become ready
before starting the remaining ranks can block distributed initialization.
Use fewer model workers when simultaneous loading exceeds available memory.
For separate hosts, bind
the frontend to a reachable interface and set YAML `model.host` accordingly.
Keep the model rendezvous, model frontend/backend and simulator ports distinct.
Themis defaults to coordinator port 15558 and simulator torchrun port 15559.

The service exits after accepting `run_end`. Start a fresh service and wait
for readiness before each independent evaluation, including a recovered run.
Keep the model recipe's existing loading and precision settings unchanged.

## Run simulation

From FluxThemis in the simulator environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ROBOLAB_ROOT=/path/to/RoboLab
python scripts/run_evaluation.py --config /path/to/deployment.yaml --smoke \
  --result-pointer /path/to/smoke-location.json
```

The pointer identifies the actual result directory. For a continuous subset,
set YAML `simulator.task` and `simulator.episodes` to `null`, use a deployment
inheriting `droid.py` and run without `--smoke` or `--full`:

```bash
python scripts/run_evaluation.py --config /path/to/deployment.yaml \
  --result-pointer /path/to/subset-location.json \
  --cfg-options \
  'themis.evaluation.environment.task_name=[BananaInBowlTask,MustardInRightBinTask,CannedFoodInBinTask]' \
  themis.evaluation.episodes_per_task_overrides.BananaInBowlTask=2
```

This selects Banana twice, then Mustard and CannedFood once each. Native batch
subsets use the same task override with `native_batch.py` and multiples of ten;
omit the two-episode override above. Use dynamic scheduling, zero simulator
restarts and no video recording for RoboLab native batching.

For all tasks, select `droid.py` or `native_batch.py` through your deployment
config, verify the dry-run quota, then replace `--smoke` with `--full`:

```bash
python scripts/run_evaluation.py --config /path/to/deployment.yaml --full \
  --gpu-ids 0,1 --workers-per-gpu 1 --scheduling dynamic \
  --result-pointer /path/to/full-location.json
```

`--full` describes requested scope, not a guarantee of successful completion.
The launcher reports an incomplete run as an error. Startup readiness timeout,
wall-clock episode timeout and native task step budgets are separate controls.

## Observations, actions and lifecycle

The default observation contains `over_shoulder_left_camera`, `wrist_cam`,
and `states`: seven arm joint positions followed by one gripper position.
Images retain native RGB values, and returned arrays own their data. Models
receive the native task instruction. Camera/field options rename supported
fields; they do not add a new native camera preset or change control semantics.

Actions are finite numeric chunks shaped `[T, 8]`: seven absolute joint-position
targets and one gripper target. The adapter executes each action once without
model-specific resizing, normalization, delta conversion or horizon assumptions.
`execute_horizon=None` consumes the horizon supplied by the model.

Each worker owns one SimulationApp, reuses the environment for the same task,
and closes it when switching tasks. Reset follows native order:
`reset_eval_state()` then two resets, taking the second observation. Native
task termination/results determine success. Outer step limits are guards;
an exhausted guard without native results is an error, not a normal failure.

Native batches contain one task and independent policy clients/action queues
per lane. A finished lane freezes while others continue; ten simulated lanes
do not imply one model batch of ten. Canonical task/episode identities and
seeds remain independent of dispatch priority.

`simulator_seed` seeds native environment creation. Scheduled
`environment_seed` and case indices identify episodes; resetting uses
RoboLab's native layout logic.

The supplied recipe selects `episode_noise_stream_v1` on both sides. Each
episode begins from its own seed, then carries generator state between action
chunks. Responses must match the episode and sequence before state advances.
The service may route consecutive requests to different workers. Models without
this capability need matching `legacy_episode` settings on both sides; do not
silently change sampling mode while comparing results.

## Results and explicit recovery

Themis writes config/plan snapshots, events, progress, `result.json`, simulator
logs and resource reports. FluxVLA writes the authoritative event journal,
per-task results, `episodes.csv` and aggregate summaries. Keep both result roots.

Native HDF5 export is authoritative. The adapter reads final
`data/demo_i/subtask/score[-1]` and the eight native trajectory metrics with
`control_dt_s = sim.dt * decimation`. Score and success are independent.
Shared batch HDF5 files are identified per episode by `native_hdf5_path` plus
`native_demo_key`; `native_batch_lane` identifies the lane.

Reports distinguish simulation time, wall-clock duration and metric computation
time. EE speed is simulation motion in cm/s, not inference throughput. Undefined
native metrics remain null with a reason and valid count; missing or unfinished
recordings are errors. The companion report supplies exact binomial SR intervals
and task/episode bootstrap Score intervals with its statistical protocol.

RoboLab does not resume live physics/rendering state or automatically restart
workers. The high-level `--resume` / YAML `resume.from_dir` path is rejected.
Explicit recovery instead retains only complete acknowledged native batches,
including successes and failures, and restarts unfinished groups with original
canonical seeds. Stop the source run first and independently verify that model,
assets, task list, quotas, batch size and evaluation settings match.

```bash
python - /path/to/stopped-run /path/to/resume_input.json <<'PY'
import sys
from fluxthemis.engines.runners.resume import atomic_json, build_bundle
atomic_json(sys.argv[2], build_bundle(sys.argv[1]))
PY
python scripts/run_evaluation.py --config /path/to/new-deployment.yaml \
  --cfg-options themis.evaluation.resume_path=/path/to/resume_input.json
```

Use a fresh output root and model service. Reproduce the source run's scope;
use `--full` only if that was its scope. Bundle validation checks identities and
settings, not remote model/asset contents. A local reporter saves the validated
bundle as `resume_input.json` before `run_start`, allowing further recovery.
Conflicting existing input or a write failure aborts startup. Retain source
journals, excluded partial batches and native recordings for reconciliation.

## Development checks

In a core environment, simulation packages are optional for CPU tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q test
pre-commit run --all-files
```

CPU checks cover configuration, optional-import isolation, lane execution,
metrics, noise-state handling, transport errors and recovery. Real GPU checks
must additionally inspect renderer assignment, native records and both reports.
