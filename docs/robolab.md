# RoboLab DROID environment adapter

`RoboLabEnvironment` connects the official RoboLab DROID joint-position
environment to the existing Themis `BaseEnvironment` contract. The standalone
configuration is [`configs/robolab/droid.py`](../configs/robolab/droid.py).
It selects all benchmark tasks with one episode per task, following the official
CLI defaults (`num_envs=1`, `num_runs=1`). Small development selections belong in
[`remote.yaml`](../configs/robolab/remote.yaml) or runtime overrides.
It does not define model
weights, tokenizers, normalization, inference precision or a fixed action chunk
length. Future policies can use the same environment-unit interface after their
own input/output recipes have been validated.

## Scope and validation status

The adapter has CPU contract tests, fixed-action replay against the official
episode runner, real-service smoke/continuous checks, a complete 120 x 1 check,
and a fault-recovered 120 x 10 evaluation. Single-worker native batch
qualification completed 30/30 episodes (29 successes); six-worker qualification
completed 60/60 (34 successes). These qualification episodes are excluded from
the 1,200-episode result. See [evaluation results](#full-suite-development-check)
and [recovered evaluation](#fault-recovered-120-x-10-evaluation).

The recovered evaluation audited 120 tasks with ten episodes each, including
native HDF5, both journals, sampling-state chains and trajectory metrics. Earlier
attempts encountered nonfinite simulation states; the final retry ran cleanly,
but the numerical instability's root cause remains unresolved. The new optional
device-map/cache helpers have CPU coverage only; that evaluation used explicit
deployment wrappers and does not qualify these helpers on real GPUs.

## Native multi-environment execution

[`configs/robolab/native_batch.py`](../configs/robolab/native_batch.py) opts into
ten native environments per simulation worker and ten episodes per task. Each
worker advances one task's environments together, with a separate policy client,
action queue and continuous sampling state for every lane. Model requests remain
single-observation requests. Configure `simulator.simulator_restart_limit: 0`,
dynamic scheduling and `save_videos: false` in the launch YAML. Batch evaluation
requires task quotas divisible by `episode_batch_size`. Explicit recovery reuses
only complete native batches and restarts incomplete batches with their original
seeds. Partial-batch results remain in the source evidence and are listed in
`native_batch_recovery.excluded_partial_group_results`; they are not selected by
success value. Automatic worker restart and video recording remain rejected.
The default `droid.py` retains one environment and one episode per task.

The coordinator builds canonical task-major episode identities and seeds before
scheduling. Optional `themis.evaluation.task_schedule` lists every selected task
once in dispatch-priority order without changing those seeds. A worker receives
a full group from one task; terminal lanes report immediately, but the worker
is available for another group only after all results have been acknowledged.
No padded or replacement episodes are added.

RoboLab's double reset, original limits, freezing and success predicates remain
native. One batch writes `run_N.hdf5` with `demo_0` through `demo_9`; reports expose
`native_hdf5_path`, `native_demo_key` and `native_batch_lane`. Identify a recording
by both path and demo key. Each lane's Score and trajectory metrics use its own
finalized demo. Wall-clock episode durations overlap and include other lanes'
inference waits; summing them does not measure job elapsed time. Simulation time
and EE Speed keep the original per-episode definitions.

The source reference is RoboLab commit
`9cce00cadd5dc575fd2b8ea3836f5bdcde574229`. Its development runtime uses
Python 3.11, IsaacLab 2.2, Isaac Sim 5.0 and Torch 2.7. Other runtime combinations
have not been validated for this adapter. Install Themis's normal lightweight
dependencies in the simulation interpreter as well; the RoboLab interpreter
alone may lack `mmengine`. No Isaac/RoboLab package becomes a mandatory Themis
dependency.

This adapter does not import RoboLab, Isaac Sim, Torch or OpenPI at module scope
or during construction. MMEngine may probe optional Torch support when building
any registered class; its existing fallback handles missing Torch. As in the
[RoboDojo adapter](robodojo.md), simulator loading happens in `setup()`:

1. Resolve optional `robolab_root` / `ROBOLAB_ROOT` in the worker process.
2. Import OpenCV before IsaacLab's `AppLauncher` and start a camera-enabled app.
3. Import RoboLab runtime/registration modules after the app starts.
4. Register the selected DROID joint-position tasks with the default background.

The adapter does not import the official policy entry point, which parses CLI
arguments and launches an app at module scope. Its imports follow that entry
point's ordering without starting a second app or loading a policy client.

## Configuration and task identity

`task_name` is a base task name, a list of base task names in execution order,
or `all` for the official `benchmark` task directory. RoboLab's factory resolves
these names into registered environment IDs. Themis `TaskSpec.task_id` is the
resolved ID; `metadata.task_name` retains the base task name. These can differ
when native registrations have variants. Background randomization and alternate
robot/control registrations are outside this first recipe.

`camera_keys` maps output names to native `image_obs` keys. `state_keys` names
the native arm and gripper fields, in that order. These options rename fields;
the DROID joint-position contract is still seven arm joints and one gripper.
`renderer` and `rendering_mode` are passed to the official factory. Device
assignment uses the existing worker `configure_simulator_device()` interface.

`device_id` is the CUDA ordinal visible to the worker. `render_device_id` is
an optional Vulkan/graphics ordinal; when omitted, IsaacLab chooses it from the
CUDA device as usual. These ordinals can differ when `CUDA_VISIBLE_DEVICES`
isolates a GPU: in the validated machine, physical GPU 6 became CUDA device 0
while retaining graphics ordinal 6. The explicit override is applied after
IsaacLab resolves its devices and before the application allocates resources.
It relies on the pinned IsaacLab app creation interface and rejects an unknown
interface. Automatic multi-GPU rendering is disabled for each worker.

`kit_args` forwards deployment-specific Kit arguments, for example rendering
and thread limits. Neither a particular GPU number nor machine paths are fixed
in the Python recipe. A launcher must verify the actual device UUID and renderer
assignment for its machine; the shared worker's CUDA assignment alone does not
resolve graphics ordinal differences.

The worker callback accepts the assigned physical GPU index or UUID and maps it
to its position in `CUDA_VISIBLE_DEVICES`; a worker isolated to one GPU uses
CUDA 0. An assignment outside that list is rejected. This callback preserves
the separately configured `render_device_id`.

For multiple workers, `render_device_map` optionally maps the exact assigned
GPU token (an index string or UUID) to its verified Vulkan ordinal. CUDA and
Vulkan numbering are resolved separately; no equality or rank-based ordering
is assumed. Set either this map or the single `render_device_id`, not both.
A missing assignment or map entry is rejected before Kit loads. Without either
override, the installed IsaacLab behavior is unchanged.

`kit_cache_root` optionally creates a unique run/worker directory before Kit is
imported, containing its portable root, logs, XDG cache/config/data, OptiX and
shader caches. The process environment is restored on normal close or startup
failure; directories remain for deployment-controlled cleanup. Use a writable
root separate from native HDF5 output. Because the pinned IsaacLab parser splits
Kit arguments on whitespace, this root must not contain whitespace. Supplying
manual `--portable-root` or `--/app/tokens/omni_global_logs` together with
`kit_cache_root` is rejected instead of silently overriding either setting.

For example, an inherited deployment Python config can contain:

```python
_base_ = '/path/to/FluxThemis/configs/robolab/native_batch.py'
themis = dict(evaluation=dict(environment=dict(
    # Example only: replace with the mapping verified on your host.
    render_device_map={'2': 1, '5': 0},
    kit_cache_root='/path/to/writable-kit-cache',
)))
```

Point the deployment YAML's `simulator.config` at this file and select the same
GPU tokens (`--gpu-ids 2,5` in this example). Multiple workers on the same GPU
share its graphics ordinal but receive different cache directories. No GPU
UUID, CPU affinity or host path is fixed in the public recipe. Verify actual
rendering devices, cache paths and resource use in a small real deployment
before scaling; CPU tests alone cannot validate Vulkan assignment.

## Remote configuration and preflight

The split deployment follows [RoboDojo](robodojo.md): start FluxVLA separately,
then use Themis to connect to its reachable ZMQ endpoint. The YAML references the
model-independent DROID Python config; it does not start or replace the model.

| Entry | Task scope | Episodes |
| --- | --- | --- |
| `configs/robolab/droid.py` | All benchmark tasks | 1 per task |
| `configs/robolab/native_batch.py` | All benchmark tasks | 10 per task, 10 native environments |
| `configs/robolab/remote.yaml` | BananaInBowlTask | 1 |
| YAML with `--smoke` | BananaInBowlTask | One native batch per configured simulator worker |
| YAML with `--full` | All benchmark tasks | Python configuration quotas |

This separation matches RoboDojo and RoboTwin, whose Python recipes describe
their complete suites and whose remote YAMLs default to one task/one episode.
Their per-task quotas remain benchmark-specific. RoboLab's one-episode default
comes from `robolab/eval/runner.py` in the reference revision above, not from the
historical ten-environment run. `--full` changes scope, not protocol equivalence.

Copy the YAML to a deployment location and adjust `simulator.config`,
`simulator.robolab_root`, `simulator.python`, `model.host` and `run.output_root`.
Paths resolve relative to that YAML; `~` and environment variables are supported.
Root precedence is YAML, Python environment config, `ROBOLAB_ROOT`, then a
RoboLab checkout beside FluxThemis. An unset referenced variable is an error.
The interpreter defaults to the launcher interpreter; set `simulator.python`
explicitly when the launcher and simulation environments differ. Install the
normal Themis dependencies in that interpreter without changing the model env.

The public template uses `gpu_ids: [0]`, like the other benchmarks. Override it
with `--gpu-ids` for the host allocation. Set the independent graphics ordinal
through `--cfg-options themis.evaluation.environment.render_device_id=...` when
CUDA isolation changes the numbering. Keep host UUIDs, private paths and Kit
thread/resource limits in deployment overrides. Run under the host's assigned
CPU affinity and thread limits; the YAML does not allocate or reserve resources.

Put space-separated `kit_args` in a private Python config that inherits
`configs/robolab/droid.py`, then point the deployment YAML's `simulator.config`
to that file. Do not pass that string through `--cfg-options`: the current
MMEngine `DictAction` removes its spaces, including those inside quotes, so
IsaacLab receives one joined argument instead of separate Kit options.
Check both `config_snapshot.py` and the effective settings after Kit starts.
CPU affinity, individual thread pools and GPU device selection are separate
controls; a configured Kit thread count does not cap every process thread.

`cpu_thread_limit` optionally forwards a positive integer to Isaac Sim's
`SimulationApp.limit_cpu_threads`. This sets its Kit tasking and TBB limits
and its USD/OpenBLAS thread environment together. The default is `None`, which
preserves the installed runtime defaults. A Kit tasking CLI override alone
does not set all of these pools. Keep CPU affinity as a separate deployment
control and verify effective runtime settings and actual thread affinities.

Resolve a prepared deployment without starting workloads:

```bash
python scripts/run_evaluation.py --config /path/to/remote.yaml --dry-run
python scripts/run_evaluation.py --config /path/to/remote.yaml --smoke --dry-run
python scripts/run_evaluation.py --config /path/to/remote.yaml --full --dry-run
```

Each call writes `plan.json`, `launch_config.yaml` and `config_snapshot.py` in
a unique run directory. RoboLab's native files are rooted under that run's
`native/` directory, then separated by worker and native instance. The snapshot
records that exact path. Dry-run performs no dependency, asset, network or GPU
checks and does not create the native output directory.

For an ordered subset, use the Python entry with existing config overrides:

```bash
python scripts/run_evaluation.py --config configs/robolab/droid.py --dry-run \
  --cfg-options \
  'themis.evaluation.environment.task_name=[MustardInRightBinTask,BananaInBowlTask]' \
  themis.evaluation.episodes_per_task=2
```

YAML `simulator.task` accepts one base task name or `all`. It controls native
registration and clears the generic task-ID filter, just as RoboDojo does.
Use Python task lists as above; a list is not a new YAML task syntax. When
passing a task-list override with a YAML deployment, set both `simulator.task`
and `simulator.episodes` to `null`, or their single-task settings take precedence.
Runtime registration remains the authority for resolved IDs and per-task quota
keys. Inspect the frozen config and the `run_start` manifest before interpreting
results.

Actual execution first runs `scripts/preflight/environment.py` with the selected
simulation interpreter, then the distributed launcher's dry-run and layout
check. RoboLab has no RoboDojo layout files. Its preflight instead reads genuine
task subclasses and literal scene paths from `robolab/tasks/benchmark`, checks
the DROID robot, table fixture, default background, object catalog and direct
ASCII USD file references, and records installed package versions. Deleted or
reordered USD list entries are not new asset dependencies. Unknown tasks,
missing/empty required files, unmaterialized Git LFS pointers and missing package
metadata fail before startup.

To run only the dependency/asset check against a generated snapshot, use the
same prepared simulation interpreter and Python path as the deployment:

```bash
PYTHONPATH="$PWD/scripts:$PWD" /path/to/simulator/python \
  -m preflight.environment /path/to/run/config_snapshot.py
```

This check imports no RoboLab/Isaac runtime and opens no USD stage. Remote URLs,
Kit resolver names and binary/transitive USD/MDL dependencies are explicitly
outside its coverage; inspect `unchecked_assets` and validate them during later
simulation acceptance. Installed version metadata does not prove binary/driver
compatibility. Removing `--dry-run` from the launcher starts real execution and
therefore requires a prepared model service and separate resource validation.

`simulator.episode_timeout_s` and `simulator.simulator_restart_limit` forward the
existing worker-supervisor limits. Other benchmarks retain defaults of 3600
seconds and three restarts. RoboLab's YAML uses zero restarts to avoid silently
restarting a native random stream. This wall-clock timeout is separate from the
task's native control-step budget. Explicit batch recovery creates a fresh
report session and retains original result provenance. It resets unfinished
batches; it does not restore live physics or renderer state. Recovered scores
must be identified as recovered runs, with native artifacts independently
audited before publication.

The distributed entry point forwards `--startup-timeout-s` to both the
coordinator and each local worker supervisor. This gives cold Isaac startup the
configured readiness budget instead of an independent 180-second cutoff. It
does not change episode timeouts, restart limits or native task step budgets.

## Explicit batch recovery

RoboLab disables automatic worker restart and does not restore a live physics,
renderer or mid-episode policy state. Explicit recovery keeps acknowledged
complete native batches and resets unfinished batches with their original
canonical seeds. Success and failure results are treated equally.

The high-level `--resume` / YAML `resume.from_dir` entry remains rejected for
RoboLab: native output relocation and asset/model identity checks have not been
integrated there. Use the explicit bundle/config route after stopping the source
run and verifying its frozen configuration, model, assets and native artifacts:

```bash
python - /path/to/stopped-run /path/to/resume_input.json <<'PY'
import sys
from fluxthemis.engines.runners.resume import atomic_json, build_bundle
atomic_json(sys.argv[2], build_bundle(sys.argv[1]))
PY
python scripts/run_evaluation.py --config /path/to/new-deployment.yaml --full \
  --cfg-options themis.evaluation.resume_path=/path/to/resume_input.json
```

Keep task inventory, batch size, quotas, seeds and evaluation settings unchanged;
use a fresh output directory and a ready model service. Bundle validation checks
episode identities and settings, not the contents of remote model/asset files.
Do not bypass deployment identity checks merely because the bundle validates.

Before publishing `run_start`, a reporter with a local event journal saves the
validated bundle as `resume_input.json` beside that journal. Matching existing
content is accepted, conflicting content or a write error aborts startup. This
makes further recovery possible even if the new run stops during result replay.
A reporter without a local journal cannot provide a self-contained local
recovery directory. Keep source journals, excluded partial batches, native
recordings and all recovery inputs; disclose the recovery lineage in results.

## Model service and real execution

Use the prepared model interpreter and the matching FluxVLA recipe. For the
validated pi0.5 recipe, set its three asset paths in the model environment:

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

This command starts one model worker with the selected interpreter. The stats
path is the converted FluxVLA dataset-statistics file prepared for this recipe,
not the original OpenPI `norm_stats.json`. The checkpoint and tokenizer must
match the recipe. Keep its existing loading and mixed-precision settings; the
model recipe does not become part of the simulator config. Other model families
need independently validated recipes.

The example binds the frontend to localhost. For a different simulator host,
bind it to an appropriate reachable interface and set the deployment YAML's
`model.host` accordingly, or use an existing tunnel. The optional Themis model
wrapper requires newer FluxVLA replica/GPU CLI support; check that server
version's `--help` before using the wrapper. The validated command above uses
the existing FluxVLA Python entry directly.

In the prepared simulator environment, from FluxThemis:

```bash
python scripts/run_evaluation.py --config /path/to/deployment.yaml --smoke \
  --result-pointer /path/to/smoke-location.json
```

Resolve that same deployment with `--dry-run` first. The YAML must point to the
prepared simulator Python and assets and use its allocated GPU tokens. Supply
any required graphics ordinal, Kit library/cache paths and resource limits as
deployment overrides, as described above. Separate model/simulator interpreters
must each contain their own dependencies; do not add the entire model
site-packages directory to the simulator's Python path.

The validated FluxVLA broker exits after accepting `run_end`. Start a fresh
service before the next evaluation and check its readiness. To exercise task
reuse and switching, set the private YAML's `task` and `episodes` to `null`,
use one simulator worker and run without `--smoke` or `--full`:

```bash
python scripts/run_evaluation.py --config /path/to/deployment.yaml \
  --result-pointer /path/to/continuous-location.json \
  --cfg-options \
  'themis.evaluation.environment.task_name=[BananaInBowlTask,MustardInRightBinTask,CannedFoodInBinTask]' \
  themis.evaluation.episodes_per_task_overrides.BananaInBowlTask=2
```

With the unchanged `droid.py` default of one episode per task, this plans
Banana/Banana/Mustard/CannedFood in one run. These are development overrides;
the public Python config still selects all tasks, and the public YAML still
defaults to one Banana episode on GPU `[0]`. They are not a full benchmark
protocol or a requirement to use a particular host/GPU.

## Episode noise streams

The DROID client and matching FluxVLA pi0.5 serving recipe enable
`episode_noise_stream_v1`. Each episode keeps its own policy seed. The first
request initializes a Torch generator from the seeded state after preprocessing;
subsequent action chunks advance that generator instead of reusing the first
noise sample. The client carries the returned generator state to the next
request, together with episode identity, request sequence and a runtime format.
This keeps the stream independent of which model worker handles the request.
Replaying the same observation and input state reproduces the same noise and
output state in the validated runtime. Resetting or closing the client clears
the episode state.

Both client and server must enable the mode, and the model must explicitly
support `noise_generator`. Other configurations retain `legacy_episode` by
default. The protocol imposes no fixed noise shape, action width or chunk
length. Its serialized state is runtime-specific: incompatible Torch/CUDA/device
formats are rejected. Cross-version or cross-hardware bitwise reproducibility
is not promised. Transferring a live episode's noise state does not implement
simulation resume or recover renderer/physics state.

This matches OpenPI's mechanism of drawing fresh flow-matching noise at each
replanning request. It keeps independent episode seeds instead of reproducing
OpenPI's historical process-wide stream across episodes. The validated pi0.5
recipe still uses its accepted mixed precision: 775 FP32 and 37 BF16 parameter
tensors with BF16 autocast. This is not an all-FP32 or strictly OpenPI-equivalent
computation path. The converted checkpoint file itself stores 812 FP32 tensors;
the 37 BF16 norm containers belong to the loaded model.

For leaderboard comparisons, disclose task count and episodes per task as well
as SR/Score. The public DROID configuration uses 120 tasks with one episode each;
the official pi0.5 leaderboard uses ten episodes per task. Its reported RoboLab
build is `09545754 (v0.1.0)`, while this integration is validated against
`9cce00ca`. The development model runtime uses Python 3.10, Torch 2.8.0+cu128
and Transformers 5.3.0; simulation uses the separate runtime described above.
Record the actual GPU, worker count, asset/config hashes and inference precision
with each run. The default recipe uses current EGO-L/over-shoulder-left and wrist
RGB images, proprioception, the default instruction, and 15-step joint-position
chunks. Other models can choose their own validated action horizon.

The official model-size value of approximately 3.3B and its OpenPI inference
requirement of more than 8 GB are upstream metadata, not measurements of this
deployment. Model-only memory measurements must be separated from simulator
and rendering allocations; aggregate GPU usage is not a model VRAM requirement.

## Evaluation events and local reports

As in [RoboDojo](robodojo.md) and [RoboCasa](robocasa.md), Themis sends
`run_start`, `episode_start`, `episode_end` and `run_end` events to the FluxVLA
service. The existing reporter mirrors events locally after the server accepts
them. FluxVLA validates session/sequence and episode identity, journals accepted
events and writes summaries at `run_end`. No extra report service is required.
[RoboTwin](robotwin.md) uses similar summary filenames in its evaluation tools;
its separate runner/manager reporting path is not a RoboLab dependency.

In FluxVLA, `configs/pi05/pi05_paligemma_robolab_droid_jointpos_inference.py`
enables `themis.server.evaluation_reporting` with `report_kind='robolab'` and
explicitly selects its existing `eval` dataset section. Report metadata lives in
that section; it does not change the model, transforms or precision. The actual
task manifest and per-task quotas come from Themis's `run_start`, including
subsets and overrides, rather than a second task list in the model config.

`themis.server.evaluation_reporting.result_output_dir` optionally selects the
model-side report root. An absolute path is used directly; a relative path is
resolved against the checkpoint's grandparent directory. Omitting it uses that
grandparent. Each run has a unique directory under
`eval_runs/<checkpoint_stem>/EVAL-robolab-<model_family>-<timestamp>[-<run_name>]`.
Set paths in the model service's deployment config or `--cfg-options`, not in
the environment recipe. Disabling `evaluation_reporting.enabled` is suitable
for inference-only diagnostics, not a Themis evaluation expecting report ACKs.

| Owner | Artifacts |
| --- | --- |
| RoboLab on the simulator host | Native HDF5 under the Themis run's `native/worker/instance` directories |
| Themis | Run plan, configuration snapshot, local event mirror and evaluation outputs |
| FluxVLA | Accepted `events.jsonl`, `rank_progress/rank0.json`, `robolab/task<index>_results.json`, `summary.json/csv/txt`, `episodes.csv`, `task_success_rates.csv`, `task_status/` and `failed_tasks.txt` |

Per-task JSON preserves the runtime task ID, description and metadata, plus
each episode's identity, prediction metadata and full environment `info`.
`native_output_dir`, `environment_instance` and `instance_episode` identify the
native evidence; `simulator_seed` remains distinct from scheduling seeds. These
are simulator-host paths, not proof that a file exists or is readable from the
model host. Reports set `native_artifacts_verified=false`; an independent
artifact audit must check the native files against accepted events. There are
no additional global `gpuN_task...` mirrors.

RoboLab reporting uses the native success boolean and final recorded subtask
score. It preserves RoboLab's own metrics alongside the common event/report
structure used by other benchmarks.

After native freeze and HDF5 export, the simulation worker reads
`data/<native_demo_key>/subtask/score[-1]` and calls the pinned native
`robolab.core.metrics.compute_metrics` functions once. Score is stored in
`[0, 1]` and displayed on a 0-100 scale. It is the final score, not the maximum
observed score; failed episodes can have partial credit. No metrics computation
is added to the model process, and HDF5/Isaac dependencies remain optional
outside simulation.

Each episode records final score, the eight native trajectory metrics, metric
status, control timestep, simulated duration, wall-clock duration, metrics
computation time and its HDF5 path. The original `duration_s` includes terminal
metrics computation; `metrics_compute_s` exposes that cost separately. Prediction
metadata records the sampling mode and input/output generator-state hashes.
An initial-observation hash is also retained, since identical physical resets
do not guarantee pixel-identical RTX images.

| Metric | Aggregation |
| --- | --- |
| SR | Native successes / native episodes; 95% Clopper-Pearson interval |
| Score /100 | Episode mean and task macro mean; 95% two-stage task/episode bootstrap interval for the task macro mean |
| EE Speed | Mean of episode `ee_speed_mean`, converted from m/s to cm/s; population standard deviation across episodes (`ddof=0`) |
| EE SPARC | Mean native `ee_sparc` across valid episodes; values nearer zero indicate smoother movement |

All valid success and failure trajectories contribute. The report also includes
`joint_sparc_mean`, `ee_isj`, `joint_isj`, `ee_path_length`, `joint_rmse_mean` and
`ee_speed_max`, with a separate valid-episode count for each metric. EE Speed
uses simulation time and excludes inference waiting; it is not evaluation
wall-clock throughput. Interpret speed and smoothness together with SR/Score,
since stalled trajectories can be slow or smooth.

The statistical protocol follows the online
[RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html),
which differs from some dashboard defaults in the pinned RoboLab checkout.
Bootstrap uses 10,000 samples, a dedicated NumPy PCG64 generator with seed 0,
and 2.5/97.5 percentiles. Official bootstrap sample-count/seed details have not
been verified; the report discloses these choices and the NumPy version rather
than claiming identical interval endpoints. Statistical RNG does not consume
the policy or simulation streams. One episode per task cannot estimate
within-task seed variance, even when a bootstrap interval is reported.

Missing or corrupt native recordings raise a technical error with the native
result and HDF5 path. A mathematically undefined metric instead becomes JSON
`null` with a reason such as `native_undefined` or `too_few_samples`. Values are
not replaced with zero. `metrics_report_complete` tracks recording coverage
separately from native-result completeness, and valid counts disclose undefined
values. The model host records artifact paths; independent HDF5/event auditing
is still required.

| Field | Meaning |
| --- | --- |
| `planned_episodes` | Quota in this run's selected task manifest |
| `completed_episodes` | Accepted `episode_end` events |
| `native_episodes` | Accepted episodes with a frozen, complete native result |
| `unscored_episodes` | Accepted episodes without a native terminal result |
| `missing_episodes` | Planned episodes without an accepted end event |
| `success_rate_pct` | Native successes / native episodes, multiplied by 100; null/blank when there are none |
| `plan_complete` / `results_complete` | All planned ends accepted / all planned native results present |
| `report_complete` | Run status is `finished` and native results cover every planned episode |
| `metrics_report_complete` | Complete native results and final score/metric records for every planned episode; valid counts separately describe undefined numerical values |

For example, 1/1 successes on one task and 1/3 on another give 2/4 = 50% overall,
not an unweighted average of task rates. A wrapper `max_steps` cutoff before
native freeze is unscored, not an ordinary native failure. A success claim
without native evidence, conflicting success flags or conflicting episode
identity is rejected. Partial reports retain the observed rate and its native
denominator alongside unscored/missing counts; that rate is not a full result.
`duration_s` in task/overall statistics sums recorded episode wall times; the
top-level duration is the run wall time, not simulated control time.

The run's `finished`, `failed` or `interrupted` status and error are preserved.
An inconsistent `finished` label cannot make missing quota complete. Task status
`COMPLETE` means native results cover the quota, even if every episode failed;
its record is `status|successes|native_episodes|planned_episodes`.
`failed_tasks.txt` contains task indexes with missing/unscored results, not a
list of policy failures. `full_suite=null` means full benchmark coverage is
unverified: completing a selected plan does not prove a full benchmark run.

This integration writes local reports and explicitly skips Feishu.
If the process is killed before `run_end`, accepted
events may exist without a final summary. Use the explicit complete-batch
recovery route above; automatic worker restart and live-state restoration remain
unsupported. Synthetic-event CPU tests cover reporting only, not transport,
native artifact export or model-driven evaluation.

## Lifecycle and seeds

A worker starts one app. The first episode creates a task environment, following
episodes of that task reuse it, and switching tasks closes the old environment
before creating the next. Each normal episode starts with:

```text
reset_eval_state() -> reset() -> reset() -> use the second observation
```

Run the simulator in a dedicated worker process. Isaac Sim's default fast
shutdown can terminate Python during the final application `close()`. Save
episode results first and verify worker completion externally; a zero exit code
alone is insufficient. The existing Themis worker protocol already distinguishes
an expected shutdown from an exit before the worker receives its stop message.

The native recorder is then assigned a separate `run_N.hdf5`, following
`robolab/eval/episode.py`; the scalar path uses demo zero and the batch path uses
one demo per lane. `N` counts runs within the current native instance.
A unique worker directory and separate instance
directories prevent overwrites when a task is revisited or an episode identity
is repeated. `native_output_dir`, `environment_instance` and `instance_episode`
in reset/step info link these artifacts to the scheduled episode. Native HDF5
export remains RoboLab's responsibility. Video uses Themis's ordinary `render()`
path (first mapped camera); native multi-camera video export is not implemented.

The seed convention follows the distinction already used by
[LIBERO](libero.md):

| Field | Meaning in this adapter |
| --- | --- |
| `simulator_seed` | Applied once per native environment creation; default 0 |
| `environment_seed` | Themis scheduling identity, recorded without reseeding |
| `environment_case_index` | Task-local episode identity, not a layout index |
| `policy_seed` | Managed by the policy protocol, not the environment wrapper |

Normal resets do not pass a seed. The adapter does not call `seed()` per episode,
map case indices to nonexistent layouts, or rewrite the scheduling seed in
results. Both seed values are explicitly recorded. This matches the official
runner's creation/reuse/reset pattern for a single native environment, but does
not establish numerical equivalence to the historical ten-environment protocol.
Returning to a task, restarting a worker, retries and dynamic worker assignment
can restart native random streams. Complete-batch recovery has been audited,
but it does not promise the same trajectories as an uninterrupted run. Validate
each deployment's device assignment and resource limits separately.

## Observations and actions

| Output | Official source | Contract |
| --- | --- | --- |
| `over_shoulder_left_camera` | `image_obs.over_shoulder_left_camera[0]` | HWC RGB uint8 |
| `wrist_cam` | `image_obs.wrist_cam[0]` | HWC RGB uint8 |
| `states` | `arm_joint_pos[0]` then `gripper_pos[0]` | 8 values; retain precision |
| `task_description` | Official default instruction selection | Original text |

Arrays are copied to CPU NumPy memory. The gripper observation has already been
scaled by RoboLab and is not scaled again. Missing images, incorrect dimensions
and nonfinite states raise errors. In the pinned runtime, both the observation
and native action manager use `panda_joint1` through `panda_joint7` in that order,
followed by `finger_joint`. No image resizing, padding,
tokenization or normalization occurs here.

Each `step()` receives one finite numeric `[8]` action row: seven absolute joint
targets and a numeric gripper command. It becomes native FP32 `[1,8]` on the
environment device. There is no denormalization, state addition, clipping or
gripper inversion. RoboLab closes its gripper only when the numeric value is
strictly greater than 0.5. Boolean actions are rejected because RoboLab's boolean
branch has different semantics. The shared episode runner handles action chunks;
`execute_horizon=None` leaves the chunk length to the policy response.

The shared serving/client/execution boundary accepts integer and floating-point
actions and returns finite FP32 values. Boolean, string, complex and object
actions are rejected before conversion, including mixed boolean/numeric lists.
Values that overflow FP32 are rejected as well. This check applies to every
policy using these boundaries; it does not change model-internal masks, tokens,
denormalization, action dimensions or gripper thresholds. The RoboLab adapter
continues to enforce its own eight-value action row.

A completed or truncated episode discards the remainder of its action chunk.
The next episode resets the policy context and request sequence. A ZMQ timeout
replaces the request socket without replaying the prediction; a late reply cannot
be consumed as the next request. A worker exit with code zero is only an orderly
stop after the coordinator's shutdown message. Unexpected exits consume the
configured restart budget; the remote development template keeps that budget at
zero. A CPU test of a permitted restart checks job identity, not equivalence of
native simulator random state after recovery.

Transport/model failures abort evaluation instead of becoming ordinary native
task failures. When the report service remains available, the failed run records
missing episodes. A fatal broker timeout can prevent a final report from being
written: retain the partial event journal and process logs, and treat the run as
incomplete. Do not infer success from the existence of an output directory.

## End conditions and limits

An early raw termination can be an initialization artifact that RoboLab resets
internally. The adapter does not add resets or count this signal as a completed
episode. It waits for `all_terminated` and reads `get_env_results()` for the
native aggregate success, including tasks with multiple success conditions.

Unfinished transitions have `success=None`. A frozen native success produces
`terminated=True`; a frozen native failure produces `truncated=True`, reflecting
RoboLab's success-versus-timeout result convention. Raw term/trunc flags, native
episode length, wrapper step count and the aggregate result remain in `info`.
The native reward is retained; final Score is read from the completed native
recording as described above.

The adapter applies the official outer-loop budget `env.max_episode_length`.
If an early internal reset clears the native counter, the wrapper still uses
the original outer-loop budget, matching the reference episode loop. It does
not add replacement steps. If that budget expires without native freeze, the
error records wrapper steps, native episode length and raw term/trunc flags;
the episode is not converted to an ordinary benchmark failure. CPU tests cover
internal resets on the first and second steps, including later success and
exhaustion without a native result.
Current benchmark tasks span 300 to 4,500 control steps at 15 Hz. The config's
outer Themis limit covers that maximum. A frozen environment without a complete
native result, or exhaustion of the native outer-loop budget without a result,
raises an error instead of silently recording an ordinary failure. Review that
diagnostic in the later evaluation/report integration.

## CPU checks and simulation evidence

```bash
python -m pytest test/test_environments/test_robolab.py -q
python -m pytest test/test_robolab_launch.py test/test_robolab_preflight.py -q
python -m pytest test/test_model_clients.py test/test_transports.py test/test_engines/ -q
```

CPU tests cover reuse/switching, clear/double reset order without reseeding,
separate seed metadata, observation ownership/precision, absolute action
forwarding, artifact/final-result handling, resource cleanup, config parsing and
imports with unavailable optional dependencies. Run the existing Themis tests
as regression coverage too.

The transport checks use local ZMQ sockets. Worker/coordinator tests cover stale
identities, restart budgets, shutdown acknowledgements and report acknowledgement
before counting a result. These portable tests need no simulator or checkpoint.
Machine-specific cross-repository process fixtures and their synthetic task
quotas stay outside the repositories; they are development evidence, not a
replacement for the official Python task selection or deployment preflight.

On the FluxVLA side, run
`test/test_runners/test_robolab_evaluation_report_writer.py` together with the
existing report/serving tests. These use synthetic events and an empty temporary
checkpoint to check report construction without loading weights. On a CPU-only
development installation, use the repository's existing `--cpu-model-tests`
import isolation with GPUs hidden; it does not validate production model
imports. No simulator package, real checkpoint or external report endpoint is
needed by these tests.

The development simulation check used one native environment, creation seed 0,
the default instruction and realtime renderer. It replayed saved FP32 action
rows through the unmodified official `create_env` / `run_episode` functions
twice, then through this adapter once. No model service was used. Each sequence
reused the app and ran these episodes consecutively:

| Episode | Official run 1 | Official run 2 | Wrapped run |
| --- | --- | --- | --- |
| BananaInBowlTask | Success, 168 steps | Success, 168 steps | Success, 168 steps |
| BananaInBowlTask, reused env | Tape exhausted at 180 | Tape exhausted at 180 | Tape exhausted at 180 |
| MustardInRightBinTask | Success, 128 steps | Success, 128 steps | Success, 128 steps |
| CannedFoodInBinTask | Native timeout, 900 steps | Native timeout, 900 steps | Native timeout, 900 steps |

Across these 12 replay attempts and 4,128 control steps, the recorded reset
states, per-step physical states, executed actions and native results matched
exactly between runs. Each sequence created and closed three native instances,
seeded only at creation, and cleared evaluation state before each double reset.
The nine completed episodes' HDF5 action rows, sample counts and success flags
matched the native returns. The three exhausted tapes left the native result
unset and exported no completed demo; exhaustion is a limitation of the saved
input sequence, not a benchmark failure.

All 1,380 wrapped observations (four resets plus 1,376 steps) matched their
same-call native camera and state values. Reward, termination flags, effective
gripper commands and result metadata were also checked. Independent rendered
images were not pixel-identical even between the two native repeats; this is
not a claim of image determinism. No early initialization artifact occurred in
these simulations, so that branch retains CPU-only coverage. The app processes
exited and released GPU memory; the small sequence does not establish long-term
leak freedom or multi-worker equivalence.

This check validates the environment boundary for three tasks and saved actions.
It does not measure policy success rate, validate the full launcher/reporting
path, or establish equivalence to the historical ten-environment evaluation.

## Full-suite development check

The 2026-11-05 check evaluated all 120 benchmark tasks once, using the production
continuous-noise protocol after its CUDA qualification. Native results, both
event journals, all 120 HDF5 recordings, the sampling-state chains and all eight
trajectory metrics were audited. All 87 failures reached their native task
limits; there were no missing, duplicate or unscored episodes. This is a
development check, not a 1,200-episode leaderboard submission.

| Run | Episodes | Successes | SR (%) | Score /100 | EE Speed (cm/s) | EE SPARC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous fixed-noise check | 120 | 16 | 13.33 | 27.49 | 3.35 ± 1.55 | −7.95 |
| Continuous-noise check | 120 | 33 | 27.50 | 41.76 | 5.45 ± 1.54 | −7.13 |
| Official pi0.5 reference | 1200 | 336 | 28.00 | 43.40 | 5.35 ± 1.59 | −8.34 |

The new SR 95% interval is [19.75, 36.40]%; the Score bootstrap interval is
[34.24, 49.50]. Equal task quotas make the episode and task macro Score means
identical here. The official values come from the linked leaderboard and use
a different task quota and RoboLab build, as described above.

With the same task/seed mapping, 24 tasks changed from failure to success and
seven changed from success to failure; nine of the original 16 successes were
retained. The total control-step count fell from 155,119 to 141,750 (8.62%),
and the time from starting the 120-task stage through audit fell from 7,427 to
6,972 seconds (6.12%). Mean inference time remained about 144 ms. Failed
episodes still accounted for 90.69% of control steps, so the success-rate gain
did not halve evaluation time.

Score and success remain separate native measurements. For example,
MustardAboveRaisinTask completed successfully in 202 steps with final Score 0,
whereas ReorientWhiteMugsTask timed out at 900 steps with final Score 1. These
values matched the native recordings. The report does not infer either field
from the other; task-specific differences between scoring and success logic
were not investigated in this check.

The deployment used seven NVIDIA RTX PRO 5000 72GB Blackwell GPUs, one simulator
worker per GPU and one model worker sharing GPU 0. The maximum observed
container memory was 142.49 GiB; maximum per-GPU total memory was 28.63 GiB,
including simulation/rendering, so it is not a model-only VRAM measurement.
Registered processes exited and the dedicated cache was cleared after archive
hash verification. This does not establish minimum hardware requirements,
1,200-episode stability, cross-GPU action equivalence or simulation resume.

The inference configuration consumes deployment-provided converted checkpoint,
statistics and tokenizer files. This change does not publish checkpoint
artifacts. Generated `episodes.csv`, per-task JSON and summary files retain the
per-episode evidence; preserve them with each run's native recordings and
frozen configuration. Source changes were uncommitted during the development
run and were frozen by content hashes, alongside FluxVLA base `6ef7040` and
FluxThemis base `3d36581`. Documentation was updated after the run.

## Fault-recovered 120 x 10 evaluation

The 2026-11-06 evaluation completed all 1,200 episodes across 120 tasks. Six
simulation workers each advanced ten native environments; six model workers
served individual observations through the shared request queue.

| Run | Episodes | Successes | SR (%) | Score /100 | EE Speed (cm/s) | EE SPARC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FluxVLA + Themis, fault-recovered | 1200 | 317 | 26.42 | 40.56 | 5.43 ± 1.54 | −7.44 |
| Official pi0.5 reference | 1200 | 336 | 28.00 | 43.40 | 5.35 ± 1.59 | −8.34 |

The SR 95% Clopper-Pearson interval is [23.94, 29.01]%; the Score bootstrap
interval is [34.77, 46.45]. SR is 1.58 percentage points below the official
reference. Inclusion of its point estimate in an interval does not establish
statistical equivalence. Version, hardware and precision differences remain
as described above.

This result combines 60 complete task batches from the first run (600 episodes,
129 successes) with the final retry's remaining 60 batches (600 episodes,
188 successes). Inherited results were unchanged apart from new report identity,
counts and provenance fields. Earlier interrupted groups with 1, 7 and 5 completed
episodes were retained as evidence and excluded as whole groups; their original
seeds were used for the replacement groups. Normal task failures were not
selectively rerun. This is a fault-recovered evaluation, not an uninterrupted run.

Final auditing verified 1,200 unique episode/task-index/native-file-demo
identities, 120 HDF5 files, 2,402 matching events on both reporting sides,
97,559 model requests and sampling-state chains, and all native metrics.
Independent archive readback checked all 435 closed output files and every
native HDF5; no hash mismatch remained. Processes exited and dedicated caches
were cleaned after verification. The final retry had no numerical warnings;
earlier nonfinite simulation states remain an unresolved robustness limitation.

The deployment used RTX PRO 5000 72GB GPUs. Observed peaks in the final retry
were 34.26 GiB per GPU and 153.98 GiB container memory; these include simulation
and rendering and are not minimum model requirements. Its new 600 episodes,
audit and cleanup took 1 h 6 min 42 s, not the elapsed time for all 1,200 episodes.
The evaluated sources were frozen by content hashes at FluxVLA base `6ef7040`
and FluxThemis base `3d36581`. The later public device-map/cache helpers and
automatic journal-local recovery-input persistence have CPU validation only;
the measured run used explicit deployment scripts for those responsibilities.

## Real-service validation

A CUDA qualification for `episode_noise_stream_v1` used fixed archived
observations for BananaInBowlTask (seed 94), MustardAboveRaisinTask (57) and
ReorientWhiteMugsTask (80). Each episode advanced for 12 chunks and every
request was replayed, giving 72 actual ZMQ requests across interleaved clients.
The production implementation exactly matched the diagnostic continuous-noise
implementation's noise and actions; replayed output states also matched. A
separate generator check reproduced all 116 archived noise blocks for these
three diagnostic episodes. Two CPU model workers verified state transfer
independent of worker order. This qualifies sampling and transport in the
frozen runtime; it does not establish task success rates or cross-GPU numerical
equivalence.

The earlier real-service check completed one Banana smoke episode and four
continuous episodes (Banana twice, MustardInRightBin and CannedFoodInBin) with
native HDF5, both journals and report files cross-checked. It exercised the
formal YAML launcher, same-task reuse and task switching, but did not cover
native failures or a full task suite. The later suite evaluations cover those
broader execution paths.

The early check passed space-separated Kit arguments through a CLI parser that
removed their spaces. Its intended Kit thread limit was therefore not verified;
a later startup-only check confirmed correct forwarding from an inherited
Python config. That check did not rerun episodes or change native task limits.
Keep Kit arguments in the Python deployment config, inspect effective runtime
settings, and verify CPU affinity separately. A configured Kit worker count
does not bound all OS threads or establish absence of transient allocations.
