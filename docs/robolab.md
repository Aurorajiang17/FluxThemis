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

The adapter has CPU contract tests and single-worker, fixed-action validation
against the official RoboLab episode runner. The checked scope includes real
application startup, same-application task switching, joint ordering, observation
mapping and native results. The remote YAML/configuration path and static
deployment preflight have CPU coverage. RoboLab event validation and local report
summaries also have CPU coverage with synthetic events. Real model/worker
integration, continuous model-driven episodes and the historical evaluation
protocol remain separate validation work. The commands below prepare/check a
deployment; they do not establish a validated full-evaluation protocol.

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
the separately configured `render_device_id`. Multi-GPU graphics assignment
requires separate deployment validation; start with one worker.

## Remote configuration and preflight

The split deployment follows [RoboDojo](robodojo.md): start FluxVLA separately,
then use Themis to connect to its reachable ZMQ endpoint. The YAML references the
model-independent DROID Python config; it does not start or replace the model.

| Entry | Task scope | Episodes |
| --- | --- | --- |
| `configs/robolab/droid.py` | All benchmark tasks | 1 per task |
| `configs/robolab/remote.yaml` | BananaInBowlTask | 1 |
| YAML with `--smoke` | BananaInBowlTask | One per configured simulator worker |
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
Use Python task lists as above; a list is not a new YAML task syntax. Runtime
registration remains the authority for the resolved environment IDs.

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
task's native control-step budget. RoboLab resume is explicitly rejected until
random-stream and output relocation behavior is validated.

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
| FluxVLA | Accepted `events.jsonl`, `rank_progress/rank0.json`, `robolab/task<index>_results.json`, `summary.json/csv/txt`, `task_success_rates.csv`, `task_status/` and `failed_tasks.txt` |

Per-task JSON preserves the runtime task ID, description and metadata, plus
each episode's identity, prediction metadata and full environment `info`.
`native_output_dir`, `environment_instance` and `instance_episode` identify the
native evidence; `simulator_seed` remains distinct from scheduling seeds. These
are simulator-host paths, not proof that a file exists or is readable from the
model host. Reports set `native_artifacts_verified=false`; later real execution
must check the native files against accepted events. There are no additional
global `gpuN_task...` mirrors.

RoboLab reporting uses the native success boolean, following the reference
`robolab/eval/summarize.py` and `robolab/core/logging/results.py`. It does not
substitute RoboDojo score/categories or RoboTwin Easy/Hard groupings.

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

This integration does not calculate the optional native subtask score,
trajectory metrics or confidence intervals. It records local reports only and
explicitly skips Feishu. If the process is killed before `run_end`, accepted
events may exist without a final summary; automatic recovery/resume is not
implemented. Synthetic-event CPU tests cover reporting only, not transport,
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

The native recorder is then assigned a separate `run_N.hdf5` with native env
index zero, following `robolab/eval/episode.py`. `N` counts episodes within the
current native instance. A unique worker directory and separate instance
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
can restart random streams; production scheduling and resume equivalence require
separate validation. Use one worker and consecutive task episodes initially.

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

## End conditions and limits

An early raw termination can be an initialization artifact that RoboLab resets
internally. The adapter does not add resets or count this signal as a completed
episode. It waits for `all_terminated` and reads `get_env_results()` for the
native aggregate success, including tasks with multiple success conditions.

Unfinished transitions have `success=None`. A frozen native success produces
`terminated=True`; a frozen native failure produces `truncated=True`, reflecting
RoboLab's success-versus-timeout result convention. Raw term/trunc flags, native
episode length, wrapper step count and the aggregate result remain in `info`.
The native reward is retained; an unverified progress score is not invented.

The adapter applies the official outer-loop budget `env.max_episode_length`.
Current benchmark tasks span 300 to 4,500 control steps at 15 Hz. The config's
outer Themis limit covers that maximum. A frozen environment without a complete
native result, or exhaustion of the native outer-loop budget without a result,
raises an error instead of silently recording an ordinary failure. Review that
diagnostic in the later evaluation/report integration.

## CPU checks and simulation evidence

```bash
python -m pytest test/test_environments/test_robolab.py -q
python -m pytest test/test_robolab_launch.py test/test_robolab_preflight.py -q
```

CPU tests cover reuse/switching, clear/double reset order without reseeding,
separate seed metadata, observation ownership/precision, absolute action
forwarding, artifact/final-result handling, resource cleanup, config parsing and
imports with unavailable optional dependencies. Run the existing Themis tests
as regression coverage too.

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
