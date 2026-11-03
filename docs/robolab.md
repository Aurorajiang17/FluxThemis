# RoboLab DROID environment adapter

`RoboLabEnvironment` connects the official RoboLab DROID joint-position
environment to the existing Themis `BaseEnvironment` contract. The standalone
configuration is [`configs/robolab/droid.py`](../configs/robolab/droid.py).
It selects one task and two development episodes. It does not define model
weights, tokenizers, normalization, inference precision or a fixed action chunk
length. Future policies can use the same environment-unit interface after their
own input/output recipes have been validated.

## Scope and validation status

The initial implementation has CPU contract tests using simulator doubles.
Real Isaac application startup, same-application task switching, joint ordering,
rendering and fixed-action trajectory comparisons require a separate simulation
validation. The full remote launcher, preflight checks, RoboLab report summary
and historical evaluation protocol are separate integration work. This document
does not provide a validated full-evaluation launch command.

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

## Lifecycle and seeds

A worker starts one app. The first episode creates a task environment, following
episodes of that task reuse it, and switching tasks closes the old environment
before creating the next. Each normal episode starts with:

```text
reset_eval_state() -> reset() -> reset() -> use the second observation
```

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
and nonfinite states raise errors. Real joint ordering still needs simulation
verification against the native action manager. No image resizing, padding,
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

## CPU checks and remaining simulation checks

```bash
python -m pytest test/test_environments/test_robolab.py -q
```

CPU tests cover reuse/switching, clear/double reset order without reseeding,
separate seed metadata, observation ownership/precision, absolute action
forwarding, artifact/final-result handling, resource cleanup, config parsing and
imports with unavailable optional dependencies. Run the existing Themis tests
as regression coverage too.

The next simulation check should compare official and wrapped execution of
task A twice followed by task B, with the same seed, renderer, environment count
and fixed action sequence. Check real reset states, camera layout, joint order,
termination results and resource release. CPU doubles cannot prove these facts.
