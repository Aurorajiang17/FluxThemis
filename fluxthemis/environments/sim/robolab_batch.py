"""RoboLab native-batch lifecycle, sharing the scalar adapter's runtime."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from fluxthemis.structures import StepResult, TaskSpec

from .robolab_metrics import collect_episode_metrics


def reset_batch(adapter, task, indices, seeds, cases):
    from .robolab import _nonnegative_integer

    adapter._require_setup()
    if not isinstance(task, TaskSpec):
        raise TypeError('task must be a TaskSpec')
    if task.task_id not in adapter._task_ids:
        raise KeyError(f'Unknown RoboLab task id: {task.task_id}')
    if any(
            len(values) != adapter.batch_size
            for values in (indices, seeds, cases)):
        raise ValueError('Reset metadata must match native batch_size')
    indices = [_nonnegative_integer(i, 'episode_index') for i in indices]
    if len(set(indices)) != len(indices):
        raise ValueError('Batch episode indices must be unique')
    seeds = [
        None if s is None else _nonnegative_integer(s, 'environment_seed')
        for s in seeds
    ]
    cases = [
        None if c is None else _nonnegative_integer(c, 'case_index')
        for c in cases
    ]
    adapter._episode_active = False
    adapter._last_observation = None
    try:
        if task.task_id != adapter._active_task_id:
            adapter._activate_task(task.task_id)
        env = adapter._env
        with adapter._runtime.torch.inference_mode():
            env.reset_eval_state()
            env.reset()
            raw, _ = env.reset()
        recorder = env.recorder_manager
        if recorder is not None and hasattr(recorder, 'set_hdf5_file'):
            recorder.set_hdf5_file(f'run_{adapter._instance_episode}.hdf5')
            for lane in range(adapter.batch_size):
                recorder.set_episode_index(lane, env_ids=[lane])
        adapter._steps = 0
        adapter._batch_active = set(range(adapter.batch_size))
        adapter._batch_info = []
        results = []
        for lane in range(adapter.batch_size):
            obs = adapter._observation(raw, lane)
            info = dict(
                benchmark='robolab',
                task_id=task.task_id,
                episode_index=indices[lane],
                environment_seed=seeds[lane],
                environment_case_index=cases[lane],
                simulator_seed=adapter.simulator_seed,
                environment_instance=adapter._instance,
                instance_episode=adapter._instance_episode,
                native_batch_size=adapter.batch_size,
                native_batch_lane=lane,
                native_hdf5_path=str(adapter._native_dir /
                                     f'run_{adapter._instance_episode}.hdf5'),
                native_demo_key=f'demo_{lane}',
                initial_observation_sha256={
                    key: hashlib.sha256(value.tobytes()).hexdigest()
                    for key, value in obs.items()
                    if isinstance(value, np.ndarray)
                },
                native_output_dir=str(adapter._native_dir),
                native_max_episode_steps=adapter._max_steps)
            adapter._batch_info.append(info)
            results.append((obs, dict(info)))
        adapter._instance_episode += 1
        adapter._episode_active = True
        return results
    except Exception:
        adapter.close()
        raise


def step_batch(adapter, actions):
    from .robolab import _numpy

    adapter._require_setup()
    if not adapter._episode_active:
        raise RuntimeError('Call reset_batch before stepping a native batch')
    active = adapter._batch_active
    if not isinstance(actions, Mapping) or set(actions) != active:
        raise ValueError('Actions must match the active native lanes')
    env = adapter._env
    if set(env.active_env_ids) != active:
        raise RuntimeError('Native active lanes changed unexpectedly')
    values = np.zeros((adapter.batch_size, 8), dtype=np.float32)
    for lane, action in actions.items():
        value = _numpy(action)
        if value.shape != (8, ):
            raise ValueError('DROID action must have shape [8] per lane')
        if (value.dtype.kind not in 'fiu' or isinstance(action, (list, tuple))
                and any(isinstance(v, (bool, np.bool_)) for v in action)):
            raise TypeError('DROID actions must be real numbers')
        with np.errstate(over='ignore'):
            values[lane] = value
    if not np.isfinite(values).all():
        raise ValueError('DROID actions must be finite FP32 values')
    native = adapter._runtime.torch.as_tensor(
        values, dtype=adapter._runtime.torch.float32, device=env.device)
    adapter._episode_active = False
    with adapter._runtime.torch.inference_mode():
        raw, reward, term, trunc, _ = env.step(native)
    adapter._steps += 1
    native_results = env.get_env_results()
    if (len(native_results) != adapter.batch_size
            or [r.get('env_id')
                for r in native_results] != list(range(adapter.batch_size))):
        raise RuntimeError('RoboLab returned invalid batch results')
    still_active = set(env.active_env_ids)
    if not still_active <= active or bool(
            env.all_terminated) != (not still_active):
        raise RuntimeError('RoboLab native frozen state is inconsistent')
    reward, term, trunc, native_steps = (
        _numpy(v).reshape(-1)
        for v in (reward, term, trunc, env.episode_length_buf))
    if any(v.size != adapter.batch_size
           for v in (reward, term, trunc, native_steps)):
        raise RuntimeError('RoboLab step arrays must match batch_size')
    transitions = {}
    for lane in sorted(active):
        frozen = lane not in still_active
        result = dict(native_results[lane])
        success = result.get('success')
        if frozen:
            if (not isinstance(success, (bool, np.bool_))
                    or not isinstance(result.get('step'), Integral)):
                raise RuntimeError('Frozen RoboLab lane has no final result')
            success = bool(success)
        elif success is not None or result.get('step') is not None:
            raise RuntimeError('Unfrozen RoboLab lane has a final result')
        elif adapter._steps >= adapter._max_steps:
            raise RuntimeError(
                f'RoboLab reached the native outer-loop limit without a '
                f'final result: lane={lane}, steps={adapter._steps}, '
                f'native_episode_step={native_steps[lane]}, '
                f'raw_terminated={term[lane]}, raw_truncated={trunc[lane]}')
        if not np.isfinite(reward[lane]):
            raise ValueError('RoboLab reward is not finite')
        info = dict(adapter._batch_info[lane])
        info.update(
            wrapper_steps=adapter._steps,
            native_episode_step=int(native_steps[lane]),
            raw_terminated=bool(term[lane]),
            raw_truncated=bool(trunc[lane]),
            native_frozen=frozen,
            native_result=result)
        score = None
        if frozen:
            score, metrics = collect_episode_metrics(
                info['native_hdf5_path'],
                adapter._control_dt_s,
                success,
                int(result['step']),
                demo_key=info['native_demo_key'])
            info.update(metrics)
        transitions[lane] = StepResult(
            observation=adapter._observation(raw, lane),
            reward=float(reward[lane]),
            terminated=frozen and success is True,
            truncated=frozen and success is False,
            success=success,
            score=score,
            info=info)
    adapter._batch_active = still_active
    adapter._episode_active = bool(still_active)
    return transitions
