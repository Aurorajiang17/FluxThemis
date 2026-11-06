"""Native batch execution with separate policy state and chunks per lane."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping

import numpy as np

from fluxthemis.structures import EpisodeResult, StepResult

from .episode import build_context, validate_action_chunk


def execute_episode_batch(environment,
                          policy_clients,
                          jobs,
                          max_episode_steps,
                          execute_horizon,
                          stop_on_success,
                          on_episode_started=None,
                          on_episode_finished=None):
    """Run a full same-task group, reporting each lane as soon as it ends.

    Calls remain synchronous and single-observation; simulation advances all
    active lanes once per step. Wall-clock durations include other lanes'
    inference and terminal reporting, since these share the simulation clock.
    """
    size = environment.batch_size
    if not jobs or len(jobs) != size or len(policy_clients) != size:
        raise ValueError('jobs and policy_clients must match batch_size')
    if len({id(client) for client in policy_clients}) != size:
        raise ValueError('Each batch lane requires an independent client')
    if any(job.task != jobs[0].task for job in jobs):
        raise ValueError('A native batch must contain the same task')
    if len({job.episode_id for job in jobs}) != size:
        raise ValueError('Batch episode IDs must be unique')
    reset = environment.reset_batch(
        jobs[0].task, [job.episode_index for job in jobs],
        environment_seeds=[job.environment_seed for job in jobs],
        environment_case_indices=[job.environment_case_index for job in jobs])
    if len(reset) != size:
        raise RuntimeError('reset_batch returned an incorrect lane count')
    observations, infos = [], []
    for lane, (observation, info) in enumerate(reset):
        if not isinstance(info, Mapping):
            raise TypeError('Environment reset info must be a mapping')
        observations.append(observation)
        infos.append(dict(info))
        client, job = policy_clients[lane], jobs[lane]
        client.reset(job.task, job.episode_index, job.policy_seed)
        client.set_episode_id(job.episode_id)
        if on_episode_started is not None:
            on_episode_started(job)

    started = time.perf_counter()
    queues = [deque() for _ in jobs]
    metadata = [[] for _ in jobs]
    returns = [0.0] * size
    steps, calls = [0] * size, [0] * size
    successes, scores = [False] * size, [None] * size
    results, active = [None] * size, set(range(size))
    while active:
        actions = {}
        for lane in sorted(active):
            job = jobs[lane]
            if not queues[lane]:
                context = build_context(job.task, job.episode_index,
                                        job.policy_seed, infos[lane],
                                        job.episode_id, observations[lane])
                chunk = policy_clients[lane].predict_action(
                    observations[lane], context)
                values = validate_action_chunk(chunk)
                if not isinstance(chunk.metadata, Mapping):
                    raise TypeError('ActionChunk.metadata must be a mapping')
                horizon = (
                    len(values) if execute_horizon is None else min(
                        len(values), execute_horizon))
                if horizon < 1:
                    raise ValueError('execute_horizon must be positive')
                queues[lane].extend(values[:horizon])
                calls[lane] += 1
                metadata[lane].append(dict(chunk.metadata))
            actions[lane] = queues[lane].popleft().copy()
        transitions = environment.step_batch(actions)
        if not isinstance(transitions, Mapping) or set(transitions) != active:
            raise RuntimeError('step_batch must return exactly active lanes')
        for lane in sorted(active):
            transition = transitions[lane]
            if not isinstance(transition, StepResult):
                raise TypeError('step_batch values must be StepResult')
            if not isinstance(transition.info, Mapping):
                raise TypeError('StepResult.info must be a mapping')
            observations[lane], infos[lane] = (transition.observation,
                                               dict(transition.info))
            returns[lane] += float(transition.reward)
            steps[lane] += 1
            if transition.success is not None:
                successes[lane] |= bool(transition.success)
            if transition.score is not None:
                score = float(transition.score)
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Environment score must be finite [0,1]')
                scores[lane] = score
            reason = None
            if successes[lane] and stop_on_success:
                reason = 'success'
            elif transition.terminated:
                reason = 'terminated'
            elif transition.truncated:
                reason = 'truncated'
            elif steps[lane] >= max_episode_steps:
                reason = 'max_steps'
            if reason is None:
                continue
            job = jobs[lane]
            infos[lane].update(native_batch_size=size, native_batch_lane=lane)
            result = EpisodeResult(
                task_id=job.task.task_id,
                episode_index=job.episode_index,
                policy_seed=job.policy_seed,
                environment_seed=job.environment_seed,
                environment_case_index=job.environment_case_index,
                success=successes[lane],
                episode_return=returns[lane],
                steps=steps[lane],
                duration_s=time.perf_counter() - started,
                termination_reason=reason,
                model_calls=calls[lane],
                score=scores[lane],
                info=infos[lane],
                prediction_metadata=metadata[lane])
            job.validate_result(result)
            results[lane] = result
            active.remove(lane)
            queues[lane].clear()
            if on_episode_finished is not None:
                on_episode_finished(job, result)
    return tuple(results)
