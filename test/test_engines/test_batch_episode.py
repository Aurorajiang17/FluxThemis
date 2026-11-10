"""Native batch execution, failure propagation and unsupported resume."""

import json
from collections import defaultdict

import numpy as np
import pytest

from fluxthemis.engines.runners.batch_episode import execute_episode_batch
from fluxthemis.engines.runners.episode import build_episode_jobs
from fluxthemis.engines.runners.evaluation import EvaluationSettings
from fluxthemis.engines.runners.resume import build_bundle
from fluxthemis.structures import ActionChunk, StepResult, TaskSpec


class BatchEnvironment:
    batch_size = 3

    def __init__(self):
        self.steps = defaultdict(int)
        self.active = {0, 1, 2}
        self.received = []

    def reset_batch(self, task, indices, **kwargs):
        assert task.task_id == 'task'
        assert tuple(indices) == (0, 1, 2)
        assert tuple(kwargs['environment_seeds']) == (7, 8, 9)
        return [(np.array([i, 0]), {}) for i in indices]

    def step_batch(self, actions):
        assert set(actions) == self.active
        self.received.append(set(actions))
        result = {}
        for lane, action in actions.items():
            self.steps[lane] += 1
            np.testing.assert_array_equal(action, [lane, self.steps[lane]])
            done = self.steps[lane] == (2, 5, 3)[lane]
            if done:
                self.active.remove(lane)
            result[lane] = StepResult(
                observation=np.array([lane, self.steps[lane]]),
                reward=lane + 1,
                terminated=done and lane != 1,
                truncated=done and lane == 1,
                success=(lane != 1) if done else None,
                score=0.5 if done else None,
                info={'lane': lane})
        return result


class Policy:

    def __init__(self, lane):
        self.lane = lane
        self.calls = []

    def reset(self, task, index, seed):
        assert index == self.lane and seed == 7 + self.lane

    def set_episode_id(self, episode_id):
        assert episode_id == f'run:{self.lane + 1}'

    def predict_action(self, obs, context):
        assert obs[0] == self.lane
        assert context['policy_seed'] == 7 + self.lane
        self.calls.append(int(obs[1]))
        return ActionChunk(
            actions=np.array([[self.lane, obs[1] + j + 1]
                              for j in range(self.lane + 1)]),
            metadata={'lane': self.lane})


def test_independent_chunks_results_and_immediate_completion():
    env = BatchEnvironment()
    clients = [Policy(i) for i in range(3)]
    jobs = build_episode_jobs([TaskSpec('task')], 3, 7, 'run')
    starts, finishes = [], []
    results = execute_episode_batch(
        env,
        clients,
        jobs,
        20,
        None,
        True,
        on_episode_started=lambda job: starts.append(job.episode_id),
        on_episode_finished=lambda job, result: finishes.append(
            (job.episode_id, dict(env.steps), result.success)))
    assert starts == ['run:1', 'run:2', 'run:3']
    assert [f[0] for f in finishes] == ['run:1', 'run:3', 'run:2']
    assert finishes[0][1] == {0: 2, 1: 2, 2: 2}
    assert [r.steps for r in results] == [2, 5, 3]
    assert [r.episode_return for r in results] == [2, 10, 9]
    assert [r.termination_reason
            for r in results] == ['success', 'truncated', 'success']
    assert [c.calls for c in clients] == [[0, 1], [0, 2, 4], [0]]
    assert env.received == [{0, 1, 2}, {0, 1, 2}, {1, 2}, {1}, {1}]


def test_batch_requires_independent_clients_and_full_same_task_group():
    env = BatchEnvironment()
    jobs = build_episode_jobs([TaskSpec('task')], 3, 7, 'run')
    client = Policy(0)
    with pytest.raises(ValueError, match='independent'):
        execute_episode_batch(env, [client] * 3, jobs, 20, None, True)
    with pytest.raises(ValueError, match='batch_size'):
        execute_episode_batch(env, [Policy(0)], jobs[:1], 20, None, True)
    mixed = build_episode_jobs(
        [TaskSpec('a'), TaskSpec('b'),
         TaskSpec('c')], 1, 7, 'run')
    with pytest.raises(ValueError, match='same task'):
        execute_episode_batch(env, [Policy(i) for i in range(3)], mixed, 20,
                              None, True)


def test_batch_rejects_missing_transition():
    env = BatchEnvironment()
    env.step_batch = lambda actions: {}
    jobs = build_episode_jobs([TaskSpec('task')], 3, 7, 'run')
    with pytest.raises(RuntimeError, match='lane'):
        execute_episode_batch(env, [Policy(i) for i in range(3)], jobs, 20,
                              None, True)


def _journal(directory, batch_size=3):
    start = dict(
        base_seed=7,
        episodes_per_task=3,
        episode_batch_size=batch_size,
        max_episode_steps=100,
        execute_horizon=None,
        stop_on_success=True,
        strict_worker_assignment=False,
        task_ids=['a', 'b'],
        tasks=[
            dict(
                task_id=t,
                task_index=i,
                description=TaskSpec(t).description,
                metadata={}) for i, t in enumerate(['a', 'b'])
        ],
        total_episodes=6,
        episodes_per_task_by_task={
            'a': 3,
            'b': 3
        })
    timestamp = '2000-01-01T00:00:00+00:00'
    events = []

    def event(kind, body):
        events.append(
            dict(
                event_type=kind,
                payload=body,
                sequence=len(events) + 1,
                run_session_id='old',
                recorded_at=timestamp))

    event('run_start', start)
    for i in range(4):
        body = dict(
            task_id='a' if i < 3 else 'b',
            episode_index=i % 3,
            policy_seed=7 + i,
            environment_seed=7 + i,
            environment_case_index=i % 3,
            episode_id=f'old:{i + 1}',
            started_at=timestamp)
        event('episode_start', body)
        event(
            'episode_end',
            dict(
                body,
                completed_episodes=i + 1,
                total_episodes=6,
                success=i % 2 == 1,
                duration_s=1.,
                episode_return=0.,
                steps=1,
                model_calls=1,
                finished_at=timestamp,
                termination_reason='truncated'))
    event(
        'run_end',
        dict(
            completed_episodes=4,
            total_episodes=6,
            finished_at=timestamp,
            status='interrupted'))
    (directory / 'events.jsonl').write_text(''.join(
        json.dumps(e) + '\n' for e in events))


@pytest.mark.parametrize('nonfinite', [False, True])
def test_technical_fault_stops_without_retry(monkeypatch, nonfinite):
    from fluxthemis.engines.runners import worker as module
    from fluxthemis.environments.base import NonFiniteObservationError
    jobs = build_episode_jobs([TaskSpec('task')], 3, 7, 'run')
    env = BatchEnvironment()
    attempts = []

    def fail(actions):
        attempts.append(actions)
        if nonfinite:
            raise NonFiniteObservationError('arm_joint_pos', 1, (3, 7), [[0]])
        raise ValueError('invalid observation shape')

    env.step_batch = fail
    obj = module._SimulatorWorker.__new__(module._SimulatorWorker)
    obj.worker_id = 'worker'
    obj.deployment_id, obj.generation = 'deployment', 0
    obj.episode_timeout_s = 30
    events = []
    monkeypatch.setattr(module, '_send_episode_event',
                        lambda *a, **kw: events.append((a[2], a[3])))
    with pytest.raises(ValueError):
        obj._execute_batch(None, env, [Policy(i) for i in range(3)],
                           dict(max_episode_steps=20), jobs)
    assert len(attempts) == 1
    assert not [e for e in events if e[0] == 'episode_end']


def test_native_resume_is_rejected_before_reading_input(tmp_path):
    with pytest.raises(ValueError, match='resume is unsupported'):
        EvaluationSettings(
            episode_batch_size=3, resume_path=str(tmp_path / 'absent.json'))
    _journal(tmp_path)
    with pytest.raises(ValueError, match='resume is unsupported'):
        build_bundle(tmp_path)


def test_shared_scalar_resume_keeps_completed_results(tmp_path):
    _journal(tmp_path, batch_size=1)
    bundle = build_bundle(tmp_path)
    assert len(bundle['completed']) == 4
    assert [p['end']['success']
            for p in bundle['completed']] == [False, True, False, True]
