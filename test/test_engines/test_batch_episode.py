from collections import defaultdict

import numpy as np
import pytest

from fluxthemis.engines.runners.batch_episode import execute_episode_batch
from fluxthemis.engines.runners.episode import build_episode_jobs
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
