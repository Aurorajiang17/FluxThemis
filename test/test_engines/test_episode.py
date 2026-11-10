import numpy as np
import pytest

from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_episode,
                                                validate_action_chunk)
from fluxthemis.environments.mock import MockEnvironment
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import ActionChunk, StepResult, TaskSpec


def test_episode_jobs_and_execution_contract():
    jobs = build_episode_jobs(
        [TaskSpec('a'), TaskSpec('b')],
        episodes_per_task=2,
        base_seed=7,
        run_session_id='run',
    )

    assert [job.policy_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_case_index for job in jobs] == [0, 1, 0, 1]

    environment = MockEnvironment(tasks=['a'], success_after=2)
    policy_client = MockPolicyClient(action_dim=7, chunk_size=1)
    environment.setup()
    result = execute_episode(
        environment=environment,
        policy_client=policy_client,
        job=jobs[0],
        max_episode_steps=3,
        execute_horizon=1,
        stop_on_success=True,
    )

    assert result.success
    assert result.steps == 2
    assert result.policy_seed == 7
    assert result.environment_seed == 7
    assert result.environment_case_index == 0


@pytest.mark.parametrize('value', [
    np.ones((2, 8), dtype=bool),
    np.full((2, 8), '0.25'),
    np.full((2, 8), 0.25 + 1j),
    np.zeros((2, 8), dtype=object),
    [[True] + [0.25] * 7] * 2,
    [np.zeros(8), np.ones(8, dtype=bool)],
])
def test_episode_rejects_non_real_action_chunks(value):
    with pytest.raises(TypeError, match='real numbers'):
        validate_action_chunk(ActionChunk(actions=value))


@pytest.mark.parametrize('value', [
    np.empty((0, 8)),
    np.empty((2, 0)),
    np.zeros(8),
    np.full((2, 8), np.nan),
    np.full((2, 8), np.inf),
    np.full((2, 8), 1e100)
])
def test_episode_rejects_invalid_chunks(value):
    with pytest.raises(ValueError):
        validate_action_chunk(ActionChunk(actions=value))


class _CountingEnvironment:

    def __init__(self, terminal_step=4, terminal='success'):
        self.terminal_step = terminal_step
        self.terminal = terminal
        self.actions = []
        self.resets = []

    def reset(self, task, episode_index, **kwargs):
        self.resets.append((task.task_id, episode_index, kwargs))
        self.actions.append([])
        return {'step': 0}, {}

    def step(self, action):
        self.actions[-1].append(action.copy())
        step = len(self.actions[-1])
        finished = step == self.terminal_step
        return StepResult(
            observation={'step': step},
            reward=0.0,
            terminated=finished and self.terminal == 'terminated',
            truncated=finished and self.terminal == 'truncated',
            success=finished and self.terminal == 'success',
            info={})


class _NumberedPolicy(MockPolicyClient):

    def __init__(self, lengths=(7, 2, 5)):
        super().__init__(action_dim=8)
        self.lengths = lengths
        self.observed_steps = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.next_number = 0
        self.observed_steps.append([])

    def predict_action(self, observation, context):
        self.observed_steps[-1].append(observation['step'])
        length = self.lengths[self.call_count % len(self.lengths)]
        numbers = np.arange(self.next_number, self.next_number + length)
        self.next_number += length
        self.call_count += 1
        return ActionChunk(np.repeat(numbers[:, None], 8, axis=1))


@pytest.mark.parametrize('terminal', ['success', 'terminated', 'truncated'])
def test_episode_discards_remaining_actions_on_every_terminal(terminal):
    jobs = build_episode_jobs([TaskSpec('a'), TaskSpec('b')], 2, 7, 'run')
    environment = _CountingEnvironment(terminal=terminal)
    policy = _NumberedPolicy()
    for job in jobs:
        result = execute_episode(environment, policy, job, 20, None, True)
        assert result.steps == 4
        assert result.model_calls == 1
    assert [x[0] for x in environment.resets] == ['a', 'a', 'b', 'b']
    for actions in environment.actions:
        assert [a[0] for a in actions] == [0, 1, 2, 3]
    assert policy.observed_steps == [[0]] * 4


@pytest.mark.parametrize('horizon,expected,observed', [
    (None, list(range(12)), [0, 7, 9]),
    (3, [0, 1, 2, 7, 8, 9, 10, 11, 14, 15, 16, 21], [0, 3, 5, 8, 11]),
])
def test_episode_replans_from_latest_observation_without_padding(
        horizon, expected, observed):
    job = build_episode_jobs([TaskSpec('a')], 1, 0, 'run')[0]
    environment = _CountingEnvironment(terminal_step=100)
    policy = _NumberedPolicy()
    result = execute_episode(environment, policy, job, 12, horizon, True)
    assert result.steps == 12
    assert [a[0] for a in environment.actions[0]] == expected
    assert policy.observed_steps[0] == observed
