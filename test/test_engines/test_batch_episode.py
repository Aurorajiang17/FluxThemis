"""Native batch execution, bounded retries and complete-batch recovery."""

import copy
import json
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import pytest

from fluxthemis.engines.runners.batch_episode import execute_episode_batch
from fluxthemis.engines.runners.coordinator import (
    DistributedEvaluationCoordinator, _SchedulingState)
from fluxthemis.engines.runners.episode import build_episode_jobs
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings,
                                                   _validate_resume_settings)
from fluxthemis.engines.runners.resume import (atomic_json, build_bundle,
                                               validate_bundle)
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.reporters import fluxvla_reporter as reporter_module
from fluxthemis.reporters.base import BaseEvaluationReporter
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


@pytest.mark.parametrize('fault',
                         ['nonfinite', 'shape', 'normal_failure', 'exhausted'])
def test_bounded_native_retry_preserves_model_and_stages_results(
        tmp_path, monkeypatch, fault):
    import json

    from fluxthemis.engines.runners import worker as module
    from fluxthemis.environments.base import NonFiniteObservationError
    jobs = build_episode_jobs([TaskSpec('task')], 3, 7, 'run')
    env = BatchEnvironment()
    env.output_dir = tmp_path
    original_step = env.step_batch
    failed = []

    def step(actions):
        if fault in ('nonfinite', 'exhausted') and len(
                env.received) == 2 and not failed:
            failed.append(True)
            raise NonFiniteObservationError('arm_joint_pos', 1, (3, 7), [[0]])
        if fault == 'shape':
            raise ValueError('shape differs')
        return original_step(actions)

    resets = []

    def recover():
        resets.append(True)
        BatchEnvironment.__init__(env)

    env.step_batch = step
    env.recover_after_observation_fault = recover
    clients = [Policy(i) for i in range(3)]
    for c in clients:
        c.sampling_mode = 'process_noise_stream_v1'
    obj = module._SimulatorWorker.__new__(module._SimulatorWorker)
    obj.worker_id = 'worker'
    obj.deployment_id, obj.generation = 'deployment', 0
    obj.episode_timeout_s = 30
    events = []
    monkeypatch.setattr(module, '_send_episode_event',
                        lambda *a, **kw: events.append((a[2], a[3])))
    config = dict(
        nonfinite_batch_retry_limit=2,
        nonfinite_batch_retry_budget=0 if fault == 'exhausted' else 1,
        max_episode_steps=20)
    if fault in ('shape', 'exhausted'):
        with pytest.raises(ValueError):
            obj._execute_retryable_batch(None, env, clients, config, jobs)
        assert not [e for e in events if e[0] == 'episode_end']
        assert not resets
    else:
        obj._execute_retryable_batch(None, env, clients, config, jobs)
        assert len([e for e in events if e[0] == 'episode_end']) == 3
        # Each attempt refreshes the deadline; reports deduplicate.
        expected_starts = 6 if fault == 'nonfinite' else 3
        starts = [e for e in events if e[0] == 'episode_start']
        assert len(starts) == expected_starts
        assert len(resets) == (1 if fault == 'nonfinite' else 0)
    records = [json.loads(p.read_text()) for p in tmp_path.rglob('*.json')]
    assert len(records) == (1 if fault in ('nonfinite', 'exhausted') else 0)
    if records:
        assert len(records[0]['prediction_metadata']) == 3
        assert records[0][
            'partial_results']  # completed lane from interrupted attempt


def test_failed_attempt_requests_and_resume_offset_are_audited(monkeypatch):
    import copy
    from pathlib import Path
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / 'scripts'))
    from evaluation.progress import validate_process_chain

    def metadata(sequence):
        return dict(
            sampling_mode='process_noise_stream_v1',
            sequence=1,
            episode_id='e',
            policy_seed=0,
            process_instance_id='p',
            process_sequence=sequence)

    record = dict(
        task_id='task',
        episode_id='e',
        policy_seed=0,
        model_calls=1,
        prediction_metadata=[metadata(12)])
    retry = dict(
        task_id='task',
        episode_ids=['e'],
        attempt=1,
        retry=True,
        confirmed_nonfinite=dict(lane=0, invalid_indices=[[0]]),
        prediction_metadata=[[metadata(11)]])
    assert validate_process_chain([record], [retry], 10) == 'p'
    for change in ('gap', 'duplicate', 'instance', 'fake_fault'):
        bad = copy.deepcopy(retry)
        if change == 'gap':
            bad['prediction_metadata'][0][0]['process_sequence'] = 9
        if change == 'duplicate':
            bad['prediction_metadata'][0][0]['process_sequence'] = 12
        if change == 'instance':
            bad['prediction_metadata'][0][0]['process_instance_id'] = 'other'
        if change == 'fake_fault':
            bad['confirmed_nonfinite']['invalid_indices'] = []
        with pytest.raises(ValueError):
            validate_process_chain([record], [bad], 10)


def _journal(directory):
    start = dict(
        base_seed=7,
        episodes_per_task=3,
        episode_batch_size=3,
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


def test_recovery_keeps_whole_batches_and_original_seeds(tmp_path):
    _journal(tmp_path)
    bundle = build_bundle(tmp_path)
    assert len(bundle['completed']) == 3
    assert [p['end']['success']
            for p in bundle['completed']] == [False, True, False]
    excluded = bundle['native_batch_recovery'][
        'excluded_partial_group_results']
    assert len(excluded) == 1 and excluded[0]['end']['success'] is True
    assert [j['policy_seed'] for j in bundle['jobs']] == list(range(7, 13))
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 10
    validate_bundle(bundle)
    partial = copy.deepcopy(bundle)
    partial['completed'].extend(excluded)
    with pytest.raises(ValueError, match='whole completed groups'):
        validate_bundle(partial)


def test_recovery_cannot_change_batch_size_or_layout(tmp_path):
    _journal(tmp_path)
    bundle = build_bundle(tmp_path)
    settings = EvaluationSettings(
        base_seed=7,
        episodes_per_task=3,
        max_episode_steps=100,
        episode_batch_size=1)
    with pytest.raises(ValueError, match='settings differ'):
        _validate_resume_settings(settings, bundle['start'])
    bundle['jobs'][3]['environment_case_index'] = 3
    bundle['next_environment_case_index_by_task']['b'] = 4
    with pytest.raises(ValueError, match='layout'):
        validate_bundle(bundle)


def test_restore_preserves_priority_and_canonical_seed_mapping():
    jobs = build_episode_jobs([TaskSpec('a'), TaskSpec('b')], 3, 7, 'new')
    state = _SchedulingState.from_jobs(jobs)
    state.pending_jobs = deque(jobs[3:] + jobs[:3])
    coordinator = DistributedEvaluationCoordinator('tcp://127.0.0.1:15558', 1)
    coordinator._scheduling_state = state
    data = dict(
        jobs=[
            dict(
                global_episode_number=j.global_episode_number,
                environment_case_index=j.environment_case_index) for j in jobs
        ],
        next_environment_case_index_by_task={
            'a': 3,
            'b': 3
        })
    coordinator._restore_jobs(SimpleNamespace(resume_data=data))
    assert [j.policy_seed for j in state.pending_jobs] == [10, 11, 12, 7, 8, 9]


def _resumed_run(tmp_path, monkeypatch, bundle=None):
    tmp_path.mkdir(exist_ok=True)
    if bundle is None:
        source = tmp_path / 'source'
        source.mkdir()
        _journal(source)
        bundle = build_bundle(source)
    input_path = tmp_path / 'external-input.json'
    atomic_json(input_path, bundle)
    monkeypatch.setattr(reporter_module, 'FluxVLAZMQTransport',
                        lambda config: SimpleNamespace())
    reporter = reporter_module.FluxVLAEvaluationReporter(
        {}, event_log=str(tmp_path / 'resumed' / 'events.jsonl'))
    reporter._run_session_id = 'new'
    rows = []

    def report(kind, payload):
        # The journal must be self-contained before a run_start is sent.
        assert json.loads((reporter._event_log.parent /
                           'resume_input.json').read_text()) == bundle
        rows.append(
            dict(
                event_type=kind,
                payload=payload,
                sequence=len(rows) + 1,
                run_session_id='new',
                recorded_at='2000-01-01T00:00:00+00:00'))
        reporter._event_log.write_text(''.join(
            json.dumps(row) + '\n' for row in rows))

    monkeypatch.setattr(reporter, 'report_evaluation', report)
    run = EvaluationRun(
        EvaluationSettings(
            base_seed=7,
            episodes_per_task=3,
            episode_batch_size=3,
            max_episode_steps=100,
            task_ids=['a', 'b'],
            resume_path=str(input_path)), SuccessRateEvaluator(), reporter)
    return run, reporter, bundle, rows


def test_config_resume_persists_input_for_chained_recovery(
        tmp_path, monkeypatch):
    run, reporter, bundle, rows = _resumed_run(tmp_path, monkeypatch)
    run.start([TaskSpec('a'), TaskSpec('b')], 6)
    # Interrupted immediately after run_start, before inherited results replay.
    first = build_bundle(reporter._event_log.parent)
    assert first['completed'] == bundle['completed']
    assert [j['policy_seed'] for j in first['jobs']] == list(range(7, 13))
    # Stopping midway through replay also preserves the full batch.
    pair = copy.deepcopy(bundle['completed'][0])
    for kind, payload in [('episode_start', pair['start']),
                          ('episode_end', pair['end'])]:
        payload['episode_id'] = 'new:1'
        payload['resumed_from'] = bundle['source']
        if kind == 'episode_end':
            payload['completed_episodes'] = 1
        reporter.report_evaluation(kind, payload)
    second = build_bundle(reporter._event_log.parent)
    assert len(second['completed']) == 3
    assert [p['end']['success']
            for p in second['completed']] == [False, True, False]
    assert len(rows) == 3
    next_run, next_reporter, _, _ = _resumed_run(tmp_path / 'next',
                                                 monkeypatch, second)
    next_run.start([TaskSpec('a'), TaskSpec('b')], 6)
    third = build_bundle(next_reporter._event_log.parent)
    assert third['completed'] == second['completed']
    assert third['jobs'] == second['jobs']


def test_resume_input_reuse_conflict_and_failure(tmp_path, monkeypatch):
    run, reporter, bundle, rows = _resumed_run(tmp_path, monkeypatch)
    path = reporter._event_log.parent / 'resume_input.json'
    reporter.persist_resume_input(bundle)
    original = path.read_bytes()
    reporter.persist_resume_input(bundle)
    assert path.read_bytes() == original
    other = copy.deepcopy(bundle)
    other['source'] += '-different'
    atomic_json(path, other)
    with pytest.raises(ValueError, match='[Cc]onflict'):
        run.start([TaskSpec('a'), TaskSpec('b')], 6)
    assert not rows
    path.unlink()

    def fail_write(*args):
        raise PermissionError('test write denied')

    monkeypatch.setattr(reporter_module, 'atomic_json', fail_write)
    with pytest.raises(PermissionError, match='test write denied'):
        run.start([TaskSpec('a'), TaskSpec('b')], 6)
    assert not rows


def test_reporter_without_journal_keeps_resume_compatibility(
        tmp_path, monkeypatch):
    run, reporter, bundle, _ = _resumed_run(tmp_path, monkeypatch)
    reporter._event_log = None
    reporter.persist_resume_input(bundle)
    assert not (tmp_path / 'resumed').exists()
    # Existing reporters need not implement a new abstract method.
    BaseEvaluationReporter.persist_resume_input(SimpleNamespace(), bundle)
