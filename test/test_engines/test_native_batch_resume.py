import copy
import json
from collections import deque
from types import SimpleNamespace

import pytest

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
from fluxthemis.structures import TaskSpec


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
    timestamp = '2026-11-06T00:00:00+00:00'
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
                recorded_at='2026-11-06T00:00:00+00:00'))
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
