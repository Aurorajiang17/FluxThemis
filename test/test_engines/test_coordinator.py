import pytest

from fluxthemis.engines.runners.coordinator import (
    DistributedEvaluationCoordinator, _SchedulingState, _WorkerSlot)
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_episode)
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings)
from fluxthemis.engines.utils import to_jsonable
from fluxthemis.environments.mock import MockEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import TaskSpec
from fluxthemis.transports.zmq_protocol import (empty_payload, encode_frame,
                                                encode_header)


class _RouterSocket:

    def __init__(self, incoming):
        self.incoming = incoming
        self.sent = []

    def recv_multipart(self):
        return self.incoming

    def send_multipart(self, frames):
        self.sent.append(frames)


def _coordinator_with_active_job(job):
    evaluation_run = EvaluationRun(
        EvaluationSettings(run_name='test-run'),
        SuccessRateEvaluator(),
        MockEvaluationReporter(),
    )
    coordinator = DistributedEvaluationCoordinator(
        rollout_bind='tcp://127.0.0.1:15558',
        expected_workers=1,
        deployment_id='deployment',
    )
    routing_id = b'worker-0'
    slot = _WorkerSlot(
        worker_id='worker-0',
        routing_id=routing_id,
        generation=0,
        device='0',
        task_manifest=(job.task, ),
        current_job=job,
        simulator_ready=True,
    )
    coordinator._worker_slots[slot.worker_id] = slot
    coordinator._worker_router = _RouterSocket([
        routing_id,
        encode_header(
            'environment_case_rejected',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=slot.generation,
            episode_id=job.episode_id,
        ),
        empty_payload(),
    ])
    return coordinator, slot, evaluation_run


def test_rejected_job_requeues_same_slot_with_next_environment_case():
    jobs = build_episode_jobs(
        [TaskSpec('task')],
        episodes_per_task=2,
        base_seed=7,
        run_session_id='run',
    )
    job = jobs[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    scheduling_state = _SchedulingState.from_jobs(jobs)
    scheduling_state.pending_jobs.clear()
    scheduling_state.started_episode_ids.add(job.episode_id)
    coordinator._scheduling_state = scheduling_state

    coordinator._receive_worker_message(evaluation_run)

    retry = scheduling_state.pending_jobs.popleft()
    assert retry.episode_id == job.episode_id
    assert retry.episode_index == job.episode_index
    assert retry.policy_seed == job.policy_seed
    assert retry.environment_seed == job.environment_seed
    assert retry.environment_case_index == 2
    assert scheduling_state.jobs_by_episode_id[job.episode_id] == retry
    assert scheduling_state.results_by_episode_id == {}
    assert scheduling_state.started_episode_ids == set()
    assert scheduling_state.next_environment_case_index_by_task == {'task': 3}
    assert slot.is_idle
    assert coordinator._worker_router.sent == []


def test_reconnecting_worker_requeues_active_episode():
    jobs = build_episode_jobs(
        [TaskSpec('task')],
        episodes_per_task=1,
        base_seed=7,
        run_session_id='run',
    )
    job = jobs[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    scheduling_state = _SchedulingState.from_jobs(jobs)
    scheduling_state.pending_jobs.clear()
    scheduling_state.started_episode_ids.add(job.episode_id)
    coordinator._scheduling_state = scheduling_state
    replacement_routing_id = b'worker-0:1'
    coordinator._worker_router.incoming = [
        replacement_routing_id,
        encode_header(
            'register_worker',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=1,
            device='0',
        ),
        empty_payload(),
    ]

    coordinator._receive_worker_message(evaluation_run)

    assert scheduling_state.pending_jobs.popleft() == job
    assert scheduling_state.restart_counts_by_episode_id == {job.episode_id: 1}
    assert not slot.is_ready
    assert slot.current_job is None
    assert slot.routing_id == replacement_routing_id
    assert slot.generation == 1
    assert slot.recovery_deadline is not None
    assert coordinator._worker_router.sent == []


@pytest.mark.parametrize('field,value', [
    ('deployment_id', 'other'),
    ('episode_id', 'other'),
    ('generation', 1),
    ('worker_id', 'unknown'),
])
def test_coordinator_rejects_wrong_active_identity(field, value):
    job = build_episode_jobs([TaskSpec('task')], 1, 7, 'run')[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    state = _SchedulingState.from_jobs([job])
    coordinator._scheduling_state = state
    header = dict(
        deployment_id='deployment',
        worker_id=slot.worker_id,
        generation=0,
        episode_id=job.episode_id)
    header[field] = value
    coordinator._worker_router.incoming = [
        slot.routing_id,
        encode_header('episode_start', **header),
        empty_payload()
    ]
    with pytest.raises(RuntimeError):
        coordinator._receive_worker_message(evaluation_run)
    assert state.started_episode_ids == set()
    assert state.results_by_episode_id == {}


def test_old_generation_cannot_complete_current_job():
    job = build_episode_jobs([TaskSpec('task')], 1, 7, 'run')[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    slot.generation = 1
    state = _SchedulingState.from_jobs([job])
    coordinator._scheduling_state = state
    coordinator._worker_router.incoming = [
        b'old-worker-route',
        encode_header(
            'episode_end',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=0,
            episode_id=job.episode_id),
        empty_payload()
    ]
    coordinator._receive_worker_message(evaluation_run)
    assert slot.current_job == job
    assert state.results_by_episode_id == {}


@pytest.mark.parametrize('startup', [False, True])
def test_restart_limit_zero_rejects_first_restart(startup):
    job = build_episode_jobs([TaskSpec('task')], 1, 7, 'run')[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    coordinator.simulator_restart_limit = 0
    coordinator._scheduling_state = _SchedulingState.from_jobs([job])
    if startup:
        slot.current_job = None
        slot.simulator_ready = False
    coordinator._worker_router.incoming = [
        b'worker-0:1',
        encode_header(
            'register_worker',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=1,
            device='0'),
        empty_payload()
    ]
    with pytest.raises(RuntimeError, match='restart limit 0'):
        coordinator._receive_worker_message(evaluation_run)


def test_report_failure_does_not_complete_or_release_job(monkeypatch):
    job = build_episode_jobs([TaskSpec('task')], 1, 7, 'run')[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    state = _SchedulingState.from_jobs([job])
    coordinator._scheduling_state = state
    environment = MockEnvironment(tasks=['task'], success_after=1)
    environment.setup()
    result = execute_episode(
        environment=environment,
        policy_client=MockPolicyClient(action_dim=8),
        job=job,
        max_episode_steps=2,
        execute_horizon=None,
        stop_on_success=True)
    environment.close()
    coordinator._worker_router.incoming = [
        slot.routing_id,
        encode_header(
            'episode_end',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=0,
            episode_id=job.episode_id),
        encode_frame(to_jsonable(result))
    ]

    def reject_report(*args):
        raise RuntimeError('report not acknowledged')

    monkeypatch.setattr(evaluation_run, 'on_episode_end', reject_report)
    with pytest.raises(RuntimeError, match='report not acknowledged'):
        coordinator._receive_worker_message(evaluation_run)
    assert slot.current_job == job
    assert state.results_by_episode_id == {}
