"""Bounded CPU checks for supervisor shutdown and restart semantics."""

from threading import Event
from unittest.mock import Mock

import pytest

from fluxthemis.engines.runners import worker


class _Process:

    def __init__(self, kwargs, outcome):
        self.kwargs = kwargs
        self.exitcode, self.stopped = outcome

    def start(self):
        self.kwargs['ready_event'].set()
        if self.stopped:
            self.kwargs['stop_received'].set()

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


@pytest.mark.parametrize('outcomes', [
    [(0, True)],
    [(0, False), (0, True)],
    [(1, True), (0, True)],
    [(-14, False), (0, False), (0, True)],
])
def test_supervisor_requires_shutdown_ack_and_fresh_generation_flags(
        monkeypatch, outcomes):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(worker.time, 'sleep', lambda _: None)
    processes = []

    def process(**values):
        child = _Process(values['kwargs'], outcomes[len(processes)])
        processes.append(child)
        return child

    context = Mock(Event=Event, Process=process)
    monkeypatch.setattr(worker.multiprocessing, 'get_context',
                        lambda _: context)
    supervisor = worker.SimulatorWorkerSupervisor(
        'config.py',
        'tcp://127.0.0.1:15558',
        'worker-0',
        'deployment',
        '-1',
        episode_timeout_s=1)
    supervisor.run()

    assert len(processes) == len(outcomes)
    assert [p.kwargs['generation']
            for p in processes] == list(range(len(outcomes)))
    assert len({id(p.kwargs['stop_received'])
                for p in processes}) == len(outcomes)
    assert len({id(p.kwargs['ready_event'])
                for p in processes}) == len(outcomes)


def test_terminate_escalates_only_when_process_stays_alive():
    process = Mock()
    process.is_alive.side_effect = [True, True]
    worker._terminate_process(process)
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.join.call_args_list == [((), {'timeout': 10.0}), ((), {})]


@pytest.mark.parametrize('field,value', [
    ('deployment_id', 'wrong'),
    ('worker_id', 'wrong'),
    ('generation', 0),
])
def test_worker_rejects_commands_for_other_identity(field, value):
    child = worker._SimulatorWorker(
        config_path='config.py',
        rollout_endpoint='tcp://127.0.0.1:15558',
        worker_id='worker-0',
        deployment_id='deployment',
        generation=1,
        device='-1',
        episode_timeout_s=1,
        config_overrides={},
        ready_event=Event(),
        stop_received=Event())
    header = dict(child._message_identity)
    header[field] = value
    with pytest.raises(RuntimeError, match='mismatch'):
        child._validate_header(header)
