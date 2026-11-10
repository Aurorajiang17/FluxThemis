import numpy as np
import pytest

from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient
from fluxthemis.reporters.fluxvla_reporter import FluxVLAEvaluationReporter
from fluxthemis.structures import TaskSpec
from fluxthemis.transports.zmq_protocol import (decode_frame, decode_header,
                                                encode_frame, encode_header)


class _FakeTransport:

    def __init__(self, response_type):
        self.response_type = response_type
        self.server_deployment_id = 'server-deployment'
        self.request_timeout_s = 5.0
        self.setup_requirements = []
        self.requests = []
        self.closed = False

    def setup(self, *, require_evaluation_reporting=False):
        self.setup_requirements.append(require_evaluation_reporting)

    def request(self, header, payload):
        request_header = decode_header(header)
        request_payload = decode_frame(payload)
        self.requests.append((request_header, request_payload))
        request_id = request_header['request_id']
        if self.response_type == 'predict_response':
            return (
                decode_header(
                    encode_header(
                        'predict_response',
                        request_id=request_id,
                        ok=True,
                        inference_time_s=0.25,
                    )),
                encode_frame({'actions': np.ones((2, 3), dtype=np.float32)}),
            )
        return (
            decode_header(
                encode_header(
                    'report_response', request_id=request_id, ok=True)),
            encode_frame({
                'accepted': True,
                'run_dir': '/tmp/evaluation',
            }),
        )

    @staticmethod
    def validate_response(header, request_id):
        assert header['request_id'] == request_id
        assert header['ok'] is True

    def close(self):
        self.closed = True


def test_policy_client_only_handles_episode_inference():
    client = FluxVLAPolicyClient(transport={
        'image_keys': ['image'],
        'state_keys': ['state'],
    })
    transport = _FakeTransport('predict_response')
    client.transport = transport

    client.setup()
    client.reset(TaskSpec('pick'), episode_index=2, policy_seed=11)
    client.set_episode_id('session:3')
    chunk = client.predict_action(
        {
            'image': np.zeros((4, 4, 3), dtype=np.uint8),
            'state': np.zeros(7),
        }, {
            'task_description': 'pick the object',
        })

    header, payload = transport.requests[0]
    assert transport.setup_requirements == [False]
    assert header['type'] == 'predict_action'
    assert header['policy_seed'] == 11
    assert payload['task_description'] == 'pick the object'
    assert chunk.actions.shape == (2, 3)
    assert chunk.metadata['episode_id'] == 'session:3'
    assert not hasattr(client, 'report_evaluation')


def test_reporter_only_handles_evaluation_lifecycle():
    reporter = FluxVLAEvaluationReporter(transport={})
    transport = _FakeTransport('report_response')
    reporter.transport = transport

    reporter.setup()
    run_session_id = reporter.run_session_id
    run_dir = reporter.report_evaluation('run_start', {'base_seed': 7})
    reporter.report_evaluation('run_end', {'status': 'finished'})

    first_header, _ = transport.requests[0]
    second_header, _ = transport.requests[1]
    assert transport.setup_requirements == [True]
    assert first_header['run_session_id'] == run_session_id
    assert first_header['sequence'] == 1
    assert second_header['sequence'] == 2
    assert run_dir == '/tmp/evaluation'
    assert reporter.evaluation_report_run_dir == '/tmp/evaluation'
    assert not hasattr(reporter, 'predict_action')


@pytest.mark.parametrize('value', [
    np.ones((2, 8), dtype=bool),
    np.full((2, 8), '0.25'),
    np.full((2, 8), 0.25 + 1j),
    [[True] + [0.25] * 7] * 2,
    [np.zeros(8), np.ones(8, dtype=bool)],
])
def test_policy_rejects_non_real_response_actions(value):
    client = FluxVLAPolicyClient({
        'image_keys': ['image'],
        'state_keys': ['state']
    })
    header = dict(type='predict_response', ok=True, request_id='request')
    with pytest.raises(TypeError, match='real numbers'):
        client._validate_response(header, encode_frame({'actions': value}),
                                  'request')


@pytest.mark.parametrize('value', [np.nan, np.inf, 1e100])
def test_policy_rejects_nonfinite_or_overflowed_actions(value):
    client = FluxVLAPolicyClient({
        'image_keys': ['image'],
        'state_keys': ['state']
    })
    header = dict(type='predict_response', ok=True, request_id='request')
    with pytest.raises(RuntimeError, match='NaN|infinite|FP32'):
        client._validate_response(
            header, encode_frame({'actions': np.full((2, 8), value)}),
            'request')


@pytest.mark.parametrize('width', [8, 14, 29])
def test_policy_preserves_numeric_actions_and_resets_request_identity(width):
    client = FluxVLAPolicyClient({
        'image_keys': ['image'],
        'state_keys': ['state']
    })
    transport = _FakeTransport('predict_response')
    client.transport = transport
    observation = dict(
        image=np.zeros((4, 4, 3), np.uint8), state=np.zeros(width))
    context = {'task_description': 'pick'}
    for episode in range(2):
        client.reset(TaskSpec('pick'), episode, policy_seed=10 + episode)
        client.set_episode_id(f'run:{episode + 1}')
        for sequence in (1, 2):
            chunk = client.predict_action(observation, context)
            assert chunk.metadata['sequence'] == sequence
            assert chunk.metadata['episode_id'] == f'run:{episode + 1}'
            assert chunk.metadata['policy_seed'] == 10 + episode
    assert len({h['request_id'] for h, _ in transport.requests}) == 4
    values = np.arange(width * 3, dtype=np.float64).reshape(3, width) / 8
    actual, _ = client._validate_response(
        dict(type='predict_response', ok=True, request_id='request'),
        encode_frame({'actions': values}), 'request')
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, values.astype(np.float32))


class StreamTransport:
    server_deployment_id = 'server'
    request_timeout_s = 5

    def __init__(self):
        self.headers = []
        self.bad = False

    def request(self, header, payload):
        header = decode_header(header)
        self.headers.append(header)
        response = dict(
            type='predict_response',
            ok=True,
            request_id=header['request_id'],
            episode_id='wrong' if self.bad else header['episode_id'],
            sequence=header['sequence'],
            sampling_mode=header['sampling_mode'])
        return response, encode_frame(
            dict(
                actions=np.ones((2, 8)),
                rng_state=bytes([header['sequence']]) * 16,
                rng_format='test-runtime-format'))

    def validate_response(self, header, request_id):
        assert header['ok'] and header['request_id'] == request_id

    def close(self):
        pass


def test_state_roundtrip_commits_only_valid_response_and_resets():
    client = FluxVLAPolicyClient(
        dict(image_keys=['image'], state_keys=['states']),
        sampling_mode='episode_noise_stream_v1')
    transport = StreamTransport()
    client.transport = transport
    observation = dict(image=np.zeros((2, 2, 3), np.uint8), states=np.zeros(8))
    context = dict(task_description='pick')
    client.reset(TaskSpec('pick'), 0, 57)
    first = client.predict_action(observation, context)
    second = client.predict_action(observation, context)
    assert transport.headers[0]['rng_state'] is None
    assert transport.headers[1]['rng_state'] == bytes([1]) * 16
    assert second.metadata['input_rng_state_sha256'] == first.metadata[
        'output_rng_state_sha256']
    before = client._rng_state
    transport.bad = True
    with pytest.raises(RuntimeError, match='identity'):
        client.predict_action(observation, context)
    assert client._rng_state == before
    client.set_episode_id('new-id')
    assert client._rng_state is None and client._prediction_sequence == 0
    transport.bad = False
    client.predict_action(observation, context)
    assert transport.headers[-1]['sequence'] == 1
    assert transport.headers[-1]['rng_state'] is None
    client.reset(TaskSpec('pick'), 1, 1057)
    assert client._rng_state is None and client._rng_format is None
    client.close()
    assert client._rng_state is None


class ProcessTransport:
    server_deployment_id = 'process-server'
    request_timeout_s = 5

    def __init__(self, shape=(3, 14)):
        self.shape = shape
        self.ordinal = 0
        self.instance = 'model-process-1'
        self.change = {}
        self.requests = []

    def request(self, header, payload):
        header = decode_header(header)
        self.requests.append(header)
        self.ordinal += 1
        response = dict(
            type='predict_response',
            ok=True,
            request_id=header['request_id'],
            inference_time_s=0.,
            episode_id=header['episode_id'],
            sequence=header['sequence'],
            sampling_mode='process_noise_stream_v1',
            process_instance_id=self.instance,
            process_sequence=self.ordinal)
        response.update(self.change)
        return response, encode_frame(
            dict(actions=np.ones(self.shape, np.float32)))

    def validate_response(self, header, request_id):
        assert header['ok'] and header['request_id'] == request_id


def client(transport):
    value = FluxVLAPolicyClient(
        dict(image_keys=['image'], state_keys=['states']),
        sampling_mode='process_noise_stream_v1')
    value.transport = transport
    return value


def predict(value):
    return value.predict_action(
        dict(image=np.zeros((2, 2, 3), np.uint8), states=np.zeros(14)),
        dict(task_description='pick'))


def test_ten_lanes_and_task_reset_preserve_one_process_identity():
    transport = ProcessTransport()
    clients = [client(transport) for _ in range(10)]
    for task in ['a', 'b']:
        for lane, value in enumerate(clients):
            value.reset(TaskSpec(task), lane, lane)
            value.set_episode_id(f'{task}:{lane}')
            chunk = predict(value)
            assert chunk.metadata['process_instance_id'] == transport.instance
            assert chunk.metadata['sampling_rng_format'] is None
    assert transport.ordinal == 20
    assert all(h['rng_state'] is None for h in transport.requests)
    transport.instance = 'replacement'
    clients[0].reset(TaskSpec('c'), 0, 200)
    with pytest.raises(RuntimeError, match='restarted or changed worker'):
        predict(clients[0])


@pytest.mark.parametrize('change', [
    dict(process_sequence=1),
    dict(process_sequence=True),
    dict(episode_id='wrong'),
    dict(sampling_mode='legacy_episode'),
])
def test_invalid_response_does_not_advance_process_position(change):
    transport = ProcessTransport()
    value = client(transport)
    value.reset(TaskSpec('a'), 0, 1)
    predict(value)
    transport.change = change
    with pytest.raises(
            RuntimeError, match='process sampling response mismatch'):
        predict(value)
    assert value._process_sequence == 1


@pytest.mark.parametrize('shape', [(1, 8), (15, 8), (3, 14), (32, 7)])
def test_process_client_keeps_model_action_chunk_shape(shape):
    transport = ProcessTransport(shape)
    value = client(transport)
    value.reset(TaskSpec('custom-policy'), 0, 0)
    chunk = predict(value)
    assert chunk.actions.shape == shape
    np.testing.assert_array_equal(chunk.actions, np.ones(shape, np.float32))
