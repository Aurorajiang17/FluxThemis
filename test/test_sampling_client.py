import numpy as np
import pytest

from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient
from fluxthemis.structures import TaskSpec
from fluxthemis.transports.zmq_protocol import decode_header, encode_frame


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
