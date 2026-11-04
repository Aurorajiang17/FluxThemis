"""Exercise the real ZMQ framing and REQ socket timeout recovery."""

from contextlib import contextmanager
from queue import Queue
from threading import Event, Thread

import pytest
import zmq

from fluxthemis.transports.fluxvla_zmq import FluxVLAZMQTransport
from fluxthemis.transports.zmq_protocol import (decode_frame, decode_header,
                                                encode_frame, encode_header)


@contextmanager
def _server(handler):
    endpoints, errors = Queue(), Queue()
    stop = Event()

    def serve():
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            port = socket.bind_to_random_port('tcp://127.0.0.1')
            endpoints.put(f'tcp://127.0.0.1:{port}')
            while not stop.is_set():
                if socket.poll(20):
                    handler(socket, socket.recv_multipart())
        except BaseException as exc:
            errors.put(exc)
        finally:
            socket.close()
            context.term()

    thread = Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield endpoints.get(timeout=5)
    finally:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive(), 'test ZMQ server did not stop'
        if not errors.empty():
            raise errors.get()


def _reply(socket, frames, response_type, body, **overrides):
    request = decode_header(frames[-2])
    values = dict(request_id=request['request_id'], ok=True)
    values.update(overrides)
    socket.send_multipart([
        *frames[:-2],
        encode_header(response_type, **values),
        encode_frame(body)
    ])


def _status(socket, frames, **overrides):
    body = dict(
        deployment_id='server',
        server_mode='stateless',
        evaluation_reporting=True,
        workers_ready=1,
        workers_expected=1,
        status='ready')
    body.update(overrides)
    _reply(socket, frames, 'status_response', body)


@pytest.mark.parametrize('overrides', [
    {},
    {
        'server_mode': 'stateful'
    },
    {
        'evaluation_reporting': False
    },
    {
        'workers_ready': 0
    },
    {
        'workers_expected': True
    },
    {
        'deployment_id': ''
    },
])
def test_real_status_handshake_and_validation(overrides):
    with _server(lambda socket, frames: _status(socket, frames, **overrides)
                 ) as endpoint:
        transport = FluxVLAZMQTransport(
            dict(endpoint=endpoint, connect_timeout_s=2))
        try:
            if overrides:
                with pytest.raises(ValueError):
                    transport.setup(require_evaluation_reporting=True)
                assert transport._socket is None
                assert transport._context is None
            else:
                transport.setup(require_evaluation_reporting=True)
                assert transport.server_deployment_id == 'server'
        finally:
            transport.close()


def test_timeout_replaces_socket_without_replaying_or_accepting_late_reply():
    seen, held, routes = [], [], []

    def handle(socket, frames):
        header = decode_header(frames[-2])
        if header['type'] == 'status':
            return _status(socket, frames)
        seen.append(header['request_id'])
        routes.append(frames[0])
        if header['request_id'] == 'first':
            held.append(frames)
            return
        # Reply to the abandoned socket before replying to its replacement.
        _reply(socket, held[0], 'predict_response', {'value': 'late'})
        _reply(socket, frames, 'predict_response', {'value': 'current'})

    with _server(handle) as endpoint:
        transport = FluxVLAZMQTransport(
            dict(endpoint=endpoint, request_timeout_s=0.1))
        try:
            transport.setup()
            first_socket = transport._socket
            with pytest.raises(TimeoutError):
                transport.request(
                    encode_header('predict_action', request_id='first'),
                    encode_frame({}))
            assert first_socket.closed
            assert transport._socket is not first_socket
            header, payload = transport.request(
                encode_header('predict_action', request_id='second'),
                encode_frame({}),
                timeout_s=2)
            transport.validate_response(header, 'second')
            assert decode_frame(payload) == {'value': 'current'}
            assert seen == ['first', 'second']
            assert len(routes) == 2 and routes[0] != routes[1]
        finally:
            transport.close()


@pytest.mark.parametrize('fault', ['identity', 'server_error', 'frames'])
def test_real_response_protocol_faults_are_rejected(fault):

    def handle(socket, frames):
        if decode_header(frames[-2])['type'] == 'status':
            return _status(socket, frames)
        if fault == 'frames':
            socket.send_multipart([*frames[:-2], encode_frame({})])
        else:
            overrides = ({
                'request_id': 'wrong'
            } if fault == 'identity' else dict(
                ok=False,
                error_type='inference_error',
                error='fixture failure'))
            _reply(socket, frames, 'predict_response', {}, **overrides)

    with _server(handle) as endpoint:
        transport = FluxVLAZMQTransport(dict(endpoint=endpoint))
        try:
            transport.setup()
            with pytest.raises(
                    RuntimeError, match='mismatch|fixture failure|frames'):
                header, _ = transport.request(
                    encode_header('predict_action', request_id='expected'),
                    encode_frame({}),
                    timeout_s=2)
                transport.validate_response(header, 'expected')
        finally:
            transport.close()
