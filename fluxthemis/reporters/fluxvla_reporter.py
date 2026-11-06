"""FluxVLA evaluation lifecycle reporter used by the coordinator."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fluxthemis.engines.runners.resume import (atomic_json, digest,
                                               read_bundle, validate_bundle)
from fluxthemis.engines.utils import EVALUATION_REPORTERS
from fluxthemis.transports.fluxvla_zmq import FluxVLAZMQTransport
from fluxthemis.transports.zmq_protocol import (decode_frame, encode_frame,
                                                encode_header)

from .base import BaseEvaluationReporter

REPORT_EVENT_TYPES = ('run_start', 'episode_start', 'episode_end', 'run_end')


@EVALUATION_REPORTERS.register_module()
class FluxVLAEvaluationReporter(BaseEvaluationReporter):
    """Publish lifecycle events to FluxVLA's authoritative report service."""

    def __init__(self,
                 transport: Mapping[str, Any],
                 event_log: str | None = None) -> None:
        self.transport = FluxVLAZMQTransport(transport)
        self._run_session_id: str | None = None
        self._sequence = 1
        self._run_dir: str | None = None
        self._event_log = Path(event_log) if event_log else None
        self._event_stream = None

    def setup(self) -> None:
        self.transport.setup(require_evaluation_reporting=True)
        self._run_session_id = uuid.uuid4().hex
        self._sequence = 1
        self._run_dir = None
        if self._event_log is not None:
            try:
                self._event_log.parent.mkdir(parents=True, exist_ok=True)
                self._event_stream = self._event_log.open(
                    'x', encoding='utf-8')
            except OSError:
                self.close()
                raise

    @property
    def run_session_id(self) -> str:
        if self._run_session_id is None:
            raise RuntimeError(
                'FluxVLAEvaluationReporter.setup() was not called')
        return self._run_session_id

    @property
    def evaluation_report_run_dir(self) -> str | None:
        return self._run_dir

    def persist_resume_input(self, bundle: Mapping[str, Any]) -> None:
        if self._event_log is None:
            return
        validate_bundle(bundle)
        path = self._event_log.parent / 'resume_input.json'
        if path.exists():
            if digest(read_bundle(path)) != digest(bundle):
                raise ValueError('Conflicting journal-local resume_input.json')
            return
        atomic_json(path, bundle)

    def report_evaluation(self, event_type: str, payload: Mapping[str,
                                                                  Any]) -> str:
        if event_type not in REPORT_EVENT_TYPES:
            raise ValueError(
                f'Unsupported evaluation report event type {event_type!r}; '
                f'expected one of {sorted(REPORT_EVENT_TYPES)}')
        if not isinstance(payload, Mapping):
            raise TypeError('evaluation report payload must be a mapping')

        request_id = uuid.uuid4().hex
        header = encode_header(
            'report_event',
            deployment_id=self.transport.server_deployment_id,
            request_id=request_id,
            run_session_id=self.run_session_id,
            event_type=event_type,
            sequence=self._sequence,
        )
        response_header, response_payload = self.transport.request(
            header, encode_frame(dict(payload)))
        run_dir = self._validate_response(response_header, response_payload,
                                          request_id)
        if self._run_dir is not None and run_dir != self._run_dir:
            raise RuntimeError(
                'FluxVLA ZMQ report service changed the authoritative '
                f'run_dir from {self._run_dir!r} to {run_dir!r}')
        self._run_dir = run_dir
        self._sequence += 1
        # Only mirror events acknowledged by the authoritative remote writer.
        if self._event_stream is not None:
            self._event_stream.write(
                json.dumps(
                    dict(
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                        event_type=event_type,
                        run_session_id=self.run_session_id,
                        sequence=self._sequence - 1,
                        run_dir=run_dir,
                        payload=dict(payload))) + '\n')
            self._event_stream.flush()
            os.fsync(self._event_stream.fileno())
        return run_dir

    def close(self) -> None:
        self.transport.close()
        if self._event_stream is not None:
            self._event_stream.close()
            self._event_stream = None
        self._run_session_id = None

    def _validate_response(self, header: Mapping[str, Any], payload: bytes,
                           request_id: str) -> str:
        self.transport.validate_response(header, request_id)
        if header.get('type') != 'report_response':
            raise RuntimeError('FluxVLA returned an invalid report type')
        body = decode_frame(payload)
        if not isinstance(body, Mapping):
            raise TypeError('FluxVLA report response must be a mapping')
        if not bool(body.get('accepted', False)):
            raise RuntimeError(f'FluxVLA rejected evaluation report: '
                               f"{body.get('error') or 'unknown error'}")
        run_dir = body.get('run_dir')
        if not isinstance(run_dir, str) or not run_dir.strip():
            raise ValueError('report run_dir must be a non-empty string')
        return run_dir.strip()


__all__ = ['FluxVLAEvaluationReporter']
