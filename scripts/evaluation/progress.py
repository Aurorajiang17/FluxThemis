"""Common stages and acknowledged episode progress for all benchmarks."""

import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

STAGES = ('Configuration', 'Simulator preflight', 'Model readiness',
          'Evaluation', 'Result validation')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Progress:

    def __init__(self):
        self.stage = 0
        self.started = time.monotonic()
        self.output = None
        self.stream = sys.stdout
        self._line_width = 0
        self._last_log = None
        self._last_log_time = 0
        self._episode_counts = None

    def close(self):
        if self._line_width:
            print(file=self.stream, flush=True)
            self._line_width = 0

    def enter(self, stage):
        self.close()
        self.stage = stage
        self._episode_counts = None
        self.started = time.monotonic()
        self.show('Starting')

    def show(self,
             message,
             current=None,
             total=None,
             status='Running',
             *,
             live=False,
             eta_s=None):
        now = time.monotonic()
        elapsed = int(now - self.started)
        if (status in ('Failed', 'Cancelled') and current is None
                and self._episode_counts is not None):
            current, total = self._episode_counts
        fraction = (f' {current}/{total} ({100 * current / total:.1f}%)'
                    if total else '')
        if live:
            self._episode_counts = (current, total)
            filled = min(20, int(20 * current / total)) if total else 20
            bar = '#' * filled + '-' * (20 - filled)
            eta = '--:--:--'
            if eta_s is not None:
                seconds = max(0, int(eta_s))
                eta = (f'{seconds // 3600:02d}:'
                       f'{seconds // 60 % 60:02d}:{seconds % 60:02d}')
            line = f'{message} [{bar}] {current}/{total} ETA {eta}'
            if status != 'Evaluating':
                line += f' ({status})'
        else:
            line = (f'[{self.stage}/{len(STAGES)} '
                    f'{STAGES[self.stage - 1]}][{status}]'
                    f'{fraction} {message}')

        # Keep redirected output sparse while progress.json stays current.
        key = (self.stage, status, message, current, total)
        log = (
            self._last_log is None or key[:3] != self._last_log[:3]
            or (current == total and key != self._last_log)
            or now - self._last_log_time >= 60)
        if live and self.stream.isatty():
            width = max(1, (shutil.get_terminal_size().columns or 80) - 1)
            visible = line[:width]
            print(
                '\r' + visible.ljust(min(self._line_width, width)),
                end='',
                file=self.stream,
                flush=True)
            self._line_width = len(visible)
        elif log:
            self.close()
            print(line, file=self.stream, flush=True)

        if self.output:
            if log:
                with (self.output / 'progress.log').open('a') as stream:
                    stream.write(line + '\n')
            write_json(
                self.output / 'progress.json',
                dict(
                    stage=self.stage,
                    title=STAGES[self.stage - 1],
                    status=status,
                    message=message,
                    current=current,
                    total=total,
                    eta_s=eta_s,
                    stage_elapsed_s=elapsed))
        if log:
            self._last_log = key
            self._last_log_time = now


class Events:

    def __init__(self):
        self.position = 0
        self.pending = ''
        self.sequence = 0
        self.session = None
        self.total = None
        self.episodes = {}
        self.episode_starts = {}
        self.restored = set()
        self.end = None
        self.run_dir = None

    def read(self, path):
        if not path.is_file():
            return
        with path.open() as stream:
            stream.seek(self.position)
            self.pending += stream.read()
            self.position = stream.tell()
        lines = self.pending.split('\n')
        self.pending = lines.pop()
        for line in lines:
            self.accept(json.loads(line))

    def accept(self, event):
        if event['sequence'] != self.sequence + 1:
            raise ValueError('Local event sequence is not contiguous')
        session = event['run_session_id']
        if self.session is not None and self.session != session:
            raise ValueError(
                'Local events contain different evaluation sessions')
        self.sequence = event['sequence']
        self.session = event['run_session_id']
        self.run_dir = event['run_dir']
        kind, body = event['event_type'], event['payload']
        if kind == 'run_start':
            if self.total is not None:
                raise ValueError('Duplicate run_start')
            self.total = body['total_episodes']
        elif kind == 'episode_start':
            self.episode_starts[body['episode_id']] = body.get('started_at')
            if body.get('resumed_from'):
                self.restored.add(body['episode_id'])
        elif kind == 'episode_end':
            key = body['episode_id']
            if key in self.episodes:
                raise ValueError('Duplicate episode_end')
            self.episodes[key] = body
            if body['completed_episodes'] != len(self.episodes):
                raise ValueError(
                    'Completed episode count does not match episode records')
        elif kind == 'run_end':
            self.end = body

    def result(self, exit_code):
        successes = sum(row['success'] for row in self.episodes.values())
        complete = bool(exit_code == 0 and self.end
                        and self.end['status'] == 'finished'
                        and self.total == len(self.episodes)
                        and self.end['completed_episodes'] == self.total
                        and self.end['total_episodes'] == self.total)
        if complete:
            overall = self.end['metrics']['overall']
            complete = (
                overall['episodes'] == self.total
                and overall['successes'] == successes)
        return dict(
            complete=complete,
            simulator_exit_code=exit_code,
            completed=len(self.episodes),
            total=self.total,
            successes=successes,
            remote_result_dir=self.run_dir,
            metrics=self.end.get('metrics') if self.end else None)


class SimulatorStartupLog:
    """Track current readiness and surface errors hidden in the raw log."""

    def __init__(self):
        self.ready = set()
        self.lines_seen = 0
        self.last_error = ''
        self.last_change = time.monotonic()

    def update(self, text):
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith('\n'):
            lines.pop()  # A writer may still be completing the final line.
        for line in lines[self.lines_seen:]:
            ready = re.search(r'simulator worker ready id=(\S+)', line)
            restart = re.search(
                r'restarting (simulator-worker-\S+) '
                r'after exit code (\S+)', line)
            if ready:
                self.ready.add(ready.group(1))
                self.last_change = time.monotonic()
            if restart:
                self.ready.discard(restart.group(1))
            if restart or re.search(
                    r'Traceback|Fatal Python error|(?:Error|Exception):',
                    line):
                self.last_error = line.strip()[-400:]
                print('[SIM_STARTUP_ERROR] ' + self.last_error, flush=True)
        self.lines_seen = len(lines)
        idle = int(time.monotonic() - self.last_change)
        message = ('; waiting for workers to become ready'
                   if idle >= 60 else '')
        if self.last_error:
            message += '; latest error: ' + self.last_error
        return message


def validate_process_chain(records):
    """Require one model instance and a complete sequence starting at one."""
    instances, sequences = set(), []

    def consume(metadata, record):
        for ordinal, item in enumerate(metadata, 1):
            if (item['sampling_mode'] != 'process_noise_stream_v1'
                    or item['sequence'] != ordinal
                    or item['episode_id'] != record['episode_id']
                    or item['policy_seed'] != record['policy_seed']
                    or not item['process_instance_id']
                    or type(item['process_sequence']) is not int):
                raise ValueError('Model request identity differs')
            instances.add(item['process_instance_id'])
            sequences.append(item['process_sequence'])

    for record in records:
        metadata = record['prediction_metadata']
        if not metadata or len(metadata) != record['model_calls']:
            raise ValueError('Missing model request metadata')
        consume(metadata, record)
    if len(instances) != 1 or sorted(sequences) != list(
            range(1,
                  len(sequences) + 1)):
        raise ValueError(
            'Process changed or request sequence has gaps/duplicates')
    return instances.pop()


def stream_results(settings, row):
    directory = Path(row['directory'])
    pointer = json.loads((directory / 'result-pointer.json').read_text())
    local = Path(pointer['directory'])
    result = json.loads((local / 'result.json').read_text())
    events = Events()
    events.read(local / 'events.jsonl')
    computed = events.result(result['simulator_exit_code'])
    if events.pending or not computed['complete'] or not result['complete']:
        raise ValueError('Incomplete stream journal')
    if any(computed[k] != result[k]
           for k in ('completed', 'total', 'successes', 'remote_result_dir')):
        raise ValueError('Result summary disagrees with journal')
    model_root = (directory / 'model-results').resolve()
    remote = Path(events.run_dir).resolve()
    if model_root not in remote.parents:
        raise ValueError(
            'Model journal is outside this stream result directory')
    left = [
        json.loads(line)
        for line in (local / 'events.jsonl').read_text().splitlines()
    ]
    right = [
        json.loads(line)
        for line in (remote / 'events.jsonl').read_text().splitlines()
    ]
    fields = ('event_type', 'sequence', 'run_session_id', 'payload')
    if [[e[k] for k in fields] for e in left] != [[e[k] for k in fields]
                                                  for e in right]:
        raise ValueError('Model and simulator journals disagree')
    if left[0]['event_type'] != 'run_start' or left[-1][
            'event_type'] != 'run_end':
        raise ValueError('Missing terminal event boundary')
    start = left[0]['payload']
    if [t['task_id'] for t in start['tasks']] != row['tasks']:
        raise ValueError('Started task order differs from planned stream')
    records = list(events.episodes.values())
    batch = settings['batch_size']
    expected = {(task, lane) for task in row['tasks'] for lane in range(batch)}
    identities = [(r['task_id'], r['episode_index']) for r in records]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError('Duplicate, missing or unexpected task/lane identity')
    for record in records:
        task, lane = record['task_id'], record['episode_index']
        seed = settings['report_base_seed'] + batch * settings['manifest'][
            'canonical_tasks'].index(task) + lane
        if record['policy_seed'] != seed or record['environment_seed'] != seed:
            raise ValueError('Canonical task/lane seed differs')
        if type(record['success']) is not bool or type(
                record['score']) not in (int, float) or not math.isfinite(
                    record['score']):
            raise ValueError('Episode success/Score is missing or invalid')
    return records, validate_process_chain(records)


def collect_results(plan):
    repeats, all_instances = {}, set()
    for seed in plan['settings']['process_seeds']:
        records = {}
        for row in (r for r in plan['streams'] if r['seed'] == seed):
            values, instance = stream_results(plan['settings'], row)
            if instance in all_instances:
                raise ValueError(
                    'Independent streams shared one model process')
            all_instances.add(instance)
            for value in values:
                key = (value['task_id'], value['episode_index'])
                if key in records:
                    raise ValueError('Duplicate task/lane across streams')
                records[key] = value
        settings = plan['settings']
        expected = {(t, i)
                    for t in settings['manifest']['canonical_tasks']
                    for i in range(settings['batch_size'])}
        if set(records) != expected:
            raise ValueError(
                'Final episode coverage differs from requested manifest')
        values = [
            records[(t, i)] for t in settings['manifest']['canonical_tasks']
            for i in range(settings['batch_size'])
        ]
        successes = sum(r['success'] for r in values)
        repeats[str(seed)] = dict(
            completed=len(values),
            successes=successes,
            success_rate=successes / len(values),
            score=sum(r['score'] for r in values) / len(values))
        write_json(
            Path(plan['directory']) / f'episodes-seed-{seed}.json', values)
    write_json(
        Path(plan['directory']) / 'summary.json',
        dict(repeats=repeats, status='complete'))
    return repeats
