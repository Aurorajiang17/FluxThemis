#!/usr/bin/env python
"""Run fixed RoboLab process streams through the public FluxVLA/Themis CLIs.

Use --dry-run to inspect all expanded configs and commands without launching.
Each GPU runs its assigned streams sequentially; GPUs run concurrently.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation.process_plan import ROOT, load_process_plan, materialize
from evaluation.progress import collect_results, stream_results, write_json
from evaluation.runtime import ProcessTree


def child_environment(settings, row, role):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('TORCHELASTIC_') or key in {
                'PYTHONPATH',
                'PYTHONHOME',
                'RANK',
                'WORLD_SIZE',
                'LOCAL_RANK',
                'LOCAL_WORLD_SIZE',
                'GROUP_RANK',
                'MASTER_ADDR',
                'MASTER_PORT',
                'PYTHON_EXEC',
                'NPROC_PER_NODE',
                'FLUXVLA_SDPA_BACKENDS',
        }:
            env.pop(key, None)
    env.update(settings['model_env' if role == 'model' else 'simulator_env'])
    vla = settings['fluxvla_root']
    env.update(
        PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',
        CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES=row['gpu'],
        TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',
        ROBOLAB_ROOT=settings['robolab_root'],
        FLUXTHEMIS_PYTHON=settings['simulator_python'],
        FLUXVLA_DEPLOYMENT_ID=(f'{Path(row["directory"]).parent.parent.name}'
                               f'-s{row["seed"]}-g{row["stream"]}'),
        PYTHONPATH=os.pathsep.join([vla, str(ROOT)]) if role == 'model' else
        os.pathsep.join([str(ROOT /
                             'scripts'), str(ROOT)]))
    threads = 2 if role == 'model' else len(row['simulator_cpus'])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'PXR_WORK_THREAD_LIMIT'):
        env[key] = str(threads)
    for key, sub in [('XDG_RUNTIME_DIR', 'runtime'),
                     ('XDG_CONFIG_HOME', 'config'), ('TMPDIR', 'tmp'),
                     ('XDG_CACHE_HOME', 'xdg'), ('XDG_DATA_HOME', 'data'),
                     ('CUDA_CACHE_PATH', 'cuda'),
                     ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                     ('TRITON_CACHE_DIR', 'triton'),
                     ('OPTIX_CACHE_PATH', 'optix'),
                     ('TORCH_EXTENSIONS_DIR', 'extensions'),
                     ('MPLCONFIGDIR', 'matplotlib'),
                     ('__GL_SHADER_DISK_CACHE_PATH', 'shaders')]:
        directory = Path(row['cache']) / role / sub
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    Path(env['XDG_RUNTIME_DIR']).chmod(0o700)
    return env


def stop_child(process, tree):
    tree.signal_groups(signal.SIGTERM)
    until = time.monotonic() + 15
    while tree.members() and time.monotonic() < until:
        process.poll()
        time.sleep(.2)
    if tree.members():
        tree.signal_groups(signal.SIGKILL)
    process.wait(timeout=10)
    if tree.members():
        raise RuntimeError('Stream descendants did not exit')


def run_stream(settings, row, cancel):
    directory = Path(row['directory'])
    for path, digest in row['config_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Prepared config changed: {path}')
    if (directory / 'execution.json').exists():
        raise RuntimeError(
            'Stream already started; automatic restarts are disabled')
    started = time.time()
    state = dict(
        seed=row['seed'],
        stream=row['stream'],
        started=started,
        status='starting',
        processes={})
    children, logs = {}, []
    write_json(directory / 'execution.json', state)
    try:
        for role in ('model', 'simulator'):
            log = (directory / f'{role}.log').open('x')
            logs.append(log)
            process = subprocess.Popen(
                row[f'{role}_command'],
                cwd=settings['fluxvla_root'] if role == 'model' else ROOT,
                env=child_environment(settings, row, role),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True)
            tree = ProcessTree(process.pid)
            children[role] = (process, tree)
            state['processes'][role] = dict(
                pid=process.pid, start_ticks=tree.known.get(process.pid))
        state['status'] = 'running'
        write_json(directory / 'execution.json', state)
        model, model_tree = children['model']
        simulator, sim_tree = children['simulator']
        while simulator.poll() is None:
            for process, tree in children.values():
                tree.members()
            if cancel.is_set():
                raise RuntimeError(
                    'Run interrupted; preserved incomplete evidence')
            if time.time() - started > settings['stream_timeout_s']:
                raise TimeoutError('Stream wall-clock limit exceeded')
            if model.poll() not in (None, 0):
                raise RuntimeError(f'Model exited with {model.returncode}')
            time.sleep(1)
        if simulator.returncode:
            raise RuntimeError(
                f'Simulator exited with {simulator.returncode}; see logs')
        if model.wait(timeout=60):
            raise RuntimeError('Model did not finish successfully')
        pointer = json.loads((directory / 'result-pointer.json').read_text())
        result = json.loads(
            (Path(pointer['directory']) / 'result.json').read_text())
        if not result['complete'] or result['completed'] != row['episodes']:
            raise RuntimeError(
                'Stream incomplete; missing episodes remain unscored')
        stream_results(settings, row)
        state.update(status='complete', result=result)
        return row
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cleanup_errors = []
        for process, tree in children.values():
            try:
                stop_child(process, tree)
            except Exception as error:
                cleanup_errors.append(f'{type(error).__name__}: {error}')
        for log in logs:
            log.close()
        if cleanup_errors:
            state.update(status='failed', cleanup_errors=cleanup_errors)
            cancel.set()
        state['finished'] = time.time()
        write_json(directory / 'execution.json', state)
        if cleanup_errors and 'error' not in state:
            raise RuntimeError('; '.join(cleanup_errors))


def execute(plan):
    cancel = threading.Event()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: cancel.set())

    failures = []

    def slot(gpu):
        # Physical GPUs may serve several complete logical streams serially.
        # Never combine their model processes or reseed mid-stream.
        indices = sorted(
            {r['stream']
             for r in plan['streams'] if r['gpu'] == gpu})
        for index in indices:
            rows = [r for r in plan['streams'] if r['stream'] == index]
            if index % 2:
                rows.reverse()
            for row in rows:
                if cancel.is_set():
                    return
                try:
                    run_stream(plan['settings'], row, cancel)
                except Exception as error:
                    failures.append(
                        dict(seed=row['seed'], stream=index, error=str(error)))

    try:
        devices = plan['settings']['gpu_ids']
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(slot, gpu) for gpu in devices]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException:
                    cancel.set()
                    raise
        if cancel.is_set():
            raise RuntimeError('Evaluation interrupted')
        if failures:
            write_json(
                Path(plan['directory']) / 'failed-streams.json', failures)
            raise RuntimeError(
                'Some streams failed; healthy streams completed')
        return collect_results(plan)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path, default=ROOT / 'configs/robolab/process.yaml')
    parser.add_argument('--process-seeds', type=int, nargs='+')
    parser.add_argument(
        '--gpu-ids', help='Comma-separated GPU indices or UUIDs')
    parser.add_argument(
        '--render-devices',
        type=int,
        nargs='+',
        help='Verified Vulkan ordinal for each selected GPU')
    parser.add_argument('--num-streams', type=int)
    parser.add_argument('--tasks', nargs='+', help='Task names, or all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    settings = load_process_plan(
        args.config, args.process_seeds,
        args.gpu_ids.split(',') if args.gpu_ids is not None else None,
        args.render_devices, args.num_streams, args.tasks)
    folder = Path(settings['output_root']) / ('process_' +
                                              time.strftime('%Y%m%d_%H%M%S') +
                                              '_' + uuid.uuid4().hex[:6])
    plan = materialize(settings, folder)
    print(
        f'Prepared public configs and commands: {folder / "plan.json"}',
        flush=True)
    if args.dry_run:
        print('Dry run only: no model, simulation, network or GPU work.')
        return 0
    print(json.dumps(execute(plan)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
