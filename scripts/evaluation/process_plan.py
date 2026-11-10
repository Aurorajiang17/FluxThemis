"""Materialize explicit, independent model/simulator process streams.

Planning reads configs only. All execution uses the ordinary public runners.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

from mmengine import Config

from .config import integer, mapping, path_value, read_yaml, select_gpus

ROOT = Path(__file__).resolve().parents[2]


def model_recipe(path, environment):
    """Resolve model asset variables using the model's declared environment."""
    previous = {key: os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        return Config.fromfile(path, lazy_import=False).to_dict()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_process_plan(path,
                      process_seeds=None,
                      gpu_ids=None,
                      render_devices=None,
                      num_streams=None,
                      tasks=None):
    path = Path(path).expanduser().resolve(strict=True)
    values = mapping(
        read_yaml(path), 'process launch',
        dict(
            version=1,
            fluxvla_root=None,
            model_python=None,
            simulator_python=None,
            robolab_root=None,
            output_root=None,
            cache_root=None,
            streams=None,
            tasks='all',
            num_streams=1,
            report_base_seed=0,
            model_config=None,
            simulator_config=None,
            process_seeds=[0],
            gpu_ids=None,
            render_devices=None,
            cpus_per_gpu=8,
            cpu_base=0,
            port_base=20800,
            startup_timeout_s=1200,
            episode_timeout_s=3600,
            stream_timeout_s=21600,
            allow_root=False,
            model_env={},
            simulator_env={}))
    integer(values['version'], 'version', maximum=1)
    for key in ('fluxvla_root', 'model_python', 'simulator_python',
                'robolab_root', 'output_root', 'cache_root', 'model_config',
                'simulator_config'):
        values[key] = path_value(
            values[key],
            key,
            path.parent,
            resolve=key not in ('model_python', 'simulator_python'))
    output, cache = Path(values['output_root']), Path(values['cache_root'])
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError('Output and cache directories must not overlap')
    for key in ('startup_timeout_s', 'episode_timeout_s', 'stream_timeout_s'):
        integer(values[key], key)
    integer(values['cpus_per_gpu'], 'cpus_per_gpu', minimum=3)
    integer(values['cpu_base'], 'cpu_base', minimum=0)
    integer(values['port_base'], 'port_base', maximum=65535)
    if type(values['allow_root']) is not bool:
        raise ValueError('allow_root must be boolean')
    for key in ('model_env', 'simulator_env'):
        env = values[key]
        if not isinstance(env, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in env.items()):
            raise ValueError(f'{key} must map environment names to strings')
        forbidden = {
            'PYTHONPATH', 'PYTHONHOME', 'CUDA_VISIBLE_DEVICES', 'WORLD_SIZE',
            'RANK', 'LOCAL_RANK', 'FLUXVLA_SDPA_BACKENDS'
        }
        if set(env) & forbidden:
            raise ValueError(
                f'{key} cannot override worker routing or Python imports')
    seeds = values['process_seeds'] if process_seeds is None else process_seeds
    if not isinstance(seeds, list) or not seeds or len(
            set(seeds)) != len(seeds):
        raise ValueError('process_seeds must be a nonempty unique list')
    for seed in seeds:
        integer(seed, 'process_seed', minimum=0, maximum=2**32 - 1)
    if values['streams'] is not None:
        if num_streams is not None or tasks is not None:
            raise ValueError(
                'An explicit stream manifest fixes tasks and groups')
        values['streams'] = path_value(values['streams'], 'streams',
                                       path.parent)
        manifest = json.loads(Path(values['streams']).read_text())
    else:
        from preflight.robolab import _task_sources
        selected = values['tasks'] if tasks is None else tasks
        canonical = sorted(
            _task_sources(Path(values['robolab_root']), selected))
        count = integer(
            values['num_streams'] if num_streams is None else num_streams,
            'num_streams',
            maximum=len(canonical))
        sim = Config.fromfile(values['simulator_config'], lazy_import=False)
        batch = sim.themis.evaluation.episode_batch_size
        manifest = dict(
            version=1,
            canonical_tasks=canonical,
            episode_batch_size=batch,
            report_base_seed=values['report_base_seed'],
            streams=[
                dict(id=i, tasks=canonical[i::count]) for i in range(count)
            ])

    canonical, streams = manifest['canonical_tasks'], manifest['streams']
    if (manifest['version'] != 1 or len(canonical) != len(set(canonical))
            or not canonical
            or any(not isinstance(t, str) or not t for t in canonical)):
        raise ValueError('Invalid canonical task manifest')
    tasks = [t for stream in streams for t in stream['tasks']]
    if (sorted(tasks) != sorted(canonical) or len(set(tasks)) != len(tasks)
            or [s['id'] for s in streams] != list(range(len(streams)))
            or any(not s['tasks'] for s in streams)):
        raise ValueError(
            'Each canonical task must occur in exactly one ordered stream')
    batch = integer(manifest['episode_batch_size'], 'episode_batch_size')
    report_seed = integer(
        manifest['report_base_seed'], 'report_base_seed', minimum=0)
    devices = values['gpu_ids'] if gpu_ids is None else gpu_ids
    if (not isinstance(devices, list) or not 1 <= len(devices) <= len(streams)
            or any(type(d) not in (str, int) for d in devices)
            or len(set(map(str, devices))) != len(devices)):
        raise ValueError(
            'gpu_ids must list unique available GPUs, up to the stream count')
    values['gpu_ids'] = select_gpus(explicit=','.join(map(str, devices)))
    render = values[
        'render_devices'] if render_devices is None else render_devices
    if not isinstance(render, list) or len(render) != len(devices):
        raise ValueError('render_devices must match gpu_ids')
    for ordinal in render:
        integer(ordinal, 'render_device', minimum=0)
    values['render_devices'] = render
    if values['port_base'] + 4 * len(streams) - 1 > 65535:
        raise ValueError('Stream ports exceed the valid range')
    values.update(
        process_seeds=seeds,
        manifest=manifest,
        batch_size=batch,
        report_base_seed=report_seed,
        source=str(path),
        manifest_sha256=hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    return values


def materialize(plan, folder):
    """Write public runner configs and explicit argv/env for every stream."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    vla = Path(plan['fluxvla_root'])
    model_base = Path(plan['model_config'])
    sim_base = Path(plan['simulator_config'])
    # Expanded configs are saved so inheritance is reviewable before execution.
    model_template = model_recipe(model_base, plan['model_env'])
    sim_template = Config.fromfile(sim_base, lazy_import=False).to_dict()
    if model_template['themis']['server']['runner'][
            'sampling_mode'] != 'process_noise_stream_v1':
        raise ValueError(
            'Companion FluxVLA does not supply the process recipe')
    if (sim_template['themis']['evaluation']['episode_batch_size']
            != plan['batch_size']
            or sim_template['themis']['evaluation']['episodes_per_task']
            != plan['batch_size']):
        raise ValueError('Stream manifest and native batch recipe differ')
    canonical = plan['manifest']['canonical_tasks']
    result = dict(settings=plan, directory=str(folder), streams=[])
    for seed in plan['process_seeds']:
        for stream in plan['manifest']['streams']:
            index = stream['id']
            run = folder / f'seed-{seed}' / f'stream-{index}'
            run.mkdir(parents=True)
            slot = index % len(plan['gpu_ids'])
            gpu = str(plan['gpu_ids'][slot])
            start = plan['cpu_base'] + plan['cpus_per_gpu'] * slot
            model_cpus, sim_cpus = list(range(start, start + 2)), list(
                range(start + 2, start + plan['cpus_per_gpu']))
            port = plan['port_base'] + index * 4
            cache = Path(plan['cache_root']
                         ) / folder.name / f'seed-{seed}' / f'stream-{index}'
            model = Config(copy.deepcopy(model_template))
            model.themis.server.runner.process_seed = seed
            model.themis.server.evaluation_reporting.result_output_dir = str(
                run / 'model-results')
            model_file = run / 'model.py'
            model.dump(model_file)
            sim = Config(copy.deepcopy(sim_template))
            evaluation = sim.themis.evaluation
            evaluation.environment.update(
                task_name=list(stream['tasks']),
                robolab_root=plan['robolab_root'],
                cpu_thread_limit=len(sim_cpus),
                rendering_mode='balanced',
                render_device_map={gpu: plan['render_devices'][slot]},
                kit_cache_root=str(cache / 'sim-kit'),
                kit_args=('--allow-root ' if plan['allow_root'] else '') +
                '--/renderer/multiGpu/enabled=false')
            evaluation.update(
                task_ids=list(stream['tasks']),
                task_schedule=list(stream['tasks']),
                base_seed=plan['report_base_seed'],
                task_seed_offsets={
                    t: plan['batch_size'] * canonical.index(t)
                    for t in stream['tasks']
                })
            sim_file = run / 'sim.py'
            sim.dump(sim_file)
            deployment = dict(
                version=1,
                simulator=dict(
                    python=plan['simulator_python'],
                    config=str(sim_file),
                    robolab_root=plan['robolab_root'],
                    gpu_ids=[gpu],
                    workers_per_gpu=1,
                    coordinator_port=port + 2,
                    scheduling='dynamic',
                    simulator_restart_limit=0,
                    save_videos=False,
                    episode_timeout_s=plan['episode_timeout_s']),
                model=dict(
                    host='127.0.0.1',
                    port=port,
                    expected_workers=1,
                    wait_seconds=plan['startup_timeout_s']),
                run=dict(output_root=str(run / 'simulation'), max_minutes=0))
            import yaml
            deployment_file = run / 'deployment.yaml'
            deployment_file.write_text(
                yaml.safe_dump(deployment, sort_keys=False))
            resume = model.themis.server.runner.get('process_resume')
            offset = 0
            if resume:
                if len(plan['process_seeds']) * len(
                        plan['manifest']['streams']) != 1:
                    raise ValueError(
                        'A resume state belongs to one logical stream')
                raw = Path(resume['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != resume['sha256']:
                    raise ValueError('Process resume state hash mismatch')
                offset = json.loads(raw)['sequence']
            row = dict(
                process_request_offset=offset,
                seed=seed,
                stream=index,
                gpu=gpu,
                tasks=stream['tasks'],
                episodes=len(stream['tasks']) * plan['batch_size'],
                directory=str(run),
                cache=str(cache),
                model_cpus=model_cpus,
                simulator_cpus=sim_cpus,
                model_config=str(model_file),
                simulator_config=str(sim_file),
                deployment=str(deployment_file))
            row['model_command'] = [
                'taskset', '-c', ','.join(map(str, model_cpus)),
                plan['model_python'], '-B',
                str(vla / 'scripts/distributed_zmq_server.py'), '--config',
                str(model_file), '--frontend-bind', f'tcp://127.0.0.1:{port}',
                '--backend-bind', f'tcp://127.0.0.1:{port + 1}',
                '--backend-endpoint', f'tcp://127.0.0.1:{port + 1}',
                '--startup-timeout-s',
                str(plan['startup_timeout_s']), '--request-timeout-s', '120'
            ]
            row['simulator_command'] = [
                'taskset', '-c', ','.join(map(str, sim_cpus)),
                plan['simulator_python'], '-B',
                str(ROOT / 'scripts/run_evaluation.py'), '--config',
                str(deployment_file), '--result-pointer',
                str(run / 'result-pointer.json')
            ]
            row['config_sha256'] = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (model_file, sim_file, deployment_file)
            }
            result['streams'].append(row)
    (folder / 'plan.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
