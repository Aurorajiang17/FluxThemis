"""Public process-stream planning, lifecycle and evidence checks on CPU."""

import copy
import importlib
import json
import os
import sys
import threading
from pathlib import Path

import pytest
import yaml
from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    return importlib.import_module('run_robolab_process')


@pytest.fixture
def config(tmp_path):
    # A synthetic companion recipe makes these tests independent of FluxVLA.
    vla = tmp_path / 'vla'
    recipe = vla / 'configs/custom_policy/process_inference.py'
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        "themis = dict(server=dict(runner=dict("
        "sampling_mode='process_noise_stream_v1', process_seed=0), "
        "evaluation_reporting=dict()))")
    inherited = recipe.with_name('model_base.py')
    inherited.write_text('import os\n' + recipe.read_text())
    recipe.write_text("_base_ = './model_base.py'\n")
    manifest = tmp_path / 'streams.json'
    canonical = [f'task-{i:03d}' for i in range(120)]
    manifest.write_text(
        json.dumps(
            dict(
                version=1,
                canonical_tasks=canonical,
                episode_batch_size=10,
                report_base_seed=1200,
                streams=[dict(id=i, tasks=canonical[i::6])
                         for i in range(6)])))
    value = dict(
        version=1,
        fluxvla_root=str(vla),
        model_python=sys.executable,
        simulator_python=sys.executable,
        robolab_root=str(tmp_path / 'robolab'),
        output_root=str(tmp_path / 'outputs'),
        cache_root=str(tmp_path / 'cache'),
        streams=str(manifest),
        model_config=str(recipe),
        simulator_config=str(ROOT / 'configs/robolab/droid.py'),
        gpu_ids=list(range(6)),
        render_devices=list(range(6)))
    path = tmp_path / 'process.yaml'
    path.write_text(yaml.safe_dump(value))
    return path


def test_two_repeats_materialize_complete_independent_configs(
        launch, config, tmp_path):
    from evaluation.config import load_config

    from fluxthemis.engines.runners.config import \
        build_distributed_evaluation_run
    plan = launch.materialize(
        launch.load_process_plan(config, [0, 1]), tmp_path / 'plan')
    identities = {0: set(), 1: set()}
    assert len(plan['streams']) == 12
    for row in plan['streams']:
        model = Config.fromfile(row['model_config'])
        assert model.themis.server.runner.process_seed == row['seed']
        deployment = load_config(row['deployment'])
        assert deployment['model']['expected_workers'] == 1
        assert deployment['simulator_workers'] == 1
        settings = build_distributed_evaluation_run(
            row['simulator_config']).settings
        assert settings.task_ids == tuple(row['tasks'])
        assert settings.task_schedule == tuple(row['tasks'])
        assert row['episodes'] == 200
        assert len(row['model_cpus']) == 2 and len(row['simulator_cpus']) == 6
        assert deployment[
            'config'].themis.evaluation.environment.simulator_seed == 0
        for task in row['tasks']:
            identities[row['seed']].update(settings.base_seed +
                                           settings.task_seed_offsets[task] + i
                                           for i in range(10))
    assert identities == {0: set(range(1200, 2400)), 1: set(range(1200, 2400))}
    assert not Path(plan['settings']['cache_root']).exists()


@pytest.mark.parametrize('values', [
    dict(gpu_ids=[0, 0, 2, 3, 4, 5]),
    dict(gpu_ids=[-1, 1, 2, 3, 4, 5]),
    dict(process_seeds=[0, 0]),
    dict(model_env={'PYTHONPATH': '/private'}),
    dict(simulator_env={'CUDA_VISIBLE_DEVICES': 'hidden'})
])
def test_invalid_launch_plan(launch, config, values):
    data = yaml.safe_load(config.read_text())
    data.update(values)
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        launch.load_process_plan(config)


def test_dry_run_cannot_launch_or_allocate_cache(launch, config, monkeypatch):
    monkeypatch.setattr(launch.subprocess, 'Popen',
                        lambda *_a, **_k: pytest.fail('subprocess started'))
    assert launch.main(
        ['--config',
         str(config), '--process-seeds', '0', '1', '--dry-run']) == 0
    data = yaml.safe_load(config.read_text())
    assert not Path(data['cache_root']).exists()


def test_venv_interpreter_symlink_is_preserved(launch, config, tmp_path):
    path = tmp_path / 'venv/bin/python'
    path.parent.mkdir(parents=True)
    path.symlink_to(sys.executable)
    data = yaml.safe_load(config.read_text())
    data['simulator_python'] = str(path)
    config.write_text(yaml.safe_dump(data))
    assert launch.load_process_plan(config)['simulator_python'] == str(path)


def test_task_shards_preserve_old_ordinals_and_canonical_seeds():
    from fluxthemis.engines.runners.coordinator import map_task_seeds
    from fluxthemis.engines.runners.episode import build_episode_jobs
    from fluxthemis.engines.runners.evaluation import EvaluationSettings
    from fluxthemis.structures import TaskSpec
    jobs = build_episode_jobs([TaskSpec('b'), TaskSpec('a')],
                              episodes_per_task=10,
                              base_seed=1200,
                              run_session_id='test')
    assert map_task_seeds(jobs, EvaluationSettings()) is jobs
    settings = EvaluationSettings(
        base_seed=1200, task_seed_offsets={
            'b': 70,
            'a': 20
        })
    mapped = map_task_seeds(jobs, settings)
    for old, new in zip(jobs, mapped):
        assert new.episode_id == old.episode_id
        assert new.environment_case_index == old.environment_case_index
        assert new.policy_seed == 1200 + settings.task_seed_offsets[
            new.task.task_id] + new.episode_index
        assert new.environment_seed == new.policy_seed
    with pytest.raises(ValueError, match='overlapping'):
        map_task_seeds(jobs,
                       EvaluationSettings(task_seed_offsets={
                           'b': 0,
                           'a': 0
                       }))


def test_process_high_level_rejects_shared_workers_and_resume(
        launch, config, tmp_path):
    from evaluation.config import load_config
    plan = launch.materialize(
        launch.load_process_plan(config), tmp_path / 'plan')
    row = plan['streams'][0]
    for overrides in (
            dict(workers_per_gpu=2),
            dict(
                cfg_options={'themis.evaluation.resume_path': '/old/result'})):
        with pytest.raises(ValueError, match='Process sampling'):
            load_config(row['deployment'], overrides=overrides)


def test_fixed_slot_order_and_failure_isolated(launch, monkeypatch, tmp_path):
    seen = {0: [], 1: []}
    plan = dict(
        settings=dict(gpu_ids=['0', '1']),
        streams=[
            dict(seed=seed, stream=gpu, gpu=str(gpu)) for seed in (0, 1)
            for gpu in (0, 1)
        ])
    monkeypatch.setattr(
        launch, 'run_stream',
        lambda _s, row, _c: seen[row['stream']].append(row['seed']))
    monkeypatch.setattr(launch, 'collect_results', lambda _p: seen)
    assert launch.execute(plan) == {0: [0, 1], 1: [1, 0]}

    def fail(_s, row, cancel):
        if row['stream'] == 0:
            raise ValueError('failed before launch')
        assert not cancel.is_set()
        seen[row['stream']].append(row['seed'])

    monkeypatch.setattr(launch, 'run_stream', fail)
    plan['directory'] = str(tmp_path)
    with pytest.raises(RuntimeError, match='healthy streams completed'):
        launch.execute(plan)


@pytest.mark.parametrize('devices', [['0'], ['0', '1', '2']])
def test_physical_gpu_count_does_not_merge_logical_streams(
        launch, config, tmp_path, monkeypatch, devices):
    plan = launch.materialize(
        launch.load_process_plan(config, [0, 1], devices,
                                 list(range(len(devices)))), tmp_path / 'plan')
    assert len(plan['streams']) == 12
    for row in plan['streams']:
        assert row['gpu'] == devices[row['stream'] % len(devices)]
    active, visited, lock = set(), [], threading.Lock()

    def run(_s, row, _c):
        with lock:
            assert row['gpu'] not in active
            active.add(row['gpu'])
            visited.append((row['stream'], row['seed']))
            active.remove(row['gpu'])

    monkeypatch.setattr(launch, 'run_stream', run)
    monkeypatch.setattr(launch, 'collect_results', lambda _p: visited)
    launch.execute(plan)
    assert len(set(visited)) == 12
    for index in range(6):
        assert [s for i, s in visited
                if i == index] == ([1, 0] if index % 2 else [0, 1])


def write_evidence(directory):
    directory.mkdir(parents=True, exist_ok=True)
    local = directory / 'simulation/run'
    remote = directory / 'model-results/run'
    local.mkdir(parents=True)
    remote.mkdir(parents=True)
    record = dict(
        task_id='task',
        episode_index=0,
        episode_id='case0',
        policy_seed=1200,
        environment_seed=1200,
        success=True,
        score=50.,
        completed_episodes=1,
        model_calls=1,
        prediction_metadata=[
            dict(
                sampling_mode='process_noise_stream_v1',
                sequence=1,
                episode_id='case0',
                policy_seed=1200,
                process_instance_id='model0',
                process_sequence=1)
        ])
    bodies = [
        dict(total_episodes=1, tasks=[dict(task_id='task')]), record,
        dict(
            status='finished',
            total_episodes=1,
            completed_episodes=1,
            metrics=dict(overall=dict(episodes=1, successes=1)))
    ]
    events = [
        dict(
            event_type=kind,
            payload=body,
            sequence=i,
            run_session_id='run',
            run_dir=str(remote)) for i, (kind, body) in enumerate(
                zip(('run_start', 'episode_end', 'run_end'), bodies), 1)
    ]
    for root in (local, remote):
        (root / 'events.jsonl').write_text(''.join(
            json.dumps(e) + '\n' for e in events))
    (directory / 'result-pointer.json').write_text(
        json.dumps(dict(directory=str(local))))
    (local / 'result.json').write_text(
        json.dumps(
            dict(
                complete=True,
                completed=1,
                total=1,
                successes=1,
                simulator_exit_code=0,
                remote_result_dir=str(remote))))
    settings = dict(
        process_seeds=[0],
        batch_size=1,
        report_base_seed=1200,
        manifest=dict(canonical_tasks=['task']))
    sim = directory / 'sim.py'
    sim.write_text(
        "themis = dict(evaluation=dict(environment=dict(output_dir=" +
        repr(str(local / 'native')) + ")))")
    row = dict(
        simulator_config=str(sim),
        seed=0,
        stream=0,
        tasks=['task'],
        episodes=1,
        directory=str(directory),
        config_sha256={},
        gpu='0',
        simulator_cpus=[0],
        cache=str(directory / 'cache'))
    return settings, row, events, local, remote


def test_dual_journals_and_process_chain_validation(launch, tmp_path):
    from evaluation.progress import stream_results
    settings, row, events, local, remote = write_evidence(tmp_path / 'stream')
    assert stream_results(settings, row)[1] == 'model0'
    for mutation in ('chain', 'score', 'seed', 'identity', 'model_journal'):
        changed = copy.deepcopy(events)
        record = changed[1]['payload']
        if mutation == 'chain':
            record['prediction_metadata'][0]['process_sequence'] = 2
        if mutation == 'score':
            record['score'] = None
        if mutation == 'seed':
            record['environment_seed'] = 999
        if mutation == 'identity':
            record['episode_index'] = 1
        if mutation == 'model_journal':
            record['score'] = 75.
        for root in (local, remote):
            content = (
                events
                if mutation == 'model_journal' and root == local else changed)
            (root / 'events.jsonl').write_text(''.join(
                json.dumps(e) + '\n' for e in content))
        with pytest.raises(ValueError):
            stream_results(settings, row)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process ownership')
def test_cpu_child_lifecycle_and_failure_cleanup(launch, tmp_path,
                                                 monkeypatch):
    settings, row, *_ = write_evidence(tmp_path / 'success')
    settings.update(
        fluxvla_root=str(tmp_path),
        robolab_root=str(tmp_path),
        simulator_python=sys.executable,
        model_env={},
        simulator_env={},
        stream_timeout_s=10)
    done = tmp_path / 'done'
    row['model_command'] = [
        sys.executable, '-c', "import time; from pathlib import Path\n"
        f"while not Path({str(done)!r}).exists(): time.sleep(.02)"
    ]
    row['simulator_command'] = [
        sys.executable, '-c',
        f"from pathlib import Path; Path({str(done)!r}).touch()"
    ]
    monkeypatch.setenv('PYTHONPATH', 'private-hook')
    env = launch.child_environment(settings, row, 'simulator')
    assert 'private-hook' not in env['PYTHONPATH']
    launch.run_stream(settings, row, threading.Event())
    assert json.loads((Path(row['directory']) /
                       'execution.json').read_text())['status'] == 'complete'
    settings2, row2, *_ = write_evidence(tmp_path / 'failure')
    row2['model_command'] = [sys.executable, '-c', 'raise SystemExit(2)']
    row2['simulator_command'] = [
        sys.executable, '-c', 'import time; time.sleep(60)'
    ]
    cancel = threading.Event()
    with pytest.raises(RuntimeError, match='Model exited'):
        launch.run_stream(settings, row2, cancel)
    state = json.loads(
        (Path(row2['directory']) / 'execution.json').read_text())
    assert state['status'] == 'failed' and not cancel.is_set()
    snapshot = launch.ProcessTree.snapshot()
    assert all(
        p['pid'] not in snapshot or snapshot[p['pid']][2] != p['start_ticks']
        for p in state['processes'].values())


def test_public_defaults_discover_tasks_without_private_manifest(
        launch, config, tmp_path):
    values = yaml.safe_load(config.read_text())
    values.update(streams=None, gpu_ids=[0], render_devices=[0])
    tasks = Path(values['robolab_root']) / 'robolab/tasks/benchmark'
    tasks.mkdir(parents=True)
    for name in ['CTask', 'ATask', 'BTask']:
        (tasks / (name + '.py')).write_text(f'class {name}(Task): pass\n')
    config.write_text(yaml.safe_dump(values))
    plan = launch.load_process_plan(config)
    assert plan['manifest']['canonical_tasks'] == ['ATask', 'BTask', 'CTask']
    assert len(plan['manifest']['streams']) == 1
    divided = launch.load_process_plan(config, num_streams=2)
    assert [s['tasks']
            for s in divided['manifest']['streams']] == [['ATask', 'CTask'],
                                                         ['BTask']]
    small = launch.load_process_plan(config, tasks=['CTask'])
    assert small['manifest']['canonical_tasks'] == ['CTask']
    public = yaml.safe_load(
        (ROOT / 'configs/robolab/process.yaml').read_text())
    assert public['streams'] is None and public['num_streams'] == 1
    assert public['gpu_ids'] == [0] and public['process_seeds'] == [0]


def test_model_environment_resolves_assets_without_leaking(
        launch, config, tmp_path, monkeypatch):
    values = yaml.safe_load(config.read_text())
    recipe = Path(values['model_config']).with_name('model_base.py')
    recipe.write_text(recipe.read_text() +
                      "\nasset_marker = os.environ.get('MODEL_TEST_ASSET')\n")
    values['model_env'] = {'MODEL_TEST_ASSET': '/model/assets'}
    config.write_text(yaml.safe_dump(values))
    monkeypatch.setenv('MODEL_TEST_ASSET', '/caller/assets')
    plan = launch.materialize(
        launch.load_process_plan(config), tmp_path / 'asset-plan')
    actual = Config.fromfile(plan['streams'][0]['model_config'])
    assert actual.asset_marker == '/model/assets'
    assert os.environ['MODEL_TEST_ASSET'] == '/caller/assets'


def test_public_process_config_uses_native_batch_and_seed_zero():
    config = Config.fromfile(ROOT / 'configs/robolab/droid.py')
    evaluation = config.themis.evaluation
    assert evaluation.policy_client.sampling_mode == 'process_noise_stream_v1'
    assert evaluation.environment.num_envs == 10
    assert evaluation.episode_batch_size == evaluation.episodes_per_task == 10
    assert evaluation.environment.simulator_seed == evaluation.base_seed == 0
    assert evaluation.execute_horizon is None
    assert evaluation.stop_on_success is True


@pytest.mark.parametrize('feature', ['resume', 'retry'])
def test_process_plan_rejects_development_recovery(launch, config, tmp_path,
                                                   feature):
    data = yaml.safe_load(config.read_text())
    key = 'model_config' if feature == 'resume' else 'simulator_config'
    recipe = Config.fromfile(data[key], lazy_import=False)
    if feature == 'resume':
        recipe.themis.server.runner.process_resume = dict(path='/absent')
    else:
        recipe.themis.evaluation.nonfinite_batch_retry_limit = 1
    path = tmp_path / 'unsupported.py'
    recipe.dump(str(path))
    data[key] = str(path)
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='unsupported'):
        launch.materialize(launch.load_process_plan(config), tmp_path / 'plan')


def test_continuous_request_chain_rejects_gaps_and_changed_identity():
    import copy

    from evaluation.progress import validate_process_chain
    metadata = dict(
        sampling_mode='process_noise_stream_v1',
        sequence=1,
        episode_id='e',
        policy_seed=0,
        process_instance_id='p',
        process_sequence=1)
    record = dict(
        task_id='task',
        episode_id='e',
        policy_seed=0,
        model_calls=2,
        prediction_metadata=[
            metadata, dict(metadata, sequence=2, process_sequence=2)
        ])
    assert validate_process_chain([record]) == 'p'
    for field, value in [('process_sequence', 1), ('process_sequence', 3),
                         ('process_instance_id', 'other'), ('episode_id', 'x'),
                         ('policy_seed', 1)]:
        bad = copy.deepcopy(record)
        bad['prediction_metadata'][1][field] = value
        with pytest.raises(ValueError):
            validate_process_chain([bad])
