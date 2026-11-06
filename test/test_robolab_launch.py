"""CPU launch/preflight checks; no model or simulator is started."""

import importlib
import json
from pathlib import Path

import pytest
import yaml
from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('ROBOLAB_ROOT', raising=False)
    return importlib.import_module('run_evaluation')


def write_launch(tmp_path, **simulator):
    path = tmp_path / 'launch.yaml'
    data = dict(
        simulator=dict(
            config=str(ROOT / 'configs/robolab/droid.py'),
            gpu_ids=[0],
            **simulator),
        run=dict(output_root=str(tmp_path / 'outputs')))
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize('task', ['BananaInBowlTask', 'MustardInRightBinTask'])
def test_yaml_task_selects_registration_and_quota(launch, tmp_path, task):
    settings = launch.load_config(
        write_launch(tmp_path, task=task, episodes=2))
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == task
    assert evaluation.task_ids is None
    assert evaluation.episodes_per_task == 2
    assert evaluation.episodes_per_task_overrides == {task: 2}


def test_full_restores_official_cli_scope(launch, tmp_path):
    settings = launch.load_config(
        write_launch(tmp_path, task='BananaInBowlTask', episodes=2),
        mode='full')
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == 'all'
    assert evaluation.task_ids is None
    assert evaluation.episodes_per_task == 1
    assert not evaluation.get('episodes_per_task_overrides')
    assert evaluation.environment.num_envs == 1
    assert evaluation.environment.simulator_seed == 0


def test_smoke_scales_to_workers(launch, tmp_path):
    settings = launch.load_config(
        write_launch(tmp_path, task='MustardInRightBinTask', episodes=9),
        mode='smoke',
        overrides=dict(gpu_ids='6', workers_per_gpu=2))
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == 'BananaInBowlTask'
    assert evaluation.task_ids is None
    assert evaluation.episodes_per_task_overrides == {'BananaInBowlTask': 2}


def test_python_task_list_preserves_order(launch):
    tasks = ['MustardInRightBinTask', 'BananaInBowlTask']
    settings = launch.load_config(
        ROOT / 'configs/robolab/droid.py',
        overrides=dict(
            cfg_options={
                'themis.evaluation.environment.task_name': tasks,
                'themis.evaluation.episodes_per_task': 3,
            }))
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == tasks
    assert evaluation.episodes_per_task == 3


def test_native_batch_launch_validates_before_simulation(launch, tmp_path):
    path = write_launch(tmp_path, simulator_restart_limit=0)
    options = {
        'themis.evaluation.episode_batch_size': 10,
        'themis.evaluation.episodes_per_task': 10,
        'themis.evaluation.environment.num_envs': 10
    }
    settings = launch.load_config(path, overrides=dict(cfg_options=options))
    assert settings['config'].themis.evaluation.episode_batch_size == 10
    smoke = launch.load_config(
        path,
        mode='smoke',
        overrides=dict(cfg_options=options, workers_per_gpu=2))
    assert smoke['config'].themis.evaluation.episodes_per_task == 20
    for key, value in [('themis.evaluation.episodes_per_task', 9),
                       ('themis.evaluation.environment.num_envs', 1)]:
        with pytest.raises(ValueError):
            launch.load_config(
                path, overrides=dict(cfg_options={
                    **options, key: value
                }))
    with pytest.raises(ValueError, match='restart_limit=0'):
        launch.load_config(
            write_launch(tmp_path), overrides=dict(cfg_options=options))


def test_root_precedence_and_child_environment(launch, tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOLAB_ROOT', str(tmp_path / 'fallback'))
    monkeypatch.setenv('ROBO_TEST_ROOT', str(tmp_path / 'installed'))
    config = write_launch(tmp_path, robolab_root='${ROBO_TEST_ROOT}/../actual')
    settings = launch.load_config(config)
    root = str(tmp_path / 'actual')
    assert settings['local']['robolab_root'] == root
    assert settings[
        'config'].themis.evaluation.environment.robolab_root == root
    runtime = launch.ProcessRunner(settings)
    assert runtime.env['ROBOLAB_ROOT'] == root
    assert runtime.python == settings['local']['python']
    settings = launch.load_config(write_launch(tmp_path))
    assert settings['local']['robolab_root'] == str(tmp_path / 'fallback')
    settings = launch.load_config(
        write_launch(tmp_path),
        overrides=dict(
            cfg_options={
                'themis.evaluation.environment.robolab_root': '../python_root'
            }))
    assert settings['local']['robolab_root'] == str(tmp_path.parent /
                                                    'python_root')


@pytest.mark.parametrize('values, message', [
    ({
        'robolab_root': '$UNSET_ROBOLAB_PATH'
    }, 'unset environment'),
    ({
        'episode_timeout_s': 0
    }, 'episode_timeout_s'),
    ({
        'simulator_restart_limit': -1
    }, 'simulator_restart_limit'),
    ({
        'task': ['BananaInBowlTask']
    }, 'simulator.task'),
    ({
        'unknown': True
    }, 'unknown fields'),
])
def test_invalid_launch_fields(launch, tmp_path, values, message):
    with pytest.raises(ValueError, match=message):
        launch.load_config(write_launch(tmp_path, **values))


def test_dry_run_freezes_output_and_launch_limits(launch, tmp_path):
    config = write_launch(
        tmp_path,
        robolab_root='../RoboLab',
        task='MustardInRightBinTask',
        episodes=2,
        episode_timeout_s=300,
        simulator_restart_limit=0)
    assert launch.main(
        ['--config', str(config), '--dry-run', '--gpu-ids', '6']) == 0
    output, = (tmp_path / 'outputs').iterdir()
    plan = json.loads((output / 'plan.json').read_text())
    cfg = Config.fromfile(str(output / 'config_snapshot.py'))
    assert cfg.themis.evaluation.environment.output_dir == str(output /
                                                               'native')
    args = launch.simulator_arguments(plan)
    assert args[args.index('--episode-timeout-s') + 1] == '300'
    assert args[args.index('--simulator-restart-limit') + 1] == '0'
    assert 'themis.evaluation.environment.robolab_root=' + str(
        tmp_path.parent / 'RoboLab') in args
    assert not (output / 'preflight.log').exists()
    assert not (output / 'native').exists()


@pytest.mark.parametrize('benchmark, smoke_task', [
    ('robodojo', 'put_bottles_into_dustbin'),
    ('robotwin', None),
])
def test_existing_remote_defaults_unchanged(launch, benchmark, smoke_task):
    settings = launch.load_config(ROOT / f'configs/{benchmark}/remote.yaml')
    assert settings['local']['gpus'] == '0'
    assert settings['local']['workers'] == 1
    assert settings['local']['episodes'] == 1
    assert settings['local']['episode_timeout_s'] == 3600
    assert settings['local']['simulator_restart_limit'] == 3
    if smoke_task:
        assert settings['local']['task'] == smoke_task


def test_public_robolab_template_is_small_and_portable(launch):
    settings = launch.load_config(ROOT / 'configs/robolab/remote.yaml')
    assert settings['local']['gpus'] == '0'
    assert settings['local']['task'] == 'BananaInBowlTask'
    assert settings['local']['episodes'] == 1
    assert settings['local']['simulator_restart_limit'] == 0


def test_robolab_root_cannot_affect_other_benchmarks(launch, tmp_path):
    path = write_launch(tmp_path, robolab_root='../RoboLab')
    data = yaml.safe_load(path.read_text())
    data['simulator']['config'] = str(ROOT / 'configs/robodojo/arx_x5.py')
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='only to RoboLab'):
        launch.load_config(path)


def test_invalid_interpreter_fails_before_subprocess(launch, tmp_path):
    settings = launch.load_config(
        write_launch(tmp_path, python='./missing-python'))
    with pytest.raises(ValueError, match='Python is not executable'):
        launch.ProcessRunner(settings).connect()


def test_resume_requires_separate_robolab_validation(launch, tmp_path, capsys):
    path = write_launch(tmp_path)
    assert launch.run_config(path, dry_run=True, resume=tmp_path / 'old') == 1
    assert 'RoboLab resume requires separate validation' in capsys.readouterr(
    ).out
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('visible, assigned, expected', [
    ('6', '6', 0),
    ('GPU-example', 'GPU-example', 0),
    ('2,6', '6', 1),
    (None, '6', 6),
])
def test_worker_cuda_mapping(monkeypatch, visible, assigned, expected):
    from fluxthemis.environments.sim.robolab import RoboLabEnvironment
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    env = RoboLabEnvironment(task_name='all', render_device_id=6)
    env.configure_simulator_device(assigned)
    assert env.device_id == expected
    assert env.render_device_id == 6


@pytest.mark.parametrize('visible, assigned', [
    ('6', '0'),
    ('', '6'),
    ('6', 'GPU-other'),
    (None, 'GPU-example'),
    ('6,6', '6'),
    (None, '-1'),
])
def test_worker_rejects_invisible_device(monkeypatch, visible, assigned):
    from fluxthemis.environments.sim.robolab import RoboLabEnvironment
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match='GPU|CUDA_VISIBLE_DEVICES'):
        RoboLabEnvironment(
            task_name='all').configure_simulator_device(assigned)
