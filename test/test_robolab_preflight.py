"""Static RoboLab input checks use small source/asset fixtures."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def preflight(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('preflight.robolab')


def write_file(root, name, text='asset'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def checkout(tmp_path, preflight, monkeypatch):
    monkeypatch.setattr(preflight.importlib.metadata, 'version',
                        lambda _: '2.2.0')
    for name in preflight.REQUIRED_SOURCES:
        write_file(tmp_path, name, '# source\n')
    for name in [
            'assets/robots/franka_robotiq_2f_85_flattened.usd',
            'assets/fixtures/franka_table.usd',
            'assets/backgrounds/default/home_office.exr',
            'assets/objects/object_catalog.json'
    ]:
        write_file(tmp_path, name)
    write_file(tmp_path, 'assets/scenes/test.usda',
               '#usda 1.0\n def Xform "test" {}\n')
    for task in ['BananaInBowlTask', 'MustardInRightBinTask']:
        write_file(
            tmp_path, f'robolab/tasks/benchmark/{task}.py',
            f'class {task}(Task):\n'
            '    scene = import_scene("test.usda")\n')
    return tmp_path


def test_discover_real_source_without_importing(preflight, checkout):
    # Executing these fixtures would raise NameError (no Task/Isaac imports).
    result = preflight.check_environment(checkout, 'all')
    assert result['status'] == 'ok'
    assert set(result['task_sources']) == {
        'BananaInBowlTask', 'MustardInRightBinTask'
    }
    result = preflight.check_environment(
        checkout, ['MustardInRightBinTask', 'BananaInBowlTask'])
    assert list(result['task_sources']) == [
        'MustardInRightBinTask', 'BananaInBowlTask'
    ]


def test_unknown_or_invalid_tasks_rejected(preflight, checkout):
    selections = ['MissingTask', [], ['all', 'BananaInBowlTask']]
    selections.append(['BananaInBowlTask'] * 2)
    for selection in selections:
        with pytest.raises(ValueError, match='task'):
            preflight.check_environment(checkout, selection)


@pytest.mark.parametrize('name', [
    'robolab/core/environments/runtime.py',
    'assets/robots/franka_robotiq_2f_85_flattened.usd',
    'assets/fixtures/franka_table.usd',
    'assets/backgrounds/default/home_office.exr',
    'assets/scenes/test.usda',
    'assets/objects/object_catalog.json',
])
def test_missing_required_input_rejected(preflight, checkout, name):
    (checkout / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        preflight.check_environment(checkout, 'BananaInBowlTask')


def test_empty_asset_and_missing_scene_reference(preflight, checkout):
    asset = checkout / 'assets/fixtures/franka_table.usd'
    asset.write_text('')
    with pytest.raises(ValueError, match='Empty'):
        preflight.check_environment(checkout)
    asset.write_text('asset')
    (checkout / 'assets/scenes/test.usda').write_text(
        '#usda 1.0\n def Xform "x" '
        '(references = @../objects/missing.usd@) {}\n')
    with pytest.raises(FileNotFoundError, match='missing.usd'):
        preflight.check_environment(checkout)


def test_missing_simulator_dependency_rejected(preflight, checkout,
                                               monkeypatch):

    def missing(name):
        if name == 'isaacsim':
            raise importlib.metadata.PackageNotFoundError(name)
        return '2.2.0'

    monkeypatch.setattr(preflight.importlib.metadata, 'version', missing)
    with pytest.raises(
            importlib.metadata.PackageNotFoundError, match='isaacsim'):
        preflight.check_environment(checkout)


def test_lfs_pointer_is_not_an_installed_asset(preflight, checkout):
    (checkout / 'assets/fixtures/franka_table.usd').write_text(
        'version https://git-lfs.github.com/spec/v1\n'
        'oid sha256:example\nsize 123\n')
    with pytest.raises(ValueError, match='Unmaterialized.*LFS'):
        preflight.check_environment(checkout)


def test_removed_usd_payload_is_not_a_required_asset(preflight, checkout):
    (checkout / 'assets/scenes/test.usda').write_text(
        '#usda 1.0\n def Xform "bin" (\n'
        '    delete payload = @../fixtures/old.usd@\n'
        '    reorder references = [\n        @missing.usd@\n    ]\n'
        '    prepend payload = @https://example.com/bin.usd@\n) {}\n')
    result = preflight.check_environment(checkout)
    assert result['unchecked_assets'] == ['https://example.com/bin.usd']


def test_dynamic_scene_cannot_silently_pass(preflight, checkout):
    write_file(
        checkout, 'robolab/tasks/benchmark/BananaInBowlTask.py',
        'class BananaInBowlTask(Task):\n'
        '    scene = import_scene(select_scene())\n')
    with pytest.raises(ValueError, match='literal scene'):
        preflight.check_environment(checkout, 'BananaInBowlTask')


def test_environment_dispatch_uses_robolab_checker(preflight, checkout,
                                                   monkeypatch):
    from mmengine import Config
    environment = importlib.import_module('preflight.environment')
    monkeypatch.delenv('ROBOLAB_ROOT', raising=False)
    path = checkout / 'config.py'
    Config(
        dict(
            themis=dict(
                evaluation=dict(
                    environment=dict(
                        type='RoboLabEnvironment',
                        robolab_root=str(checkout),
                        task_name='MustardInRightBinTask'))))).dump(str(path))
    result = environment.check_environment(str(path))
    assert list(result['task_sources']) == ['MustardInRightBinTask']
    monkeypatch.setenv('ROBOLAB_ROOT', str(checkout / 'missing'))
    with pytest.raises(FileNotFoundError):
        environment.check_environment(str(path))
