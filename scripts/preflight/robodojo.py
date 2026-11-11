"""RoboDojo-specific source, package and generated robot-path checks."""

import argparse
import ast
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def prepare(root, *, check=False):
    """Check camera interfaces and apply the remaining PhysX reset fix."""
    root = Path(root).resolve(strict=True)
    interfaces = {
        'env/camera_manager/capture/render_sync.py':
        {'add_zero_delay_kit_args', 'zero_delay_active'},
        'env/observation_manager/obs_manager.py': {'render_for_capture'},
    }
    for name, required in interfaces.items():
        source = root / name
        defined = ({
            node.name
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.FunctionDef)
        } if source.is_file() else set())
        if not required <= defined:
            raise ValueError('Missing RoboDojo camera synchronization '
                             'interfaces; use the source selected by '
                             'scripts/install_robodojo.sh')
    patch = (
        Path(__file__).resolve().parents[1] / 'patches' /
        'robodojo-spring-button-reset.patch')
    # Restored archives may sit inside another repository's work_dirs.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(root.parent))

    def apply(*args):
        return subprocess.run(
            ['git', 'apply', *args, str(patch)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True)

    if apply('--reverse', '--check').returncode == 0:
        return {patch.name: 'already_applied'}
    result = apply('--check')
    if result.returncode:
        raise ValueError('Incompatible or partial RoboDojo reset patch:\n' +
                         result.stderr)
    if check:
        raise ValueError('Missing RoboDojo PhysX reset fix; run '
                         'python scripts/preflight/robodojo.py '
                         f'--root "{root}" before starting workers')
    result = apply()
    if result.returncode:
        raise RuntimeError(result.stderr)
    return {patch.name: 'applied'}


def check_environment(root):
    """Inspect packages, source and asset paths without starting Isaac Sim."""
    root = Path(root).resolve(strict=True)
    patches = prepare(root, check=True)
    for module in ('zmq', 'msgpack', 'mmengine', 'gymnasium', 'fluxthemis'):
        importlib.import_module(module)
    packages = {
        name: importlib.metadata.version(name)
        for name in ('torch', 'isaacsim', 'isaaclab', 'numpy', 'mmengine',
                     'pyzmq', 'msgpack', 'gymnasium')
    }
    required = ('env/global_configs.py', 'env_cfg/arx_x5.yml',
                'src/eval_client/eval_env.py',
                'src/eval_client/physx_warning_monitor.py',
                'task/RoboDojo/task_registry.py', 'Assets/Robots',
                'Assets/Object', 'Assets/Material', 'Assets/Eval_Layout')
    missing = [
        str(root / name) for name in required if not (root / name).exists()
    ]
    if missing:
        raise ValueError('Missing RoboDojo files: ' + ', '.join(missing))
    templates = list((root / 'Assets/Robots').rglob('*_tmp.yml'))
    if not templates:
        raise ValueError('No robot path templates in Assets/Robots')
    for template in templates:
        expected = template.read_text().replace('${ASSETS_PATH}', str(root))
        expected = expected.replace('$ASSETS_PATH', str(root))
        target = template.with_name(template.name.replace('_tmp.yml', '.yml'))
        if not target.is_file() or target.read_text() != expected:
            raise ValueError(f'Robot paths need preparation: {target}; run '
                             'utils/update_embodiment_config_path.py in '
                             f'{root} before evaluation')
    return dict(
        status='ok',
        python=sys.version,
        packages=packages,
        patches=patches,
        coverage='Python and simulator package versions; '
        'camera interfaces and PhysX reset patch; not the full repository, '
        'CUDA driver '
        'or installed binary contents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=prepare.__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check source requirements without modifying it')
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, check=args.check), indent=2))
