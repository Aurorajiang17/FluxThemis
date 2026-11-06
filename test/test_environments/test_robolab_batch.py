import numpy as np
import pytest

from fluxthemis.environments.sim.robolab import RoboLabEnvironment


def test_native_batch_configuration_is_opt_in():
    env = RoboLabEnvironment(task_name='all', num_envs=10)
    assert env.batch_size == 10
    assert RoboLabEnvironment(task_name='all').batch_size == 1
    for invalid in (0, -1, True, 1.5):
        with pytest.raises((ValueError, TypeError)):
            RoboLabEnvironment(task_name='all', num_envs=invalid)


def test_observations_select_the_requested_native_lane():
    env = RoboLabEnvironment(task_name='all', num_envs=3)
    env._instruction = 'Pick'
    raw = dict(
        image_obs={
            key: np.stack([np.full((2, 3, 3), i, np.uint8) for i in range(3)])
            for key in env.camera_keys.values()
        },
        proprio_obs=dict(
            arm_joint_pos=np.arange(21).reshape(3, 7),
            gripper_pos=np.array([[0], [0.5], [1]])))
    obs = env._observation(raw, env_id=2)
    assert obs['wrist_cam'][0, 0, 0] == 2
    np.testing.assert_array_equal(obs['states'],
                                  [14, 15, 16, 17, 18, 19, 20, 1])
    obs['wrist_cam'][:] = 99
    assert raw['image_obs']['wrist_cam'][2, 0, 0, 0] == 2
