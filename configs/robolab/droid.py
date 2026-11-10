"""RoboLab DROID evaluation with ten native lanes and process sampling.

Use process.yaml with run_robolab_process.py. Model weights, preprocessing,
precision and action horizon are supplied by the companion FluxVLA recipe.
Native environment seed is zero; the launcher sets the model startup seed.
"""

themis = dict(
    evaluation=dict(
        base_seed=0,
        environment=dict(
            headless=True,
            num_envs=10,
            output_dir='work_dirs/robolab',
            robolab_root=None,
            simulator_seed=0,
            task_name='all',
            type='RoboLabEnvironment'),
        episode_batch_size=10,
        episodes_per_task=10,
        evaluator=dict(type='SuccessRateEvaluator'),
        execute_horizon=None,
        max_episode_steps=4500,
        policy_client=dict(
            sampling_mode='process_noise_stream_v1',
            type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        stop_on_success=True,
        strict_worker_assignment=False),
    transport=dict(
        image_keys=[
            'over_shoulder_left_camera',
            'wrist_cam',
        ],
        state_keys=[
            'states',
        ],
        unnorm_key='robolab_droid'))
