"""RoboLab DROID environment interface for single-worker development.

Uses the official joint-position registration and default instruction field.
Model weights, preprocessing, precision and action horizon live in FluxVLA.
ROBOLAB_ROOT optionally selects an installed simulator checkout.
This is not the historical 10-environment / 1,200-episode evaluation recipe.
"""

themis = dict(
    transport=dict(
        image_keys=['over_shoulder_left_camera', 'wrist_cam'],
        state_keys=['states'],
        unnorm_key='robolab_droid',
    ),
    evaluation=dict(
        environment=dict(
            type='RoboLabEnvironment',
            task_name='BananaInBowlTask',
            robolab_root=None,
            output_dir='work_dirs/robolab',
            simulator_seed=0,
            num_envs=1,
            headless=True),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        base_seed=0,
        episodes_per_task=2,
        # Current benchmark tasks need at most 4,500 native control steps.
        # The adapter enforces each task's own native outer-loop limit.
        max_episode_steps=4500,
        execute_horizon=None,
        stop_on_success=True,
    ),
)
