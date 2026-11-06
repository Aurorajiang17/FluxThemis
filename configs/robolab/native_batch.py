"""Ten native environments per worker and ten episodes per benchmark task.

Use dynamic scheduling, simulator_restart_limit=0 and no video recording.
Explicit recovery reuses complete batches; unfinished batches reset with their
original seeds. Live simulator state and automatic worker restart remain
unsupported.
Model workers still serve one observation per request; choose their count and
GPU allocation in the deployment configuration after resource qualification.
"""

_base_ = './droid.py'

themis = dict(
    evaluation=dict(
        environment=dict(num_envs=10),
        episode_batch_size=10,
        episodes_per_task=10,
        strict_worker_assignment=False))
