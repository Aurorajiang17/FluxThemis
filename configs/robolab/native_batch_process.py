"""Native ten-lane batches sharing a persistent model process RNG stream.

Pair with FluxVLA's process inference recipe and one model worker per endpoint.
Keep task order and worker assignment fixed for reproducible noise allocation.
Episode seed metadata remains available without reseeding the model.
"""

_base_ = './native_batch.py'

themis = dict(
    evaluation=dict(
        policy_client=dict(sampling_mode='process_noise_stream_v1')))
