# FluxVLA + FluxThemis Quickstart

## 1. Install

Create and install exactly one selected simulator environment.

For LIBERO:

```bash
cd /root/projects/FluxThemis
conda create -n fluxthemis-libero python=3.10 -y
bash scripts/install_libero.sh
```

For RoboCasa:

```bash
cd /root/projects/FluxThemis
conda create -n fluxthemis-robocasa python=3.10 -y
bash scripts/install_robocasa.sh
```

For RoboDojo:

```bash
cd /root/projects/FluxThemis
bash scripts/install_robodojo.sh
```

For RoboLab:

```bash
cd /root/projects/FluxThemis
bash scripts/install_robolab.sh
```

This uses RoboLab's locked uv environment and native LFS assets. See
[installation prerequisites and prepared images](robolab.md#install).

Choose the launch path for your benchmark. Sections 3 and 4 below are a
RoboDojo example with separate model and simulator commands. For RoboLab's
default process workflow, use the following entry instead; it starts both
sides locally with their own Python environments.

## 2. Start RoboLab (Process Mode)

Prepare the matching FluxVLA process recipe, converted weights, statistics
and tokenizer described in [required configuration](robolab.md#required-configuration).
The selected FluxVLA version must include the companion process runtime.

```bash
export FLUXVLA_ROOT=/root/projects/FluxVLA
export FLUXVLA_PYTHON=/path/to/model-env/bin/python
export ROBOLAB_ROOT=/root/projects/RoboLab
export FLUXTHEMIS_PYTHON="$ROBOLAB_ROOT/.venv/bin/python"
export FLUXVLA_ROBOLAB_CHECKPOINT=/path/to/model.safetensors
export FLUXVLA_ROBOLAB_STATS=/path/to/dataset_statistics.json
export FLUXVLA_ROBOLAB_TOKENIZER=/path/to/paligemma_tokenizer.model
export ROBOLAB_EVAL_OUTPUT=/path/to/results
export ROBOLAB_EVAL_CACHE=/path/to/cache
cd /root/projects/FluxThemis
"$FLUXTHEMIS_PYTHON" scripts/run_robolab_process.py \
  --tasks BananaInBowlTask --dry-run
```

The default selects startup **seed 0**, one GPU and ten native episodes for
this task. Inspect `configs/robolab/process.yaml` for GPU/Vulkan mapping,
CPU allocation and paths; set `allow_root: true` if running as root. Accept
the simulator license before setting `OMNI_KIT_ACCEPT_EULA=YES`, then remove
`--dry-run` to execute. Removing `--tasks BananaInBowlTask` selects all tasks
(1,200 episodes at the pinned RoboLab revision).

Keep the model and simulator results under `ROBOLAB_EVAL_OUTPUT`; native HDF5
records supply Score and trajectory metrics. See the
[RoboLab guide](robolab.md#progress-and-results) for reports and runtime requirements.

## 3. Start FluxVLA (RoboDojo Example)

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/distributed_zmq_server.sh \
  --config configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --ckpt-path /path/to/checkpoints/model.safetensors
```

The command starts one stateless model Worker per torchrun rank. Rank 0 also
starts the Coordinator.

## 4. Start FluxThemis (RoboDojo Example)

```bash
conda activate RoboDojo
cd /root/projects/FluxThemis
CUDA_VISIBLE_DEVICES=2,3 \
bash scripts/distributed_eval.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --run-name smoke \
  --save-videos \
  --video-dir /persistent/eval/smoke-videos \
  --cfg-options \
    themis.evaluation.environment.task_name=put_bottles_into_dustbin \
    themis.evaluation.episodes_per_task=2
```

The client may be launched first. It waits for server readiness instead of
failing because of startup order.

Each launcher uses the `torchrun` from its active Conda environment and starts
one worker per visible GPU. The default rendezvous ports are `29500` for
FluxVLA and `29501` for FluxThemis.

## 5. Locate Results

In the two-launcher workflow above, FluxVLA prints the authoritative run
directory after `run_start` and again at
`run_end`. The default root is beside the checkpoint. Videos are written to the
evaluation-level `video_dir`.

## 6. Multi-Node

For the two-launcher workflow, use the same shell launchers on each node.
The RoboLab process entry manages streams on one host. See ``Multi-Node Evaluation`` in the
repository README for the required node environment variables and discovery
options.
