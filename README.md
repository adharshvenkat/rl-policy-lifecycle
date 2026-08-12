# rl-policy-lifecycle

Scripts covering the training-to-deployment lifecycle for RL locomotion and
manipulation policies trained with [mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
(JAX/MJX, Brax PPO): checkpoint export to ONNX, interactive sim2sim deployment,
and a policy robustness evaluator.

This is not a fork of mujoco_playground. It's a small set of original scripts
that sit on top of it. mujoco_playground and mujoco_menagerie must be
installed separately (see Setup).

## Why the evaluator exists

`eval/episode_reward` from a training run is a scalar average over whatever
commands got sampled that eval pass. It doesn't say anything about *which*
commands the policy handles well. Two checkpoints from the same H1 humanoid
locomotion run illustrate the gap:

| checkpoint | eval reward | fall rate | avg survival |
|---|---|---|---|
| H1 humanoid, 103M steps | 74.4 | 87.5% | 37.8% |
| H1 humanoid, 300M steps | 68.2 (peaked 77.8 @ 68.8M, declined after) | 68.8% | 73.8% |

By reward alone the 103M checkpoint looks equal or better. `fall_diagnostic.py`
rolls out N episodes with independently sampled commands and tracks per-episode
survival against the env's own fall/termination condition, decoupled from the
reward signal entirely. Under that measure the 300M checkpoint is roughly twice
as robust. The reward curve was not wrong, it just wasn't answering the
question that mattered for picking a checkpoint to deploy.

The same script, unmodified, also measures a completely different morphology:

| checkpoint | fall rate | avg survival |
|---|---|---|
| Go1 quadruped, rough terrain | 18.8% | 94.2% |

Go1 is far more robust than either H1 checkpoint on the same command range.
That tracks with the underlying physics (a 4-point stance with a lower center
of mass is structurally easier to balance than a biped), but the point is this
is a measured number, not an assumption from watching a rollout video look
smooth for a few seconds.

## Scripts

**`export_onnx.py`** - loads a brax PPO checkpoint, rebuilds the policy as a
Keras MLP with weights transferred layer-by-layer, converts to ONNX, and
verifies the ONNX output matches the source JAX policy on a random observation
before accepting the export (fails loudly above 1e-3 max abs diff; typical
observed diff is ~1e-6).

```
uv run python scripts/export_onnx.py \
    --env_name Go1JoystickRoughTerrain \
    --checkpoint_path <path-to-checkpoint> \
    --output_path assets/go1_rough_policy.onnx
```

**`deploy_go1_keyboard.py`** - runs an exported ONNX policy interactively in
MuJoCo's passive viewer, with a keyboard-driven velocity command (W/A/S/D
translate, Q/E turn, space to zero) in place of a physical gamepad.

```
uv run python scripts/deploy_go1_keyboard.py --terrain rough
```

<p align="center">
  <img src="assets/go1_rollout.gif" alt="Go1 rough-terrain rollout" width="420">
</p>

**`fall_diagnostic.py`** - restores a checkpoint, rolls out N episodes in
parallel with independently sampled commands, and reports per-episode survival
length against the env's fall/termination condition, plus aggregate fall rate.
Env-agnostic; verified against both H1JoystickGaitTracking (humanoid) and
Go1JoystickRoughTerrain (quadruped).

```
uv run python scripts/fall_diagnostic.py \
    --env_name Go1JoystickRoughTerrain \
    --checkpoint_path <path-to-checkpoint> \
    --num_episodes 32
```

## Setup

Requires a working mujoco_playground install (JAX/MJX, Brax, mujoco_menagerie
assets) - see the [mujoco_playground README](https://github.com/google-deepmind/mujoco_playground)
for install instructions, GPU requirements, and menagerie setup. `export_onnx.py`
additionally needs `tensorflow`, `tf2onnx`, and `onnxruntime`.

Run these scripts from the root of a mujoco_playground checkout, with this
repo's `scripts/` copied in or on `PYTHONPATH`.

## License

Code in this repo is MIT licensed (see `LICENSE`). mujoco_playground is
Apache-2.0 licensed; this repo depends on it but does not redistribute its
source.
