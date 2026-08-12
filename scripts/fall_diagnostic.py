"""Run a deterministic fall-rate/survival diagnostic on a trained locomotion policy checkpoint.

Rolls out N episodes in parallel (each with an independently sampled command
via the env's own reset()), and reports per-episode survival length and
overall fall rate. Useful for judging policy robustness beyond the scalar
eval/episode_reward curve, which can look fine even when a policy has failed
to generalize across the command space (e.g. forward-only competence).

Works against any registered locomotion env exposing the standard
info["command"] / done interface (verified against both H1JoystickGaitTracking
and Go1JoystickRoughTerrain).

Usage:
  uv run python scripts/fall_diagnostic.py \
      --env_name Go1JoystickRoughTerrain \
      --checkpoint_path logs/Go1JoystickRoughTerrain-20260801-112323/checkpoints/000206438400
"""
import argparse
import functools
import tempfile

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--env_name", default="H1JoystickGaitTracking")
  parser.add_argument("--checkpoint_path", required=True)
  parser.add_argument("--num_episodes", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  env_cfg = registry.get_default_config(args.env_name)
  ppo_params = locomotion_params.brax_ppo_config(args.env_name)
  network_factory = (
      functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
      if "network_factory" in ppo_params
      else ppo_networks.make_ppo_networks
  )

  env = registry.load(args.env_name, config=env_cfg)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]
  num_eval_envs = training_params.pop("num_eval_envs", 128)
  training_params["num_timesteps"] = 0  # play-only: restore weights, don't train

  make_inference_fn, params, _ = ppo.train(
      **training_params,
      network_factory=network_factory,
      seed=args.seed,
      restore_checkpoint_path=args.checkpoint_path,
      save_checkpoint_path=tempfile.mkdtemp(),
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      environment=env,
      eval_env=env,
      progress_fn=lambda *a: None,
      policy_params_fn=lambda *a: None,
  )

  print("Restored OK")

  jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

  wrapped_env = wrapper.wrap_for_brax_training(
      env, episode_length=ppo_params.episode_length, action_repeat=ppo_params.action_repeat
  )

  rng = jax.random.split(jax.random.PRNGKey(args.seed), args.num_episodes)
  reset_states = jax.jit(wrapped_env.reset)(rng)

  def step_fn(carry, _):
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act_keys = jax.random.split(act_key, args.num_episodes)
    act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
    state = wrapped_env.step(state, act)
    return (state, rng), state.done

  (final_state, _), done_traj = jax.lax.scan(
      step_fn, (reset_states, jax.random.PRNGKey(args.seed + 1)), None,
      length=ppo_params.episode_length,
  )

  done_np = np.array(done_traj)
  commands_np = np.array(reset_states.info["command"])

  survival_steps = []
  for lane in range(args.num_episodes):
    fell_at = np.nonzero(done_np[:, lane])[0]
    survival_steps.append(
        int(fell_at[0]) + 1 if len(fell_at) else ppo_params.episode_length
    )

  print(f"\n{'lane':>4}  {'cmd_vx':>7}  {'survived':>12}  fell")
  for lane in range(args.num_episodes):
    vx = commands_np[lane, 0]
    survived = survival_steps[lane]
    fell = survived < ppo_params.episode_length
    print(f"{lane:>4}  {vx:>+7.2f}  {survived:>5}/{ppo_params.episode_length}  {fell}")

  fall_rate = np.mean([s < ppo_params.episode_length for s in survival_steps])
  avg_survival_pct = np.mean(survival_steps) / ppo_params.episode_length
  
  print(f"\noverall fall rate: {fall_rate*100:.1f}%   avg survival: {avg_survival_pct*100:.1f}%")


if __name__ == "__main__":
  main()