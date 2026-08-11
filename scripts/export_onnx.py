"""Export a trained brax PPO locomotion policy checkpoint to ONNX.

Adapted from mujoco_playground/experimental/brax_network_to_onnx.ipynb,
generalized to take env_name/checkpoint/output as CLI args and to verify
the ONNX output matches the source JAX policy before writing.

Usage:
  uv run python scripts/export_onnx.py \
      --env_name Go1JoystickRoughTerrain \
      --checkpoint_path logs/Go1JoystickRoughTerrain-20260801-112323/checkpoints/000206438400 \
      --output_path scripts/onnx/go1_rough_policy.onnx
"""
import argparse
import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import onnxruntime as rt
import jax
import jax.numpy as jp

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training.checkpoint import load

from mujoco_playground.config import locomotion_params
from mujoco_playground import locomotion


class MLP(tf.keras.Model):
  """TF mirror of brax's MLP policy network, for weight transfer."""

  def __init__(self, layer_sizes, activation=tf.nn.swish, mean_std=None):
    super().__init__()
    self.mlp_block = tf.keras.Sequential(name="MLP_0")
    for i, size in enumerate(layer_sizes):
      self.mlp_block.add(
          layers.Dense(
              size,
              activation=activation,
              kernel_initializer="lecun_uniform",
              name=f"hidden_{i}",
          )
      )
    if self.mlp_block.layers:
      self.mlp_block.layers[-1].activation = None
    if mean_std is not None:
      self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
      self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
    else:
      self.mean = None
      self.std = None

  def call(self, inputs):
    if isinstance(inputs, list):
      inputs = inputs[0]
    if self.mean is not None:
      inputs = (inputs - self.mean) / self.std
    logits = self.mlp_block(inputs)
    loc, _ = tf.split(logits, 2, axis=-1)
    return tf.tanh(loc)


def transfer_weights(jax_params, tf_model):
  for layer_name, layer_params in jax_params.items():
    tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
    kernel = np.array(layer_params["kernel"])
    bias = np.array(layer_params["bias"])
    tf_layer.set_weights([kernel, bias])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--env_name", required=True)
  p.add_argument("--checkpoint_path", required=True)
  p.add_argument("--output_path", required=True)
  args = p.parse_args()

  ppo_params = locomotion_params.brax_ppo_config(args.env_name)
  env_cfg = locomotion.get_default_config(args.env_name)
  env = locomotion.load(args.env_name, config=env_cfg)

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory,
      preprocess_observations_fn=running_statistics.normalize,
  )
  obs_size = env.observation_size
  act_size = env.action_size
  ppo_network = network_factory(obs_size, act_size)

  params = load(args.checkpoint_path)
  params = (params[0], params[1])

  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
  inference_fn = make_inference_fn(params, deterministic=True)

  mean = params[0].mean["state"]
  std = params[0].std["state"]
  mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

  tf_policy = MLP(
      layer_sizes=list(ppo_params.network_factory.policy_hidden_layer_sizes)
      + [act_size * 2],
      mean_std=mean_std,
  )
  state_obs_size = obs_size["state"][0]
  tf_policy(tf.zeros((1, state_obs_size)))  # build
  transfer_weights(params[1]["params"], tf_policy)

  spec = [tf.TensorSpec(shape=(1, state_obs_size), dtype=tf.float32, name="obs")]
  tf_policy.output_names = ["continuous_actions"]
  tf2onnx.convert.from_keras(
      tf_policy, input_signature=spec, opset=11, output_path=args.output_path
  )

  # Verify ONNX output matches the source JAX policy on a random obs.
  rng = np.random.RandomState(0)
  test_obs_np = rng.randn(1, state_obs_size).astype(np.float32)
  sess = rt.InferenceSession(args.output_path, providers=["CPUExecutionProvider"])
  onnx_pred = sess.run(["continuous_actions"], {"obs": test_obs_np})[0][0]

  test_obs_jax = {
      "state": jp.array(test_obs_np[0]),
      "privileged_state": jp.zeros(obs_size["privileged_state"]),
  }
  jax_pred, _ = inference_fn(test_obs_jax, jax.random.PRNGKey(0))

  max_diff = float(np.max(np.abs(onnx_pred - np.array(jax_pred))))
  print(f"Exported: {args.output_path}")
  print(f"Max |onnx - jax| action diff on random obs: {max_diff:.6f}")
  if max_diff > 1e-3:
    raise RuntimeError(
        f"ONNX export diverges from JAX policy (max diff {max_diff}), do not deploy this."
    )
  print("Verification passed (diff < 1e-3).")


if __name__ == "__main__":
  main()
