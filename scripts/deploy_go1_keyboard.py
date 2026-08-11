"""Deploy an ONNX-exported Go1 policy in an interactive MuJoCo viewer.

Same idea as experimental/sim2sim/play_go1_joystick.py (ONNX policy +
mujoco.set_mjcb_control), but drives the velocity command from the
keyboard instead of a Logitech F710 gamepad, via launch_passive's
key_callback -- no extra HID dependency, no physical controller needed.

Controls (while the viewer window has focus):
  W / S       forward / backward (x velocity)
  A / D       strafe left / right (y velocity)
  Q / E       turn left / right (yaw rate)
  SPACE       zero the command (stop)

Usage:
  uv run python scripts/deploy_go1_keyboard.py --terrain rough
  uv run python scripts/deploy_go1_keyboard.py --terrain flat --policy scripts/onnx/go1_flat_policy.onnx
"""
import argparse
import time

import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.locomotion.go1 import go1_constants
from mujoco_playground._src.locomotion.go1.base import get_assets

_VEL_STEP_X = 0.5
_VEL_STEP_Y = 0.4
_VEL_STEP_YAW = 0.7
_MAX_X, _MAX_Y, _MAX_YAW = 1.5, 0.8, 2 * np.pi


class KeyboardCommand:
  """Shared [x, y, yaw] velocity command, updated via viewer key_callback."""

  def __init__(self):
    self.command = np.zeros(3, dtype=np.float32)

  def key_callback(self, keycode):
    key = chr(keycode) if 0 <= keycode < 256 else ""
    if key == "W":
      self.command[0] = min(self.command[0] + _VEL_STEP_X, _MAX_X)
    elif key == "S":
      self.command[0] = max(self.command[0] - _VEL_STEP_X, -_MAX_X)
    elif key == "A":
      self.command[1] = min(self.command[1] + _VEL_STEP_Y, _MAX_Y)
    elif key == "D":
      self.command[1] = max(self.command[1] - _VEL_STEP_Y, -_MAX_Y)
    elif key == "Q":
      self.command[2] = min(self.command[2] + _VEL_STEP_YAW, _MAX_YAW)
    elif key == "E":
      self.command[2] = max(self.command[2] - _VEL_STEP_YAW, -_MAX_YAW)
    elif keycode == 32:  # SPACE
      self.command[:] = 0.0
    else:
      return
    print(f"command (x,y,yaw) = {self.command.round(2)}")


class OnnxController:
  """ONNX controller for the Go-1 robot (same obs layout as go1/joystick.py)."""

  def __init__(self, policy_path, default_angles, n_substeps, keyboard,
               action_scale=0.5):
    self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    self._output_names = ["continuous_actions"]
    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._counter = 0
    self._n_substeps = n_substeps
    self._keyboard = keyboard

  def get_obs(self, model, data):
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    obs = np.hstack([
        linvel, gyro, gravity, joint_angles, joint_velocities,
        self._last_action, self._keyboard.command,
    ])
    return obs.astype(np.float32)

  def get_control(self, model, data):
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_pred = self._policy.run(self._output_names, {"obs": obs.reshape(1, -1)})[0][0]
      self._last_action = onnx_pred.copy()
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--terrain", choices=["flat", "rough"], default="rough")
  p.add_argument("--policy", default=None, help="Override ONNX policy path")
  args = p.parse_args()

  xml_path = (
      go1_constants.FEET_ONLY_ROUGH_TERRAIN_XML
      if args.terrain == "rough"
      else go1_constants.FEET_ONLY_FLAT_TERRAIN_XML
  )
  policy_path = args.policy or (
      "scripts/onnx/go1_rough_policy.onnx"
      if args.terrain == "rough"
      else "scripts/onnx/go1_flat_policy.onnx"
  )

  model = mujoco.MjModel.from_xml_path(xml_path.as_posix(), assets=get_assets())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02
  sim_dt = 0.004
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  keyboard = KeyboardCommand()
  policy = OnnxController(
      policy_path=policy_path,
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      n_substeps=n_substeps,
      keyboard=keyboard,
      action_scale=0.5,
  )
  mujoco.set_mjcb_control(policy.get_control)

  print("Controls: W/S=fwd/back  A/D=strafe  Q/E=turn  SPACE=stop")
  with viewer.launch_passive(
      model, data, key_callback=keyboard.key_callback
  ) as v:
    v.cam.trackbodyid = model.body("trunk").id if model.body("trunk").id >= 0 else -1
    while v.is_running():
      step_start = time.time()
      mujoco.mj_step(model, data)
      v.sync()
      dt_left = model.opt.timestep - (time.time() - step_start)
      if dt_left > 0:
        time.sleep(dt_left)


if __name__ == "__main__":
  main()
