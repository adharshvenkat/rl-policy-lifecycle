import time

import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from mujoco_playground._src.locomotion.go1 import go1_constants
from mujoco_playground._src.locomotion.go1.base import get_assets

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster

_MAX_X, _MAX_Y, _MAX_YAW = 1.5, 0.8, 2 * np.pi


class OnnxController:
  """Same policy interface as deploy_go1_keyboard.py's OnnxController."""

  def __init__(self, policy_path, default_angles, n_substeps, get_command_fn,
               action_scale=0.5):
    self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    self._output_names = ["continuous_actions"]
    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._counter = 0
    self._n_substeps = n_substeps
    self._get_command = get_command_fn

  def get_obs(self, model, data):
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    obs = np.hstack([
        linvel, gyro, gravity, joint_angles, joint_velocities,
        self._last_action, self._get_command(),
    ])
    return obs.astype(np.float32)

  def get_control(self, model, data):
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_pred = self._policy.run(self._output_names, {"obs": obs.reshape(1, -1)})[0][0]
      self._last_action = onnx_pred.copy()
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles


class PolicyBridgeNode(Node):

  def __init__(self):
    super().__init__("go1_policy_bridge")

    self.declare_parameter("policy_path", "")
    self.declare_parameter("terrain", "rough")
    policy_path = self.get_parameter("policy_path").value
    terrain = self.get_parameter("terrain").value
    if not policy_path:
      raise ValueError("must set the policy_path parameter")

    self._command = np.zeros(3, dtype=np.float32)
    self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)

    xml_path = (
        go1_constants.FEET_ONLY_ROUGH_TERRAIN_XML
        if terrain == "rough"
        else go1_constants.FEET_ONLY_FLAT_TERRAIN_XML
    )
    self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix(), assets=get_assets())
    self._data = mujoco.MjData(self._model)
    mujoco.mj_resetDataKeyframe(self._model, self._data, 0)

    self._sim_dt = 0.004
    ctrl_dt = 0.02
    n_substeps = int(round(ctrl_dt / self._sim_dt))
    self._model.opt.timestep = self._sim_dt

    self._policy = OnnxController(
        policy_path=policy_path,
        default_angles=np.array(self._model.keyframe("home").qpos[7:]),
        n_substeps=n_substeps,
        get_command_fn=lambda: self._command,
        action_scale=0.5,
    )

    self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
    self._tf_broadcaster = TransformBroadcaster(self)
    self._n_substeps = n_substeps
    self._step_count = 0

    mujoco.set_mjcb_control(self._policy.get_control)

    self.get_logger().info(f"go1_policy_bridge started, policy={policy_path}, terrain={terrain}")

  def _cmd_vel_callback(self, msg):
    self._command[0] = float(np.clip(msg.linear.x, -_MAX_X, _MAX_X))
    self._command[1] = float(np.clip(msg.linear.y, -_MAX_Y, _MAX_Y))
    self._command[2] = float(np.clip(msg.angular.z, -_MAX_YAW, _MAX_YAW))

  def _publish_odom(self):
    pos = self._data.sensor("position").data
    quat = self._data.sensor("orientation").data
    linvel = self._data.sensor("local_linvel").data
    angvel = self._data.sensor("gyro").data
    ros_quat = Quaternion(x=float(quat[1]), y=float(quat[2]), z=float(quat[3]), w=float(quat[0]))
    stamp = self.get_clock().now().to_msg()

    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = "odom"
    odom.child_frame_id = "trunk"
    odom.pose.pose.position.x = float(pos[0])
    odom.pose.pose.position.y = float(pos[1])
    odom.pose.pose.position.z = float(pos[2])
    odom.pose.pose.orientation = ros_quat
    odom.twist.twist.linear.x = float(linvel[0])
    odom.twist.twist.linear.y = float(linvel[1])
    odom.twist.twist.linear.z = float(linvel[2])
    odom.twist.twist.angular.x = float(angvel[0])
    odom.twist.twist.angular.y = float(angvel[1])
    odom.twist.twist.angular.z = float(angvel[2])
    self._odom_pub.publish(odom)

    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = "odom"
    tf.child_frame_id = "trunk"
    tf.transform.translation.x = float(pos[0])
    tf.transform.translation.y = float(pos[1])
    tf.transform.translation.z = float(pos[2])
    tf.transform.rotation = ros_quat
    self._tf_broadcaster.sendTransform(tf)

  def run(self):
    with viewer.launch_passive(self._model, self._data) as v:
      v.cam.trackbodyid = self._model.body("trunk").id
      while v.is_running() and rclpy.ok():
        step_start = time.time()
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        if self._step_count % self._n_substeps == 0:
          self._publish_odom()
        v.sync()
        rclpy.spin_once(self, timeout_sec=0)
        dt_left = self._sim_dt - (time.time() - step_start)
        if dt_left > 0:
          time.sleep(dt_left)


def main():
  rclpy.init()
  node = PolicyBridgeNode()
  try:
    node.run()
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
