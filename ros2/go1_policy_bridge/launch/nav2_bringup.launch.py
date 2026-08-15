import os

from launch import LaunchDescription
from launch_ros.actions import Node

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_launch_description():
  params_file = os.path.join(_PKG_DIR, "config", "nav2_params.yaml")
  map_yaml_file = os.path.join(_PKG_DIR, "config", "maps", "go1_rough_bounds.yaml")

  return LaunchDescription([
      Node(
          package="nav2_controller",
          executable="controller_server",
          name="controller_server",
          output="screen",
          parameters=[params_file],
      ),
      Node(
          package="nav2_planner",
          executable="planner_server",
          name="planner_server",
          output="screen",
          parameters=[params_file],
      ),
      Node(
          package="nav2_bt_navigator",
          executable="bt_navigator",
          name="bt_navigator",
          output="screen",
          parameters=[params_file],
      ),
      Node(
          package="nav2_lifecycle_manager",
          executable="lifecycle_manager",
          name="lifecycle_manager_navigation",
          output="screen",
          parameters=[params_file],
      ),
      Node(
          package="rviz2",
          executable="rviz2",
          name="rviz2",
          output="screen",
      ),
      Node(
          package="nav2_behaviors",
          executable="behavior_server",
          name="behavior_server",
          output="screen",
          parameters=[params_file],
      ),
      Node(
          package="nav2_map_server",
          executable="map_server",
          name="map_server",
          output="screen",
          parameters=[params_file, {"yaml_filename": map_yaml_file}],
      ),
  ])