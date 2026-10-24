"""Jetson: USB camera (v4l2_camera) + vision_node.
    ros2 launch edgehand jetson.launch.py               (video_device:=/dev/video0 show:=true)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("show", default_value="false"),
        Node(package="v4l2_camera", executable="v4l2_camera_node", output="screen",
             parameters=[{"video_device": LaunchConfiguration("video_device"), "image_size": [1280, 720],
                          "output_encoding": "rgb8"}]),
        Node(package="edgehand", executable="vision_node", output="screen",
             parameters=[{"show": ParameterValue(LaunchConfiguration("show"), value_type=bool)}]),
    ])
