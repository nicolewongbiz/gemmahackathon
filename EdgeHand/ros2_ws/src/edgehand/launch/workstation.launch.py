"""Workstation: MoveIt + ros2_kortex driver (one launch, it starts the driver itself) + EdgeHand nodes.
    ros2 launch edgehand workstation.launch.py robot_ip:=192.168.1.10
    ros2 launch edgehand workstation.launch.py robot_ip:=0.0.0.0 use_fake_hardware:=true require_pi:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [DeclareLaunchArgument("robot_ip"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("require_pi", default_value="true"),
            DeclareLaunchArgument("with_moveit", default_value="true")]
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("kinova_gen3_7dof_robotiq_2f_85_moveit_config"), "launch", "robot.launch.py"])),
        launch_arguments={"robot_ip": LaunchConfiguration("robot_ip"),
                          "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
                          "launch_rviz": LaunchConfiguration("launch_rviz")}.items(),
        condition=IfCondition(LaunchConfiguration("with_moveit")))
    return LaunchDescription(args + [
        moveit,
        Node(package="edgehand", executable="intent_node", output="screen"),
        Node(package="edgehand", executable="handover_node", output="screen"),
        Node(package="edgehand", executable="pi_bridge", output="screen",
             parameters=[{"require_pi": ParameterValue(LaunchConfiguration("require_pi"), value_type=bool)}]),
    ])
