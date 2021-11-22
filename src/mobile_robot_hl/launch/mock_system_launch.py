import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('mobile_robot_hl'),
        'config',
        'config.yaml'
        )
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'gui_node',
            parameters= [config]
        ),
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'supervisor_node',
            parameters= [config]
        ),
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'agent_node',
            parameters= [config]
        ),
        launch_ros.actions.Node(
            package = 'mobile_robot_hl',
            executable = 'joystick_node',
            parameters=[config]
        ),
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'mock_up_node',
        ),
    ])
