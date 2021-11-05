import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    try:
        image_raw_topic_name = os.environ['MOBILE_ROBOT_HL_IMAGE_RAW_TOPIC']
    except:
        image_raw_topic_name = "image_raw/uncompressed"

    config = os.path.join(
        get_package_share_directory('mobile_robot_hl'),
        'config',
        'config.yaml'
        )
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'supervisor_node'
        ),
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'agent_node',
            parameters= [config]
        ),
        launch_ros.actions.Node(
            package= 'image_transport',
            executable= 'republish',
            arguments=[
                ('compressed'),
                ],
            remappings=[
                    ('in/compressed','image_raw/compressed'),
                    ('out',image_raw_topic_name),
                ],
        ),
    ])
