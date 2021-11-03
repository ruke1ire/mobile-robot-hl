import launch
import launch.actions
import launch.substitutions
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'supervisor_node'
        ),
        launch_ros.actions.Node(
            package= 'mobile_robot_hl',
            executable= 'agent_node'
        ),
        launch_ros.actions.Node(
            package= 'image_transport',
            executable= 'republish',
            arguments=[
                ('compressed'),
                ],
            remappings=[
                    ('in/compressed','image_raw/compressed'),
                    ('out','image_raw/uncompressed'),
                ],
        ),
    ])
