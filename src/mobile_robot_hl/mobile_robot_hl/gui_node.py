import rclpy
from rclpy.qos import *
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import StringTrigger, FloatTrigger
from custom_interfaces.msg import AgentOutput
from std_srvs.srv import Trigger
from std_msgs.msg import Bool, String, Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3

from threading import Thread
import numpy as np
import os
from PIL import Image as PImage
import copy

from mobile_robot_hl.episode_data import *
from mobile_robot_hl.gui import *
from mobile_robot_hl.model import *

import ros2_numpy as rnp

class GUINode(Node):

    def __init__(self):
        super().__init__('gui')

        demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']
        model_path = os.environ['MOBILE_ROBOT_HL_MODEL_PATH']
        
        try:
            desired_velocity_topic_name = os.environ['MOBILE_ROBOT_HL_DESIRED_VELOCITY_TOPIC']
        except:
            desired_velocity_topic_name = "desired_velocity"
        try:
            image_raw_topic_name = os.environ['MOBILE_ROBOT_HL_IMAGE_RAW_TOPIC']
        except:
            image_raw_topic_name = "image_raw/uncompressed"

        self.demo_handler = DemoHandler(path=demo_path)
        self.task_handler = TaskHandler(path=task_path, demo_handler = self.demo_handler)
        self.model_handler = ModelHandler(path=model_path)

        self.variables = GUIVariable()
        self.constants = GUIConstant()

        self.gui = GUI(ros_node = self)

        self.variable_trigger = dict(
            episode = [
                self.gui.display.episode.update_image, 
                self.gui.display.episode.update_plot_full, 
                self.gui.display.episode.update_plot_sel,
                self.gui.display.current.info.update_info
                ],
            episode_type = [
                self.gui.display.current.info.update_info
                ],
            episode_name = [
                self.gui.display.current.info.update_info
                ],
            episode_id = [
                self.gui.display.current.info.update_info
                ],
            model_name = [
                self.gui.display.current.info.update_info
                ],
            model_id = [
                self.gui.display.current.info.update_info
                ],
            image_raw = [
                self.gui.display.current.update_image
                ],
            supervisor_state = [
                self.gui.display.current.info.update_info,
                ],
            demo_names = [
                self.gui.control.demo.update_entry
                ],
            task_names = [
                ],
            ids = [
                self.gui.control.selection.update_id
                ],
            model_names = [
                self.gui.control.model.update_entries_name
            ],
            model_ids = [
                self.gui.control.model.update_entries_id
            ],
            task_queue = [
            ],
            episode_index = [
                self.gui.display.episode.update_image,
                self.gui.display.episode.update_plot_sel
            ],
        )

        self.get_logger().info("Initializing Node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_linear_velocity', None),
                ('max_angular_velocity', None),
            ])

        self.constants.max_linear_vel = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.constants.max_angular_vel = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.get_logger().info(f"Parameter <max_linear_velocity> = {self.constants.max_linear_vel}")
        self.get_logger().info(f"Parameter <max_angular_velocity> = {self.constants.max_angular_vel}")

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.task_image_subscriber = self.create_subscription(Image, 'task_image', self.task_image_callback, best_effort_qos, callback_group=ReentrantCallbackGroup())
        self.desired_velocity_subscriber = self.create_subscription(Twist, desired_velocity_topic_name, self.desired_velocity_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.termination_flag_subscriber = self.create_subscription(Bool, 'termination_flag', self.termination_flag_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.action_controller_subscriber = self.create_subscription(String, 'action_controller', self.action_controller_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.frame_no_subscriber = self.create_subscription(Int32, 'action_controller', self.frame_no_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.agent_output_subscriber = self.create_subscription(AgentOutput, 'agent_output', self.agent_output_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.user_velocity_subscriber = self.create_subscription(Twist, 'user_velocity', self.user_velocity_callback, best_effort_qos, callback_group = ReentrantCallbackGroup())

        self.image_raw_subscriber = self.create_subscription(Image, image_raw_topic_name, self.image_raw_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.supervisor_state_subscriber = self.create_subscription(String, 'supervisor_state', self.supervisor_state_callback, callback_group=ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")
    
    def image_raw_callback(self, img):
        self.variables.image_raw = PImage.fromarray(rnp.numpify(img))
    
    def supervisor_state_callback(self, msg):
        self.variables.supervisor_state = SupervisorState[msg.data]

    def call_service(self, service_name, command=None):
        self.get_logger().info(f'Calling service "{service_name}"')
        if self.services_[service_name].wait_for_service(timeout_sec=0.1) == False:
            self.get_logger().warn(f'{service_name} service not available')
        else:
            if(command == None):
                request = Trigger.Request()
            else:
                if(type(command) == float):
                    request = FloatTrigger.Request()
                    request.command = command
                else:
                    request = StringTrigger.Request()
                    request.command = str(command)
            response = self.services_[service_name].call(request)

            if(response.success == True):
                self.get_logger().info(f'Service successful: "{service_name}"')
            else:
                self.get_logger().warn(f'"{service_name}" service error: {response.message}')
            return response

    def update_episode(self, type, name, id):
        if type == InformationType.DEMO:
            self.variables.episode = self.demo_handler.get(name, id)
        elif type == InformationType.TASK_EPISODE:
            self.variables.episode = self.task_handler.get(name, id)
    
    def update_state_loop(self):
        self.prev_variables = copy.deepcopy(self.variables)
        while True:
            for var_type in GUIVariables:
                if(self.variables[var_type.name] != self.prev_variables[var_type.name]):
                    for trigger in self.variable_trigger[var_type.name]:
                        Thread(target=lambda: trigger()).start()

def spin_thread_(node):
    while True:
        rclpy.spin_once(node)

def main():
    rclpy.init()

    node = GUINode()

    spin_thread = Thread(target=spin_thread_, args=(node,))
    spin_thread.start()
    node.update_state_loop()

if __name__ == '__main__':
    main()


