import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.msg import AgentOutput
from custom_interfaces.srv import StringTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from .supervisor_gui import SupervisorGUI

import ros2_numpy as rnp

import threading
import tkinter

class SupervisorNode(Node):

    def __init__(self):
        super().__init__('supervisor')

        self.get_logger().info("Initializing Node")
        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.desired_velocity_publisher = self.create_publisher(Twist, 'desired_velocity', reliable_qos)
        self.termination_flag_publisher = self.create_publisher(Bool, 'termination_flag', reliable_qos)
        self.agent_output_subscriber = self.create_subscription(AgentOutput, 'agent_output', self.agent_output_callback ,best_effort_qos)
        self.agent_input_subscriber = self.create_subscription(Image, 'agent_input', self.agent_input_callback, reliable_qos)
        self.user_velocity_subscriber = self.create_subscription(Twist, 'user_input/velocity', self.user_velocity_callback, best_effort_qos)
        self.user_termination_flag_subscriber = self.create_subscription(Bool, 'user_input/termination_flag', self.user_termination_flag_callback, best_effort_qos)
        self.image_raw_subscriber = self.create_subscription(Image, 'image_raw/uncompressed', self.image_raw_callback ,best_effort_qos)

        agent_prefix = "agent/"
        self.agent_start_client = self.create_client(Trigger, agent_prefix+'start')
        self.agent_pause_client = self.create_client(Trigger, agent_prefix+'pause')
        self.agent_stop_client = self.create_client(Trigger, agent_prefix+'stop')
        self.agent_take_over_client = self.create_client(Trigger, agent_prefix+'take_over')
        self.agent_select_demonstration_client = self.create_client(StringTrigger, agent_prefix+'select_demonstration')
        self.agent_select_model_client = self.create_client(StringTrigger, agent_prefix+'select_model')
        self.agent_select_mode_client = self.create_client(StringTrigger, agent_prefix+'select_mode')

        trainer_prefix='trainer/'
        self.trainer_select_model_client = self.create_client(StringTrigger, trainer_prefix+'select_model')
        self.trainer_start_client = self.create_client(Trigger, trainer_prefix+'start')
        self.trainer_pause_client = self.create_client(Trigger, trainer_prefix+'pause')
        self.trainer_stop_client = self.create_client(Trigger, trainer_prefix+'stop')
        self.trainer_save_client = self.create_client(Trigger, trainer_prefix+'save')
        self.trainer_delete_client = self.create_client(Trigger, trainer_prefix+'delete')
        self.trainer_pre_train_client = self.create_client(Trigger, trainer_prefix+'pre_train')

        self.gui = SupervisorGUI()
        self.get_logger().info("Initialized Node")
        self.image_raw = None
        self.agent_output = {}
        self.agent_input = []
        self.user_output = {}

    def agent_output_callback(self, msg):
        velocity = msg.velocity
        termination_flag = msg.termination_flag
        self.agent_output['velocity'] = velocity
        self.gui.update_current_action_plot(agent_vel={'linear':velocity.linear.x, 'angular': velocity.angular.z})
        self.agent_output['termination_flag'] = termination_flag
        self.get_logger().info(f"got agent_output {self.agent_output}")

    def agent_input_callback(self, img):
        image = rnp.numpify(img)
        self.agent_input.append(image)
        self.get_logger().info(f"got agent_input {self.agent_input[-1]}")

    def user_velocity_callback(self, vel):
        self.user_output['velocity'] = vel
        self.gui.update_current_action_plot(user_vel={'linear':vel.linear.x, 'angular': vel.angular.z})
        self.get_logger().info(f"got user velocity {self.user_output['velocity']}")

    def user_termination_flag_callback(self, msg):
        self.user_output['termination_flag'] = msg.data
        self.get_logger().info(f"got user termination flag {self.user_output['termination_flag']}")
    
    def image_raw_callback(self, img):
        self.image_raw = rnp.numpify(img)
        self.gui.update_image_current(self.image_raw)
        self.get_logger().info(f"got image raw {self.image_raw.shape}")

def spin_thread_(node):
    while True:
        rclpy.spin_once(node)

def main():
    rclpy.init()

    node = SupervisorNode()

    spin_thread = threading.Thread(target=spin_thread_, args=(node,))
    spin_thread.start()
    
    node.gui.window.mainloop()

if __name__ == '__main__':
    main()