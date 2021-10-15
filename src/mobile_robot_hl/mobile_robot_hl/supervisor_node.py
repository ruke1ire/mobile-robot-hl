import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.msg import AgentOutput
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from .supervisor_gui import SupervisorGUI

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

        self.gui = SupervisorGUI()
        self.get_logger().info("Initialized Node")

    def agent_output_callback(self, msg):
        self.get_logger().info("got agent_output")

    def agent_input_callback(self, msg):
        self.get_logger().info("got agent_input")

    def user_velocity_callback(self, msg):
        self.get_logger().info("got user velocity")

    def user_termination_flag_callback(self, msg):
        self.get_logger().info("got user termination flag")

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