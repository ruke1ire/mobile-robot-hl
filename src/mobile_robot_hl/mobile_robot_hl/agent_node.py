import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.msg import AgentOutput
from custom_interfaces.srv import StringTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

import threading

class AgentNode(Node):

    def __init__(self):
        super().__init__('agent')

        self.get_logger().info("Initializing Node")

        service_prefix = 'agent/'

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.agent_output_publisher = self.create_publisher(AgentOutput, 'agent_output', reliable_qos)
        self.agent_input_publisher = self.create_publisher(Image, 'agent_input', reliable_qos)
        self.image_raw_subscriber = self.create_subscription(Image, 'image_raw', self.image_raw_callback ,best_effort_qos)
        self.desired_velocity_subscriber = self.create_subscription(Twist, 'desired_velocity', self.desired_velocity_callback, reliable_qos)
        self.termination_flag_subscriber = self.create_subscription(Bool, 'termination_flag', self.termination_flag_callback, reliable_qos)

        self.start_service = self.create_service(Trigger, service_prefix+'start', self.start_service_callback)
        self.pause_service = self.create_service(Trigger, service_prefix+'pause', self.pause_service_callback)
        self.stop_service = self.create_service(Trigger, service_prefix+'stop', self.stop_service_callback)
        self.take_over_service = self.create_service(Trigger, service_prefix+'take_over', self.take_over_service_callback)
        self.select_demonstration_service = self.create_service(StringTrigger, service_prefix+'select_demonstration', self.take_over_service_callback)
        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_service_callback)
        self.select_mode_service = self.create_service(StringTrigger, service_prefix+'select_mode', self.select_mode_service_callback)

        self.get_logger().info("Initialized Node")
    
    def image_raw_callback(self, msg):
        self.get_logger().info("got image raw")

    def desired_velocity_callback(self, msg):
        self.get_logger().info("got desired velocity")

    def termination_flag_callback(self, msg):
        data = msg.data
        self.get_logger().info(f"got termination flag {data}")
    
    def start_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def pause_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def stop_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def take_over_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def select_demonstration_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def select_model_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def select_mode_service_callback(self):
        response = Trigger()
        response.success = True
        return response

def spin_thread_(node):
    while True:
        rclpy.spin_once(node)

def main():
    rclpy.init()

    node = AgentNode()

    spin_thread = threading.Thread(target=spin_thread_, args=(node,))
    spin_thread.start()
    spin_thread.join()

if __name__ == '__main__':
    main()
