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
import os
from enum import Enum
import numpy as np

import ros2_numpy as rnp

from .utils import *

class AgentNode(Node):

    def __init__(self):
        super().__init__('agent')

        self.demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        try:
            desired_velocity_topic_name = os.environ['MOBILE_ROBOT_HL_DESIRED_VELOCITY_TOPIC']
        except:
            desired_velocity_topic_name = "desired_velocity"
        try:
            image_raw_topic_name = os.environ['MOBILE_ROBOT_HL_IMAGE_RAW_TOPIC']
        except:
            image_raw_topic_name = "image_raw/uncompressed"

        self.get_logger().info("Initializing Node")

        self.declare_parameter('frequency', 5.0)
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value

        service_prefix = 'agent/'

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.agent_output_publisher = self.create_publisher(AgentOutput, 'agent_output', reliable_qos)
        self.agent_input_publisher = self.create_publisher(Image, 'agent_input', reliable_qos)
        self.image_raw_subscriber = self.create_subscription(Image, image_raw_topic_name, self.image_raw_callback ,best_effort_qos)
        self.desired_velocity_subscriber = self.create_subscription(Twist, desired_velocity_topic_name, self.desired_velocity_callback, reliable_qos)
        self.termination_flag_subscriber = self.create_subscription(Bool, 'termination_flag', self.termination_flag_callback, reliable_qos)

        self.start_service = self.create_service(Trigger, service_prefix+'start', self.start_service_callback)
        self.pause_service = self.create_service(Trigger, service_prefix+'pause', self.pause_service_callback)
        self.stop_service = self.create_service(Trigger, service_prefix+'stop', self.stop_service_callback)
        self.take_over_service = self.create_service(Trigger, service_prefix+'take_over', self.take_over_service_callback)
        self.select_demonstration_service = self.create_service(StringTrigger, service_prefix+'select_demonstration', self.select_demonstration_service_callback)
        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_service_callback)
        self.select_mode_service = self.create_service(StringTrigger, service_prefix+'select_mode', self.select_mode_service_callback)

        self.control_loop = self.create_timer(1/self.frequency, self.control_callback)

        self.get_logger().info("Initialized Node")
        self.image_raw = None
        self.fill_int = 255
        img_tmp = np.zeros([240,320,3],dtype=np.uint8)
        img_tmp.fill(self.fill_int)
        self.image_raw_msg = rnp.msgify(Image,img_tmp, encoding='rgb8')

        self.state = AgentState.STANDBY
        self.demo = None
    
    def image_raw_callback(self, img):
        self.fill_int == None
        self.image_raw_msg = img
        self.image_raw = rnp.numpify(img)
        self.get_logger().info(f"got image {self.image_raw.shape}")

    def desired_velocity_callback(self, msg):
        self.get_logger().info("got desired velocity")

    def termination_flag_callback(self, msg):
        data = msg.data
        self.get_logger().info(f"got termination flag {data}")
    
    def start_service_callback(self):
        response = Trigger()
        if(self.demo == None):
            response.success = False
        else:
            self.state = AgentState.RUNNING
            response.success = True
        return response

    def pause_service_callback(self):
        response = Trigger()
        if(self.state == AgentState.RUNNING or self.state == AgentState.TAKE_OVER):
            self.state = AgentState.PAUSED
            response.success = True
        else:
            response.success = False
        return response

    def stop_service_callback(self):
        response = Trigger()
        if(self.state == AgentState.RUNNING or self.state == AgentState.TAKE_OVER or self.state == AgentState.PAUSED):
            self.state = AgentState.STANDBY
            self.demo = None
            response.success = True
        else:
            response.success = False
        return response

    def take_over_service_callback(self):
        response = Trigger()
        if(self.state == AgentState.RUNNING or self.state == AgentState.PAUSED):
            self.state = AgentState.TAKE_OVER
            response.success = True
        else:
            response.success = False
        return response

    def select_demonstration_service_callback(self, msg):
        response = Trigger()
        if(self.state == AgentState.STANDBY):
            demo_split = msg.command.split('.')
            demo_name = demo_split[0]
            demo_id = demo_split[1]
            #TODO: load the demo given the demo_name and demo_id
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

    def control_callback(self):
        if(self.fill_int != None):
            img_tmp = np.zeros([240,320,3],dtype=np.uint8)
            if(self.fill_int == 255):
                self.fill_int = 0
            else:
                self.fill_int = 255
            img_tmp.fill(self.fill_int)
            self.image_raw_msg = rnp.msgify(Image,img_tmp,encoding='rgb8')

        msg = self.image_raw_msg
        self.agent_input_publisher.publish(msg)

        #TODO: below will be where the model inference be

class AgentState(Enum):
    STANDBY = 0
    RUNNING = 1
    PAUSED = 2
    TAKE_OVER = 3

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
