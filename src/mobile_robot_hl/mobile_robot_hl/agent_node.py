import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.msg import AgentOutput
from custom_interfaces.srv import StringTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import threading
import os
from enum import Enum
import numpy as np
import time

import ros2_numpy as rnp

from .utils import *

class AgentNode(Node):

    def __init__(self):
        super().__init__('agent')

        demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']
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
        self.received_desired_vel = False
        self.received_termination_flag = False
        self.received_action_controller = False
        self.image_raw = None
        self.fill_int = 255
        img_tmp = np.zeros([240,320,3],dtype=np.uint8)
        img_tmp.fill(self.fill_int)
        self.image_raw_msg = rnp.msgify(Image,img_tmp, encoding='rgb8')
        self.state = AgentState.STANDBY
        self.episode = EpisodeData(data=None, structure=DataStructure.LIST_DICT)
        self.wait = False

        self.get_logger().info("Initializing Node")

        self.declare_parameter('frequency', 0.5)
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value

        service_prefix = 'agent/'

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.agent_output_publisher = self.create_publisher(AgentOutput, 'agent_output', reliable_qos, callback_group=ReentrantCallbackGroup())
        self.agent_input_publisher = self.create_publisher(Image, 'agent_input', reliable_qos, callback_group=ReentrantCallbackGroup())
        self.image_raw_subscriber = self.create_subscription(Image, image_raw_topic_name, self.image_raw_callback ,best_effort_qos, callback_group=ReentrantCallbackGroup())
        self.desired_velocity_subscriber = self.create_subscription(Twist, desired_velocity_topic_name, self.desired_velocity_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.termination_flag_subscriber = self.create_subscription(Bool, 'termination_flag', self.termination_flag_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.action_controller_subscriber = self.create_subscription(String, 'action_controller', self.action_controller_callback, reliable_qos, callback_group=ReentrantCallbackGroup())

        self.start_service = self.create_service(Trigger, service_prefix+'start', self.start_service_callback)
        self.pause_service = self.create_service(Trigger, service_prefix+'pause', self.pause_service_callback)
        self.stop_service = self.create_service(Trigger, service_prefix+'stop', self.stop_service_callback)
        self.select_demonstration_service = self.create_service(StringTrigger, service_prefix+'select_demonstration', self.select_demonstration_service_callback)
        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_service_callback)
        self.select_mode_service = self.create_service(StringTrigger, service_prefix+'select_mode', self.select_mode_service_callback)

        self.control_loop = self.create_timer(1/self.frequency, self.control_callback, callback_group=ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")
    
    def image_raw_callback(self, img):
        self.fill_int = None
        self.image_raw_msg = img
        self.image_raw = rnp.numpify(img)

    def desired_velocity_callback(self, msg):
        self.desired_vel = {'linear':msg.linear.x, 'angular':msg.angular.z}
        self.received_desired_vel = True

    def termination_flag_callback(self, msg):
        self.termination_flag = msg.data
        self.received_termination_flag = True
    
    def action_controller_callback(self, msg):
        self.action_controller = ControllerType[msg.data]
        self.received_action_controller = True
    
    def start_service_callback(self, request, response):
        if(self.episode.data_empty == True):
            response.success = False
        else:
            if(self.state == AgentState.STANDBY):
                self.wait = True
                time.sleep(2.0)
                self.wait = False
            self.state = AgentState.RUNNING
            response.success = True
        return response

    def pause_service_callback(self, request, response):
        if(self.state == AgentState.RUNNING or self.state == AgentState.TAKE_OVER or self.state == AgentState.PAUSED):
            self.state = AgentState.PAUSED
            response.success = True
        else:
            response.success = False
        return response

    def stop_service_callback(self, request, response):
        self.state = AgentState.STANDBY
        self.episode = EpisodeData(data=None, structure=DataStructure.LIST_DICT)
        response.success = True
        return response

    def select_demonstration_service_callback(self, request, response):
        if(self.state == AgentState.STANDBY):
            demo_split = request.command.split('.')
            demo_name = demo_split[0]
            demo_id = demo_split[1]
            episode = self.demo_handler.get(demo_name, demo_id)
            episode_data = EpisodeData(data=episode.data, structure=DataStructure.DICT_LIST)
            episode_data.restructure(DataStructure.LIST_DICT)
            self.episode = episode_data
            self.received_desired_vel = True
            self.received_termination_flag = True
            self.received_action_controller = True

            self.get_logger().info(f"Selected demonstration: {request.command}")
        response.success = True
        return response

    def select_model_service_callback(self, request, response):
        response.success = True
        return response

    def select_mode_service_callback(self, request, response):
        response = Trigger()
        response.success = True
        return response

    def control_callback(self):
        if(self.wait == True):
            return
        self.get_logger().info("Publishing agent_in")
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
        self.get_logger().info("Published agent_in")

        if(self.state == AgentState.RUNNING):
            self.get_logger().info("Verifying that previous actions are received")
            while(self.received_desired_vel == False or self.received_termination_flag == False or self.received_action_controller == False):
                pass
            self.received_desired_vel = False
            self.received_termination_flag = False
            self.received_action_controller = False
            self.get_logger().info("Computing agent_output")
            self.get_logger().info("Publishing agent_output")
            #TODO: model inference + publish agent_output
            agent_out = AgentOutput(velocity = Twist(linear=Vector3(x=0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=0.0)))
            self.agent_output_publisher.publish(agent_out)
            self.get_logger().info("Published agent_output")
            #TODO: this is where new information will be appended to the model
            self.episode.append_data(
                image=None,
                agent_linear_vel=None,
                agent_angular_vel=None,
                agent_termination_flag=None,
                user_linear_vel=None,
                user_angular_vel=None,
                user_termination_flag=None,
                controller=None
            )

        elif(self.state == AgentState.PAUSED):
            self.get_logger().info("Computing agent_output")
            self.get_logger().info("Publishing agent_output")
            #TODO: model inference + publish agent_output, information is NOT appended
            agent_out = AgentOutput(velocity = Twist(linear=Vector3(x=0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=0.0)))
            self.agent_output_publisher.publish(agent_out)
            self.get_logger().info("Published agent_output")
            pass
        else:
            pass

        self.get_logger().info(f"Current Episode Length: {self.episode.get_episode_length()}")


class AgentState(Enum):
    STANDBY = 0
    RUNNING = 1
    PAUSED = 2

def spin_thread_(node, executor):
    rclpy.spin(node, executor)

def main():
    rclpy.init()

    node = AgentNode()
    executor = MultiThreadedExecutor()

    spin_thread = threading.Thread(target=spin_thread_, args=(node, executor, ))
    spin_thread.start()
    spin_thread.join()

if __name__ == '__main__':
    main()
