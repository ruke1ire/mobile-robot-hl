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
import os
import glob

class SupervisorNode(Node):

    def __init__(self):
        super().__init__('supervisor')

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
        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.desired_velocity_publisher = self.create_publisher(Twist, desired_velocity_topic_name, reliable_qos)
        self.termination_flag_publisher = self.create_publisher(Bool, 'termination_flag', reliable_qos)
        self.agent_output_subscriber = self.create_subscription(AgentOutput, 'agent_output', self.agent_output_callback ,best_effort_qos)
        self.agent_input_subscriber = self.create_subscription(Image, 'agent_input', self.agent_input_callback, reliable_qos)
        self.user_velocity_subscriber = self.create_subscription(Twist, 'user_input/velocity', self.user_velocity_callback, best_effort_qos)
        self.user_termination_flag_subscriber = self.create_subscription(Bool, 'user_input/termination_flag', self.user_termination_flag_callback, best_effort_qos)
        self.image_raw_subscriber = self.create_subscription(Image, image_raw_topic_name, self.image_raw_callback ,best_effort_qos)

        agent_prefix = "agent/"
        trainer_prefix='trainer/'

        self.services = {}
        self.services[agent_prefix+'start'] = self.create_client(Trigger, agent_prefix+'start')
        self.services[agent_prefix+'pause'] = self.create_client(Trigger, agent_prefix+'pause')
        self.services[agent_prefix+'stop'] = self.create_client(Trigger, agent_prefix+'stop')
        self.services[agent_prefix+'take_over'] = self.create_client(Trigger, agent_prefix+'take_over')
        self.services[agent_prefix+'select_demonstration'] = self.create_client(StringTrigger, agent_prefix+'select_demonstration')
        self.services[agent_prefix+'select_model'] = self.create_client(StringTrigger, agent_prefix+'select_model')
        self.services[agent_prefix+'select_mode'] = self.create_client(StringTrigger, agent_prefix+'select_mode')

        self.services[trainer_prefix+'select_model'] = self.create_client(StringTrigger, trainer_prefix+'select_model')
        self.services[trainer_prefix+'start'] = self.create_client(Trigger, trainer_prefix+'start')
        self.services[trainer_prefix+'pause'] = self.create_client(Trigger, trainer_prefix+'pause')
        self.services[trainer_prefix+'stop'] = self.create_client(Trigger, trainer_prefix+'stop')
        self.services[trainer_prefix+'save'] = self.create_client(Trigger, trainer_prefix+'save')
        self.services[trainer_prefix+'delete'] = self.create_client(Trigger, trainer_prefix+'delete')
        self.services[trainer_prefix+'pre_train'] = self.create_client(Trigger, trainer_prefix+'pre_train')

        self.gui = SupervisorGUI()
        self.gui.saved_demo = self.get_available_demo_names()
        self.get_logger().info("Initialized Node")
        self.image_raw = None
        self.agent_output = {}
        self.agent_input = []
        self.user_output = {}
        self.demo = []

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
    
    def call_service(self, service_name, command=None):
        if self.services[service_name].wait_for_service(timeout_sec=0.1) == False:
            self.get_logger().info(f'{service_name} service not available')
        else:
            if(command == None):
                request = Trigger.Request()
            else:
                request = StringTrigger.Request()
                request.command = str(command)

            response = self.services[service_name].call(request)
            if(response.success == True):
                pass
            else:
                self.get_logger().info(f'{service_name} service error: {response.message}')
    
    def get_available_demo_names(self):
        '''get the name of all the available demos'''
        demos = [os.basename(x) for x in glob.glob(self.demo_path+"/*")]
        return demos
    
    def get_demo(self, demo_name):
        '''return the array of images, actions, etc.'''
        raise NotImplementedError()
    
    def save_demo(self, demo_name):
        '''save a demonstration'''
        raise NotImplementedError()

    def append_demo(self, image, action):
        '''append a data point to the demonstration'''
        raise NotImplementedError()
    
    def start_automatic(self):
        '''start automatic control'''
        raise NotImplementedError()

    def pause_automatic(self):
        '''pause automatic control'''
        raise NotImplementedError()
    
    def stop_automatic(self):
        '''stop automatic control'''
        raise NotImplementedError()

    def take_over_automatic(self):
        '''take-over automatic control'''
        raise NotImplementedError()

def supervisor_node_thread_(node):
    while True:
        pass

def spin_thread_(node):
    while True:
        rclpy.spin_once(node)

def main():
    rclpy.init()

    node = SupervisorNode()

    spin_thread = threading.Thread(target=spin_thread_, args=(node,))
    supervisor_node_thread = threading.Thread(target=supervisor_node_thread_, args=(node,))

    spin_thread.start()
    supervisor_node_thread.start()
    
    node.gui.window.mainloop()

if __name__ == '__main__':
    main()