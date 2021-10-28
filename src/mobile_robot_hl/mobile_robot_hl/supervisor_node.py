import rclpy
from rclpy.node import Node
from rclpy.qos import *
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.msg import AgentOutput
from custom_interfaces.srv import StringTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import Vector3

from .supervisor_gui import SupervisorGUI, SupervisorState
from .utils import *

import ros2_numpy as rnp

import threading
import tkinter
import os
import glob
from PIL import Image as PImage
import yaml
from enum import Enum
import time

class SupervisorNode(Node):

    def __init__(self):
        super().__init__('supervisor')

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
        
        self.get_logger().info("Initializing Node")

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
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

        self.services_ = dict()
        self.services_[agent_prefix+'start'] = self.create_client(Trigger, agent_prefix+'start')
        self.services_[agent_prefix+'pause'] = self.create_client(Trigger, agent_prefix+'pause')
        self.services_[agent_prefix+'stop'] = self.create_client(Trigger, agent_prefix+'stop')
        self.services_[agent_prefix+'take_over'] = self.create_client(Trigger, agent_prefix+'take_over')
        self.services_[agent_prefix+'select_demonstration'] = self.create_client(StringTrigger, agent_prefix+'select_demonstration')
        self.services_[agent_prefix+'select_model'] = self.create_client(StringTrigger, agent_prefix+'select_model')
        self.services_[agent_prefix+'select_mode'] = self.create_client(StringTrigger, agent_prefix+'select_mode')

        self.services_[trainer_prefix+'select_model'] = self.create_client(StringTrigger, trainer_prefix+'select_model')
        self.services_[trainer_prefix+'start'] = self.create_client(Trigger, trainer_prefix+'start')
        self.services_[trainer_prefix+'pause'] = self.create_client(Trigger, trainer_prefix+'pause')
        self.services_[trainer_prefix+'stop'] = self.create_client(Trigger, trainer_prefix+'stop')
        self.services_[trainer_prefix+'save'] = self.create_client(Trigger, trainer_prefix+'save')
        self.services_[trainer_prefix+'delete'] = self.create_client(Trigger, trainer_prefix+'delete')
        self.services_[trainer_prefix+'pre_train'] = self.create_client(Trigger, trainer_prefix+'pre_train')

        self.gui = SupervisorGUI(ros_node=self)
        self.gui.update_available_demo_name(self.demo_handler.get_names())

        self.get_logger().info("Initialized Node")
        self.image_raw = None
        self.agent_output = {'velocity':{'linear':0.0, 'angular': 0.0}, 'termination_flag':False}
        self.agent_input = None
        self.user_output =  {'velocity':{'linear':0.0, 'angular': 0.0}, 'termination_flag':False}
        self.demo = [] # list(dict(image, velocity, termination_flag)
        self.task_episode = [] # list(dict(image, velocity, termination_flag, controller))

        self.desired_velocity = {'linear':0.0, 'angular': 0.0}
        self.state = SupervisorState.STANDBY

    def agent_output_callback(self, msg):
        velocity = msg.velocity
        termination_flag = msg.termination_flag
        self.agent_output['velocity'] = {'linear':velocity.linear.x, 'angular': velocity.angular.z}
        self.agent_output['termination_flag'] = termination_flag
        self.gui.update_current_action_plot(agent_vel=self.agent_output['velocity'])
        self.get_logger().info(f"got agent_output {self.agent_output}")

    def agent_input_callback(self, img):
        image = rnp.numpify(img)
        self.agent_input = image
        #self.get_logger().info(f"got agent_input")
        if(self.state == SupervisorState.TASK_RUNNING):
            if(len(self.task_episode) == 0):
                self.task_episode.append({'image':image})
                return
            velocity_msg = Twist(linear=Vector3(x=self.agent_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.agent_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(velocity_msg)
            bool_msg = Bool(data=self.agent_output['termination_flag'])
            self.termination_flag_publisher.publish(bool_msg)
            self.task_episode[-1]['velocity'] = self.agent_output['velocity'].copy()
            self.task_episode[-1]['termination_flag'] = self.agent_output['termination_flag']
            self.task_episode[-1]['controller'] = ControllerType.AGENT
            self.get_logger().info(f'Episode Length: {len(self.task_episode)}')
            self.task_episode.append({'image':image})
        elif(self.state == SupervisorState.TASK_TAKE_OVER):
            if(len(self.task_episode) == 0):
                self.task_episode.append({'image':image})
                return
            velocity_msg = Twist(linear=Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(velocity_msg)
            bool_msg = Bool(data=self.user_output['termination_flag'])
            self.termination_flag_publisher.publish(bool_msg)
            self.task_episode[-1]['velocity'] = self.user_output['velocity'].copy()
            self.task_episode[-1]['termination_flag'] = self.user_output['termination_flag']
            self.task_episode[-1]['controller'] = ControllerType.USER
            self.get_logger().info(f'Episode Length: {len(self.task_episode)}')
            self.task_episode.append({'image':image})
        elif(self.state == SupervisorState.DEMO_RECORDING):
            if(len(self.demo) == 0):
                self.demo.append({'image':image})
                return
            self.demo[-1]['velocity'] = self.user_output['velocity'].copy()
            self.demo[-1]['termination_flag'] = self.user_output['termination_flag']
            self.get_logger().info(f'Demonstration Length: {len(self.demo)}')
            self.demo.append({'image':image})

    def user_velocity_callback(self, vel):
        self.user_output['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
        self.gui.update_current_action_plot(user_vel=self.user_output['velocity'])
        self.gui.update_info(user_vel=self.user_output['velocity'])
        self.get_logger().info(f"got user velocity {self.user_output['velocity']}")

    def user_termination_flag_callback(self, msg):
        self.user_output['termination_flag'] = msg.data
        self.get_logger().info(f"got user termination flag {self.user_output['termination_flag']}")
    
    def image_raw_callback(self, img):
        self.image_raw = rnp.numpify(img)
        self.gui.update_image_current(self.image_raw)
        #self.get_logger().info(f"got image raw {self.image_raw.shape}")
    
    def call_service(self, service_name, command=None):
        self.get_logger().info(f'calling service {service_name}.')
        if self.services_[service_name].wait_for_service(timeout_sec=0.1) == False:
            self.get_logger().info(f'{service_name} service not available')
        else:
            if(command == None):
                request = Trigger.Request()
            else:
                request = StringTrigger.Request()
                request.command = str(command)

            response = self.services_[service_name].call(request)
            if(response.success == True):
                self.get_logger().info(f'service successful: {service_name}')
            else:
                self.get_logger().info(f'{service_name} service error: {response.message}')
    
    def get_current_demo(self):
        images = [data['image'] for data in self.demo[:-1]]
        velocity = [data['velocity'] for data in self.demo[:-1]]
        termination_flag = [data['termination_flag'] for data in self.demo[:-1]]
        return images, velocity, termination_flag
    
    def append_demo(self, image, velocity, termination_flag):
        '''append a data point to the demonstration'''
        self.demo.append({'image':image,'velocity':velocity,'termination_flag':termination_flag})
    
    def reset_demo(self):
        self.demo = []

    def get_current_task_episode(self):
        images = [data['image'] for data in self.task_episode]
        velocity = [data['velocity'] for data in self.task_episode]
        termination_flag = [data['termination_flag'] for data in self.task_episode]
        controller = [data['controller'] for data in self.task_episode]
        return images, velocity, termination_flag, controller
    
    def append_task_episode(self, image, velocity, termination_flag, controller):
        self.task_episode.append({'image':image,'velocity':velocity,'termination_flag':termination_flag, 'controller':controller})
    
    def reset_task_episode(self):
        self.task_episode = []
    
    def update_state(self, state):
        self.state = state

    def reset_velocity(self):
        self.desired_velocity = {'linear':0.0, 'angular':0.0}

    def save_demo(self, name):
        self.demo_handler.save(self.demo, name)
    
    def save_task_episode(self, demo_name, demo_id):
        self.task_handler.save(self.task_episode, demo_name, demo_id)

def supervisor_node_thread_(node):
    while True:
        pass

def spin_thread_(node, executor):
    rclpy.spin(node, executor=executor)

def main():
    rclpy.init()

    node = SupervisorNode()
    executor = MultiThreadedExecutor()

    spin_thread = threading.Thread(target=spin_thread_, args=(node,executor,))
    supervisor_node_thread = threading.Thread(target=supervisor_node_thread_, args=(node,))

    spin_thread.start()
    supervisor_node_thread.start()
    
    node.gui.window.mainloop()

if __name__ == '__main__':
    main()