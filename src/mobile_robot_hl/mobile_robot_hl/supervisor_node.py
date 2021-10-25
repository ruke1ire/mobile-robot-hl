import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.msg import AgentOutput
from custom_interfaces.srv import StringTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from .supervisor_gui import SupervisorGUI, SupervisorState

import ros2_numpy as rnp

import threading
import tkinter
import os
import glob
from PIL import Image as PImage
import yaml
from enum import Enum

class SupervisorNode(Node):

    def __init__(self):
        super().__init__('supervisor')

        self.demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        self.task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']
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

        self.gui = SupervisorGUI()
        self.gui.saved_demo = self.get_available_demo_names()
        self.get_logger().info("Initialized Node")
        self.image_raw = None
        self.agent_output = {}
        self.agent_input = None
        self.user_output = {}
        self.demo = [] # {"image", "velocity", "termination_flag"}
        self.task_episode = [] # {"image", "velocity", "termination_flag", "controller"}

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
        self.get_logger().info(f"got agent_input {self.agent_input}")
        if(self.state == SupervisorState.TASK_RUNNING):
            if(len(self.task_episode) == 0):
                self.task_episode.append({'image':image})
                return
            output_msg = Twist(linear=Twist.Vector3(x=self.agent_output['velocity']['linear'],y=0.0,z=0.0),angular=Twist.Vector3(x=0.0,y=0.0,z=self.agent_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(output_msg)
            self.termination_flag_publisher.publish(self.agent_output['termination_flag'])
            self.task_episode[-1]['velocity'] = self.agent_output['velocity']
            self.task_episode[-1]['termination_flag'] = self.agent_output['termination_flag']
            self.task_episode[-1]['controller'] = ControllerType.AGENT
            self.task_episode.append({'image':image})
        elif(self.state == SupervisorState.TASK_TAKE_OVER):
            if(len(self.task_episode) == 0):
                self.task_episode.append({'image':image})
                return
            output_msg = Twist(linear=Twist.Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Twist.Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(output_msg)
            self.termination_flag_publisher.publish(self.user_output['termination_flag'])
            self.task_episode[-1]['velocity'] = self.user_output['velocity']
            self.task_episode[-1]['termination_flag'] = self.user_output['termination_flag']
            self.task_episode[-1]['controller'] = ControllerType.USER
            self.task_episode.append({'image':image})
        elif(self.state == SupervisorState.DEMO_RECORDING):
            if(len(self.task_episode) == 0):
                self.demo.append({'image':image})
                return
            self.demo[-1]['velocity'] = self.user_output['velocity']
            self.demo[-1]['termination_flag'] = self.user_output['termination_flag']
            self.demo.append({'image':image})

    def user_velocity_callback(self, vel):
        self.user_output['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
        self.gui.update_current_action_plot(user_vel=self.user_output['velocity'])
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
    
    def get_available_demo_id(self, demo_name):
        demos = [os.basename(x) for x in glob.glob(self.demo_path+"/"+demo_name+"/*") if "yaml" not in x]
        return demos
    
    def get_demo(self, demo_name, demo_id):
        '''return the array of images, actions, etc.'''
        info = yaml.load(self.demo_path+"/"+demo_name+"/"+demo_id+"/demo_info.yaml")
        velocity = info['actions']['velocity']
        termination_flag = info['actions']['termination_flag']
        images = [PImage.open(self.demo_path+"/"+demo_name+"/"+demo_id+"/"+image_id+".png") 
                    for image_id in info['observations']['image_id']]
        return images, velocity, termination_flag
    
    def get_current_demo(self):
        images = [data['image'] for data in self.demo]
        velocity = [data['velocity'] for data in self.demo]
        termination_flag = [data['termination_flag'] for data in self.demo]
        return images, velocity, termination_flag
    
    def save_demo(self, demo_name, demo_id=None):
        '''save a demonstration'''
        if(demo_id == None):
            try:
                taken_id = max([x for x in self.get_available_demo_id() if type(x) == int])
                next_id = taken_id + 1
            except:
                next_id = 0

        image_ids = list(range(len(self.demo)))
        for i in image_ids:
            img = PImage.fromarray(self.demo[i]["image"])
            img.save(f"{self.demo_path}/{demo_name}/{next_id}/{i}.png")
        
        velocity = [data['velocity'] for data in self.demo]
        termination_flag = [data['termination_flag'] for data in self.demo]
        demo_dict = {
                        'observations':{
                            'image_id':image_ids
                        },
                        'actions':{
                            'velocity': velocity,
                            'termination_flag': termination_flag
                        }
                    }
        
        with open(f'{self.demo_path}/{demo_name}/{next_id}/demo_info.yaml', 'w') as outfile:
            yaml.dump(demo_dict, outfile)

    def append_demo(self, image, velocity, termination_flag):
        '''append a data point to the demonstration'''
        self.demo.append({'image':image,'velocity':velocity,'termination_flag':termination_flag})
    
    def reset_demo(self):
        self.demo = []

    def get_available_task_id(self, demo_name):
        task_episodes = [os.basename(x) for x in glob.glob(self.task_path+"/"+demo_name+"/*") if "yaml" not in x]
        return task_episodes
    
    def get_task_episode(self, demo_name, task_id):
        info = yaml.load(self.demo_path+"/"+demo_name+"/"+task_id+"/task_info.yaml")
        demo_id = info['demo_id']
        demo_info = yaml.load(self.demo_path+"/"+demo_name+"/"+demo_id+"demo_info.yaml")

        demo_velocity = demo_info['actions']['velocity']
        demo_termination_flag = demo_info['actions']['termination_flag']
        demo_images = [PImage.open(self.demo_path+"/"+demo_name+"/"+demo_id+"/"+image_id+".png") 
                    for image_id in demo_info['observations']['image_id']]
        
        task_velocity = info['actions']['velocity']
        task_termination_flag = info['actions']['termination_flag']
        task_controller = info['actions']['controller']
        task_images = [PImage.open(self.task_path+"/"+demo_name+"/"+task_id+"/"+image_id+".png") 
                    for image_id in demo_info['observations']['image_id']]
        
        velocity = demo_velocity+task_velocity
        termination_flag = demo_termination_flag + task_termination_flag
        images = demo_images + task_images

        return images, velocity, termination_flag, task_controller

    def get_current_task_episode(self):
        images = [data['image'] for data in self.task_episode]
        velocity = [data['velocity'] for data in self.task_episode]
        termination_flag = [data['termination_flag'] for data in self.task_episode]
        controller = [data['controller'] for data in self.task_episode]
        return images, velocity, termination_flag, controller
    
    def save_task_episode(self, demo_name, demo_id, task_id = None):
        if(task_id == None):
            try:
                taken_id = max([x for x in self.get_available_task_id() if type(x) == int])
                next_id = taken_id + 1
            except:
                next_id = 0

        image_ids = list(range(len(self.task_episode)))
        for i in image_ids:
            img = PImage.fromarray(self.task_episode[i]["image"])
            img.save(f"{self.task_path}/{demo_name}/{next_id}/{i}.png")
        
        velocity = [data['velocity'] for data in self.task_episode]
        termination_flag = [data['termination_flag'] for data in self.task_episode]
        controller = [data['controller'] for data in self.task_episode]
        task_episode_dict = {
                        'observations':{
                            'image_id':image_ids
                        },
                        'actions':{
                            'velocity': velocity,
                            'termination_flag': termination_flag,
                            'controller': controller
                        },
                        'demonstration': demo_id
                    }
        
        with open('task_info_info.yaml', 'w') as outfile:
            yaml.dump(task_episode_dict, outfile)
    
    def append_task_episode(self, image, velocity, termination_flag, controller):
        self.task_episode.append({'image':image,'velocity':velocity,'termination_flag':termination_flag, 'controller':controller})
    
    def reset_task_episode(self):
        self.task_episode = []
    
    def update_state(self, state):
        self.state = state
        if(self.state == SupervisorState.STANDBY):
            self.call_service('agent/pause')
        elif(self.state == SupervisorState.TASK_RUNNING):
            self.call_service('agent/start')
        elif(self.state == SupervisorState.TASK_PAUSED):
            self.call_service('agent/pause')
        elif(self.state == SupervisorState.TASK_TAKE_OVER):
            self.call_service('agent/take_over')

    def reset_velocity(self):
        self.desired_velocity = {'linear':0.0, 'angular':0.0}

class ControllerType(Enum):
    USER = 0
    AGENT = 1

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