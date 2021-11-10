import rclpy
from rclpy.node import Node
from rclpy.qos import *
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.msg import AgentOutput
from custom_interfaces.srv import StringTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Vector3

from .supervisor_gui import SupervisorGUI, SupervisorState
from .joy_handler import JoyHandler, InterfaceType
from .utils import *
from .model.utils import *

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
        self.model_handler = ModelHandler(path = model_path)

        self.image_raw = None
        self.agent_input = None
        self.agent_output = {'velocity':{'linear':0.0, 'angular': 0.0}, 'termination_flag':False}
        self.user_output =  {'velocity':{'linear':0.0, 'angular': 0.0}, 'termination_flag':False}
        self.episode = EpisodeData(data=None)
        self.received_agent_output = False
        self.agent_in_callback_lock = False

        self.state = SupervisorState.STANDBY

        
        self.get_logger().info("Initializing Node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_linear_velocity', 0.05),
                ('max_angular_velocity', 0.125),
            ])
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.get_logger().info(f"Parameter <max_linear_velocity> = {self.max_linear_velocity}")
        self.get_logger().info(f"Parameter <max_angular_velocity> = {self.max_angular_velocity}")

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        self.desired_velocity_publisher = self.create_publisher(Twist, desired_velocity_topic_name, reliable_qos, callback_group= ReentrantCallbackGroup())
        self.termination_flag_publisher = self.create_publisher(Bool, 'termination_flag', reliable_qos, callback_group = ReentrantCallbackGroup())
        self.action_controller_publisher = self.create_publisher(String, 'action_controller', reliable_qos, callback_group = ReentrantCallbackGroup())
        self.agent_output_subscriber = self.create_subscription(AgentOutput, 'agent_output', self.agent_output_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.agent_input_subscriber = self.create_subscription(Image, 'agent_input', self.agent_input_callback, reliable_qos, callback_group = ReentrantCallbackGroup())
        self.user_velocity_subscriber = self.create_subscription(Twist, 'user_input/velocity', self.user_velocity_callback, best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.user_termination_flag_subscriber = self.create_subscription(Bool, 'user_input/termination_flag', self.user_termination_flag_callback, best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.image_raw_subscriber = self.create_subscription(Image, image_raw_topic_name, self.image_raw_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())

        agent_prefix = "agent/"
        trainer_prefix='trainer/'

        self.agent_service_group = ReentrantCallbackGroup()
        self.services_ = dict()
        self.services_[agent_prefix+'start'] = self.create_client(StringTrigger, agent_prefix+'start', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'pause'] = self.create_client(Trigger, agent_prefix+'pause', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'stop'] = self.create_client(Trigger, agent_prefix+'stop', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'select_model'] = self.create_client(StringTrigger, agent_prefix+'select_model', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'select_mode'] = self.create_client(StringTrigger, agent_prefix+'select_mode', callback_group=self.agent_service_group)

        self.trainer_service_group = ReentrantCallbackGroup()
        self.services_[trainer_prefix+'select_model'] = self.create_client(StringTrigger, trainer_prefix+'select_model', callback_group=self.trainer_service_group)
        self.services_[trainer_prefix+'start'] = self.create_client(Trigger, trainer_prefix+'start', callback_group=self.trainer_service_group)
        self.services_[trainer_prefix+'pause'] = self.create_client(Trigger, trainer_prefix+'pause', callback_group=self.trainer_service_group)
        self.services_[trainer_prefix+'stop'] = self.create_client(Trigger, trainer_prefix+'stop', callback_group=self.trainer_service_group)
        self.services_[trainer_prefix+'save'] = self.create_client(Trigger, trainer_prefix+'save', callback_group=self.trainer_service_group)
        self.services_[trainer_prefix+'delete'] = self.create_client(Trigger, trainer_prefix+'delete', callback_group=self.trainer_service_group)
        self.services_[trainer_prefix+'pre_train'] = self.create_client(Trigger, trainer_prefix+'pre_train', callback_group=self.trainer_service_group)

        self.gui = SupervisorGUI(ros_node=self)
        self.gui.update_available_demo_name(self.demo_handler.get_names())
        self.gui.update_available_task_episode_name(self.task_handler.get_names())

        self.get_logger().info("Initialized Node")

        self.joy_handler = JoyHandler(self.max_linear_velocity, self.max_angular_velocity)

        threading.Thread(target=lambda: self.joy_event_handler_thread()).start()
        threading.Thread(target=lambda: self.joy_action_handler_thread()).start()

    def agent_output_callback(self, msg):
        velocity = msg.velocity
        termination_flag = msg.termination_flag
        self.agent_output['velocity'] = {'linear':velocity.linear.x, 'angular': velocity.angular.z}
        self.agent_output['termination_flag'] = termination_flag
        self.received_agent_output = True
        threading.Thread(target=self.gui.update_info(agent_vel = self.agent_output['velocity'], agent_termination=self.agent_output['termination_flag'])).start()

    def agent_input_callback(self, img):
        self.get_logger().info("Received agent_input")
        current_state = self.state
        self.agent_in_callback_lock = True
        image = PImage.fromarray(rnp.numpify(img))
        self.agent_input = image
        if(self.state == SupervisorState.TASK_RUNNING):
            self.gui.update_info(controller=ControllerType.AGENT)
            self.get_logger().info("Waiting for agent_output")
            while(self.received_agent_output == False):
                if(self.state == SupervisorState.STANDBY):
                    self.get_logger().info("State changed to STANDBY")
                    return
            self.get_logger().info("Received agent_output")

            velocity_msg = Twist(linear=Vector3(x=self.agent_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.agent_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(velocity_msg)
            bool_msg = Bool(data=self.agent_output['termination_flag'])
            self.termination_flag_publisher.publish(bool_msg)
            controller_msg = String(data=ControllerType.AGENT.name)
            self.action_controller_publisher.publish(controller_msg)
            self.get_logger().info("Published desired_velocity, termination_flag, and action_controller")

            #self.get_logger().info(str(self.episode.get_data()))

            if(self.episode.get_data(index=-1)['action']['controller'] in [ControllerType.NONE,None]):
                length = self.episode.get_episode_length()
                self.episode.set_data(
                    index = length-1,
                    image=self.agent_input,
                    agent_linear_vel=self.agent_output['velocity']['linear'],
                    agent_angular_vel=self.agent_output['velocity']['angular'],
                    agent_termination_flag=self.agent_output['termination_flag'],
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.AGENT
                )

            else:
                self.episode.append_data(
                    image=self.agent_input,
                    agent_linear_vel=self.agent_output['velocity']['linear'],
                    agent_angular_vel=self.agent_output['velocity']['angular'],
                    agent_termination_flag=self.agent_output['termination_flag'],
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.AGENT
                    )

            #set_ep_thread = threading.Thread(target=lambda: self.gui.set_episode(self.episode)).start()
            self.gui.set_episode(self.episode)
            self.get_logger().info("Completed setting up frame to GUI")
            self.get_logger().info(f'Current Episode Length: {self.episode.get_episode_length()}')

        elif(self.state == SupervisorState.TASK_TAKE_OVER):
            self.gui.update_info(controller=ControllerType.USER)
            self.get_logger().info("Waiting for agent_output")
            while(self.received_agent_output == False):
                if(self.state == SupervisorState.STANDBY):
                    self.get_logger().info("State changed to STANDBY")
                    return
            self.get_logger().info("Received agent_output")

            velocity_msg = Twist(linear=Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(velocity_msg)
            bool_msg = Bool(data=self.user_output['termination_flag'])
            self.termination_flag_publisher.publish(bool_msg)
            controller_msg = String(data=ControllerType.USER.name)
            self.action_controller_publisher.publish(controller_msg)
            self.get_logger().info("Published desired_velocity, termination_flag, and action_controller")

            if(self.episode.get_data(index=-1)['action']['controller'] in [ControllerType.NONE,None]):
                length = self.episode.get_episode_length()
                self.episode.set_data(
                    index = length-1,
                    image=self.agent_input,
                    agent_linear_vel=self.agent_output['velocity']['linear'],
                    agent_angular_vel=self.agent_output['velocity']['angular'],
                    agent_termination_flag=self.agent_output['termination_flag'],
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.USER
                )
            else:
                self.episode.append_data(
                    image=self.agent_input,
                    agent_linear_vel=self.agent_output['velocity']['linear'],
                    agent_angular_vel=self.agent_output['velocity']['angular'],
                    agent_termination_flag=self.agent_output['termination_flag'],
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.USER
                    )

            if(current_state in [SupervisorState.TASK_RUNNING, SupervisorState.TASK_TAKE_OVER] and self.state == SupervisorState.STANDBY):
                self.get_logger().info("Skipped frame and resetting episode")
                self.reset_episode()
                return
            self.gui.set_episode(self.episode)
            self.get_logger().info("Completed setting up frame to GUI")
            self.get_logger().info(f'Current Episode Length: {self.episode.get_episode_length()}')

        elif(self.state == SupervisorState.DEMO_RECORDING):
            self.gui.update_info(controller=ControllerType.USER)
            velocity_msg = Twist(linear=Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(velocity_msg)
            bool_msg = Bool(data=self.user_output['termination_flag'])
            self.termination_flag_publisher.publish(bool_msg)
            controller_msg = String(data=ControllerType.USER.name)
            self.action_controller_publisher.publish(controller_msg)
            self.get_logger().info("Published desired_velocity, termination_flag, and action_controller")

            if(self.episode.get_data(index = -1)['action']['controller'] in [ControllerType.NONE,None]):
                length = self.episode.get_episode_length()
                self.episode.set_data(
                    index = length-1,
                    image=self.agent_input,
                    agent_linear_vel=None,
                    agent_angular_vel=None,
                    agent_termination_flag=None,
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.USER
                )
            else:
                self.episode.append_data(
                    image=self.agent_input,
                    agent_linear_vel=None,
                    agent_angular_vel=None,
                    agent_termination_flag=None,
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.USER
                    )

            self.gui.set_episode(self.episode)
            self.get_logger().info("Completed setting up frame to GUI")
            self.get_logger().info(f'Current Episode Length: {self.episode.get_episode_length()}')
        else:
            self.gui.update_info(controller=ControllerType.USER)
            if(self.episode.get_data(index = -1)['action']['controller'] in [ControllerType.NONE,None]):
                length = self.episode.get_episode_length()
                self.episode.set_data(
                    index = length-1,
                    image=image,
                    agent_linear_vel=None,
                    agent_angular_vel=None,
                    agent_termination_flag=None,
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.NONE
                )
            else:
                self.episode.append_data(
                    image=image,
                    agent_linear_vel=None,
                    agent_angular_vel=None,
                    agent_termination_flag=None,
                    user_linear_vel=self.user_output['velocity']['linear'],
                    user_angular_vel=self.user_output['velocity']['angular'],
                    user_termination_flag=self.user_output['termination_flag'],
                    controller=ControllerType.NONE
                )
            self.gui.set_episode(self.episode)
            self.get_logger().info("Completed setting up frame to GUI")

        self.agent_in_callback_lock = False

    def user_velocity_callback(self, vel):
        self.user_output['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
        self.gui.update_info(user_vel=self.user_output['velocity'])
        self.get_logger().info(f"got user velocity {self.user_output['velocity']}")

    def user_termination_flag_callback(self, msg):
        self.user_output['termination_flag'] = msg.data
        self.get_logger().info(f"got user termination flag {self.user_output['termination_flag']}")
    
    def image_raw_callback(self, img):
        self.image_raw = PImage.fromarray(rnp.numpify(img))
        #self.gui.update_image_current(self.image_raw)
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
            return response
    
    def update_state(self, state):
        self.state = state

    def save_demo(self, name):
        self.get_logger().info("Saving demonstration")
        self.demo_handler.save(self.episode, name)
        self.get_logger().info("Saved demonstration")
    
    def save_task_episode(self, demo_name, demo_id):
        self.task_handler.save(self.episode, demo_name, demo_id)

    def reset_episode(self):
        self.episode.init_empty_structure()
        self.received_agent_output = False

    def joy_action_handler_thread(self):
        while True:
            joy_state = self.joy_handler.get_state()

            self.user_output =  {'velocity':{'linear':joy_state[InterfaceType.LINEAR_VELOCITY.name], 'angular': joy_state[InterfaceType.ANGULAR_VELOCITY.name]}, 'termination_flag':joy_state[InterfaceType.TERMINATION_FLAG.name]}

            self.episode.set_key_value(key = 'action.user.velocity.linear', value = self.user_output['velocity']['linear'], index = -1)
            self.episode.set_key_value(key = 'action.user.velocity.angular', value = self.user_output['velocity']['angular'], index = -1)
            self.episode.set_key_value(key = 'action.user.termination_flag', value = self.user_output['termination_flag'], index = -1)

            if(self.state == SupervisorState.STANDBY):
                velocity_msg = Twist(linear=Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
                self.desired_velocity_publisher.publish(velocity_msg)
                bool_msg = Bool(data=self.user_output['termination_flag'])
                self.termination_flag_publisher.publish(bool_msg)
                controller_msg = String(data=ControllerType.USER.name)
                self.action_controller_publisher.publish(controller_msg)

            self.gui.update_info(user_vel=self.user_output['velocity'], user_termination=self.user_output['termination_flag'])
            time.sleep(0.1)

    def joy_event_handler_thread(self):
        prev_joy_state = self.joy_handler.get_state()
        while True:
            joy_state = self.joy_handler.get_state()
            if(prev_joy_state[InterfaceType.STOP.name] == False and joy_state[InterfaceType.STOP.name] == True):
                if(self.gui.state in [SupervisorState.TASK_RUNNING, SupervisorState.TASK_PAUSED, SupervisorState.TASK_TAKE_OVER]):
                    self.gui.agent_stop_button_trigger()
                elif(self.gui.state in [SupervisorState.DEMO_RECORDING, SupervisorState.DEMO_PAUSED]):
                    self.gui.demo_stop_button_trigger()
            elif(prev_joy_state[InterfaceType.START_PAUSE_TASK.name] == False and joy_state[InterfaceType.START_PAUSE_TASK.name] == True):
                if(self.gui.state in [SupervisorState.STANDBY, SupervisorState.TASK_RUNNING, SupervisorState.TASK_PAUSED, SupervisorState.TASK_TAKE_OVER]):
                    self.gui.agent_start_button_trigger()
            elif(prev_joy_state[InterfaceType.TAKE_OVER_TASK.name] == False and joy_state[InterfaceType.TAKE_OVER_TASK.name] == True):
                if(self.gui.state in [SupervisorState.TASK_RUNNING, SupervisorState.TASK_PAUSED, SupervisorState.TASK_TAKE_OVER]):
                    self.gui.agent_take_over_button_trigger()
            elif(prev_joy_state[InterfaceType.START_PAUSE_DEMO.name] == False and joy_state[InterfaceType.START_PAUSE_DEMO.name] == True):
                if(self.gui.state in [SupervisorState.STANDBY, SupervisorState.DEMO_RECORDING, SupervisorState.DEMO_PAUSED]):
                    self.gui.demo_start_button_trigger()

            prev_joy_state = joy_state.copy()
            time.sleep(0.1)



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