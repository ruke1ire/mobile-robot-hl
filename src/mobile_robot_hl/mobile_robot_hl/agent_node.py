from torch.autograd.grad_mode import inference_mode
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
from .model.model import MimeticSNAIL
from .model.utils import *
from .model.model_handler import ModelHandler

class AgentNode(Node):

    def __init__(self):
        super().__init__('agent')

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

        self.received_desired_vel = False
        self.received_termination_flag = False
        self.received_action_controller = False

        self.image_raw = None
        self.fill_int = 255
        img_tmp = np.zeros([240,320,3],dtype=np.uint8)
        img_tmp.fill(self.fill_int)
        self.image_raw_msg = rnp.msgify(Image,img_tmp, encoding='rgb8')
        self.desired_vel = dict(linear = 0.0, angular = 0.0)
        self.termination_flag = False

        self.state = AgentState.STANDBY
        self.episode = EpisodeData(data=None)
        self.wait = False

        self.model = None

        self.get_logger().info("Initializing Node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('frequency', 0.6),
                ('max_linear_velocity', 0.05),
                ('max_angular_velocity', 0.125),
            ])

        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value

        self.get_logger().info(f"Parameter <frequency> = {self.frequency}")
        self.get_logger().info(f"Parameter <max_linear_velocity> = {self.max_linear_velocity}")
        self.get_logger().info(f"Parameter <max_angular_velocity> = {self.max_angular_velocity}")

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

        self.service_group = ReentrantCallbackGroup()

        self.start_service = self.create_service(StringTrigger, service_prefix+'start', self.start_service_callback, callback_group = self.service_group)
        self.pause_service = self.create_service(Trigger, service_prefix+'pause', self.pause_service_callback, callback_group = self.service_group)
        self.stop_service = self.create_service(Trigger, service_prefix+'stop', self.stop_service_callback, callback_group = self.service_group)
        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_service_callback, callback_group = self.service_group)
        self.select_mode_service = self.create_service(StringTrigger, service_prefix+'select_mode', self.select_mode_service_callback, callback_group = self.service_group)

        self.control_loop = self.create_timer(1/self.frequency, self.control_callback, callback_group=ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")
    
    def image_raw_callback(self, img):
        self.fill_int = None
        self.image_raw_msg = img
        self.image_raw = rnp.numpify(img)

    def desired_velocity_callback(self, msg):
        self.desired_vel = dict(linear = msg.linear.x, angular = msg.angular.z)
        self.received_desired_vel = True

    def termination_flag_callback(self, msg):
        self.termination_flag = msg.data
        self.received_termination_flag = True
    
    def action_controller_callback(self, msg):
        self.action_controller = ControllerType[msg.data]
        self.received_action_controller = True
    
    def start_service_callback(self, request, response):
        if(self.model == None):
            response.success = False
            response.message = "Model not selected"
        else:
            if(self.state == AgentState.STANDBY):
                self.wait = True
                demo_split = request.command.split('.')
                demo_name = demo_split[0]
                demo_id = demo_split[1]
                self.episode = self.demo_handler.get(demo_name, demo_id)
                observations, latent_vec = self.episode_to_model_input()
                self.get_logger().info(str(observations.shape)+ str(latent_vec.shape))
                # conditioning the model
                self.model(input = observations, input_latent = latent_vec, inference_mode = InferenceMode.STORE)
                self.received_desired_vel = True
                self.received_termination_flag = True
                self.received_action_controller = True

                self.get_logger().info(f"Selected demonstration: {request.command}")
                self.get_logger().info(f"Episode Length: {self.episode.get_episode_length()}")

                time.sleep(3.0)
                self.wait = False
            self.get_logger().info(f"Starting Task")
            response.success = True
            self.state = AgentState.RUNNING
        return response

    def pause_service_callback(self, request, response):
        if(self.state == AgentState.RUNNING or self.state == AgentState.PAUSED):
            self.state = AgentState.PAUSED
            self.wait = True
            time.sleep(2.0)
            self.wait = False
            response.success = True
        else:
            response.success = False
        return response

    def stop_service_callback(self, request, response):
        self.state = AgentState.STANDBY
        self.episode = EpisodeData(data=None)
        self.wait = True
        time.sleep(2.0)
        self.wait = False
        response.success = True
        return response

    def select_model_service_callback(self, request, response):
        self.get_logger().info("Selecting Model")
        if(self.state == AgentState.STANDBY):
            command_str = request.command
            if command_str == "":
                self.model = None
                response.success = True
                self.get_logger().info("Selected nothing")
            else:
                try:
                    self.get_logger().info("Split command")
                    command_split = request.command.split('/')
                    model_name = command_split[0]
                    self.get_logger().info(f"Model name : {model_name}")
                    model_id = command_split[1]
                    self.get_logger().info(f"Model ID : {model_id}")
                    self.select_model(model_name, model_id)
                    response.success = True
                    self.get_logger().info("Selected Model")
                except Exception as e:
                    self.get_logger().warn(f"Model selection failed, {e}")
                    response.success = False
                    response.message = str(e)
        else:
            response.message = "Model selection failed, agent is currently RUNNING"
            response.success = False
            self.get_logger().warn("Model selection failed, agent is currently RUNNING")

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
        image_raw = self.image_raw
        self.get_logger().info("Published agent_in")

        if(self.state == AgentState.RUNNING):
            self.get_logger().info("Verifying that previous actions are received")
            while(self.received_desired_vel == False or self.received_termination_flag == False or self.received_action_controller == False):
                pass

            action_controller = self.action_controller
            observation, latent_input = self.convert_to_model_input(image_raw, self.desired_vel, self.termination_flag, self.action_controller)

            self.received_desired_vel = False
            self.received_termination_flag = False
            self.received_action_controller = False
            self.get_logger().info("Computing agent_output")
            self.get_logger().info("Publishing agent_output")

            actions = self.model(input = observation, input_latent = latent_input, inference_mode = InferenceMode.STORE)
            agent_linear_vel = actions[0]
            agent_angular_vel = actions[1]
            agent_termination_flag = actions[2]
            if(agent_termination_flag >= 0.5):
                agent_termination_flag = True
            else: 
                agent_termination_flag = False

            agent_out = AgentOutput(velocity = Twist(linear=Vector3(x=agent_linear_vel,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=agent_angular_vel)), termination_flag = agent_termination_flag)

            self.agent_output_publisher.publish(agent_out)
            self.get_logger().info("Published agent_output")

            self.episode.append_data(
                image=PImage.fromarray(rnp.numpify(image_raw)),
                agent_linear_vel=agent_linear_vel,
                agent_angular_vel=agent_angular_vel,
                agent_termination_flag=agent_termination_flag,
                user_linear_vel=None,
                user_angular_vel=None,
                user_termination_flag=None,
                controller=action_controller
            )

        elif(self.state == AgentState.PAUSED):
            self.get_logger().info("Computing agent_output")
            self.get_logger().info("Publishing agent_output")

            #TODO: model inference + publish agent_output, information is NOT appended
            # observation, latent_input = self.convert_to_model_input(image_raw, self.desired_vel, self.termination_flag, self.action_controller)
            # actions = self.model(input = observation, input_latent = latent_input, inference_mode = InferenceMode.ONLY_LAST_FRAME)
            agent_out = AgentOutput(velocity = Twist(linear=Vector3(x=0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=0.0)))
            self.agent_output_publisher.publish(agent_out)
            self.get_logger().info("Published agent_output")
            pass
        else:
            pass

        self.get_logger().info(f"Current Episode Length: {self.episode.get_episode_length()}")

    def select_model(self, name, id_):
        self.model = self.model_handler.get(ModelType.ACTOR, name, id_)
    
    def convert_to_model_input(self, image_raw, desired_vel, termination_flag, action_controller):
        linear_vel = desired_vel['linear']
        angular_vel = desired_vel['angular']
        if(termination_flag == False):
            termination_flag = 0.0
        else:
            termination_flag = 1.0
        if(action_controller == ControllerType.USER):
            demo_flag = 1.0
        else:
            demo_flag = 0.0

        latent_vec = torch.tensor([linear_vel, angular_vel, termination_flag, demo_flag])
        observation = torch.tensor(image_raw).permute((2,0,1))
        return observation, latent_vec
    
    def episode_to_model_input(self):
        episode = self.episode.get_data()
        observations = torch.tensor(np.stack(episode['observation']['image'])).permute((2,0,1))
        controller = episode['action']['controller']
        agent_linear_vel = episode['action']['agent']['velocity']['linear']
        agent_angular_vel = episode['action']['agent']['velocity']['angular']
        agent_termination_flag = episode['action']['agent']['termination_flag']
        user_linear_vel = episode['action']['user']['velocity']['linear']
        user_angular_vel = episode['action']['user']['velocity']['angular']
        user_termination_flag = episode['action']['user']['termination_flag']

        desired_linear_vel = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_linear_vel, agent_linear_vel, controller)])
        desired_angular_vel = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_angular_vel, agent_angular_vel, controller)])
        desired_termination_flag = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_termination_flag, agent_termination_flag, controller)], dtype = torch.float32)
        demonstration_flag = torch.tensor([True if cont == ControllerType.USER else False for cont in controller], dtype = torch.float32)

        latent_vec = torch.stack([desired_linear_vel, desired_angular_vel, desired_termination_flag, demonstration_flag])

        return observations, latent_vec



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
