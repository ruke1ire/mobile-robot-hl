import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.msg import AgentOutput
from custom_interfaces.srv import StringTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool, String, Int32
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import threading
import os
from enum import Enum
import numpy as np
import time
import traceback
from PIL import Image as PImage

import ros2_numpy as rnp

from .utils import *
from .model.model_handler import ModelHandler
from .episode_data import *
from .model.utils import *

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
        
        self.demo_handler = DemoHandler(path=demo_path)
        self.task_handler = TaskHandler(path=task_path, demo_handler = self.demo_handler)
        self.model_handler = ModelHandler(path=model_path)

        self.model_name = None
        self.model_id = None

        self.received_desired_vel = False
        self.received_termination_flag = False
        self.received_action_controller = False
        self.received_frame_no = False

        self.fill_int = 255
        img_tmp = np.zeros([240,320,3],dtype=np.uint8)
        img_tmp.fill(self.fill_int)
        self.image_raw = img_tmp
        self.image_raw_msg = rnp.msgify(Image,img_tmp, encoding='rgb8')
        self.desired_vel = dict(linear = 0.0, angular = 0.0)
        self.termination_flag = False
        self.frame_no = 0

        self.state = AgentState.STANDBY

        self.model = None

        self.get_logger().info("Initializing Node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', 'cpu'),
            ])

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.get_logger().info(f"Parameter <device> = {self.device}")

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.task_image_subscriber = self.create_subscription(Image, 'task_image', self.task_image_callback, best_effort_qos, callback_group=ReentrantCallbackGroup())
        self.desired_velocity_subscriber = self.create_subscription(Twist, desired_velocity_topic_name, self.desired_velocity_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.termination_flag_subscriber = self.create_subscription(Bool, 'termination_flag', self.termination_flag_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.action_controller_subscriber = self.create_subscription(String, 'action_controller', self.action_controller_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.frame_no_subscriber = self.create_subscription(Int32, 'action_controller', self.frame_no_callback, reliable_qos, callback_group=ReentrantCallbackGroup())

        self.agent_output_publisher = self.create_publisher(AgentOutput, 'agent_output', reliable_qos, callback_group=ReentrantCallbackGroup())

        service_prefix = 'agent/'
        self.service_group = ReentrantCallbackGroup()

        self.select_data_service = self.create_service(StringTrigger, service_prefix+'select_data', self.select_data_callback, callback_group = self.service_group)
        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_callback, callback_group = self.service_group)
        self.reset_model_service = self.create_service(Trigger, service_prefix+'reset_model', self.reset_model_callback, callback_group = self.service_group)

        self.get_logger().info("Initialized Node")
    
    # SUBSCRIBER CALLBACKS

    def task_image_callback(self, img):
        self.fill_int = None
        self.task_image = rnp.numpify(img)
        # check whether "demo" and model has been selected otherwise fail
        # TODO: Model inference here

    def desired_velocity_callback(self, msg):
        self.desired_vel = dict(linear = msg.linear.x, angular = msg.angular.z)
        self.received_desired_vel = True

    def termination_flag_callback(self, msg):
        self.termination_flag = msg.data
        self.received_termination_flag = True
    
    def action_controller_callback(self, msg):
        self.action_controller = ControllerType[msg.data]
        self.received_action_controller = True
    
    def frame_no_callback(self, msg):
        self.frame_no = msg.data
        self.received_frame_no = True
    
    # SERVICE CALLBACKS

    def select_data_callback(self, request, response):
        # if model = None, fail
        # get data from demo handler or task handler
        # condition model on data
        pass

    def select_model_callback(self, request, response):
        self.get_logger().info("Selecting Model")
        if(self.state == AgentState.STANDBY):
            command_str = request.command
            if command_str == "":
                self.model = None
                self.model_name = None
                self.model_id = None
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
                    self.model_name = model_name
                    self.model_id = model_id
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
    
    def reset_model_callback(self, request, response):
        if(self.model_name is None or self.model_id is None or self.model is None):
            response.message = "Unable to reset model as model is not yet selected"
            response.success = False
        else:
            self.select_model(self.model_name, self.model_id)

#    def control_callback(self):
#        if(self.wait == True):
#            return
#        self.get_logger().info("Publishing agent_in")
#        if(self.fill_int != None):
#            img_tmp = np.zeros([240,320,3],dtype=np.uint8)
#            if(self.fill_int == 255):
#                self.fill_int = 0
#            else:
#                self.fill_int = 255
#            img_tmp.fill(self.fill_int)
#            self.image_raw_msg = rnp.msgify(Image,img_tmp,encoding='rgb8')
#
#        msg = self.image_raw_msg
#        self.agent_input_publisher.publish(msg)
#        image_raw = self.image_raw
#        self.get_logger().info("Published agent_in")
#
#        if(self.state == AgentState.RUNNING):
#            self.get_logger().info("Verifying that previous actions are received")
#            while(self.received_desired_vel == False or self.received_termination_flag == False or self.received_action_controller == False):
#                pass
#
#            try:
#                action_controller = self.action_controller
#                observation, latent_input = self.convert_to_model_input(image_raw, self.desired_vel, self.termination_flag, self.action_controller)
#
#                self.received_desired_vel = False
#                self.received_termination_flag = False
#                self.received_action_controller = False
#                self.get_logger().info("Computing agent_output")
#                self.get_logger().info("Publishing agent_output")
#
#                actions = self.model(input = observation, input_latent = latent_input, inference_mode = InferenceMode.STORE)
#                agent_linear_vel = actions[0].item()
#                agent_angular_vel = actions[1].item()
#                agent_termination_flag = actions[2].item()
#                if(agent_termination_flag >= 0.5):
#                    agent_termination_flag = True
#                else: 
#                    agent_termination_flag = False
#                
#
#                agent_out = AgentOutput(velocity = Twist(linear=Vector3(x=agent_linear_vel,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=agent_angular_vel)), termination_flag = agent_termination_flag)
#
#                self.agent_output_publisher.publish(agent_out)
#                self.get_logger().info("Published agent_output")
#
#                self.episode.append_data(
#                    image=PImage.fromarray(image_raw),
#                    agent_linear_vel=agent_linear_vel,
#                    agent_angular_vel=agent_angular_vel,
#                    agent_termination_flag=agent_termination_flag,
#                    user_linear_vel=0.0,
#                    user_angular_vel=0.0,
#                    user_termination_flag=False,
#                    controller=action_controller
#                )
#
#            except:
#                self.get_logger().info(str(traceback.format_exc()))
#
#        elif(self.state == AgentState.PAUSED):
#            self.get_logger().info("Computing agent_output")
#            self.get_logger().info("Publishing agent_output")
#
#            #TODO: model inference + publish agent_output, information is NOT appended
#            # observation, latent_input = self.convert_to_model_input(image_raw, self.desired_vel, self.termination_flag, self.action_controller)
#            # actions = self.model(input = observation, input_latent = latent_input, inference_mode = InferenceMode.ONLY_LAST_FRAME)
#            agent_out = AgentOutput(velocity = Twist(linear=Vector3(x=0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=0.0)))
#            self.agent_output_publisher.publish(agent_out)
#            self.get_logger().info("Published agent_output")
#            pass
#        else:
#            pass
#
#        self.get_logger().info(f"Current Episode Length: {self.episode.get_episode_length()}")

    # UTILS

    def select_model(self, name, id_):
        self.model, model_info = self.model_handler.get(ModelType.ACTOR, name, id_)
        self.model = self.model.to(self.device)
    
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
        observation = torch.tensor(image_raw, dtype = torch.float32).permute((2,0,1))/255-0.5


        return observation, latent_vec
    
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
