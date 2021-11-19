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
import numpy as np
import traceback
import json

import ros2_numpy as rnp

from mobile_robot_hl.model.model_handler import ModelHandler
from mobile_robot_hl.episode_data import *
from mobile_robot_hl.model.utils import *
from mobile_robot_hl.utils import *

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

        self.reset_variables()

        self.model = None
        self.model_name = None
        self.model_id = None
        self.state = SupervisorState.STANDBY
        self.selected_data = None

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
        self.frame_no_subscriber = self.create_subscription(Int32, 'frame_no', self.frame_no_callback, reliable_qos, callback_group=ReentrantCallbackGroup())
        self.supervisor_state_subscriber = self.create_subscription(String, 'supervisor_state', self.supervisor_state_callback, best_effort_qos, callback_group=ReentrantCallbackGroup())

        self.agent_velocity_publisher = self.create_publisher(Twist, 'agent_velocity', reliable_qos, callback_group=ReentrantCallbackGroup())

        service_prefix = 'agent/'
        self.service_group = ReentrantCallbackGroup()

        self.select_data_service = self.create_service(StringTrigger, service_prefix+'select_data', self.select_data_callback, callback_group = self.service_group)
        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_callback, callback_group = self.service_group)
        self.reset_model_service = self.create_service(Trigger, service_prefix+'reset_model', self.reset_model_callback, callback_group = self.service_group)

        self.termination_flag_client = self.create_client(StringTrigger, 'supervisor/termination_flag', callback_group = ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")
    
    # SUBSCRIBER CALLBACKS

    def task_image_callback(self, img):
        self.task_image = rnp.numpify(img)
        # check whether "demo" and model has been selected otherwise fail
        if(self.model == None and self.selected_data == None):
            self.get_logger().warn("Model or data not yet selected")
            return
        
        # 1. Verify previous actions and frame_no are received
        while(self.received_desired_vel == False or self.received_termination_flag == False or self.received_action_controller == False or self.received_frame_no == False):
            pass

        self.received_desired_vel = False
        self.received_termination_flag = False
        self.received_action_controller = False
        self.received_frame_no = False

        prev_vel = self.desired_vel
        prev_termination_flag = self.desired_termination_flag
        prev_action_controller = self.action_controller
        frame_no = self.frame_no

        # 2. Convert information to tensor
        image_tensor, latent_tensor, frame_no_tensor = self.convert_to_model_input(self.task_image, prev_vel, prev_termination_flag, prev_action_controller, frame_no)
        
        # 3. Inference and processing
        output_tensor = self.model(input = image_tensor, input_latent = latent_tensor, frame_no = frame_no_tensor, inference_mode = InferenceMode.STORE)
        output_tensor = process_actor_output(output_tensor)

        # 4. Run model post processing to convert model output to appropriate values
        agent_linear_vel = output_tensor[0].item()
        agent_angular_vel = output_tensor[1].item()
        agent_termination_flag = output_tensor[2].item()

        # 5. Call supervisor/termination_flag if raised
        if(agent_termination_flag >= 0.5):
            self.termination_flag_client.call(Trigger.Request())
                
        # 6. Publish agent output
        velocity_msg = Twist(linear=Vector3(x=agent_linear_vel,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=agent_angular_vel))
        self.agent_velocity_publisher.publish(velocity_msg)

    def desired_velocity_callback(self, msg):
        self.desired_vel = dict(linear = msg.linear.x, angular = msg.angular.z)
        self.received_desired_vel = True

    def termination_flag_callback(self, msg):
        self.desired_termination_flag = msg.data
        self.received_termination_flag = True
    
    def action_controller_callback(self, msg):
        self.action_controller = ControllerType[msg.data]
        self.received_action_controller = True
    
    def frame_no_callback(self, msg):
        self.frame_no = msg.data
        self.received_frame_no = True
    
    def supervisor_state_callback(self, msg):
        state = json.loads(msg.data)
        state['state'] = SupervisorState[state['state']]
        state['controller'] = ControllerType[state['controller']]
        self.state = state
    
    # SERVICE CALLBACKS

    def select_data_callback(self, request, response):
        # if model = None, fail
        if(self.model == None):
            response.success = False
            response.message = "Data selection failed as model is not yet selected"
            return response
        try:
            # reset model and prev_actions
            self.model.reset()
            self.reset_variables()
            # get data from demo handler or task handler
            selected_data = json.loads(request.data)
            selected_data['type'] = InformationType[selected_data['type']]
            if(selected_data['type'] == InformationType.TASK_EPISODE):
                episode = self.task_handler.get(selected_data['name'], selected_data['id'])
            elif(self.selected_data['type'] == InformationType.DEMO):
                episode = self.demo_handler.get(selected_data['name'], selected_data['id'])
            # condition model on data
            image_tensor, latent_tensor, frame_no_tensor = episode.get_tensor()
            self.model(input = image_tensor, input_latent = latent_tensor, frame_no = frame_no_tensor, inference_mode = InferenceMode.STORE)
        except Exception:
            self.selected_data = None
            response.success = False
            response.message = str(traceback.format_exc())
            return response
        self.selected_data = selected_data
        response.success = True
        return response

    def select_model_callback(self, request, response):
        self.get_logger().info("Selecting Model")
        if(self.state == SupervisorState.STANDBY):
            command_dict = json.loads(request.command)
            model_name = command_dict['name']
            model_id = command_dict['id']
            self.get_logger().info(f"Model name : {model_name}")
            self.get_logger().info(f"Model ID : {model_id}")
            try:
                self.select_model(model_name, model_id)
                self.model_name = model_name
                self.model_id = model_id
                self.get_logger().info("Selected Model")
            except Exception:
                self.get_logger().warn(f"Model selection failed, {traceback.format_exc()}")
                self.model = None
                self.model_name = None
                self.model_id = None
                response.success = False
                response.message = str(traceback.format_exc())
                return response
            self.reset_variables()
        else:
            response.message = "Model selection failed, agent is currently RUNNING"
            response.success = False
            self.get_logger().warn("Model selection failed, agent is currently RUNNING")
            return response

        response.success = True
        return response
    
    def reset_model_callback(self, request, response):
        if(self.model_name is None or self.model_id is None or self.model is None):
            response.message = "Unable to reset model as model is not yet selected"
            response.success = False
        else:
            self.model.reset()
            self.reset_variables()

    # UTILS
    def reset_variables(self):
        self.desired_vel = dict(linear = 0.0, angular = 0.0)
        self.desired_termination_flag = False
        self.action_controller = ControllerType.USER
        self.frame_no = 0
        self.received_desired_vel = True
        self.received_termination_flag = True
        self.received_action_controller = True
        self.received_frame_no = False

    def select_model(self, name, id_):
        self.model, model_info = self.model_handler.get(ModelType.ACTOR, name, id_)
        self.model = self.model.to(self.device)
    
    def convert_to_model_input(self, image_raw, velocity, termination_flag, action_controller, frame_no):
        linear_vel = velocity['linear']
        angular_vel = velocity['angular']
        if(termination_flag == False):
            termination_flag = 0.0
        else:
            termination_flag = 1.0
        if(action_controller == ControllerType.USER):
            demo_flag = 1.0
        else:
            demo_flag = 0.0

        latent_tensor = torch.tensor([linear_vel, angular_vel, termination_flag, demo_flag], dtype = torch.float32)
        image_tensor = torch.tensor(image_raw, dtype = torch.float32).permute((2,0,1))/255-0.5
        frame_no_tensor = torch.tensor(frame_no, dtype = torch.float32)

        return image_tensor, latent_tensor, frame_no_tensor
    
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
