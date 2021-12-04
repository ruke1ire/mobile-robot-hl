import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.srv import StringTrigger, FloatTrigger
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
import gc

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
        self.max_linear_velocity = float(os.environ['MOBILE_ROBOT_HL_MAX_LINEAR_VEL'])
        self.max_angular_velocity = float(os.environ['MOBILE_ROBOT_HL_MAX_ANGULAR_VEL'])
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
        self.state = dict(state = SupervisorState.STANDBY, controller = ControllerType.USER)
        self.selected_data = None

        self.noise = 0.0

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
        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_callback, callback_group = ReentrantCallbackGroup())
        self.reset_model_service = self.create_service(Trigger, service_prefix+'reset_model', self.reset_model_callback, callback_group = self.service_group)
        self.configure_disturbance_service = self.create_service(FloatTrigger, service_prefix+'configure_disturbance', self.configure_disturbance_callback, callback_group=ReentrantCallbackGroup())

        self.termination_flag_client = self.create_client(StringTrigger, 'supervisor/termination_flag', callback_group = ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")
    
    # SUBSCRIBER CALLBACKS

    def task_image_callback(self, img):
        try:
            if(self.state['state'] == SupervisorState.TASK_RUNNING):
                self.get_logger().debug("Received task_image")
                self.task_image = rnp.numpify(img)
                # check whether "demo" and model has been selected otherwise fail
                if(self.model == None and self.selected_data == None):
                    self.get_logger().warn("Model or data not yet selected")
                    return
                
                # 1. Verify previous actions and frame_no are received
                self.get_logger().debug("Receiving previous actions and frame_no")
                while(self.received_desired_vel == False or self.received_termination_flag == False or self.received_action_controller == False or self.received_frame_no == False):
                    pass
                    #raise Exception("Task failed, previous actions unknown")

                self.get_logger().debug("Received previous actions and frame_no")
                self.received_desired_vel = False
                self.received_termination_flag = False
                self.received_action_controller = False
                self.received_frame_no = False

                prev_vel = self.desired_vel
                prev_termination_flag = self.desired_termination_flag
                prev_action_controller = self.action_controller
                frame_no = self.frame_no

                # 2. Convert information to tensor
                self.get_logger().debug("Converting information to tensors")
                image_tensor, latent_tensor, frame_no_tensor = self.convert_to_model_input(self.task_image, prev_vel, prev_termination_flag, prev_action_controller, frame_no)
                
                # 3. Inference and processing
                self.get_logger().debug(f"Computing model output with noise = {self.noise}")
                output_tensor = self.model(input = image_tensor, input_latent = latent_tensor, frame_no = frame_no_tensor, noise = self.noise, inference_mode = InferenceMode.STORE)

                # 4. Run model post processing to convert model output to appropriate values
                agent_linear_vel = output_tensor[0].item()
                agent_angular_vel = output_tensor[1].item()
                agent_termination_flag = output_tensor[2].item()

                # 5. Call supervisor/termination_flag if raised
                if(agent_termination_flag >= 0.5):
                    service_request = StringTrigger.Request()
                    service_request.command = "agent"
                    self.termination_flag_client.call(service_request)
                        
                # 6. Publish agent output
                self.get_logger().debug("Publishing agent velocity")
                velocity_msg = Twist(linear=Vector3(x=agent_linear_vel,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=agent_angular_vel))
                self.agent_velocity_publisher.publish(velocity_msg)
                self.get_logger().info("Completed a control frame")

        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))

    def desired_velocity_callback(self, msg):
        try:
            self.desired_vel = dict(linear = msg.linear.x, angular = msg.angular.z)
            self.received_desired_vel = True
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))

    def termination_flag_callback(self, msg):
        try:
            self.desired_termination_flag = msg.data
            self.received_termination_flag = True
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))
    
    def action_controller_callback(self, msg):
        try:
            self.action_controller = ControllerType[msg.data]
            self.received_action_controller = True
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))
    
    def frame_no_callback(self, msg):
        try:
            self.frame_no = msg.data
            self.received_frame_no = True
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))
    
    def supervisor_state_callback(self, msg):
        try:
            state = json.loads(msg.data)
            state['state'] = SupervisorState[state['state']]
            state['controller'] = ControllerType[state['controller']]
            self.state = state
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))
    
    # SERVICE CALLBACKS

    def select_data_callback(self, request, response):
        try:
            self.get_logger().info("<select_data> service called")

            # if model = None, fail
            if(self.model == None):
                raise Exception("Data selection failed as model is not yet selected")

            self.get_logger().debug("Resetting model")
            # reset model and prev_actions
            self.model.reset()
            self.reset_variables()
            gc.collect()
            # get data from demo handler or task handler
            self.get_logger().debug("Retrieving data")
            selected_data = json.loads(request.command)
            selected_data['type'] = InformationType[selected_data['type']]
            if(selected_data['type'] == InformationType.TASK_EPISODE):
                episode = self.task_handler.get(selected_data['name'], selected_data['id'])
            elif(selected_data['type'] == InformationType.DEMO):
                episode = self.demo_handler.get(selected_data['name'], selected_data['id'])
            else:
                raise Exception("Invalid request. Unable to select data")
            self.get_logger().debug("Retrieved data")

            self.get_logger().debug("Conditioning model")
            # condition model on data
            image_tensor, latent_tensor, frame_no_tensor = episode.get_tensor()

            initial_action = torch.zeros_like(latent_tensor[:,0])
            initial_action[3] = 1.0
            prev_latent = torch.cat((initial_action.unsqueeze(1), latent_tensor[:,:-1]), dim = 1)
            self.model(input = image_tensor, input_latent = prev_latent, frame_no = frame_no_tensor, inference_mode = InferenceMode.STORE)

            self.desired_vel = dict(linear = latent_tensor[0,-1].item(), angular = latent_tensor[1,-1].item())
            if(latent_tensor[2, -1] >= 0.5):
                self.desired_termination_flag = True
            else:
                self.desired_termination_flag = False
            if(latent_tensor[3, -1] == 1.0):
                self.action_controller = ControllerType.USER
            else:
                self.action_controller = ControllerType.AGENT
            
            self.received_desired_vel = True
            self.received_termination_flag = True
            self.received_action_controller = True

            self.get_logger().debug("Conditioned model")
            self.selected_data = selected_data
            self.get_logger().info("<select_data> service completed")
            response.success = True
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    def select_model_callback(self, request, response):
        try:
            self.get_logger().info("<select_model> service called")
            if(self.state['state'] == SupervisorState.STANDBY):
                command_dict = json.loads(request.command)
                model_name = command_dict['name']
                model_id = command_dict['id']
                self.get_logger().debug(f"Model name : {model_name}")
                self.get_logger().debug(f"Model ID : {model_id}")
                self.select_model(model_name, model_id)
                self.model_name = model_name
                self.model_id = model_id
                self.reset_variables()
            else:
                raise Exception("Model selection failed, agent is currently RUNNING")

            response.success = True
            self.get_logger().info("<select_model> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            self.model = None
            self.model_name = None
            self.model_id = None
            response.message = str(e)
            response.success = False
            return response
    
    def reset_model_callback(self, request, response):
        try:
            self.get_logger().info("<reset_model> service called")
            if(self.model_name is None or self.model_id is None or self.model is None):
                response.message = "Unable to reset model as model is not yet selected"
                response.success = False
            else:
                self.model.reset()
                self.reset_variables()
            response.success = True
            self.get_logger().info("<reset_model> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    def configure_disturbance_callback(self, request, response):
        self.get_logger().info("Configuring disturbance")
        if(self.state['state'] == SupervisorState.STANDBY):
            self.noise = request.command
            self.get_logger().info(f"Got noise value of {self.noise}")
            response.success  = True
            self.get_logger().info("Configured disturbance")
        else:
            response.success  = False
            response.message  = "Unable to configure disturbance"
            self.get_logger().warn("Unable to configure disturbance")
        return response

    # UTILS
    def reset_variables(self):
        self.desired_vel = dict(linear = 0.0, angular = 0.0)
        self.desired_termination_flag = False
        self.action_controller = ControllerType.USER
        self.frame_no = 0
        self.received_desired_vel = False
        self.received_termination_flag = False
        self.received_action_controller = False
        self.received_frame_no = False

    def select_model(self, name, id_):
        self.get_logger().info("Selecting model")
        self.model, model_info = self.model_handler.get(ModelType.ACTOR, name, id_)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.get_logger().info("Selected model")

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
        image_tensor = torch.tensor(image_raw, dtype = torch.float32).permute((2,0,1))/255.0 - 0.5
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
