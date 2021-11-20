import rclpy
from rclpy.node import Node
from rclpy.qos import *
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import StringTrigger, FloatTrigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, Int32
from std_srvs.srv import Trigger
from geometry_msgs.msg import Vector3

from .utils import *
from mobile_robot_hl.episode_data import *

import ros2_numpy as rnp

import os
from PIL import Image as PImage
import json
import copy

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

        self.image_raw_msg = Image()
        self.agent_output = {'velocity':{'linear':0.0, 'angular': 0.0}, 'termination_flag':False}
        self.user_output =  {'velocity':{'linear':0.0, 'angular': 0.0}, 'termination_flag':False}
        self.episode = EpisodeData()
        self.selected_data = dict(type = InformationType.NONE, name = "", id = "")
        self.frame_no = 0
        self.received_agent_velocity = False

        self.state = SupervisorState.STANDBY
        self.controller = ControllerType.USER
        
        self.get_logger().info("Initializing Node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('frequency', 0.6),
                ('max_linear_velocity', None),
                ('max_angular_velocity', None),
            ])

        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value

        self.get_logger().info(f"Parameter <frequency> = {self.frequency}")
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
        self.task_image_publisher = self.create_publisher(Image, 'task_image', reliable_qos, callback_group = ReentrantCallbackGroup())
        self.frame_no_publisher = self.create_publisher(Int32, 'frame_no', reliable_qos, callback_group = ReentrantCallbackGroup())
        self.supervisor_state_publisher = self.create_publisher(String, 'supervisor_state', best_effort_qos, callback_group = ReentrantCallbackGroup())

        self.image_raw_subscriber = self.create_subscription(Image, image_raw_topic_name, self.image_raw_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.agent_velocity_subscriber = self.create_subscription(Twist, 'agent_velocity', self.agent_velocity_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.user_velocity_subscriber = self.create_subscription(Twist, 'user_velocity', self.user_velocity_callback, best_effort_qos, callback_group = ReentrantCallbackGroup())

        agent_prefix = "agent/"
        supervisor_prefix='supervisor/'

        self.services_ = dict()

        self.supervisor_service_group = ReentrantCallbackGroup()
        self.services_[supervisor_prefix+'start'] = self.create_service(StringTrigger, supervisor_prefix+'start', self.start_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'pause'] = self.create_service(Trigger, supervisor_prefix+'pause', self.pause_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'stop'] = self.create_service(Trigger, supervisor_prefix+'stop', self.stop_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'save'] = self.create_service(Trigger, supervisor_prefix+'save', self.save_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'termination_flag'] = self.create_service(StringTrigger, supervisor_prefix+'termination_flag', self.termination_flag_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'select_data'] = self.create_service(StringTrigger, supervisor_prefix+'select_data', self.select_data_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'select_controller'] = self.create_service(StringTrigger, supervisor_prefix+'select_controller', self.select_controller_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'configure_disturbance'] = self.create_service(FloatTrigger, supervisor_prefix+'configure_disturbance', self.configure_disturbance_callback, callback_group=self.supervisor_service_group)

        self.agent_service_group = ReentrantCallbackGroup()
        self.services_[agent_prefix+'select_data'] = self.create_client(StringTrigger, agent_prefix+'select_data', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'select_model'] = self.create_client(StringTrigger, agent_prefix+'select_model', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'reset_model'] = self.create_client(Trigger, agent_prefix+'reset_model', callback_group=self.agent_service_group)

        self.control_loop = self.create_timer(1/self.frequency, self.control_callback, callback_group=ReentrantCallbackGroup())
        self.state_publish_loop = self.create_timer(0.1, self.publish_state, callback_group=ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")

    # SUBSCRIPTION CALLBACKS

    def image_raw_callback(self, img):
        self.image_raw_msg = img

    def agent_velocity_callback(self, vel):
        self.agent_velocity['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
        self.received_agent_velocity = True

    def user_velocity_callback(self, vel):
        self.user_output['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
        if(self.state == SupervisorState.STANDBY):
            velocity_msg = Twist(linear=Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(velocity_msg)

    ### SERVICE CALLBACKS

    def start_callback(self, request, response):
        # check type of start (task vs demo)
        start_type = request.data
        if(start_type == 'task'):
            # state == standby
            if(self.state == SupervisorState.STANDBY):
                # check if episode has been selected
                if(self.selected_data['name'] == None or self.selected_data['id'] == None):
                    response.success = False
                    response.message = "Episode not yet selected"
                    return response
                # call agent service to select demo
                selected_data = copy.deepcopy(self.selected_data)
                selected_data['type'] = selected_data['type'].name
                select_data_response = self.call_service('agent/select_data', json.dumps(selected_data))
                if(select_data_response == False):
                    response.success = False
                    response.message = "agent/select_data service not successful"
                    return response
                # set self.episode
                try:
                    if(self.selected_data['type'] == InformationType.TASK_EPISODE):
                        self.episode = self.task_handler.get(self.selected_data['name'], self.selected_data['id'])
                    elif(self.selected_data['type'] == InformationType.DEMO):
                        self.episode = self.demo_handler.get(self.selected_data['name'], self.selected_data['id'])
                    else:
                        raise Exception()
                except:
                    response.success = False
                    response.message = f"Unable to get episode {self.selected_data}"
                    return response
            
            # state == task_paused or task_running
            elif(self.state in [SupervisorState.TASK_PAUSED, SupervisorState.TASK_RUNNING]):
                # do nothing
                pass
            # state == any other state => fail
            else:
                response.success = False
                response.message = f"Unable to start task as the current state == {self.state.name}"
                return response

            self.controller = ControllerType.AGENT
            self.state = SupervisorState.TASK_RUNNING

        elif(start_type == 'demo'):
            if(self.state == SupervisorState.STANDBY):
                # check if demo name has been selected
                if(self.selected_data['name'] == None or self.selected_data['type'] == InformationType.TASK_EPISODE):
                    response.success = False
                    response.message = "Demo name not yet selected"
                    return response
                # if the ID is selected, then set self.episode
                if(self.selected_data['id'] != None):
                    try:
                        self.episode = self.demo_handler.get(self.selected_data['name'], self.selected_data['id'])
                    except:
                        response.success = False
                        response.message = f"Unable to get episode {self.selected_data}"
                        return response
                self.state = SupervisorState.DEMO_RECORDING
                self.controller = ControllerType.USER
            elif(self.state == SupervisorState.DEMO_PAUSED):
                self.state = SupervisorState.DEMO_RECORDING
                self.controller = ControllerType.USER
            else:
                response.success = False
                response.message = f"Unable to start demo as the current state == {self.state.name}"
                return response
        else:
            response.success = False
            response.message = f"Invalid start type"
            return response

        response.success = True
        return response

    def pause_callback(self, request, response):
        try:
            self.pause()
        except Exception as e:
            response.success = False
            response.message = str(e)
            return response
        response.success = True
        return response
    
    def pause(self):
        if(self.state in [SupervisorState.TASK_RUNNING]):
            self.state = SupervisorState.TASK_PAUSED
        elif(self.state == SupervisorState.DEMO_RECORDING):
            self.state = SupervisorState.DEMO_PAUSED
        else:
            raise Exception(f"Unable to pause as state == {self.state}")

    def stop_callback(self, request, response):
        self.episode.reset()
        self.selected_data = dict(string = "", type = InformationType.NONE, name = "", id = "")
        self.frame_no = 0
        self.state = SupervisorState.STANDBY
        response.success = True
        return response

    def save_callback(self, request, response):
        if(self.state == SupervisorState.STANDBY):
            response.success = False
            response.message = f"Unable to save episode because state == {self.state}"
            return response

        # make sure episode is not empty
        if(self.episode.length() != 0):
            if(self.state in [SupervisorState.TASK_RUNNING, SupervisorState.TASK_PAUSED]):
                self.state = SupervisorState.TASK_PAUSED
                self.task_handler.save(self.episode, self.selected_data['name'], self.selected_data['id'])
            
            elif(self.state in [SupervisorState.DEMO_PAUSED, SupervisorState.DEMO_RECORDING]):
                self.state = SupervisorState.DEMO_PAUSED
                self.demo_handler.save(self.episode, self.selected_data['name'])
        else:
            response.success = False
            response.message = f"Unable to save episode as episode is empty"
            return response
        
        response.success = True
        return response

    def termination_flag_callback(self, request, response):
        requestor = ControllerType[request.data]

        if(requestor == ControllerType.AGENT):
            self.pause()
            self.agent_output['termination_flag'] = True
        elif(requestor == ControllerType.USER):
            self.user_output['termination_flag'] = True
        else:
            response.success = False
            response.message = f"Invalid requestor {requestor}"
            return response

        response.success = True
        return response

    def select_data_callback(self, request, response):
        try:
            data_dict = json.loads(request.data)
            self.selected_data = dict(type = InformationType.NONE, name = "", id = "")
            for key, value in data_dict.items():
                if(key == 'type'):
                    value = InformationType[value]
                self.selected_data[key] = value
        except Exception as e:
            response.success = False
            response.message = str(e)
            return response
        response.success = True
        return response

    def select_controller_callback(self, request, response):
        if(self.state in [SupervisorState.TASK_PAUSED, SupervisorState.TASK_RUNNING]):
            try:
                self.controller = ControllerType[request.data.upper()]
            except Exception as e:
                response.success = False
                response.message = str(e)
                return response
        else:
            response.success = False
            response.message = f"Unable to select controller as state == {self.state.name}"
            return response
        
        response.success = True
        return response

    def configure_disturbance_callback(self, request, response):
        pass

    ### TIMER CALLBACK

    def control_callback(self):
        # state = DEMO_RECORDING, TASK_RUNNING
        state = copy.deepcopy(self.state)
        if(state in [SupervisorState.DEMO_RECORDING, SupervisorState.TASK_RUNNING]):
            # 1. send image_raw_msg to agent 
            frame_no = self.frame_no + 1
            desired_output = dict(velocity = dict(linear = None, angular = None), termination_flag = None)

            # publish image and wait for agent output
            if(self.received_agent_velocity == False):
                self.task_image_msg = self.image_raw_msg
                self.task_image_publisher.publish(self.task_image_msg)
                self.frame_no_publisher.publish(Int32(data=frame_no))

                while(self.received_agent_velocity == False):
                    pass

            agent_output = self.agent_output
            self.received_agent_velocity = False

            # convert image to PImage to be appended to the episode
            image = PImage.fromarray(rnp.numpify(self.task_image_msg))

            # get user output
            user_output = self.user_output
            
            # set desired_output based on self.state and termination flags
            if(self.state == SupervisorState.DEMO_RECORDING):
                agent_output = dict(velocity = dict(linear = None, angular = None), terminatinon_flag = None)
                desired_output = user_output
            else:
                if(user_output['termination_flag'] == True and agent_output['terminatinon_flag'] == False):
                    self.controller = ControllerType.USER

                if(self.controller == ControllerType.USER):
                    desired_output = user_output
                elif(self.controller == ControllerType.AGENT):
                    desired_output = agent_output
                else:
                    raise Exception("Invalid controller when recording demo or running task")

            # get final decision on the controller
            controller = self.controller
            
            # if the state changed during this process, then cancel this frame
            if(self.state != state):
                self.received_agent_velocity = True
                return
            else:
                # publish desired velocity, desired termination flag, and action controller
                desired_velocity_msg = Twist(linear=Vector3(x=desired_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=desired_output['velocity']['angular']))
                self.desired_velocity_publisher.publish(desired_velocity_msg)
                termination_flag_msg = Bool(data=desired_output['termination_flag'])
                self.termination_flag_publisher.publish(termination_flag_msg)
                action_controller_msg = String(data=controller.name)
                self.action_controller_publisher.publish(action_controller_msg)
                # append frame to self.episode
                episode_frame = EpisodeData(
                    observation = dict(image=image), 
                    action = dict(
                        user = user_output,
                        agent = agent_output,
                        controller = controller), 
                    frame_no = frame_no)
                self.episode.append(episode_frame)

                # reset termination_flag
                self.user_output['termination_flag'] = False
                self.agent_output['termination_flag'] = False
                # pause and reset frame_no if termination_flag was raised
                if(desired_output['termination_flag'] == True):
                    self.frame_no = 0
                    self.pause()
                else:
                    self.frame_no = frame_no

    def publish_state(self):
        state = json.dumps(dict(state = self.state.name, controller = self.controller.name))
        msg = String(data=state)
        self.supervisor_state_publisher.publish(msg)

    ### UTILS
    def call_service(self, service_name, command=None):
        self.get_logger().info(f'calling service {service_name}.')
        if self.services_[service_name].wait_for_service(timeout_sec=0.1) == False:
            self.get_logger().info(f'{service_name} service not available')
            return False
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
            return response.success
    
def main():
    rclpy.init()

    node = SupervisorNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)

if __name__ == '__main__':
    main()