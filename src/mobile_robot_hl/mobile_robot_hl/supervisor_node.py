from numpy.core.fromnumeric import trace
import rclpy
from rclpy.node import Node
from rclpy.qos import *
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import StringTrigger, FloatTrigger
from custom_interfaces.msg import EpisodeFrame
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
import traceback
import numpy as np

class SupervisorNode(Node):

    def __init__(self):
        super().__init__('supervisor')

        demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']
        self.max_linear_velocity = float(os.environ['MOBILE_ROBOT_HL_MAX_LINEAR_VEL'])
        self.max_angular_velocity = float(os.environ['MOBILE_ROBOT_HL_MAX_ANGULAR_VEL'])
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

        self.image_raw_msg = rnp.msgify(Image,np.zeros((240,320,3),dtype = np.uint8), 'rgb8')
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
                ('frequency', 0.3),
                ('max_episode_length', 100),
            ])

        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.max_episode_length = self.get_parameter('max_episode_length').get_parameter_value().integer_value

        self.get_logger().info(f"Parameter <frequency> = {self.frequency}")
        self.get_logger().info(f"Parameter <max_episode_length> = {self.max_episode_length}")

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.desired_velocity_publisher = self.create_publisher(Twist, desired_velocity_topic_name, reliable_qos, callback_group= ReentrantCallbackGroup())
        self.episode_frame_publisher = self.create_publisher(EpisodeFrame, 'episode_frame', reliable_qos, callback_group = ReentrantCallbackGroup())
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

        self.agent_service_group = ReentrantCallbackGroup()
        self.services_[agent_prefix+'select_data'] = self.create_client(StringTrigger, agent_prefix+'select_data', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'select_model'] = self.create_client(StringTrigger, agent_prefix+'select_model', callback_group=self.agent_service_group)
        self.services_[agent_prefix+'reset_model'] = self.create_client(Trigger, agent_prefix+'reset_model', callback_group=self.agent_service_group)

        self.control_loop = self.create_timer(1/self.frequency, self.control_callback, callback_group=ReentrantCallbackGroup())
        self.state_publish_loop = self.create_timer(0.1, self.publish_state, callback_group=ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")

    # SUBSCRIPTION CALLBACKS

    def image_raw_callback(self, img):
        try:
            self.image_raw_msg = img
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))

    def agent_velocity_callback(self, vel):
        try:
            self.agent_output['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
            self.received_agent_velocity = True
            self.get_logger().debug("Got agent velocity")
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))

    def user_velocity_callback(self, vel):
        try:
            self.user_output['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
            if(self.state == SupervisorState.STANDBY):
                velocity_msg = Twist(linear=Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
                self.desired_velocity_publisher.publish(velocity_msg)
        except Exception:
            self.get_logger().warn(str(traceback.format_exc))

    ### SERVICE CALLBACKS

    def start_callback(self, request, response):
        try:
            # check type of start (task vs demo)
            start_type = request.command
            self.get_logger().info(f"<start> service called by {start_type}")
            if(start_type == 'task'):
                self.get_logger().info("Starting task")
                # state == standby
                if(self.state == SupervisorState.STANDBY):
                    # check if episode has been selected
                    if(self.selected_data['name'] == None or self.selected_data['id'] == None):
                        raise Exception("Episode not yet selected")
                    # call agent service to select demo
                    self.get_logger().debug("Calling agent/select_data")
                    selected_data = copy.deepcopy(self.selected_data)
                    selected_data['type'] = selected_data['type'].name
                    select_data_response = self.call_service('agent/select_data', json.dumps(selected_data))
                    if(select_data_response == False):
                        raise Exception("agent/select_data service not successful")
                    # set self.episode
                    if(self.selected_data['type'] == InformationType.TASK_EPISODE):
                        self.episode = self.task_handler.get(self.selected_data['name'], self.selected_data['id'])
                    elif(self.selected_data['type'] == InformationType.DEMO):
                        self.episode = self.demo_handler.get(self.selected_data['name'], self.selected_data['id'])
                    else:
                        raise Exception("Invalid data type")
                    self.controller = ControllerType.AGENT
                
                # state == task_paused or task_running
                elif(self.state in [SupervisorState.TASK_PAUSED, SupervisorState.TASK_RUNNING]):
                    # do nothing
                    pass
                # state == any other state => fail
                else:
                    raise Exception(f"Unable to start task as the current state == {self.state.name}")

                self.state = SupervisorState.TASK_RUNNING

            elif(start_type == 'demo'):
                self.get_logger().info("Starting demo")
                if(self.state == SupervisorState.STANDBY):
                    # check if demo name has been selected
                    if(self.selected_data['name'] in [None, ''] or self.selected_data['type'] == InformationType.TASK_EPISODE):
                        raise Exception("Demo name not yet selected")
                    # if the ID is selected, then set self.episode
                    if(self.selected_data['id'] not in [None, '']):
                        self.get_logger().debug("Retrieving episode")
                        try:
                            self.episode = self.demo_handler.get(self.selected_data['name'], self.selected_data['id'])
                        except:
                            raise Exception(f"Unable to retrieve episode {self.selected_data}")
                elif(self.state == SupervisorState.DEMO_PAUSED):
                    pass
                else:
                    raise Exception(f"Unable to start demo as the current state == {self.state.name}")
                self.state = SupervisorState.DEMO_RECORDING
                self.controller = ControllerType.USER
            else:
                raise Exception(f"Invalid start type")

            self.get_logger().info("<start> service completed")
            response.success = True
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    def pause_callback(self, request, response):
        try:
            self.get_logger().info("<pause> service called")
            self.pause()
            self.stop_vehicle()
            response.success = True
            self.get_logger().info("<pause> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response
    
    def stop_callback(self, request, response):
        try:
            self.get_logger().info("<stop> service called")
            self.episode.reset()
            self.stop_vehicle()
            self.selected_data = dict(string = "", type = InformationType.NONE, name = "", id = "")
            self.frame_no = 0
            self.state = SupervisorState.STANDBY
            response.success = True
            self.get_logger().info("<stop> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    def save_callback(self, request, response):
        try:
            self.get_logger().info("<save> service called")
            if(self.state == SupervisorState.STANDBY):
                raise Exception(f"Unable to save episode because state == {self.state}")

            # make sure episode is not empty
            if(self.episode.length() != 0):
                if(self.state in [SupervisorState.TASK_RUNNING, SupervisorState.TASK_PAUSED]):
                    self.state = SupervisorState.TASK_PAUSED
                    self.get_logger().debug("Saving task episode")
                    self.task_handler.save(self.episode, self.selected_data['name'], self.selected_data['id'])
                
                elif(self.state in [SupervisorState.DEMO_PAUSED, SupervisorState.DEMO_RECORDING]):
                    self.state = SupervisorState.DEMO_PAUSED
                    self.get_logger().debug("Saving demo episode")
                    self.demo_handler.save(self.episode, self.selected_data['name'])
            else:
                raise Exception(f"Unable to save episode as episode is empty")
            
            response.success = True
            self.get_logger().info("<save> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    def termination_flag_callback(self, request, response):
        try:
            requestor = ControllerType[request.command.upper()]
            self.get_logger().info(f"<termination_flag> service called by {requestor.name}")

            if(requestor == ControllerType.AGENT):
                if(self.controller == ControllerType.AGENT):
                    self.pause()
                    self.stop_vehicle()
                self.agent_output['termination_flag'] = True
            elif(requestor == ControllerType.USER):
                self.user_output['termination_flag'] = True
            else:
                raise Exception(f"Invalid requestor {requestor}")

            response.success = True
            self.get_logger().info("<termination_flag> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    def select_data_callback(self, request, response):
        try:
            self.get_logger().info("<select_data> service called")
            data_dict = json.loads(request.command)
            self.selected_data = dict(type = InformationType.NONE, name = "", id = "")
            for key, value in data_dict.items():
                if(key == 'type'):
                    value = InformationType[value]
                self.selected_data[key] = value
            response.success = True
            self.get_logger().info("<select_data> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    def select_controller_callback(self, request, response):
        try:
            self.get_logger().info("<select_controller> service called")
            if(self.state in [SupervisorState.TASK_PAUSED, SupervisorState.TASK_RUNNING, SupervisorState.STANDBY]):
                self.controller = ControllerType[request.command.upper()]
            else:
                raise Exception(f"Unable to select controller as state == {self.state.name}")
            
            response.success = True
            self.get_logger().info("<select_controller> service completed")
            return response
        except Exception as e:
            self.get_logger().warn(str(traceback.format_exc()))
            response.message = str(e)
            response.success = False
            return response

    ### TIMER CALLBACK

    def control_callback(self):
        try:
            state = copy.deepcopy(self.state)
            if(state in [SupervisorState.DEMO_RECORDING, SupervisorState.TASK_RUNNING]):
                if(state == SupervisorState.TASK_RUNNING):
                    if(self.episode.length() == (self.max_episode_length*2)):
                        self.pause()
                        return
                else:
                    if(self.episode.length() == (self.max_episode_length)):
                        self.pause()
                        return
                # 1. send image_raw_msg to agent 
                frame_no = self.frame_no + 1
                desired_output = dict(velocity = dict(linear = 0.0, angular = 0.0), termination_flag = False)

                # publish image and wait for agent output
                if(self.received_agent_velocity == False):
                    self.get_logger().info("=========Started a control frame=========")
                    self.get_logger().debug("Publishing task_image and frame_no")
                    self.task_image_msg = self.image_raw_msg
                    #self.get_logger().info(str(type(self.task_image_msg)))
                    self.frame_no_publisher.publish(Int32(data=frame_no))
                    self.task_image_publisher.publish(self.task_image_msg)

                    if(state == SupervisorState.TASK_RUNNING):
                        while(self.received_agent_velocity == False):
                            pass

                agent_output = self.agent_output
                self.received_agent_velocity = False

                # convert image to PImage to be appended to the episode
                image = PImage.fromarray(rnp.numpify(self.task_image_msg))

                # get user output
                user_output = copy.deepcopy(self.user_output)
                
                # set desired_output based on self.state and termination flags
                if(self.state == SupervisorState.DEMO_RECORDING):
                    agent_output = dict(velocity = dict(linear = 0.0, angular = 0.0), termination_flag = False)
                    desired_output = user_output
                    demonstration_flag = True
                else:
                    if(self.controller == ControllerType.USER):
                        desired_output = copy.deepcopy(user_output)
                        demonstration_flag = True
                    elif(self.controller == ControllerType.AGENT):
                        desired_output = copy.deepcopy(agent_output)
                        demonstration_flag = False
                    else:
                        raise Exception("Invalid controller when recording demo or running task")
                    
                    desired_output['termination_flag'] = copy.deepcopy(user_output['termination_flag'])

                # get final decision on the controller
                controller = self.controller
                
                # if the state changed during this process, then cancel this frame
                if(self.state != state):
                    self.received_agent_velocity = True
                    self.get_logger().warn("Cancelled current frame as the state changed during execution")
                    return
                else:
                    self.get_logger().debug("Publishing desired actions")
                    # publish desired velocity, desired termination flag, and action controller
                    desired_velocity_msg = Twist(linear=Vector3(x=desired_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=desired_output['velocity']['angular']))
                    self.desired_velocity_publisher.publish(desired_velocity_msg)
                    termination_flag_msg = Bool(data=desired_output['termination_flag'])
                    self.termination_flag_publisher.publish(termination_flag_msg)
                    action_controller_msg = String(data=controller.name)
                    self.action_controller_publisher.publish(action_controller_msg)
                    episode_frame_msg = EpisodeFrame(
                        user_velocity = Twist(
                            linear = Vector3(x=user_output['velocity']['linear'], y=0.0, z=0.0),
                            angular = Vector3(x=0.0, y=0.0, z=user_output['velocity']['angular'])
                        ),
                        agent_velocity = Twist(
                            linear = Vector3(x=agent_output['velocity']['linear'], y=0.0, z=0.0),
                            angular = Vector3(x=0.0, y=0.0, z=agent_output['velocity']['angular'])
                        ),
                        user_termination_flag = user_output['termination_flag'],
                        agent_termination_flag = agent_output['termination_flag'],
                        demonstration_flag = demonstration_flag,
                        observation = self.task_image_msg,
                        frame_no = self.frame_no
                    )
                    self.episode_frame_publisher.publish(episode_frame_msg)
                    # append frame to self.episode
                    episode_frame = EpisodeData(
                        observation = dict(image=image), 
                        action = dict(
                            user = user_output,
                            agent = agent_output,
                            controller = controller), 
                        frame_no = frame_no)
                    self.episode.append(episode_frame)

                    # pause and reset frame_no if termination_flag was raised
                    if(desired_output['termination_flag'] == True):
                        self.frame_no = 0
                        self.pause()
                        self.stop_vehicle()
                    else:
                        self.frame_no = frame_no

                    # reset termination_flag
                    self.user_output['termination_flag'] = False
                    self.agent_output['termination_flag'] = False

                    if(state == SupervisorState.TASK_RUNNING):
                        if(self.episode.length() == (self.max_episode_length*2-1)):
                            self.pause()
                    else:
                        if(self.episode.length() == (self.max_episode_length-1)):
                            self.pause()

                self.get_logger().info(f"Episode Length = {self.episode.length()}")
                self.get_logger().info("Completed a control frame")
        except:
            self.get_logger().warn(str(traceback.format_exc()))

    def publish_state(self):
        state = json.dumps(dict(state = self.state.name, controller = self.controller.name, episode_type = self.selected_data['type'].name, episode_name = self.selected_data['name'], episode_id = self.selected_data['id']))
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

    def pause(self):
        self.get_logger().info("Pausing episode")
        if(self.state in [SupervisorState.TASK_RUNNING]):
            self.state = SupervisorState.TASK_PAUSED
        elif(self.state == SupervisorState.DEMO_RECORDING):
            self.state = SupervisorState.DEMO_PAUSED
        else:
            raise Exception(f"Unable to pause as state == {self.state}")
    
    def stop_vehicle(self):
        desired_velocity_msg = Twist(linear=Vector3(x=0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=0.0))
        self.desired_velocity_publisher.publish(desired_velocity_msg)
    
def main():
    rclpy.init()

    node = SupervisorNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)

if __name__ == '__main__':
    main()