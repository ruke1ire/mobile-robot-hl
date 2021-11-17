import rclpy
from rclpy.node import Node
from rclpy.qos import *
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.msg import AgentOutput
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
        self.episode = EpisodeData(data=None)
        self.received_agent_output = False

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
        self.agent_output_subscriber = self.create_subscription(AgentOutput, 'agent_output', self.agent_output_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.user_velocity_subscriber = self.create_subscription(Twist, 'user_velocity', self.user_velocity_callback, best_effort_qos, callback_group = ReentrantCallbackGroup())

        agent_prefix = "agent/"
        supervisor_prefix='supervisor/'

        self.services_ = dict()

        self.supervisor_service_group = ReentrantCallbackGroup()
        self.services_[supervisor_prefix+'start'] = self.create_service(StringTrigger, supervisor_prefix+'start', self.start_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'pause'] = self.create_service(Trigger, supervisor_prefix+'pause', self.pause_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'stop'] = self.create_service(Trigger, supervisor_prefix+'stop', self.stop_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'save'] = self.create_service(Trigger, supervisor_prefix+'save', self.save_callback, callback_group=self.supervisor_service_group)
        self.services_[supervisor_prefix+'termination_flag'] = self.create_service(Trigger, supervisor_prefix+'termination_flag', self.termination_flag_callback, callback_group=self.supervisor_service_group)
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

    def agent_output_callback(self, msg):
        velocity = msg.velocity
        termination_flag = msg.termination_flag
        self.agent_output['velocity'] = {'linear':velocity.linear.x, 'angular': velocity.angular.z}
        self.agent_output['termination_flag'] = termination_flag
        self.received_agent_output = True

    def user_velocity_callback(self, vel):
        self.user_output['velocity'] = {'linear':vel.linear.x, 'angular': vel.angular.z}
        if(self.state == SupervisorState.STANDBY):
            velocity_msg = Twist(linear=Vector3(x=self.user_output['velocity']['linear'],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=self.user_output['velocity']['angular']))
            self.desired_velocity_publisher.publish(velocity_msg)

    ### SERVICE CALLBACKS

    def start_callback(self, request, response):
        # check type of start (task vs demo)
            # type = task
                # state == standby:
                    # check if "demo" is selected otherwise fail
                    # call agent service select_demo
                    # condition episode on "demo"
                # state == task_paused, task_running
                    # do nothing
                # state == anything else
                    # fail
                # controller => AGENT
                # state => TASK_RUNNING
            # type = demo
                # state == standby:
                    # check if "demo" is selected otherwise fail
                    # condition episode on "demo"
                # state == demo_paused
                    # state => DEMO_RECORDING
                # any other state 
                    # fail
        pass

    def pause_callback(self, request, response):
        # state == TASK_RUNNING
            # state => TASK_PAUSED
        # state == DEMO_RECORDING
            # state => DEMO_PAUSED
        # state == anything else
            # fail
        pass

    def stop_callback(self, request, response):
        # reset episode
        # state => standby
        pass

    def save_callback(self, request, response):
        # make sure episode is not empty
        # if state == STANDBY
            # fail
        # if state == TASK_RUNNING or TASK_PAUSED
            # state => TASK_PAUSED
            # save task episode
        # if state == DEMO_RECORDING or DEMO_PAUSED
            # state => DEMO_PAUSED
            # save demo episode
        pass

    def termination_flag_callback(self, request, response):
        pass

    def select_data_callback(self, request, response):
        # set self.selected_data variable = request.data
        pass

    def select_controller_callback(self, request, response):
        # state = TASK_PAUSED, TASK_RUNNING
            # controller => request.data
        # state = DEMO_RECORDING, DEMO_PAUSED
            # fail
        # state = STANDBY
            # fail
        pass

    def configure_disturbance_callback(self, request, response):
        pass

    ### TIMER CALLBACK

    def control_callback(self):
        # 1. Check current supervisor state
            # state = DEMO_RECORDING, TASK_RUNNING
                # 1. convert image_raw_msg to PImage
                # 2. get user_velocity
                    # controller = USER
                            # state DEMO_RECORDING
                                # 1. set agent actions = None
                            # state TASK_RUNNING
                                # 1. get agent_actions
                        # 1. set termination_flag = False
                        # 2. set controller type = USER
                    # controller = AGENT
                        # 1. get agent_actions
                        # 2. set termination_flag = False
                        # 3. set controller type = AGENT
                # 3. Publish desired_velocity
                # 4. append data into episode_data
            # state = DEMO_PAUSED, TASK_PAUSED
                # 1. Do nothing?
            # state = STANDBY
                # do nothing
        pass

    def publish_state(self):
        msg = String(data=self.state.name)
        self.supervisor_state_publisher.publish(msg)

    ### UTILS
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
    
def main():
    rclpy.init()

    node = SupervisorNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)

if __name__ == '__main__':
    main()