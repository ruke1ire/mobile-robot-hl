import rclpy
from rclpy.qos import *
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import StringTrigger, FloatTrigger
from custom_interfaces.msg import EpisodeFrame
from std_srvs.srv import Trigger
from std_msgs.msg import Bool, String, Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3

from queue import Queue
from threading import Thread, active_count
import numpy as np
import os
from PIL import Image as PImage
import copy
import json
import seaborn as sns
import traceback

from mobile_robot_hl.episode_data import *
from mobile_robot_hl.gui import *
from mobile_robot_hl.model import *

import ros2_numpy as rnp

class GUINode(Node):

    def __init__(self):
        super().__init__('gui')

        demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']
        model_path = os.environ['MOBILE_ROBOT_HL_MODEL_PATH']
        self.constants.max_linear_vel = float(os.environ['MOBILE_ROBOT_HL_MAX_LINEAR_VEL'])
        self.constants.max_angular_vel = float(os.environ['MOBILE_ROBOT_HL_MAX_ANGULAR_VEL'])
        
        try:
            image_raw_topic_name = os.environ['MOBILE_ROBOT_HL_IMAGE_RAW_TOPIC']
        except:
            image_raw_topic_name = "image_raw/uncompressed"

        self.demo_handler = DemoHandler(path=demo_path)
        self.task_handler = TaskHandler(path=task_path, demo_handler = self.demo_handler)
        self.model_handler = ModelHandler(path=model_path)

        self.variables = GUIVariable()
        self.init_variables()
        self.constants = GUIConstant()
        self.conditioned_episode = False

        self.get_logger().info("Initializing Node")

        self.episode_event_queue = Queue()

        self.user_output = dict(velocity = dict(linear = 0.0, angular = 0.0), termination_flag = False)
        self.agent_output = dict(velocity = dict(linear = 0.0, angular = 0.0), termination_flag = False)

        self.gui = GUI(ros_node = self)

        self.variable_trigger = dict(
            episode = [
                self.gui.display.episode.update_image, 
                self.gui.display.episode.update_plot_full, 
                self.gui.display.current.info.update_info,
                self.gui.display.episode.update_episode_index
                ],
            episode_type = [
                self.gui.display.current.info.update_info
                ],
            episode_name = [
                self.gui.display.current.info.update_info
                ],
            episode_id = [
                self.gui.display.current.info.update_info
                ],
            model_name = [
                self.gui.display.current.info.update_info
                ],
            model_id = [
                self.gui.display.current.info.update_info
                ],
            image_raw = [
                self.gui.display.current.update_image
                ],
            supervisor_state = [
                self.gui.display.current.info.update_info,
                self.update_episode_event,
                self.gui.control.task.update_buttons,
                self.gui.control.demo.update_buttons,
                self.gui.control.model.update_buttons,
                ],
            supervisor_controller = [
                self.gui.control.task.update_buttons,
                self.gui.control.demo.update_buttons,
                self.gui.control.model.update_buttons,
            ],
            supervisor_episode_type = [],
            supervisor_episode_name = [],
            supervisor_episode_id = [],
            demo_names = [
                self.gui.control.demo.update_entry,
                self.gui.control.selection.update_demo
                ],
            task_names = [
                self.gui.control.selection.update_task
                ],
            ids = [
                self.gui.control.selection.update_id
                ],
            model_names = [
                self.gui.control.model.update_entries_name
            ],
            model_ids = [
                self.gui.control.model.update_entries_id
            ],
            task_queue = [
                self.gui.control.selection.update_queue
            ],
            episode_index = [
                self.gui.display.episode.update_image,
                self.gui.display.current.info.update_info
                #self.gui.display.episode.update_plot_sel
            ],
            user_velocity = [
                #self.gui.display.episode.update_plot_sel_live_velocity
            ],
        )

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.episode_frame_subscriber = self.create_subscription(EpisodeFrame, 'episode_frame', self.episode_frame_callback, reliable_qos, callback_group = ReentrantCallbackGroup())
        self.user_velocity_subscriber = self.create_subscription(Twist, 'user_velocity', self.user_velocity_callback, best_effort_qos, callback_group = ReentrantCallbackGroup())

        self.image_raw_subscriber = self.create_subscription(Image, image_raw_topic_name, self.image_raw_callback ,best_effort_qos, callback_group = ReentrantCallbackGroup())
        self.supervisor_state_subscriber = self.create_subscription(String, 'supervisor_state', self.supervisor_state_callback, best_effort_qos, callback_group=ReentrantCallbackGroup())

        self.client_callback_group = ReentrantCallbackGroup()
        self.services_ = dict()
        self.services_['supervisor/start'] = self.create_client(StringTrigger, 'supervisor/start', callback_group=self.client_callback_group)
        self.services_['supervisor/pause'] = self.create_client(Trigger, 'supervisor/pause', callback_group=self.client_callback_group)
        self.services_['supervisor/stop'] = self.create_client(Trigger, 'supervisor/stop', callback_group=self.client_callback_group)
        self.services_['supervisor/select_data'] = self.create_client(StringTrigger, 'supervisor/select_data', callback_group=self.client_callback_group)
        self.services_['supervisor/termination_flag'] = self.create_client(StringTrigger, 'supervisor/termination_flag', callback_group=self.client_callback_group)
        self.services_['supervisor/select_controller'] = self.create_client(StringTrigger, 'supervisor/select_controller', callback_group=self.client_callback_group)
        self.services_['supervisor/save'] = self.create_client(Trigger, 'supervisor/save', callback_group=self.client_callback_group)
        self.services_['agent/select_model'] = self.create_client(StringTrigger, 'agent/select_model', callback_group=self.client_callback_group)
        self.services_['agent/configure_disturbance'] = self.create_client(FloatTrigger, 'agent/configure_disturbance', callback_group=self.client_callback_group)

        self.get_logger().info("Initialized Node")
    
    def episode_frame_callback(self, episode_frame_msg):
        self.get_logger().debug("Receiving episode frame")

        try:
            user_velocity = episode_frame_msg.user_velocity
            agent_velocity = episode_frame_msg.agent_velocity
            user_termination_flag = episode_frame_msg.user_termination_flag
            agent_termination_flag = episode_frame_msg.agent_termination_flag
            demonstration_flag = episode_frame_msg.demonstration_flag
            observation = PImage.fromarray(rnp.numpify(episode_frame_msg.observation))
            frame_no = episode_frame_msg.frame_no
            if(demonstration_flag == True):
                controller = ControllerType.USER
            else:
                controller = ControllerType.AGENT

            episode_data = EpisodeData(
                observation = dict(image = observation),
                action = dict(
                    user = dict(
                        velocity = dict(
                            linear = user_velocity.linear.x,
                            angular = user_velocity.angular.z
                        ),
                        termination_flag = user_termination_flag,
                    ),
                    agent = dict(
                        velocity = dict(
                            linear = agent_velocity.linear.x,
                            angular = agent_velocity.angular.z
                        ),
                        termination_flag = agent_termination_flag,
                    ),
                    controller = controller
                ),
                frame_no = frame_no
                )

            episode_event = dict(function = self.variables.episode.append, kwargs = episode_data)
            self.episode_event_queue.put(episode_event)
            self.get_logger().debug("Received episode frame")
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))

    def user_velocity_callback(self, vel):
        try:
            self.variables.user_velocity = {'linear':vel.linear.x, 'angular': vel.angular.z}
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))
    
    def image_raw_callback(self, img):
        try:
            self.variables.image_raw = PImage.fromarray(rnp.numpify(img))
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))
    
    def supervisor_state_callback(self, msg):
        try:
            supervisor_state = json.loads(msg.data)
            self.variables.supervisor_state = SupervisorState[supervisor_state['state']]
            self.variables.supervisor_controller = ControllerType[supervisor_state['controller']]
            if(supervisor_state['episode_type'] == None):
                self.variables.supervisor_episode_type = InformationType.NONE
            else:
                self.variables.supervisor_episode_type = InformationType[supervisor_state['episode_type']]
            self.variables.supervisor_episode_name = supervisor_state['episode_name']
            self.variables.supervisor_episode_id = supervisor_state['episode_id']
        except Exception:
            self.get_logger().warn(str(traceback.format_exc()))

    def call_service(self, service_name, command=None):
        self.get_logger().info(f'Calling service <{service_name}>')
        if self.services_[service_name].wait_for_service(timeout_sec=0.1) == False:
            self.get_logger().warn(f'{service_name} service not available')
            return False
        else:
            if(command == None):
                request = Trigger.Request()
            else:
                if(type(command) == float):
                    request = FloatTrigger.Request()
                    request.command = command
                else:
                    request = StringTrigger.Request()
                    request.command = str(command)
            response = self.services_[service_name].call(request)

            if(response.success == True):
                self.get_logger().info(f'Service successful: <{service_name}>')
            else:
                self.get_logger().warn(f'<{service_name}> service error: {response.message}')
            return response.success

    def update_episode(self, type, name, id, condition = False):
        self.get_logger().info(f"Updating episode with type = {type}, name = {name}, id = {id}")
        if type == InformationType.DEMO:
            if(id is None):
                self.variables.episode.reset()
            elif(name is None):
                raise Exception("Cannot update episode")
            else:
                self.variables.episode = self.demo_handler.get(name, id)
            self.variables.episode_type = InformationType.DEMO
            self.conditioned_episode = condition
        elif type == InformationType.TASK_EPISODE:
            if(name is None or id is None):
                raise Exception("Cannot update episode")
            else:
                self.variables.episode = self.task_handler.get(name, id)
            self.variables.episode_type = InformationType.TASK_EPISODE
            self.conditioned_episode = condition
        else:
            self.variables.episode.reset()
            self.variables.episode_name = "None"
            self.variables.episode_id = "None"
            self.variables.episode_type = InformationType.NONE
            self.conditioned_episode = False
            return
        self.variables.episode_name = str(name)
        self.variables.episode_id = str(id)
    
    def update_episode_event(self):
        if(self.variables.supervisor_state == SupervisorState.STANDBY):
            type_ = InformationType.NONE
            name = None
            id_ = None
            episode_event = dict(function = self.update_episode, kwargs = dict(type = type_, name = name, id = id_, condition = False))
            self.episode_event_queue.put(episode_event)
        elif(self.variables.supervisor_state == SupervisorState.DEMO_RECORDING and self.conditioned_episode == False):
            type_ = self.variables.supervisor_episode_type
            name = self.variables.supervisor_episode_name
            id_ = self.variables.supervisor_episode_id
            episode_event = dict(function = self.update_episode, kwargs = dict(type = type_, name = name, id = id_, condition = True))
            self.episode_event_queue.put(episode_event)
        elif(self.variables.supervisor_state == SupervisorState.TASK_RUNNING and self.conditioned_episode == False):
            type_ = self.variables.supervisor_episode_type
            name = self.variables.supervisor_episode_name
            id_ = self.variables.supervisor_episode_id
            episode_event = dict(function = self.update_episode, kwargs = dict(type = type_, name = name, id = id_, condition = True))
            self.episode_event_queue.put(episode_event)
    
    def update_state_loop(self):
        prev_variables = GUIVariable()
        while True:
            for var_type in self.variables.__dict__.keys():
                if(self.variables.__dict__[var_type] != prev_variables.__dict__[var_type]):
                    #print(self.variables.__dict__[var_type])
                    for trigger in self.variable_trigger[var_type]:
                        Thread(target=trigger).start()
                        #print(active_count())
                        #self.get_logger().info(str(self.variables.task_queue))
                        if(var_type == "episode"):
                            prev_variables.episode = EpisodeData(**self.variables.episode.get())
                        else:
                            exec(f"prev_variables.{var_type} = copy.deepcopy(self.variables.{var_type})")
    
    def run_episode_event_queue(self):
        self.get_logger().info("Starting execution of episode events")
        while True:
            try:
                episode_event = self.episode_event_queue.get()
                #print(str(episode_event))
                try:
                    episode_event['function'](**episode_event['kwargs'])
                except:
                    episode_event['function'](episode_event['kwargs'])
            except Exception:
                self.get_logger().warn(str(traceback.format_exc()))
    
    def init_variables(self):
        self.variables.demo_names = self.demo_handler.get_names()
        self.variables.task_names = self.task_handler.get_names()
        self.variables.model_names = self.model_handler.get_names(ModelType.ACTOR)

def spin_thread_(node):
    while True:
        rclpy.spin_once(node)

def main():
    rclpy.init()

    sns.set('notebook')
    sns.set_style("white")
    node = GUINode()

    Thread(target = node.update_state_loop).start()
    Thread(target =node.run_episode_event_queue).start()
    Thread(target= spin_thread_, args=(node,)).start()
    node.gui.window.mainloop()

if __name__ == '__main__':
    main()