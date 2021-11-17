import rclpy
from rclpy.node import Node
from rclpy.qos import *
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from custom_interfaces.srv import StringTrigger
from std_srvs.srv import Trigger

from threading import Thread
import time

from .joystick import *

supervisor_prefix = 'supervisor/'

class JoystickNode(Node):
    def __init__(self):
        super().__init__('joystick')

        best_effort_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=1, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

        self.termination_flag_state = False
        self.start_pause_task_state = False
        self.take_over_state = False
        self.start_pause_demo_state = False
        self.stop_state = False

        self.get_logger().info("Initializing Node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_linear_velocity', None),
                ('max_angular_velocity', None),
            ])

        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value

        self.joy_handler = JoyHandler(max_linear_vel=self.max_linear_velocity, max_angular_vel=self.max_angular_velocity)

        self.get_logger().info(f"Parameter <frequency> = {self.frequency}")
        self.get_logger().info(f"Parameter <max_linear_velocity> = {self.max_linear_velocity}")
        self.get_logger().info(f"Parameter <max_angular_velocity> = {self.max_angular_velocity}")

        self.user_velocity_group = ReentrantCallbackGroup()
        self.user_velocity_publisher = self.create_publisher(Twist, 'user_velocity', best_effort_qos, callback_group=self.user_velocity_group)
        # TODO: self.supervisor_state_subscriber = self.create_subscription()

        self.services_ = {}

        self.service_group = ReentrantCallbackGroup()
        self.services_[supervisor_prefix+'termination_flag'] = self.create_client(Trigger, supervisor_prefix+'termination_flag', callback_group=self.service_group)
        self.services_[supervisor_prefix+'start'] = self.create_client(StringTrigger, supervisor_prefix+'start')
        self.services_[supervisor_prefix+'pause'] = self.create_client(StringTrigger, supervisor_prefix+'start')
        self.services_[supervisor_prefix+'stop'] = self.create_client(StringTrigger, supervisor_prefix+'start')

        self.get_logger().info("Initialized Node")

    def velocity_loop(self):
        self.get_logger().info("Publishing user_velocity")
        while True:
            state = self.joy_handler.get_state()
            velocity_msg = Twist(linear=Vector3(x=state[InterfaceType.LINEAR_VELOCITY.name],y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z=state[InterfaceType.ANGULAR_VELOCITY.name]))
            self.user_velocity_publisher.publish(velocity_msg)
            time.sleep(0.1)
    
    def service_loop(self):
        prev_joy_state = self.joy_handler.get_state()
        while True:
            joy_state = self.joy_handler.get_state()
            if(prev_joy_state[InterfaceType.STOP.name] == False and joy_state[InterfaceType.STOP.name] == True):
                self.call_service(supervisor_prefix+'stop')
            elif(prev_joy_state[InterfaceType.START_PAUSE_TASK.name] == False and joy_state[InterfaceType.START_PAUSE_TASK.name] == True):
                # TODO: need some logic here to determine if it should be start or pause
                self.call_service(supervisor_prefix+'start', command = 'task')
            elif(prev_joy_state[InterfaceType.TAKE_OVER_TASK.name] == False and joy_state[InterfaceType.TAKE_OVER_TASK.name] == True):
                self.call_service(supervisor_prefix+'take_over')
            elif(prev_joy_state[InterfaceType.START_PAUSE_DEMO.name] == False and joy_state[InterfaceType.START_PAUSE_DEMO.name] == True):
                # TODO: need some logic here to determine if it should be start or pause
                self.call_service(supervisor_prefix+'start', command = 'demo')
            elif(prev_joy_state[InterfaceType.TERMINATION_FLAG.name] == False and joy_state[InterfaceType.TERMINATION_FLAG.name] == True):
                self.call_service(supervisor_prefix+'termination_flag')

            prev_joy_state = joy_state.copy()
            time.sleep(0.1)
    
    def call_service(self, service_name, command=None):
        self.get_logger().info(f'Calling service {service_name}.')
        if self.services_[service_name].wait_for_service(timeout_sec=0.1) == False:
            self.get_logger().warn(f'{service_name} service not available')
        else:
            if(command == None):
                request = Trigger.Request()
            else:
                request = StringTrigger.Request()
                request.command = str(command)
            response = self.services_[service_name].call(request)

            if(response.success == True):
                self.get_logger().info(f'Service successful: {service_name}')
            else:
                self.get_logger().warn(f'{service_name} service error: {response.message}')
            return response

def main():
    rclpy.init()

    joystick_node = JoystickNode()
    executor = MultiThreadedExecutor()

    Thread(target=lambda: rclpy.spin(joystick_node, executor = executor)).start()
    Thread(target=lambda: joystick_node.velocity_loop()).start()
    joystick_node.service_loop()

if __name__ == "__main__":
    main()