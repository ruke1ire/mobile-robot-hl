import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.srv import StringTrigger
from std_srvs.srv import Trigger

import threading
import os

class TrainerNode(Node):

    def __init__(self):
        super().__init__('trainer')

        self.demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        self.get_logger().info("Initializing Node")

        service_prefix = 'trainer/'

        self.select_model_service = self.create_service(StringTrigger, service_prefix+'select_model', self.select_model_service_callback)
        self.start_service = self.create_service(Trigger, service_prefix+'start', self.start_service_callback)
        self.pause_service = self.create_service(Trigger, service_prefix+'pause', self.pause_service_callback)
        self.stop_service = self.create_service(Trigger, service_prefix+'stop', self.stop_service_callback)
        self.save_service = self.create_service(Trigger, service_prefix+'save', self.save_service_callback)
        self.delete_service = self.create_service(Trigger, service_prefix+'delete', self.delete_service_callback)
        self.pre_train_service = self.create_service(Trigger, service_prefix+'pre_train', self.pre_train_service_callback)

        self.get_logger().info("Initialized Node")

    def select_model_service_callback(self):
        response = StringTrigger()
        response.success = True
        return response

    def start_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def pause_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def stop_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def save_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def delete_service_callback(self):
        response = Trigger()
        response.success = True
        return response

    def pre_train_service_callback(self):
        response = Trigger()
        response.success = True
        return response

def spin_thread_(node):
    while True:
        rclpy.spin_once(node)

def main():
    rclpy.init()

    node = TrainerNode()

    spin_thread = threading.Thread(target=spin_thread_, args=(node,))
    spin_thread.start()
    spin_thread.join()

if __name__ == '__main__':
    main()

