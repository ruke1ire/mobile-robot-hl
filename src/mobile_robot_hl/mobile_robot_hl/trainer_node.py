import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.srv import StringTrigger

import threading
import os

from mobile_robot_hl.trainer.trainer import Trainer

class TrainerNode(Node):

    def __init__(self):
        super().__init__('trainer')

        self.demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
        self.get_logger().info("Initializing Node")

        service_prefix = 'trainer/'

        self.command_service = self.create_service(StringTrigger, service_prefix+'command', self.command_callback)

        self.get_logger().info("Initialized Node")

    def command_callback(self, request, response):
        command = request.command
        self.get_logger().info(f"Got command {command}")

        response = StringTrigger()
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

