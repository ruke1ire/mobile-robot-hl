from tkinter import Image
from mobile_robot_hl.episode_data import *
import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.srv import StringTrigger, FloatTrigger
from std_srvs.srv import Trigger

import threading
import numpy as np
import os
from PIL import Image as PImage

from mobile_robot_hl.gui import *

class GuiNode(Node):

    def __init__(self):
        super().__init__('gui')

        self.episode = EpisodeData()
        self.episode_type = InformationType.NONE
        self.episode_name = None
        self.episode_id = None
        self.model_name = None
        self.model_id = None
        self.image_raw = PImage.fromarray(np.zeros([360, 480, 3], dtype = np.uint8).fill(100))

        #self.model_handler =

    def call_service(self, service_name, command=None):
        self.get_logger().info(f'Calling service "{service_name}"')
        if self.services_[service_name].wait_for_service(timeout_sec=0.1) == False:
            self.get_logger().warn(f'{service_name} service not available')
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
                self.get_logger().info(f'Service successful: "{service_name}"')
            else:
                self.get_logger().warn(f'"{service_name}" service error: {response.message}')
            return response

def spin_thread_(node):
    while True:
        rclpy.spin_once(node)

def main():
    rclpy.init()

    node = GuiNode()

    spin_thread = threading.Thread(target=spin_thread_, args=(node,))
    spin_thread.start()
    spin_thread.join()

if __name__ == '__main__':
    main()


