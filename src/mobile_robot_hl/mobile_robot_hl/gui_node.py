import rclpy
from rclpy.node import Node
from rclpy.qos import *
from custom_interfaces.srv import StringTrigger

import threading
import os

from mobile_robot_hl.gui import *

class GuiNode(Node):

    def __init__(self):
        super().__init__('gui')
        pass

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


