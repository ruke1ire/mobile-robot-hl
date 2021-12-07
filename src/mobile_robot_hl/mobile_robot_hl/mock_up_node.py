import rclpy
from rclpy.node import Node
from rclpy.qos import *
from sensor_msgs.msg import Image
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import threading
import os
import numpy as np
import time

import ros2_numpy as rnp

class AgentNode(Node):

    def __init__(self):
        super().__init__('agent')

        try:
            image_raw_topic_name = os.environ['MOBILE_ROBOT_HL_IMAGE_RAW_TOPIC']
        except:
            image_raw_topic_name = "image_raw/uncompressed"

        self.get_logger().info("Initializing Node")

        reliable_qos = QoSProfile(history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, 
                                        depth=10, 
                                        reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)

        self.image_raw_publisher = self.create_publisher(Image, image_raw_topic_name, reliable_qos, callback_group = ReentrantCallbackGroup())

        self.get_logger().info("Initialized Node")
    
    def loop(self):
        img_tmp = np.zeros([30,40,3],dtype=np.uint8)
        plus = True
        i = 0
        while True:
            img_tmp.fill(i)
            image_raw_msg = rnp.msgify(Image,img_tmp, encoding='rgb8')
            self.image_raw_publisher.publish(image_raw_msg)
            if(i == 255):
                plus = False
            elif(i == 0):
                plus = True

            if(plus == True):
                i += 1
            else:
                i -= 1
            time.sleep(0.1)
    
def spin_thread_(node, executor):
    rclpy.spin(node, executor)

def main():
    rclpy.init()

    node = AgentNode()
    executor = MultiThreadedExecutor()

    spin_thread = threading.Thread(target=spin_thread_, args=(node, executor, ))
    spin_thread.start()
    node.loop()

if __name__ == '__main__':
    main()
