import rclpy
from rclpy.node import Node

class AgentNode(Node):

    def __init__(self):
        super().__init__('agent')
        #self.agent_output_publisher = self.create_publisher(String, 'topic', 10)
        #self.agent_input_publisher = self.create_publisher()
        #self.image_raw_subscriber = self.create_subscription()
        #self.desired_velocity_subscriber = self.create_subscription()
        #self.termination_flag_subscriber = self.create_subscription()

def main():
    pass


if __name__ == '__main__':
    main()
