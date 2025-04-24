#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from parallel_parking_interfaces.msg import Traj

class MPPINode(Node):
    def __init__(self):
        super().__init__("mppi_node")
        self.get_logger().info("Python mppi_node has been started.")

def main(args=None):
    rclpy.init(args=args)
    node = MPPINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


    