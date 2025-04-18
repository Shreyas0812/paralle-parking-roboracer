#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class ParallelParkingNode(Node):
    def __init__(self):
        super().__init__("parallel_paking_node")
        self.get_logger().info("Python parallel_paking_node has been started.")

def main(args=None):
    rclpy.init(args=args)
    node = ParallelParkingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()