#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PointStamped

import numpy as np
from transforms3d.euler import quat2euler

class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_node")
        self.get_logger().info("Python mpc_node has been started.")

def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()