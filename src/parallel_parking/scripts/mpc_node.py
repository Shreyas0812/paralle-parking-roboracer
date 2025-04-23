#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PointStamped

from ament_index_python.packages import get_package_share_directory

import os
import numpy as np
from transforms3d.euler import quat2euler

from parallel_parking.utils import load_waypoints, generateAckermannWaypoints

class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_node")
        self.get_logger().info("Python mpc_node has been started.")

        self.declare_parameter("pose_topic", '/ego_racecar/odom')


        package_share_dir = get_package_share_directory("parallel_parking")

        waypoint_file_path = os.path.join(package_share_dir, 'config', waypoint_file_name)
        self.waypoints = np.array(load_waypoints(waypoint_file_path))

        qos_pos = QoSProfile(depth=10)

        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value

        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_pos)

    def pose_callback(self, msg):
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Convert quaternion to Euler angles
        euler_angles = quat2euler([qw, qx, qy, qz])
        yaw = euler_angles[2]

        self.get_logger().info(f"Waypoints: {self.waypoints}")

        # self.waypoints = generateAckermannWaypoints(start_x=pos_x, start_y=pos_y, start_yaw=yaw)



def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()