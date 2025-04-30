#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry

from transforms3d.euler import quat2euler

import numpy as np

from ament_index_python.packages import get_package_share_directory

from parallel_parking.utils import load_waypoints
from geometry_msgs.msg import PointStamped

class NextWaypoint(Node):
    def __init__(self):
        super().__init__("next_wp_node")
        self.get_logger().info("Python next_wp_node has been started.")

        self.declare_parameter('waypoint_file_name', 'waypoints_park2.csv')
        self.declare_parameter('pose_topic', '/ego_racecar/odom')
        self.declare_parameter('next_wp_topic', '/next_waypoint')
        self.declare_parameter('lookahead_distance', 0.04)
        
        waypoint_file_name = self.get_parameter('waypoint_file_name').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        next_wp_topic = self.get_parameter('next_wp_topic').get_parameter_value().string_value

        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        package_share_dir = get_package_share_directory("parallel_parking")

        waypoint_file_path = os.path.join(package_share_dir, 'config', waypoint_file_name)
        self.waypoints = np.array(load_waypoints(waypoint_file_path))

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )   

        self.pose_subscriber_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_profile)

        self.next_waypoint_publisher_ = self.create_publisher(PointStamped, next_wp_topic, qos_profile)


        self.wp_index_final = 0

    def pose_callback(self, msg):
        # Curret pose of the vehicle
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        euler = quat2euler([qw, qx, qy, qz])
        yaw = euler[2]


        # TODO: We can use some other parameters to determine the next waypoint as well, such as check the current velocity of the car, if positive something, if -ve something else, etc.

        # TODO: Use server client architecture for verfication of the next waypoint

        # 2 point parking
        wp1_index = 0
        wp1 = self.waypoints[wp1_index] # 1st point -- Right beside the car in front

        wp2_index = 1
        wp2 = self.waypoints[wp2_index] #2nd point -- Parking spot

        # Distance from the car to the 1st point
        dx = wp1[0] - pos_x
        dy = wp1[1] - pos_y
        distance = np.sqrt(dx**2 + dy**2)

        yaw_diff = abs(wp1[2] - yaw)

        if self.wp_index_final != wp2_index:

            if distance < self.lookahead_distance and yaw_diff < 0.1:
                self.wp_index_final = wp2_index
            else:
                self.wp_index_final = wp1_index

        wp_ego_next = self.waypoints[self.wp_index_final]

        self.publish_next_waypoint(wp_ego_next)

    def publish_next_waypoint(self, next_wp):
        # NOTE: We Can change this to more complicated waypoint message if needed
        # e.g. PointStamped with yaw and velocity
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.point.x = next_wp[0]
        msg.point.y = next_wp[1]
        msg.point.z = 0.0

        self.next_waypoint_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = NextWaypoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()