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

        self.declare_parameter('waypoint_file_name', 'waypoints_park1.csv')
        self.declare_parameter('pose_topic', '/ego_racecar/odom')
        self.declare_parameter('next_wp_topic', '/next_waypoint')
        self.declare_parameter('lookahead_distance', 0.8)
        self.declare_parameter('y_ego_threshold', 1.2)
        
        waypoint_file_name = self.get_parameter('waypoint_file_name').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        next_wp_topic = self.get_parameter('next_wp_topic').get_parameter_value().string_value

        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.y_ego_threshold = self.get_parameter('y_ego_threshold').get_parameter_value().double_value

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

        wp_ego, wp_index =  self.find_lookup_waypoint(pos_x, pos_y, yaw)

        next_wp_index = wp_index + 1
        if wp_index == len(self.waypoints)-1:
            next_wp_index = 0

        wp_ego_next = self.waypoints[next_wp_index]

        self.publish_next_waypoint(wp_ego_next)

    def find_lookup_waypoint(self, pos_x, pos_y, yaw):

        waypoints_ego = []
        distances = []
        indices = []

        for i, waypoint in enumerate(self.waypoints):
            wp_x, wp_y, wp_yaw, _, _, _, _ = waypoint
            
            dx = wp_x - pos_x
            dy = wp_y - pos_y

            distance = np.sqrt(dx**2 + dy**2)

            x_ego = dx * np.cos(-yaw) - dy * np.sin(-yaw)
            y_ego = dx * np.sin(-yaw) + dy * np.cos(-yaw)

            if x_ego >= 0 and abs(y_ego) <= self.y_ego_threshold:
                waypoints_ego.append((wp_x, wp_y))
                distances.append(distance)
                indices.append(i)


        distances = np.array(distances)
        waypoints_ego = np.array(waypoints_ego)

        if len(waypoints_ego) != 0:
            
            distances[distances < self.lookahead_distance] = np.inf
            min_index = np.argmin(distances)

            self.prev_valid_index = min_index
            self.prev_ego_waypoints = waypoints_ego

            return waypoints_ego[min_index], indices[min_index]
        
        else:
            return self.prev_ego_waypoints[self.prev_valid_index], self.prev_valid_index

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