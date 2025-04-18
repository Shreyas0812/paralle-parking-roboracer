#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

from transforms3d.euler import quat2euler

import csv
import numpy as np

from ament_index_python.packages import get_package_share_directory

from scipy.interpolate import CubicSpline

from parallel_parking.utils import load_waypoints

class VisualizeNode(Node):
    def __init__(self):
        super().__init__("visualize_node")
        self.get_logger().info("Visualize Node Launched")

        self.declare_parameter('waypoint_file_name', 'waypoints_levine.csv')
        self.declare_parameter('pose_topic', '/ego_racecar/odom')
        self.declare_parameter('visualize_wp_topic', '/visualization/waypoints')
        self.declare_parameter('visualize_next_wp_topic', '/visualization/next_waypoint')
        self.declare_parameter('lookahead_distance', 0.8)
        self.declare_parameter('y_ego_threshold', 1.2)

        waypoint_file_name = self.get_parameter('waypoint_file_name').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        visualize_wp_topic = self.get_parameter('visualize_wp_topic').get_parameter_value().string_value
        visualize_next_wp_topic = self.get_parameter('visualize_next_wp_topic').get_parameter_value().string_value
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

        # Subscribers
        self.pose_subscriber_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_profile)

        # Publishers
        self.waypoint_marker_publisher_ = self.create_publisher(MarkerArray, visualize_wp_topic, qos_profile)
        self.next_waypoint_publisher_ = self.create_publisher(MarkerArray, visualize_next_wp_topic, qos_profile)
        
        # self.visualization_timer = self.create_timer(1, self.visualize_waypoints)

        # Visualize the waypoints
        self.visualize_waypoints()

        self.prev_valid_index = 0
        self.prev_ego_waypoints = []
    
    def visualize_waypoints(self):
        marker_array = MarkerArray()
        for i, wp in enumerate(self.waypoints):
            x, y, yaw, qw, qx, qy, qz = wp

            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose = Pose()
            marker.pose.position = Point(x=x, y=y, z=0.0)
            marker.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

            marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)

            marker.lifetime.sec = 0

            marker_array.markers.append(marker)

        self.waypoint_marker_publisher_.publish(marker_array)
        self.get_logger().info("Waypoints Visualized")

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

        self.visualize_next_waypoint(wp_ego_next)

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

    def visualize_next_waypoint(self, wp_ego_next):
        x, y = wp_ego_next[:2]

        cur_marker_array = MarkerArray()

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose = Pose()
        marker.pose.position = Point(x=x, y=y, z=0.0)
        marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        marker.lifetime.sec = 0

        cur_marker_array.markers.append(marker)

        self.next_waypoint_publisher_.publish(cur_marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()