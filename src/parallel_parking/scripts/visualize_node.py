#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, Quaternion, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

from parallel_parking_interfaces.msg import Traj

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

        self.declare_parameter('waypoint_file_name', 'waypoints_park1.csv')
        self.declare_parameter('visualize_wp_topic', '/visualization/waypoints')
        self.declare_parameter('extrapolated_path_topic', '/extrapolated_path')
        self.declare_parameter('visualize_extrapolated_wp_topic', '/visualization/extrapolated_path')

        waypoint_file_name = self.get_parameter('waypoint_file_name').get_parameter_value().string_value
        visualize_wp_topic = self.get_parameter('visualize_wp_topic').get_parameter_value().string_value
        extrapolated_path_topic = self.get_parameter('extrapolated_path_topic').get_parameter_value().string_value
        visualize_extrapolated_wp_topic = self.get_parameter('visualize_extrapolated_wp_topic').get_parameter_value().string_value

        package_share_dir = get_package_share_directory("parallel_parking")

        waypoint_file_path = os.path.join(package_share_dir, 'config', waypoint_file_name)
        self.waypoints = np.array(load_waypoints(waypoint_file_path))

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )   

        # Subscribers
        self.extrapolated_path_subscriber_ = self.create_subscription(Traj, extrapolated_path_topic, self.visualize_extrapolated_path, qos_profile)

        # Publishers
        self.waypoint_marker_publisher_ = self.create_publisher(MarkerArray, visualize_wp_topic, qos_profile)
        self.extrapolated_path_marker_publisher_ = self.create_publisher(MarkerArray, visualize_extrapolated_wp_topic, qos_profile)
        
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
    
    def visualize_extrapolated_path(self, msg):

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 4200
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale = Vector3(x=0.1, y=0.0, z=0.0)
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

        for i, pose in enumerate(msg.traj):
            p = Point()
            p.x = pose.position.x
            p.y = pose.position.y
            p.z = 0.1
            marker.points.append(p)
        
        marker.lifetime.sec = 1

        arrow_marker = Marker()
        arrow_marker.header = marker.header
        arrow_marker.id = 4201
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.scale = Vector3(x=0.3, y=0.15, z=0.1)
        arrow_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)



        if len(msg.traj) > 0:

            p1 = msg.traj[-2].position
            p2 = msg.traj[-1].position

            arrow_marker.pose.position.x = p2.x
            arrow_marker.pose.position.y = p2.y
            arrow_marker.pose.position.z = 0.2

            dx = p2.x - p1.x
            dy = p2.y - p1.y
            angle = np.arctan2(dy, dx)

            arrow_marker.pose.orientation.z = np.sin(angle / 2.0)
            arrow_marker.pose.orientation.w = np.cos(angle / 2.0)

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        if len(msg.traj) > 2:
            marker_array.markers.append(arrow_marker)
        
        self.extrapolated_path_marker_publisher_.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()