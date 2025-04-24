#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose, Quaternion, Vector3, Point



from ament_index_python.packages import get_package_share_directory

import os
import numpy as np
from transforms3d.euler import quat2euler

from parallel_parking.utils import load_waypoints, generateAckermannWaypoints

class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_node")
        self.get_logger().info("Python mpc_node has been started.")

        self.declare_parameter('waypoint_file_name', 'waypoints_park1.csv')
        self.declare_parameter("pose_topic", '/ego_racecar/odom')
        self.declare_parameter('extrapolated_path_topic', '/extrapolated_path')

        package_share_dir = get_package_share_directory("parallel_parking")

        waypoint_file_name = self.get_parameter("waypoint_file_name").get_parameter_value().string_value
        extrapolated_path_topic = self.get_parameter('extrapolated_path_topic').get_parameter_value().string_value
         
        waypoint_file_path = os.path.join(package_share_dir, 'config', waypoint_file_name)
        waypoints = np.array(load_waypoints(waypoint_file_path))

        self.wp1 = waypoints[0]
        self.wp_rest = waypoints[1:-1]
        self.wp2 = waypoints[-1]

        qos_pos = QoSProfile(depth=10)

        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value

        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_pos)

        qos_profile = QoSProfile(depth=10)

        # Publishers
        self.extrapolated_path_publisher_ = self.create_publisher(Path, extrapolated_path_topic, qos_profile)

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

        extrapolated_waypoints = generateAckermannWaypoints(start_x=pos_x, start_y=pos_y, start_yaw=yaw, goal_x=self.wp1[0], goal_y=self.wp1[1], goal_yaw=self.wp1[2], wheelbase=0.32, dt=0.1, max_steering_angle=0.52, velocity=1.0)

        # self.get_logger().info(f"Waypoints: {extrapolated_waypoints}", throttle_duration_sec=1.0)

        self.publish_extrapolated_path(extrapolated_waypoints)

    def publish_extrapolated_path(self, extrapolated_waypoints):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for i, wp in enumerate(extrapolated_waypoints):
            x, y, yaw = wp

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = Point(x=x, y=y, z=0.0)
            pose.pose.orientation = Quaternion()

            path.poses.append(pose)
        
        for i, wp in enumerate(self.wp_rest):
            x, y, yaw, qw, qx, qy, qz = wp

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = Point(x=x, y=y, z=0.0)
            pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            
            path.poses.append(pose)

        self.extrapolated_path_publisher_.publish(path)
        self.get_logger().info("Extrapolated Path Published", once=True)

def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()