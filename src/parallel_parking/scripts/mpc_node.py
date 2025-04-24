#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PointStamped

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
        self.declare_parameter('visualize_wp_topic', '/visualization/extrapolated_waypoints')


        package_share_dir = get_package_share_directory("parallel_parking")

        waypoint_file_name = self.get_parameter("waypoint_file_name").get_parameter_value().string_value
        visualize_wp_topic = self.get_parameter('visualize_wp_topic').get_parameter_value().string_value

        waypoint_file_path = os.path.join(package_share_dir, 'config', waypoint_file_name)
        waypoints = np.array(load_waypoints(waypoint_file_path))

        self.wp1 = waypoints[0]
        self.wp_rest = waypoints[1:-1]
        self.wp2 = waypoints[-1]

        qos_pos = QoSProfile(depth=10)

        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value

        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_pos)


        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )   

        # Publishers
        self.waypoint_marker_publisher_ = self.create_publisher(MarkerArray, visualize_wp_topic, qos_profile)

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

        extrapolated_waypoints.extend(self.wp_rest)

        # self.get_logger().info(f"Waypoints: {extrapolated_waypoints}", throttle_duration_sec=1.0)

        self.visualize_waypoints(extrapolated_waypoints)

    def visualize_waypoints(self, extrapolated_waypoints):
        marker_array = MarkerArray()
        for i, wp in enumerate(extrapolated_waypoints):
            x, y, yaw, *others = wp

            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose = Pose()
            marker.pose.position = Point(x=x, y=y, z=0.0)
            marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)

            marker.lifetime.sec = 0

            marker_array.markers.append(marker)

        self.waypoint_marker_publisher_.publish(marker_array)
        # self.get_logger().info("Waypoints Visualized")



def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()