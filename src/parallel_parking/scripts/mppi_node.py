#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from nav_msgs.msg import Odometry

from parallel_parking_interfaces.msg import Traj

from transforms3d.euler import quat2euler

class MPPINode(Node):
    def __init__(self):
        super().__init__("mppi_node")
        self.get_logger().info("Python mppi_node has been started.")

        self.declare_parameter("pose_topic", '/ego_racecar/odom')
        self.declare_parameter('extrapolated_path_topic', '/extrapolated_path')

        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        extrapolated_path_topic = self.get_parameter('extrapolated_path_topic').get_parameter_value().string_value
        
        qos_profile = QoSProfile(depth=10)

        # Subscribers
        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_profile)
        self.extrapolated_path_subscriber_ = self.create_subscription(Traj, extrapolated_path_topic, self.extrapolated_path_callback, qos_profile)

    def extrapolated_path_callback(self, msg):
        # Process the received message
        self.extrapolated_traj = msg.traj
        self.end_pose = msg.end_pose

    def pose_callback(self, msg):
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        yaw = quat2euler([qw, qx, qy, qz])[2]

        if not hasattr(self, 'extrapolated_traj'):
            self.get_logger().info("Extrapolated trajectory not received yet.", throttle_duration_sec=1.0)
        
        # Position and Yaw
        # pos_x, pos_y, yaw

        # Extrapolated Trajectory
        # self.extrapolated_traj

        # End Pose of the Extrapolated Trajectory
        # self.end_pose    

        self.get_logger().info(f"\n\n\nMPPI Node Ready\n\n\n", once=True)


def main(args=None):
    rclpy.init(args=args)
    node = MPPINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


    