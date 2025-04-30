#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from nav_msgs.msg import Odometry

from geometry_msgs.msg import Pose, Quaternion, Point

from parallel_parking_interfaces.msg import Traj

from ament_index_python.packages import get_package_share_directory

import os
import numpy as np
from transforms3d.euler import quat2euler

from parallel_parking.utils import load_waypoints, generateAckermannWaypoints, normalize_angle

class TrajGen(Node):
    def __init__(self):
        super().__init__("traj_gen_node")
        self.get_logger().info("Python traj_gen_node has been started.")

        self.declare_parameter('waypoint_file_name', 'waypoints_park1.csv')
        self.declare_parameter("pose_topic", '/ego_racecar/odom')
        self.declare_parameter('extrapolated_path_topic', '/extrapolated_path')
        self.declare_parameter('wp1_dist_thresh', 0.5)
        self.declare_parameter('wp2_dist_thresh', 0.5)
        self.declare_parameter('wp1_angle_thresh', 0.34)
        self.declare_parameter('wp2_angle_thresh', 0.15)
        self.declare_parameter('switch_wp_index', 1)

        package_share_dir = get_package_share_directory("parallel_parking")

        waypoint_file_name = self.get_parameter("waypoint_file_name").get_parameter_value().string_value
        extrapolated_path_topic = self.get_parameter('extrapolated_path_topic').get_parameter_value().string_value
        self.wp1_dist_thresh = self.get_parameter('wp1_dist_thresh').get_parameter_value().double_value
        self.wp2_dist_thresh = self.get_parameter('wp2_dist_thresh').get_parameter_value().double_value
        self.wp1_angle_thresh = self.get_parameter('wp1_angle_thresh').get_parameter_value().double_value
        self.wp2_angle_thresh = self.get_parameter('wp2_angle_thresh').get_parameter_value().double_value
        self.switch_wp_index = self.get_parameter('switch_wp_index').get_parameter_value().integer_value
         
        waypoint_file_path = os.path.join(package_share_dir, 'config', waypoint_file_name)
        waypoints = np.array(load_waypoints(waypoint_file_path))

        # self.wp1 = waypoints[0]
        # self.wp_rest = waypoints[1:-1]
        # self.wp2 = waypoints[-1]

        self.wp1 = waypoints[0]
        self.wp2 = waypoints[1]
        self.wp3 = waypoints[2]
    

        qos_pos = QoSProfile(depth=10)

        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value

        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_pos)

        qos_profile = QoSProfile(depth=10)

        # Publishers
        self.extrapolated_path_publisher_ = self.create_publisher(Traj, extrapolated_path_topic, qos_profile)


        self.gotowp1 = True

        self.parked = False

    # def waypoint_to_follow(self, cur_wp):
    #     dis



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

        if not self.parked:
            # Distances to the waypoints
            dist_to_wp1 = np.sqrt((pos_x - self.wp1[0])**2 + (pos_y - self.wp1[1])**2)
            angle_to_wp = normalize_angle(np.arctan2(self.wp1[1] - pos_y, self.wp1[0] - pos_x) - yaw)

            dist_to_wp2 = np.sqrt((pos_x - self.wp2[0])**2 + (pos_y - self.wp2[1])**2)
            angle_to_wp2 = normalize_angle(np.arctan2(self.wp2[1] - pos_y, self.wp2[0] - pos_x) - yaw)


            # Always go to wp1 first
            if self.gotowp1:        
                # Check if the car is close to the first waypoint
                if dist_to_wp1 < self.wp1_dist_thresh and abs(angle_to_wp) < self.wp1_angle_thresh:
                    self.gotowp1 = False
                    return
                
                reversed_extrapolated_pts = generateAckermannWaypoints(start_x=self.wp1[0], start_y=self.wp1[1], start_yaw=self.wp1[2], goal_x=pos_x, goal_y=pos_y, goal_yaw=yaw, wheelbase=0.32, dt=0.1, max_steering_angle=0.52, velocity=-1.0)
                extrapolated_waypoints = reversed_extrapolated_pts[::-1]
                
            else:
                # Check if the car is Parked
                if dist_to_wp2 < self.wp2_dist_thresh and abs(angle_to_wp2) < self.wp2_angle_thresh:
                    self.parked = True
                    return
                elif dist_to_wp2 < self.wp2_dist_thresh: # Only distance is satisfied, not yaw we go to wp3
                    reversed_extrapolated_pts = generateAckermannWaypoints(start_x=self.wp3[0], start_y=self.wp3[1], start_yaw=self.wp3[2], goal_x=pos_x, goal_y=pos_y, goal_yaw=yaw, wheelbase=0.32, dt=0.1, max_steering_angle=0.52, velocity=-1.0)
                    extrapolated_waypoints = reversed_extrapolated_pts[::-1]
                else: # if the car is not close to the second waypoint, we go to the second waypoint
                    reversed_extrapolated_pts = generateAckermannWaypoints(start_x=self.wp2[0], start_y=self.wp2[1], start_yaw=self.wp2[2], goal_x=pos_x, goal_y=pos_y, goal_yaw=yaw, wheelbase=0.32, dt=0.1, max_steering_angle=0.52, velocity=-1.0)
                    extrapolated_waypoints = reversed_extrapolated_pts[::-1]
            self.publish_extrapolated_path(extrapolated_waypoints)


        else:
            self.get_logger().info("Car is parked, no need to follow the waypoints anymore.")
            

    def publish_extrapolated_path(self, extrapolated_waypoints):
        path = Traj()

        for i, wp in enumerate(extrapolated_waypoints):
            x, y, yaw = wp

            pose = Pose()
            pose.position = Point(x=x, y=y, z=0.0)
            pose.orientation = Quaternion()

            path.traj.append(pose)
        
        # if not self.switched_path:
        #     for i, wp in enumerate(self.wp_rest):
        #         x, y, yaw, qw, qx, qy, qz = wp

        #         pose = Pose()
        #         pose.position = Point(x=x, y=y, z=0.0)
        #         pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                
        #         path.traj.append(pose)

        #         if i == len(self.wp_rest) - 1:
        #             path.end_pose = pose
            

        self.extrapolated_path_publisher_.publish(path)
        # self.get_logger().info("Extrapolated Path Published")

def main(args=None):
    rclpy.init(args=args)
    node = TrajGen()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()