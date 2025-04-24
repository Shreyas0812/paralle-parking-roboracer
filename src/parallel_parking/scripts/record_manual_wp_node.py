#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

from transforms3d.euler import quat2euler

class RecodManualWPNode(Node):
    def __init__(self):
        super().__init__("record_manual_wp_node")
        self.get_logger().info("Manual Waypoint Record Node Launched")

        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('visualize_wp_topic', '/visualization/waypoints')
        self.declare_parameter('waypoint_file_path', '/home/shreyas/Documents/ESE6150_F1_Tenth/parallel-parking-roboracer/src/parallel_parking/config/waypoints_manual.csv')

        goal_pose_topic = self.get_parameter('goal_pose_topic').get_parameter_value().string_value
        visualize_wp_topic = self.get_parameter('visualize_wp_topic').get_parameter_value().string_value
        waypoint_file_path = self.get_parameter('waypoint_file_path').get_parameter_value().string_value

        # Shared Variables
        self.waypoint_file = open(waypoint_file_path, 'w')
        self.waypoint_file.write("x,y,yaw,qw,qx,qy,qz\n")

        self.marker_array = MarkerArray()

        # Subscribers
        self.manual_pose_sub_ = self.create_subscription(PoseStamped, goal_pose_topic, self.manual_pose_callback, QoSProfile(depth=10))

        # Publishers
        self.waypoint_pub_ = self.create_publisher(MarkerArray, visualize_wp_topic, QoSProfile(depth=10))
    
    def manual_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        euler = quat2euler([qw, qx, qy, qz], axes='sxyz')
        yaw = euler[2]

        self.waypoint_file.write(f"{x},{y},{yaw},{qw},{qx},{qy},{qz}\n")

        self.get_logger().info(f"Recorded waypoint: x={x}, y={y}, yaw={yaw}")

        marker = Marker()

        marker.header = Header()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.id = len(self.marker_array.markers)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose = Pose()
        marker.pose.position = Point(x=x, y=y, z=0.0)
        marker.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        marker.scale = Vector3(x=0.1, y=0.1, z=0.1)

        marker.color = ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0)

        marker.lifetime.sec = 0

        self.marker_array.markers.append(marker)

        self.waypoint_pub_.publish(self.marker_array)

    def close_waypoint_file(self):
        self.get_logger().info("Closing waypoint file")
        self.waypoint_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = RecodManualWPNode()
    try:
        rclpy.spin(node)
    finally:
        node.close_waypoint_file()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()