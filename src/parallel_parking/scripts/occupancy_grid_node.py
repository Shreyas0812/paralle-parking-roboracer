#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid

import numpy as np
from transforms3d.euler import quat2euler

from parallel_parking.utils import get_grid_coordinates, get_world_coordinates, update_grid_with_ray

class OccupancyGridNode(Node):
    def __init__(self):
        super().__init__("occupancy_grid_node")
        self.get_logger().info("Python occupancy_grid_node has been started.")

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('pose_topic', '/ego_racecar/odom')
        self.declare_parameter('original_map_topic', '/map')
        self.declare_parameter('occupancy_grid_topic', '/occupancy_grid')

        self.declare_parameter('map_height', 713)
        self.declare_parameter('map_width', 727)
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_origin', (-20.2, -5.68))

        self.declare_parameter('expand_occ_size', 2)

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        original_map_topic = self.get_parameter('original_map_topic').get_parameter_value().string_value
        occupancy_grid_topic = self.get_parameter('occupancy_grid_topic').get_parameter_value().string_value

        self.map_height = self.get_parameter('map_height').get_parameter_value().integer_value
        self.map_width = self.get_parameter('map_width').get_parameter_value().integer_value
        self.map_resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_origin = self.get_parameter('map_origin').get_parameter_value().double_array_value

        self.expand_occ_size = self.get_parameter('expand_occ_size').get_parameter_value().integer_value
       
        # Subscribers
        
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # NOTE: self.occupancy_grid is defined and set inside map_callback
        self.map_subscriber_ = self.create_subscription(OccupancyGrid, original_map_topic, self.map_callback, map_qos_profile)


        qos_profile = QoSProfile(depth=10)

        # NOTE: self.cur_pos and self.cur_yaw is defined and set inside pose_callback
        self.pose_subscriber_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, qos_profile)

        # NOTE: self.occupancy_grid is updated and published inside scan_callback
        self.scan_subscriber_ = self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos_profile)         

        # Publisher
        
        occupancy_grid_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.occupancy_grid_publisher_ = self.create_publisher(OccupancyGrid, occupancy_grid_topic, occupancy_grid_qos_profile)

    # Callbacks
    def map_callback(self, map_msg):
        # Process the map message

        self.map_height = map_msg.info.height
        self.map_width = map_msg.info.width
        self.map_resolution = map_msg.info.resolution
        self.map_origin = (map_msg.info.origin.position.x, map_msg.info.origin.position.y)

        self.occupancy_grid = np.array(map_msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))

        self.get_logger().info(f"Received map and occupancy_grid set: \n"
                               f"height: {self.map_height}, width: {self.map_width},\n"
                               f"resolution: {self.map_resolution}, \n"
                               f"origin: {self.map_origin}"
                               )
        
    def pose_callback(self, pose_msg):
        # Process the Odometry message
        self.cur_pos = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
        self.cur_yaw = quat2euler([pose_msg.pose.pose.orientation.w,
                                          pose_msg.pose.pose.orientation.x,
                                          pose_msg.pose.pose.orientation.y,
                                          pose_msg.pose.pose.orientation.z])[2]

        # self.get_logger().info(f"Received pose data: {self.cur_pos}, yaw: {self.cur_yaw}")

    def scan_callback(self, scan_msg):
        # Process the LaserScan message
        
        if not hasattr(self, 'occupancy_grid'):
            self.get_logger().warn("Occupancy grid not set yet. Waiting for map data...", throttle_duration_sec=1.0)
            return
        if not hasattr(self, 'cur_pos') or not hasattr(self, 'cur_yaw'):
            self.get_logger().warn("Current position or yaw not set yet. Waiting for pose data...", throttle_duration_sec=1.0)
            return
        
        # Convert the scan data to a numpy array
        scan_data = np.array(scan_msg.ranges)

        cur_grid_x, cur_grid_y = get_grid_coordinates(self.cur_pos[0], self.cur_pos[1], self.map_origin, self.map_resolution)

        cur_grid_x = np.clip(cur_grid_x, 0, self.map_width - 1)
        cur_grid_y = np.clip(cur_grid_y, 0, self.map_height - 1)

        updated_occupancy_grid = np.copy(self.occupancy_grid)
        for i, distance in enumerate(scan_data):
            if distance == float('inf'):
                distance = scan_msg.range_max

            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            angle_world = angle + self.cur_yaw

            updated_occupancy_grid = update_grid_with_ray(
                updated_occupancy_grid,
                cur_grid_x,
                cur_grid_y,
                angle_world,
                distance,
                self.map_resolution,
                self.map_width,
                self.map_height,
                area_size=self.expand_occ_size
            )

        self.occupancy_grid = updated_occupancy_grid
        self.publish_occupancy_grid()

        # self.get_logger().info(f"Updated occupancy grid with scan data at position: {self.cur_pos}, yaw: {self.cur_yaw}")

    def publish_occupancy_grid(self):
        # Create and publish the updated occupancy grid
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = "map"

        occupancy_grid_msg.info.height = self.map_height
        occupancy_grid_msg.info.width = self.map_width
        occupancy_grid_msg.info.resolution = self.map_resolution
        occupancy_grid_msg.info.origin.position.x = self.map_origin[0]
        occupancy_grid_msg.info.origin.position.y = self.map_origin[1]

        occupancy_grid_msg.data = self.occupancy_grid.flatten().astype(np.int8).tolist()

        # Publish the updated occupancy grid
        self.occupancy_grid_publisher_.publish(occupancy_grid_msg)
       

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()