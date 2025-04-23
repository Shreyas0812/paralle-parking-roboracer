#!/usr/bin/env python3

import time, os, sys
import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32
from ament_index_python.packages import get_package_share_directory

from infer_env import InferEnv
from mppi_tracking import MPPI
import utils.utils as utils
from utils.jax_utils import numpify
import utils.jax_utils as jax_utils
from utils.Track import Track

from visualization_msgs.msg import Marker

# jax.config.update("jax_compilation_cache_dir", "/home/nvidia/jax_cache") 
jax.config.update("jax_compilation_cache_dir", "/cache/jax")

## This is a demosntration of how to use the MPPI planner with the Roboracer
## Zirui Zang 2025/04/07

class MPPI_Node(Node):
    def __init__(self):
        super().__init__('lmppi_node')
        self.config = utils.ConfigYAML()
        config_dir = get_package_share_directory('mppi')
        config_path = os.path.join(config_dir, 'config.yaml')
        self.config.load_file(config_path)
        self.config.norm_params = np.array(self.config.norm_params).T
        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)    
        map_dir = os.path.join(config_dir, 'waypoints/')
        map_info = np.genfromtxt(map_dir + 'map_info.txt', delimiter='|', dtype='str')
        track, self.config = Track.load_map(map_dir, map_info, self.config.map_ind, self.config)
        # track.waypoints[:, 3] += 0.5 * np.pi
        self.infer_env = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = MPPI(self.config, self.infer_env, jrng)

        # Do a dummy call on the MPPI to initialize the variables
        state_c_0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.control = np.asarray([0.0, 0.0]) # [steering angle, speed]
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0.copy(), self.config.ref_vel, self.config.n_steps)
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        self.get_logger().info('MPPI initialized')
        
        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=10,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        
        # create subscribers
        self.grid_sub = self.create_subscription(OccupancyGrid, "/occupancy_grid", self.grid_callback, qos)        

        if self.config.is_sim:
            self.pose_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, qos)
        else:
            self.pose_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pose_callback, qos)

        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)
        
        self.obstacle_marker_pub = self.create_publisher(Marker, "/obstacle_points_marker", 10)

        # map info
        self.map_received = False
        self.width = None
        self.height = None
        self.resolution = None
        self.origin = None
        self.occup_pos = None


    # @TODO: add a callback for the laser scan data and get the scan data from obstacles to add the cost to the MPPI
    def grid_callback(self, grid_msg):
        if self.width is None or self.height is None or self.resolution is None or self.origin is None:
            self.width = grid_msg.info.width
            self.height = grid_msg.info.height
            self.resolution = grid_msg.info.resolution
            self.origin = grid_msg.info.origin.position

        grid = np.array(grid_msg.data).reshape((self.height, self.width))
        occupied_indices = np.argwhere(grid >= 100)[::5]
        occupied_pos = self.grid_to_world_batch(occupied_indices, self.origin, self.resolution)
        self.occup_pos = occupied_pos

        self.map_received = True
        
    def grid_to_world_batch(self, occupied_indices, origin, resolution):
        i = occupied_indices[:, 0]
        j = occupied_indices[:, 1]

        x = origin.x + j * resolution + resolution / 2.0
        y = origin.y + i * resolution + resolution / 2.0

        return np.stack((x, y), axis=-1)  # shape: [N, 2]

    def uniform_resample(self, obstacles: np.ndarray, max_obstacles: int) -> np.ndarray:
        """
        If len(obs) > max_obstacles: uniformly subsample.
        If len(obs) < max_obstacles: randomly replicate existing points
        (sampling with replacement) until you have exactly max_obstacles.
        Returns an array of shape (max_obstacles,2).
        """
        M = obstacles.shape[0]
        if M == 0:
            return np.zeros((max_obstacles, 2), dtype=obstacles.dtype)

        if M >= max_obstacles:
            # down-sample uniformly
            idxs = np.linspace(0, M - 1, max_obstacles, dtype=int)
            return obstacles[idxs]

        # up-sample by random sampling WITH replacement
        # keep all original M, then sample (max_obstacles - M) extras
        extra_n = max_obstacles - M
        # choose random indices from [0..M-1]
        extra_idxs = np.random.choice(M, size=extra_n, replace=True)
        extras = obstacles[extra_idxs]
        # concatenate original + extras
        out = np.vstack([obstacles, extras])

        return out

    def filtering_roi_obstacles(self, state, obstacle_world_coords, roi_area=(5.0, 3.0), max_obstacles=200):
        '''
        Filters obstacles within a rectangular ROI in front of the car.
        
        Args:
            state (np.ndarray): current vehicle state [x, y, steering, v, yaw, ..., ...]
            obstacle_world_coords (np.ndarray): shape [N, 2] array of world-frame (x, y) obstacle positions
            roi_area (tuple): ROI dimensions (length, width), in meters

        Returns:
            np.ndarray: filtered (x, y) world coordinates of obstacles in ROI
        '''
        x, y, yaw = state[0], state[1], state[4]
        dx = obstacle_world_coords[:, 0] - x
        dy = obstacle_world_coords[:, 1] - y

        # Transform to robot frame
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        x_local = cos_yaw * dx - sin_yaw * dy
        y_local = sin_yaw * dx + cos_yaw * dy

        length, width = roi_area
        x_min, x_max = 0.0, length
        y_min, y_max = -width / 2.0, width / 2.0

        # Filter points inside ROI
        mask = (x_local >= x_min) & (x_local <= x_max) & \
            (y_local >= y_min) & (y_local <= y_max)
        filtered_obstacles = obstacle_world_coords[mask]

        filtered_obstacles = self.uniform_resample(filtered_obstacles, max_obstacles)    

        return filtered_obstacles

    def publish_obstacle_points(self, obstacle_world_coords):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Visual appearance
        marker.scale.x = 0.1  # point width
        marker.scale.y = 0.1  # point height
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Add points
        for x, y in obstacle_world_coords:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        self.obstacle_marker_pub.publish(marker)

    def world_to_grid(self, x, y, origin, resolution):
        i = int((y - origin.y) / resolution)
        j = int((x - origin.x) / resolution)
        return i, j

    def pose_callback(self, pose_msg):
        """
        Callback function for subscribing to particle filter's inferred pose.
        This funcion saves the current pose of the car and obtain the goal
        waypoint from the pure pursuit module.

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        """
        if not self.map_received:
            self.get_logger().warning("Waiting for map data...")
            return

        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist

        # Beta calculated by the arctan of the lateral velocity and the longitudinal velocity
        beta = np.arctan2(twist.linear.y, twist.linear.x)

        # For demonstration, let’s assume we have these quaternion values
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert quaternion to Euler angles
        euler = tf_transformations.euler_from_quaternion(quaternion)

        # Extract the Z-angle (yaw)
        theta = euler[2]  # Yaw is the third element

        state_c_0 = np.asarray([
            pose.position.x,
            pose.position.y,
            self.control[0],
            max(twist.linear.x, self.config.init_vel),
            theta,
            twist.angular.z,
            beta,
        ])
        filtered_obstacles = self.filtering_roi_obstacles(state_c_0, self.occup_pos)
        self.publish_obstacle_points(filtered_obstacles)
        
        find_waypoint_vel = max(self.config.ref_vel, twist.linear.x)
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0.copy(), find_waypoint_vel, self.config.n_steps)

        ## MPPI call
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj), jnp.asarray(filtered_obstacles))
        # self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2]/2
        self.control[0] = float(mppi_control[0]) * self.config.sim_time_step + self.control[0]
        self.control[1] = float(mppi_control[1]) * self.config.sim_time_step + twist.linear.x
        
        if self.reference_pub.get_subscription_count() > 0:
            ref_traj_cpu = numpify(reference_traj)
            arr_msg = to_multiarray_f32(ref_traj_cpu.astype(np.float32))
            self.reference_pub.publish(arr_msg)

        if self.opt_traj_pub.get_subscription_count() > 0:
            opt_traj_cpu = numpify(self.mppi.traj_opt)
            arr_msg = to_multiarray_f32(opt_traj_cpu.astype(np.float32))
            self.opt_traj_pub.publish(arr_msg)

        if twist.linear.x < self.config.init_vel:
            self.control = [0.0, self.config.init_vel * 2]

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        # Publish the control command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = self.control[1]
        self.get_logger().info(f"Steering Angle: {drive_msg.drive.steering_angle}, Speed: {drive_msg.drive.speed}")
        self.drive_pub.publish(drive_msg)
        

def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI_Node()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()