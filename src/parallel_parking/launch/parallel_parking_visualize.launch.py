from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    waypoint_file_name = "waypoints_agh300a_bf.csv"
    
    scan_topic = "/scan"
    pose_topic = "/ego_racecar/odom"
    visualize_wp_topic = "/visualization/waypoints"
    next_wp_topic = "/next_waypoint"
    original_map_topic = "/map"
    occupancy_grid_topic = "/occupancy_grid"
    extrapolated_path_topic = "/extrapolated_path"
    
    lookahead_distance = 0.1
    y_ego_threshold = 1.2
    
    map_height = 713
    map_width = 727
    map_resolution = 0.05
    map_origin = [-20.2, -5.68]

    expand_occ_size = 1

    return LaunchDescription([
        # parallel_parking node
        Node(
            package="parallel_parking",
            executable="parallel_parking_node.py",
            name="parallel_parking_node",
            output="screen",
            parameters=[
                {"pose_topic": pose_topic},
                {"occupancy_grid_topic": occupancy_grid_topic},
                {"next_wp_topic": next_wp_topic},
                {"map_height": map_height},
                {"map_width": map_width},
                {"map_resolution": map_resolution},
                {"map_origin": map_origin}
            ]
        ),
        # Visualize node
        Node(
            package="parallel_parking",
            executable="visualize_node.py",
            name="visualize_node",
            output="screen",
            parameters=[
                {"waypoint_file_name": waypoint_file_name},
                {"visualize_wp_topic": visualize_wp_topic},
            ]
        ),
        # Occupancy grid node
        Node(
            package="parallel_parking",
            executable="occupancy_grid_node.py",
            name="occupancy_grid_node",
            output="screen",
            parameters=[
                {"scan_topic": scan_topic},
                {"pose_topic": pose_topic},
                {"original_map_topic": original_map_topic},
                {"occupancy_grid_topic": occupancy_grid_topic},
                {"map_height": map_height},
                {"map_width": map_width},
                {"map_resolution": map_resolution},
                {"map_origin": map_origin},
                {"expand_occ_size": expand_occ_size},
            ]
        ),
        # Next waypoint node
        Node(
            package="parallel_parking",
            executable="next_waypoint_node.py",
            name="next_waypoint_node",
            output="screen",
            parameters=[
                {"waypoint_file_name": waypoint_file_name},
                {"pose_topic": pose_topic},
                {"next_wp_topic": next_wp_topic},
                {"lookahead_distance": lookahead_distance},
                {"y_ego_threshold": y_ego_threshold}
            ]
        ),
        # Traj Generation node
        Node(
            package="parallel_parking",
            executable="traj_gen_node.py",
            name="traj_gen_node",
            output="screen",
            parameters=[
                {"waypoint_file_name": waypoint_file_name},
                {"pose_topic": pose_topic},
                {"extrapolated_path_topic": extrapolated_path_topic},
                {"wp1_dist_thresh": 0.5},
                {"wp2_dist_thresh": 0.2},
                {"wp1_angle_thresh": 3.14},
                {"wp2_angle_thresh": 0.15},
                {"switch_wp_index": 1}
            ]
        ),
        # MPPI node
        Node(
            package="parallel_parking",
            executable="mppi_node.py",
            name="mppi_node",
            output="screen",
            parameters=[
                {"pose_topic": pose_topic},
                {"extrapolated_path_topic": extrapolated_path_topic},
            ]
        ),
    ])