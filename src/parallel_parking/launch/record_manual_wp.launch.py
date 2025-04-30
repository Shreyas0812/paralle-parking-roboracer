from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='parallel_parking',
            executable='record_manual_wp_node.py',
            name='record_manual_wp_node',
            output='screen',
            parameters=[{
                'goal_pose_topic': '/goal_pose',
                'visualize_wp_topic': '/visualization/waypoints',
                # 'waypoint_file_path': '/sim_ws/src/parallel_parking/config/waypoints_manual.csv'
                'waypoint_file_path': '/home/shreyas/Documents/ESE6150_F1_Tenth/parallel-parking-roboracer/src/parallel_parking/config/waypoints_manual.csv'
            }]
        )
    ])