from utils import load_waypoints, get_world_coordinates

import numpy as np

import csv
# save waypoints to csv
def save_waypoints(waypoints, file_path):
    try:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'yaw'])  # Write header
            for waypoint in waypoints:
                writer.writerow(waypoint)
    except Exception as e:
        print(f"Failed to save waypoints: {e}")
# Load grid waypoints
grid_waypoints = load_waypoints("/home/yufeiyang/Documents/paralle-parking-roboracer/waypoints.csv")

print(grid_waypoints)

# Convert grid waypoints to world coordinates
resolution = 0.5
origin = (-5.13, -4.19)
world_waypoints = []
for waypoint in grid_waypoints:
    grid_x, grid_y, yaw = waypoint
    world_x, world_y = get_world_coordinates(grid_x, grid_y, origin, resolution)
    world_waypoints.append((world_x, world_y, yaw))
    # print(f"Grid coordinates: ({grid_x}, {grid_y}) -> World coordinates: ({world_x}, {world_y})")

output_file = "world_waypoints.csv"
save_waypoints(world_waypoints, output_file)