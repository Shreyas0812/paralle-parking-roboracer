from utils import load_waypoints, get_world_coordinates

import numpy as np

import csv

# Load grid waypoints
grid_waypoints = load_waypoints('/home/shreyas/Documents/ESE6150_F1_Tenth/parallel-parking-roboracer/src/parallel_parking/config/grid_waypoints.csv')

print(grid_waypoints)