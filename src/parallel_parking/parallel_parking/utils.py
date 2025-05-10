import csv
import numpy as np

def load_waypoints(waypoint_file_path):
    waypoints = []
    try:
        with open(waypoint_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # Skip the header
            for row in reader:
                x = float(row[0])
                y = float(row[1])
                yaw = float(row[2])
                qw = float(row[3])
                qx = float(row[4])
                qy = float(row[5])
                qz = float(row[6])
                waypoints.append((x, y, yaw, qw, qx, qy, qz))
    except Exception as e:
        print(f"Failed to read waypoints: {e}")

    return waypoints

def get_grid_coordinates(x, y, map_origin, map_resolution) -> tuple[int, int]:
        """
        Convert world coordinates to grid coordinates
        Args:
            x (float): x coordinate in world frame
            y (float): y coordinate in world frame
            map_origin (tuple): origin of the map in world frame
            map_resolution (float): resolution of the map
        Returns:
            (int, int): x and y coordinates in grid frame
        """
        grid_x = int((x - map_origin[0]) / map_resolution)
        grid_y = int((y - map_origin[1]) / map_resolution)

        return grid_x, grid_y
    
def get_world_coordinates(grid_x, grid_y, map_origin, map_resolution) -> tuple[float, float]:
    """
    Convert grid coordinates to world coordinates
    Args:
        grid_x (int): x coordinate in grid frame
        grid_y (int): y coordinate in grid frame
    Returns:
        (float, float): x and y coordinates in world frame
    """
    world_x = grid_x * map_resolution + map_origin[0]
    world_y = grid_y * map_resolution + map_origin[1]

    return world_x, world_y

def update_grid_with_ray(
    grid: np.ndarray,
    start_x: int,
    start_y: int,
    angle: float,  # Input in radians
    distance: float,
    map_resolution: float,
    map_width: int,
    map_height: int,
    area_size: int
) -> np.ndarray:
    """
    Updates an occupancy grid by tracing a ray from (start_x, start_y) to a point at 
    `distance` meters away along `angle`, marking free cells along the ray and occupied 
    cells in a square area around the endpoint.

    Args:
        grid (np.ndarray): 2D occupancy grid (0=free, 100=occupied).
        start_x (int): Starting x-coordinate in grid cells.
        start_y (int): Starting y-coordinate in grid cells.
        angle (float): Ray angle in radians.
        distance (float): Maximum ray distance in meters.
        map_resolution (float): Grid resolution (meters/cell).
        map_width (int): Grid width in cells.
        map_height (int): Grid height in cells.
        area_size (int): Half-size of the square area around the endpoint to mark as occupied.

    Returns:
        np.ndarray: Updated occupancy grid.
    """
    # Validate inputs
    if not (0 <= start_x < map_width and 0 <= start_y < map_height):
        raise ValueError("Start position out of grid bounds.")
    if map_resolution <= 0:
        raise ValueError("map_resolution must be positive.")

    # Calculate endpoint
    range_in_cells = int(round(distance / map_resolution))
    end_x = int(round(start_x + range_in_cells * np.cos(angle)))
    end_y = int(round(start_y + range_in_cells * np.sin(angle)))

    # Bresenham's line algorithm
    x, y = start_x, start_y
    dx = abs(end_x - x)
    dy = -abs(end_y - y)
    sx = 1 if end_x > x else -1
    sy = 1 if end_y > y else -1
    err = dx + dy

    max_x = map_width - 1
    max_y = map_height - 1

    while True:
        if 0 <= x <= max_x and 0 <= y <= max_y:
            grid[y, x] = 0  # Free cell

        if x == end_x and y == end_y:
            break

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    # Mark occupied area around endpoint
    for dx in range(-area_size, area_size + 1):
        for dy in range(-area_size, area_size + 1):
            new_x = end_x + dx
            new_y = end_y + dy
            if 0 <= new_x < map_width and 0 <= new_y < map_height:
                grid[new_y, new_x] = 100  # Occupied cell

    return grid


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].
    
    Args:
        angle (float): Angle in radians.
        
    Returns:
        float: Normalized angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def generateAckermannWaypoints(
        start_x: float = 0.0,
        start_y: float = 0.0,
        start_yaw: float = 0.0,
        goal_x: float = 0.0,
        goal_y: float = 0.0,
        goal_yaw: float = 0.0,
        wheelbase: float = 0.32,
        dt: float = 0.1,
        max_steering_angle: float = 0.52,
        velocity: float = 1.0,
):
    """
    Generate waypoints for an Ackermann vehicle from start to goal position/orientation
    
    Args:
        start_x, start_y: Initial position
        start_yaw: Initial orientation (radians)
        goal_x, goal_y: Goal position
        goal_yaw: Goal orientation (radians)
        wheelbase: Vehicle wheelbase (distance between front and rear axles)
        dt: Time step for simulation
        max_steering_angle: Maximum steering angle in radians
        velocity: Constant velocity for path generation
        
    Returns:
        waypoints: List of (x, y, yaw) tuples representing the vehicle path
    """

    # Initialize current state
    x, y, yaw = start_x, start_y, start_yaw

    k_p = 0.5  
    lookahead_distance = 0.5  

    pos_threshold = 1.0
    yaw_threshold = np.deg2rad(5)

    waypoints = [(x, y, yaw)]

    max_time = 30.0
    time = 0.0

    while time < max_time:
        dist_to_goal = np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

        if dist_to_goal < pos_threshold:
            break
            yaw_error = normalize_angle(goal_yaw - yaw)
            if abs(yaw_error) < yaw_threshold:
                break # Reached goal

            steering_angle = np.clip(k_p * yaw_error, -max_steering_angle, max_steering_angle)

            current_velocity = min(velocity, 1.0)

        else:
            angle_to_goal = np.arctan2(goal_y - y, goal_x - x)
            yaw_error = normalize_angle(angle_to_goal - yaw)

            ld = min(lookahead_distance, dist_to_goal)
            steering_angle = np.arctan2(2 * wheelbase * np.sin(yaw_error), ld)

            steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)

            current_velocity = velocity
        
        # Update state
        x += current_velocity * np.cos(yaw) * dt
        y += current_velocity * np.sin(yaw) * dt
        yaw += (current_velocity / wheelbase) * np.tan(steering_angle) * dt
        yaw = normalize_angle(yaw)

        waypoints.append((x, y, yaw))

        time += dt
        
    return waypoints

def generate_s_curve_waypoints(
<<<<<<< Updated upstream
    start_x, start_y, goal_x, goal_y, num_points=50, epsilon=0.01
):
    """
    Generate waypoints along an S-shaped sigmoid curve from start to goal.
    Returns: list of (x, y, yaw) tuples
    """
    # Ensure start_x < goal_x for parameterization
    if goal_x < start_x:
        start_x, goal_x = goal_x, start_x
        start_y, goal_y = goal_y, start_y

    # Sigmoid parameters
    L = start_y
    U = goal_y

    # Steepness parameter for sigmoid
    k = 2 * np.log((1 - epsilon) / epsilon) / (goal_x - start_x)
    x0 = (start_x + goal_x) / 2

    def sigmoid(x):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def y_curve(x):
        return L + (U - L) * sigmoid(x)

    # Sample points between start_x and goal_x
    x_vals = np.linspace(start_x, goal_x, num_points)
    y_vals = y_curve(x_vals)

    # Compute yaw (heading) at each point using the derivative
    dy_dx = np.gradient(y_vals, x_vals)
    yaws = np.arctan2(dy_dx, 1.0)

    waypoints = [(float(x), float(y), float(yaw)) for x, y, yaw in zip(x_vals, y_vals, yaws)]
    return waypoints


=======
    start_x, start_y,
    goal_x,  goal_y,
    num_points=50,
    epsilon=1e-2,
    horizontal_first=True
):
    """
    Generate waypoints along an S-shaped sigmoid curve.

    If horizontal_first is False (default):
        1. x is the parameter we sample,  y = f(x)
        2. The vehicle starts heading roughly along +x  (horizontal).

    If horizontal_first is True:
        1. y is the parameter we sample,  x = f(y)
        2. The vehicle starts heading roughly along +y  (vertical).

    Returns
    waypoints : list[(float, float, float)]
        Tuples of (x, y, yaw) in radians.
    """

    def build_sigmoid(L, U, p_min, p_max, p):
        k  = 2.0 * np.log((1 - epsilon) / epsilon) / (p_max - p_min)
        p0 = 0.5 * (p_min + p_max)
        sigma = 1.0 / (1.0 + np.exp(-k * (p - p0)))
        return L + (U - L) * sigma

    if horizontal_first is False:
        # sample x from start_x ➜ goal_x
        x_vals = np.linspace(start_x, goal_x, num_points)
        y_vals = build_sigmoid(start_y, goal_y, start_x, goal_x, x_vals)

        # dy/dx
        dy_dx = np.gradient(y_vals, x_vals)
        yaws  = np.arctan2(dy_dx, 1.0)       # (dx,dy) ≈ (1,dy/dx)

    else:                                    # horizontal‑first variant
        # sample y from start_y ➜ goal_y
        y_vals = np.linspace(start_y, goal_y, num_points)
        x_vals = build_sigmoid(start_x, goal_x, start_y, goal_y, y_vals)

        # dx/dy
        dx_dy = np.gradient(x_vals, y_vals)
        yaws  = np.arctan2(1.0, dx_dy)       # (dx,dy) ≈ (dx/dy,1)

    waypoints = [
        (float(x), float(y), float(yaw))
        for x, y, yaw in zip(x_vals, y_vals, yaws)
    ]
    return waypoints

### work for top slot
>>>>>>> Stashed changes
# def generate_s_curve_waypoints(
#     start_x, start_y, goal_x, goal_y, start_yaw=0.0, goal_yaw=0.0, num_points=50, invert=False, epsilon=0.01
# ):
#     """
#     Generate S-curve waypoints from start to goal, aligned with start and goal pose.
#     Returns: list of (x, y, yaw) tuples in global frame.
#     """
#     # Compute local vector from start to goal
#     dx = goal_x - start_x
#     dy = goal_y - start_y
#     distance = np.hypot(dx, dy)
#     angle = np.arctan2(dy, dx)

#     # Generate S-curve in local frame (from (0,0) to (distance, 0))
#     x_vals = np.linspace(0, distance, num_points)
#     # S-curve in y: goes from 0 to 0, with S-shape in between
#     k = 2 * np.log((1 - epsilon) / epsilon) / distance
#     x0 = distance / 2

#     # S-curve: y = amplitude * (sigmoid(x) - 0.5)
#     amplitude = distance / 3  # Adjust for how "curvy" you want it
#     if invert:
#         amplitude = -amplitude

#     def sigmoid(x):
#         return 1 / (1 + np.exp(-k * (x - x0)))

#     y_vals = amplitude * (sigmoid(x_vals) - 0.5) * 2  # Range: -amplitude to +amplitude

#     # Compute yaw at each point (tangent angle)
#     dy_dx = np.gradient(y_vals, x_vals)
#     yaws = np.arctan2(dy_dx, 1.0)

#     # Now rotate and translate to global frame
#     cos_a = np.cos(1.5*angle)
#     sin_a = np.sin(1.5*angle)
#     waypoints = []
#     for x_local, y_local, yaw_local in zip(x_vals, y_vals, yaws):
#         x_global = cos_a * x_local - sin_a * y_local + start_x +0.5
#         y_global = sin_a * x_local + cos_a * y_local + start_y
#         yaw_global = normalize_angle(yaw_local + angle)
#         waypoints.append((x_global, y_global, yaw_global))
#     return waypoints

