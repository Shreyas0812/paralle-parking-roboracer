import numpy as np

def generate_s_curve_waypoints(
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

waypoints = generate_s_curve_waypoints(
    start_x=-5.448110103607178,
    start_y=3.888681411743164,
    goal_x=-5.082984447479248,
    goal_y=3.9115869998931885,
    num_points=50
)

print(waypoints)