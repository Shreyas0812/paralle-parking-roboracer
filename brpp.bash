source /opt/ros/humble/setup.bash

colcon build --packages-select parallel_parking

source install/setup.bash

ros2 launch parallel_parking parallel_parking_visualize.launch.py 
