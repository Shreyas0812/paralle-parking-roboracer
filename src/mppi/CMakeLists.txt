cmake_minimum_required(VERSION 3.8)
project(mppi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros)
find_package(parallel_parking_interfaces REQUIRED)


# Python
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  infer_env.py
  mppi_node.py
  mppi_tracking.py
  vis_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY dynamics_models utils
  DESTINATION lib/${PROJECT_NAME}
)

# yaml
install(FILES config.yaml config_new_map.yaml config_map_edited.yaml config_park.yaml
  DESTINATION share/${PROJECT_NAME}
)

# Directories
install(DIRECTORY waypoints
  DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()