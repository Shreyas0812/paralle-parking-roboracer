# paralle-parking-roboracer

### Steps to run:

Note: Make sure you are at the correct location as given by the text before `$`

```bash
/paralle-parking-roboracer/src$ git clone -b parallel-parking-roboracer --recurse-submodules https://github.com/Shreyas0812/f1tenth_gym_ros.git
```

```bash
/SBAMP/src/f1tenth_gym_ros/f1tenth_gym$ pip install -e .
```

```bash
/paralle-parking-roboracer$ rosdep install --from-paths src -y --ignore-src
```

#### manual waypoint logging:

```bash
paralle-parking-roboracer$ ros2 launch parallel_parking record_manual_wp.launch.py 
```
