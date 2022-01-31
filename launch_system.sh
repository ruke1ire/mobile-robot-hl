#!/usr/bin/bash

source /opt/ros/foxy/setup.bash
source install/setup.bash
source src/utils/init.sh

ros2 launch mobile_robot_hl full_system_launch.py
