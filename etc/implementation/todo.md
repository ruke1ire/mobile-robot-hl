# TODO

The todo list is written to this file.

- [ ] Implement supervisor_node
    - [ ] Design GUI design
    - [ ] Implement every service and topic
    - [x] Create simple GUI
    - [x] Setup base-line ros communication topics and services
- [ ] Implement agent_node
    - [ ] Implement every service and topic
    - [x] Setup base-line ros communication topics and services
- [ ] Implement trainer_node
    - [ ] Implement every service and topic
    - [x] Setup base-line ros communication topics and services
    - [ ] Implement neural network model
- [x] Other nodes
    - [ ] Add launch files for starting up all the other necessary nodes 
        - ros2 run image_transport republish compressed in/compressed:=image_raw/compressed raw out:=image_raw/uncompressed
        - ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/diffbot_base_controller/cmd_vel_unstamped
        - ros2 run rviz2 rviz2 -d ./src/control/ros2_control_demo_description/diffbot_description/config/diffbot.rviz
    - [x] Install and buy necessary hardware for robot parts
- [x] Create custom interfaces
    - [x] agent_output

