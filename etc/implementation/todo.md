# TODO

The todo list is written to this file.

## In Progress

- [ ] Implement supervisor_node
    - [ ] Implement every service and topic
    - [ ] Restructure the functions according to the efficient information structure for each scenario
    - [ ] Create classes for demonstration and task episodes and they should have functions to help append/set/get information with ease
        - Appending information should be to add the observation
        - Setting the information should be to set the actions performed for that observation
        - Get function should assist in returning specific information in a specific data structure
    - [ ] Create function to display information in the current episode
        - [x] demonstration
        - [ ] saved task episdoe
        - [ ] current task episode
    - [ ] Use blit for matplotlib
    - [ ] Create list box for task episodes
- [ ] Implement agent_node
    - [ ] Implement every service and topic
- [ ] Implement trainer_node
    - [ ] Implement every service and topic
    - [ ] Implement neural network model
- [ ] Other nodes
    - [ ] Add launch files for starting up all the other necessary nodes 
        - ros2 run image_transport republish compressed in/compressed:=image_raw/compressed raw out:=image_raw/uncompressed
        - ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/diffbot_base_controller/cmd_vel_unstamped
        - ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=user_input/velocity
        - ros2 run rviz2 rviz2 -d ./src/control/ros2_control_demo_description/diffbot_description/config/diffbot.rviz
    - [ ] Create node to input actions from an xbox controller

## Done

- [ ] Implement supervisor_node
    - [x] Design GUI 
    - [x] Create simple GUI
    - [x] Setup base-line ros communication topics and services
    - [x] Create an entry for selecting the demonstration whenever starting the automatic controller, and make it required to select a demonstration, otherwise, disable the automatic start button
    - [x] Test out demo_handler and task_handler
    - [x] Add another listbox for the demo selection process, one should be to select the demo name, the other should be to select the demo id
    - [x] Finish function to save/delete the task episodes using the task handler
- [ ] Implement agent_node
    - [x] Setup base-line ros communication topics and services
- [ ] Implement trainer_node
    - [x] Setup base-line ros communication topics and services
- [ ] Other nodes
    - [x] Install and buy necessary hardware for robot parts
- [ ] Create custom interfaces
    - [x] agent_output