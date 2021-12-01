# TODO

The todo list is written to this file.

## In Progress

- [ ] Make global config file?
- [ ] Think about what can be logged 
- [ ] Make the noise function better
- [ ] Make sure all the images are normalized correctly
    - training
    - inference
        - demo
        - live
- [ ] Supervisor
    - [ ] When saving the episode, put information about the control frequency too?
- [ ] Record demonstrations
    - about 20 path following tasks
- [ ] Train with RL
- Make neural network lighter
    - [ ] Convert all information to be in 16 bit
        - https://medium.com/@dwightfoster03/fp16-in-pytorch-a042e9967f7e
    - [ ] https://pytorch.org/docs/master/amp.html Autocast
- [ ] algorithms.py
    - implement il algorithm
- [ ] Put this on the mobile_robot_base repo
    - ros2 run rviz2 rviz2 -d ./src/control/ros2_control_demo_description/diffbot_description/config/diffbot.rviz