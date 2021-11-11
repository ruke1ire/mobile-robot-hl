# TODO

The todo list is written to this file.

## In Progress

- [ ] Think about what data structure is ideal for training the actor and the critic using td3
- [ ] Change the dataset class so that the data returned is in the ideal form
    - if i use a tuple of numpy arrays, the original collate_fn in the dataloader can be used.
- [ ] Implement supervisor_node
    - [ ] Can i do multiprocessing instead of multithreading?
    - [ ] Use blit for matplotlib
- [ ] Implement agent_node
    - [ ] Implement every service and topic
    - [ ] Let model do inference in agent_node
- [ ] Implement trainer_node
    - [ ] Implement every service and topic
    - [ ] Create trainer class
        - select neural network to train
        - train actor
        - train critic
- [ ] Other nodes
    - [x] Add launch files for starting up all the other necessary nodes 
        - ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/diffbot_base_controller/cmd_vel_unstamped
        - ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=user_input/velocity
        - ros2 run rviz2 rviz2 -d ./src/control/ros2_control_demo_description/diffbot_description/config/diffbot.rviz

## Done

- [ ] Implement supervisor_node
    - [x] Design GUI 
    - [x] Create simple GUI
    - [x] Setup base-line ros communication topics and services
    - [x] Create an entry for selecting the demonstration whenever starting the automatic controller, and make it required to select a demonstration, otherwise, disable the automatic start button
    - [x] Test out demo_handler and task_handler
    - [x] Add another listbox for the demo selection process, one should be to select the demo name, the other should be to select the demo id
    - [x] Finish function to save/delete the task episodes using the task handler
    - [x] Create classes for demonstration and task episodes and they should have functions to help append/set/get information with ease
        - Appending information should be to add the observation
        - Setting the information should be to set the actions performed for that observation
        - Get function should assist in returning specific information in a specific data structure
    - [x] Restructure the functions according to the efficient information structure for each scenario
    - [x] Supervisor gui should run a separate thread for updating the plots and images
    - [x] Create function to display information in the current episode
        - [x] demonstration
        - [x] saved task episdoe
        - [x] current task episode
    - [x] Create list box for task episodes
    - [x] Make sure agent information and supervisor information is in sync
        - [x] agent_output should be used to verify valid timesteps when using automatic control mode, demonstration doesn't need any validation because the user_output doesn't necessarily come at the same frequency as the controller
    - [x] Make the information panel better make it update with the correct information
    - [x] Make so that the episdoe data structure is always dict(list)
    - [x] Removed select_demo service in favour of only have start service with demo variable
    - [x] Implement every service and topic
    - [x] synchronize agent and supervisor because it sometimes isn't depending on when I click call the agent_start command | solved by adding time.sleep in the agent's service so that all callback's are first processed before another button is pressed
- [ ] Implement agent_node
    - [x] Setup base-line ros communication topics and services
    - [x] Create config file for setting the frequency
    - [x] Add reentrant callback group to the services
    - [x] Add model onto agent node
- [ ] Implement trainer_node
    - [x] Setup base-line ros communication topics and services
    - [x] List out all the commands that the trainer node should be able to do
- [ ] Other nodes
    - [x] Install and buy necessary hardware for robot parts
    - [x] Create node to input actions from an xbox controller | DO THIS NEXT!!
- [ ] Create custom interfaces
    - [x] agent_output
- [x] Implement neural network module
    - The dense blocks should not use the constant padding because sometimes we do not want any padding (the const padding should only be used on the very first frame)
- [x] Implement SNAIL model
    - [x] should be able to be used for both inference modes
    - [x] architecture design should be able to be inserted set when initializing the model
- [x] Find suitable NN for base layer CNN
- [x] Create full model
- [x] Implement model handler
- [x] Think about how and where new models should be created
    - create them in trainer nodes?