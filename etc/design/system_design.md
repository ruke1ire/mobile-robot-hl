# System Design

The system design ideas of this package is discussed in this file. Note that the ros2 framework is used therefore terms such as "node", "topic", and "service" refers to their definition in the ros2 framework.

## Functional Requirements

1. Manual control
    - Create and save user demonstrations
2. Automatic control
    - Setup commands
        - Select/Queue user demonstrations
        - Select mode: Inference/Exploration
    - Control commands
        - Start: Start automatic control
        - Pause: Pause automatic control
        - Stop: Stop automatic control and discard data (sequence of observed images/command velocity) stored for current episode
        - Take-over: Pause automatic control and start saving data from the supervisor (manual control)
        - Save: Save current episode data
3. Training model
    - Imitation Learning Pre-training
        - Goes through all saved demonstrations and performs imitation learning
    - Reinforcement Learning
        - Goes through the replay buffer (demonstration+policy action+reward) to train the Q-network. Then use the Q-network to train the policy network.

## Nodes and functionalities

### Supervisor Node

The supervisor node handles all output/input to/from the supervisor (user).

***Functional Requirements***

- [ ] Display
    - [ ] Information that the agent is conditioned on
    - [ ] Current and previous outputs of the agent and supervisor
    - [ ] Live video stream
- [ ] Controls
    - [ ] Start/Pause/Stop automatic control
    - [ ] Start/Pause/Stop creating user demonstration
    - [ ] Save episode or user demonstration
    - [ ] Select/Queue user demonstration for agent (can random too)
    - [ ] Start/Stop training of the model

***Params***
- Demonstration file path

***Topics***
- Publishes to desired_velocity (QOS: Reliable)
- Publishes to termination_flag (QOS: Reliable)
- Subscribes to agent_output (QOS: Reliable)
- Subscribes to agent_input (QOS: Reliable)
- Subscribes to user_velocity (QOS: Best Effort, Twist)
- Subscribes to user_termination_flag (QOS: Best Effort, Bool)
- Subscribes to image_raw (QOS: Best Effort)

***Other information***
- Input for supervisor control will come from other external nodes such as teleop_twist_keyboard
- When creating user demonstrations or supervisor take-overs, the frequency of the control output from the supervisor will be limited to the control frequency of the agent node therefore this requires the agent node to be present.
- Supervisor take-over is similar to a user-demonstration therefore the start/pause buttons of the user demonstration can be used to start/pause the supervisor take-over. To restart the automatic control press the start button in the automatic control section.
- Defaults to manual control mode where there isn't any frequency limitation for controlling the mobile robot. This mode changes if the start buttons are pressed for automatic control or creating user demonstrations.
- Information panel should contain the following informations:
    - Selected demonstration
    - User action
    - Agent action
    - Controller
- GUI design can be found in [etc/design/supervisor_GUI_design.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/supervisor_GUI_design.md)
- Below is the state-flow-diagram 

![etc/design/supervisor_GUI_state_diagram.png](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/supervisor_GUI_state_diagram.png)

---

### Agent Node

The agent node outputs the automatic control signals using a neural network. 

***Functional Requirements***
- [ ] Controls
    - [ ] Select User demonstration
    - [ ] Start/Pause/Take-over/Stop automatic control
    - [ ] Select neural network model for agent (defaults to latest model)

***Params***
- CPU/GPU for neural network inference
- Demonstration file path
- Control frequency
- Path to neural networks

***Topics***
- Publishes to agent_output (QOS: Reliable)
    - predicted_velocity
    - predicted_termination_flag
- Publishes to agent_input (QOS: Reliable)
- Subscribes to image_raw (QOS: Best effort)
- Subscribes to desired_velocity (QOS: Reliable)
- Subscribes to termination_flag (QOS: Reliable)

***Services***
- start: Start automatic control or continue automatic control after supervisor take-over
- pause: Pause automatic control
- stop: Stop automatic control and clear current episode data
- take-over: Pause automatic control but continue to condition the model with supervisor input
- select_demonstration: Select a user demonstration to condition the model
- select_model: Select the neural network model for the agent
- select_mode: Select between inference or exploration mode

***Other information***
- This node will always output the agent_in at the specified control frequency
- The agent_output will only output when automatic control is started
- When take-over occurs, the desired_velocity and termination_flag will be used to condition the model.
- Directly subscribing to compressed visual images can be done through the image_transport_plugins but it is only available in c++. Therefore I will use the "image_transport republish" node to first convert the compressed image from the mobile-robot-base to raw images.

---

### Trainer Node

The trainer node trains the neural networks.

***Functional Requirements***
- [ ] Controls
    - [ ] Select model
    - [ ] Create and select model
    - [ ] Start/Pause training
        - Type of model can be specified
            - actor 
            - critic
        - Type of training can be specified
            - imitation learning
            - reinforcement learning
        - Epochs to train can be specified
        - Saving every X epochs can be specified
    - [ ] Restart neural network model
    - [ ] Save/Delete neural network

***Params***
- Demonstration file path
    - For imitation pre-training
- Task Episode file path
    - For reinforcement learning
- Path to neural networks
- Saving interval

***Services***
- command: Receives a string command and does whatever that string tells it to

***Other Information***
- 2 models can be trained in parallel
    - the actor and the critic model
- the ros node is only used as a communication tool and all the functionalities are implemented in the Trainer class
- Protocol for the string commands can be found [here](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/implementation/trainer_protocol.md).

## Episode Data Structure

The following section discusses the data structures that will be used to store information about the episode.

### dict(each_variable=list(variable))

- Eg: {'linear':[1,2,3,4,5], 'angular':[1,2,3,4,5]}
- Efficient for plotting, mini-batch model inference
- This structure is good for:
    - plotting across time
    - conditioning model
    - storing information

### list(dict(each variable))

- Eg: [{'linear':1, 'angular':1}, {'linear':2, 'angular':2}]
- Efficient for appending live information, single-data model inference
- This structure is good for:
    - plotting live information
    - live model inference
    - temporary data storange for recording demos/tasks

This program will primarily use the dict(list) data structure. But in order to make it easier to work with live inference/plotting, helper functions will be made to:
- 1. Get data -> returns data dict(each variable) format
- 2. Append data -> allows easily appending information for storing while recording demos/tasks

## Saved Files

### Demonstration Files

Demonstration files store observed images and actions performed in a demonstration.

- Information regarding the types of tasks/demonstrations are discussed in [etc/implementation/dataset.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/implementation/dataset.md)
- All demonstrations are saved into a path specified in the environment variable **MOBILE_ROBOT_HL_DEMO_PATH**
- The demonstration name is the name of the task
- Each unique demonstration name has a unique directory
    - There will be a yaml file which gives the information about this demonstration name
- Each unique demonstration also has a unique directory and a unique id
    - Inside this directory the sequence of images are stored with the file name corresponding to the ordering of the images in the demonstration
    - There will also be a yaml file which gives information about this demonstration file such as the supervisor action

### Task Episode Files

Task episode files stores the actions an agent/supervisor took when performing a task corresponding to a particular demonstration file.

- All task episodes are saved into a path specified in the environment variable **MOBILE_ROBOT_HL_TASK_PATH**
- Each task references a demonstration file, therefore it will inherit the same directory structure as the corresponding demonstration file
    - eg: if a task references a demonstration file in **MOBILE_ROBOT_HL_DEMO_PATH**/task1/1/ the data for that task is stored in **MOBILE_ROBOT_HL_TASK_PATH**/task1/1/
- The files inside the directory containing an episode are as follows:
    - observed images
    - yaml file containing information about 
        - agent actions
        - supervisor take-over action
        - reward?

### Neural Network Models

- Tensorboard will be used to visualize the training process
- Each training run will have a unique directory
- Inside the unique directory will be a yaml file containing information about the model such as the architecture, date-time of creation
- Each type of model such as the *actor model*, *critic model* will be saved in a separate directory within the same training run directory
    - each type of model will have a specific user defined model namme/id
    - there will be a yaml file inside the model directory which provides information about the model and the training process
- The model will be saved every *x* training iteration. The name of the file will increment from 1 onwards.