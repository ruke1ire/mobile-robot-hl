# Design

The design ideas of this package is discussed in this file. Note that the ros2 framework is used therefore terms such as "node", "topic", and "service" refers to their definition in the ros2 framework.

## Required Functionalities

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

**User Interface**

- Display the information the agent is conditioned on from the agent_in topic
- Display the current and previous output of the agent from the agent_output topic
- User should be able to simply start/pause/stop the automatic control as well as save the information in the current episode to be used for training the model.
- User should be able to select (or random) a demonstration and condition the agent model with that demonstration.
- User should be able to condition model with manual control.
- User should be able to start/stop the training of the model.

**Params**
- Demonstration file path

**Topics**
- Publishes to desired_velocity (QOS: Reliable)
- Subscribes to agent_output (QOS: Reliable)
- Subscribes to agent_input (QOS: Reliable)
- Subscribes to user_input 
    - velocity
    - termination_flag

**Other information**
- When creating user demonstrations or supervisor take-overs, the frequency of the control output from the supervisor will be limited to the control frequency of the agent node therefore this requires the agent node to be present.
- 

### Agent Node

The agent node outputs the automatic control signals using a neural network. 

**Params**
- CPU/GPU for neural network inference
- Demonstration file path
- Control frequency
- Path to neural networks

**Topics**
- Publishes to agent_output (QOS: Reliable)
    - predicted_velocity
    - predicted_termination_flag
- Publishes to agent_input (QOS: Reliable)
- Subscribes to image_raw (QOS: Best effort)
- Subscribes to desired_velocity (QOS: Reliable)

**Services**
- start: Start automatic control
- pause: Pause automatic control
- stop: Stop automatic control and clear current episode data
- save: Save the current episode to demonstration file
- condition_model: 
    - user demonstration
    - supervisor actions

### Trainer Node

The trainer node trains the neural networks.

**Params**
- CPU/GPU for neural network training
- Demonstration file path
- Path to neural networks

**Services**
- start: Start training model
- stop: Stop training the model
- save: Save the current model to a specific name/path
