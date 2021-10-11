# Design

The design ideas of this package is discussed in this file.

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

**Modes of operation**

- Manual control mode: let the supervisor manually control the vehicle
- Automatic control mode: let the supervisor monitor the automatic controller

**Params**
- Demonstration file path

**Topics**
- Publishes to desired_velocity (QOS: Reliable)
- Subscribes to agent_output (QOS: Reliable)
- Subscribes to agent_input (QOS: Reliable)

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
    - predicted_task_termination
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
