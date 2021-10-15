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

***Other information***
- Input for supervisor control will come from other external nodes such as teleop_twist_keyboard
- When creating user demonstrations or supervisor take-overs, the frequency of the control output from the supervisor will be limited to the control frequency of the agent node therefore this requires the agent node to be present.
- Supervisor take-over is similar to a user-demonstration therefore the start/pause buttons of the user demonstration can be used to start/pause the supervisor take-over. To restart the automatic control press the start button in the automatic control section.
- Defaults to manual control mode where there isn't any frequency limitation for controlling the mobile robot. This mode changes if the start buttons are pressed for automatic control or creating user demonstrations.
- GUI design can be found in [etc/design/supervisor_GUI_design.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/supervisor_GUI_design.md)

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
- Directly subscribing to raw visual images can be done through the image_transport_plugins but it is only available in c++. Therefore I will use the "image_transport republish" node to first convert the compressed image from the mobile-robot-base to raw images.

---

### Trainer Node

The trainer node trains the neural networks.

***Functional Requirements***
- [ ] Controls
    - [ ] Select model
    - [ ] Start/Pause training of actor and critic neural network
    - [ ] Restart neural network model
    - [ ] Save/Delete neural network
    - [ ] Pre-train actor model with imitation learning

***Params***
- CPU/GPU for neural network training
- Demonstration file path
- Path to neural networks
- Saving interval

***Services***
- select_model: Select model to be trained
- start: Start training model
- pause: Pause training the model
- stop: Stop training the model and restart it with new random weights
- save: Save the current model to a specific name/path
- delete: Delete neural networks
- pre-train: Pre-train the current actor model 