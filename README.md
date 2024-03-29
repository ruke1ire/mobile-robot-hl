# mobile-robot-high-level

This repo contains code that allows mobile robots to be trained for any navigation task which uses a monocular camera as its perception. It borrows concepts from meta-learning, imitation learning, and reinforcement learning. This mobile robot controller is a high-level controller that publishes and subscribes to ros2 topics of a [low-level controller](https://github.com/ruke1ire/mobile-robot-base).

## Files

- System design can be found in [etc/design/system_design.md](https://github.com/ruke1ire/mobile-robot-hl/blob/test_branch/etc/design/system_design.md)
- Neural network design can be found in [etc/design/neural_network.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/neural_network.md) and the implementation details can be found in [etc/implementation/neural_network.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/implementation/neural_network.md)
- Neural Network trainer command protocol can be found in [etc/implementation/trainer_protocol.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/implementation/trainer_protocol.md).
- Todo list can be found in [etc/implementation/todo.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/implementation/todo.md)

## Requirements

- [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation.html)

Install additional requirements
```
pip3 install -r ./etc/requirements/requirements.txt
```

## Startup

### Source files

```
source /opt/ros/foxy/setup.bash
source install/setup.bash
source src/utils/init.sh
```

### Launch ROS nodes

```
ros2 launch mobile_robot_hl full_system_launch.py 
ros2 launch mobile_robot_hl mock_system_launch.py # launches the system along with a mock_up node that publishes the raw images
```

### Start trainer.console

```
./trainer.sh
```

### Transfer data files

#### Demo, task, run_setup, source -> server
```
./send.sh -dtrs
```

#### Model -> local

```
./recv.sh -m
```