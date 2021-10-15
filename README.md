# mobile-robot-high-level

This repo contains code that allows mobile robots to be trained for any navigation task which uses a monocular camera as its perception. It borrows concepts from meta-learning, imitation learning, and reinforcement learning. This mobile robot controller is a high-level controller that publishes and subscribes to ros2 topics of a [low-level controller](https://github.com/ruke1ire/mobile-robot-base).

## Files

- System design can be found in [etc/design/design.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/system_design.md)
- Todo list can be found in [etc/implementation/todo.md](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/implementation/todo.md)

## Requirements

- [ros2 foxy](https://docs.ros.org/en/foxy/Installation.html)
