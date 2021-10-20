# Supervisor GUI Design

This file discusses the design of the Graphic interface for the supervisor node.

## Notes

- TKinter is used as the framework for building the GUI
- The functional requirement of the GUI is according to the [system design](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/system_design.md)

## Design

- Display
    - Information that the agent is conditioned on or show preview of user demonstration (play/pause/stop button, slider for selecting frame no., each frame will also accompany other information such as the action taken, demo flag, etc.)
    - Current and previous outputs of the agent and supervisor (image+matplotlib)
    - Live video stream (image)
- Controls
    - Start/Pause/Stop/Take-over automatic control (Buttons)
    - Start/Pause/Stop creating user demonstration (Buttons)
    - Save episode or user demonstration (Button + pop up text box)
    - Select/Queue user demonstration for agent (Add/remove button onto a list)
    - Start/Stop training of the model (Button)

Below is the GUI Draft.
![GUI Draft](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/gui.jpeg)

![GUI Current](https://github.com/ruke1ire/mobile-robot-hl/blob/main/etc/design/gui.png)
