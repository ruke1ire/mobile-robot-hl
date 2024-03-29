# Dataset

This file contains the high level information about the dataset.

## Tasks and behaviors

The following are the tasks and behaviors that will be demonstrated to train the agent.

### Tasks (Episodic)

- Line following
	- follow line and stop where demonstrator stopped
- Lane keeping
	- stay in lane and stop where demonstrator stopped
- Path following
	- follow the path the demonstrator took

### Behaviors (Non-Episodic)

- Line following
	- follow the line until max-steps is reached
- Lane keeping
	- stay in a land until max-steps is reached
- Object following
	- follow an object until max-steps is reached

### Default Behaviors 

- Obstacle avoidance
- Avoid falling from stairs

## Information

### Recording location

- Home (Tokyo, Saitama)
- University (Basement)

## Number of demonstrations

- 10 for each task/behavior/location

## Demonstration Dataset

- Numpy array Demonstration observations
	- Shape = (Time, Channels, Width, Height)
- Numpy array Demonstration actions
	- Shape = (Time, ActionSize)

## Task Episode Dataset

- Numpy array of Task Episode observations
	- Shape = (Time, Channels, Width, Height)
- Numpy array of Task Episode actions
	- Shape = (Time, ActionSize)
- Numpy array of Demonstration Flags
	- Shape = (Time, 1)
- Numpy array of Values
	- Shape = (Time, 1) 