from enum import Enum
import torch

class ControllerType(Enum):
    NONE = 0
    USER = 1
    AGENT = 2

class SupervisorState(Enum):
    STANDBY = 0
    TASK_RUNNING = 101
    TASK_PAUSED = 102
    DEMO_RECORDING = 201
    DEMO_PAUSED = 202

def process_actor_output(actor_output, max_linear_vel, max_angular_vel):
    multiplier = torch.tensor([max_linear_vel, max_angular_vel, 1.0], dtype = torch.float32)
    actor_output = torch.sigmoid(actor_output)*multiplier
    return actor_output