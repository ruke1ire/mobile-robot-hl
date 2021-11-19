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
    #TASK_TAKE_OVER = 103
    DEMO_RECORDING = 201
    DEMO_PAUSED = 202

def process_actor_output(actor_output):
    actor_output[2] = torch.sigmoid(actor_output[2])
    return actor_output