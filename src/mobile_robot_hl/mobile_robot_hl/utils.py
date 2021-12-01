from enum import Enum
import torch
import torch.nn as nn

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
