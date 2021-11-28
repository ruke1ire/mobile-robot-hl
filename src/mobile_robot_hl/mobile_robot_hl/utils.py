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

def process_actor_output(actor_output, max_linear_vel, max_angular_vel, noise = 0.0):
    multiplier = torch.tensor([max_linear_vel, max_angular_vel, 0.5], dtype = torch.float32)
    adder = torch.tensor([0.0, 0.0, 0.5], dtype = torch.float32)
    noise_tensor = noise*(torch.rand(actor_output.shape)*(2*multiplier) - multiplier)
    actor_output = torch.tanh(actor_output+noise_tensor)*multiplier + adder
    return actor_output