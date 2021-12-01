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

class OutputProcessor(nn.Module):
    def __init__(self, max_linear_vel, max_angular_vel):
        super().__init__()
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

    def forward(self, actor_output, noise = 0.0):
        multiplier = torch.tensor([self.max_linear_vel, self.max_angular_vel, 0.5], dtype = torch.float32).to(actor_output.device.type)
        adder = torch.tensor([0.0, 0.0, 0.5], dtype = torch.float32).to(actor_output.device.type)
        noise_tensor = (20*noise*(torch.rand(actor_output.shape)*(2*multiplier) - multiplier)).to(actor_output.device.type)
        actor_output = torch.tanh(actor_output+noise_tensor)*multiplier + adder
        return actor_output