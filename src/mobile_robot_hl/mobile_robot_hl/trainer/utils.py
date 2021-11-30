from enum import Enum

from mobile_robot_hl.utils import ControllerType

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np

class TrainerState(Enum):
    SLEEPING = 0 # when model hasn't been selected
    STANDBY = 1 # when model has been selected or when training has been paused
    RUNNING = 2 # when model is being trained

class TrainingComponents(Enum):
    MODEL = 0
    OPTIMIZER = 1

class TrainingType(Enum):
    IL = 0
    RL = 1

def compute_rewards(demonstration_flag, user_termination_flag, agent_termination_flag):
    if(type(demonstration_flag) == np.ndarray):
        reward = np.ones((demonstration_flag.shape[0]))
        reward[demonstration_flag == 1] = 0
        next_r = 1
        count = 0
        for i in reversed(range(reward.shape[0])):
            r = reward[i]
            if(r == 1):
                if(next_r == 0):
                    reward[i] = count-1
                    count = 0
            elif(r == 0):
                count -= 1
            next_r = r
        reward[user_termination_flag != agent_termination_flag] -= 1.0
    elif(type(demonstration_flag) == torch.Tensor):
        reward = torch.ones((demonstration_flag.shape[0]), dtype = torch.float32)
        reward[demonstration_flag == 1] = 0
        next_r = 1
        count = 0
        for i in reversed(range(reward.shape[0])):
            r = reward[i]
            if(r == 1):
                if(next_r == 0):
                    reward[i] = count-1
                    count = 0
            elif(r == 0):
                count -= 1
            next_r = r
        reward[user_termination_flag != agent_termination_flag] -= 1.0
    else:
        raise Exception("Invalid type to compute rewards")
    return reward
            
def compute_values(gamma, rewards):
    if(type(rewards) == torch.Tensor):
        size = rewards.size()[0]
    elif(type(rewards) == np.ndarray):
        size = rewards.size
    discounted_mat = create_discounted_matrix(gamma, size)
    if(type(rewards) == torch.Tensor):
        discounted_mat = torch.tensor(discounted_mat, dtype = torch.float32).to(rewards.device.type)
    values = discounted_mat@rewards
    return values

def create_discounted_matrix(gamma, size):
    mat = np.zeros((size,size))
    discout_vec = np.array([gamma**i for i in range(size)])
    for i in range(size):
        mat[i,i:] = discout_vec[:(size-i)]
    return mat

def create_optimizer_from_dict(optimizer_dict, parameters):
    optimizer_name = optimizer_dict['optimizer_name']
    optimizer_kwargs = optimizer_dict['optimizer_kwargs']
    optimizer = dict(parameters= parameters, optimizer_kwargs = optimizer_kwargs,out=None)
    exec(f"out = torch.optim.{optimizer_name}(parameters, **optimizer_kwargs)", None, optimizer)
    return optimizer['out']