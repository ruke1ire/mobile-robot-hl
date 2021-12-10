from enum import Enum

from mobile_robot_hl.utils import ControllerType
from mobile_robot_hl.logger import *

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import os

MAX_LINEAR_VELOCITY = float(os.environ['MOBILE_ROBOT_HL_MAX_LINEAR_VEL'])
MAX_ANGULAR_VELOCITY = float(os.environ['MOBILE_ROBOT_HL_MAX_ANGULAR_VEL'])

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

def compute_rewards(demonstration_flag):
    '''
    reward function rules:
        - Rewards when agent is controlling (Minimize supervisor correction)
            - 0 reward for every demonstrated timesteps
            - timestep before supervisor take-over gets a negative reward corresponding to the time that the take-over took place + 1
        - Rewards for user controlled variables (Imitate)
            - [0, 1] reward corresponding to the similarity of the user and agent's desired velocity
        - Rewards for termination flag usage
            - -1 reward for incorrect usage of termination flag 

    '''
    if(type(demonstration_flag) == torch.Tensor):
        reward_agent = torch.ones((demonstration_flag.shape[0]), dtype = torch.float32)
        reward_agent[demonstration_flag == 1.0] = 0.0
        next_flag = 0
        count = 0
        for i in reversed(range(reward_agent.shape[0])):
            flag = demonstration_flag[i]
            if(flag == 0):
                if(next_flag == 1):
                    reward_agent[i] = count-1
                    count = 0
            elif(flag == 1):
                count -= 1
            next_flag = flag

    else:
        raise Exception("Invalid type to compute rewards")
    return reward_agent

def compute_similarity(user_linear, user_angular, agent_linear, agent_angular):
    user_vel = torch.cat((user_linear.unsqueeze(1), user_angular.unsqueeze(1)), dim = 1)
    agent_vel = torch.cat((agent_linear.unsqueeze(1), agent_angular.unsqueeze(1)), dim = 1)
    e_dist = torch.sum((user_vel - agent_vel)**2, dim = 1)**0.5
    return -e_dist
            
def compute_values(gamma, rewards_velocity):
    if(type(rewards_velocity) == torch.Tensor):
        size = rewards_velocity.size()[0]
    elif(type(rewards_velocity) == np.ndarray):
        size = rewards_velocity.size
    discounted_mat = create_discounted_matrix(gamma, size)
    if(type(rewards_velocity) == torch.Tensor):
        discounted_mat = torch.tensor(discounted_mat, dtype = torch.float32).to(rewards_velocity.device)
    values = discounted_mat@rewards_velocity
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

def create_logger(run_name, run_id, config_dict, logger_name = None):
    if(logger_name == None):
        logger = EmptyLogger(run_name, run_id, config_dict)
    else:
        exec_vars = dict(out = None, config_dict = config_dict)
        exec(f"out = {logger_name}('{run_name}', '{run_id}', config_dict)", None, exec_vars)
        logger = exec_vars['out']
    return logger