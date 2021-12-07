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

def compute_rewards(
    demonstration_flag, 
    user_termination_flag, 
    agent_termination_flag, 
    user_linear_velocity, 
    agent_linear_velocity, 
    user_angular_velocity, 
    agent_angular_velocity):
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

        reward_user = compute_similarity(user_linear_velocity, user_angular_velocity, agent_linear_velocity, agent_angular_velocity)

        reward_termination_flag = torch.zeros_like(reward_agent)
        reward_termination_flag[user_termination_flag == agent_termination_flag] == 1.0

    else:
        raise Exception("Invalid type to compute rewards")
    return reward_agent, reward_user, reward_termination_flag

def compute_similarity(user_linear, user_angular, agent_linear, agent_angular):
    user_vel = torch.cat((user_linear.unsqueeze(1), user_angular.unsqueeze(1)), dim = 1)
    agent_vel = torch.cat((agent_linear.unsqueeze(1), agent_angular.unsqueeze(1)), dim = 1)
    dot = (user_vel@agent_vel)/(torch.max(user_vel, agent_vel))
    user_mag = (torch.sum(user_vel**2, dim = 1))**0.5
    agent_mag = (torch.sum(agent_vel**2, dim = 1))**0.5
    similarity = dot/torch.max(user_mag, agent_mag)
    return similarity
            
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