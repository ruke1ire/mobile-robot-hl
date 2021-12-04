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
            - -1 reward for incorrect usage of termination flag 
            - [0, 1] reward corresponding to the similarity of the user and agent's desired velocity

    '''
    if(type(demonstration_flag) == np.ndarray):
        reward_agent = np.ones((demonstration_flag.shape[0]))
        reward_agent[demonstration_flag == 1] = 0
        next_r = 1
        count = 0
        for i in reversed(range(reward_agent.shape[0])):
            r = reward_agent[i]
            if(r == 1):
                if(next_r == 0):
                    reward_agent[i] = count-1
                    count = 0
            elif(r == 0):
                count -= 1
            next_r = r

        reward_user = np.zeros_like(reward_agent)
        reward_user[user_termination_flag != agent_termination_flag] -= 1.0
        reward_user += ((
            compute_similarity(user_linear_velocity, agent_linear_velocity, 2*MAX_LINEAR_VELOCITY)**2 + \
            compute_similarity(user_angular_velocity, agent_angular_velocity, 2*MAX_ANGULAR_VELOCITY)**2)/2)**0.5

    elif(type(demonstration_flag) == torch.Tensor):
        reward_agent = torch.ones((demonstration_flag.shape[0]), dtype = torch.float32)
        reward_agent[demonstration_flag == 1] = 0
        next_r = 1
        count = 0
        for i in reversed(range(reward_agent.shape[0])):
            r = reward_agent[i]
            if(r == 1):
                if(next_r == 0):
                    reward_agent[i] = count-1
                    count = 0
            elif(r == 0):
                count -= 1
            next_r = r
        reward_user = torch.zeros_like(reward_agent)
        reward_user[user_termination_flag != agent_termination_flag] -= 1.0
        reward_user += ((
            compute_similarity(user_linear_velocity, agent_linear_velocity, 2*MAX_LINEAR_VELOCITY)**2 + \
            compute_similarity(user_angular_velocity, agent_angular_velocity, 2*MAX_ANGULAR_VELOCITY)**2)/2)**0.5
    else:
        raise Exception("Invalid type to compute rewards")
    return reward_agent, reward_user

def compute_similarity(tensor1, tensor2, range):
    similarity = 1 - abs(tensor1-tensor2)/range
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

def create_logger_from_dict(logger_dict):
    if(logger_dict == dict()):
        logger = EmptyLogger()
    else:
        exec_vars = dict(logger_dict = logger_dict, out = None)
        exec(f"out = {logger_dict['name']}(**logger_dict['kwargs'])", None, exec_vars)
        logger = exec_vars['out']
    return logger