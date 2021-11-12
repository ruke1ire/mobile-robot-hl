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

class TriggerCommand(Enum):
    PAUSE = 0
    STOP = 1
    SAVE = 2

class DemoDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def add_item(self, idx):
        pass

class TaskDataset(Dataset):
    def __init__(self, task_handler, gamma):
        self.task_handler = task_handler
        self.demo_names = set()
        self.task_ids = {}
        self.gamma = gamma
        self.data = []
        self.get_all_data()
    
    def get_all_data(self):
        demo_names = set(self.task_handler.get_names())
        new_names = demo_names - self.demo_names
        if(len(new_names) > 0):
            self.demo_names = demo_names
            for name in new_names:
                self.task_ids[name] = set()

        for name in self.demo_names:
            ids = set(self.task_handler.get_ids(name))
            new_ids = ids - self.task_ids[name]
            if(len(new_ids) > 0):
                self.task_ids[name] = new_ids
                for id_ in self.task_ids[name]:
                    data = self.task_handler.get(name, id_)
                    observations = np.stack([np.transpose(np.asarray(img), (2,0,1)) for img in data.data['observation']['image']], axis = 0)
                    user_action = data.data['action']['user']
                    agent_action = data.data['action']['agent']
                    controller = data.data['action']['controller']
                    linear_vel = np.array([action[0] if action[2] == ControllerType.USER else action[1] for action in zip(user_action['velocity']['linear'], agent_action['velocity']['linear'], controller)])
                    angular_vel = np.array([action[0] if action[2] == ControllerType.USER else action[1] for action in zip(user_action['velocity']['angular'], agent_action['velocity']['angular'], controller)])
                    termination_flag = np.array([action[0] if action[2] == ControllerType.USER else action[1] for action in zip(user_action['termination_flag'], agent_action['termination_flag'], controller)]).astype(float)
                    demonstration_flag = np.array([1.0 if d == ControllerType.USER else 0.0 for d in controller])
                    values = compute_values(self.gamma, compute_rewards(demonstration_flag))
                    self.data.append((observations, linear_vel, angular_vel, termination_flag, demonstration_flag, values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def compute_rewards(demonstration_flag):
    # TODO: add reward function rule on incorrect usage of the task termination flag and the reward at the end of a non-episodic task
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
    return reward
            
def compute_values(gamma, rewards):
    size = rewards.size
    discounted_mat = create_discounted_matrix(gamma, size)
    values = discounted_mat@rewards
    return values

def create_discounted_matrix(gamma, size):
    mat = np.zeros((size,size))
    discout_vec = np.array([gamma**i for i in range(size)])
    for i in range(size):
        mat[i,i:] = discout_vec[:(size-i)]
    return mat