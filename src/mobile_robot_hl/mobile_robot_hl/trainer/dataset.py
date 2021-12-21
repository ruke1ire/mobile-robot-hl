from mobile_robot_hl.utils import ControllerType
from mobile_robot_hl.trainer.utils import *

from torch.utils.data import Dataset
import numpy as np
import copy

class DemoDataset(Dataset):
    def __init__(self, demo_handler, list_of_names=None):
        self.demo_handler = demo_handler
        self.demo_names = set()
        self.demo_ids = {}
        self.data = []
        self.list_of_names = list_of_names
        self.get_all_data()

    def get_all_data(self):
        if(self.list_of_names == None):
            demo_names = set(self.demo_handler.get_names())
        else:
            demo_names = set(self.list_of_names)
        new_names = demo_names - self.demo_names
        if(len(new_names) > 0):
            self.demo_names = copy.deepcopy(demo_names)
            for name in new_names:
                self.demo_ids[name] = set()

        for name in self.demo_names:
            ids = set(self.demo_handler.get_ids(name))
            new_ids = ids - self.demo_ids[name]
            if(len(new_ids) > 0):
                self.demo_ids[name] = copy.deepcopy(ids)
                for id_ in new_ids:
                    data = self.demo_handler.get(name, id_)
                    image_tensor, latent_tensor, frame_no_tensor = data.get_tensor()
                    self.data.insert(0,(image_tensor, latent_tensor, frame_no_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TaskDataset(Dataset):
    def __init__(self, task_handler, list_of_names= None):
        self.task_handler = task_handler
        self.demo_names = set()
        self.task_ids = {}
        self.data = []
        self.list_of_names = list_of_names
        self.get_all_data()
    
    def get_all_data(self):
        if(self.list_of_names == None):
            demo_names = set(self.task_handler.get_names())
        else:
            demo_names = set(self.list_of_names)
        new_names = demo_names - self.demo_names
        if(len(new_names) > 0):
            self.demo_names = copy.deepcopy(demo_names)
            for name in new_names:
                self.task_ids[name] = set()

        for name in self.demo_names:
            ids = set(self.task_handler.get_ids(name))
            new_ids = ids - self.task_ids[name]
            if(len(new_ids) > 0):
                self.task_ids[name] = copy.deepcopy(ids)
                for id_ in new_ids:
                    data = self.task_handler.get(name, id_)
                    image_tensor, latent_tensor, frame_no_tensor = data.get_tensor()
                    demonstration_flag = latent_tensor[3,:]
                    desired_termination_flag = latent_tensor[2,:]
#                    agent_termination_flag = torch.tensor(data.action.agent.termination_flag.get(), dtype=torch.float32)
#                    agent_linear_vel = torch.tensor(data.action.agent.velocity.linear.get())
#                    agent_angular_vel = torch.tensor(data.action.agent.velocity.angular.get())
                    user_linear_vel = torch.tensor(data.action.user.velocity.linear.get())
                    user_angular_vel = torch.tensor(data.action.user.velocity.angular.get())
                    rewards_agent = compute_rewards(demonstration_flag)
                    self.data.insert(0,(image_tensor, latent_tensor, frame_no_tensor, rewards_agent, desired_termination_flag, user_linear_vel, user_angular_vel))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
