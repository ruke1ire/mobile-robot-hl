from mobile_robot_hl.utils import ControllerType
from mobile_robot_hl.trainer.utils import *

from torch.utils.data import Dataset
import numpy as np

class DemoDataset(Dataset):
    def __init__(self, demo_handler):
        self.demo_handler = demo_handler
        self.demo_names = set()
        self.demo_ids = {}
        self.data = []
        self.get_all_data()

    def get_all_data(self):
        demo_names = set(self.demo_handler.get_names())
        new_names = demo_names - self.demo_names
        if(len(new_names) > 0):
            self.demo_names = demo_names
            for name in new_names:
                self.demo_ids[name] = set()

        for name in self.demo_names:
            ids = set(self.demo_handler.get_ids(name))
            new_ids = ids - self.demo_ids[name]
            if(len(new_ids) > 0):
                self.demo_ids[name] = new_ids
                for id_ in self.demo_ids[name]:
                    data = self.demo_handler.get(name, id_)
                    image_tensor, latent_tensor, frame_no_tensor = data.get_tensor()
                    demonstration_flag = latent_tensor[3,:]
                    self.data.append((image_tensor, latent_tensor, frame_no_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TaskDataset(Dataset):
    def __init__(self, task_handler):
        self.task_handler = task_handler
        self.demo_names = set()
        self.task_ids = {}
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
                    image_tensor, latent_tensor, frame_no_tensor = data.get_tensor()
                    demonstration_flag = latent_tensor[3,:]
                    user_termination_flag = torch.tensor(data.action.user.termination_flag.get(), dtype=torch.float32)
                    agent_termination_flag = torch.tensor(data.action.agent.termination_flag.get(), dtype=torch.float32)
                    rewards = compute_rewards(demonstration_flag, user_termination_flag, agent_termination_flag)
                    self.data.append((image_tensor, latent_tensor, frame_no_tensor, rewards))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
