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
                    observations = np.stack([np.transpose(np.asarray(img), (2,0,1)) for img in data.data['observation']['image']], axis = 0).astype(np.float32)/255.0
                    linear_vel = np.array(data.data['action']['user']['velocity']['linear']).astype(np.float32)
                    angular_vel = np.array(data.data['action']['user']['velocity']['angular']).astype(np.float32)
                    termination_flag = np.array(data.data['action']['user']['termination_flag']).astype(np.float32)
                    self.data.append((observations, linear_vel, angular_vel, termination_flag))

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
                    observations = np.stack([np.transpose(np.asarray(img), (2,0,1)) for img in data.data['observation']['image']], axis = 0).astype(np.float32)/255.0
                    user_action = data.data['action']['user']
                    agent_action = data.data['action']['agent']
                    controller = data.data['action']['controller']

                    linear_vel = np.array([action[0] if action[2] == ControllerType.USER else action[1] for action in zip(user_action['velocity']['linear'], agent_action['velocity']['linear'], controller)]).astype(np.float32)
                    angular_vel = np.array([action[0] if action[2] == ControllerType.USER else action[1] for action in zip(user_action['velocity']['angular'], agent_action['velocity']['angular'], controller)]).astype(np.float32)
                    termination_flag = np.array([action[0] if action[2] == ControllerType.USER else action[1] for action in zip(user_action['termination_flag'], agent_action['termination_flag'], controller)]).astype(np.float32)
                    demonstration_flag = np.array([1.0 if d == ControllerType.USER else 0.0 for d in controller]).astype(np.float32)
                    rewards = compute_rewards(demonstration_flag).astype(np.float32)
                    self.data.append((observations, linear_vel, angular_vel, termination_flag, demonstration_flag, rewards))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
