from itertools import count
from threading import Thread

import torch
from torch.utils.data import Dataset, DataLoader

from mobile_robot_hl.model.utils import *
import mobile_robot_hl.model as m
from .utils import *
from .dataset import *
from .algorithms import *

class Trainer():
    def __init__(self, model_handler, demo_handler, task_handler, logger = None):
        self.model_handler = model_handler
        self.demo_hander = demo_handler
        self.task_handler = task_handler

        self.logger = logger

        self.actor_state = TrainerState.SLEEPING
        self.critic_state = TrainerState.SLEEPING
        self.stop = True

        self.actor_model = None
        self.actor_model_info = None
        self.actor_optimizer_dict = None
        self.critic_model_dict = None
        self.critic_model_info = None
        self.critic_optimizer_dict = None
        self.algorithm = None

        self.task_dataset = TaskDataset(self.task_handler)
        self.demo_dataset = DemoDataset(self.demo_handler)
        self.task_dataloader = DataLoader(self.task_dataset, batch_size = None, shuffle = True, num_workers = 4)
        self.demo_dataloader = DataLoader(self.demo_dataset, batch_size = None, shuffle = True, num_workers = 4)

    def select_model(self, model_type, model_name, model_id):
        if(model_type == ModelType.ACTOR):
            if(self.actor_state in [TrainerState.RUNNING, TrainerState.STANDBY]):
                return
            try:
                self.actor_model, self.actor_model_info = self.model_handler.get(ModelType.ACTOR, model_name, model_id)
                if(self.actor_optimizer_dict is not None):
                    self.actor_state = TrainerState.STANDBY
            except:
                self.actor_model = None
                self.actor_model_info = None
                self.actor_state = TrainerState.SLEEPING
        else:
            if(self.critic_state in [TrainerState.RUNNING, TrainerState.STANDBY]):
                return
            try:
                self.critic_model, self.critic_model_info = self.model_handler.get(ModelType.ACTOR, model_name, model_id)
                if(self.critic_optimizer_dict is not None):
                    self.critic_state = TrainerState.STANDBY
            except:
                self.critic_model = None
                self.critic_model_info = None
                self.critic_state = TrainerState.SLEEPING
    
    def create_model(self, model_type, model_name, model_architecture):
        if(model_type == ModelType.ACTOR):
            if(self.actor_state in [TrainerState.RUNNING, TrainerState.STANDBY]):
                return
            try:
                self.actor_model = m.MimeticSNAIL(**model_architecture)
                self.actor_model_info = dict(architecture = model_architecture, name = model_name)
                if(self.actor_optimizer_dict is not None):
                    self.actor_state = TrainerState.STANDBY
            except:
                self.actor_model = None
                self.actor_model_info = None
                self.actor_state = TrainerState.SLEEPING
            
        else:
            if(self.critic_state in [TrainerState.RUNNING, TrainerState.STANDBY]):
                return
            try:
                self.critic_model = m.MimeticSNAIL(**model_architecture)
                self.critic_model_info = dict(architecture = model_architecture, name = model_name)
                if(self.critic_optimizer_dict is not None):
                    self.critic_state = TrainerState.STANDBY
            except:
                self.critic_model = None
                self.critic_model_info = None
                self.critic_state = TrainerState.SLEEPING

    def set_optimizer(self, model_type, optimizer_dict):
        if(model_type == ModelType.ACTOR):
            self.actor_optimizer_dict = optimizer_dict
            if(self.actor_model is not None):
                self.actor_state = TrainerState.STANDBY
        else:
            self.critic_optimizer_dict = optimizer_dict
            if(self.critic_model is not None):
                self.critic_state = TrainerState.STANDBY
    
    def start_training(self, training_type, save_every, max_epochs, additional_algorithm_kwargs = None):
        if(training_type == TrainingType.RL):
            if(self.actor_state == TrainerState.STANDBY and self.critic_state == TrainerState.STANDBY):
                self.flag = True

                algorithm_kwargs = dict(
                    actor_model = self.actor_model,
                    critic_model = self.critic_model,
                    actor_optimizer_dict = self.actor_optimizer_dict, 
                    critic_optimizer_dict = self.critic_optimizer_dict,
                    dataloader = self.task_dataloader,
                    logger = self.logger
                    )
                if(additional_algorithm_kwargs is not None):
                    algorithm_kwargs = {**algorithm_kwargs, **additional_algorithm_kwargs}
                self.algorithm = TD3(**algorithm_kwargs)

                Thread(target = self.training_loop(save_every, max_epochs)).start()
                self.actor_state = TrainerState.RUNNING
                self.critic_state = TrainerState.RUNNING
        elif(training_type == TrainingType.IL):
            if(self.actor_state == TrainerState.STANDBY):
                self.flag = True

                algorithm_kwargs = dict(
                    actor_model = self.actor_model,
                    actor_optimizer_dict = self.actor_optimizer_dict, 
                    dataloader = self.demo_dataloader,
                    logger = self.logger
                    )
                if(additional_algorithm_kwargs is not None):
                    algorithm_kwargs = {**algorithm_kwargs, **additional_algorithm_kwargs}

                # TODO: self.algorithm = IL(**algorithm_kwargs)
                Thread(target = self.training_loop(save_every, max_epochs)).start()
                self.actor_state = TrainerState.RUNNING

    def pause_training(self):
        self.stop = True
        self.actor_state = TrainerState.STANDBY
        if(self.critic_state == TrainerState.RUNNING):
            self.critic_state = TrainerState.STANDBY

    def stop_training(self):
        self.stop = True
        self.actor_state = TrainerState.SLEEPING
        if(self.critic_state == TrainerState.RUNNING):
            self.critic_state = TrainerState.SLEEPING
        self.actor_model = None
        self.critic_model = None

    def restart_model(self, model_type):
        if(model_type == ModelType.ACTOR):
            if(self.actor_state == TrainerState.STANDBY):
                self.actor_model = m.MimeticSNAIL(**self.actor_model_info['architecture'])
            else:
                return
        else:
            if(self.critic_state == TrainerState.STANDBY):
                self.critic_model = m.MimeticSNAIL(**self.critic_model_info['architecture'])
            else:
                return
    
    def save_model(self, model_type):
        if model_type == ModelType.ACTOR:
            model = self.actor_model
            architecture = self.actor_model_info['architecture']
            name = self.actor_model_info['name']
        else:
            model = self.critic_model
            architecture = self.critic_model_info['architecture']
            name = self.critic_model_info['name']

        self.model_handler.save(model, architecture, model_type, name)
    
    def select_training_data(self, list_of_names):
        # TODO: self.dataloader = sth
        raise NotImplementedError()
    
    def select_device(self, device_name):
        if(self.actor_state == TrainerState.STANDBY):
            self.algorithm.select_device(device_name)
            self.actor_model.to(device_name)
        else:
            raise Exception(f"Unable to select device as agent_state = {self.agent_state.name}")
        if(self.critic_state == TrainerState.STANDBY):
            self.critic_model.to(device_name)
        else:
            raise Exception(f"Unable to select device as critic_state = {self.critic_state.name}")
    
    def training_loop(self, max_epochs = None, save_every = None):
        if(self.actor_state == TrainerState.SLEEPING):
            return
        self.actor_state = TrainerState.RUNNING
        if(max_epochs == None):
            max_epochs = -1

        for i in count(0):
            # TODO: Training code here
            self.algorithm.train_one_epoch(self.stop)

            if i == max_epochs:
                self.critic_state = TrainerState.STANDBY
                return
        else:
            pass