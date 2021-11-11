from threading import Thread
from enum import Enum
from itertools import count
from queue import Queue

from torch.utils.data import Dataset, DataLoader

from mobile_robot_hl.model.utils import *
import mobile_robot_hl.model as m

class TrainerState(Enum):
    SLEEPING = 0 # when model hasn't been selected
    STANDBY = 1 # when model has been selected or when training has been paused
    RUNNING = 2 # when model is being trained

class TriggerCommand(Enum):
    PAUSE = 0
    STOP = 1
    SAVE = 2

class Trainer():
    def __init__(self, model_handler, demo_handler, task_handler, logger = None):
        self.model_handler = model_handler
        self.demo_hander = demo_handler
        self.task_handler = task_handler

        self.logger = logger

        self.actor_state = TrainerState.SLEEPING
        self.critic_state = TrainerState.SLEEPING

        self.actor_model = None
        self.actor_model_info = None
        self.critic_model = None
        self.critic_model_info = None

        self.task_dataset = TaskDataset(self.task_handler)
        self.demo_dataset = DemoDataset(self.demo_handler)
        self.task_dataloader = DataLoader(self.task_dataset, batch_size = None, shuffle = True, num_workers = 4, collate_fn = collate)
        self.demo_dataloader = None

        self.actor_command_queue = Queue()
        self.critic_command_queue = Queue()

    def select_model(self, model_type, model_name, model_id):
        if(model_type == ModelType.ACTOR):
            if(self.actor_state in [TrainerState.RUNNING, TrainerState.STANDBY]):
                return
            try:
                self.actor_model, self.actor_model_info = self.model_handler.get(ModelType.ACTOR, model_name, model_id)
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
                self.critic_state = TrainerState.STANDBY
            except:
                self.critic_model = None
                self.critic_model_info = None
                self.critic_state = TrainerState.SLEEPING
    
    def start_training_actor(self, training_type, save_every, max_epochs):
        if(self.actor_state == TrainerState.STANDBY):
            self.actor_trainig_loop(training_type, save_every, max_epochs)
        else:
            return

    def start_training_critic(self, save_every, max_epochs):
        if(self.critic_state == TrainerState.STANDBY):
            self.critic_trainig_loop(save_every, max_epochs)
        else:
            return
    
    def pause_training_actor(self):
        self.actor_command_queue.put(TriggerCommand.PAUSE)
    
    def pause_training_critic(self):
        self.critic_command_queue.put(TriggerCommand.PAUSE)

    def stop_training_actor(self):
        self.actor_command_queue.put(TriggerCommand.STOP)

    def stop_training_critic(self):
        self.critic_command_queue.put(TriggerCommand.STOP)

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
    
    def select_device(self, model_type, device_name):
        if(model_type == ModelType.ACTOR):
            if(self.actor_state in [TrainerState.STANDBY, TrainerState.SLEEPING]):
                self.actor_model.to(device_name)
            else:
                raise Exception(f"Unable to select device as agent_state = {self.agent_state.name}")
        else:
            if(self.critic_state in [TrainerState.STANDBY, TrainerState.SLEEPING]):
                self.critic_model.to(device_name)
            else:
                raise Exception(f"Unable to select device as critic_state = {self.critic_state.name}")
    
    def actor_training_loop(self, training_type, max_epochs = None, save_every = None):
        if(self.actor_state == TrainerState.SLEEPING):
            return
        self.actor_state = TrainerState.RUNNING
        if(max_epochs == None):
            max_epochs = -1
        for i in count(0):
            # TODO: Training code here

            if i == max_epochs:
                self.critic_state = TrainerState.STANDBY
                return

    def critic_training_loop(self, max_epochs = None, save_every = None):
        if(self.critic_state == TrainerState.SLEEPING):
            return
        self.critic_state = TrainerState.RUNNING
        if(max_epochs == None):
            max_epochs = -1
        for i in count(0):
            # TODO: Training code here

            if i == max_epochs:
                self.critic_state = TrainerState.STANDBY
                return
    
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
                    self.data.append(self.task_handler.get(name, id_))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#def collate(episode_data):
#    episode_data[j]
#    return (images, actions, episode_values)