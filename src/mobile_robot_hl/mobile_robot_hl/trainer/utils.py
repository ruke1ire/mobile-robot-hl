from enum import Enum

from torch.utils.data import Dataset

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