from enum import Enum
import yaml
from datetime import datetime
import os
import torch
import glob

import mobile_robot_hl.model as m

class ModuleType(Enum):
    TC = 0
    ATTENTION = 1

class InferenceMode(Enum):
    ONLY_LAST_FRAME = 0
    WHOLE_BATCH = 1

class ModelType(Enum):
    ACTOR = 0
    CRITIC = 1

class ModelHandler():
    MODEL_INFO_FILE = "info.yaml"
    def __init__(self, path):
        self.path = path

    def get(self, model_type, name, id_ = None):
        '''
        returns a nn.Module object of the specified name and ID
        '''
        if id_ == "latest" or id_ == None:
            list_ids = self.get_ids(name)
            if len(list_ids) == 0:
                raise Exception("Model selection failed, there are no models in this directory")
            datetime_list = [datetime.strptime(datetime_str, "%Y-%m-%d_%H:%M") for datetime_str in list_ids]
            latest_time = max(datetime_list)
            id_ = latest_time.strftime("%Y-%m-%d_%H:%M")

        with open(f"{self.path}/{model_type.name.lower()}/{name}/{ModelHandler.MODEL_INFO_FILE}", "r") as stream:
            info_dict = yaml.safe_load(stream)
        
        model_architecture = info_dict['architecture']

        model = m.MimeticSNAIL(**model_architecture)
        model.load_state_dict(torch.load(f"{self.path}/{model_type.name.lower()}/{name}/{id_}.pth"))

        return model, info_dict

    def get_names(self, model_type):
        list_model_names = [os.path.basename(x) for x in glob.glob(f"{self.path}/{model_type.name.lower()}/*")]
        return list_model_names

    def get_ids(self, model_type, name):
        list_run_ids = [os.path.basename(x) for x in glob.glob(f"{self.path}/{model_type.name.lower()}/{name}/*") if os.path.basename(x) is not ModelHandler.MODEL_INFO_FILE]
        return list_run_ids

    def save(self, model, model_architecture, model_type, name):
        '''
        saves a nn.Module object to the specified name and ID along with a yaml file with all the information about the model

        model = nn.Module obj of the model to be saved
        model_architecture = dict() of kwargs to be set for the model
        name = name of the model
        '''

        if not os.path.exists(f"{self.path}/{model_type}/{name}"):
            os.mkdir(f"{self.path}/{model_type.name.lower()}/{name}")

        current_date_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M")
        torch.save(model.state_dict(), f"{self.path}/{model_type.name.lower()}/{name}/{current_date_time_str}.pth")

        info_dict = dict(architecture=model_architecture)

        with open(f"{self.path}/{model_type.name.lower()}/{name}/{ModelHandler.MODEL_INFO_FILE}", "w") as stream:
            yaml.dump(info_dict, stream)