from enum import Enum

class ModuleType(Enum):
    TC = 0
    ATTENTION = 1

class InferenceMode(Enum):
    ONLY_LAST_FRAME = 0
    WHOLE_BATCH = 1

class ModelHandler():
    MODEL_INFO_FILE = "info.yaml"
    def __init__(self, path):
        self.path = path

    def get(self, name, id):
        '''
        returns a nn.Module object of the specified name and ID
        '''
        # 1. Get info.yaml
        # 2. Create nn module
        # 3. Load weights
        # 4. Return model and info
        raise NotImplementedError()

    def save(self, model, model_architecture, name, id):
        '''
        saves a nn.Module object to the specified name and ID along with a yaml file with all the information about the model
        '''
        
        raise NotImplementedError()
