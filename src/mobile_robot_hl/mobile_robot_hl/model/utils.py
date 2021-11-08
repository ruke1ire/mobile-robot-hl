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
        pass

    def get(self, name, id):
        '''
        returns a nn.Module object of the specified name and ID
        '''
        raise NotImplementedError()

    def save(self, model, name, id):
        '''
        saves a nn.Module object to the specified name and ID
        '''
        raise NotImplementedError()
