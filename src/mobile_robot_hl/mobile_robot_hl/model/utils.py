from enum import Enum

class ModuleType(Enum):
    TC = 0
    ATTENTION = 1

class InferenceMode(Enum):
    ONLY_LAST_FRAME = 0
    WHOLE_BATCH = 1