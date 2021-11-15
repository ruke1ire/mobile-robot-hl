from enum import Enum
import yaml
from datetime import datetime
import os
import torch
import glob

class ModuleType(Enum):
    TC = 0
    ATTENTION = 1

class InferenceMode(Enum):
    ONLY_LAST_FRAME = 0
    WHOLE_BATCH = 1

class ModelType(Enum):
    ACTOR = 0
    CRITIC = 1