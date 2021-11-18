from mobile_robot_hl.utils import *
from mobile_robot_hl.episode_data import *
from PIL import Image as PImage
import numpy as np

class GUIVariable():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episode = EpisodeData()
        self.episode_type = InformationType.NONE
        self.episode_name = None
        self.episode_id = None
        self.model_name = None
        self.model_id = None
        self.image_raw = PImage.fromarray(np.zeros([360, 480, 3], dtype = np.uint8).fill(100))
        self.supervisor_state = SupervisorState.STANDBY
        self.demo_names = []
        self.task_names = []
        self.ids = []
        self.model_names = []
        self.model_ids = []
        self.task_queue = []
        self.episode_index = 0

class GUIConstant():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.max_linear_vel = None
        self.max_angular_vel = None