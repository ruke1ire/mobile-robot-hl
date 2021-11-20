from mobile_robot_hl.utils import *
from mobile_robot_hl.episode_data import *
from PIL import Image as PImage

import numpy as np
import copy

class GUIVariable():
    def __init__(self, variable = None):
        if variable != None:
            self.episode = EpisodeData(**variable.episode.get())
            self.episode_index = copy.deepcopy(variable.episode_index)
            self.episode_type = copy.deepcopy(variable.episode_type)
            self.episode_name = copy.deepcopy(variable.episode_name)
            self.episode_id = copy.deepcopy(variable.episode_id)

            self.model_name = copy.deepcopy(variable.model_name)
            self.model_id = copy.deepcopy(variable.model_id)

            self.image_raw = copy.deepcopy(variable.image_raw)
            self.supervisor_state = copy.deepcopy(variable.supervisor_state)
            self.supervisor_controller = copy.deepcopy(variable.supervisor_controller)

            self.demo_names = copy.deepcopy(variable.demo_names)
            self.task_names = copy.deepcopy(variable.task_names)
            self.ids = copy.deepcopy(variable.ids)
            self.task_queue = copy.deepcopy(variable.task_queue)

            self.model_names = copy.deepcopy(variable.model_names)
            self.model_ids = copy.deepcopy(variable.model_ids)

            self.user_velocity = copy.deepcopy(variable.user_velocity)

        else:
            self.reset()
    
    def reset(self):
        self.episode = EpisodeData()
        self.episode_index = 0
        self.episode_type = InformationType.NONE
        self.episode_name = None
        self.episode_id = None

        self.model_name = None
        self.model_id = None

        blank_img = np.zeros([360, 480, 3], dtype = np.uint8)
        blank_img.fill(100)
        self.image_raw = PImage.fromarray(blank_img)
        self.supervisor_state = SupervisorState.STANDBY
        self.supervisor_controller = ControllerType.NONE

        self.demo_names = []
        self.task_names = []
        self.ids = []
        self.task_queue = []

        self.model_names = []
        self.model_ids = []

        self.user_velocity = dict(linear= 0.0, angular= 0.0)

class GUIConstant():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.max_linear_vel = None
        self.max_angular_vel = None