from mobile_robot_hl.utils import ControllerType
import torch
import numpy as np
from PIL import Image as PImage
import copy

class Node():
    def __init__(self, childrens):
        self.childrens = childrens
    
    def get(self, index = None):
        tmp = dict()
        for name, obj in self.childrens.items():
            tmp[name] = obj.get(index)
        return tmp
    
    def set(self, data, index = None):
        if(type(data) == dict):
            pass
        else:
            data = data.get()
        for name, obj in self.childrens.items():
            obj.set(data[name], index)
    
    def append(self, data):
        if(type(data) == dict):
            pass
        else:
            data = data.get()
        for name, obj in self.childrens.items():
            obj.append(data[name])
    
    def remove(self, index, leftwards = True):
        for obj in self.childrens.values():
            obj.remove(index, leftwards)

    def reset(self):
        for obj in self.childrens.values():
            obj.reset()
    
class LeafNode():
    def __init__(self, type_, data):
        '''
        Initialize the object

        Keyword arguments:
        data -- Can be {None, List, Variable}

        if data == None => initialize self.data = []
        if data == Variable => initialize self.data = [data]
        if data == List => initialize self.data = data
        '''
        self.type = type_
        self.set(data, index = None)

    def get(self, index = None):
        '''
        Get data in the object
        
        Keyword arguments
        index -- Int
        '''
        if index is None:
            return self.data
        else:
            assert (type(index) == int), "Index should be of type <int>"
            if (index >= len(self.data)):
                return None
            else:
                return self.data[index]

    def set(self, data, index = None):
        data = copy.deepcopy(data)
        if index is None:
            if(data is None or data == []):
                self.data = []
            elif(type(data) == list):
                #assert (type(data[0]) == self.type), f"Input data type is not {str(self.type)}"
                self.data = data
            else:
                #assert (type(data) == self.type), f"Input data type is not {str(self.type)}"
                self.data = [data]
        else:
            assert (type(index) == int), "Index should be of type <int>"
            assert (index < len(self.data)), "Index out of range"
            #assert (type(data) == self.type), f"Input data type is not {str(self.type)}"
            self.data[index] = data

    def append(self, data):
        if(type(data) == list):
            #assert (type(data[0]) == self.type), f"Input data type is not {str(self.type)}"
            self.data += data
        else:
            self.data.append(data)

    def remove(self, index, leftwards = True):
        if leftwards == True:
            self.data = self.data[index+1:]
        else:
            self.data = self.data[:index]

    def reset(self):
        self.data = []
    
    def length(self):
        return len(self.data)
    
    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)
    
class EpisodeFloat(LeafNode):
    type = float
    def __init__(self, data = None):
        super().__init__(self.type, data)

class EpisodeInt(LeafNode):
    type = int
    def __init__(self, data = None):
        super().__init__(self.type, data)

class EpisodeString(LeafNode):
    type = str
    def __init__(self, data = None):
        super().__init__(self.type, data)

class EpisodeBool(LeafNode):
    type = bool
    def __init__(self, data = None):
        super().__init__(self.type, data)

class EpisodeImage(LeafNode):
    type = PImage
    def __init__(self, data = None):
        super().__init__(self.type, data)

class EpisodeController(LeafNode):
    type = ControllerType
    def __init__(self, data = None):
        super().__init__(self.type, data)

class EpisodeVelocity(Node):
    def __init__(self, linear = None, angular = None):
        self.linear = EpisodeFloat(linear)
        self.angular = EpisodeFloat(angular)
        childrens = dict(linear = self.linear, angular = self.angular)
        super().__init__(childrens)

class EpisodeActor(Node):
    def __init__(self, velocity = {}, termination_flag = None):
        self.velocity = EpisodeVelocity(**velocity)
        self.termination_flag = EpisodeBool(termination_flag)
        childrens = dict(velocity = self.velocity, termination_flag = self.termination_flag)
        super().__init__(childrens)

class EpisodeAction(Node):
    def __init__(self, user = {}, agent = {}, controller = None):
        self.user = EpisodeActor(**user)
        self.agent = EpisodeActor(**agent)
        self.controller = EpisodeController(controller)
        childrens = dict(user = self.user, agent = self.agent, controller = self.controller)
        super().__init__(childrens)

class EpisodeObservation(Node):
    def __init__(self, image = None):
        self.image = EpisodeImage(image)
        childrens = dict(image = self.image)
        super().__init__(childrens)

class EpisodeData(Node):
    def __init__(self, observation = {}, action = {}, frame_no = None):
        self.observation = EpisodeObservation(**observation)
        self.action = EpisodeAction(**action)
        self.frame_no = EpisodeInt(frame_no)
        childrens = dict(observation = self.observation, action = self.action, frame_no = self.frame_no)
        super().__init__(childrens)
    
    def length(self):
        return self.action.controller.length()
    
    def get_tensor(self):
        p_image_list = self.observation.image.get()
        np_image_list = [np.array(img) for img in p_image_list]

        image_tensor = torch.tensor(np.stack(np_image_list), dtype = torch.float32).permute((0,3,1,2))/255.0

        controller = self.action.controller.get()
        agent_linear_vel = self.action.agent.velocity.linear.get()
        agent_angular_vel = self.action.agent.velocity.angular.get()
        agent_termination_flag = self.action.agent.termination_flag.get()
        user_linear_vel = self.action.user.velocity.linear.get()
        user_angular_vel = self.action.user.velocity.angular.get()
        user_termination_flag = self.action.user.termination_flag.get()

        desired_linear_vel = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_linear_vel, agent_linear_vel, controller)])
        desired_angular_vel = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_angular_vel, agent_angular_vel, controller)])
        desired_termination_flag = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_termination_flag, agent_termination_flag, controller)], dtype = torch.float32)
        demonstration_flag = torch.tensor([True if cont == ControllerType.USER else False for cont in controller], dtype = torch.float32)

        latent_tensor = torch.stack([desired_linear_vel, desired_angular_vel, desired_termination_flag, demonstration_flag])
        frame_no_tensor = torch.tensor(self.frame_no.get(), dtype = torch.float32)

        return image_tensor, latent_tensor, frame_no_tensor

    def __eq__(self, other):
        """Overrides the default implementation"""
        try:
            return self.get() == other.get()
        except:
            return False