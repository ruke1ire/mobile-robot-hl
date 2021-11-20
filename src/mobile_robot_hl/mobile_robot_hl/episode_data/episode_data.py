from mobile_robot_hl.utils import ControllerType
import torch
import numpy as np
from PIL import Image

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
        elif(type(data) == self.type):
            self.data.append(data)
        else:
            raise Exception("Invalid data type")

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
    type = Image
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

#class EpisodeData:
#    def __init__(self, data):
#        if data == None:
#            self.init_empty_structure()
#        else:
#            # TODO: error if data is just an empty structure
#            self.data = copy.deepcopy(data)
#            self.length = self.get_episode_length_()
#            self.data_empty = False
#
#    def init_empty_structure(self):
#        self.data = dict(
#                        observation=dict(image=[None]),
#                        action=dict(
#                            agent=dict(
#                                velocity=dict(
#                                    linear=[None],
#                                    angular=[None]
#                                ),
#                                termination_flag = [None]
#                            ),
#                            user=dict(
#                                velocity=dict(
#                                    linear=[None],
#                                    angular=[None]
#                                ),
#                                termination_flag = [None]
#                            ),
#                            controller=[None]))
#        self.data_empty = True
#        self.length = 0
#    
#    def append_episode_data(self, episode):
#        if(episode.data_empty == True):
#            return
#        else:
#            self.data['observation']['image'] += episode.data['observation']['image']
#            self.data['action']['agent']['velocity']['linear'] += episode.data['action']['agent']['velocity']['linear']
#            self.data['action']['agent']['velocity']['angular'] += episode.data['action']['agent']['velocity']['angular']
#            self.data['action']['agent']['termination_flag'] += episode.data['action']['agent']['termination_flag']
#            self.data['action']['user']['velocity']['linear'] += episode.data['action']['user']['velocity']['linear']
#            self.data['action']['user']['velocity']['angular'] += episode.data['action']['user']['velocity']['angular']
#            self.data['action']['user']['termination_flag'] += episode.data['action']['user']['termination_flag']
#            self.data['action']['controller'] += episode.data['action']['controller']
#            self.data_empty = False
#
#            self.length += episode.length
#
#    def append_data(
#        self, 
#        image, 
#        agent_linear_vel, agent_angular_vel, agent_termination_flag,
#        user_linear_vel, user_angular_vel, user_termination_flag,
#        controller):
#        if(self.data_empty == True):
#            self.data['observation']['image'] = [image]
#            self.data['action']['agent']['velocity']['linear'] = [agent_linear_vel]
#            self.data['action']['agent']['velocity']['angular'] = [agent_angular_vel]
#            self.data['action']['agent']['termination_flag'] = [agent_termination_flag]
#            self.data['action']['user']['velocity']['linear'] = [user_linear_vel]
#            self.data['action']['user']['velocity']['angular'] = [user_angular_vel]
#            self.data['action']['user']['termination_flag'] = [user_termination_flag]
#            self.data['action']['controller'] = [controller]
#            self.data_empty = False
#        else:
#            self.data['observation']['image'].append(image)
#            self.data['action']['agent']['velocity']['linear'].append(agent_linear_vel)
#            self.data['action']['agent']['velocity']['angular'].append(agent_angular_vel)
#            self.data['action']['agent']['termination_flag'].append(agent_termination_flag)
#            self.data['action']['user']['velocity']['linear'].append(user_linear_vel)
#            self.data['action']['user']['velocity']['angular'].append(user_angular_vel)
#            self.data['action']['user']['termination_flag'].append(user_termination_flag)
#            self.data['action']['controller'].append(controller)
#
#        self.length += 1
#
#    def set_key_value(self, key, value, index = None):
#        '''
#        key = string of keys separated by "."
#        Eg. "action.agent.velocity.linear"
#        '''
#        key_split = key.split('.')
#        dict_string = ''
#        for k in key_split:
#            dict_string += "['"
#            dict_string += k
#            dict_string += "']"
#        if type(index) is not int:
#            exec(f"self.data{dict_string} = {value}")
#        else:
#            exec(f"self.data{dict_string}[{index}] = {value}")
#
#    def set_data(
#        self, 
#        index,
#        image=None, 
#        agent_linear_vel=None, agent_angular_vel=None, agent_termination_flag=None,
#        user_linear_vel=None, user_angular_vel=None, user_termination_flag=None,
#        controller=ControllerType.NONE):
#        if(self.data_empty == False):
#            assert index <= self.get_episode_length() - 1
#            self.data['observation']['image'][index] = image
#            self.data['action']['agent']['velocity']['linear'][index] = agent_linear_vel
#            self.data['action']['agent']['velocity']['angular'][index] = agent_angular_vel
#            self.data['action']['agent']['termination_flag'][index] = agent_termination_flag
#            self.data['action']['user']['velocity']['linear'][index] = user_linear_vel
#            self.data['action']['user']['velocity']['angular'][index] = user_angular_vel
#            self.data['action']['user']['termination_flag'][index] = user_termination_flag
#            self.data['action']['controller'][index] = controller
#        else:
#            self.append_data(image, agent_linear_vel, agent_angular_vel, agent_termination_flag,
#            user_linear_vel, user_angular_vel, user_termination_flag,
#            controller)
#
#    def get_episode_length_(self):
#        return len(self.data['observation']['image'])
#
#    def get_episode_length(self):
#        return self.length
#    
#    def get_data(self, index = None):
#        if(type(index) is not int):
#            return self.data
#        else:
#            if(self.data_empty == False):
#                return dict(
#                    observation = dict(
#                        image = self.data['observation']['image'][index]
#                    ),
#                    action = dict(
#                        controller = self.data['action']['controller'][index],
#                        agent = dict(
#                            velocity = dict(
#                                linear = self.data['action']['agent']['velocity']['linear'][index],
#                                angular = self.data['action']['agent']['velocity']['angular'][index]
#                            ),
#                            termination_flag = self.data['action']['agent']['termination_flag']
#                        ),
#                        user = dict(
#                            velocity = dict(
#                                linear = self.data['action']['user']['velocity']['linear'][index],
#                                angular = self.data['action']['user']['velocity']['angular'][index]
#                            ),
#                            termination_flag = self.data['action']['user']['termination_flag']
#                        ),
#                    )
#                )
#            else:
#                return dict(
#                    observation = dict(
#                        image = None
#                    ),
#                    action = dict(
#                        controller = None,
#                        agent = dict(
#                            velocity = dict(
#                                linear = None,
#                                angular = None
#                            ),
#                            termination_flag = None,
#                        ),
#                        user = dict(
#                            velocity = dict(
#                                linear = None,
#                                angular = None
#                            ),
#                            termination_flag = None
#                        ),
#                    )
#                )
#
#    def remove_data(self, index, leftwards=True):
#        list_strings = get_leaf_string(self.data)
#        for s in list_strings:
#            if leftwards == True:
#                exec(f"self.data{s} = self.data{s}[{index+1}:]")
#            else:
#                exec(f"self.data{s} = self.data{s}[:{index}]")
#        
#        self.length = self.get_episode_length_()
#        if self.length == 0:
#            self.data_empty = True
#
#    def get_tensor(self):
#        episode = self.data
#        images = [np.array(img) for img in episode['observation']['image']]
#
#        observations = torch.tensor(np.stack(images), dtype = torch.float32).permute((0,3,1,2))/255.0
#
#        controller = episode['action']['controller']
#        agent_linear_vel = episode['action']['agent']['velocity']['linear']
#        agent_angular_vel = episode['action']['agent']['velocity']['angular']
#        agent_termination_flag = episode['action']['agent']['termination_flag']
#        user_linear_vel = episode['action']['user']['velocity']['linear']
#        user_angular_vel = episode['action']['user']['velocity']['angular']
#        user_termination_flag = episode['action']['user']['termination_flag']
#
#        desired_linear_vel = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_linear_vel, agent_linear_vel, controller)])
#        desired_angular_vel = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_angular_vel, agent_angular_vel, controller)])
#        desired_termination_flag = torch.tensor([user if cont == ControllerType.USER else agent for (user, agent, cont) in zip(user_termination_flag, agent_termination_flag, controller)], dtype = torch.float32)
#        demonstration_flag = torch.tensor([True if cont == ControllerType.USER else False for cont in controller], dtype = torch.float32)
#
#        latent_vec = torch.stack([desired_linear_vel, desired_angular_vel, desired_termination_flag, demonstration_flag])
#
#        return observations, latent_vec

#def get_leaf_string(dict_, string = ""):
#    try:
#        for key in dict_.keys():
#            s = f"['{key}']"
#            ss = string + s
#            yield from get_leaf_string(dict_[key], ss)
#    except:
#        yield string
