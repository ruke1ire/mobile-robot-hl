import os
import glob
import yaml
import shutil
from PIL import Image as PImage
from enum import Enum
import copy

class ControllerType(Enum):
    NONE = 0
    USER = 1
    AGENT = 2

class InformationType(Enum):
    NONE = 0
    DEMO = 1
    TASK_EPISODE = 2

class DemoHandler():
    DEMO_ID_INFO_FILE= "info.yaml"
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok = True)

    def get(self, name, id_):
        '''
        Return the demonstration with DataStructure dict_list
        '''
        try:
            with open(f"{self.path}/{name}/{id_}/{DemoHandler.DEMO_ID_INFO_FILE}", 'r') as stream:
                info = yaml.safe_load(stream)
        except:
            raise Exception(f'Invalid demo name: {name} ID: {id_}')

        images = [PImage.open(f"{self.path}/{name}/{id_}/{image_id}.png") 
                    for image_id in info['observation']['image_id']]

        info['observation']['image'] = images
        del info['observation']['image_id']
        task_controller = [ControllerType[cont] for cont in info['action']['controller']]
        info['action']['controller'] = task_controller


        episode_data = EpisodeData(data=info)
        return episode_data
    
    def get_names(self):
        '''
        Get all available demo names
        '''
        demos = [os.path.basename(x) for x in glob.glob(self.path+"/*")]
        return demos
    
    def get_ids(self, name):
        '''
        Get all available demo ids
        '''
        demos = [int(os.path.basename(x)) for x in glob.glob(self.path+"/"+name+"/*") if "yaml" not in os.path.basename(x)]
        return demos

    def save(self, demo, name, id_=None):
        '''
        demo: demonstration data with DataStructure = list_dict
        name: name of demonstration
        id_: id of demonstration
        '''

        demo_copy = EpisodeData(data=demo.data)

        if(demo_copy.get_data(index = -1)['action']['controller'] in [None, ControllerType.NONE]):
            demo_copy.remove_data(index=-1, leftwards=False)

        if not os.path.exists(f"{self.path}/{name}"):
            os.mkdir(f"{self.path}/{name}")

        if(id_ == None):
            try:
                taken_id = max([x for x in self.get_ids(name) if type(x) == int])
                next_id = taken_id + 1
            except:
                next_id = 0
        else:
            next_id = id_

        if os.path.exists(f"{self.path}/{name}/{next_id}"):
            shutil.rmtree(f"{self.path}/{name}/{next_id}", ignore_errors=True)
        os.mkdir(f"{self.path}/{name}/{next_id}")

        dict_data = demo_copy.data.copy()

        image_ids = list(range(demo_copy.get_episode_length()))
        for i in image_ids:
            img = dict_data['observation']['image'][i]
            img.save(f"{self.path}/{name}/{next_id}/{i}.png")
        
        del dict_data['observation']['image']
        dict_data['observation']['image_id'] = image_ids
        dict_data['action']['controller'] = [cont.name for cont in dict_data['action']['controller']]

        with open(f'{self.path}/{name}/{next_id}/{DemoHandler.DEMO_ID_INFO_FILE}', 'w') as outfile:
            yaml.dump(dict_data, outfile)

class TaskHandler():
    TASK_ID_INFO_FILE= "info.yaml"
    def __init__(self, path, demo_handler):
        self.path = path
        self.demo_handler = demo_handler
        os.makedirs(self.path, exist_ok = True)

    def get(self, name, id_):
        '''
        Return the task episode with DataStructure = dict_list
        '''

        try:
            with open(f"{self.path}/{name}/{id_}/{TaskHandler.TASK_ID_INFO_FILE}", 'r') as stream:
                info = yaml.safe_load(stream)
        except:
            raise Exception(f'Invalid task name: {name} ID: {id_}')

        demo_id = info['demonstration']

        demo_episode_data = self.demo_handler.get(name, demo_id)

        task_images = [PImage.open(f"{self.path}/{name}/{id_}/{image_id}.png") 
                    for image_id in info['observation']['image_id']]
        task_controller = [ControllerType[cont] for cont in info['action']['controller']]
        
        info['observation']['image'] = task_images
        info['action']['controller'] = task_controller
        del info['observation']['image_id']
        del info['demonstration']

        task_episode_data = EpisodeData(data=info)
        demo_episode_data.append_episode_data(task_episode_data)

        return demo_episode_data
    
    def get_names(self):
        '''
        Get all available task names
        '''
        demos = [os.path.basename(x) for x in glob.glob(self.path+"/*")]
        return demos
    
    def get_ids(self, name):
        '''
        Get all available task ids
        '''
        task_episodes = [int(os.path.basename(x)) for x in glob.glob(self.path+"/"+name+"/*") if "yaml" not in os.path.basename(x)]
        return task_episodes

    def save(self, episode, demo_name, demo_id, task_id = None):
        '''
        episode: DataStructure = LIST_DICT
        demo_name: name of demonstration
        demo_id: id of demonstration
        task_id: id of task episode

        storing DataStructure = DICT_LIST
        '''

        if not os.path.exists(f"{self.path}/{demo_name}"):
            os.mkdir(f"{self.path}/{demo_name}")

        if(task_id == None):
            try:
                taken_id = max([x for x in self.get_ids(demo_name) if type(x) == int])
                next_id = taken_id + 1
            except:
                next_id = 0
        else:
            next_id = task_id

        if os.path.exists(f"{self.path}/{demo_name}/{next_id}"):
            shutil.rmtree(f"{self.path}/{demo_name}/{next_id}", ignore_errors=True)
        os.mkdir(f"{self.path}/{demo_name}/{next_id}")

        episode_copy = EpisodeData(episode.data)

        if(episode_copy.get_data(index = -1)['action']['controller'] in [None, ControllerType.NONE]):
            episode_copy.remove_data(index = -1, leftwards=False)
        
        task_index = episode_copy.data['action']['controller'].index(ControllerType.AGENT)

        episode_copy.remove_data(task_index-1, leftwards=True)

        dict_data = episode_copy.get_data()

        image_ids = list(range(episode_copy.get_episode_length()))
        for i in image_ids:
            img = dict_data['observation']['image'][i]
            img.save(f"{self.path}/{demo_name}/{next_id}/{i}.png")
        
        del dict_data['observation']['image']
        dict_data['observation']['image_id'] = image_ids
        dict_data['action']['controller'] = [cont.name for cont in dict_data['action']['controller']]
        dict_data['demonstration'] = demo_id

        with open(f'{self.path}/{demo_name}/{next_id}/{TaskHandler.TASK_ID_INFO_FILE}', 'w') as outfile:
            yaml.dump(dict_data, outfile)
    
class EpisodeData:
    def __init__(self, data):
        if data == None:
            self.init_empty_structure()
        else:
            # TODO: error if data is just an empty structure
            self.data = copy.deepcopy(data)
            self.length = self.get_episode_length_()
            self.data_empty = False

    def init_empty_structure(self):
        self.data = dict(
                        observation=dict(image=[None]),
                        action=dict(
                            agent=dict(
                                velocity=dict(
                                    linear=[None],
                                    angular=[None]
                                ),
                                termination_flag = [None]
                            ),
                            user=dict(
                                velocity=dict(
                                    linear=[None],
                                    angular=[None]
                                ),
                                termination_flag = [None]
                            ),
                            controller=[None]))
        self.data_empty = True
        self.length = 0
    
    def append_episode_data(self, episode):
        if(episode.data_empty == True):
            return
        else:
            self.data['observation']['image'] += episode.data['observation']['image']
            self.data['action']['agent']['velocity']['linear'] += episode.data['action']['agent']['velocity']['linear']
            self.data['action']['agent']['velocity']['angular'] += episode.data['action']['agent']['velocity']['angular']
            self.data['action']['agent']['termination_flag'] += episode.data['action']['agent']['termination_flag']
            self.data['action']['user']['velocity']['linear'] += episode.data['action']['user']['velocity']['linear']
            self.data['action']['user']['velocity']['angular'] += episode.data['action']['user']['velocity']['angular']
            self.data['action']['user']['termination_flag'] += episode.data['action']['user']['termination_flag']
            self.data['action']['controller'] += episode.data['action']['controller']
            self.data_empty = False

            self.length += episode.length

    def append_data(
        self, 
        image, 
        agent_linear_vel, agent_angular_vel, agent_termination_flag,
        user_linear_vel, user_angular_vel, user_termination_flag,
        controller):
        if(self.data_empty == True):
            self.data['observation']['image'] = [image]
            self.data['action']['agent']['velocity']['linear'] = [agent_linear_vel]
            self.data['action']['agent']['velocity']['angular'] = [agent_angular_vel]
            self.data['action']['agent']['termination_flag'] = [agent_termination_flag]
            self.data['action']['user']['velocity']['linear'] = [user_linear_vel]
            self.data['action']['user']['velocity']['angular'] = [user_angular_vel]
            self.data['action']['user']['termination_flag'] = [user_termination_flag]
            self.data['action']['controller'] = [controller]
            self.data_empty = False
        else:
            self.data['observation']['image'].append(image)
            self.data['action']['agent']['velocity']['linear'].append(agent_linear_vel)
            self.data['action']['agent']['velocity']['angular'].append(agent_angular_vel)
            self.data['action']['agent']['termination_flag'].append(agent_termination_flag)
            self.data['action']['user']['velocity']['linear'].append(user_linear_vel)
            self.data['action']['user']['velocity']['angular'].append(user_angular_vel)
            self.data['action']['user']['termination_flag'].append(user_termination_flag)
            self.data['action']['controller'].append(controller)

        self.length += 1

    def set_key_value(self, key, value, index = None):
        '''
        key = string of keys separated by "."
        Eg. "action.agent.velocity.linear"
        '''
        key_split = key.split('.')
        dict_string = ''
        for k in key_split:
            dict_string += "['"
            dict_string += k
            dict_string += "']"
        if type(index) is not int:
            exec(f"self.data{dict_string} = {value}")
        else:
            exec(f"self.data{dict_string}[{index}] = {value}")

    def set_data(
        self, 
        index,
        image=None, 
        agent_linear_vel=None, agent_angular_vel=None, agent_termination_flag=None,
        user_linear_vel=None, user_angular_vel=None, user_termination_flag=None,
        controller=ControllerType.NONE):
        if(self.data_empty == False):
            assert index <= self.get_episode_length() - 1
            self.data['observation']['image'][index] = image
            self.data['action']['agent']['velocity']['linear'][index] = agent_linear_vel
            self.data['action']['agent']['velocity']['angular'][index] = agent_angular_vel
            self.data['action']['agent']['termination_flag'][index] = agent_termination_flag
            self.data['action']['user']['velocity']['linear'][index] = user_linear_vel
            self.data['action']['user']['velocity']['angular'][index] = user_angular_vel
            self.data['action']['user']['termination_flag'][index] = user_termination_flag
            self.data['action']['controller'][index] = controller
        else:
            self.append_data(image, agent_linear_vel, agent_angular_vel, agent_termination_flag,
            user_linear_vel, user_angular_vel, user_termination_flag,
            controller)

    def get_episode_length_(self):
        return len(self.data['observation']['image'])

    def get_episode_length(self):
        return self.length
    
    def get_data(self, index = None):
        if(type(index) is not int):
            return self.data
        else:
            if(self.data_empty == False):
                return dict(
                    observation = dict(
                        image = self.data['observation']['image'][index]
                    ),
                    action = dict(
                        controller = self.data['action']['controller'][index],
                        agent = dict(
                            velocity = dict(
                                linear = self.data['action']['agent']['velocity']['linear'][index],
                                angular = self.data['action']['agent']['velocity']['angular'][index]
                            ),
                            termination_flag = self.data['action']['agent']['termination_flag']
                        ),
                        user = dict(
                            velocity = dict(
                                linear = self.data['action']['user']['velocity']['linear'][index],
                                angular = self.data['action']['user']['velocity']['angular'][index]
                            ),
                            termination_flag = self.data['action']['user']['termination_flag']
                        ),
                    )
                )
            else:
                return dict(
                    observation = dict(
                        image = None
                    ),
                    action = dict(
                        controller = None,
                        agent = dict(
                            velocity = dict(
                                linear = None,
                                angular = None
                            ),
                            termination_flag = None,
                        ),
                        user = dict(
                            velocity = dict(
                                linear = None,
                                angular = None
                            ),
                            termination_flag = None
                        ),
                    )
                )

    def remove_data(self, index, leftwards=True):
        list_strings = get_leaf_string(self.data)
        for s in list_strings:
            if leftwards == True:
                exec(f"self.data{s} = self.data{s}[{index+1}:]")
            else:
                exec(f"self.data{s} = self.data{s}[:{index}]")
        
        self.length = self.get_episode_length_()
        if self.length == 0:
            self.data_empty = True

class ModelHandler():
    def __init__(self):
        pass


def get_leaf_string(dict_, string = ""):
    try:
        for key in dict_.keys():
            s = f"['{key}']"
            ss = string + s
            yield from get_leaf_string(dict_[key], ss)
    except:
        yield string