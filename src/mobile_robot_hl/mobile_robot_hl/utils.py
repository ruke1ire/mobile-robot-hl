import os
import glob
import yaml
import shutil
from PIL import Image as PImage
from enum import Enum

class DemoHandler():
    DEMO_NAME_INFO_FILE = "info.yaml"
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
            raise Exception("Invalid demo name and id! Name: {name} ID: {id_}")

        images = [PImage.open(f"{self.path}/{name}/{id_}/{image_id}.png") 
                    for image_id in info['observation']['image_id']]

        info['observation']['image'] = images
        del info['observation']['image_id']
        task_controller = [ControllerType[cont] for cont in info['action']['controller']]
        info['action']['controller'] = task_controller


        episode_data = EpisodeData(data=info, structure=DataStructure.DICT_LIST)
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

        if(demo.structure == DataStructure.LIST_DICT):
            demo.restructure(DataStructure.DICT_LIST)

        dict_data = demo.data.copy()

        image_ids = list(range(demo.get_episode_length()))
        for i in image_ids:
            img = dict_data['observation']['image'][i]
            img.save(f"{self.path}/{name}/{next_id}/{i}.png")
        
        del dict_data['observation']['image']
        dict_data['observation']['image_id'] = image_ids
        dict_data['action']['controller'] = [cont.name for cont in dict_data['action']['controller']]

        with open(f'{self.path}/{name}/{next_id}/{DemoHandler.DEMO_ID_INFO_FILE}', 'w') as outfile:
            yaml.dump(dict_data, outfile)

class TaskHandler():
    TASK_NAME_INFO_FILE = "info.yaml"
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
            raise Exception("Invalid task name and id! Name: {name} ID: {id_}")
        demo_id = info['demonstration']

        demo_episode_data = self.demo_handler.get(name, demo_id)

        task_images = [PImage.open(f"{self.path}/{name}/{id_}/{image_id}.png") 
                    for image_id in info['observation']['image_id']]
        task_controller = [ControllerType[cont] for cont in info['action']['controller']]
        
        info['observation']['image'] = task_images
        info['action']['controller'] = task_controller
        del info['observation']['image_id']
        del info['demonstration']

        task_episode_data = EpisodeData(data=info, structure=DataStructure.DICT_LIST)
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

        episode_copy = EpisodeData(episode.data, structure=episode.structure)

        if(episode_copy.structure == DataStructure.LIST_DICT):
            episode_copy.restructure(DataStructure.DICT_LIST)
        
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
    
def restructure_list2dict(list_data):
    '''
    list = [dict(observation, action)]
    dict = dict(observation=list(observation), action=list(action))

    observation = dict(image)
    action = dict(user, agent, controller)
    user, agent = dict(velocity, termination_flag)
    *_velocity = dict(linear, angular)
    '''
    
    agent_linear_vel = []
    agent_angular_vel = []
    agent_termination_flag = []
    user_linear_vel = []
    user_angular_vel = []
    user_termination_flag = []
    controller = []
    image = []

    for d in list_data:
        agent_linear_vel.append(d['action']['agent']['velocity']['linear'])
        agent_angular_vel.append(d['action']['agent']['velocity']['angular'])
        agent_termination_flag.append(d['action']['agent']['termination_flag'])
        user_linear_vel.append(d['action']['user']['velocity']['linear'])
        user_angular_vel.append(d['action']['user']['velocity']['angular'])
        user_termination_flag.append(d['action']['user']['termination_flag'])
        controller.append(d['action']['controller'])
        image.append(d['observation']['image'])
    return dict(
        observation=dict(image=image), 
        action=dict(
            agent=dict(
                velocity=dict(
                    linear=agent_linear_vel,
                    angular=agent_angular_vel
                ),
                termination_flag=agent_termination_flag
            ),
            user=dict(
                velocity=dict(
                    linear=user_linear_vel,
                    angular=user_angular_vel
                ),
                termination_flag=user_termination_flag
            ),
            controller = controller
        ))

def restructure_dict2list(dict_data):
    '''
    dict = dict(observation=list(observation), action=list(action))
    list = [dict(observation, action)]

    observation = dict(image)
    action = dict(user, agent, controller)
    user, agent = dict(velocity, termination_flag)
    *_velocity = dict(linear, angular)
    '''
    agent_linear_vel = dict_data['action']['agent']['velocity']['linear']
    agent_angular_vel = dict_data['action']['agent']['velocity']['angular']
    agent_termination_flag = dict_data['action']['agent']['termination_flag']
    user_linear_vel = dict_data['action']['user']['velocity']['linear']
    user_angular_vel = dict_data['action']['user']['velocity']['angular']
    user_termination_flag = dict_data['action']['user']['termination_flag']
    controller = dict_data['action']['controller']
    image = dict_data['observation']['image']
    combined_data = zip(
        image,
        agent_linear_vel,
        agent_angular_vel,
        agent_termination_flag,
        user_linear_vel,
        user_angular_vel,
        user_termination_flag,
        controller,
    )
    return [dict(
        observation=dict(image=img),
        action=dict(
            agent=dict(
                velocity=dict(
                    linear=agent_lin,
                    angular=agent_ang
                ),
                termination_flag = agent_term
            ),
            user=dict(
                velocity=dict(
                    linear=user_lin,
                    angular=user_ang
                ),
                termination_flag = user_term
            ),
            controller=ctrl
        )) for (img, agent_lin, agent_ang, agent_term, user_lin, user_ang, user_term, ctrl) in combined_data]

class EpisodeData:
    def __init__(self, data, structure):
        self.structure = structure
        if data == None:
            self.init_empty_structure()
        else:
            self.data = data.copy()
            self.data_empty = False

    def init_empty_structure(self):
        if self.structure == DataStructure.LIST_DICT:
            self.data = [dict(
                            observation=dict(image=None),
                            action=dict(
                                agent=dict(
                                    velocity=dict(
                                        linear=None,
                                        angular=None
                                    ),
                                    termination_flag = None
                                ),
                                user=dict(
                                    velocity=dict(
                                        linear=None,
                                        angular=None
                                    ),
                                    termination_flag = None
                                ),
                                controller=None))]
        else:
            self.data = dict(
                            observation=dict(image=[]),
                            action=dict(
                                agent=dict(
                                    velocity=dict(
                                        linear=[],
                                        angular=[]
                                    ),
                                    termination_flag = []
                                ),
                                user=dict(
                                    velocity=dict(
                                        linear=[],
                                        angular=[]
                                    ),
                                    termination_flag = []
                                ),
                                controller=[]))
        self.data_empty = True
    
    def restructure(self, structure, inplace = True):
        if(self.structure != structure):
            if(self.structure == DataStructure.DICT_LIST):
                data = restructure_dict2list(self.data)
                structure = DataStructure.LIST_DICT
            elif(self.structure == DataStructure.LIST_DICT):
                data = restructure_list2dict(self.data)
                structure = DataStructure.DICT_LIST
            else:
                raise Exception("Invalid data structure")
        else:
            data = self.data
            structure = self.structure

        if inplace == True:
            self.data = data
            self.structure = structure
            return
        else:
            return data

    def append_episode_data(self, episode_data):
        if(episode_data.data_empty == True):
            return
        else:
            episode_data_data = episode_data.data
            if(self.structure == DataStructure.LIST_DICT):
                if(episode_data.structure != self.structure):
                    episode_data_data = restructure_dict2list(episode_data_data)
                self.data += episode_data_data
            else:
                if(episode_data.structure != self.structure):
                    episode_data_data = restructure_list2dict(episode_data_data)
                self.data['observation']['image'] += episode_data_data['observation']['image']
                self.data['action']['agent']['velocity']['linear'] += episode_data_data['action']['agent']['velocity']['linear']
                self.data['action']['agent']['velocity']['angular'] +=  episode_data_data['action']['agent']['velocity']['angular']
                self.data['action']['agent']['termination_flag'] +=  episode_data_data['action']['agent']['termination_flag']
                self.data['action']['user']['velocity']['linear'] +=  episode_data_data['action']['user']['velocity']['linear']
                self.data['action']['user']['velocity']['angular'] +=  episode_data_data['action']['user']['velocity']['angular']
                self.data['action']['user']['termination_flag'] +=  episode_data_data['action']['user']['termination_flag']
                self.data['action']['controller'] +=  episode_data_data['action']['controller']
            self.data_empty = False

    def append_data(
        self, 
        image, 
        agent_linear_vel, agent_angular_vel, agent_termination_flag,
        user_linear_vel, user_angular_vel, user_termination_flag,
        controller):
        if(self.data_empty == True):
            if(self.structure==DataStructure.DICT_LIST):
                self.data['observation']['image'] = [image]
                self.data['action']['agent']['velocity']['linear'] = [agent_linear_vel]
                self.data['action']['agent']['velocity']['angular'] = [agent_angular_vel]
                self.data['action']['agent']['termination_flag'] = [agent_termination_flag]
                self.data['action']['user']['velocity']['linear'] = [user_linear_vel]
                self.data['action']['user']['velocity']['angular'] = [user_angular_vel]
                self.data['action']['user']['termination_flag'] = [user_termination_flag]
                self.data['action']['controller'] = [controller]
            else:
                self.data[0]['observation']['image'] = image
                self.data[0]['action']['agent']['velocity']['linear'] = agent_linear_vel
                self.data[0]['action']['agent']['velocity']['angular'] = agent_angular_vel
                self.data[0]['action']['agent']['termination_flag'] = agent_termination_flag
                self.data[0]['action']['user']['velocity']['linear'] = user_linear_vel
                self.data[0]['action']['user']['velocity']['angular'] = user_angular_vel
                self.data[0]['action']['user']['termination_flag'] = user_termination_flag
                self.data[0]['action']['controller'] = controller
            self.data_empty = False
        else:
            if(self.structure==DataStructure.DICT_LIST):
                self.data['observation']['image'].append(image)
                self.data['action']['agent']['velocity']['linear'].append(agent_linear_vel)
                self.data['action']['agent']['velocity']['angular'].append(agent_angular_vel)
                self.data['action']['agent']['termination_flag'].append(agent_termination_flag)
                self.data['action']['user']['velocity']['linear'].append(user_linear_vel)
                self.data['action']['user']['velocity']['angular'].append(user_angular_vel)
                self.data['action']['user']['termination_flag'].append(user_termination_flag)
                self.data['action']['controller'].append(controller)
            else:
                self.data.append(dict(
                            observation=dict(image=image),
                            action=dict(
                                agent=dict(
                                    velocity=dict(
                                        linear=agent_linear_vel,
                                        angular=agent_angular_vel
                                    ),
                                    termination_flag = agent_termination_flag
                                ),
                                user=dict(
                                    velocity=dict(
                                        linear=user_linear_vel,
                                        angular=user_angular_vel
                                    ),
                                    termination_flag = user_termination_flag
                                ),
                                controller=controller)))
    def get_episode_length(self):
        if(self.data_empty == True):
            return 0
        if(self.structure == DataStructure.LIST_DICT):
            return len(self.data)
        else:
            return len(self.data['observation']['image'])
    
    def get_data(self, structure = None):
        if structure == None:
            structure = self.structure
        data = self.restructure(structure, inplace=False)
        return data
    
    def remove_data(self, index, leftwards=True):
        if self.structure == DataStructure.LIST_DICT:
            if leftwards == True:
                self.data = self.data[index+1:]
            else:
                self.data = self.data[:index]
        else:
            list_strings = get_leaf_string(self.data)
            for s in list_strings:
                if leftwards == True:
                    exec(f"self.data{s} = self.data{s}[{index+1}:]")
                else:
                    exec(f"self.data{s} = self.data{s}[:{index}]")
        
        if self.get_episode_length() == 0:
            self.data_empty = True

class ModelHandler():
    def __init__(self):
        pass

class ControllerType(Enum):
    USER = 0
    AGENT = 1

class InformationType(Enum):
    NONE = 0
    DEMO = 1
    TASK_EPISODE = 2

class DataStructure(Enum):
    LIST_DICT = 0
    DICT_LIST = 1

def get_leaf_string(dict_, string = ""):
    try:
        for key in dict_.keys():
            s = f"['{key}']"
            ss = string + s
            yield from get_leaf_string(dict_[key], ss)
    except:
        yield string